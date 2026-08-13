# Can we run sacremoses' regex operations on a GPU?

A feasibility assessment. **Nothing here was implemented.** This document is a
recommendation, a set of measurements, and a plan. No library code was changed;
this file is the only thing the investigation wrote.

---

## Verdict

**No — don't build the "shard the file, run the substitutions on GPU tensors,
collect the outputs" design.** It is a large project whose best outcome is beaten
by a two-line change, and this is measured, not predicted:

> **[measured]** Replacing **every one of `tokenize()`'s 23 regex substitutions
> with a no-op** — a perfect, infinitely fast, zero-cost accelerator — makes the
> library **1.71×** faster. Hoisting two constant sets out of `islower()` and
> `isanyalpha()` makes it **2.08×** faster.
>
> **A two-line bug fix beats the theoretical ceiling of any regex accelerator
> that will ever exist.**

That is the whole assessment. Regex is only ~30% of `tokenize()`'s runtime, so
Amdahl's law caps the entire idea below 2× before a single line of CUDA is
written.

The obstacles are also not the ones usually cited. PCIe transfer is irrelevant
here (~140,000× cheaper than the compute it replaces), and the regex dialect is
**not** a blocker — 27 of 28 sacremoses patterns compile unmodified under an
RE2-class engine. What actually caps it is that the library spends ~70% of its
time in non-regex Python over 66-character strings.

**Do this instead.** The profile found that **53% of `tokenize()` wall time is
`islower()`**, which rebuilds a 2,253-element `set` from scratch on every call
(**607 µs/call**; 392× faster once hoisted). Fix that and `isanyalpha()`, reclaim
the **28%** currently lost to CPython's cyclic GC, pre-compile the normalizer's
patterns, and take Python 3.12's documented **2–3×** on
`re.sub`-with-backreferences: that is roughly **3–5× for a few hours of work**.
If it is still not enough, the C++ `opus-fast-mosestokenizer` is already **15×
faster than sacremoses** and someone else paid for it. The GPU becomes
interesting only *after* the substitution chain is compiled into a single
automaton — and by then you will not need it.

---

## 1. What was measured, and on what

Claims are labelled **[measured]** (code run on this machine during this
investigation), **[source-verified]** (read in the cited project's source), or
**[literature]** (from the cited source, not reproduced here).

| | |
|---|---|
| CPU | **Intel Core i7-1060NG7 @ 1.20 GHz** (Ice Lake-Y, 4 physical / 8 logical) |
| RAM | 16 GB |
| GPU | **Intel Iris Plus Graphics** (integrated, 1536 MB dynamic VRAM, Metal 3) |
| OS | macOS 15.7.3 (24G419), darwin **x86_64** |
| Python | 3.9.12 (miniconda), `pyarrow` 19.0.0, `joblib` 1.3.1, `regex` 2.5.140 |
| Corpus | `big.txt` — 6,488,666 bytes, 128,458 lines, 103,501 non-empty |
| Tree | `/Users/alvas/git-stuff/sacremoses`, HEAD `53ce2d9` + concurrent audit edits |

### A correction to the brief

The task described this as "an Apple Silicon / darwin box." **It is not** — it is
an **Intel** Mac (2020 MacBook Air-class, 1.2 GHz Ice Lake-Y). That changes the
local-testability story materially. **[measured]**

- **No CUDA** — `nvcc` and `nvidia-smi` absent; `import cudf` → `ModuleNotFoundError`.
- **No PyTorch MPS** — `torch.backends.mps.is_available()` is **`False`**. PyTorch's
  Metal backend requires Apple Silicon; this Intel Mac does not get it.
- **No MLX** — `pip install mlx` → *"No matching distribution found for mlx"*.
  MLX ships Apple-Silicon-only wheels.

So there is **no usable GPU compute path on this hardware at all**. Any GPU work
would need remote NVIDIA hardware, or hand-written Metal against a 2020
integrated iGPU. That cost belongs in the decision. It is also why the
experiments below use **CPU proxies (Apache Arrow + RE2)** that reproduce the
*cost structure and feature set* of the proposed design without the hardware.

---

## 2. The CPU baseline

Single-threaded, first 20,000 non-empty lines of `big.txt` (1.680 MB), best of 2.
**[measured]**

| Operation | lines/s | MB/s | whole 6.36 MB file |
|---|---:|---:|---:|
| `MosesTokenizer.tokenize()` | **~1,035** | **~0.087** | ~100 s |
| `tokenize(return_str=True)` | ~1,000–1,110 | ~0.084–0.093 | ~75–100 s |
| `tokenize(escape=False)` | ~950 | ~0.080 | ~106 s |
| `MosesPunctNormalizer.normalize()` | ~3,163 | ~0.266 | ~24 s |
| `normalize()` → `tokenize()` | ~528 | ~0.044 | ~145 s |
| `str.split()` (no-op reference) | 124,321 | 10.441 | 0.6 s |

**Headline: sacremoses tokenizes at ~0.09 MB/s — about 1,000 lines/second.** It
is **~120× slower than merely touching every line**.

### Variance, reported honestly

The first run of this benchmark recorded 48.1 s for 20,000 lines (416 lines/s).
A careful re-run recorded 26.0 s for the identical call, and 18.7 s with GC off.
The 48.1 s figure was inflated by CPU contention from concurrent tooling and by GC
pressure. **The trustworthy figure is ~19–20 s / 20,000 lines ≈ 1,000–1,070
lines/s.** Reporting the range, not the prettiest number.

### For scale: what "fast" looks like

| Implementation | Throughput | |
|---|---|---|
| **sacremoses, this machine** | **~0.09 MB/s** | **[measured]** |
| `opus-fast-mosestokenizer` (C++) | *"6× faster than tokenizer.perl and **15× faster than sacremoses**"* | [literature] |
| HuggingFace `tokenizers` (Rust, CPU) | *"less than 20 seconds to tokenize a GB of text on a server's CPU"* ≈ **50 MB/s** | [literature] |

HuggingFace's **CPU** tokenizer is ~500× faster per megabyte than sacremoses, on
comparable hardware, with no accelerator. **That gap is entirely software.**
Reaching for a GPU while a 500× software gap sits untouched is optimising the
wrong variable.

---

## 3. What sacremoses actually does

### 3.1 The chain is sequential and data-dependent

**[measured]** by wrapping every compiled pattern in a counting proxy and running
a representative line through the public API:

| Entry point | dependent `.sub()` calls **per line** |
|---|---:|
| `tokenize()` (default, `escape=True`) | **23** |
| `tokenize(escape=False)` | 15 |
| `tokenize(aggressive_dash_splits=True)` | 24 |
| `tokenize()`, `lang="de"` | 19 |
| `MosesPunctNormalizer.normalize()`, `lang="en"` | **45** |
| `normalize(pre_replace_unicode_punct=True, post_remove_control_chars=True)` | 81 |

**A realistic normalize→tokenize pipeline is 68 sequential regex passes per line.**

They are strictly dependent — each consumes the previous one's output. The
measured chain for one 144-char line, with length after each step:

```
 1. DEDUPLICATE_SPACE          144 -> 144
 2. ASCII_JUNK                 144 -> 144
 3. PAD_NOT_ISALNUM            144 -> 168   (+24)
 4. MULTIDOT_RUN               168 -> 181   (+13)
 5. COMMA_SEPARATE_1           181 -> 183
 6. COMMA_SEPARATE_2           183 -> 185
 7. COMMA_SEPARATE_3           185 -> 185
 8-12. {EN,FR_IT}_SPECIFIC_*   185 -> 186
      << handles_nonbreaking_prefixes() -- pure Python, no .sub() -- 186 -> 174 >>
13. DEDUPLICATE_SPACE          174 -> 174
14. TRAILING_DOT_APOSTROPHE    174 -> 174
15. MULTIDOT_MARKER            174 -> 163   (-11)
16-23. ESCAPE_*                163 -> 178   (+15)
```

(Rule names repeat where `re.compile()`'s internal cache returns the same object
for identical pattern text — e.g. `EN_SPECIFIC_1` and `FR_IT_SPECIFIC_1`. The
*count* is what matters.)

Two structural facts: **the length changes at nearly every step** (+24, +13, −11,
+15 …), so nothing has a fixed shape; and **the largest single step is not a
regex at all** — the 186→174 drop is `handles_nonbreaking_prefixes()`.

### 3.2 Which regex features are used — the surprise

The expectation is that Perl-flavoured Moses rules are full of lookaround and
backreferences a GPU engine cannot express. **That expectation is wrong.**

Audit of all **88** class-level `(pattern, replacement)` constants on
`MosesTokenizer`: **[measured]**

| Feature | Occurrences | GPU-engine problem? |
|---|---|---|
| **Backreference in the *replacement*** (`\1`, `\2`) | **39 of 88 patterns** | **No.** cuDF supports `\1`–`\99` and `${1}`–`${99}`; RE2/Arrow supports `\1`. |
| `\g<1>` replacement syntax | `normalize.py` | **No** — needs a trivial textual rewrite to `\1`; RE2 rejects `\g<1>`. |
| **Lookahead `(?=`** | **exactly 1** — `AGGRESSIVE_HYPHEN_SPLIT` | Yes — but **off by default** (`aggressive_dash_splits=False`). |
| Lookbehind `(?<=` / `(?<!` | **none** | — |
| Negative lookahead `(?!` | **none** | — |
| Backreference **in the pattern** | **none** | — |
| Lazy quantifiers | **none** | — |
| `re.IGNORECASE` | only caller-supplied `protected_patterns` | Supported by cuDF and RE2. |
| `\p{C}` Unicode property | `normalize.remove_control_chars` | Supported — **[measured]** works in Arrow/RE2. |
| `(?:` non-capturing | `MULTIDOT_MARKER` | Supported. |
| `\b` | none in the tokenize chain | Supported anyway. |

**[measured] — direct expressibility test.** 28 actual sacremoses patterns fed to
`pyarrow.compute.replace_substring_regex`, backed by **Google RE2** — the same
feature class as cuDF's GPU engine:

```
accepted by RE2: 27/28   rejected: 1
  REJECTED AGGRESSIVE_HYPHEN_SPLIT: invalid perl operator: (?=
```

**The regex dialect is not the blocker.** This is the opposite of the expected
result, and it matters: "GPU regex can't express Moses" is **not** a valid reason
to reject the idea. The valid reasons are in §4 and §3.5.

### 3.3 The per-line work is tiny

**[measured]** on `big.txt`: 103,501 non-empty lines; **mean 61.4 chars, median
66, p95 72, max 2,541**.

**The unit of work is a 66-character string.** §4.4 quantifies the damage.

### 3.4 Enormous character classes — but only because of Python

**[measured]** The Perl-Unicode classes are inlined as literal character sets:

| Class | literal chars | **contiguous ranges** | compression |
|---|---:|---:|---:|
| `IsAlnum` | 11,232 | **461** | 24.4× |
| `IsAlpha` | 10,862 | **442** | 24.6× |
| `IsSo` | 2,321 | 110 | 21.1× |
| `IsLower` | 2,253 | 641 | 3.5× |
| `IsN` | 713 | 64 | 11.1× |
| `IsSc` | 53 | 18 | 2.9× |

Python's `re` compiles a literal set rather than coalescing ranges, so the
pattern *source strings* are **21–22 KB each** (`AGGRESSIVE_HYPHEN_SPLIT` 22,476
chars; `EN_SPECIFIC_2` 22,449; …). Remember **461 ranges** — §4.4 shows it is
exactly what breaks cuDF's fast path.

**But the ruleset is far simpler than it looks.** **[measured]** Classifying every
code point by its membership vector across all six classes plus the literal set:

```
universe examined            : 14,147 code points
distinct equivalence classes : 17
```

**The entire ruleset distinguishes only 17 kinds of character.** A DFA or
transducer over this ruleset needs an alphabet of ~17 symbols (~50 once
individual literals are separated), not 14,147. **Character-class size is an
artefact of the Python encoding, not an intrinsic obstacle** — which is the
single most important input to §7's "what would have to be true".

### 3.5 Where the time actually goes — the decisive measurement

**[measured]** `cProfile`, 4,000 lines, sorted by `tottime`:

```
   ncalls  tottime  cumtime  function
     5344    3.464    3.473  tokenize.py:373(islower)              <-- 53% of total
    92000    1.246    1.729  {method 'sub' of 're.Pattern' objects}
    36000    0.319    0.320  re.py:325(_subx)
     4000    0.314    4.582  tokenize.py:418(handles_nonbreaking_prefixes)   <-- 70% cumulative
   135416    0.205    0.671  re.py:198(search)
   135416    0.199    0.289  re.py:289(_compile)
     4000    0.147    6.543  tokenize.py:481(tokenize)
       22    0.067    0.067  tokenize.py:376(isanyalpha)
                      6.543  TOTAL
```

Per line: **full `tokenize()` = 1,430 µs**, of which
**`handles_nonbreaking_prefixes` = 729 µs (51%)**.

#### 3.5.1 `islower()` is 53% of the library's runtime

```python
def islower(self, text):
    return not set(text).difference(set(self.IsLower))    # rebuilds a 2,253-element set. every call.

def isanyalpha(self, text):
    return any(set(text).intersection(set(self.IsAlpha)))  # rebuilds a 10,862-element set. every call.
```

**[measured]**, hoisting the constant set out of the call:

| | per call | after hoisting | speedup |
|---|---:|---:|---:|
| `islower("the")` | **607.5 µs** | 1.5 µs | **392×** |
| `isanyalpha("The")` | **2,965.3 µs** | 1.1 µs | **2,663×** |

This is a third instance of the pattern the concurrent audit already found twice
(`replace_multidots`, `has_numeric_only`): a constant rebuilt inside a hot loop.
It is not adversarial-input-only — it fires on **ordinary text, every line**.
Hoisting both sets to `__init__` as `frozenset`s is a **two-line change worth
roughly 2× end-to-end**.

#### 3.5.2 The Amdahl ceiling on any regex accelerator

**[measured]**, running the identical 19-step regex subset of the chain both ways
over the same 20,000 lines:

| | seconds | lines/s | MB/s |
|---|---:|---:|---:|
| Full `tokenize()` | 19.29 | 1,037 | 0.087 |
| **Just the 19 regex substitutions** (Python `re`) | **5.77** | 3,465 | 0.291 |

**Regex accounts for ~30% of `tokenize()`.** Amdahl's law therefore caps any
regex accelerator at roughly 1/(1 − 0.30) ≈ 1.4×.

**This was then measured directly rather than left as an inference.** Every one
of `tokenize()`'s 23 substitutions was replaced with a no-op — modelling a
perfect accelerator that is instantaneous, free, and has no transfer cost —
and the full pipeline re-run over the same 20,000 lines: **[measured]**

| | seconds | lines/s | vs baseline |
|---|---:|---:|---:|
| Baseline `tokenize()` | 17.796 | 1,124 | 1.00× |
| **All 23 regex subs → no-op** (free accelerator) | 10.433 | 1,917 | **1.71×** |
| **`islower`/`isanyalpha` sets hoisted** (2-line fix) | **8.554** | **2,338** | **2.08×** |

> **The ceiling on *any* regex accelerator — GPU, FPGA, or magic — is 1.71× on
> this workload. A two-line bug fix delivers 2.08×.**

(The measured 1.71× exceeds the 1.4× inference because the no-op also skips work
downstream: shorter intermediate strings make the surviving Python steps cheaper.
It is therefore a generous upper bound, which is what an upper bound should be.)

The remaining ~70% is
`handles_nonbreaking_prefixes` (a per-token Python loop with list/dict membership
tests and a `re.search` per token), the two multidot passes (which use `sub()`
with a **Python callable** — a per-match interpreter callback), `.strip()`,
`.split()`, and the protected-pattern `str.replace` loop. **None of it moves to a
GPU without a rewrite of the algorithm, not just a port of the regexes.**

### 3.6 The existing joblib parallelism does not scale

**[measured]** through `sacremoses.util.parallelize_preprocess` — the exact path
`cli.py`'s `-j` flag uses:

| `n_jobs` | seconds | lines/s | speedup | efficiency |
|---:|---:|---:|---:|---:|
| 1 | 19.29 | 1,037 | 1.00× | 100% |
| 2 | 23.35 | 857 | **0.83×** | 41% |
| 4 | 19.07 | 1,049 | **1.01×** | 25% |
| 8 | 23.08 | 866 | **0.84×** | 10% |

**`-j 4` buys nothing (1.01×), and `-j 2` and `-j 8` are net losses.** The cause
is in `util.py:208`:

```python
return Parallel(n_jobs=processes)(delayed(func)(line) for line in iterator)
```

One joblib task **per line** — a 66-character payload — through process-based
`loky`, which must pickle the `MosesTokenizer` (carrying ~100 KB of inlined
character classes) to each worker. On a 1.2 GHz 4-core part the IPC and pickling
overhead consumes the entire win.

This matters for the GPU question in two directions. It removes "we already
parallelise, the next step is a GPU" from the argument — **the CPU parallelism
does not currently work**. And chunking lines into batches before dispatching is
the *same* fix the GPU design would need, obtainable here for a few lines of code.

### 3.7 Incidental findings

Not fixed (this investigation writes no library code); noted for whoever owns them.

- **`penn_tokenize()` is dead on arrival.** **[measured]**
  ```python
  >>> MosesTokenizer('en').penn_tokenize('Hello and/or world.')
  AttributeError: 'str' object has no attribute 'sub'
  ```
  `MOSES_PENN_REGEXES_1` contains `INTRATOKEN_SLASHES`, an **uncompiled `str`**,
  so `for regexp, substitution in ...: regexp.sub(...)` raises. Its replacement
  text `"$1 \@\/\@ $2"` uses Perl `$1` syntax, so it never worked in Python. Four
  constants are raw strings masquerading as live rules: `INTRATOKEN_SLASHES`,
  `LEFT_STRIP`, `MID_STRIP`, `RIGHT_STRIP` (the latter three are commented as
  reference-only; `INTRATOKEN_SLASHES` is not).
- **`normalize()` calls module-level `re.sub()` with uncompiled patterns** — 45
  per line. `re`'s internal cache prevents recompilation, but the profile shows
  the cost: 135,416 calls to `re._compile` and `re.search` in a 4,000-line run.
  Pre-compiling at `__init__` removes them.
- **CPython's cyclic GC costs 28% on bulk runs.** **[measured]** 20,000 lines:
  26.004 s retaining token lists, **18.707 s with `gc.disable()`** — ~800,000
  short-lived string objects trigger repeated collections. `gc.freeze()` after
  setup, or disabling GC around the CLI's bulk loop, buys **1.28×** for free.

---

## 4. The arithmetic on the proposed design

Design under review: shard a file → move shards to GPU → run substitutions on
GPU → collect results.

### 4.1 PCIe transfer is *not* the problem

This is the usual first objection and here it is simply wrong. A batch of *B*
bytes costs ~2*B* over PCIe (H2D + D2H). At a realistic **~25 GB/s** achievable
pinned-memory bandwidth on Gen4 ×16:

| Batch | Round-trip | sacremoses CPU time for the same bytes **[measured]** | Ratio |
|---|---:|---:|---:|
| 1 MB | ~80 µs | ~11.5 s | **~140,000×** |
| 64 MB | ~5.1 ms | ~12 min | ~140,000× |
| 1 GB | ~80 ms | ~3.2 h | ~140,000× |

Break-even is on the order of **tens of kilobytes**; any realistic shard clears
it by four orders of magnitude.

But sanity-check it against a *competent* baseline rather than this one. At
HuggingFace-`tokenizers` speed (~50 MB/s), 1 MB is 20 ms of CPU against 80 µs of
transfer — still 250×. Even against a hypothetical Hyperscan-class 1 GB/s
scanner, 1 MB is 1 ms against 80 µs. **Text is compute-heavy per byte; PCIe
genuinely is not the bottleneck.** Do not reject the design on transfer grounds.

### 4.2 Chain depth × kernel launches × allocations

23 (or 68) *dependent* substitutions cannot be fused into one kernel: each
rewrites the whole string and the next must see the rewritten bytes. Each becomes
a separate launch with a **global barrier**.

Worse, a variable-length string rewrite is not one kernel. cuDF's strings-column
model (Arrow-like: an `offsets` array plus a packed `chars` buffer) forces a
**two-pass** structure per substitution:

1. every thread computes its output *size*;
2. an **exclusive scan** builds the new offsets array;
3. a **device allocation** for the new chars buffer;
4. every thread writes its output.

So one logical `.sub()` ≈ 2 kernels + 1 scan + 1 allocation. A 68-step pipeline
is **~200 kernel launches and ~68 device allocations per batch**, each with a
full-column barrier, each materialising a complete new copy of the column.

Allocation count is not a theoretical worry: the GPUTOK paper reports **70–80% of
CUDA API time was memory allocation** in a GPU tokenizer [literature]. At
~5–10 µs/launch, ~200 launches is 1–2 ms of overhead per batch — amortisable —
but the allocations and repeated full-column materialisation scale *with* batch
size rather than being amortised by it.

### 4.3 Variable-length output: solved, but not free

The "fixed-shape tensors" worry is real but **already solved** by the offsets
design — that is exactly how cuDF represents strings and why it can do regex
replace at all. Variable-length output is **not** a fundamental blocker.

The cost is that every step allocates and copies the entire column. A 68-step
chain is 68 complete rewrites of the dataset in VRAM.

### 4.4 Divergence, occupancy — and cuDF's 2,048-byte cliff

This is where the design dies, and the evidence is cuDF's own source.

**[source-verified]** `cpp/src/strings/regex/` in `NVIDIA/cudf` (`main`):

```cpp
// regex.cuh
constexpr int32_t MAX_SHARED_MEM = 2048;  ///< Memory size for storing prog instruction data

// regexec.cpp
int32_t reprog_device::compute_shared_memory_size() const {
  return _prog_size < MAX_SHARED_MEM ? static_cast<int32_t>(_prog_size) : 0;
}
std::size_t reprog_device::compute_working_memory_size(int32_t num_threads, int32_t insts_count) {
  return relist::alloc_size(insts_count, num_threads) * 2;
}
```

`_prog_size` is `memsize + sizeof(reprog_device)` where
`memsize = insts_size + startids_size + classes_size`, and `classes_size`
includes each class's **variable-length `reclass_range` table** — one 8-byte
range pair per range (verified in `regexec.cpp`'s flat-buffer packing loop).

Now apply §3.4. `PAD_NOT_ISALNUM` runs on **every line** and is built on
`IsAlnum` = **461 ranges**. Its class table alone is `461 × 8 = 3,688 bytes` —
**1.8× over the 2,048-byte budget**, before instructions and start-ids.

**Consequence: `compute_shared_memory_size()` returns 0 for sacremoses' core
patterns.** The compiled program is not staged in shared memory; every
instruction fetch, for every character, for every thread, goes to **global
memory**. The hottest rule in the library lands in cuDF's slow path by
construction.

A second mechanism compounds it — `compute_strided_working_memory()` **halves the
thread count** until the per-thread NFA state buffer fits a budget:

```cpp
auto thread_count = rows;
auto buffer_size  = working_memory_size(thread_count);
while ((buffer_size > requested_max_size) && (thread_count > min_rows)) {
  thread_count = thread_count / 2;
  buffer_size  = working_memory_size(thread_count);
}
```

Working memory is `≈ 2 × insts_count × num_threads × ~10 bytes`, so a large
instruction count **directly reduces how many strings are processed
concurrently**. Big patterns do not merely run slower per thread — they
*deliberately throttle GPU parallelism*.

Finally, ordinary divergence applies and is measurable here. cuDF assigns roughly
one thread per row and a warp retires at its slowest lane. With **median 66 chars
but max 2,541** **[measured]**, one long line can stall its 32-lane warp for
~38× the useful work. (Length-sorting the column first mitigates this; cuDF does
not do it for you.) Regex matching also diverges *within* a character — each lane
follows a different automaton transition — so lanes diverge even on equal-length
strings. This is the same effect the ASPLOS 2020 analysis of NFA execution on
GPUs identifies as the core difficulty [literature].

### 4.5 Apple Silicon / unified memory

Unified memory drives the §4.1 transfer term to ~zero — but §4.1 showed that term
was already negligible, so **unified memory removes a cost nobody was paying**.
It does nothing about chain depth, allocation churn, divergence, or the ~70%
non-regex Python work.

Practically, **[measured]**, on this machine there is no GPU path at all (no
CUDA, MPS `False`, MLX unavailable). And even on an M-series Mac, **MLX and Metal
ship no regex engine** — you would be writing an NFA simulator in Metal Shading
Language from scratch, with no cuDF to lean on. Strictly more work than the CUDA
path, for a smaller ecosystem.

---

## 5. Prior art

### 5.1 Shipping products

**NVIDIA cuDF / RAPIDS — GPU regex is real.** libcudf has a bespoke CUDA regex
engine (not RE2) backing `Series.str.replace(regex=True)`, `.str.contains`,
`.str.extract`, `replace_with_backrefs`.
*Supported*: literals, `.`, `|`, classes/ranges/negation, `\d \w \s \D \W \S`,
anchors `^ $ \A \Z`, `\b`/`\B`, greedy **and** lazy quantifiers, capturing and
non-capturing groups, **backreferences `\1`–`\99` / `${1}`–`${99}` in the
replacement**, flags IGNORECASE / MULTILINE / EXT_NEWLINE / ASCII.
*Not supported*: lookahead/lookbehind, named groups, atomic groups, possessive
quantifiers, inline `(?i)`. Lookarounds were **closed as `wontfix`**
(rapidsai/cudf#3100). [shipping product doc / GitHub issue]

**`cudf.pandas` falls back silently.** *"Attribute lookups and method calls are
first attempted on the GPU… If that fails, the operation is attempted on the
CPU."* A lookahead in `.str.replace()` therefore runs on **CPU pandas** after a
full GPU→host copy — worse than plain pandas. Guard with
`CUDF_PANDAS_FAIL_ON_FALLBACK=1`. [official doc]

**cuDF subword tokenizer (`nvtext::subword_tokenize`) — the strongest existing
proof point, and the most instructive.** It is genuinely fast, and the reason is
that **it contains no arbitrary regex**: a fixed character-normalisation table, a
whitespace split, then WordPiece longest-match-first against a **perfect-hashed**
vocabulary. Implementation details that make it work [source-verified]: a
precomputed sdbm hash of `"##"` (`hashtag_hash = 2296000`) so continuations are
looked up without materialising strings, and an **invertible** hash
(`mod_inverse = 24320495251391`) so shrinking a candidate by one character is
O(1) instead of a rehash.
Its limits are equally instructive: *"requires about **21× the number of
character bytes** in the input strings column as working memory"*; normalisation
tables *"are currently hardcoded inside libcudf"*; `padding != "max_length"`,
`return_token_type_ids`, and `add_special_tokens` without truncation all raise
`NotImplementedError`; exceeding `max_num_rows` is *"undefined"* behaviour; and a
vocabulary with an sdbm collision **cannot be used at all** (build-time assert).

**Crucially, it is being retired — and not for being slow.** The stated reason is
inflexibility: the offline perfect-hash step made adopting arbitrary pretrained
tokenizers painful, so the replacement decomposes into `CharacterNormalizer` +
`WordPieceVocabulary` over a plain `vocab.txt` (cudf#14294, #17507). In the same
period `nvtext::byte_pair_encoding` was **deprecated** (cudf#21157, 26.04) with
the maintainer rationale *"a complex algorithm that is difficult to maintain as
libcudf evolves"*, and `edit_distance_matrix` was **removed outright** (#22644,
26.08).

> **The boundary of "GPU-fast text" is hardening around table- and
> delimiter-driven ops, and libcudf is actively shedding the complex sequential
> algorithms. A Moses substitution cascade is squarely in the class being shed.**

**NVIDIA NeMo Curator — GPU is dedup + neural classifiers only.** Fuzzy/exact/
semantic dedup and DeBERTa-class classifiers run on CUDA; the 30+ heuristic
filters, the regex text cleaners (`NewlineNormalizer`, `UrlRemover`), and PII
redaction are CPU. `PiiModifier`'s `device="gpu"` accelerates the spaCy/
transformers NER model, not Presidio's regex recognizers. [official doc]

**NVIDIA DALI — irrelevant.** Image, video and audio only; **zero text
operators**. [source code]

**HuggingFace `tokenizers` — CPU-only, confirmed.** No CUDA dependency in
`Cargo.toml` (`rayon`, `regex`, `fancy-regex`, `daachorse`); the GPU-training
request (#499) was closed unimplemented. Parallelism is rayon, **on by default**
when `TOKENIZERS_PARALLELISM` is unset. [source code]

**Hyperscan / Vectorscan (Intel) — CPU SIMD, and it definitively cannot
substitute.** It is a *streaming multi-pattern match reporter*. Its entire output
channel is one callback carrying **four integers**:

```c
int (*match_event_handler)(unsigned int id, unsigned long long from,
                           unsigned long long to, unsigned int flags, void *context);
```

It never returns matched text. The docs say *"capturing is **ignored**"*; the
NSDI '19 paper says *"the 'capturing' functionality of regexes, **which Hyperscan
does not support at all**."* Chimera adds capture **offsets** via PCRE — still no
substitution, block-mode only. Also unsupported: backreferences, **all
lookaround**, atomic grouping, possessive quantifiers, `\K`, `\R`, `\C`. And its
semantics are *all matches, end offsets, unordered* — **not leftmost-longest**;
greedy/non-greedy is a **no-op**, so `/foo.*bar/` on `fooxyzbarbar` returns
**two** matches where PCRE returns one.

**It is the wrong shape for a rewriting tokenizer**, and the honest speedup
numbers are 1.8–2.3× over PCRE2-JIT and 8.4–13.5× over `RE2::Set` (the widely
quoted 40×/183× compare a *multi-pattern* engine against one-pattern-at-a-time
PCRE).

**Status matters if anyone plans to depend on it:** open source is **frozen at
5.4** (last release v5.4.2, April 2023); **5.5+ is Intel Proprietary**. The live
path is **Vectorscan** (BSD, ARM NEON/SVE2 + Power VSX, in Debian/Fedora),
created because Intel rejected the ARM ports. Same limitations inherited.
[official doc / source code]

**Why Intel chose CPU over GPU — on the record**, NSDI '19 §7.1:

> *"We investigated GPU-based regex matching, but it imposed **unacceptable
> latency and system complexity**."*

### 5.2 Research with code

- **BlockBPE: Parallel BPE Tokenization** (You, ICML 2025 ES-FoMo workshop,
  [arXiv 2507.11941](https://arxiv.org/abs/2507.11941)) — **the single most
  relevant paper to this question.** To make BPE run on a GPU the authors
  **delete the regex pre-tokenization step entirely** ("leads to small loss in
  generation quality, but enables highly parallelized token merges within thread
  blocks"), reducing to O(nd). Claims up to 2× tiktoken / 2.5× HF at high batch.
  *Independent confirmation that the regex stage is the part that does not port.*
- **GPUTOK: GPU Accelerated Byte Level BPE Tokenization**
  ([arXiv 2603.02597](https://arxiv.org/pdf/2603.02597), 2026) — BlockBPE-style
  kernel; ~1.7× tiktoken, ~7.6× HF GPT-2. Reports **70–80% of CUDA API time is
  memory allocation**. Code: `venugopalkadamba/gpu-tokenizer`.
#### The number that should govern any "regex on GPU?" decision

**ngAP (ASPLOS 2024, Best Paper) Table 6** — absolute throughput in **MB/s on a
single 1 MB input stream**: [literature]

| App | **HyperScan (CPU)** | NFA-CG | AsyncAP | GPU-NFA | ngAP-best |
|---|---:|---:|---:|---:|---:|
| Snort | **31.54** | — | 4.29 | — | 1.83 |
| YARA | **61.31** | 0.25 | 1.52 | 0.79 | 3.35 |
| ExactMatch | **1177.86** | 1.66 | 35.71 | 1.74 | 133.28 |
| Bro217 | **285.88** | 1.69 | 39.06 | 1.49 | 111.66 |
| PowerEN | **84.03** | 1.58 | 0.08 | 1.08 | 1.16 |

**On a single stream, one CPU core running Hyperscan beats every GPU automata
engine on every workload — often by 10–70×.** The GPU papers' headline wins come
*only* from batching hundreds of concurrent streams. ANMLZoo's own conclusion
says why: these architectures *"generally benefit from **input stream
parallelism, rather than automata-level parallelism**."*

This is decisive for sacremoses. The proposed design has **one** logical stream
per substitution step, and the parallelism available is across *lines* — but each
line is 66 characters (§3.3), so the per-stream work is far below what these
engines need to amortise. sacremoses sits on the worst point of this curve.

#### The GPU headroom ceiling

**BitGen (MICRO 2025)** measured an **RTX 3090** against a **32-core Xeon
Platinum 8562Y+** and got **1.7× over multi-threaded Hyperscan** — on the *Yara*
workload, single-threaded Hyperscan (793.7 MB/s) **beat the entire GPU** (638.3).
The paper states the structural reason: the 3090 has only **~4.5× the integer
throughput** of the Xeon (17.8 vs 3.9 TIOPS). **Regex is integer and branch work,
not floating-point**, so GPUs have nothing like the headroom they enjoy in ML.
[literature]

#### Beware the baseline-inflation chain

These published claims are **not mutually consistent**, and the pattern is
instructive:

| Engine | Venue | Claim | Baseline |
|---|---|---|---|
| GPU-NFA / HotStart | ASPLOS 2020 | 26.5× | iNFAnt |
| AsyncAP | SIGMETRICS 2023 | 58× / 2.4× | GPU-NFA |
| ngAP | ASPLOS 2024 | 39.5× / 13.2× / 7.9×; **11.5×** | NFA-CG / AsyncAP / GPU-NFA; **Hyperscan** |
| BitGen | MICRO 2025 | 19.5× over ngAP; **1.7×** | ngAP; **MT-Hyperscan** |

ngAP claims 11.5× over Hyperscan; BitGen then claims 19.5× over ngAP but only
1.7× over multi-threaded Hyperscan. Both cannot be right — ngAP's Hyperscan
comparison counted 4 **timeouts** (on Hamming/Levenshtein automata, which are not
regex) as losses. **Most of this field measures against other GPU engines, not
against good CPU code.** Geoff Langdale, who did Intel's GPGPU regex work before
building Hyperscan, put it directly: *"Many papers about amazing GPGPU or FPGA
speedups (50x, 100x, 1000x) over software depend mostly on **having bad software
for comparison**."*

That warning applies to this repository with unusual force, because §2 measured
sacremoses at **0.09 MB/s** — this *is* the bad software for comparison.

- **"Why GPUs are Slow at Executing NFAs and How to Make them Faster"**
  (Liu, Pai, Jog, **ASPLOS 2020**) — the canonical diagnosis, and it is *both*
  memory and utilisation: (1) the transition table is O(N·256), too large for
  on-chip memory — iNFAnt issues **25×** and NFA-CG **18×** more global loads
  than an ideal input-only kernel; (2) **only 0.39% of NFA states are active on
  average** (max 3.05%), so one-thread-per-state wastes nearly every thread;
  (3) the obvious fix — a worklist — is *slower* (114 KB/s vs 143 KB/s), because
  some states are always active and NFAs need a **barrier every input symbol**
  with very little work per symbol. That last point is the general form of §4.2's
  68-barriers problem.
- **iNFAnt** (Cascarano et al., SIGCOMM CCR 2010) — the original NFA-on-GPU line.
  Independent reproductions: **0.01–6.07 MB/s** across 16 apps (ASPLOS 2020,
  Quadro P6000); as low as 143 KB/s on Brill. The maintained fork `iNFAnt2` was
  last pushed **2017** — dead.
- **ANMLZoo** (Wadden et al., IISWC 2016) / **AutomataZoo** (IISWC 2018) and
  **VASim** — the benchmark suites for automata processing.
- **Micron Automata Processor** (Dlugosch et al., IEEE TPDS 2014; and Fang,
  Hoang, Becchi, Chien, *Unified Automata Processor*, MICRO 2015) — purpose-built
  automata silicon: 133 MHz, 7.5 ns/symbol, 49,152 states/chip, ideal
  133.33 MB/s/chip. **Correction to an earlier draft of this document:** I could
  **not** verify a discontinuation announcement, and do not assert one. What is
  on the record is ICS 2017 stating the AP was *"not yet available on the
  market,"* and that every subsequent paper **models or projects** the AP rather
  than measuring one. What *is* verified is that its real throughput collapses
  once its **50 ms per-batch reconfiguration** is counted: **0.36 MB/s**
  (ClamAV), 0.82 (YARA), 4.21 (Snort) — i.e. dedicated automata silicon lands
  *below* this machine's Arrow/RE2 CPU number (1.374 MB/s, §6).
- **Rewrite-cascade composition theory** — Kaplan & Kay 1994, *Regular Models of
  Phonological Rule Systems* (Computational Linguistics); Mohri & Sproat 1996,
  *An Efficient Compiler for Weighted Rewrite Rules* (ACL). **A cascade of ordered
  rewrite rules can be compiled into a single finite-state transducer.** This is
  the theoretical basis for §7's "what would have to be true", and the reason
  §3.4's 17-equivalence-class result matters.

### 5.3 Claims to distrust

- **"483× faster than HuggingFace"** (NVIDIA blog, 2021) traces to a single 2020
  gist: no GPU/CPU model or versions recorded, RMM pre-warmed with a 10 GB pool
  so allocation is excluded from the cuDF side, and the input is 100 distinct
  sentences repeated 1,000× (unusually cache-friendly). **Cite the 6.2× vs the
  Rust tokenizer instead** — that is the defensible apples-to-apples number. The
  same blog notes cuDF is *slower* than both baselines below ~100 records.
- **"RE2 is faster than Python `re`"** — only sometimes. pyre2's own README:
  *"for very simple substitutions, I've found that occasionally python's regular
  `re` module is actually faster."* RE2 has high per-call setup; it wins on large
  haystacks and backtracking-prone patterns.
- **"the `regex` module is faster than `re`"** — no. `rebar` geometric means:
  `python/re` **42.10** vs `python/regex` **43.26** (lower is better). Its real
  advantages are features and **GIL release**.

### 5.4 The negative result

**No GPU implementation of the Moses tokenizer exists** — not for
`tokenizer.perl`, not for sacremoses, not for `mosestokenizer`. Searched
explicitly; nothing found. The fastest implementation of any kind is
`Helsinki-NLP/opus-fast-mosestokenizer`, a **C++/CPU** reimplementation. Even
`tokenizer.perl`'s own parallelism is line-batched CPU threads
(`-threads` + `-lines`, default 2,000 lines/thread).

That absence is evidence. This is a well-known bottleneck in a field
(machine translation) that has had abundant GPUs and strong throughput incentives
for a decade. Nobody has built it, and BlockBPE tells us why: when people put
tokenizers on GPUs, **the regex stage is the part they remove**.

**The database field reached the same conclusion.** There is no VLDB/SIGMOD/ICDE
paper on GPU regex in the modern era, and the flagship GPU-analytics paper says
why outright — **Crystal** (Shanbhag, Madden, Yu, SIGMOD 2020) dictionary-encodes
strings away entirely and states: *"Handling arbitrary strings and array data
types efficiently on GPUs is **still an open problem**."* Zero mentions of regex.
HeavyDB/OmniSci ships `REGEXP_LIKE`/`REGEXP_REPLACE` but runs them on **CPU
`boost::regex`** over distinct dictionary entries, sending only the id→id map to
the GPU.

**And cuDF's own roadmap is heading toward the CPU engines' primitive**, not away
from it: open issue rapidsai/cudf#15611 proposes making its regex faster by
adopting **shift-or** — the bit-parallel technique Hyperscan, HybridSA and BitGen
all converge on. Note also cuDF's weak contract: *"Invalid regex patterns will
result in **undefined behavior**."*

---

## 6. The honest alternatives, ranked

Speedups marked **[measured]** are from this machine; others are cited estimates.

| # | Option | Speedup | Effort | Risk | Verdict |
|---|---|---|---|---|---|
| **1** | **Hoist `set(IsLower)` / `set(IsAlpha)` out of `islower`/`isanyalpha`** | **2.08× end-to-end** **[measured]** (392× / 2,663× on the calls; 53% of runtime) | **~2 lines** | **Very low** — pure hoist, no semantic change | **Do this first, today** |
| **2** | **`gc.freeze()` / disable GC around the CLI's bulk loop** | **1.28×** **[measured]** | ~5 lines | Low | **Do it** |
| **3** | **Raise the Python floor to 3.12** | **2–3× on `re.sub` with group references** [literature] — i.e. most of the chain | Packaging only | Low (drops 3.9–3.11 users) | **Strongly worth it** |
| **4** | Pre-compile `normalize()`'s 45 patterns at `__init__` | Removes 135k+ `re._compile`/`search` calls per 4k lines **[measured]** | ~10 lines | Very low | Do it |
| **5** | **Fix the joblib batching** (chunk lines; stop sending one task per 66-char line) | Should recover real multi-core scaling; currently **1.01× on 4 cores** **[measured]** | ~20 lines | Low | Do it — the current `-j` is a no-op |
| **6** | Fix remaining algorithmic waste (the audit's `replace_multidots` ~400×, `has_numeric_only` ~28,000× on adversarial input) | Large on adversarial input, modest on typical | Small | Low | Already in progress |
| **7** | **Adopt / bind `opus-fast-mosestokenizer` (C++)** | **15× vs sacremoses** [literature] | Medium — binding + parity testing | Medium — behavioural drift vs the Python | **Best single lever if 1–5 aren't enough** |
| **8** | **Batch via Arrow/RE2 compute kernels on CPU** | **4.72× on the identical 19-step chain** **[measured]** (0.291 → 1.374 MB/s) | Medium — must port the non-regex 70% too | Medium — RE2 rejects `\g<1>` and the one lookahead | Real, but §3.5.2 caps it |
| **9** | Rewrite the hot loop in Rust/Cython | Plausibly 10–50× | High | Medium | Only if #7 is rejected |
| **10** | Swap in `re2`/`hyperscan` on CPU | Unclear; RE2 loses on simple patterns, Hyperscan **cannot substitute** | Medium | High — feature mismatch | **Not recommended** |
| **11** | **Compile the chain to a single FST/DFA** | Potentially **100×+**, and it makes every other option moot | **Very high** — research-grade | High | The only genuinely exciting option (§7) |
| **12** | **The proposed GPU design** | **≤1.71× ceiling** **[measured]** — less than option #1 — and only after porting the non-regex 70% | **Very high** — plus no local hardware | **High** | **Do not build** |

### Note on option 8 vs option 12

Option 8 is the honest CPU control for the GPU hypothesis: **the same programming
model** (columnar batch, offsets-and-chars layout, N whole-column passes, RE2-class
engine) minus the GPU. It delivers **4.72×** on the regex portion. **[measured]**

```
Arrow/RE2, 19 column passes :   1.222s ->  16,361 lines/s   1.374 MB/s
Python re, same 19 steps    :   5.772s ->   3,465 lines/s   0.291 MB/s
```

Note what dominates: `DEDUPLICATE_SPACE` (`\s+`, a **3-character** pattern) is
**37% of the Arrow chain's cost** across its two invocations, while the 22 KB
`EN_SPECIFIC_*` monsters are ~6% each. Pattern *size* is not what costs; match
*frequency* is. A GPU port would inherit exactly this profile.

Since §3.5.2 measured the whole-regex ceiling at **1.71×**, that 4.72× on the
regex portion becomes roughly **1.3–1.5× end-to-end** — for a substantial
rewrite. **A GPU is bounded by the same 1.71×.** That is the argument in one
line: option 8 gets most of what option 12 could ever get, on hardware you
already own, and both are beaten by option 1.

---

## 7. Recommendation

### 7.1 Do not build the proposed design

Stated plainly, with the reasons ranked by how decisive they are:

1. **The Amdahl ceiling is 1.71×, and a two-line fix delivers 2.08×.**
   **[measured]** — not inferred; see §3.5.2 and §7.3. Everything else is detail.
2. **~70% of the work does not port.** `handles_nonbreaking_prefixes`, the
   callable-replacement multidot passes, `.strip()`, `.split()`, and the
   protected-pattern loop are Python control flow and dictionary lookups.
   Porting them is *rewriting the tokenizer*, not accelerating it.
3. **The library's real bottleneck is not regex at all** — it is one method
   rebuilding a set 5,344 times per 4,000 lines. **[measured]**
4. **cuDF's fast path is closed to these patterns** by the 2,048-byte
   shared-memory limit. **[source-verified]**
5. **No local hardware.** No CUDA, no MPS, no MLX. **[measured]**
6. **Nobody has done it, and the people closest to the problem deleted the regex
   stage instead** (BlockBPE) or are retiring their complex-sequential GPU text
   ops (cuDF).
7. **The automata literature says CPU wins at this scale.** On a *single stream*,
   CPU Hyperscan beats every GPU automata engine on every workload, by 10–70×
   (ngAP Table 6, §5.2). GPU wins in that field require batching hundreds of
   concurrent streams; sacremoses offers 66-character lines. And when someone
   finally benchmarked a full RTX 3090 against *multi-threaded* Hyperscan rather
   than against other GPU engines, the answer was **1.7×** (BitGen, MICRO 2025).

Note carefully which arguments are **not** on this list: PCIe cost (§4.1),
variable-length output (§4.3), and regex expressibility (§3.2). Those are the
objections people expect, and they are all wrong. The design fails for better
reasons.

### 7.2 Do this instead — staged

- **Stage 0 (hours).** Hoist the two sets (#1) — **2.08× measured**. Freeze GC in
  the CLI bulk path (#2) — 1.28× measured. Pre-compile the normalizer's patterns
  (#4). Fix joblib batching (#5). Re-run the §2 benchmark and publish the new
  baseline. **Stage 0 alone should clear the 1.71× that the entire GPU project
  is capped at.**
- **Stage 1 (days).** Raise the Python floor to 3.12 (#3). Re-measure — Stage 0
  and Stage 1 interact, since #3 targets `re.sub` and #1 targets everything else.
- **Stage 2 (weeks, only if still needed).** Evaluate
  `opus-fast-mosestokenizer` (#7) as an optional fast path with the Python
  implementation retained as the correctness oracle, and a differential test
  across the corpus. This is the standard "vendor the fast path, keep our
  implementation as the guarantee" shape.
- **Stage 3 (research).** §7.4, and only with eyes open.

### 7.3 The cheapest experiment that would kill the GPU idea — already run

The right first experiment is not to write CUDA; it is to measure the *ceiling*,
because no amount of CUDA skill can beat a no-op. **That experiment has been run,
and it took minutes rather than the day budgeted.**

> **Experiment.** Replace every one of `tokenize()`'s 23 regex substitutions with
> a no-op — a perfect accelerator: instantaneous, free, no transfer cost — while
> leaving everything else (the non-breaking-prefix loop, multidots, strip/split)
> intact. Run the full pipeline over 20,000 lines of `big.txt`.
>
> **Metric.** End-to-end `tokenize()` throughput, lines/s.
>
> **Success threshold set in advance.** Pursue the GPU **only if throughput
> exceeds ~5,600 lines/s (5× over the 1,124 lines/s baseline)** — the return
> needed to justify the build cost in §4.
>
> **Result: 1,917 lines/s — 1.71×.** **[measured]**
> **Verdict: the GPU hypothesis is dead on arithmetic**, for zero hardware and
> minutes of work.
>
> **Control, run at the same time:** hoisting two constant sets (§3.5.1) gives
> **2,338 lines/s — 2.08×**, beating the perfect accelerator with a two-line diff.

Script: `noop.py` (Appendix B). Anyone can re-run it to check this conclusion.

**The experiment that would still be worth running,** if someone wants to
accelerate the regex portion regardless (option 8 in §6, capped at ~1.3–1.7×
end-to-end): take the single
hottest pattern (`DEDUPLICATE_SPACE` — 37% of the Arrow chain, §6) and run it via
cuDF on a rented GPU against a 100 MB column, comparing against Arrow/RE2 on
CPU. Instrument with `CUDF_PANDAS_FAIL_ON_FALLBACK=1` so a silent CPU fallback
cannot masquerade as a result. **Success = >10× over the Arrow/RE2 CPU number on
the same batch**, which is what would be needed to overcome §4.2's per-step
allocation and barrier costs across 68 steps.

### 7.4 What would have to be true for the GPU idea to become viable

The GPU is the wrong answer to the question as posed. But there is a *right*
question nearby, and this investigation turned up the key evidence for it.

**If the 68-step dependent chain were compiled into a single finite-state
transducer, everything changes.** The dependency chain — the thing that forces 68
barriers — collapses into one left-to-right pass. This is not speculation;
Kaplan & Kay (1994) prove it, and the wording is almost a description of this
problem:

> *"since **regular relations are closed under serial composition, a finite set
> of rules applying to each other's output in an ordered sequence also defines a
> regular relation**."* — Kaplan & Kay 1994, *Regular Models of Phonological Rule
> Systems*

Their stated goal is verbatim the goal here: *"eliminate the cascade altogether,
representing the information in the grammar as a whole in **a single more unified
device**, namely, a finite-state transducer."*

The usual objection is alphabet blowup, and **§3.4 measured that objection
away**:

> The entire sacremoses ruleset distinguishes only **17 character equivalence
> classes** across 14,147 code points. **[measured]**

An FST over ~17–50 symbols is entirely tractable. That reframes the problem
completely:

- The 2,048-byte cuDF cliff (§4.4) disappears — a range-coalesced,
  equivalence-classed automaton is small.
- The 68 barriers become **one** pass.
- Divergence drops sharply: a DFA transition is a table lookup, not a branchy
  backtracking search.

**But note the punchline: if you achieve that, you no longer need the GPU.** A
single-pass DFA over a 17-symbol alphabet runs at hundreds of MB/s to GB/s **on
one CPU core** — three to four orders of magnitude above today's 0.09 MB/s, and
far past anything the GPU design was aiming at. The GPU would be optimising a
problem that compilation already solved.

So the honest statement of the condition is:

> The GPU becomes viable if the rule cascade is first compiled to a single
> automaton **and** throughput is still the binding constraint afterwards — i.e.
> if you are processing enough text that even ~1 GB/s per core is insufficient
> and you cannot simply add cores. At that point you would be running a
> **DFA**, not a chain of `re.sub` calls, and the GPU literature (§5.2) applies
> favourably rather than adversely.

#### The conditions the theory imposes, and where the cost lands

This is sound only under real preconditions, and they must be checked against the
Moses rules before anyone commits: [literature]

- **No rule may apply unboundedly to its own output** (Johnson 1972; Kaplan &
  Kay). A self-feeding rewrite is **not** a regular relation and cannot be
  compiled at all. Most Moses rules are single-pass character-context rewrites and
  should qualify — but `replace_multidots`/`restore_multidots` and the
  `DOTMULTI`/`_ELLIPSIS_` marker round-trip are exactly the shape that needs
  checking first.
- **The blowup is in *states*, not the alphabet, and it lands at compile time.**
  Composing transducers with *n* and *m* states is O(n·m); across a chain of N
  rules that multiplies. Kaplan & Kay's construction needs six transducers per
  rule and, per Mohri & Sproat, *"two intersections of automata, two distinct
  subtractions, and **nine complementations**"* — and **complementation requires
  determinisation**, the exponential subset construction. Mohri & Sproat 1996
  improve this to five transducers with *"only… the determinization of **3
  automata** and additional work **linear** (time and space) in the size of the
  determinized automata."*
- **Transducers are not always determinisable** (unlike FSAs) — Mohri's twins
  property.

**The winning pattern is compile-once, run-forever**; compiling per request is
not viable. That fits sacremoses perfectly: the rules are fixed per language, so
the FST could be built at package-build time and shipped.

**Do not write this from scratch.** The tooling is mature and this is its exact
use case: **OpenFst**, **Thrax**, and especially **Pynini** (Gorman 2016) —
NVIDIA's own NeMo text normalization is built on Pynini. Also HFST, foma, Xerox
XFST/lexc.

Remaining honest caveats: (a) `handles_nonbreaking_prefixes` does dictionary
lookup with lookahead to the *next token*, so it must be modelled as transducer
state or kept as a separate pass; (b) state blowup needs empirical
determinisation/minimisation to confirm, not just the alphabet argument;
(c) exact output parity would need differential testing across all 39 bundled
languages. None is a reason not to try — they are why this is Stage 3, not
Stage 0.

---

## Appendix A — sources

Read during this investigation. Marked by type.

**cuDF / RAPIDS (source code, read directly)**
- `NVIDIA/cudf` → `cpp/src/strings/regex/regex.cuh` — `MAX_SHARED_MEM = 2048`, `relist::alloc_size`
- `NVIDIA/cudf` → `cpp/src/strings/regex/regexec.cpp` — `compute_shared_memory_size`, `compute_working_memory_size`, `compute_strided_working_memory`, flat class-table packing
- `NVIDIA/cudf` → `cpp/src/text/subword/wordpiece_tokenizer.cu` — `hashtag_hash`, invertible `prev_sdbm_hash`
- `NVIDIA/cudf` → `cpp/src/text/subword/detail/cp_data.h` — normalisation table (not NFD)
- `cpp/include/nvtext/subword_tokenize.hpp` — the 21× working-memory requirement

**cuDF / RAPIDS (official docs)**
- https://docs.rapids.ai/api/cudf/stable/libcudf_docs/md_regex/ — regex feature support
- https://docs.rapids.ai/api/libcudf/stable/group__nvtext__tokenize.html — nvtext ops
- https://docs.rapids.ai/api/cudf/stable/cudf_pandas/how-it-works/ — GPU→CPU fallback
- https://docs.rapids.ai/api/cudf/stable/cudf_pandas/faq/ — fallback known issues
- https://developer.nvidia.com/blog/mastering-the-cudf-pandas-profiler-for-gpu-acceleration/

**cuDF (GitHub issues / PRs)**
- rapidsai/cudf#3100 — regex lookarounds, **closed `wontfix`**
- NVIDIA/cudf#21157 — `byte_pair_encoding` deprecated (26.04)
- NVIDIA/cudf#22644 — `edit_distance_matrix` removed (26.08)
- NVIDIA/cudf#14294, #17507 — move off hashed vocab; why subword_tokenize is being retired
- NVIDIA/cudf#12403 — sdbm hash overflow
- NVIDIA/cudf#23403 — real Unicode normalisation still an open PR
- rapidsai/cudf#9657 — BPE tokenizer FEA, still open

**Blog claims (treat with the caveats in §5.3)**
- https://developer.nvidia.com/blog/run-state-of-the-art-nlp-workloads-at-scale-with-rapids-huggingface-and-dask/ — the "483×" claim
- https://medium.com/rapids-ai/preprocess-your-training-data-at-lightspeed-with-our-gpu-based-tokenizer-for-bert-language-models-561cf9c46e15 — 271× / 6.2×, and "slower below 100 records"

**Papers**
- BlockBPE — https://arxiv.org/abs/2507.11941 (ICML 2025 ES-FoMo) — *deletes regex pre-tokenization to reach the GPU*
- GPUTOK — https://arxiv.org/pdf/2603.02597 — 70–80% of CUDA API time is allocation
- LoPT — https://arxiv.org/abs/2511.04952 (ACL 2026) — parallel tokenization chunk-boundary correctness
- FlashText — https://arxiv.org/abs/1711.00046 — trie beats regex above ~500 keywords
- **"Why GPUs are Slow at Executing NFAs and How to Make them Faster"** — Liu, Pai, Jog, ASPLOS 2020 — https://dl.acm.org/doi/10.1145/3373376.3378471 · free PDF https://par.nsf.gov/servlets/purl/10206640 · artifact https://github.com/bigwater/gpunfa-artifact (dead, 2022)
- **ngAP** (ASPLOS 2024, Best Paper) — https://getianao.github.io/papers/asplos24ngap.pdf — **Table 6 is the single-stream throughput table quoted in §5.2**
- **BitGen** (MICRO 2025) — https://getianao.github.io/papers/micro25bitgen.pdf · https://github.com/getianao/BitGen — RTX 3090 vs 32-core Xeon; 1.7× over MT-Hyperscan; the TIOPS headroom argument
- **AsyncAP** — SIGMETRICS 2023; **NFA-CG** — Zu et al., PPoPP 2012
- **Demystifying Automata Processing: GPUs, FPGAs or Micron's AP?** — Nourian, Wang, Yu, Feng, Becchi, ICS 2017 — https://research.cs.vt.edu/synergy/pubs/papers/yu-demysifying-ics17.pdf — *"GPUs underperform FPGAs by up to a factor 900×"*; GPU 105–185 W vs FPGA 2.09–2.36 W
- iNFAnt — Cascarano et al., SIGCOMM CCR 2010 — https://dl.acm.org/doi/10.1145/1880153.1880157 (paywalled; **its own headline speedup could not be verified** — independent reproductions used instead). Fork: https://github.com/vqd8a/iNFAnt2 (dead, 2017)
- **ANMLZoo** — Wadden et al., IISWC 2016 — https://www.cs.virginia.edu/~skadron/Papers/IISWC16_ANMLZoo.pdf · https://github.com/jackwadden/ANMLZoo
- **AutomataZoo** — Wadden et al., IISWC 2018 — https://github.com/tjt7a/AutomataZoo
- **VASim** — Wadden & Skadron, UVA TR CS2016-03 — https://github.com/jackwadden/VASim
- Micron Automata Processor — Dlugosch et al., IEEE TPDS 2014; Fang, Hoang, Becchi, Chien, *Unified Automata Processor*, MICRO 2015. (**Discontinuation NOT verified** — see §5.2.)
- **Hyperscan** — Wang, Hong, Chang, Park, Langdale, Hu, Zhu, **NSDI '19** — https://www.usenix.org/conference/nsdi19/presentation/wang-xiang — §7.1 on rejecting GPU; §7.3 on no capturing
- **Crystal** — Shanbhag, Madden, Yu, SIGMOD 2020 — https://arxiv.org/pdf/2003.01178 — *"arbitrary strings… still an open problem"*
- **Kaplan & Kay 1994**, *Regular Models of Phonological Rule Systems*, Computational Linguistics 20(3):331–378 — https://aclanthology.org/J94-3001/
- **Mohri & Sproat 1996**, *An Efficient Compiler for Weighted Rewrite Rules*, ACL '96 pp. 231–238 — https://aclanthology.org/P96-1031/
- Russ Cox, *Regular Expression Matching Can Be Simple And Fast* — https://swtch.com/~rsc/regexp/regexp1.html — why RE2 excludes backreferences (matching with backreferences is **NP-complete**)
- FST toolkits: OpenFst (openfst.org), Thrax, **Pynini** (Gorman 2016 — used by NVIDIA NeMo text normalization), HFST, foma

**Other implementations**
- https://github.com/huggingface/tokenizers — README "1 GB in <20 s on a server's CPU"; `Cargo.toml` has no CUDA
- https://github.com/Helsinki-NLP/opus-fast-mosestokenizer — "15× faster than sacremoses"
- https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl
- https://github.com/BurntSushi/rebar — cross-engine benchmarks (`re` 42.10, `regex` 43.26, `re2` 10.39, `hyperscan` 2.37)
- https://github.com/google/re2/wiki/Syntax — RE2 unsupported constructs
- https://github.com/intel/hyperscan/issues/17 — no capture groups
- https://intel.github.io/hyperscan/dev-reference/chimera.html — Chimera's cost
- https://github.com/VectorCamp/vectorscan — the maintained BSD fork (Hyperscan 5.5+ is proprietary)
- https://www.phoronix.com/news/Intel-Hyperscan-Now-Proprietary — 2024-05-10
- https://github.com/rapidsai/cudf/issues/15611 — cuDF proposing shift-or, converging on Hyperscan's primitive
- https://arrow.apache.org/docs/python/generated/pyarrow.compute.replace_substring_regex.html — RE2-backed
- https://docs.python.org/3/whatsnew/3.12.html — *"Speed up the regular expression substitution… for replacement strings containing group references by 2–3 times"*
- https://docs.python.org/3/whatsnew/3.11.html — atomic groups, computed-goto engine
- https://docs.nvidia.com/nemo/curator/v26.02/reference/infra/gpu-processing.md — what NeMo Curator puts on GPU
- https://github.com/NVIDIA/DALI — image/video/audio only

---

## Appendix B — reproducing the measurements

Scripts live outside the repo, in this session's scratchpad:
`/private/tmp/claude-501/-Users-alvas-jean-claude/87d3c1ec-6a02-4f1d-88f6-0182d48208b6/scratchpad/gpu_study/`

| Script | Produces |
|---|---|
| `audit_regex.py` | §3.2 feature audit, §3.4 pattern sizes, class sizes |
| `chain.py` | §3.1 chain length + ordered trace, §3.3 line statistics |
| `bench.py` | §2 single-threaded baseline table |
| `bench2.py` | §3.7 GC finding, the `return_str` variance |
| `bench_par.py` | §3.6 joblib scaling table |
| `prof.py` | §3.5 profile, per-substitution costs |
| `arrow_chain.py` | §6 Arrow/RE2 vs Python `re` on the identical chain |
| `alphabet.py` | §3.4 range coalescing, §7.4 the 17 equivalence classes |
| `re2_feat.py` | §3.2 RE2 expressibility test (27/28) |
| **`noop.py`** | **§3.5.2 / §7.3 — the Amdahl ceiling experiment (1.71×) and the 2-line-fix control (2.08×)** |

All were run with `cwd=/Users/alvas/git-stuff/sacremoses` and
`sys.path.insert(0, ...)` pointing at the working tree, so they measure the
audited code rather than the copy in `site-packages`.

**Reproduction caveats.** Absolute timings on a 1.2 GHz Ice Lake-Y part under
thermal pressure vary by 30%+ run to run — see §2. Ratios (speedups, percentages,
efficiency) are far more stable than absolute seconds and are what the
conclusions rest on. The `-j` scaling table in particular should be re-run on a
machine with more than 4 physical cores before being generalised; **the finding
that per-line joblib dispatch is the wrong granularity is structural, but the
exact numbers are this machine's.**
