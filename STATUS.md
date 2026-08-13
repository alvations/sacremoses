# sacremoses — work status

Snapshot of what has been done, what is running, and what is planned.
Baseline: upstream `hplt-project/sacremoses` master `53ce2d9`.

Test suite: **106 passed, 1 skipped, 225 subtests** (was 36/1/98 at baseline).
Security harness: **9 attacks, 0 vulnerable** (8 of 9 vulnerable at baseline).

---

## DONE — landed in the working tree

Nothing is committed yet. `git status` shows the full change set.

### Security fixes

| ID | Fix | Impact |
|----|-----|--------|
| A1 | `replace_multidots`/`restore_multidots` rewritten single-pass | `'.'*16000`: **16.76s → 0.042s** (~400x) |
| A2 | `has_numeric_only` uses `\s` not `[\s]+` | `' '*50000`: **96.5s → 0.0034s** (~28,000x) |
| A3 | Protected-span placeholder escaped against the input | Forged `THISISPROTECTED000` no longer relocates a URL |
| A4 | `assert` bound replaced with `ValueError` | Bound now holds under `python -O` |
| A5/A9 | Data-as-code: 29 char classes + 39 prefix lists are Python modules | No runtime file loading; traversal is a `KeyError` |
| A6 | Truecase model parse errors name file + line | Untrusted `.truemodel` no longer raises raw `AttributeError` |
| A7 | `save_model` writes `O_EXCL\|O_NOFOLLOW` `0600` + atomic `os.replace` | No symlink follow, no world-readable model, no TOCTOU |
| — | Dependency floors | `joblib>=1.2.0` (CVE-2022-21797), `tqdm>=4.66.3` (CVE-2024-34062) |

### Correctness fixes

- **`is_cjk` range boundaries** — comparison was exclusive on closed intervals,
  dropping the first and last codepoint of all 11 CJK ranges (22 characters,
  including **U+AC00**, the most common Korean syllable). `가` and `각` — adjacent
  syllables — detokenized differently. All 11 ranges now consistent.
- **`perl_parity=True` no longer mutates class attributes**, which had leaked
  into every other normalizer in the process.
- **`open()` now pins `encoding="utf-8"`** in `tokenize.py`, `sent_tokenize.py`,
  `subwords.py`. 33 of 39 prefix lists are non-ASCII; six are not cp1252-
  decodable (hard error on a default Windows locale) and six decode to silent
  mojibake.
- **`train-truecase` produced a character-level model** — the CLI passed raw
  strings to `train()`, which expects `list(list(str))`, so each string was
  iterated per character. `truecase` was consequently a **complete no-op**.
  Now word-level; peak RSS on 50k lines dropped 313 MiB → 146 MiB.
- **`click.__version__` removed** (deprecated in 8.3, gone in 9.1). The pre-8
  branch was dead — `resultcallback` does not exist in installed Click.

### Verification

- `audit/attacks.py` — 9 attacks, benign ones kept. **0 vulnerable** on this
  tree, **8 vulnerable** on `53ce2d9` via `--ref`.
- `audit/differential.py` — **704 outputs across 12 languages** vs pristine:
  678 identical, 26 expected, **0 unexpected**.
- `audit/measure.py` — complexity classifier with a self-check.
- `sacremoses/test/test_security_harness.py` — the CI-enforced subset.
- `sacremoses/test/test_cli.py` — 40 tests, new; there were none.
- **Languages explicitly verified**: zh/ja/th/ko tokenization unchanged;
  truecase remains a no-op for caseless scripts (Han/Kana/Thai are Unicode
  category `Lo`; `SKIP_LETTERS_REGEX` matches only `Ll/Lu/Lt`, so they can
  never enter a model).

### Documentation

- `README.md` restored to the minimal pre-HPLT scope (`4203e7d`), with stale
  snippets corrected: undefined `tokens` variable, `train()` → `train_from_file()`,
  swapped `use_known` outputs, stale `--help` text.
- `audit/README.md` — findings, evidence, and the transferable lessons.

---

## IN PROGRESS — research agents still running

None of these have written to the package. All are plan-only unless noted.

| Topic | Deliverable | State |
|-------|-------------|-------|
| Improvement roadmap | `audit/ROADMAP.md` — 47 items, 127 verified URLs | **done** |
| GPU-accelerated regex feasibility | `audit/GPU-TOKENIZATION.md` — verdict: **don't build it** | **done** |
| Performance sweep | 8 optimizations, see above | **done** |
| Parallelism library survey | alternatives to joblib | running |

### GPU tokenization — answered, and the answer is no

Settled by an upper-bound experiment rather than an argument: replacing **all 23
of `tokenize()`'s regex substitutions with no-ops** — i.e. a perfect, infinitely
fast, zero-transfer accelerator — yields only **1.71x**. Hoisting the two
constant sets out of `islower`/`isanyalpha` yields **2.08x**.

**A two-line bug fix beats the theoretical ceiling of any regex accelerator that
could ever exist.** No hardware was needed to determine that.

The usual objections are all wrong, and worth recording so they are not
re-litigated: PCIe transfer is ~140,000x cheaper than the compute it replaces
(break-even is tens of kilobytes); regex expressibility is fine (27 of 28 real
patterns accepted by RE2); variable-length output is a solved problem (cuDF's
offsets/chars layout). The design fails for a different reason — regex is only
~30% of runtime, and 66-character lines sit on the worst point of the
GPU-automata curve. Corroborated by ngAP (ASPLOS'24) Table 6: on a single
stream, CPU Hyperscan beats every GPU automata engine by 10-70x, and a full
RTX 3090 manages 1.7x over multi-threaded Hyperscan.

The one direction that survives: the ruleset distinguishes only **17 character
equivalence classes** across 14,147 code points, so FST composition is tractable
(Kaplan & Kay 1994). But that punchline undercuts the premise — a one-pass DFA
runs at GB/s **on a single CPU core**.

---

## PLANNED — reported, decided, not yet applied

### High value, well evidenced

**Single-process optimizations.** Eight changes, each verified byte-identical
across 12 languages × 1,510 sentences over tokenize/penn/detokenize/normalize
and their flag combinations. Combined: **tokenize 1.98x, detokenize 2.35x,
normalize 2.01x.** Ranked by payoff:

| # | Where | Change | Speedup |
|---|-------|--------|---------|
| 1 | `normalize.py:187-200` | Precompile `substitutions` in `__init__`; 62 of 84 literal patterns → `str.replace`. 45% of normalize time was pure `re`-module dispatch — upstream's regex hoisting never touched this file. | **2.01x** |
| 2 | `tokenize.py:373-377` | Cache `frozenset(IsLower/IsAlpha)` **per instance**. They rebuild a set over a ~100K-char string on *every call*: 610 calls cost 2.4s. | **1.77x** |
| 3 | `tokenize.py:789,794,805,848` | First-char guards before the four anchored detok regexes | **1.72x** |
| 4 | `tokenize.py:780,782` | `token[0] < "ᄀ"` guard before `is_cjk` | 1.14x |
| 5 | `tokenize.py:424` | `^(\S+)\.$` → `endswith(".")` | 1.09x |
| 6 | `tokenize.py:355,367` | `".." not in text` / `"MULTI" not in text` guards | 1.08x |
| 7 | `tokenize.py:745` | `unescape_xml` → guarded `str.replace` chain | 1.06x |
| 8 | `tokenize.py:322,437` | Prefix lists → `frozenset` (changes a public attribute's type) | 1.05x |

Two traps recorded with them: #2's cache **must be per-instance**, because
`__init__` does `self.IsAlpha += cjk_chars` for `zh/ja/ko/cjk` — a class-level
cache is silently wrong for CJK. And #4 is 2.44x on Latin but **0.78x on pure
CJK**, so it is a trade, not a free win.

Measured and **discarded**: `escape_xml` → `str.translate` (1.01x), `is_cjk` via
`bisect` (0.93x on Latin — slower than the current linear scan), hoisting
`self.*` out of the detok loop (1.00x), list-append + `join` (1.00x).

**The harness earned its keep here.** A ninth candidate rewrote
`replace_multidots` as a loop; it reintroduced the CWE-407 quadratic and drove
`test_multidot_run_is_linear` from 3s to 30–230s. The regression test caught it
before it could land.

**Multi-process.**

1. **Range-compress the CJK character classes.** They enumerate every codepoint;
   93,249 → 1,277 chars gives **47x compile / 583x match** with identical
   output, and removes the pickling cost that makes `-j` slow.
2. **Replace joblib with `ProcessPoolExecutor` + `initializer=`.** For zh/ja/ko
   the current joblib path is **1.86x slower than serial** (354s vs 190s on 20k
   lines): `partial(moses.tokenize, ...)` pickles the whole tokenizer including
   ~93–186k-char compiled regexes, and the `re` cache always misses. Proposed
   design measured **5.09x faster** on zh.
3. **`-j -1` silently runs serially.** `cli.py` tests `processes == 1` but
   `util.py` tests `processes <= 1`, so joblib's own "all cores" idiom is a
   no-op with no warning.
4. **`tokenize.py:319`** — `if line not in self.NONBREAKING_PREFIXES` is O(n²)
   when loading a custom prefixes file.

### Packaging / CI

5. **`pyproject.toml`, delete `setup.py`.** Built and verified: wheel
   1677 KB → 846 KB, `twine check --strict` passes, data modules ship
   automatically. **Must land atomically with committing the `_data_*.py`
   files** — `corpus.py` imports them, so a wheel without them dies on import.
6. **Replace `.github/workflows/test.yaml`; delete `.appveyor.yml`** (dead
   4.5 years; its `CMD_IN_ENV` references a file not in the repo). Travis is
   already gone. Current workflow has **no `pull_request` trigger**, so fork PRs
   never run CI. Add Windows/macOS legs, a data-drift guard, and a packaging job.
7. **Trusted Publishing for PyPI** (OIDC, no stored token), publishing as
   **alvations**.
8. **Decide `python_requires`.** `>=3.8` is stated but 3.8 and 3.9 are both EOL,
   and `click>=8.0` means 3.8/3.9 silently resolve to click 8.1.x. Nothing in
   the code forces a bump — this is policy.

### Pre-existing defects confirmed by running them (not introduced here)

Reported by a peer research session; each re-tested locally rather than taken
on trust, and two of its claims did not survive that.

- **Combining marks are split out as punctuation, tearing words apart.**
  Marks of Unicode category `Mn` that are missing from `IsAlpha`/`IsAlnum` get
  padded by `PAD_NOT_ISALNUM`. Counted by enumerating every `Mn` codepoint per
  block rather than by sampling words:

  | Block | `Mn` marks | missing from `IsAlpha` | notable |
  |-------|-----------|------------------------|---------|
  | Thai | 16 | **7** | U+0E48–U+0E4B, all four tone marks |
  | Lao | 17 | **6** | U+0EC8–U+0ECB, the tone marks |
  | Khmer | 22 | **14** | U+17D2 coeng |
  | Tibetan | 74 | 11 | |
  | Devanagari | 23 | 4 | Vedic accents only — rare |
  | Myanmar | 28 | 3 | U+103A asat |

  Effect: `น้ำ` ("water") → `['น', '้', 'ำ']`; `ไม่` ("not") → `['ไม', '่']`;
  `ນ້ຳ` (Lao "water") → `['ນ', '້', 'ຳ']`. These are among the commonest words
  in each language.

  **Note on how this was nearly missed.** An initial spot-check used
  `สวัสดีครับ`, which happens to contain only *covered* marks, and it tokenized
  cleanly — leading to a wrong "Thai is fine" conclusion. Sampling a word tests
  the marks that word happens to use; only enumerating the block tests the
  class. Two independent agents reported the defect correctly; the spot-check
  was what was wrong.

  Tibetan's splits are mostly on U+0F0B tsek and U+0F0D shad, which are genuine
  delimiters — that part is defensible.

- **Irish and Catalan corrupt text on round-trip.** Perl groups `fr|it|ga|ca`
  in the tokenizer but only `fr|it` in the detokenizer; sacremoses has the
  grouping inverted on both sides. Measured:
  `"D'ith an cat"` → `"D 'ith an cat"`, `"L'home d'aquí"` → `"L 'home d' aquí"`.
  French and Italian round-trip correctly.
- **`MosesSentTokenizer` cannot be imported** — bare `raise NotImplementedError`
  in the class body (`sent_tokenize.py`). `split-sentences.perl` was never ported.
- **`penn_tokenize()` raises `AttributeError: 'str' object has no attribute
  'sub'`** — `INTRATOKEN_SLASHES` is an uncompiled tuple whose replacement is
  still Perl syntax. Upstream issue #151, open since 2024-07-19.
- **The Finnish/Swedish colon branch is dead text.** It sits inside a
  triple-quoted string headed `# FIXME!!!`, and `FI_SV_COLON_APOSTROPHE` is
  never defined anywhere (`hasattr` → `False`). Same for the Somali block.
  So `-l fi/sv/ca/tdt` silently get the generic path. No issue tracks this.

**Performance claim checked and refuted.** The peer flagged this tree as ~35%
slower than stock, but had compared 0.0.41 against a modified 0.1.1. Measured
against the correct baseline (`53ce2d9`, 5000 lines of `big.txt`, min of 3):

| | pristine | hardened | |
|---|---|---|---|
| tokenize | 6.414s | 6.154s | −4% |
| detokenize | 1.161s | 0.993s | −14% |
| normalize | 1.312s | 1.159s | −12% |

The hardening is neutral-to-faster on ordinary text, and 400x–28,000x faster on
adversarial input.

### Known, deliberately not done

- **`click.echo` can raise `UnicodeEncodeError` on a non-UTF8 Windows console**
  (reproduced with `PYTHONIOENCODING=cp932`/`cp437`). Fixing it changes stdout
  buffering semantics and would override a deliberate `PYTHONIOENCODING` — a
  real interface decision for the maintainer, not a silent patch.
- **CLI buffers all of stdin** (`cli.py` eager `list()` per stage). Streaming was
  assessed and rejected: `truecase_file` needs a genuine second pass, every
  progress bar would regress to a bare counter, and with `-j > 1` joblib
  materialises the result list anyway.
- **`--version` fails in a source checkout** (needs distribution metadata).
  Pre-existing; works from a pip install.
- **`is_flag=True, default=True`** on `-x`/`-q`/`-d` means those flags cannot
  turn their option off. Pre-existing interface wart; changing it would break
  the CLI contract.
- **`learn_truecase_weights` gates the first token on `i == 1`**, never true at
  `i == 0` — a pre-existing off-by-one in `truecase.py` that makes `-p` not work
  as documented. Not fixed; flagged.

---

## Next actions

1. **Commit.** `_data_perluniprops.py`, `_data_nonbreaking_prefixes.py`,
   `tools/`, `audit/`, `test_security_harness.py`, `test_cli.py` are all
   untracked; `corpus.py` already imports the data modules, so CI fails at
   import until they land. Commit as **alvations** only.
2. Decide `package_data` / `pyproject.toml` (item 5) — same commit as item 1.
3. Apply the two measured performance wins (items 1–2 above).
4. Read `audit/ROADMAP.md` and `audit/GPU-TOKENIZATION.md` when they land.
