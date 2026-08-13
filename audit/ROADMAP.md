# sacremoses — improvement roadmap

**Status: PLAN ONLY. Nothing in this document has been implemented.**

Prepared against `hplt-project/sacremoses` master `53ce2d9`, working tree at
`/Users/alvas/git-stuff/sacremoses` (with the in-progress security/performance audit applied).
Date of research: 2026-08-13.

Every item below carries a **Source** column. Sources are one of:

* a GitHub issue/PR number with a URL,
* a `file:line` in this repo (marked *"no external source — own analysis"*),
* a line in the Perl original (`moses-smt/mosesdecoder`),
* a downstream project's issue or code.

Where I could not verify something, it says so explicitly. **No issue number or URL in this
document was invented**; every one was retrieved with `gh` or `curl` during this research pass.

Findings are tagged **[verified]** when I reproduced them by running code in this checkout, and
**[unverified]** when they are a reading of the source that I did not execute.

---

## 1. Current state, and the biggest risks

### What sacremoses is

A pure-Python port of four Moses Perl scripts — `tokenizer.perl`, `detokenizer.perl`,
`normalize-punctuation.perl`, `train-truecaser.perl`/`truecase.perl`/`detruecase.perl` — created in
April 2018 because `nltk.tokenize.moses` had to be removed for licence reasons
([nltk/nltk#2000](https://github.com/nltk/nltk/issues/2000)). It is ~2,750 lines of hand-written
library code across 12 modules (`tokenize` 908, `truecase` 569, `cli` 333, `normalize` 213,
`util` 208, `subwords` 184, `corpus` 150, `indic` 73, `sent_tokenize` 71, `chinese` 27, plus the
two audit-generated `_data_*` modules), with **106 passing tests + 1 skipped** in ~128 s.

### Its position

* **195.9M total PyPI downloads, ~3.1M/month** ([pepy.tech](https://pepy.tech/projects/sacremoses));
  **28,393 GitHub repositories** list it as a dependency.
* It is **no longer a hard dependency of `transformers`**. It was in `install_requires` from
  **v2.0.0 (Sept 2019) to v4.18.x** — about 2.7 years — and was removed by commit `30ca5299` /
  [huggingface/transformers#17049](https://github.com/huggingface/transformers/pull/17049)
  ("Make the sacremoses dependency optional", merged 2022-05-02, first shipped in **v4.19.0,
  2022-05-12**). It was *demoted to an optional per-tokenizer import*, **not vendored and not
  replaced**. That 2019–2022 window is where the enormous install base came from, and it is why
  tens of thousands of pinned requirements files still name it.
* Exactly **five** `transformers` tokenizers still import it (verified against `main`):
  `xlm`, `fsmt`, `flaubert`, `biogpt` (all raise a hard `ImportError` without it), and `marian`
  (**silently** degrades — see risk R2).
* The load-bearing API is tiny. Across 100 files / 72 distinct downstream repos found by
  `gh search code "from sacremoses import"`: `MosesTokenizer` 69, `MosesDetokenizer` 57,
  `MosesPunctNormalizer` 19, `MosesTruecaser` 7.

### The five biggest risks

**R1 — Any change to tokenizer output is a silent data-corruption event downstream.**
`XLMTokenizer`, `FSMTTokenizer`, `FlaubertTokenizer` and `BioGptTokenizer` call
`MosesTokenizer(...).tokenize(...)` at inference time with *fixed* arguments baked into
`transformers`. A published model's vocabulary was built against one specific sacremoses output.
Change the output and inference silently mismatches training — no exception, just worse
translations. This already happened once at patch level:
[dmlc/gluon-nlp#1548](https://github.com/dmlc/gluon-nlp/issues/1548) —
`MosesPunctNormalizer` changed output between 0.0.43 and 0.0.44 and a downstream project had to
file an issue about it. **This risk dominates the prioritisation in §4.** The exact call sites,
which form the compatibility contract:

| transformers file | call |
|---|---|
| `models/fsmt/tokenization_fsmt.py:246` | `.tokenize(text, aggressive_dash_splits=True, return_str=False, escape=True)` |
| `models/biogpt/tokenization_biogpt.py:146` | `.tokenize(text, aggressive_dash_splits=True, return_str=False, escape=True)` |
| `models/xlm/tokenization_xlm.py:284` | `.tokenize(text, return_str=False, escape=False)` |
| `models/fsmt/tokenization_fsmt.py:240` | `MosesPunctNormalizer(lang=lang).normalize(text)` |
| `models/marian/tokenization_marian.py:174` | `MosesPunctNormalizer(source_lang).normalize` |
| `models/fsmt/…:254`, `biogpt/…:154` | `MosesDetokenizer(lang=lang).detokenize(tokens)` |

**R2 — `MarianTokenizer` degrades silently to the identity function.** Verified in
`transformers/models/marian/tokenization_marian.py` on `main`:

```python
except (ImportError, FileNotFoundError):
    warnings.warn("Recommended: pip install sacremoses.")
    self.punc_normalizer = lambda x: x
```

Every Opus-MT/Marian translation therefore differs depending on whether an optional package
happens to be installed. sacremoses cannot fix this in `transformers`, but it is the reason
`MosesPunctNormalizer` output stability matters more than its 19 direct importers suggest.

**R3 — The project is effectively unmaintained, and the queue proves it.** Last code commit
2023-10-30; the only 2026 commit added an EU funding line. Last PyPI release **0.1.1, 2023-10-30**
(~2.8 years stale). 28 open issues, **median age 2,243 days (6.1 years)**, 13 with zero replies.
Four open PRs, including a measured **2x speedup unreviewed for 2.2 years**
([#149](https://github.com/hplt-project/sacremoses/pull/149)). The project's own test suite is
reported red ([#153](https://github.com/hplt-project/sacremoses/issues/153)).

**R4 — sacremoses is the slowest Moses tokenizer in common use.** Third-party benchmark
([mingruimingrui/fast-mosestokenizer bench README](https://github.com/mingruimingrui/fast-mosestokenizer/blob/master/bench/README.md),
IWSLT EN-ZH, 100k lines, i7-9750H):

| Tool | EN sent/sec | ZH sent/sec |
|---|---|---|
| fast-tokenizer (C++) | 127,758 (6.5x) | 153,884 (3.6x) |
| `tokenizer.perl` | 19,760 (1.0x) | 43,300 (1.0x) |
| **sacremoses** | **7,961 (0.4x)** | **582 (0.01x, timed out)** |

That is ~2.5x slower than the Perl it replaced. The OPUS/Marian ecosystem responded by maintaining
[Helsinki-NLP/opus-fast-mosestokenizer](https://github.com/Helsinki-NLP/opus-fast-mosestokenizer)
instead. **The good news: over half of that gap is one line** — see P1, which I measured at
**3.29x with byte-identical output**.
*(Counter-evidence, stated for honesty:* [bitextor/bicleaner#35](https://github.com/bitextor/bicleaner/issues/35)
*benchmarked sacremoses* faster *than the Perl-via-subprocess `mosestokenizer` — 46.7K vs 40.5K
sents/sec. The deficit is against C++ and against Perl-in-process, not against Perl-over-a-pipe.)*

**R5 — Packaging metadata is wrong in ways that leak into distro packages.** PyPI `0.1.1` reports
`author: ''`, `summary: 'SacreMoses'`, `license: ''`, and unconstrained
`requires_dist: ['regex','click','joblib','tqdm']`. **conda-forge still ships `0.0.53` and still
declares `LGPL-2.1-or-later`** ([feedstock#7](https://github.com/conda-forge/sacremoses-feedstock/issues/7),
open with zero replies since 2020-10-13; [feedstock#15](https://github.com/conda-forge/sacremoses-feedstock/pull/15),
the 0.1.1 bump, open since 2023-10-30). The licence confusion is what started the transformers
divorce in the first place
([transformers#2453](https://github.com/huggingface/transformers/issues/2453): *"sacremoses does not
have an accepted licence type for my system"*). **sacremoses is not in Debian at all** (verified:
`tracker.debian.org/pkg/sacremoses` → 404).

---

## 2. Proposed work items

Impact = value to users. Effort: S ≈ <½ day, M ≈ 1–3 days, L ≈ >3 days.
**Back-compat risk** is specifically *"could this change bytes that a `transformers` tokenizer
emits?"* — see R1.

### Performance

| ID | Title | Category | Impact | Effort | Back-compat risk | Source(s) |
|---|---|---|---|---|---|---|
| **P1** | Hoist `set(self.IsLower)` / `set(self.IsAlpha)` out of `islower()`/`isanyalpha()` into module-level `frozenset`s | perf | **High** | **S** | **None** (output-identical) | [PR #149](https://github.com/hplt-project/sacremoses/pull/149) "2x faster MosesTokenizer" (de9uch1, open since 2024-05-26) · own analysis of `sacremoses/tokenize.py:373-377` **[verified]** |
| **P2** | Precompile the ~9 `re.search` calls in `MosesDetokenizer.tokenize`'s per-token loop | perf | Med | S | **None** | [PR #148](https://github.com/hplt-project/sacremoses/pull/148) "MosesDetokenizer performance" (jelmervdl, open since 2023-10-30) · [PR #140](https://github.com/hplt-project/sacremoses/pull/140) (closed, superseded) · own analysis of `tokenize.py:796,816,818,819,839,840,850,864,883` |
| **P3** | Precompile `MosesPunctNormalizer.substitutions` once in `__init__` | perf | Med | S | **None** (output-identical) | own analysis of `sacremoses/normalize.py:196-198` **[verified]** |
| **P4** | Compile `^(\S+)\.$` and `^[0-9]+` once instead of per token | perf | Low-Med | S | **None** | own analysis of `tokenize.py:424,452` **[verified]** |
| **P5** | Stream `train_from_file` instead of `fin.readlines()` | perf/mem | Med | S | None | [#105](https://github.com/hplt-project/sacremoses/issues/105) "truecase training killed: 9" (open, OOM at 16GB) · [#55](https://github.com/hplt-project/sacremoses/issues/55) "Truecaser crashes for large corpora (>8M segments)" · own analysis of `truecase.py:196-200, 224-226` **[unverified]** |

**P1 evidence (measured in this checkout, CPython 3.9.12, macOS):**

`cProfile` over 1,500 lines of `big.txt`, `MosesTokenizer(lang='en').tokenize`:

```
       1500  0.139   9.778  tokenize.py:481(tokenize)
       1500  0.533   7.032  tokenize.py:418(handles_nonbreaking_prefixes)
       2870  5.266   5.283  tokenize.py:373(islower)      <-- 54% of total runtime
```

`islower()` is `not set(text).difference(set(self.IsLower))`. `self.IsLower` is a **2,253-character
string**; `self.IsAlpha` is **10,862 characters**. Both are rebuilt into a `set` on *every call*.
Measured cost: `isanyalpha` **11 ms/call**, `islower` **2.6 ms/call**.

Monkeypatching P1+P4 (no repo edits) over 3,000 real lines of `big.txt`:

```
tokenize   before: 11.61s (258 lines/s)   after: 3.53s (851 lines/s)   speedup 3.29x
output identical on all 3000 lines:  True
normalize  before:  2.50s               after: 1.75s               speedup 1.43x  (P3)
output identical:  True
```

This is the highest impact-per-effort item in the document, it is already written as
[PR #149](https://github.com/hplt-project/sacremoses/pull/149), and it carries **zero** back-compat
risk. It should be merged before anything else in this roadmap.

### Correctness — bugs with no back-compat cost

| ID | Title | Category | Impact | Effort | Back-compat risk | Source(s) |
|---|---|---|---|---|---|---|
| **C1** | `penn_tokenize()` raises `AttributeError` for **every** input and **every** language | correctness | **High** | S | **None** (nothing can depend on a method that always raises) | [#151](https://github.com/hplt-project/sacremoses/issues/151) "Regular expression error of penn_tokenize" (open since 2024-07-19) **[verified — still broken]** |
| **C2** | `penn_tokenize()` ignores per-language `__init__` overrides | correctness | Med | S | None (blocked behind C1) | [#151](https://github.com/hplt-project/sacremoses/issues/151), jelmervdl: *"two bugs here: 1) that expression not being compiled, and 2) changing lang doesn't update the penn expressions"* · own analysis of `tokenize.py:204-255` vs `341-349` |
| **C3** | Czech detokenizer crashes: `TypeError: 'list' object is not an iterator` | correctness | Med | S | **None** (currently a crash) | own analysis of `tokenize.py:844` vs `detokenizer.perl:146-147` **[verified]** — no external issue exists |
| **C4** | Czech decimal rule reads the **last two tokens of the sentence** instead of the two preceding tokens | correctness | Med | S | Low (`lang="cs"` only) | own analysis of `tokenize.py:816,818` vs `detokenizer.perl:132` **[verified]** — no external issue exists |
| **C5** | `MosesTruecaser.Titlecase_Letter` is a copy-paste of `Uppercase_Letter`; 31 Unicode titlecase characters are invisible to the truecaser | correctness | Low-Med | S | Low (truecaser only, 7 direct importers) | own analysis of `truecase.py:34` **[verified]** — no external issue exists |
| **C6** | `assert hasattr(self, "model")` is stripped by `python -O` | robustness | Low | S | None | own analysis of `truecase.py:251` **[unverified]** — same class as the protected-token assert the current audit already fixed, but a *different* assert |
| **C7** | `_write_casing` shadows its loop variable and passes 3 args to a 2-placeholder format string | maintainability | Low | S | None | own analysis of `truecase.py:429-437` **[unverified]** |

**C1 is the single most embarrassing open bug.** `MOSES_PENN_REGEXES_1` contains
`INTRATOKEN_SLASHES`, which at `tokenize.py:93-96` is a bare tuple of **strings** (never
`re.compile`d) whose replacement is still in **Perl** syntax (`r"$1 \@\/\@ $2"`). Verified:

```
>>> MosesTokenizer(lang='en').penn_tokenize("-LRB- This is very nice -RRB-")
AttributeError: 'str' object has no attribute 'sub'
>>> MosesTokenizer(lang='zh').penn_tokenize("hello world")
AttributeError: 'str' object has no attribute 'sub'
```

Downstream consequence found in the wild: `BLKSerene/Wordless` carries a commit
*"Utils: Remove Sacremoses's Penn Treebank tokenizer"*.

**C3/C4 are new findings.** Perl's `detokenizer.perl:146-147` uses a C-style index loop and does
`$i++` to skip the dash. The Python port at `tokenize.py:844` translated that into
`next(tokens, None)` — but `tokens` is a `list`, not an iterator. Verified:

```
>>> MosesDetokenizer(lang='cs').detokenize(["Poslal","e'","-","mail","dnes"])
TypeError: 'list' object is not an iterator
```

`detokenizer.perl:132` reads `$words[$i-2]` and `$words[$i-1]`; `tokenize.py:816,818` reads
`tokens[-2]` and `tokens[-1]`. Verified consequence:

```
>>> MosesDetokenizer(lang='cs').detokenize(["9","je","tady","7","."])
'9 je tady7.'          # "tady" and "7" wrongly glued
```

### Correctness — divergences from the Perl original (all change output)

These are real gaps against `tokenizer.perl`, but every one of them changes tokenizer output and
therefore lands squarely on risk R1. **My recommendation for all of them is a single opt-in
`perl_parity=True` constructor flag on `MosesTokenizer`, mirroring the one
`MosesPunctNormalizer` already has** ([PR #146](https://github.com/hplt-project/sacremoses/pull/146)
"Parity with perl normalize", merged) — not a change to the default path.

| ID | Title | Impact | Effort | Back-compat risk | Source(s) |
|---|---|---|---|---|---|
| **C8** | Missing Perl's *"split last word independently"* branch — a sentence-final non-breaking prefix is never split | **High** | S (2 lines) | **HIGH** | [#97](https://github.com/hplt-project/sacremoses/issues/97), [#21](https://github.com/hplt-project/sacremoses/issues/21), [#110](https://github.com/hplt-project/sacremoses/issues/110) · `tokenizer.perl:362-365` vs `tokenize.py:418-457` **[verified]** |
| **C9** | Finnish/Swedish colon rule is commented out, and the constants it names do not exist | Med | S | **HIGH** for `fi`/`sv` | `tokenizer.perl:261-268` vs `tokenize.py:534-548` (`# FIXME!!!`, references undefined `FI_SV_COLON_APOSTROPHE`, `FI_SV_COLON_NO_LOWER_FOLLOW`) **[verified]** |
| **C10** | Catalan middle-dot rule missing entirely | Med | S | **HIGH** for `ca` | `tokenizer.perl:275-281` **[verified]** · alvations in [#43](https://github.com/hplt-project/sacremoses/issues/43): *"having Catalan specific rules would be awesome! I've vested interest for Catalan text processing"* |
| **C11** | `ga` and `ca` are missing from the FR/IT apostrophe group — **breaks tokenize→detokenize round-tripping for Irish and Catalan** | **High** | S | **HIGH** for `ga`/`ca` | `tokenizer.perl:336` (`fr`\|`it`\|`ga`\|`ca`) vs `tokenize.py:572` (`["fr","it"]`) **[verified]** |
| **C12** | `so`/`tdt` glottal-stop rule commented out | Low | S | Med | `tokenizer.perl:340-346` vs `tokenize.py:575-578` |
| **C13** | Detokenizer's English-contraction rule drops Perl's left-context guard, so it fires more often than Perl | Low-Med | S | **HIGH** for `en` | `detokenizer.perl:128` vs `tokenize.py:802-807` (the guard is present but **commented out**) **[verified]** |
| **C14** | Detokenizer adds `ga` to the FR/IT right-shift group that Perl does not have | Low | S | Med | `detokenizer.perl:136` (`fr`\|`it`) vs `tokenize.py:826` (`["fr","it","ga"]`) **[verified]** |

**C13 evidence.** Perl guards the left-shift with `$words[$i-1] =~ /[\p{IsAlnum}]$/`
(`detokenizer.perl:128`); `tokenize.py:807` carries that guard as a **comment**. So sacremoses
left-shifts a contraction onto a non-alphanumeric predecessor, which Perl would not:

```
>>> MosesDetokenizer('en').detokenize(["--", "'ll", "go"])
"--'ll go"                                  # perl would leave a space
>>> MosesDetokenizer('en').detokenize(["(", "'s", "test"])
"('s test"
```

**C11 and C14 together are the same mistake made in opposite directions.** Perl's *tokenizer*
groups `fr|it|ga|ca`, and sacremoses' tokenizer drops `ga` and `ca` (C11). Perl's *detokenizer*
groups only `fr|it`, and sacremoses' detokenizer **adds** `ga` (C14). Verified:

```
lang=fr  ["d'", "esprit"] -> "d'esprit"
lang=ga  ["d'", "esprit"] -> "d'esprit"     # sacremoses; perl gives "d' esprit"
lang=es  ["d'", "esprit"] -> "d' esprit"    # the generic path = what perl does for ga
```

Net effect: Irish text is tokenized with the *generic* rule and then detokenized with the
*French* rule, so `MosesTokenizer('ga')` and `MosesDetokenizer('ga')` **do not round-trip — they
corrupt the text**. Verified:

```
ga  "D'ith an cat"     -> ['D', "'", 'ith', 'an', 'cat']  -> "D 'ith an cat"     roundtrip=False
ca  "L'amistat forta"  -> ['L', "'", 'amistat', 'forta']  -> "L 'amistat forta"  roundtrip=False
fr  "L'amitié forte"   -> ["L'", 'amitié', 'forte']       -> "L'amitié forte"    roundtrip=True
```

A space migrates across the apostrophe on every Irish and Catalan clitic. This raises **C11 from
"parity nit" to a real correctness bug**, and it is the strongest argument in this document for
doing C10/C11 rather than deferring them: unlike C8/C9, restoring Perl's behaviour here *fixes*
round-tripping rather than merely changing it. It is still an output change, so it still belongs
behind `perl_parity=True` — but `ga`/`ca` are the two languages where the current default is
indefensible.
| **C15** | No warning on unknown-language fallback to English prefixes | **High** | S | **None** (stderr only) | `tokenizer.perl:570-571` prints *"WARNING: No known abbreviations for language '%s', attempting fall-back to English version…"*; sacremoses is silent **[verified]** · [#125](https://github.com/hplt-project/sacremoses/issues/125) is exactly this confusion |

**C8 evidence.** Perl has an explicit first branch that sacremoses never ported:

```perl
if ($i == scalar(@words)-1) {
    # split last words independently as they are unlikely to be non-breaking prefixes
    $word = $pre." .";
}
```

Adding it reproduces the Perl reference output quoted verbatim in issue #97:

```
current:            ['John', 'vs.', 'Doe', ',', 'says', 'vs.']
with Perl branch:   ['John', 'vs.', 'Doe', ',', 'says', 'vs', '.']
perl (from #97):     John vs. Doe , says vs .
```

alvations wrote in [#97](https://github.com/hplt-project/sacremoses/issues/97) that this is
*"a difference kept on purpose"*. The Perl source shows it is not a design choice but a missing
branch — however, after seven years of shipped output, *keeping* it is now the safer default.
Hence: `perl_parity=True`, not a default change.

**C15 evidence.** Any unrecognised language silently gets the 105 English prefixes with no signal:

```
th nbp=105   km nbp=105   lo nbp=105   ar nbp=105   tr nbp=105   xx nbp=105   ja nbp=105
```

This matters more than it looks: `MarianTokenizer` passes `self.source_lang` straight through, and
Opus-MT models use codes like `mul`, `roa`, `cel`. Those all silently become English.

### Language coverage

sacremoses bundles **38 non-breaking-prefix files** (`sacremoses/data/nonbreaking_prefixes/`, plus a
`README.txt`), exposed under **76 keys** (ISO code + English name aliases):
`as bn ca cs de el en es et fi fr ga gu hi hu is it kn lt lv ml mni mr nl or pa pl pt ro ru sk sl sv ta tdt te yue zh`.
Everything else falls back to English **silently** (C15).

Per-language behaviour actually implemented in `sacremoses/tokenize.py`:

| Branch | Languages | Line |
|---|---|---|
| English apostrophe rules | `en` | `tokenize.py:569-571` |
| FR/IT apostrophe rules | `fr`, `it` | `tokenize.py:572-574` |
| Generic apostrophe split | everything else | `tokenize.py:579-581` |
| CJK added to `IsAlpha`/`IsAlnum` + 3 regexes recompiled | `zh`, `ja`, `ko`, `cjk` | `tokenize.py:328-349` |
| Detokenizer: skip CJK left-shift | `ko` | `tokenize.py:780` |
| Detokenizer: NBSP before `?!:;\%` | `fr` | `tokenize.py:796` |
| Detokenizer: decimal + `-li`/`e-mail` | `cs` | `tokenize.py:812-845` |
| Detokenizer: colon morphology | `fi` | `tokenize.py:881-889` |

| ID | Title | Impact | Effort | Back-compat risk | Source(s) |
|---|---|---|---|---|---|
| **L1** | Thai / Khmer / Lao / Burmese are shattered at every combining mark | **High** | M | **HIGH** (changes output) | own analysis + **[verified]**, same mechanism as [#42](https://github.com/hplt-project/sacremoses/issues/42) (Hindi) and [#73](https://github.com/hplt-project/sacremoses/issues/73) (Tamil, Russian) · fix pattern already exists in `sacremoses/indic.py` via [PR #103](https://github.com/hplt-project/sacremoses/pull/103) (merged) |
| **L2** | Document that `MosesTokenizer(lang='zh'/'ja')` is effectively a **pass-through** | **High** | S | **None** (docs) | [#120](https://github.com/hplt-project/sacremoses/issues/120), [#42](https://github.com/hplt-project/sacremoses/issues/42), [#125](https://github.com/hplt-project/sacremoses/issues/125) · XapaJIaMnu in #120: *"For tokenizing Chinese you should use something like `spacy_pkuseg`"* **[verified]** |
| **L3** | Chinese punctuation is split inconsistently: `。` and `、` stay glued, `，` `！` `？` split | Med | S | **HIGH** for `zh` | [#120](https://github.com/hplt-project/sacremoses/issues/120) "Chinese full stop 。 can't be split" (open since 2021-07-08) · own analysis of `tokenize.py:333` + `data/perluniprops/Han.txt` **[verified]** |
| **L4** | Finish Tetun Dili (`tdt`) — prefixes ship, code does not | Low-Med | S | Low (new language) | [PR #114](https://github.com/hplt-project/sacremoses/pull/114) (raphaelmerx, open since 2021-03-27) · [PR #144](https://github.com/hplt-project/sacremoses/pull/144) (BLKSerene, closed unmerged) · [mosesdecoder PR #224](https://github.com/moses-smt/mosesdecoder/pull/224) · `tokenizer.perl:269-274` **[verified: `nonbreaking_prefix.tdt` loads 191 prefixes but no `tdt` branch exists in `tokenize.py`]** |

**L1 evidence.** The generic `PAD_NOT_ISALNUM` at `tokenize.py:44` pads every character not in
`IsAlnum`. Thai/Lao/Khmer/Myanmar combining marks are not in `IsAlnum`, so words break apart:

```
th  22 chars ->  4 tokens: ['สวัสดีครับ', 'ผมชื', '่', 'อสมชาย']
km  20 chars ->  8 tokens: ['សួស', '្', 'តី', 'ខ', '្', 'ញុំឈ', '្', 'មោះសុខ']
lo  21 chars ->  6 tokens: ['ສະບາຍດີ', 'ຂ', '້', 'ອຍຊື', '່', 'ສົມຊາຍ']
my   9 chars ->  3 tokens: ['မင', '်', '္ဂလာပါ']
```

Khmer is the worst affected: 20 characters become 8 tokens, split at every *coeng* (U+17D2).

**The precise mechanism** (independently reproduced with different inputs): every character that
gets torn off is Unicode general category **`Mn` (nonspacing mark)** — U+0E48/U+0E49 Thai MAI EK /
MAI THO, U+17D2 KHMER COENG, U+0EC8/U+0EC9 Lao, U+103A Myanmar ASAT. The base consonants are in
`IsAlpha`; the marks are not. So `PAD_NOT_ISALNUM` (`tokenize.py:44`) pads them **as if they were
punctuation**, tearing tone marks off their bases and splitting Khmer subscript clusters at the
coeng. The output is therefore not merely over-segmented — it contains **invalid grapheme
clusters**, which is strictly worse than leaving the string alone. Confirmed:
`grep -i "thai\|khmer\|lao\|burmese\|myanmar"` over `tokenize.py` returns **zero hits**.

This is the identical failure mode as Hindi #42 and Tamil #73, and the identical fix shape as
`indic.py`'s `VIRAMAS`/`NUKTAS` (merged as PR #103): add the relevant combining marks to `IsAlnum`.
**Note honestly: `tokenizer.perl` has the same behaviour** — yannvgn confirmed this for Tamil in
[#73](https://github.com/hplt-project/sacremoses/issues/73). So this is *improving on* Perl, not
restoring parity, and it changes output. It should ship behind an opt-in flag or a major version.

**But consider not doing it at all.** ICU has solved exactly this set of languages for years. Per
the ICU user guide, ICU ships dictionary-based word boundaries for **Chinese, Japanese, Thai, Lao,
Khmer and Burmese**, automatically, with *"no separate API, and no extra programming steps
required"* — one `BreakIterator.createWordInstance()`. That is precisely the six languages
sacremoses either corrupts (th/km/lo/my) or passes through (zh/ja). It is reachable from Python
today via **PyICU 2.16.2 (2026-03-20)**. A defensible position is that sacremoses should **detect
and refuse/warn** on these scripts (C15/L2) and point at ICU, rather than grow a second-rate
dictionary segmenter. See §5 N9.

**L3 evidence.** For `lang='zh'`, `tokenize.py:333` adds `perluniprops.chars("Han")` to `IsAlnum`,
and the bundled `Han.txt` includes CJK *punctuation*:

```
'。' U+3002  is_cjk=True   in zh IsAlnum? True    (base IsAlnum? False)
'、' U+3001  is_cjk=True
'，' U+FF0C  is_cjk=False  -> gets split
'！' U+FF01  is_cjk=False  -> gets split
'？' U+FF1F  is_cjk=False  -> gets split
```

So `。` survives only because a *data* file classes it as a Han character. Nobody in issue #120
diagnosed this; the thread closed with a "use a real Chinese segmenter" recommendation. The precise
statement for the issue is: **this is a defect in `data/perluniprops/Han.txt`, not in the
tokenizer logic.**

### API ergonomics and dead code

| ID | Title | Impact | Effort | Back-compat risk | Source(s) |
|---|---|---|---|---|---|
| **A1** | CLI boolean flags are *negative* flags with *positive* names and help text | **High** | S | Med (CLI only) | [#98](https://github.com/hplt-project/sacremoses/issues/98) "Tokenizer -x option is confusing" (open) · [#90](https://github.com/hplt-project/sacremoses/issues/90) "Escape option on command line" · own analysis of `cli.py:92-98,149-155,177-186` **[verified]** |
| **A2** | Delete or gate `sacremoses/sent_tokenize.py` — `import sacremoses.sent_tokenize` raises `NotImplementedError` | Med | S | None | own analysis of `sent_tokenize.py:19` **[verified]** · history: [#91](https://github.com/hplt-project/sacremoses/issues/91), [#86](https://github.com/hplt-project/sacremoses/issues/86) (this file shipped broken in 0.0.38 and broke `pip install` itself) |
| **A3** | Decide the fate of `subwords.py` (commented out of `__init__.py:6`) and `chinese.py` (unreferenced anywhere) | Low | S | None | own analysis of `__init__.py:6`, `chinese.py` **[verified: `chinese.py` has zero importers in the whole repo]** |
| **A4** | Delete dead regex constants `REPLACE_DOT_WITH_LITERALSTRING_1/2/3`, `MID_STRIP`, `LEFT_STRIP`, `RIGHT_STRIP` | Low | S | None | own analysis of `tokenize.py:39-41,54-56` **[verified: `REPLACE_DOT_WITH_LITERALSTRING_1`'s replacement is `' DOTMULTI\x01'` — a non-raw-string bug — but the constant is unused since the audit's `replace_multidots` rewrite]** |
| **A5** | Add type hints + `py.typed` to the four public classes | Med | M | None | no external source — own analysis; the public surface is only 4 classes (see §1) |
| **A6** | Rename `MosesDetokenizer.tokenize()` → `detokenize()` as primary, keep `tokenize` as the alias | Low | S | Low | own analysis of `tokenize.py:750,903-905` |
| **A7** | Add an incremental-training path to `MosesTruecaser` | Low | M | None | [#102](https://github.com/hplt-project/sacremoses/issues/102) "Possible to retrain/keep training an existing model?" — alvations: *"I'm thinking about how to put this feature in… please keep this issue open"* (open since 2020-06-22) |

**A1 evidence.** `click`'s `is_flag=True, default=True` sets `flag_value = not default`, so passing
the flag *disables* the behaviour its name and help text advertise:

```
$ sacremoses -l en tokenize   < in     ->  It &apos;s a test .
$ sacremoses -l en tokenize -x < in     ->  It 's a test .
    help text says:  "-x, --xml-escape   Escape special characters for XML."
```

The same inversion applies to `detokenize -x/--xml-unescape`, `normalize -q/--normalize-quote-commas`
and `normalize -d/--normalize-numbers` (`cli.py:149-155, 177-186`). Perl calls this `-no-escape`,
which is what ZJaume and bricksdont both asked for in #98. Note also that `normalize -q` collides
visually with the group-level `-q/--quiet`.

### Packaging, release and CI

| ID | Title | Impact | Effort | Back-compat risk | Source(s) |
|---|---|---|---|---|---|
| **K1** | **Cut a release.** 0.1.1 is 2.8 years old and predates every fix in the current audit | **High** | S | n/a | [#147](https://github.com/hplt-project/sacremoses/issues/147), [#129](https://github.com/hplt-project/sacremoses/issues/129), [#115](https://github.com/hplt-project/sacremoses/issues/115), [#75](https://github.com/hplt-project/sacremoses/issues/75) · [PyPI history](https://pypi.org/project/sacremoses/#history) **[verified: latest = 0.1.1, 2023-10-30]** |
| **K2** | Tag GitHub releases; only one exists (`0.1.0`) for 52 PyPI versions | Med | S | None | [#115](https://github.com/hplt-project/sacremoses/issues/115) (open since 2021-04-07), [#75](https://github.com/hplt-project/sacremoses/issues/75) **[verified via `gh api …/releases`]** |
| **K3** | Fix PyPI metadata: `author`, `summary`, `license` are empty strings | Med | S | None | own analysis of `setup.py:22,25,28` + live PyPI JSON **[verified]** · downstream harm: [conda-forge/sacremoses-feedstock#7](https://github.com/conda-forge/sacremoses-feedstock/issues/7) still declares LGPL-2.1 · [#104](https://github.com/hplt-project/sacremoses/issues/104) "License Clarification" |
| **K4** | Migrate to `pyproject.toml` (PEP 621) | Med | S | None | no `pyproject.toml` exists **[verified]** · [#126](https://github.com/hplt-project/sacremoses/issues/126) "distutils is deprecated in Python 3.10" |
| **K5** | Extend CI to Python 3.12/3.13; refresh `actions/checkout@v3`→v4+, `setup-python@v4`→v5+, `cache@v3`→v4 | **High** | S | None | own analysis of `.github/workflows/test.yaml` **[verified: matrix stops at 3.11]** · [PR #141](https://github.com/hplt-project/sacremoses/pull/141) (Travis→Actions, merged) |
| **K6** | Raise `python_requires` off EOL 3.8 | Low | S | Low | `setup.py:45` **[verified]** |
| **K7** | Stop shipping `sacremoses/data/` (1.6 MB) now that the audit turned it into `_data_*.py` (1.5 MB); also drop the `README.txt` key the generator sweeps in | Med | S | Low | own analysis of `setup.py:26` vs `corpus.py` (no file I/O remains) + `tools/generate_data_modules.py` **[verified: wheel would carry ~3.1 MB of which half is unread; `NONBREAKING_PREFIXES` has 39 keys for 38 languages because the generator globs `README.txt` in as a language]** |
| **K8** | Make `joblib`/`tqdm`/`click` CLI-only extras | Med | M | Med (import surface) | [tlkh/serverless-transformers#48](https://github.com/tlkh/serverless-transformers/issues/48) — Whitesource flagged CVE-2022-21797 with the chain `sacremoses-0.0.43 → joblib-0.14.1` ([GHSA-6hrg-qmvc-2xh8](https://github.com/advisories/GHSA-6hrg-qmvc-2xh8)) · [#130](https://github.com/hplt-project/sacremoses/issues/130) click 8.1 breakage · alvations in #130: *"In the long run, seems like getting rid of click is the way to go"* |
| **K9** | Publish a conda-forge bump and correct the licence there | Med | S | None | [feedstock#15](https://github.com/conda-forge/sacremoses-feedstock/pull/15) (open since 2023-10-30), [feedstock#7](https://github.com/conda-forge/sacremoses-feedstock/issues/7) (open, zero replies, since 2020-10-13) |

### Documentation and testing

| ID | Title | Impact | Effort | Back-compat risk | Source(s) |
|---|---|---|---|---|---|
| **D1** | Publish a supported-language table + an explicit "what this tool is not" section | **High** | S | None | [#12](https://github.com/hplt-project/sacremoses/issues/12) "Documentation" (open since 2018-12-18), [#125](https://github.com/hplt-project/sacremoses/issues/125), [#120](https://github.com/hplt-project/sacremoses/issues/120), [#42](https://github.com/hplt-project/sacremoses/issues/42) **[verified: README contains no language list]** |
| **D2** | Publish the Moses↔sacremoses cheatsheet from issue #44 in the README | Med | S | None | [#44](https://github.com/hplt-project/sacremoses/issues/44) — alvations wrote the table in-thread; loretoparisi: *"It would be worth to add it to the docs!"* |
| **D3** | Document that `MosesDetruecaser()` takes no model | Low | S | None | [#107](https://github.com/hplt-project/sacremoses/issues/107) "error: loading model for detruecaser" (open, unanswered) **[verified: `TypeError: __init__() takes 1 positional argument but 2 were given`]** |
| **T1** | Add a Perl-parity regression suite (golden files generated from `tokenizer.perl`) | **High** | L | None | [#37](https://github.com/hplt-project/sacremoses/issues/37) "Testing CLI commands" (open since 2019-03-18; alvations: *"Add tests to test sacremoses output against Moses' perl scripts"*) · [#150](https://github.com/hplt-project/sacremoses/issues/150) · jelmervdl on [PR #144](https://github.com/hplt-project/sacremoses/pull/144): *"I'd like to add tests for all supported languages so we can make sure we don't break/change anything by accident"* |
| **T2** | Add CLI chaining tests | Med | M | None | [#150](https://github.com/hplt-project/sacremoses/issues/150), [#37](https://github.com/hplt-project/sacremoses/issues/37), [#24](https://github.com/hplt-project/sacremoses/issues/24) · [WMT20 paper §](https://aclanthology.org/2020.wmt-1.88.pdf) cited by alvations in #150: *"we noticed that some sacremoses command line flags were broken"* |
| **T3** | Add a golden-output test pinned to the four `transformers` call signatures | **High** | S | None | no external source — own analysis; see R1 table. This is the guard that makes every other change in this document safe |
| **T4** | Cover the untested modules: `penn_tokenize`, `cs`/`fi`/`ga` detokenizer paths, `chinese.py`, `indic.py`, `util.py` doctests | Med | M | None | own analysis of `sacremoses/test/` **[verified: 106 tests, none touching `penn_tokenize`, `cs` detok, or `chinese.py`]** |

### Issues that can be closed now

Verified against this checkout — these no longer reproduce, or are already resolved by the
in-progress audit:

| Issue | Finding |
|---|---|
| [#153](https://github.com/hplt-project/sacremoses/issues/153) "1 test fails" | **Root cause identified and already fixed by the current audit.** Not a normalizer bug at all — a *test-pollution* bug. See below. |
| [#106](https://github.com/hplt-project/sacremoses/issues/106) "detrucase Error: unsupported operand type(s)" | No longer reproduces; `truecase(..., use_known=True)` returns `'the ADVENTURES OF SHERLOCK HOLMES'`. Fixed by [PR #121](https://github.com/hplt-project/sacremoses/pull/121) / commit `2d5bb85` **[verified]** |
| [#113](https://github.com/hplt-project/sacremoses/issues/113) "Truecaser seems do not process sentences beginning with quotation marks" | No longer reproduces; output now matches the `truecase.perl` reference quoted in the issue **[verified]** |
| [#112](https://github.com/hplt-project/sacremoses/issues/112) "bug report" (`int`+`str` in truecaser) | Superseded — see [PR #111](https://github.com/hplt-project/sacremoses/pull/111), jelmervdl: *"Fixed in 2d5bb85…"* |
| [#117](https://github.com/hplt-project/sacremoses/issues/117) "can't tokenise the period properly" | **Working as designed and Perl-faithful.** `tokenizer.perl:366` has the same `$words[$i+1] =~ /^[\p{IsLower}]/` guard. Close with an explanation, don't "fix" **[verified]** |

**#153 root cause (proven).** The reported failure string is
`assert 'yesterday "s reception' == "yesterday 's reception"`. On upstream 0.1.1, merely
*constructing* a `perl_parity=True` normalizer permanently rewrites the shared class attribute
`NORMALIZE_UNICODE[11]` for every other normalizer in the process:

```
# upstream 0.1.1 (pre-audit)
before parity : "yesterday 's reception"
MosesPunctNormalizer('en', perl_parity=True)      # <-- just constructing it
after  parity : 'yesterday "s reception'          # <-- verbatim the string in issue #153

# current worktree (audit applied)
before parity : "yesterday 's reception"
after  parity : "yesterday 's reception"
```

So `test_moses_noralize_single_apostrophe` fails **only when `test_moses_normalize_with_perl_parity`
happens to run first** — which is why it fails on a FreeBSD port build and passes in the project's
own `unittest discover` CI. Nobody in the thread diagnosed this. The report is genuine, the
diagnosis in the title ("1 test fails") is misleading, and the fix already landed in this audit.

---

## 3. How sacremoses compares

All versions and dates below were pulled from the PyPI JSON API during this research pass and are
**verified**. sacremoses sits at **0.1.1, 2023-10-30**.

### The Perl original (`moses-smt/mosesdecoder`)

`tokenizer.perl` is 596 lines. Its complete set of language branches is small and entirely about
apostrophes, clitics and one punctuation character — **there is no segmentation logic for any
language**:

| Perl branch | Lines | In sacremoses? |
|---|---|---|
| `fi`/`sv` colon-inside-word | 261-268 | **No** — commented out, constants undefined (C9) |
| `tdt` apostrophe-inside-word | 269-274 | **No** (L4) |
| `ca` middle dot | 275-281 | **No** (C10) |
| `en` apostrophe (split right) | 324-331 | Yes |
| `fr`/`it`/`ga`/`ca` apostrophe (split left) | 336-339 | **Partial** — `fr`/`it` only (C11) |
| `so`/`tdt` glottal | 344-346 | **No** — commented out (C12) |
| generic apostrophe | 350 | Yes |
| last-word split | 362-365 | **No** (C8) |
| unknown-language warning | 570-571 | **No** (C15) |

CLI flags: `-b` (unbuffer), `-l`, `-q`, `-h`, `-x` (skip XML), `-a`, `-time`, `-protected`,
`-threads`, `-lines`, `-penn`, `-no-escape`. sacremoses' CLI implements `-l`, `-q`, `-a`,
`-protected`, and `-j` (≈`-threads`); it **lacks** `-b`, `-time`, `-lines`, `-penn`, and its `-x`
is the inverted flag of A1. `penn_tokenize()` exists in the library but always raises (C1). The
detokenizer's `-penn` mode has no port at all — that is
[#116](https://github.com/hplt-project/sacremoses/issues/116) "No detokenize_penn?", open since
2021.

### Direct replacements for the Moses scripts

| Tool | Latest | Released | Status | vs sacremoses |
|---|---|---|---|---|
| **sacremoses** | 0.1.1 | 2023-10-30 | stalled | — |
| `mosestokenizer` (luismsgomes) | 1.2.1 | 2021-10-22 | dormant | Wraps the **actual Perl scripts** over subprocess pipes, so parity is exact by construction; costs a process per call and a Perl runtime. Includes `MosesSentenceSplitter` (`split-sentences.perl`), which sacremoses lacks entirely (N6) |
| `fast-mosestokenizer` (mingruimingrui) | 0.0.8.2 | 2021-10-29 | dormant | C++ reimplementation; source of the R4 benchmark |
| **`opus-fast-mosestokenizer`** (Helsinki-NLP) | **0.0.8.11** | **2025-12-03** | **active** | The maintained fork of the above. **This is the live competitor**: same job, ~15x faster, and released 2 years more recently than sacremoses. It is what the OPUS/Marian ecosystem actually uses |

The honest read: **`opus-fast-mosestokenizer` beats sacremoses on both speed and release cadence.**
sacremoses' remaining advantages are that it is pure Python (no compiler, no wheels-per-platform
problem, works on PyPy — which the CI matrix tests), that it is what tens of thousands of pinned
requirements files already name, and that it is the exact code four `transformers` tokenizers were
calibrated against.

### Adjacent tools that are often confused with it

| Tool | Latest | Released | Relationship |
|---|---|---|---|
| `sacrebleu` | 2.6.0 | 2026-01-12 | **Different job, similar name.** Its `13a` is a port of `mteval-v13a.pl`'s scoring tokenizer, **not** `tokenizer.perl` — using it as a preprocessing tokenizer is a category error. Also ships `intl` (Unicode-property based), `zh`, `ja-mecab`, `ko-mecab`, `flores101`/`flores200` (SPM). Actively maintained |
| HF `tokenizers` (Rust) | 0.23.1 | 2026-04-27 | **No Moses anything.** A code search across the repo for `moses`/`Moses`/`MosesTokenizer` returns **zero** results (control search for `metaspace` returns 10+, so the search works). Its pre-tokenizers are `Whitespace`, `BertPreTokenizer`, `Metaspace`, `ByteLevel` etc. It solves subword vocabulary, not Moses-style pre-tokenization — the two compose, they do not compete |
| `sentencepiece` | 0.2.2 | 2026-07-12 | What actually replaced Moses tokenization in modern NMT: language-agnostic, no pre-tokenization required, trains directly on raw text. This is why sacremoses' addressable market has been shrinking since ~2019 |
| `subword-nmt` | 0.3.8 | 2021-12-08 | No PyPI release in 4.7 years, but the repo is **not archived** and has commits to 2024-08-07. Best described as quiet maintenance for reproducibility, not abandonment |
| `nltk` | 3.10.3 | 2026-08-12 | `nltk.tokenize.moses` was **removed** over the LGPL/Apache conflict in [nltk/nltk#2000](https://github.com/nltk/nltk/issues/2000) (2018-04-10). sacremoses exists because of this. Very much alive |
| `spacy` | 3.8.15 | 2026-08-07 | **Delegates** for no-delimiter languages rather than implementing: `th`→pythainlp, `zh`→jieba/spacy-pkuseg, `ja`→SudachiPy, `ko`→mecab-ko, `vi`→pyvi. 79 language dirs |
| `stanza` | 1.14.0 | 2026-07-15 | 80 languages on UD v2.12; one of the few general toolkits with **native Thai and Burmese**. Neural, so far heavier than sacremoses |

### Languages without word delimiters — the honest scorecard

This is where sacremoses is weakest and where the owner's concern is best directed.

| Language | sacremoses today | Best-in-class alternative |
|---|---|---|
| Chinese | pass-through (whole sentence = 1 token); `。`/`、` glued, `，`/`！`/`？` split (L3) | jieba 0.42.1 (2020-01-20, dormant), spacy-pkuseg, **ICU** |
| Japanese | pass-through | fugashi 1.5.2 (2025-10-24), SudachiPy, **ICU** |
| Korean | splits on spaces only (Hangul added to `IsAlnum`) | mecab-ko |
| **Thai** | **corrupts** — tears `Mn` marks off bases (L1) | pythainlp 5.3.5 (2026-07-29), stanza, **ICU** |
| **Khmer** | **corrupts** — splits at every coeng (L1) | khmer-nltk 1.6 (2023-09-19), **ICU** |
| **Lao** | **corrupts** (L1) | **ICU** |
| **Burmese** | **corrupts** (L1) | stanza, **ICU** |
| Tibetan | not supported | botok 1.1.6 (2026-07-22) |

**ICU is the single answer to that whole column.** The ICU user guide states it ships dictionary
word boundaries for *"Chinese, Japanese, Thai, Lao, Khmer and Burmese… Use of the dictionaries is
automatic… There is no separate API, and no extra programming steps required."* Reachable from
Python via **PyICU 2.16.2 (2026-03-20)**. This is the basis of N9: sacremoses should point at ICU,
not grow a segmenter.

### What sacremoses is uniquely good at

1. **It is the reference implementation for four shipped `transformers` tokenizers.** XLM, FSMT,
   FlauBERT and BioGPT vocabularies were built against *its* output, quirks included. No other
   tool can substitute without changing those models' behaviour. This is not a technical advantage
   — it is a compatibility monopoly, and it is the entire reason the roadmap prioritises stability.
2. **Pure Python, four small dependencies, no compiler, no model files.** `opus-fast-mosestokenizer`
   needs wheels per platform; `stanza` needs torch; ICU needs a C++ library. sacremoses runs on
   PyPy and in a locked-down build container.
3. **It ports the *detokenizer* and *truecaser*, not just the tokenizer.** `MosesDetokenizer` has
   57 downstream importers. sacrebleu, HF `tokenizers` and sentencepiece have no equivalent —
   detokenization is a genuinely under-served niche.
4. **38 bundled non-breaking-prefix lists**, kept in sync with mosesdecoder.

### Where it is clearly behind

1. **Speed** — slowest Moses tokenizer in common use (R4), and over half of that is one unmerged
   patch (P1).
2. **Release cadence** — 2.8 years since 0.1.1; its live competitor shipped 2025-12-03.
3. **Delimiter-less languages** — actively corrupting for four of them (L1).
4. **Perl parity** — nine documented divergences (C8–C15), two of which break round-tripping.
5. **Sentence splitting** — no `split-sentences.perl` port; `mosestokenizer` has one (N6).
6. **Packaging hygiene** — empty metadata, wrong licence downstream, absent from Debian (R5).

## 4. Tiers

### Tier 1 — Quick wins (ship these first; all are S effort, none changes tokenizer output)

Ranked by impact ÷ effort.

1. **P1** — merge [PR #149](https://github.com/hplt-project/sacremoses/pull/149). 3.29x measured,
   output-identical, written 2.2 years ago, sitting unreviewed. Nothing else here comes close.
2. **T3** — pin golden outputs for the four `transformers` call signatures. Do this *before*
   anything in Tier 2, because it is what converts R1 from a hope into a test.
3. **C1 + C2** — fix `penn_tokenize`. It cannot regress anyone: it currently raises for 100% of
   inputs.
4. **K1** — release. Every fix in the current security audit is invisible until this happens, and
   four separate issues ask for it.
5. **P3, P2, P4** — the remaining free speed.
6. **C15** — emit the unknown-language warning Perl emits. One `warnings.warn`, and it defuses a
   whole class of confusion (#125, #120, #42) plus the silent Marian `source_lang` fallback.
7. **D1 + D2** — the language table and the Moses↔sacremoses cheatsheet. Issue #12 has been open
   for 7.6 years and the cheatsheet is already written in #44.
8. **C3, C4, C5** — three genuine crashes/misbehaviours with narrow blast radius.
9. **K5** — Python 3.12/3.13 in CI. Cheap, and 3.13 is current.
10. **K3, K7, A4, A2** — metadata, the 1.6 MB of unread data files, dead constants, the module that
    raises on import.
11. **Close #153, #106, #113, #112, #117** with the evidence in §2.

### Tier 2 — Substantial (worth doing, needs care and a design decision)

* **T1** — the Perl-parity golden suite. This is the load-bearing investment. Every item in Tier 3
  is gated on it, and jelmervdl already asked for it on PR #144. Estimate L.
* **`perl_parity=True` on `MosesTokenizer`** — the vehicle for **C8–C14**. Mirrors the flag
  `MosesPunctNormalizer` already has from merged PR #146. Lets the project be honest about the
  divergences without breaking a single existing user. **Within that, do C10+C11 (`ga`/`ca`) first**
  — they are the only divergences where the current default actively corrupts text on a
  tokenize→detokenize round-trip, and neither language appears in any `transformers` tokenizer's
  hard-coded call, so the R1 blast radius is small.
* **A1** — rename the inverted CLI flags, keeping the old spellings as hidden aliases. Fixes #98
  and #90.
* **A5** — type hints + `py.typed`. Cheap for a 4-class API, and it is the kind of thing that makes
  a dependency feel maintained.
* **K8** — move `click`/`joblib`/`tqdm` to a `[cli]` extra. Library users currently inherit
  joblib's CVE surface for a feature they never call, and `click` has already broken the package
  once (#130).
* **P5** — stream truecaser training. #105 has been an unanswered OOM report for 6 years.
* **L4** — finish Tetun Dili. Two PRs and the upstream Moses PR already exist; only the tokenizer
  branch is missing.

### Tier 3 — Speculative (needs a decision from the owner before any code)

* **L1** — Thai/Khmer/Lao/Burmese combining marks. Highest linguistic value in the document, but it
  changes output *and* deviates from Perl. Only behind an opt-in flag + a major version, and only
  with T1 in place. The repo owner has explicitly warned against breaking delimiter-less
  languages — this item is how you *un*-break them, but it must be opt-in. **Scope it narrowly**:
  the goal is "stop emitting invalid grapheme clusters", not "segment words" (see §5 N9). The cheap
  90% of the value is L2+C15 — *documenting* and *warning* — which are Tier 1 and carry no
  back-compat risk at all.
* **L3** — Chinese punctuation consistency. The honest fix is to correct
  `data/perluniprops/Han.txt`, which is a data change with wide blast radius.
* **A7** — incremental truecaser training (#102). Low demand: `MosesTruecaser` has only 7 importers
  in the 100-file downstream sample.
* **A3** — decide `subwords.py`/`chinese.py`. `sentencepiece` and `subword-nmt` own this space now;
  the honest answer is probably deletion.
* **A6** — `MosesDetokenizer.detokenize` as primary.
* **A JOSS submission or Zenodo DOI** — [#145](https://github.com/hplt-project/sacremoses/issues/145).
  jelmervdl declined (*"It really is just a tool, not a research project"*); faroit pushed back
  (*"This project is extremely valuable to carry out research… This is what JOSS is for"*).
  Genuinely the owner's call, not a technical question.

---

## 5. NOT recommended

Things users have asked for that I think should be **declined**, with reasoning.

**N1 — Do not enable protected patterns by default.**
Source: [PR #138](https://github.com/hplt-project/sacremoses/pull/138) "Web and basic protected
patterns by default" (samirsalman, open since 2023-03-20).
The request is reasonable in isolation — `tokenize("http://www.someurl.com")` shredding a URL is
surprising. But this changes default tokenizer output for every user, including the four
`transformers` tokenizers in R1, for text that merely *contains* a URL or an `@handle`. It also
puts the protected-token path — the one the current audit had to harden twice (placeholder
overflow, forgeable sentinel) — on every call. `tokenizer.perl` has no default `-protected` either.
**Decline; document `protected_patterns=` prominently instead** (it is already reachable, and via
`:web:`/`:basic:` on the CLI).

**N2 — Do not "fix" #117.**
Source: [#117](https://github.com/hplt-project/sacremoses/issues/117) "can't tokenise the period
properly".
`meaning.` staying glued when followed by a lowercase word is exactly `tokenizer.perl:366`'s
`$words[$i+1] =~ /^[\p{IsLower}]/` guard. Changing it would break parity *and* every downstream
vocabulary. Close with the Perl citation.

**N3 — Do not add a `lowercase` command.**
Source: [#82](https://github.com/hplt-project/sacremoses/issues/82) "Add lowercase script?" — the
**most-discussed open issue** (20 comments).
The strongest argument in-thread is noe's: GNU `tr [:upper:] [:lower:]` is not Unicode-aware
(verified in-thread on Cyrillic and Greek), so people do want *something*. But that something is
`str.lower()`, which is already correct, already Unicode-aware, and already one line. Moses shipped
`lowercase.perl` because Perl needed it, not because the operation is hard. Adding a command means
owning final-sigma/Turkish-dotless-i edge cases forever for zero algorithmic content.
**Decline the command; add a three-line README recipe** covering the `tr` trap that noe documented.
*(If mjpost's chaining request — `sacremoses normalize … lowercase …` — is the real motivation,
that is an argument about CLI composition, not about lowercasing.)*

**N4 — Do not switch from `re` to `regex`.**
Source: [#76](https://github.com/hplt-project/sacremoses/issues/76) "Difference between regex and
re library" (alvations' own question).
Both substantive replies in-thread argued against it. goodmami: *"I don't think there's a lot to
gain unless you really need greater PCRE compatibility."* BLKSerene noted `regex` should be
*slower*, having more features. And `regex`'s `V1` flag turns existing warnings into errors. The
measured hot spot is not the regex engine at all — it is `set()` reconstruction (P1). **Decline,
and close #76 with the P1 profile as the answer.** Note `regex` is still needed for
`remove_control_chars`'s `\p{C}` (`normalize.py:213`), so the dependency stays.

**N5 — Do not replace the space-dedup regex with `" ".join(text.split())`.**
Source: [#79](https://github.com/hplt-project/sacremoses/issues/79) "Space deduplication"
(alvations' own question).
goodmami did the Unicode analysis in-thread and alvations' own conclusion was the right one:
*"considering things like em-space and ideographic space might be problematic because it affects
other regexes in Moses."* `str.split()` and `\s` disagree on characters that other Moses rules
depend on. This is a micro-optimisation with a real correctness tail. **Decline; close #79.**

**N6 — Do not build a sentence tokenizer.**
Source: [#99](https://github.com/hplt-project/sacremoses/issues/99) "Is there a plan to have
sent_tokenize in this library?"; alvations: *"that `sent_tokenize` is a can of worms thus the
reluctance to complete the code."*
Six years later the can of worms has not shrunk, `split-sentences.perl` is a different script with
different data files, and the space is well served (`pysbd`, `blingfire`, NLTK punkt, `stanza`).
The half-written `sent_tokenize.py` should be **deleted** (A2), not finished — it has already
broken `pip install` once (#91, #86). If the demand is real, the honest answer is a README pointer.
*(Noted as a genuine gap:* [OpenNMT/CTranslate2#1140](https://github.com/OpenNMT/CTranslate2/issues/1140)
*has a user bridging to Perl over pipes because of exactly this. That user wants
`split-sentences.perl`, which is a separate project.)*

**N7 — Do not port sacremoses to Java or C++.**
Source: [#123](https://github.com/hplt-project/sacremoses/issues/123) "Is there a way to use
sacremoses in java?"
[Helsinki-NLP/opus-fast-mosestokenizer](https://github.com/Helsinki-NLP/opus-fast-mosestokenizer)
already exists, is maintained, and is 15x faster. Point people at it.

**N8 — Do not chase a "deep detokenizer".**
Source: [#118](https://github.com/hplt-project/sacremoses/issues/118) "deep detokenizer" — a
pointer to a fine-tuned DeBERTa model that reverses sacremoses.
Adding a neural dependency to a zero-model regex library would be a category error. A README
"see also" line is the right response, which is all the reporter actually asked for.

**N9 — Do not build a word segmenter for Chinese/Japanese/Thai/Khmer/Lao/Burmese.**
Sources: [#120](https://github.com/hplt-project/sacremoses/issues/120),
[#42](https://github.com/hplt-project/sacremoses/issues/42),
[#125](https://github.com/hplt-project/sacremoses/issues/125),
[#123](https://github.com/hplt-project/sacremoses/issues/123).
This is the natural over-reach from L1, and it should be resisted. `tokenizer.perl` has **zero**
language branches for any of these — its only branches are `fi/sv`, `tdt`, `ca`, `en`, `fr/it/ga/ca`
and `so/tdt`, all apostrophe/clitic rules — so segmentation was never in scope for the thing
sacremoses is a port of. XapaJIaMnu already gave the right answer in #120: *"tokenizing Chinese is
a more complicated problem than what sacremoses is designed to handle."* ICU covers all six
languages with one API (see L1), and jieba/spacy-pkuseg, SudachiPy/fugashi, pythainlp, botok and
stanza cover them individually. **L1 should be scoped strictly to "stop producing invalid grapheme
clusters", not to "segment words".** Those are different projects, and only the first one is
sacremoses'.

---

## 6. Where the evidence is thin

Stated plainly, so nobody over-trusts this document:

* **P5** (truecaser OOM) is a read of `readlines()` at `truecase.py:196-200`. It is a very likely
  cause of #105/#55 but I did not reproduce an OOM.
* **L1's fix shape** is inferred from `indic.py`'s precedent. I verified the *breakage* but did not
  build or test the Thai/Khmer/Lao/Myanmar mark tables; the exact character set needs a linguist's
  eye, not a regex author's.
* **All timings are from one machine** (macOS, CPython 3.9.12, `regex` 2.5.140). The 3.29x for P1 is
  robust because the mechanism (rebuilding a 2,253-element set per call) is machine-independent,
  but absolute lines/sec will vary. The fast-mosestokenizer numbers in R4 are *their* benchmark on
  *their* hardware, not mine.
* **`#153` is verified as a mechanism**, but I did not reproduce it on FreeBSD/Python 3.11 as the
  reporter's environment. The failure string matches exactly, which is strong but not conclusive.
* **No published benchmark of sacremoses vs `sacrebleu`/HF `tokenizers`/spaCy/stanza exists.** The
  only quantified rival benchmark is fast-mosestokenizer's. Any claim about relative speed against
  those tools would be invention.
* **Stack Overflow is a dead end for this project.** The Stack Exchange API returns **zero**
  questions with "sacremoses" in the title. 69 mention it in the body, almost all incidentally
  inside pasted `pip install transformers` logs. The "notorious sacremoses install problem" is a
  real PyPI packaging fact (see K1 and §7) but it is *not* a widely discussed topic, and I found no
  blog post or Reddit thread about it. Treat the absence of Reddit hits as inconclusive rather
  than as evidence of absence — the search backend may not index it.
* **conda-forge, nixpkgs and MacPorts licence metadata**: I verified conda-forge directly (still
  `LGPL-2.1-or-later`, still `0.0.53`). nixpkgs/MacPorts/AUR/FreeBSD were reported by a delegated
  web sweep and **I did not re-fetch them**; treat as unverified.
* Which exact `transformers` release contains
  [PR #18321](https://github.com/huggingface/transformers/pull/18321) is **unverified** (4.22.0 is
  plausible).

---

## 7. Appendix — sources consulted

### Repositories read

| Repo | What was read |
|---|---|
| `hplt-project/sacremoses` | **upstream.** All 83 issues + all 70 PRs (`--state all`), full bodies and comments for ~40 |
| `alvations/sacremoses` | verified via `gh repo view`: now a **fork** of hplt-project, issues **disabled**, 0 PRs, last push 2023-04-27, 126 stars. All tracker history lives upstream |
| `moses-smt/mosesdecoder` | `scripts/tokenizer/tokenizer.perl` (596 lines), `detokenizer.perl` (373), `normalize-punctuation.perl` (90), `scripts/recaser/train-truecaser.perl` (127) — fetched raw and diffed against the port |
| `huggingface/transformers` | `gh search code "sacremoses"`; read `models/{fsmt,xlm,marian,biogpt}/tokenization_*.py` via the contents API |
| `conda-forge/sacremoses-feedstock` | live `meta.yaml` (still LGPL-2.1, still 0.0.53) |
| local checkout | every file under `sacremoses/`, `setup.py`, `requirements.txt`, `.github/workflows/test.yaml`, `SECURITY.md`, `tools/generate_data_modules.py` |

### sacremoses issues cited

Open: [#3](https://github.com/hplt-project/sacremoses/issues/3) ·
[#12](https://github.com/hplt-project/sacremoses/issues/12) ·
[#21](https://github.com/hplt-project/sacremoses/issues/21) ·
[#24](https://github.com/hplt-project/sacremoses/issues/24) ·
[#37](https://github.com/hplt-project/sacremoses/issues/37) ·
[#42](https://github.com/hplt-project/sacremoses/issues/42) ·
[#43](https://github.com/hplt-project/sacremoses/issues/43) ·
[#74](https://github.com/hplt-project/sacremoses/issues/74) ·
[#76](https://github.com/hplt-project/sacremoses/issues/76) ·
[#79](https://github.com/hplt-project/sacremoses/issues/79) ·
[#82](https://github.com/hplt-project/sacremoses/issues/82) ·
[#98](https://github.com/hplt-project/sacremoses/issues/98) ·
[#99](https://github.com/hplt-project/sacremoses/issues/99) ·
[#102](https://github.com/hplt-project/sacremoses/issues/102) ·
[#105](https://github.com/hplt-project/sacremoses/issues/105) ·
[#106](https://github.com/hplt-project/sacremoses/issues/106) ·
[#107](https://github.com/hplt-project/sacremoses/issues/107) ·
[#112](https://github.com/hplt-project/sacremoses/issues/112) ·
[#113](https://github.com/hplt-project/sacremoses/issues/113) ·
[#115](https://github.com/hplt-project/sacremoses/issues/115) ·
[#116](https://github.com/hplt-project/sacremoses/issues/116) ·
[#117](https://github.com/hplt-project/sacremoses/issues/117) ·
[#118](https://github.com/hplt-project/sacremoses/issues/118) ·
[#120](https://github.com/hplt-project/sacremoses/issues/120) ·
[#150](https://github.com/hplt-project/sacremoses/issues/150) ·
[#151](https://github.com/hplt-project/sacremoses/issues/151) ·
[#152](https://github.com/hplt-project/sacremoses/issues/152) ·
[#153](https://github.com/hplt-project/sacremoses/issues/153)

Closed: [#7](https://github.com/hplt-project/sacremoses/issues/7) ·
[#13](https://github.com/hplt-project/sacremoses/issues/13) ·
[#35](https://github.com/hplt-project/sacremoses/issues/35) ·
[#44](https://github.com/hplt-project/sacremoses/issues/44) ·
[#51](https://github.com/hplt-project/sacremoses/issues/51) ·
[#52](https://github.com/hplt-project/sacremoses/issues/52) ·
[#53](https://github.com/hplt-project/sacremoses/issues/53) ·
[#55](https://github.com/hplt-project/sacremoses/issues/55) ·
[#61](https://github.com/hplt-project/sacremoses/issues/61) ·
[#68](https://github.com/hplt-project/sacremoses/issues/68) ·
[#71](https://github.com/hplt-project/sacremoses/issues/71) ·
[#73](https://github.com/hplt-project/sacremoses/issues/73) ·
[#83](https://github.com/hplt-project/sacremoses/issues/83) ·
[#86](https://github.com/hplt-project/sacremoses/issues/86) ·
[#91](https://github.com/hplt-project/sacremoses/issues/91) ·
[#97](https://github.com/hplt-project/sacremoses/issues/97) ·
[#104](https://github.com/hplt-project/sacremoses/issues/104) ·
[#110](https://github.com/hplt-project/sacremoses/issues/110) ·
[#122](https://github.com/hplt-project/sacremoses/issues/122) ·
[#123](https://github.com/hplt-project/sacremoses/issues/123) ·
[#125](https://github.com/hplt-project/sacremoses/issues/125) ·
[#126](https://github.com/hplt-project/sacremoses/issues/126) ·
[#129](https://github.com/hplt-project/sacremoses/issues/129) ·
[#130](https://github.com/hplt-project/sacremoses/issues/130) ·
[#135](https://github.com/hplt-project/sacremoses/issues/135) ·
[#136](https://github.com/hplt-project/sacremoses/issues/136) ·
[#137](https://github.com/hplt-project/sacremoses/issues/137) ·
[#145](https://github.com/hplt-project/sacremoses/issues/145) ·
[#147](https://github.com/hplt-project/sacremoses/issues/147)

### sacremoses PRs cited

Open: [#114](https://github.com/hplt-project/sacremoses/pull/114) ·
[#138](https://github.com/hplt-project/sacremoses/pull/138) ·
[#148](https://github.com/hplt-project/sacremoses/pull/148) ·
[#149](https://github.com/hplt-project/sacremoses/pull/149)

Merged/closed: [#103](https://github.com/hplt-project/sacremoses/pull/103) ·
[#111](https://github.com/hplt-project/sacremoses/pull/111) ·
[#121](https://github.com/hplt-project/sacremoses/pull/121) ·
[#133](https://github.com/hplt-project/sacremoses/pull/133) ·
[#140](https://github.com/hplt-project/sacremoses/pull/140) ·
[#141](https://github.com/hplt-project/sacremoses/pull/141) ·
[#143](https://github.com/hplt-project/sacremoses/pull/143) ·
[#144](https://github.com/hplt-project/sacremoses/pull/144) ·
[#146](https://github.com/hplt-project/sacremoses/pull/146)

### External issues and PRs cited

* [nltk/nltk#2000](https://github.com/nltk/nltk/issues/2000) — "Is the Moses Tokenizer in violation of it's license?" (2018-04-10). The origin of sacremoses; `nltk.tokenize.moses` removed for LGPL/Apache incompatibility.
* [pytorch/text#306](https://github.com/pytorch/text/issues/306) — "MosesTokenizer has been moved out of NLTK due to licensing issues".
* [pytorch/text#2276](https://github.com/pytorch/text/issues/2276) + [PR #2281](https://github.com/pytorch/text/pull/2281) — torchtext points at the `alvations` fork; alvations in-thread: *"Pointing to HPLT is correct."* Both still open; torchtext archived.
* [huggingface/transformers#2453](https://github.com/huggingface/transformers/issues/2453) — "Installation of Transformers without Sacremoses" (2020-01-08). The licence complaint that started it.
* [huggingface/transformers#3307](https://github.com/huggingface/transformers/pull/3307) — "Make sacremoses dependency optional due to GPL license." **Closed, never merged.**
* [huggingface/transformers#17049](https://github.com/huggingface/transformers/pull/17049) — **"Make the sacremoses dependency optional"**, merged 2022-05-02, shipped in v4.19.0. *"This is an artifact of the past."*
* [huggingface/transformers#17096](https://github.com/huggingface/transformers/issues/17096) — `pip install "sacremoses>=0.0.50"` breaks on SageMaker Studio; workaround was pinning `==0.0.49`, the last version with a wheel.
* [huggingface/transformers#18321](https://github.com/huggingface/transformers/pull/18321) — follow-up unguarded-import fix.
* [huggingface/transformers#25380](https://github.com/huggingface/transformers/issues/25380) — "sacremoses library missing" on AzureML; sgugger declines to re-add.
* [huggingface/transformers#29504](https://github.com/huggingface/transformers/pull/29504) — 49 BioGPT tokenizer tests failing from the missing optional dep.
* [facebookresearch/fairseq#4882](https://github.com/facebookresearch/fairseq/issues/4882) — "Sacremoses error on recent versions"; `sacremoses tokenize -l en` → `Error: no such option: -l`. **Open since 2022-11-30**; fairseq's own documented example is broken.
* [hplt-project/OpusCleaner#143](https://github.com/hplt-project/OpusCleaner/issues/143) — the same `-l` breakage in HPLT's own tool.
* [flairNLP/flair#2763](https://github.com/flairNLP/flair/issues/2763), [#2765](https://github.com/flairNLP/flair/issues/2765) — flair broke the day transformers 4.19.0 shipped.
* [dmlc/gluon-nlp#1548](https://github.com/dmlc/gluon-nlp/issues/1548) — `MosesPunctNormalizer` output changed silently between 0.0.43 and 0.0.44.
* [bitextor/bicleaner#35](https://github.com/bitextor/bicleaner/issues/35) — adoption thread; contains the counter-benchmark (sacremoses faster than Perl-over-pipes) *and* ZJaume's initial refusal citing #43.
* [mozilla/translations#322](https://github.com/mozilla/translations/issues/322) — unbuildable training env from a sacremoses version conflict.
* [argosopentech/argos-translate#454](https://github.com/argosopentech/argos-translate/issues/454) — pinned at 0.0.53; upgrading treated as a risk.
* [OpenNMT/CTranslate2#1140](https://github.com/OpenNMT/CTranslate2/issues/1140) — user bridges to Perl because `split-sentences.perl` has no port.
* [tlkh/serverless-transformers#48](https://github.com/tlkh/serverless-transformers/issues/48) — CVE-2022-21797 reported through the `sacremoses → joblib` chain.
* [TigreGotico/linguonnx#33](https://github.com/TigreGotico/linguonnx/issues/33) — production 500s from a missing sacremoses, **opened 2026-08-03** (the most recent instance found).
* [conda-forge/sacremoses-feedstock#7](https://github.com/conda-forge/sacremoses-feedstock/issues/7) — wrong licence, open with zero replies since 2020-10-13.
* [conda-forge/sacremoses-feedstock#15](https://github.com/conda-forge/sacremoses-feedstock/pull/15) — the 0.1.1 bump, open since 2023-10-30.
* [moses-smt/mosesdecoder#224](https://github.com/moses-smt/mosesdecoder/pull/224) — the upstream Tetun Dili PR that #114/#144 mirror.
* [python/cpython#15030](https://github.com/python/cpython/pull/15030) — yannvgn's CPython `sre_parse._uniq` fix, filed from sacremoses issue #61.
* [GHSA-6hrg-qmvc-2xh8](https://github.com/advisories/GHSA-6hrg-qmvc-2xh8) — CVE-2022-21797 (joblib).

### Other URLs read

* [pypi.org/project/sacremoses](https://pypi.org/project/sacremoses/) — full release JSON. Latest **0.1.1, 2023-10-30**; 52 releases; `author`/`summary`/`license` empty; unpinned `requires_dist`. Wheel history is erratic: no wheel at all for the first 41 releases (0.0.0 → 0.0.44); **0.0.50–0.0.53 are sdist-only, and 0.0.50/0.0.51/0.0.52 are yanked** ("over-thinking", "Broken.").
* [pepy.tech/projects/sacremoses](https://pepy.tech/projects/sacremoses) — 195,922,480 total downloads, ~3.1M/month.
* [github.com/hplt-project/sacremoses/releases](https://github.com/hplt-project/sacremoses/releases) — exactly one release, `0.1.0`.
* [Helsinki-NLP/opus-fast-mosestokenizer](https://github.com/Helsinki-NLP/opus-fast-mosestokenizer) and [mingruimingrui/fast-mosestokenizer bench README](https://github.com/mingruimingrui/fast-mosestokenizer/blob/master/bench/README.md) — the C++ replacement and the only published benchmark including sacremoses.
* [aclanthology.org/2020.wmt-1.88.pdf](https://aclanthology.org/2020.wmt-1.88.pdf) — WMT20 system paper cited by alvations in #150: *"we noticed that some sacremoses command line flags were broken."*
* [tracker.debian.org/pkg/sacremoses](https://tracker.debian.org/pkg/sacremoses) — **404**; sacremoses is not in Debian.
* [huggingface.co/dsilin/detok-deberta-xl](https://huggingface.co/dsilin/detok-deberta-xl) — the "deep detokenizer" from #118.

### Alternatives surveyed (§3)

All versions/dates below were read from `https://pypi.org/pypi/<name>/json` during this pass and
are **verified by me directly**, not delegated:

| Package | Version | Released |
|---|---|---|
| [sacremoses](https://pypi.org/project/sacremoses/) | 0.1.1 | 2023-10-30 |
| [sacrebleu](https://pypi.org/project/sacrebleu/) | 2.6.0 | 2026-01-12 |
| [mosestokenizer](https://pypi.org/project/mosestokenizer/) | 1.2.1 | 2021-10-22 |
| [fast-mosestokenizer](https://pypi.org/project/fast-mosestokenizer/) | 0.0.8.2 | 2021-10-29 |
| [opus-fast-mosestokenizer](https://pypi.org/project/opus-fast-mosestokenizer/) | 0.0.8.11 | 2025-12-03 |
| [tokenizers](https://pypi.org/project/tokenizers/) | 0.23.1 | 2026-04-27 |
| [sentencepiece](https://pypi.org/project/sentencepiece/) | 0.2.2 | 2026-07-12 |
| [subword-nmt](https://pypi.org/project/subword-nmt/) | 0.3.8 | 2021-12-08 |
| [nltk](https://pypi.org/project/nltk/) | 3.10.3 | 2026-08-12 |
| [spacy](https://pypi.org/project/spacy/) | 3.8.15 | 2026-08-07 |
| [stanza](https://pypi.org/project/stanza/) | 1.14.0 | 2026-07-15 |
| [pythainlp](https://pypi.org/project/pythainlp/) | 5.3.5 | 2026-07-29 |
| [khmer-nltk](https://pypi.org/project/khmer-nltk/) | 1.6 | 2023-09-19 |
| [jieba](https://pypi.org/project/jieba/) | 0.42.1 | 2020-01-20 |
| [fugashi](https://pypi.org/project/fugashi/) | 1.5.2 | 2025-10-24 |
| [botok](https://pypi.org/project/botok/) | 1.1.6 | 2026-07-22 |
| [PyICU](https://pypi.org/project/PyICU/) | 2.16.2 | 2026-03-20 |

Also consulted: the [ICU user guide on boundary analysis](https://unicode-org.github.io/icu/userguide/boundaryanalysis/)
(dictionary support for zh/ja/th/lo/km/my, *"no separate API… required"*), and the
`huggingface/tokenizers` repository (code search for `moses`/`Moses`/`MosesTokenizer` → **0 hits**;
control search for `metaspace` → 10+ hits, confirming the search was functional).

**Not independently re-verified** (delegated, treat with caution): nixpkgs / MacPorts / AUR /
FreeBSD licence metadata for sacremoses; the two HuggingFace forum threads
([XLMTokenizer in Hosted Inference API](https://discuss.huggingface.co/t/xlmtokenizer-in-hosted-inference-api/19292),
[Need to install sacremoses in HF hub](https://discuss.huggingface.co/t/need-to-install-sacremoses-in-hf-hub/37574));
`pythainlp`'s exact engine list; and the `spacy`/`stanza` per-language delegation details.

### Commands used (for retraceability)

```bash
gh repo view hplt-project/sacremoses --json ...
gh issue list --repo hplt-project/sacremoses --state all --limit 300 \
    --json number,title,state,createdAt,author,labels,comments,body
gh pr    list --repo hplt-project/sacremoses --state all --limit 300 --json ...
gh search issues "sacremoses" --limit 100
gh search code   "sacremoses" --repo huggingface/transformers
gh search code   "from sacremoses import"
gh api repos/hplt-project/sacremoses/releases
curl -s https://pypi.org/pypi/sacremoses/json
curl -sL https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/{tokenizer,detokenizer,normalize-punctuation}.perl
curl -sL https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/recaser/train-truecaser.perl
```

Note for retracing: **`gh issue list --repo alvations/sacremoses` fails** with *"the
'alvations/sacremoses' repository has disabled issues"*. That repo is now a fork; the entire
tracker lives at `hplt-project/sacremoses`.
