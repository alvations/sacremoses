# TODO — correctness bugs

Deferred from the 2026-08 security audit. None of these is a security issue;
the security work shipped in **0.2.0**. These are correctness defects found
while auditing, kept here so they are not lost.

Every item was reproduced by running it. The repro line is copy-pasteable.

Back-compat note that governs all of them: four `transformers` tokenizers
(FSMT, XLM, FlauBERT, BioGPT) are calibrated against sacremoses' exact output
bytes. Anything that changes tokenizer output should land behind an opt-in
flag, not as a new default. See `audit/ROADMAP.md` for the full 47-item
analysis with sources.

---

## 1. Combining marks are split off their base characters

**Effect:** words in Thai, Lao, Khmer and Burmese are torn into pieces.

```python
>>> MosesTokenizer(lang="th").tokenize("น้ำ")     # "water"
['น', '้', 'ำ']
>>> MosesTokenizer(lang="th").tokenize("ไม่")     # "not"
['ไม', '่']
>>> MosesTokenizer(lang="km").tokenize("ភាសាខ្មែរ")
['ភាសាខ', '្', 'មែរ']
```

**Cause:** marks of Unicode category `Mn` that are absent from `IsAlpha` /
`IsAlnum` are padded as punctuation by `PAD_NOT_ISALNUM` (`tokenize.py`).
Counted by enumerating every `Mn` codepoint per block, not by sampling words:

| Block | `Mn` marks | missing from `IsAlpha` | notable |
|-------|-----------|------------------------|---------|
| Thai | 16 | **7** | U+0E48–U+0E4B, all four tone marks |
| Lao | 17 | **6** | U+0EC8–U+0ECB, the tone marks |
| Khmer | 22 | **14** | U+17D2 coeng |
| Tibetan | 74 | 11 | mostly genuine delimiters — see below |
| Devanagari | 23 | 4 | Vedic accents only, rare |
| Myanmar | 28 | 3 | U+103A asat |

**Careful:** Tibetan's splits are mostly on U+0F0B tsek and U+0F0D shad, which
*are* delimiters. Do not "fix" those.

**Also careful:** an early spot-check used `สวัสดีครับ`, which happens to
contain only the covered marks, and tokenized cleanly — producing a wrong
"Thai is fine" conclusion. Sampling a word tests the marks that word uses;
only enumerating the block tests the class.

**Do not** build a word segmenter for these languages. ICU already covers
zh/ja/th/km/lo/my transparently — see the ROADMAP entry.

## 2. Irish and Catalan corrupt on round-trip

```python
>>> t, d = MosesTokenizer(lang="ga"), MosesDetokenizer(lang="ga")
>>> d.detokenize(t.tokenize("D'ith an cat", escape=False))
"D 'ith an cat"          # expected: "D'ith an cat"
>>> # Catalan: "L'home d'aquí" -> "L 'home d' aquí"
```

`tokenizer.perl` groups `fr|it|ga|ca`, but `detokenizer.perl` groups only
`fr|it`. sacremoses has the grouping **inverted on both sides**. French and
Italian round-trip correctly.

## 3. `penn_tokenize()` raises on every input

```python
>>> MosesTokenizer(lang="en").penn_tokenize("Hello world.")
AttributeError: 'str' object has no attribute 'sub'
```

`INTRATOKEN_SLASHES` (`tokenize.py:93-96`) is an uncompiled `(str, str)` tuple
whose replacement is still Perl syntax (`r"$1 \@\/\@ $2"` rather than
`\1/\2`). It sits in `MOSES_PENN_REGEXES_1`, which `penn_tokenize()` iterates
calling `regexp.sub()`. Introduced by the regex-hoisting commit `4de6194`;
it has **zero test coverage**, which is why it went unnoticed. Upstream issue
hplt-project/sacremoses#151, open since 2024-07-19.

## 4. `MosesSentTokenizer` cannot be imported

```python
>>> from sacremoses.sent_tokenize import MosesSentTokenizer
NotImplementedError
```

A bare `raise NotImplementedError` in the class body, so merely importing the
module raises. `split-sentences.perl` was never ported. Either finish it or
remove the module — but note upstream #99 has already broken `pip install`
once, so removal is the safer direction.

## 5. Finnish, Swedish, Catalan and Tetum branches are dead code

`tokenize.py` around line 534 holds the `fi`/`sv` colon-handling branch inside
a **triple-quoted string** headed `# FIXME!!!`, and `FI_SV_COLON_APOSTROPHE` /
`FI_SV_COLON_NO_LOWER_FOLLOW` are never defined anywhere:

```python
>>> hasattr(MosesTokenizer, "FI_SV_COLON_APOSTROPHE")
False
```

Because the `else:` is inside the string too, `PAD_NOT_ISALNUM` runs
unconditionally for every language. A second `# FIXME!!!` near line 575 does
the same for Somali. Observable divergence from Perl:

| input | lang | Perl | sacremoses |
|-------|------|------|------------|
| `EU:ssa USA:n S:t 20:een` | fi/sv | keeps the colons in-word | `EU : ssa USA : n …` |
| `il·lusio` | ca | keeps the middle dot | `il · lusio` |
| `ha'u nia` | tdt | keeps the apostrophe | `ha &apos; u nia` |

No GitHub issue tracks any of this.

## 6. Two Czech detokenizer bugs

- `tokenize.py:844` calls `next(tokens, None)` on a **list**, which is a no-op
  (Perl used `$i++`), so the `-li` merge never happens:
  `'Je to pravda - li e-mail.'` instead of `'pravda-li'`.
- `tokenize.py:816-818` indexes `tokens[-2]` / `tokens[-1]` — the *last*
  elements — where Perl uses `$words[$i-2]` / `$words[$i-1]`. Produces
  `'9 je tady7.'`.

## 7. `learn_truecase_weights` first-token gate is off by one

`truecase.py:112` gates the sentence-initial token on `i == 1`, which is never
true at `i == 0`, so `--possibly-use-first-token` does not do what it
documents.

---

## Performance, not correctness

Measured, byte-identical, but deferred with the rest. Combined they give
**tokenize 1.98x, detokenize 2.35x, normalize 2.01x**.

1. **`islower()` / `isanyalpha()` rebuild a `set()` over a ~100K-char string on
   every call** — 53% of all tokenize time. Found independently by four
   separate analyses. Already upstream as PR #149, open and unreviewed since
   2024-05-26. **Cache per instance, not at class level:** `__init__` does
   `self.IsAlpha += cjk_chars` for `zh/ja/ko/cjk`, so a class-level cache is
   silently wrong for CJK.
2. **`normalize.py:187-200`** — substitutions are uncompiled; 45% of normalize
   time is pure `re`-module dispatch. Precompiling plus turning 62 of 84
   literal patterns into `str.replace` gives 2.01x on its own.
3. **`-j` does nothing** — measured `-j 4` = 1.01x, and for zh/ja/ko the joblib
   path is **1.86x slower than serial**, because `partial(moses.tokenize, ...)`
   pickles the whole tokenizer including ~93–186K-char compiled regexes.
4. **`-j -1` silently runs serially** — `cli.py` tests `processes == 1` but
   `util.py` tests `processes <= 1`, so joblib's own "all cores" idiom is a
   no-op with no warning.

Do **not** pursue GPU acceleration. Replacing all 23 regex substitutions with
no-ops — a perfect, free accelerator — yields only 1.71x, while item 1 above
yields 2.08x. See `audit/GPU-TOKENIZATION.md`.

---

## Deliberately declined

- **CLI stdin buffering.** Streaming breaks `truecase_file`'s second pass,
  regresses every progress bar to a bare counter, and saves nothing with
  `-j > 1` since joblib materialises the list anyway.
- **`click.echo` `UnicodeEncodeError`** on non-UTF8 Windows consoles
  (reproduced with `PYTHONIOENCODING=cp932`, `cp437`). Fixing it changes stdout
  buffering and would override a deliberate `PYTHONIOENCODING` — a maintainer
  interface decision.
- **`is_flag=True, default=True`** on `-x`/`-q`/`-d` means those flags cannot
  turn their option off. Changing it would break the CLI contract.
