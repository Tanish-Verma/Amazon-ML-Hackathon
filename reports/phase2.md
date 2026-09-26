# Phase 2 — Normalisation: verification report

Author of the module: teammate (as `notebooks/normalize_phase2.ipynb`).
Extracted to `src/normalize.py`, tested and verified on `cmslab` 2026-09-26.

## Verdict: correct. Phase 3 can proceed.

The notebook's own 19 checks all passed, but every one used invented strings and
none had run on the server. Below is verification against real data on real
hardware.

## 1. Crash / integrity sweep — all 24.2M records

```
train/s1 2,206,821   test/s1 1,732,544
train/s2 5,034,616   test/s2 4,887,273
train/s3 5,285,603   test/s3 5,082,316
-------------------------------------------------
total 24,229,173 records | 0 crashes | 0 non-empty names normalising to empty
```

Nothing crashes, and no record is silently annihilated — the failure mode that
would quietly destroy recall.

## 2. Throughput

**37,463 records/sec single-core** (includes `clean_text` plus
`extract_numeric_keys`, over real varied names and addresses).

This is **below** the ~100k/sec/core bar `PLAN.md` set, so the bar was not met as
written. In practice it does not matter: the full 24.2M-record corpus normalises
in ~11 minutes on one core, and the box has 48. Normalisation is not the
bottleneck, so this is recorded as a missed target rather than a problem to fix.
(The notebook reported 78k/sec, but that measured `clean_text` alone over three
short repeated strings, so the two numbers are not comparable.)

## 3. DF-derived stoplists on real per-country corpora

Built from the test split, which includes France. The threshold matters more than
expected:

| Threshold | France | India | US | Assessment |
|---|---|---|---|---|
| 0.02 (notebook default) | 30 tokens | 11 | 16 | **Too aggressive.** US loses `care`, `associates`, `center`, `group`, `partners`, `corp` — discriminative business words. |
| **0.05** | 9 | 5 | 2 | **Best.** Catches legal forms and country names, keeps discriminative words. |
| 0.10 | 2 | 4 | 2 | Too loose for France — drops `eurl`, `sasu`, `sci`. |

At 0.05 the derived stoplists are:

```
France : sarl(28%) sas(20%) club(9%) france(8%) de(7%) eurl(7%) ecole(6%) amicale(6%) comite(5%)
India  : limited(59%) private(49%) ltd(16%) pvt(14%) india(7%)
US     : llc(27%) inc(18%)
```

**The France mechanism works on real data with no French-specific code** —
`sarl`/`sas`/`eurl` are caught purely by frequency, alongside the function word
`de` that an English stoplist would have missed.

Note the per-country asymmetry is a feature, not a bug: France's corpus genuinely
has higher-DF category words (`club`, `ecole`, `amicale`) than the US's, and a
single global threshold adapts to each country's own distribution.

**Change applied:** default `min_df_fraction` moved from 0.02 to 0.05 on this
evidence. Phase 3 should still tune it against end-to-end F0.5.

## 4. Reachability regression — the test that matters

Does this normaliser preserve the lexical ceiling the whole plan rests on?
Identical sample and thresholds as `reports/eda.md`:

| Channel | This run | reports/eda.md | Delta |
|---|---|---|---|
| Name | 90.43% | 90.43% | +0.00 |
| Address | 94.82% | 94.82% | +0.00 |
| **Either (ceiling)** | **99.88%** | **99.88%** | **-0.00** |

Exact match. No regression.

Honest caveat: exact equality means this metric cannot *distinguish* the two
normalisers — it confirms nothing was lost, but does not demonstrate an
improvement over the EDA's inline version. That is the right result to want here.

## 5. Two gaps, neither blocking

**No address normalisation output.** `NormalizedRecord` exposes name-side fields
plus `numeric_keys`; the address text is used only for digit extraction. Phase 3's
address channel needs rare address *word* tokens too. Not blocking —
`clean_text()` is script-agnostic and works on addresses directly
(`"6(29), C.I.T. Colony, Chennai"` → `"6 29 c i t colony chennai"`). Phase 3
should call it and consider adding `address_clean`/`address_tokens` to the record.

**TLD not stripped from `joined`.** `maurewilliamscolombier.com` →
`maurewilliamscolombiercom`, which will not exact-match the plain name's
`maurewilliamscolombier`. Measured impact: negligible, because `joined` exact
matching is not the retrieval mechanism — char-3gram Jaccard between those two
strings is **0.870**, far above the 0.15 blocking bar. Left as-is.

## 6. Test-suite note

The notebook's check #9, "no hardcoded country-name branching", referenced an
undefined `_country_literal_findings` variable and therefore could never fail.
`tests/test_normalize.py` replaces it with one that actually reads the module
source (excluding the feature-flagged suffix table and all comments). It passes.

The ported suite also adds coverage the notebook lacked: all nine Indic scripts
rather than Devanagari alone, the zero-padding equivalence `AF-684` ≡ `AF-0684`,
a real cross-record numeric-key check, and a guarantee that `core_tokens` never
returns empty.

**14/14 tests pass** on `cmslab`.
