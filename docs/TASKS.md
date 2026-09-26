# Project status, what's left, and how to do it

**Updated:** 2026-09-26 · **Deadline:** 2026-09-27 23:59 IST

New here? Read in this order:
1. `docs/README-for-the-team.md` — what the problem is, in plain English
2. `code/business_entity_resolution/README.md` — architecture, folder layout, how to run
3. `docs/DATA.md` — which data files ship and how to regenerate the rest
4. This file — status, and which prompt to pick up

---

## 1. Status

| Phase | What it is | Status |
|---|---|---|
| 0 | Environment, schema checks, output plumbing | ✅ done |
| 1 | EDA — the noise, the ceiling, what actually matters | ✅ done |
| 2 | Normalisation (script-agnostic, 9 Indic scripts + French) | ✅ done, verified on 24.2M records |
| 3 | **Blocking + reranking → `candidate_pairs.tsv`** | ✅ **done, ~98.2% recall** — production run in progress |
| 4 | Features for the final matcher | ⬜ **ready to start** |
| 5 | Matching model + decision rule | ⬜ **highest-value phase** |
| 6 | Validation harness | ⬜ |
| 7 | Unseen-country (France) stress test | ⬜ gates touching test data |
| 8 | Full test inference + packaging | ⬜ |
| 9 | Reproducibility + methodology write-up | ⬜ |

**We already have a submittable file.** `output/matching_results.tsv` currently
predicts "no match" for everything and scores ~0.056. Worthless as a score, but it
passes the official validator — so we can never end up with nothing to submit.

## 2. What Phase 3 delivered

| measurement | value |
|---|---|
| Blocking pool recall (depth 1000) | 98.8% India / 99.3% US |
| After reranking, at K=50 | **97.4% India / 99.3% US** |
| Projected test-set recall | **~98.2%** |
| First (broken) implementation | 71.0% |

`candidate_pairs.tsv` gives every Source-1 entity **50 candidates**, containing
~98% of its true matches. Everything downstream works inside that 50.

## 3. What's left, and how to do it

Each phase has a **self-contained prompt**. Paste it into a fresh Claude Code
session — it carries the machine access, the standing rules, and every established
fact, so nothing gets re-derived.

| Phase | Prompt | What you're building |
|---|---|---|
| 4 | `docs/handoff-phase4.md` | Feature matrix for the matcher. Extends `rerank_vec.py`; adds rank/margin/contention features. |
| 5 | `docs/handoff-phase5.md` | The matcher + decision rule. **Most of the remaining score lives here.** |
| 6 | `docs/handoff-phase6.md` | Validation splits, bootstrap CIs, regression check. |
| 7 | `docs/handoff-phase7.md` | Hide-a-country stress tests. Gates touching test data. |
| 8 | `docs/handoff-phase8.md` | Full inference, validator PASS, submission zip. |
| 9 | `docs/handoff-phase9.md` | `Documentation_template.md`, pinned deps, clean-room repro. |

Each prompt ends by telling Claude to write the next one, so the chain continues.

**Phases 4 and 6 can run in parallel** with different people — 6 only needs the
existing scorer. Phase 5 depends on 4. Phase 7 depends on 5.

## 4. Why Phase 5 matters most

The metric is **F0.5**: precision counts double. At precision 0.9:

* four extra points of blocking **recall** ≈ **+0.008** F0.5
* seven extra points of **precision** ≈ **+0.055** F0.5

Blocking is finished. The score is now won or lost in the decision rule.

**The biggest single lever is already identified and verified:** every Source-2/3
record belongs to **at most one** Source-1 entity — checked across all 7,638,365
ground-truth links. Turning that into an assignment constraint (each record goes to
one entity, contention resolved by score) removes a whole class of false merges that
no per-pair threshold can catch. Combined with 5.58% true singletons, where a single
false match costs a full point, that is where the remaining points are.

## 5. Things not to undo

Each cost real time to discover:

* **Country is a partition, never a feature.** One-hot country is exactly what
  breaks on France — 15% of the score, with no training data.
* **Stoplists are derived from corpus frequency, never hand-written.** That is why
  French `sarl`/`sas`/`eurl` work with no French-specific code, and why French
  function words (`de`, `du`, `des`) are caught too.
* **Fuse ranked lists with RRF, not by summing scores.** Summing cost 16 points of
  recall, because channel score magnitudes differ by an order of magnitude.
* **No per-record Python object stores at corpus scale.** 40 GB of frozensets plus
  forked workers exhausted a 125 GB machine twice. Sparse matrices: 0.68 GB.
* **Don't change memory parameters between a smoke test and the real run.** That
  invalidates the test, and it is exactly how the first OOM happened.
* **Reachability is not retrievability.** "99.88% of pairs share some signal" is not
  "98% rank in the top 50 of 4M records". Always quote a depth.

## 6. Measured dead ends — don't spend time re-testing these

| Idea | Result |
|---|---|
| Multilingual embedding channel | **+0.12% recall.** Not worth GPU time; the address channel already solves transliteration because digits survive script change. |
| LightGBM as reranker | 54% @10 vs 94% for logistic regression — though it was given a classification objective for a ranking problem. Headroom above LR is ≤1.4 points either way. |
| Tighter `max_df` | Every early sweep went the wrong way; 0.002 cost ~5 points vs 0.01. |
| Shallower rerank pool | depth 700 = −0.17%, depth 400 = −0.43/−0.64%. Only 1.43× for the former. Left at 1000. |
| Per-query broadcast to avoid duplicated gathers | **10% slower** — ~100k small scipy calls beat by one large vectorised gather. |
| Forked scorer processes | 2.8× and now safe, but deliberately not shipped (see `docs/handoff-phase4.md` context). |
