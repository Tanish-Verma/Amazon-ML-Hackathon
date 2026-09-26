# Where we are, and who can pick up what

Last updated: 2026-09-26 · Blocking is **done and measured**. Phases 4–9 remain.

## Status

| Phase | What it is | Status |
|---|---|---|
| 0 | Environment, schema checks, output plumbing | ✅ done |
| 1 | EDA — understand the noise and the ceiling | ✅ done |
| 2 | Normalisation (script-agnostic) | ✅ done, verified on 24.2M records |
| 3 | **Blocking + reranking → `candidate_pairs.tsv`** | ✅ code done, **~98.2%** recall; production run in progress |
| 4 | Features for the final matcher | ⬜ ready to start — `docs/handoff-phase4.md` |
| 5 | Matching model + decision rule | ⬜ `docs/handoff-phase5.md` |
| 6 | Validation harness | ⬜ `docs/handoff-phase6.md` |
| 7 | Unseen-country (France) stress test | ⬜ `docs/handoff-phase7.md` |
| 8 | Full test inference + packaging | ⬜ `docs/handoff-phase8.md` |
| 9 | Reproducibility + methodology write-up | ⬜ `docs/handoff-phase9.md` |

We already have a **valid submission file** that scores ~0.056 (predicts "no match"
for everything). Worthless as a score, but it proves our output format is accepted,
so we can never end up with nothing to submit.

## How to pick up a phase

1. Read `code/business_entity_resolution/README.md` — pipeline, folder structure,
   and why each decision was made.
2. Open `docs/handoff-phase<N>.md`, paste it into a fresh Claude Code session. Each
   is self-contained: it carries the machine details, the standing rules, and every
   established fact so nothing gets re-derived.
3. Work one phase at a time. Each prompt ends by telling Claude to write the next
   phase's prompt, so the chain continues itself.

## Which phase matters most

**Phase 5.** The metric weights precision 2×, so at precision 0.9, four extra
points of blocking recall are worth about **+0.008** F0.5, while seven points of
precision are worth about **+0.055**. Blocking is finished; the score is now won or
lost in the decision rule.

The biggest single lever there is already identified and verified: **every S2/S3
record belongs to at most one S1 entity** (checked across all 7,638,365 ground-truth
links). Turning that into an assignment constraint removes a whole class of false
merges that no per-pair threshold can catch.

## Things not to undo

These each cost real time to discover:

- **Country is a partition, never a feature.** One-hot country is exactly what
  breaks on France, which is 15% of the score and has no training data.
- **Stoplists are derived from corpus frequency, never hand-written.** That is why
  French `sarl`/`sas`/`eurl` are handled with no French-specific code.
- **Fuse ranked lists with RRF, not by summing scores.** Summing cost 16 points of
  recall because channel score magnitudes differ.
- **No per-record Python object stores at corpus scale.** 40 GB of frozensets plus
  forked workers exhausted a 125 GB machine twice. Sparse matrices: ~4 GB.
- **Don't change memory parameters between a smoke test and the real run.** That
  invalidates the smoke test, and it is how the first OOM happened.
