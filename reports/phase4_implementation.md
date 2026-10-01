# Phase 4 — implementation checkpoint (not a final validation report)

## Implemented

- Preserved the 16 original `src.rerank_vec.pair_features_vec` columns.
- Added forward rank, margin, ratio, candidate count, reverse rank, rare address
  overlap and corpus-IDF-weighted trailing address component agreement.
- Reverse ranks are computed across all staged contenders of one country via
  SQLite, rather than independently within query batches.
- Country is used only to choose a partition and its document frequencies; it
  is not a feature.
- Added per-feature ROC AUC (including reversed-direction interpretation),
  finite-value checks, throughput logging and 87M-pair extrapolation.
- Added an opt-in Phase 3 metadata sidecar to avoid silently guessing RRF or
  reranker scores discarded by the original two-column writer.

## Verification available now

- Six tests passed: Phase 4 feature tests and existing reranker parity tests.
- Python syntax compilation passed for all three new source files.
- Full integration was NOT run here: this environment lacks pyarrow and the
  actual training parquet store. Run the smoke test on cmslab.

## Outstanding before marking Phase 4 done

1. Check the challenge PDF against the implementation and input schema.
2. Produce enriched candidate pairs for the complete train partition (one
   country at a time), subject to reviewer approval for the Phase 3 rerun.
3. Measure AUC on a query-held-out validation split and examine near-dead
   features with both AUC directions. The supplied runner's sampled AUC is
   preliminary and not a substitute for query-held-out validation.
4. Benchmark throughput, disk pressure and RAM/swap; assess whether SQLite
   reverse ranking is affordable for 87M pairs before a full test run.
5. Review corpus-derived location proxy against raw address formats.
6. Record actual measurements in the live reports/phase4.md, update the live
   PLAN.md after review, commit locally, and refresh docs/handoff-phase5.md.

**Do not report Phase 4 as complete until these measurements exist.**
