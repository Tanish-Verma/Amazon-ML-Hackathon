# Phase 4 implementation — apply as an additive patch

The ZIP contains **new files only** (plus the new `src/features.py` if your checkout
already has the earlier starter). It does not overwrite PLAN.md, existing Phase 3
code, existing reports, or the existing Phase 5 handoff.

## Why an enriched Phase 3 output is required

`src/run_blocking.py` currently writes **only** `source1_entity_id` and a comma-
separated `candidate_entity_ids` list. It drops the RRF score, channel hit count,
best channel rank and actual reranker score. The full 23-feature matrix cannot be
reconstructed exactly from the two-column output. **Do not invent zeros** for
these features or treat a candidate's ordinal position as its real score.

`src/run_blocking_enriched.py` is a copy of the existing Phase 3 entry point with
an **optional** `--metadata-out` argument. It preserves the existing two-column
candidate file and simultaneously writes an enriched one-row-per-pair sidecar.
It never overwrites existing output files. Producing that sidecar requires a
new Phase 3 run (the already running jobs cannot be retroactively changed).
Discuss the cost with your team before launching it.

## Installation and smoke test

1. On the server, `cd ~/amlc && git status` and check free RAM with `free -h`.
2. Extract the ZIP **without overwriting** (`unzip -n ...`). Verify `git diff`.
3. `source ~/amlc/venv/bin/activate` and
   `cd ~/amlc/code/business_entity_resolution`.
4. `python -m pytest -q tests/test_features.py tests/test_rerank_parity.py`.
5. Use the following on a small TRAIN partition, using a NEW output path.

```bash
# This is a *new* Phase 3 smoke run; it does not reuse the old two-column files.
python -m src.run_blocking_enriched \
  --store ~/amlc/work/store --split train --countries India \
  --model ~/amlc/work/reranker.pkl \
  --out ~/amlc/work/phase4_smoke_candidates.tsv \
  --metadata-out ~/amlc/work/phase4_smoke_pairs.tsv \
  --max-queries 100 --workers 4 --batch 100 --top-k 50

python -m src.run_features \
  --pairs ~/amlc/work/phase4_smoke_pairs.tsv \
  --store ~/amlc/work/store --split train --country India \
  --gt ~/amlc/dataset/train/train_ground_truth.tsv \
  --out ~/amlc/work/phase4_smoke_features --workers 4 \
  --query-batch 50 --min-free-gb 25
```

**Important:** reverse rank on a 100-query smoke sample is only a correctness
smoke test; it is NOT the complete-partition reverse rank. For meaningful
validation, export all train contenders for one country, then run
`src.run_features` on that full country's enriched TSV. Repeat country by
country. Don't alter memory-related parameters between smoke and full runs
without reviewing the memory evidence.

`run_features.py` writes chunked parquet files plus `feature_auc.tsv` when the
sample contains both labels. AUC is sampled from the first N labelled pairs;
for publication-grade estimates use a reproducible query-level holdout and
calculate AUC on that holdout, not the training set. The output `label` column
must NOT be fed into the feature matrix.

`run_features.py` stages the country's candidate metadata and reverse ranks in
SQLite **inside your home directory**. At ~87M pairs, staging and SQL sorting
may be expensive; benchmark and inspect free disk and RAM before any full-test
run. The code reports observed throughput and an 87M-pair extrapolation; do not
report a fabricated runtime before it has run on the server.

## File roles

- `src/features.py`: 16 original vectorized features + seven Phase 4 features.
- `src/run_blocking_enriched.py`: optional Phase 3 sidecar with exact metadata.
- `src/run_features.py`: full-partition reverse rank, per-country feature
  extraction, parquet chunks, per-feature AUC and throughput logs.
- `tests/test_features.py`: feature and reverse-rank unit tests.
- `reports/phase4_implementation.md`: honest local test status and next steps.

No git push. Do not replace your live PLAN.md or current handoff with drafts.
