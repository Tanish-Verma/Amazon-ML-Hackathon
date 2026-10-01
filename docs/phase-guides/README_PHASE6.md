# Phase 6 — local validation harness

This package adds `src/split.py`, `src/validate_phase6.py`, and `tests/test_phase6.py`. **Do not overwrite the existing `src/scoring.py`**; the package imports it unchanged. This is a validated harness component, **not a verified full end-to-end pipeline**: the blocking → features → model command must be wired to your team's working CLI and executed on real train data. No measured real-data F0.5 is claimed.

## Install safely

Inspect `git status` and the ZIP contents; extract in a new staging folder under your home directory and copy only approved new files. Do not overwrite team changes. No deletes, no pushes, no `/tmp` writes. From `~/amlc/code/business_entity_resolution` run:

    python -m pytest -q tests/test_phase6.py tests/test_scoring.py

## 1. Create dev/confirm holdouts

Use the **actual** training Source-1 TSV path; check `ls ~/amlc/dataset/train/` for its name. Run after checking `free -h`, `df -h`:

    python -m src.split --gt ~/amlc/dataset/train/train_ground_truth.tsv --source1 /ACTUAL/PATH/train_source1.tsv --out ~/amlc/work/phase6_split_v1 --dev-size 50000 --confirm-size 300000 --seed 42

These two holdouts are disjoint and reproducible. Keep BOTH out of classifier fitting, probability calibration, threshold tuning, corpus-derived feature fitting and other supervised optimization. If Phase 5 was previously fitted on these IDs, **retrain Phase 5** with a disjoint fit set before treating scores as holdout results.

**Important:** blocking on each holdout must still use the **full train S2/S3 corpus**, never a subset. The selected S1 IDs define the queries, not the retrieval corpus. Preserve country partitions as opaque labels. The current `src.run_blocking` CLI may require an adapter to accept a query-ID filter; do not silently evaluate on an easier reduced corpus.

## 2. Produce holdout predictions

Run the existing Phase 3 → Phase 4 → Phase 5 pipeline for selected Source-1 IDs, with the full S2/S3 train corpus. Produce an exact two-column UTF-8 TSV with header:

    source1_entity_id\tmatched_entity_ids

Write **one row for every holdout S1**, including explicit empty `matched_entity_ids` for predicted singletons. Do not feed Phase 5 test predictions into this harness.

## 3. Score + bootstrap CIs

    python -m src.validate_phase6 --gt ~/amlc/dataset/train/train_ground_truth.tsv --source1 /ACTUAL/PATH/train_source1.tsv --ids ~/amlc/work/phase6_split_v1/dev_ids.tsv --predictions /ACTUAL/PATH/dev_matching_results.tsv --out ~/amlc/reports/phase6_dev_v1.json --bootstrap 1000 --seed 42

Repeat for `confirm_ids.tsv` and its own predictions. Both overall and per-country metrics have stratified entity-bootstrap 95% intervals. Report dev and confirm independently; never choose thresholds on confirm and then call it an untouched holdout.

## Status / remaining integration

- Local unit tests verify deterministic disjoint splits, official worked example, singleton edge cases, reproducible bootstrap and singleton baseline.
- Full end-to-end regression and real-data measurements **cannot be claimed** until the real training corpus and Phase 3/4/5 artifacts are available and run on the server.
- The package intentionally does not change `PLAN.md`, commit, or push; document actual scores in `reports/phase6.md` after server execution and reviewer approval.
