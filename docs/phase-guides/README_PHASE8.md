# Phase 8: inference and packaging

This adds `src/phase8.py` and tests; it does not replace existing Phase 3-5 code or overwrite existing files. Run from `~/amlc/code/business_entity_resolution` in `~/amlc/venv`.

**Prerequisites:** `src/run_blocking_enriched.py` (Phase 4 ZIP), `src/run_features.py` and `src/features.py` (Phase 4), `src/predict_phase5.py` (Phase 5), both trained pickle models, and completed test Parquet partitions. Phase 4 and 5 must already pass actual-data smoke tests. The original two-column candidate files alone cannot supply all Phase 4 features. Never fabricate missing metadata.

## Install

Upload ZIP to `~/` on server; `cd ~/amlc && git status && unzip -l ~/phase8_inference_code.zip`. Extract without overwriting: `unzip -n ~/phase8_inference_code.zip`. Activate venv and `cd code/business_entity_resolution`. Run `python -m pytest -q tests/test_phase8.py`.

## Option A: assemble existing completed country files (no rerun)

If each country has a complete `candidates.tsv` (2-column Phase 3) and Phase 5 `matches_nonempty.tsv`, run:

```bash
python -m src.phase8 assemble \
  --store ~/amlc/work/store \
  --candidates ~/amlc/prod/cand_US.tsv ~/amlc/prod/cand_India.tsv ~/amlc/prod/cand_France.tsv \
  --matches ~/amlc/work/p5_US/matches_nonempty.tsv ~/amlc/work/p5_India/matches_nonempty.tsv ~/amlc/work/p5_France/matches_nonempty.tsv \
  --out ~/amlc/output/phase8_v1
```

**These paths are examples, not verified server paths.** Check `ls -lh` and actual country names. Every candidate TSV must contain one row for every S1 in its country. Empty candidate lists are allowed, but missing candidate rows are not silently fabricated. Phase 5 `matches_nonempty.tsv` may omit singleton rows.

## Option B: full per-country run (CPU-intensive; get team approval)

```bash
python -m src.phase8 run \
  --store ~/amlc/work/store \
  --blocking-model ~/amlc/work/reranker.pkl \
  --matching-model ~/amlc/work/phase5_model_v1/model.pkl \
  --countries US India France \
  --out ~/amlc/output/phase8_full_v1 \
  --workers 8 --top-k 50 --min-free-gb 25
```

The `run` subcommand executes **blocking (enriched sidecar) → Phase 4 features → Phase 5 predictions** for each country, then assembles final files. It will NOT run if `--out` exists. It does not redo raw TSV → Parquet preparation; use `python -m src.cli prep ...` beforehand if required. It does not automatically retrain models. Verify actual country partition names and trained model paths. On a shared server, do not start while Phase 3 jobs are still running.

## Audit and official validator

```bash
python -m src.phase8 audit \
  --store ~/amlc/work/store \
  --candidates ~/amlc/output/phase8_v1/candidate_pairs.tsv \
  --matches ~/amlc/output/phase8_v1/matching_results.tsv \
  --check-ids

python3 utils/validate_submission.py \
  -m ~/amlc/output/phase8_v1/matching_results.tsv \
  -c ~/amlc/output/phase8_v1/candidate_pairs.tsv \
  -t ~/amlc/dataset/test --check-ids
```

**Only claim PASS if the official validator actually prints PASS.** Confirm the real location of `utils/validate_submission.py` (it was absent from the provided project ZIP) and its CLI options before running. The internal audit checks exact headers, expected 1,732,544 rows, synchronized rows, duplicate S1 IDs, duplicate IDs within lists, S2/S3 prefixes, matches subset of candidates, global exclusive assignment, and optionally existence of S2/S3 IDs. It uses SQLite files under your home directory and does not delete pre-existing user files. `audit` uses Python's `TemporaryDirectory` to clean up only its own newly created audit scratch folder.

`assembly_metrics.json` records predicted singleton rate and assembly time. `runtime.json` records per-country stage times for `run`. **Peak RSS is not yet instrumented** (recorded as null): use `/usr/bin/time -v` or scheduler accounting for production peak memory. The official validator and full real-data pipeline have not been run in this package; test on a small complete synthetic/test subset first with `--expected` adjusted.
