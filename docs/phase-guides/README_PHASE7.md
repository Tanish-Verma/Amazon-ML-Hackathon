# Phase 7 — unseen-country generalization and audit

This package implements **evaluation, provenance checks, code audit and unlabeled drift checks**. It does not pretend to fit six independent blocking/feature/model runs automatically: those must be executed with the actual working Phase 3–5 CLI, and training-time country leakage must be verified. No actual F0.5 or France accuracy has been measured here.

## Required experimental matrix

| Run | Fit country | Evaluation | Dictionaries |
|---|---|---|---|
| `us_baseline` | US | US held-out | on |
| `us_to_india` | US | India held-out | on |
| `us_to_india_ablated` | US | same India IDs | off |
| `india_baseline` | India | India held-out | on |
| `india_to_us` | India | US held-out | on |
| `india_to_us_ablated` | India | same US IDs | off |

**Do not fit IDF, classifiers, scalers, probability calibrators or thresholds on the target country's labels.** Rebuilding unlabeled target-country retrieval indexes is allowed if declared in the manifest as `target_unlabeled_only`; it is a transductive retrieval setup, not a strict zero-shot setup. For strict zero-shot, use `training_only` and document how retrieval works without target statistics. For each experiment, preserve the *full target-country S2/S3 corpus*, while filtering only the S1 queries to the holdout. Ensure target S1 IDs and their labels have not entered fitting, threshold tuning or dictionary construction. All three evaluations for a target must use identical IDs.

The dictionary ablation must actually disable **both** handwritten legal suffix and abbreviation maps in the production normalization/feature code, and rerun blocking/features/prediction as appropriate. A configuration flag that is ignored is not an ablation. Compare with the on-dictionary transfer run, using the same target IDs and otherwise equivalent settings. Do not infer ablation effects from precomputed features generated with dictionaries enabled.

## Install

Stage and review the ZIP before copying new files into `~/amlc/code/business_entity_resolution`. Do not overwrite team edits. No deletion, push, or writes outside your home directory.

```bash
cd ~/amlc
source venv/bin/activate
cd code/business_entity_resolution
python -m pytest -q tests/test_generalization.py tests/test_phase6.py tests/test_scoring.py
```

## Generate evaluation JSON

For each of the six runs, use the Phase 6 evaluator, with its own prediction TSV and the correct target-country holdout IDs. Its `--source1` and `--gt` are the **full train** Source-1 and ground-truth files. It rejects missing S1 rows; output must include explicit singleton rows.

```bash
python -m src.validate_phase6 \
  --gt ~/amlc/dataset/train/train_ground_truth.tsv \
  --source1 /ACTUAL/PATH/train_source1.tsv \
  --ids /ACTUAL/PATH/india_holdout_ids.tsv \
  --predictions /ACTUAL/PATH/us_to_india_predictions.tsv \
  --out ~/amlc/reports/p7_us_to_india.json --bootstrap 1000 --seed 42
```

Repeat with each run's own paths; keep same target holdout IDs across baseline/transfer/ablation. Inspect the generated JSON to confirm its metric keys (`micro_precision`, `micro_recall`, `macro_f05`) as implemented in the supplied `src.scoring`. Precision/recall are **micro**; F0.5 is **macro**. Undefined micro metrics are reported as `null` in Phase 7. The code deliberately fails instead of silently guessing keys.

## Provenance manifest

Create `~/amlc/work/phase7_manifest.json` with `runs` containing **exactly** the six names in the matrix. For each run, include:

```json
{
  "fit_country": "US",
  "eval_country": "India",
  "dictionaries_enabled": true,
  "fit_ids_sha256": "REAL_SHA256_OF_FIT_ID_LIST",
  "eval_ids_sha256": "REAL_SHA256_OF_TARGET_ID_LIST",
  "model_sha256": "REAL_SHA256_OF_TRAINED_MODEL",
  "prediction_path": "/home/co24btech11023/amlc/work/.../predictions.tsv",
  "phase6_report_path": "/home/co24btech11023/amlc/reports/p7_us_to_india.json",
  "blocking_config_sha256": "REAL_SHA256",
  "normalization_config_sha256": "REAL_SHA256",
  "feature_config_sha256": "REAL_SHA256",
  "threshold_config_sha256": "REAL_SHA256",
  "corpus_stats_provenance": "target_unlabeled_only"
}
```

The above object is the value for the `us_to_india` key inside `runs`, **not** a complete manifest. Generate hashes using `sha256sum FILE`; never insert placeholder hashes. Baseline and transfer for the same fit country must reuse the exact model and threshold config. Ablation can require recomputing the feature pipeline; if that also changes model fitting, document it and do not interpret the result as a dictionary-only ablation. The manifest checker validates declared metadata, not the actual history of training: review run logs and code separately.

```bash
python -m src.generalization report \
  --manifest ~/amlc/work/phase7_manifest.json \
  --out ~/amlc/reports/phase7_generalization.json
```

## Literal-country audit

```bash
python -m src.generalization audit \
  --root ~/amlc/code/business_entity_resolution/src \
  --out ~/amlc/reports/phase7_audit.json
```

Review **every** hit manually. Opaque country partition values are acceptable; country-conditioned thresholds, hard-coded France dictionaries and script-specific fallback rules are not. This audit catches Python string literals only; also manually inspect JSON/YAML configs, shell scripts and data-derived rules.

## Unlabeled France drift

Use the **actual test-set** `candidate_pairs.tsv` and `matching_results.tsv`, or per-country equivalents, with headers `source1_entity_id`, `candidate_entity_ids` and `source1_entity_id`, `matched_entity_ids`. If your current candidate output is pair-per-row, convert it to the competition's list-per-S1 format first. For a mixed-country file, filter by S1 country before calling the command. The command checks every match belongs to the candidate list.

```bash
python -m src.generalization drift --country France \
  --candidates /ACTUAL/PATH/france_candidate_pairs.tsv \
  --predictions /ACTUAL/PATH/france_matching_results.tsv \
  --out ~/amlc/reports/phase7_france_drift.json
```

Repeat for US and India; compare candidate count percentiles and predicted singleton rates. This implementation **does not** report score distributions unless the production inference step also exports a per-pair score file. Add those when available. Do not claim France precision, recall or F0.5 without France ground truth.

## Completion gate

Write `reports/generalization.md` with the six run commands/config hashes, split hashes, actual Phase 6 scores and bootstrap CIs, both transfer deltas, dictionary ablations, reviewed audit hits, and France/US/India drift statistics. Phase 7 is complete only after all experiments are executed on the server and reviewed. Do not touch the official submission outputs until the team approves the results.
