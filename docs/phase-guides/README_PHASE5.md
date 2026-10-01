# Phase 5: Matching model + decision rule

This is an implementation for the **specific uploaded project**, not a verified leaderboard submission. It assumes Phase 4 has produced **complete**, labeled train Parquet and unlabeled test Parquet files containing the 23 `src.features.FEATURE_NAMES` plus `source1_entity_id` and `candidate_entity_id`. Train Parquet also requires `label`. **Phase 4 must be validated first.** France is test-only; the pooled India/US model can be applied to it, but its quality cannot be measured using train labels.

## Files
- `src/model.py`: bounded training-pair sampling, entity-level fit/calibration/tune split, LightGBM, isotonic calibration, threshold/margin grid, macro-F0.5, per-country report.
- `src/decide.py`: precision-first per-entity margin and globally exclusive candidate ownership.
- `src/predict_phase5.py`: one-country-at-a-time streaming inference, disk-backed global assignment (SQLite) and memory checks.
- `src/assemble_phase5.py`: add all S1 test entities, including predicted singletons, to matching_results.tsv.
- `tests/test_decide.py`: rule tests.

## Installation (do not overwrite team files)
Unzip in a NEW staging folder; compare with your existing repo and copy only files approved by your team. No existing files need replacing. Use your existing `~/amlc/venv` and installed `lightgbm`, `pyarrow`, `scikit-learn`, `psutil`.

From `~/amlc/code/business_entity_resolution`, run `python -m pytest -q tests/test_decide.py`.

## Train, after Phase 4 labeled feature generation

    python -m src.model --features ~/amlc/work/phase4_train --gt ~/amlc/dataset/train/train_ground_truth.tsv --out ~/amlc/work/phase5_model_v1 --threads 8 --max-rows 800000 --tune-entities 3000 --min-free-gb 25

Review `~/amlc/work/phase5_model_v1/metrics.json`. The code reports sampled validation; it does not establish the challenge's final score. The tuning grid includes exclusive/nonexclusive ablation, but reports only the top 20 grid rows. A dedicated larger holdout and margin-stealing experiments remain for reviewer approval.

## Test inference (one country at a time)

    python -m src.predict_phase5 --features ~/amlc/work/phase4_test/India --model ~/amlc/work/phase5_model_v1/model.pkl --out ~/amlc/work/phase5_india_v1 --min-free-gb 25
    python -m src.predict_phase5 --features ~/amlc/work/phase4_test/US --model ~/amlc/work/phase5_model_v1/model.pkl --out ~/amlc/work/phase5_us_v1 --min-free-gb 25
    python -m src.predict_phase5 --features ~/amlc/work/phase4_test/France --model ~/amlc/work/phase5_model_v1/model.pkl --out ~/amlc/work/phase5_france_v1 --min-free-gb 25

Run each command **only after the previous one finishes**. Country folder names are examples: use actual Phase 4 output directories.

## Final matching file

    python -m src.assemble_phase5 --store ~/amlc/work/store --country-results ~/amlc/work/phase5_india_v1/matches_nonempty.tsv ~/amlc/work/phase5_us_v1/matches_nonempty.tsv ~/amlc/work/phase5_france_v1/matches_nonempty.tsv --out ~/amlc/output/matching_results_phase5_v1.tsv

The organizer also requires **candidate_pairs.tsv**, the exact Phase 3 post-rerank candidate set. Preserve and assemble it separately using the project's existing Phase 3 writer. Run the organizer's validator against both files before submission. **Do not assume the Phase 5 matching file is itself `candidate_pairs.tsv`.**

## Safety and limitations
- No deletes, no pushes, no use of `/tmp` in this package. SQLite scratch stays in the output folder; `TMPDIR` is set for the Python process, but external native libraries may use their own temp policies.
- Check `free -h`, `df -h` and other running jobs before any bulk job. SQLite may need substantial home-disk space for country-wide eligible pairs; this package does not pre-estimate disk requirements.
- The training sampler selects pairs uniformly, so the sampled tuning entities are recovered in full with a second pass. This is a *development* validation, not a formal untouched final holdout.
- Calibrator and threshold are trained using India/US labels only. France is out-of-domain; verify generalization as Phase 6 describes.
- No trained model or real validation metrics are included; they require the actual data and execution on cmslab.
