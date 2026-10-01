# Integrated Phase 4–9 runner (existing candidate file)

**This is a new integration draft. It is not certified against your server or the official competition validator.** It uses the already-generated **test** `candidate_pairs.tsv`, never reruns Phase 3 and never overwrites it. The original project and prior Phase 4–9 modules are included.

## Critical input: labeled TRAIN candidates

You **must** supply a separate two-column `train_candidate_pairs.tsv` built using the training S1 queries against the training S2/S3 corpus. The test `candidate_pairs.tsv` has no ground truth and **cannot** be used for training. If you do not have training candidates, pass `--blocking-model /path/to/your/trained_phase3_model.pkl` and the runner generates them using your existing Phase 3 blocker on the TRAIN split only (not the test split). Do not fabricate positive-only training candidates: that makes the classifier invalid.

## Missing Phase 3 metadata

`src.existing_candidates` computes a NEW deterministic lexical score from genuine name/address/digit similarities for the pairs in the existing TSV. It sets unavailable RRF/channel fields to **zero** in both training and inference. This is a **different feature distribution and a newly trained model**, not a reconstruction of Phase 3 metadata. Model quality and runtime are unmeasured. Reverse rank is computed globally per country using the new score.

## Install safely

Extract into a **new** directory, not over the shared team's repo. Activate the existing virtualenv. From `code/business_entity_resolution`, run `python -m pytest -q tests/` and `python -m src.pipeline --help`. If your existing repo has newer changes, reconcile before deploying.

## One-command smoke test (replace all placeholder paths)

```bash
cd ~/amlc_integrated/code/business_entity_resolution
source ~/amlc/venv/bin/activate
python -m src.pipeline \
 --store ~/amlc/work/store \
 --train-candidates ~/amlc/prod/train_candidate_pairs.tsv \
 --test-candidates ~/amlc/prod/candidate_pairs.tsv \
 --gt ~/amlc/dataset/train/train_ground_truth.tsv \
 --out ~/amlc/work/integrated_smoke_v1 \
 --max-queries 50 --skip-audit --workers 2 --min-free-gb 25
```

## Full run

Use a NEW output directory, remove `--max-queries` and `--skip-audit`, verify actual training candidate coverage first. The runner sequentially: (4) recomputes available metadata and builds 23-column country feature Parquets, (5) trains and predicts with LightGBM, (8) writes matching_results.tsv aligned to the original candidate file and audits both. Phase 6 and 7 experimental modules remain separate because holdout and transfer evaluations require separate training experiments; Phase 9 packaging is optional and requires an official-format manifest. **Do not claim that phases 6/7 have been run by this command.**

## Important limitations

- The per-query adapter builds a corpus-wide sparse store per country and computes lexical features per query. This is functional but may be **too slow** for 1.73M test entities; benchmark 50/500/5000 first.
- `run_features.py` builds another corpus store and disk-backed SQLite ranks. Check free RAM/disk and run one country at a time.
- Phase 5 tuning is based on labeled training candidates; inspect validation and coverage before trusting it.
- The official submission archive layout must be checked against the current competition statement. `src.phase9` requires an explicit reviewed manifest.
- The integrated runner does not automatically execute Phase 6 or Phase 7 generalization experiments; their existing CLI modules are included.
