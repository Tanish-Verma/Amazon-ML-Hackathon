# Phase 9 — reproducibility and submission package

This package supplies a **safe, exact-layout, manifest-driven archive builder**, tests, a methodology drafting template, and a final checklist. It **does not claim the final submission is ready**: the actual competition problem statement PDF and `Documentation_template.md` were not in the supplied project snapshot, and the Phase 4–8 real-data results have not been supplied. Never invent the required archive layout, metrics, licenses, or test outcomes.

## Install and test

Upload the ZIP to your home directory on `cmslab`. Check `git status` and `unzip -l ~/phase9_code.zip`, then extract in `~/amlc` using `unzip -n ~/phase9_code.zip`. This adds `src/phase9.py`, `tests/test_phase9.py`, and docs without overwriting existing files. Activate `~/amlc/venv`; from `~/amlc/code/business_entity_resolution`, run:

```bash
python -m pytest -q tests/test_phase9.py
python -m src.phase9 --help
python -m src.phase9 environment --out ~/amlc/reports/phase9_environment.json
```

## Prerequisites to complete Phase 9

1. Finish Phases 4–8 with actual data. Record measured macro F0.5, confidence intervals, thresholds, error analysis, country-transfer measurements, wall time and peak RSS. Run the official validator on both completed TSVs.
2. Open the **official problem statement** and **original Documentation_template.md**. Fill the template's *actual* fields using the accompanying `docs/phase9_methodology_draft.md` as a source. Do not submit the draft instead of the organiser's template unless permitted.
3. Audit imports in the actual final code and pin only needed runtime dependencies. Build a **fresh venv** from the pinned requirements, then run all tests. Do not blindly replace the existing requirements file.
4. Confirm model licensing from the versions and official repositories actually shipped. For the final CPU-only pipeline, check LightGBM and scikit-learn licensing; no embedding model should be claimed unless actually shipped.
5. In a **fresh directory under your home**, reproduce the final outputs from raw input data. Keep this separate from the production repo; don't delete or overwrite existing files.

## Exact submission ZIP

Create `~/amlc/docs/phase9_manifest.json` with an entry for **every file required by the official problem statement**. `archive_path` is the path *inside the ZIP*, including any mandated top-level team directory. The following is an **illustrative manifest, not the verified competition layout**:

```json
{
  "files": [
    {"source": "/home/co24btech11023/amlc/output/phase8_v1/candidate_pairs.tsv", "archive_path": "candidate_pairs.tsv"},
    {"source": "/home/co24btech11023/amlc/output/phase8_v1/matching_results.tsv", "archive_path": "matching_results.tsv"},
    {"source": "/home/co24btech11023/amlc/Documentation_template.md", "archive_path": "Documentation_template.md"}
  ]
}
```

**Replace this example with the exact required structure**; include source code, requirements, and trained models if the statement demands them. Do not assume the above three files are sufficient. Use the actual team name in the filename, and create a new output path each time:

```bash
python -m src.phase9 build \
  --manifest ~/amlc/docs/phase9_manifest.json \
  --out ~/amlc/output/TEAM_NAME_submission_v1.zip
python -m src.phase9 verify --archive ~/amlc/output/TEAM_NAME_submission_v1.zip
unzip -l ~/amlc/output/TEAM_NAME_submission_v1.zip
```

The builder refuses to overwrite, rejects path traversal, checks ZIP CRCs and verifies SHA-256 for every file. It **does not** substitute for the official submission validator or the required clean-room reproduction.

## Clean-room reproduction

Under a new directory in `~/`, check out the exact final local commit, create a fresh Python 3.11 venv, install the audited pinned requirements, run the entire test suite, prepare Parquet from raw TSV, retrain or load the **shipped** model as the official rules require, regenerate both final TSVs, and run `utils/validate_submission.py --check-ids` using its actual CLI. Record the commands, checksums, versions, elapsed times and peak RSS in `reports/phase9_reproduction.md`. Never use `/tmp`, `/scratch`, or `/dev/shm` on the shared machine.

Do not `git push`; commit locally only after review. Do not delete existing files.
