# Business Entity Resolution — ML Challenge 2026

Resolves which Source-2 / Source-3 business records refer to the same real-world
entity as each Source-1 record, across US, India and (test-only) France.

> **Status: Phase 0 complete.** Environment, schema verification and the output
> path are in place. Blocking, features and the matching model follow. This file
> is filled out fully in Phase 9; run instructions below are already accurate.

## Setup

```bash
bash src/bootstrap.sh /path/to/project_root      # creates venv, installs pinned deps
source /path/to/project_root/venv/bin/activate
```

Requires Python 3.11. The embedding stage additionally needs an NVIDIA GPU with a
CUDA 12.x driver; the rest of the pipeline is CPU-only.

## Layout

```
dataset/{train,test}/*.tsv     # inputs, tab-separated, no quoting
output/matching_results.tsv    # final matches  (scored on the leaderboard)
output/candidate_pairs.tsv     # blocking candidate set fed to the matcher
```

## Running

Each stage is separately runnable, so blocking recall can be inspected
independently of final matching precision.

```bash
python -m src.cli --data-dir dataset verify                 # schema assertions
python -m src.cli --data-dir dataset --work-dir work prep   # TSV -> partitioned parquet
python -m src.cli --data-dir dataset --out output baseline  # all-singleton baseline
```

Validate any submission before uploading:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test --check-ids
```

## Design notes

* **Country is a partition key, never a feature.** Ground truth confirms no true
  match crosses a country boundary, so partitioning on the literal label both
  shrinks the candidate space and generalises to unseen countries for free.
  Nothing in the pipeline is conditioned on the set `{US, India}`.
* **Matches are one-to-many.** Every S2/S3 record is claimed by at most one S1
  entity (verified over all 7,638,365 ground-truth links), which the final
  decision rule exploits as an assignment constraint to suppress false merges.
* **The metric is precision-heavy** (F0.5, macro-averaged per S1 entity,
  singletons included), so the decision rule abstains rather than guessing.
