# Submission checklist — final verification record

**Challenge:** Amazon ML Challenge 2026, Business Entity Resolution
**Completed:** 2026-09-27 · **Final leaderboard F0.5: ~0.90**

## Deliverables

| Artefact | Status | Verification |
|---|---|---|
| `output/matching_results.tsv` | ✅ | 1,732,544 rows · 113,807 singletons (6.57%) · mean 3.06 matches · validator **PASS** |
| `output/candidate_pairs.tsv` | ✅ | 1,732,544 rows · 50 candidates each (99.99%) · validator **PASS** |
| `code/business_entity_resolution/src/` | ✅ | 30 modules |
| `code/business_entity_resolution/tests/` | ✅ | **52 tests passing** |
| `code/business_entity_resolution/README.md` | ✅ | architecture, layout, run instructions |
| `code/business_entity_resolution/requirements.txt` | ✅ | pinned |
| `Documentation_template.md` | ✅ | all sections filled with measured numbers |

## Official validator

```
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test --check-ids

  required S1 entities: 1732544
  valid S2/S3 match IDs: 9969589
  matching_results.tsv: 1732544 rows (113807 empty, 1618737 non-empty)
  candidate_pairs.tsv:  1732544 rows (10 empty, 1732534 non-empty)

PASS — no blocking issues found. Safe to submit.
```

Run on the 125 GB server — `--check-ids` loads ~10M IDs and will OOM a small machine.

## Rule compliance

| Rule | Status |
|---|---|
| Output format exactly as specified | ✅ validator PASS |
| `matched_entity_ids` references only S2/S3 IDs in the test set | ✅ checked with `--check-ids` |
| Every Source-1 entity appears exactly once | ✅ 1,732,544 of 1,732,544 |
| No duplicate IDs within any list | ✅ checked |
| **Model MIT/Apache-2.0 and ≤8B parameters** | ✅ LightGBM (MIT) + scikit-learn (BSD-3), **trained from scratch, no pretrained model ships** |
| **No external data lookup** | ✅ no API, database, geocoder or internet source is consulted anywhere; the pipeline reads only the provided TSVs |
| `country` not hardcoded to `{US, India}` | ✅ partition key only; France handled with no France-specific code |

On the embedding model: `multilingual-e5-small` (MIT, 117.7M parameters, confirmed from the
live model card) was **benchmarked and rejected** — worth +0.12% recall. It does not ship
and no GPU is used at any stage.

## Measured results

| | |
|---|---|
| Blocking recall @ K=50, production path, 300k train entities | **0.9809** |
| Matcher validation macro-F0.5 (held-out entities) | **0.9043** |
| Leaderboard F0.5 | **~0.90** |

Validation tracked the leaderboard to within ~0.005.

## Reproducibility

Artefacts shipped via Git LFS so the pipeline can be retrained without rebuilding:

| | size | contents |
|---|---|---|
| `work/paircache/*.npz` | 460 MB | 23.2M pre-featurised labelled pairs |
| `trainsplit/*.tsv` | 197 MB | 300k labelled train candidates at production K=50 |
| `models/reranker.pkl` | 2.2 KB | trained blocking reranker |

Total 657 MB — inside GitHub's 1 GB free LFS tier.

Not shipped: `dataset/` (organisers' data), `work/store/` (regenerable in 23s),
`output/*.tsv` (~1.2 GB, regenerable). See `docs/DATA.md`.
