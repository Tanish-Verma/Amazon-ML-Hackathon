# Phase 0 — Repo & environment audit

Date: 2026-09-25 · Machine: `cmslab` (2x Xeon Gold 6240R, 125 GB RAM, RTX A2000 12 GB)

## Verified dataset facts

| File | Rows | Countries | Empty address |
|---|---|---|---|
| train_source1 | 2,206,821 | India, US | 0 |
| train_source2 | 5,034,616 | India, US | 168,967 |
| train_source3 | 5,285,603 | India, US | 175,916 |
| test_source1 | **1,732,544** | France, India, US | 0 |
| test_source2 | 4,887,273 | France, India, US | 129,408 |
| test_source3 | 5,082,316 | France, India, US | 136,098 |

Ground truth: 2,206,821 entities · 123,247 singletons (5.58%) · 7,638,365 matched IDs.

Structural checks, all passing:

* No duplicate `entity_id` in any source; every ID carries the correct `S1-`/`S2-`/`S3-` prefix.
* Ground truth covers exactly the Source-1 ID set.
* **One-to-many holds:** 0 of 7,638,365 S2/S3 IDs are claimed by more than one S1 entity.
  The matching decision rule exploits this as an assignment constraint.
* Row counts are identical on the laptop and on `cmslab`, which is also the transfer-integrity check.

## Environment

* Python 3.11.9, isolated venv, pinned deps (`src/bootstrap.sh`, `requirements.txt`).
* torch 2.5.1+**cu121** — chosen to match the 535.x driver (CUDA 12.2) exactly rather than
  relying on CUDA minor-version compatibility. CUDA available, 12.44 GB VRAM free.
* Measured GPU throughput: **21.1 TFLOPS** fp16 dense matmul.

## Embedding model benchmark (task 0d)

20,000 real test rows (`name | address`, mean 84 chars), fp16, max_len 64:

| Model | Licence† | Params‡ | Dim | texts/sec | Full 20M pass |
|---|---|---|---|---|---|
| `intfloat/multilingual-e5-small` | MIT | 117.7M | 384 | 5,001 | **1.11 h** |
| `intfloat/multilingual-e5-base` | MIT | 278.0M | 768 | 1,951 | **2.85 h** |

† read from model-card metadata via the HF API, not from recollection.
‡ summed from the loaded weights.

Both satisfy the MIT/Apache-2.0 + ≤8B constraint with a wide margin.

**Recommendation:** `multilingual-e5-small` as the default. Its 384-dim output needs 7.7 GB to
store test embeddings versus 15 GB for `e5-base` at 768 dims, and disk is a tighter constraint
here than GPU hours. `e5-base` stays available as an upgrade if Phase 1 shows the transliteration
problem is large enough to justify the storage.

## Columnar store

15 country partitions, 864 MB zstd parquet, built in 22.6 s. The three France partitions were
created automatically from the data — partitioning is on whatever `country` label is present, with
no fixed set anywhere in the code.

## Baseline submission (rung R0)

All-singleton prediction for every test Source-1 entity. Passes
`utils/validate_submission.py --check-ids`: 1,732,544 rows, exact headers, tab-separated, UTF-8.
Expected score ≈ the singleton rate, 0.056. Its purpose is to prove the output path end-to-end.
