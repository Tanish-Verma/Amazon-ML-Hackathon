# Business Entity Resolution — ML Challenge 2026

For each of **1,732,544** Source-1 business records, find every matching record
among ~10M Source-2 / Source-3 records, across **US, India and France** — where
France appears only in the test set and never in training.

Scored by **macro-averaged F0.5**: precision counts double, and a false match on
a true singleton costs a full point. When unsure, predict nothing.

---

## 1. What the pipeline does

```
dataset/*.tsv
     │
 [0] PREP ──────── one-time ──────── country-partitioned Parquet store
     │
     ▼  ══════════ per country (India / US / France) ══════════
     │
 [1] NORMALISE every record                         src/normalize.py
     │    Unicode-category cleanup (keeps Indic vowel marks), casefold,
     │    diacritic-strip Latin only, digit-run extraction
     │
 [2] BUILD 4 RETRIEVAL INDEXES over the S2+S3 corpus     src/blocking.py
     │    name-trigrams │ address │ name-tokens │ digit-keys
     │    each an IDF-weighted sparse matrix           ← paid once per country
     │
 [3] BUILD CORPUS RERANK STORE                      src/rerank_vec.py
     │    5 binary sparse matrices + 2 string lists (~4 GB)  ← paid once
     │
     ▼  ══════════ then stream query batches of 20,000 ══════════
     │
 [4] RETRIEVE top-1000 per channel   (sp_matmul_topn, 24 threads)
 [5] FUSE      reciprocal rank fusion → pool of ~1000 candidates
 [6] FEATURES  16 pairwise features, vectorised       src/rerank_vec.py
 [7] SCORE     pooled logistic regression (one matmul)
 [8] CUT       top-50 per query
 [9] WRITE     append rows to TSV, free the batch, next batch
```

**Why it streams:** 1.73M queries × 1000 candidates is ~1.7 **billion** pool
entries. Per-country fixed costs are paid once, then queries are walked in
batches and released, so peak memory depends on the *corpus*, not the query count.

## 2. Measured results

| stage | measurement |
|---|---|
| Blocking pool recall (depth 1000) | **98.8% India / 99.3% US** |
| After rerank, at K=50 | **97.4% India / 99.3% US** |
| Projected test-set recall @ K=50 | **~98.2%** |
| First (broken) implementation, for contrast | 71.0% |

Full evidence: `reports/phase3_blocking.md`.

## 3. Folder structure

```
code/business_entity_resolution/
├── README.md                  ← this file
├── requirements.txt           pinned dependencies
├── src/
│   ├── bootstrap.sh           creates the venv, installs pinned deps
│   ├── config.py              ★ ALL tuned constants, each with its evidence
│   ├── io_utils.py            streaming TSV readers, Parquet store, TSV writers
│   ├── normalize.py           Phase 2: script-agnostic text normalisation
│   ├── blocking.py            Phase 3: the 4 channels + sparse retrieval index
│   ├── candidates.py          per-country orchestration, parallel feature extraction
│   ├── eval_blocking.py       recall / reduction ratio / size distribution
│   ├── rerank.py              reference feature implementation + FEATURE_NAMES
│   ├── rerank_vec.py          ★ production vectorised features (sparse matrices)
│   ├── train_reranker.py      fits + saves the pooled logistic regression
│   ├── run_blocking.py        ★ production entry point → candidate_pairs.tsv
│   ├── scoring.py             macro-F0.5 metric, spec-exact
│   ├── eda.py                 Phase 1 exploratory analysis
│   ├── cli.py                 verify / prep / baseline / block subcommands
│   └── diagnostics/           experiments behind the reported numbers (re-runnable,
│                              not on the production path)
├── tests/
│   ├── test_scoring.py        pins F0.5 to the problem statement's worked example
│   ├── test_normalize.py      9 Indic scripts, French, real noisy records
│   ├── test_rerank_parity.py  ★ vectorised features == reference features
│   └── smoke_candidates.py    output-format validator for a candidate TSV
└── (repo root)
    ├── PLAN.md                the phase plan and every decision with reasoning
    ├── reports/               measurements per phase
    ├── docs/                  team explainer + per-phase handoff prompts
    ├── notebooks/             teammate's original Phase 2 notebook (provenance)
    ├── output/                matching_results.tsv + candidate_pairs.tsv
    └── utils/                 organiser-provided submission validator
```

★ = the files that matter most if you are reading this to understand the system.

## 4. Setup

```bash
bash src/bootstrap.sh ~/amlc          # venv + pinned deps
source ~/amlc/venv/bin/activate
```

Python 3.11. CPU-only; no GPU is needed anywhere in the final pipeline.

## 5. Running it

```bash
# 0. sanity-check the data and build the columnar store  (~40s total)
python -m src.cli --data-dir dataset verify
python -m src.cli --data-dir dataset --work-dir work prep

# 1. build labelled pair features from train  (retrieval + features, cached)
python -m src.diagnostics.diag_pooled --store work/store \
    --gt dataset/train/train_ground_truth.tsv --countries India US \
    --sample 4000 --cache-dir work/paircache

# 2. fit the pooled reranker
python -m src.train_reranker --cache-dir work/paircache --out work/reranker.pkl

# 3. produce candidates for a split, one country at a time
python -m src.run_blocking --store work/store --split test \
    --countries India --model work/reranker.pkl \
    --out work/cand_India.tsv --top-k 50 --workers 16 --batch 20000

# 4. validate any candidate TSV's format
python tests/smoke_candidates.py work/cand_India.tsv 50

# 5. validate the final submission
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

Countries are run separately on purpose: a failure costs one partition, and peak
memory stays bounded to one corpus.

## 6. Design decisions that matter

**Country is a partition key, never a feature.** Ground truth confirms no true
match crosses a country boundary, so partitioning on the literal label both
shrinks the search space and generalises to France for free. Nothing anywhere
branches on `{US, India}`.

**Matches are one-to-many.** Every S2/S3 record is claimed by at most one S1
entity — verified across all 7,638,365 ground-truth links. Phase 5's decision rule
exploits this as an assignment constraint.

**Stoplists are derived, not written.** Per-country token document frequency is
computed at runtime, so France's `sarl` (28.3%) and `sas` (20.1%) are caught by the
same mechanism as US `llc` (26.9%) and `inc` (18.0%) — with no French table. It
also catches French function words (`de`, `du`, `des`) that an English stoplist
would miss.

**The address channel solves transliteration, not embeddings.** For every Indic
script the name channel reaches 0.2–1.0%, but the union reaches 98–99%, because
street numbers survive a change of script. Measured worth of adding an embedding
channel: **0.12% recall** — which is why the final pipeline has no neural model
and needs no GPU.

**Precision, not recall, decides the score.** 26% of S2/S3 records match nothing
(~2.7M distractors) and 5.58% of S1 entities are true singletons. At precision
0.9, four extra points of blocking recall are worth ~+0.008 F0.5; seven points of
precision are worth ~+0.055.

## 7. Lessons paid for in wall-clock time

Recorded because they are easy to repeat:

* **Fusing channels by summing raw scores cost 16 points of recall.** Cosine
  magnitudes differ per channel, so name candidates evicted address candidates.
  RRF is scale-free and fixed it.
* **Per-record Python objects do not survive fork.** 40 GB of frozensets, and
  CPython's refcount writes meant workers *copied* rather than shared them —
  6 workers added 68 GB and exhausted a 125 GB machine. Binary sparse matrices
  (~4 GB, no per-element refcounts) removed the failure mode; lowering the worker
  count never could.
* **Tuning the wrong knob.** Every early sweep went *tighter* on `max_df` when the
  problem was that it was already too tight.
* **Reachability ≠ retrievability.** Phase 1's "99.88% ceiling" measured whether a
  pair shared any signal, not whether it ranks in the top-K of 4M records. Quote
  the pool ceiling at a stated depth instead.
