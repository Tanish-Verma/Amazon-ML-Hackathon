# Correction for Phases 4–9: actual candidate_pairs.tsv schema

Actual file: `source1_entity_id<TAB>candidate_entity_ids`; the second field is a comma-separated list of source-prefixed S2/S3 IDs, already in Phase 3 reranked order.

## Phase 4

`src.run_features` requires a **different**, enriched one-row-per-pair sidecar with exactly these six columns, in order:
`source1_entity_id, candidate_entity_id, rrf_score, n_channels_hit, best_channel_rank, reranker_score` (tab-separated). The two-column candidate file **cannot recover** RRF scores, channel counts, original best channel ranks or actual reranker scores. The candidate's list position is a rank, NOT a score. Do not fill missing metadata with fabricated values.

The existing `src.run_blocking_enriched` from the Phase 4 package can regenerate the sidecar while preserving the two-column official file; run only after coordinating with teammates and verifying resource capacity. The sidecar must represent the **same** candidate pairs as the official file. If a new run produces different pairs, use its own two-column output alongside its sidecar, never mix different runs. For Phase 4 TRAIN AUC, generate train candidates and train sidecar, not test candidates.

Inspect the existing file:

```bash
python -m src.candidate_format inspect --candidates ~/amlc/prod/candidate_pairs.tsv --expected 1732544
```

Only if you need an ID-only long-form file for inspection (this is NOT a feature matrix):

```bash
python -m src.candidate_format expand --candidates ~/amlc/prod/candidate_pairs.tsv --out ~/amlc/work/candidate_pairs_long_v1.tsv
```

After regenerating an exact enriched sidecar:

```bash
python -m src.candidate_format verify-sidecar \
  --candidates ~/amlc/work/phase4_candidates_v2.tsv \
  --enriched ~/amlc/work/phase4_pairs_v2.tsv \
  --db ~/amlc/work/phase4_verify_v2.sqlite
```

Then `src.run_features --pairs` accepts the **enriched** sidecar, never the original two-column file. See `README_PHASE4.md`.

## Phase 5

Train and predict from **Phase 4 Parquet feature chunks**, not `candidate_pairs.tsv`. Predictions should only contain candidate IDs that occurred in those feature chunks. Preserve exact source-prefixed IDs.

## Phase 6

Validation candidate recall must expand the second TSV field into a set of IDs per S1. Score final predicted match sets separately. Do not interpret a comma-separated candidate list as a single ID. The official test candidate file is unlabeled and cannot yield recall or F0.5 without corresponding truth.

## Phase 7

Use country-specific **labeled train** Phase 4 features for cross-country generalization; France test candidate lists are unlabeled, so distribution drift is measurable but France F0.5 is not.

## Phase 8

`src.phase8` already expects the **correct two-column candidate schema** (`source1_entity_id`, `candidate_entity_ids`) and checks that predicted matches are subsets. Use original completed country files or the original combined file; do not feed it the expanded long-form file or enriched metadata sidecar. Ensure the Phase 8 `--candidates` inputs have no duplicate S1 rows across files. Keep the official column names and comma separators exactly.

## Phase 9

Package the **original two-column** `candidate_pairs.tsv` and the separate two-column `matching_results.tsv`. Never submit the expanded long-form TSV, enriched metadata or Phase 4 Parquet in place of the official candidate file. Validate against the official competition specification before submission.

## Installation

Additive patch; no existing scripts or PLAN.md are overwritten. Extract at `~/amlc`, then:

```bash
source ~/amlc/venv/bin/activate
cd ~/amlc/code/business_entity_resolution
python -m pytest -q tests/test_candidate_format.py
```

The original Phase 4–9 ZIPs are **not** a verified end-to-end system; their earlier local unit tests cannot establish correct integration or submission validity. The adapter does not regenerate lost Phase 3 metadata.
