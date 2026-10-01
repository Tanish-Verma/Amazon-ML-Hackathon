# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

We resolve 1,732,544 Source-1 business records against ~10M Source-2/3 records using a
four-channel lexical retrieval stage, reciprocal rank fusion, a learned reranker, and a
precision-first LightGBM matcher with globally exclusive assignment. Blocking recall is
**0.9809** at 50 candidates per entity; the matcher achieves **0.9043 macro-F0.5** on a
held-out validation split, which tracked the leaderboard closely (**~0.90**).

Two decisions shaped the result. First, we measured that a multilingual embedding channel
was worth only **+0.12% recall** — because address digit runs survive transliteration
intact — so the final system is entirely CPU-based with no neural component. Second, we
verified that **every Source-2/3 record belongs to at most one Source-1 entity** across all
7,638,365 ground-truth links, and enforced that as a globally exclusive assignment, which
removes a class of false merges no per-pair threshold can reach.

---

## 2. Methodology

### 2.1 Problem Analysis

Measured from the training data (`reports/eda.md`):

| Property | Value | Consequence |
|---|---|---|
| Singleton rate | 5.58% | Each correct abstention is worth a full 1.0 |
| Mean matches per matched entity | 3.67 (max 11) | Candidate sets of 50 give ample headroom |
| **S2/S3 records matching nothing** | **26.6% / 25.4%** | ~2.7M distractors — precision is the hard problem |
| **Record claimed by >1 S1 entity** | **0 of 7,638,365** | The relation is strictly one-to-many |
| Cross-country matches | 0 of 188,806 groups | `country` is a safe partition key |
| Empty addresses | 3.3% of S2/S3 | Address alone cannot carry the system |

**Noise patterns.** Injected typos (`PAYNE-ENRTPRMISES`), spurious diacritics
(`Payne Énterprises`), token reordering (`Hendricks and Inc Flowers`), domain-style names
(`maurewilliamscolombier.com`), whole-name replacement recoverable only by address
(`Dréxkor`), and full transliteration into nine Indic scripts — Devanagari, Bengali,
Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam.

**The finding that determined the architecture.** We measured retrieval reachability per
channel:

| Channel | True matches reachable |
|---|---|
| Name only | 90.43% |
| Address only | 94.82% |
| **Either** | **99.88%** |
| Neither | 0.12% |

Broken down by script, the name channel reaches only **0.2–1.0%** for transliterated
records, while the union reaches **98–99%**. The reason is simple and exploitable:
**a street number reads identically in Devanagari, Tamil, French and Latin.** The address
channel — not an embedding model — solves transliteration. This is why the final pipeline
has no neural network.

**France.** 15% of test entities, absent from training. We profiled token document
frequency on the test split (a frequency count needs no labels): French `sarl` 28.3% and
`sas` 20.1% occupy the same band as US `llc` 26.9% and `inc` 18.0%. A DF-threshold rule
therefore handles French legal suffixes with no French-specific code — and also catches
French function words (`de`, `du`, `des`) that a hardcoded English stoplist would miss.

### 2.2 Solution Strategy

**Approach Type:** Multi-channel lexical blocking → rank fusion → learned reranking →
gradient-boosted matching with constrained assignment.

**Core Innovation:** Treating the one-to-many property of the data as a hard constraint on
the output rather than a property of individual pair scores. Combined with rank-based
fusion of channels whose score distributions are not comparable, this is what converted
high recall into high precision.

**Design rule applied throughout:** `country` is a *partition key*, never a feature. It is
never one-hot encoded and nothing branches on its value. Country-conditional information
enters only through per-partition document-frequency statistics, computed for whatever
labels appear in the data. This is what makes France work without French-specific code.

---

## 3. Candidate Generation (Blocking)

### Blocking keys used

Four independent channels, each an IDF-weighted sparse index over the Source-2/3 corpus of
one country, queried for the top 1000:

| Channel | Key | Solo recall@1000 | Role |
|---|---|---|---|
| `address` | digit runs + address word tokens | **92.64%** | The workhorse; recovers transliterations |
| `name_ngram` | character 3-grams of the name | 71.93% | Typo- and script-robust |
| `name_token` | whole name tokens | 67.80% | Catches word reordering |
| `digit_key` | address digit runs alone | 43.93% | Tiny, selective, ~0.1s — nearly free |

`digit_key` is separated from `address` deliberately: inside the combined channel, digit
features share L2 normalisation with dozens of street and city tokens and get diluted by
the unshared ones.

Queries retain only their **rarest 14–20 features**; this scored identically to using all
features while greatly reducing cost. Features present in more than 1% of the corpus are
dropped (0.2% for digit keys).

### Fusion and reranking

Channels are combined by **reciprocal rank fusion**, `Σ w/(20 + rank)`. This was a
correction, not an initial design: our first implementation summed raw cosine scores and
scored **16 points worse** than its own best single channel, because character-trigram
cosines are an order of magnitude larger than address-token cosines, so name candidates
systematically evicted address candidates during the final cut — catastrophic for exactly
the transliterated records whose only route is the address.

The ~1000-candidate pool is then reranked by a **pooled logistic regression** over 16
cheap pairwise features (set Jaccards, containment, Jaro-Winkler, token-set ratio, digit
overlap, retrieval-context features) and cut to the top 50.

The reranker is pooled across countries rather than fitted per country. This is required,
not preferred: France has no labels, so no France-specific model can exist. Measured cost
of pooling: −0.04 points on India, −0.33 on US.

### Candidate pairs generated

**86,626,700** pairs — 1,732,544 entities × 50 candidates (10 entities received none).

### How we ensured true matches were not lost

Measured by running the **production code path** on a 300,000-entity training split and
scoring against ground truth (`reports/phase3_production_accuracy.md`):

| | India | US | Overall |
|---|---|---|---|
| Pool recall @ depth 1000 | 98.8% | 99.3% | — |
| **Macro recall @ K=50** | **0.9716** | **0.9902** | **0.9809** |
| Reduction ratio | 0.999988 | 0.999992 | 0.99999 |

Only **541 of 300,000** entities (0.18%) lost every true match. These are cases where the
name is transliterated *and* the address is truncated beyond overlap.

This verification mattered: all earlier figures came from diagnostics reading cached
features, and the production path had since been refactored (candidate pools moved from
Python dicts to numpy arrays). Predicted 97.4%/99.3%, delivered 97.16%/99.02% — within
sampling noise.

**Depth was chosen by measurement.** Recall@50 against rerank depth: 1000 → 98.22%,
700 → 98.06%, 500 → 97.89%, 400 → 97.78%, 200 → 97.00% (India). Depth 700 saves 1.43× for
0.17% recall; depth 400 costs 0.43–0.64%. We kept 1000.

---

## 4. Matching Model

### Features used

The matcher scores the 16 features produced by `src/rerank_vec.py`, computed as vectorised
sparse-matrix operations:

**Name:** character-trigram Jaccard · token Jaccard · token containment · Jaro-Winkler ·
token-set ratio (reordering-invariant) · length ratio
**Address:** trigram Jaccard · token Jaccard · Jaro-Winkler · both-empty flag
**Digits:** digit-key Jaccard · shared-digit count
**Retrieval context:** RRF score · number of channels that hit · best channel rank ·
source indicator (S2 vs S3)

Class separation on training pairs, positives versus negatives:

| Feature | mean(pos) | mean(neg) | separation |
|---|---|---|---|
| `addr_tok_jac` | 0.6881 | 0.0700 | **+0.618** |
| `addr_tri_jac` | 0.6401 | 0.0557 | **+0.584** |
| `digit_jac` | 0.7010 | 0.1216 | **+0.579** |
| `name_tri_jac` | 0.5728 | 0.1661 | +0.407 |
| `best_channel_rank` | 11.2 | 486.2 | **−475** |
| `addr_both_empty` | 0.0391 | 0.0221 | +0.017 |

Address features separate best, consistent with the EDA. `best_channel_rank` is starkly
discriminative: true matches surface near the top of *some* channel, which is precisely
what RRF is built to exploit. `addr_both_empty` is near-dead and a candidate for removal.

A 23-column extension (`src/features.py`) adding forward rank, score margin, candidate
count, **reverse rank** — how this entity ranks among all entities competing for the same
record — and corpus-IDF-weighted address-component agreement is implemented and tested in
the repository, but the **submitted result used the 16-feature model**.

### Model type

**LightGBM** gradient-boosted trees with **isotonic calibration**, trained on an
entity-level split (never a pair-level split, which would leak a query's candidates across
both sides).

LightGBM is a tree ensemble trained from scratch on the provided training data only — a
few MB of tree structure, MIT licensed. The ≤8B-parameter and MIT/Apache-2.0 constraints
are satisfied with enormous margin. The pipeline ships **no pretrained model of any kind**:
scikit-learn (BSD-3) and LightGBM (MIT) are the only model dependencies, and no GPU is
used at any stage.

**Rejected alternative, with measurement.** A multilingual embedding channel
(`multilingual-e5-small`, MIT, 117.7M parameters — licence and parameter count confirmed
from the live model card, and benchmarked at 5,001 texts/sec on an RTX A2000) was worth
**+0.12% recall**. Under a precision-weighted metric that cannot repay the GPU hours or
the 7.7 GB of embedding storage. We dropped it.

### Threshold selection method

Direct macro-F0.5 optimisation on the validation split, tuning two parameters jointly:

* **probability threshold = 0.6**
* **per-entity margin = 0.2** — a candidate is accepted only if its calibrated score is
  within 0.2 of that entity's best, so one confident match does not drag in weak siblings

**Decision rule (the precision lever).** Not a flat cutoff:

1. Accept candidates above the calibrated threshold.
2. **Enforce globally exclusive assignment** — each Source-2/3 record goes to at most one
   Source-1 entity, with contention resolved by score. This is the one-to-many property of
   §2.1 applied as an output constraint.
3. Apply the per-entity relative margin.
4. **Abstain explicitly** — emit an empty list when nothing clears the threshold.
   Singletons are 5.58% of entities and each is worth a full point.

---

## 5. Results & Error Analysis

### Scores

| Metric | Value |
|---|---|
| **Validation macro-F0.5** (held-out entities) | **0.9043** |
| **Leaderboard F0.5** | **~0.90** |
| Blocking recall ceiling @ K=50 | 0.9809 |

Validation tracked the leaderboard to within ~0.005, which suggests the entity-level split
and full-corpus distractor density were realistic.

### Submission characteristics

| | Submission | Training ground truth |
|---|---|---|
| Rows | 1,732,544 | — |
| Predicted singletons | 113,807 (**6.57%**) | 5.58% |
| Mean matches per matched entity | **3.06** | 3.67 |
| Max matches | 14 | 11 |

The submission is deliberately **more conservative than the training distribution** — more
abstentions, fewer matches each. Under F0.5 that is the correct bias: a false merge on a
singleton costs a full point, while a missed match costs far less.

Official validator: **PASS** with `--check-ids` on both output files.

### Common false positives (wrong merges)

1. **Shared address, different business.** Office buildings and commercial complexes host
   many entities at one address. When the name is transliterated the name features are all
   ~0 *whether or not the match is real*, so the classifier is near-blind and leans on the
   address — which is identical for every tenant. The exclusive-assignment constraint
   mitigates this by forcing co-located candidates to compete rather than all being
   accepted.
2. **Near-identical chain names.** Branches of one chain in the same city differ only by a
   street number, so a mangled house number (`AF-684` vs `AF-0684`, which we canonicalise,
   versus `48` vs `4-8`, which we cannot) flips the decision.
3. **High-DF name tokens surviving the stoplist.** Generic names reduced to `limited`,
   `private` or `centre` after stoplisting produce spurious trigram overlap.

### Common false negatives (missed matches)

1. **Transliterated name plus truncated address** — the 0.18% that never enter the
   candidate pool at all. Unrecoverable by any downstream model.
2. **Exact-key misses on digits.** `48` and `4-8` produce disjoint keys under any exact
   canonicalisation. Documented limitation.
3. **Threshold abstentions.** With threshold 0.6 and margin 0.2 tuned for precision, some
   true matches scoring just below are deliberately dropped. This is the intended trade
   under F0.5.

### Per-country behaviour

India trails US by ~1.9 points of blocking recall throughout, which is structural rather
than a tuning failure: 13% of India's Source-2 names are non-Latin, where the name channel
contributes almost nothing and the address channel does nearly all the work. US and France
are entirely Latin, so all four channels contribute.

---

## 6. Conclusion

A carefully measured lexical pipeline beat the obvious neural approach on this problem: the
multilingual embedding channel we expected to need turned out to be worth 0.12% recall,
because street numbers are already script-invariant. The score came instead from two
places — correcting a rank-fusion bug worth 16 points of recall, and exploiting a
structural property of the data (each record belongs to one entity) as a hard output
constraint.

The broader lesson was about measurement discipline. Several of our confident intuitions
were wrong in measurable ways: fusing by raw score, pruning high-frequency features,
parallelising a memory-bandwidth-bound operation, and avoiding a "wasteful" duplicated
gather that turned out to be cheaper than the loop replacing it. Every one was caught by
measuring rather than reasoning, and the final system is simpler and faster for it.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` — all source under `src/`, 11 test suites under
`tests/`, pinned `requirements.txt`, and a `README.md` with full run instructions.

**Reproducing the outputs:**

```bash
bash src/bootstrap.sh ~/amlc && source ~/amlc/venv/bin/activate
cd code/business_entity_resolution

python -m src.cli --data-dir dataset verify                       # schema assertions
python -m src.cli --data-dir dataset --work-dir work prep          # → parquet store

# Blocking → candidate_pairs.tsv (one country at a time; ~4.2h all three concurrently)
python -m src.run_blocking --store work/store --split test --countries India \
  --model models/reranker.pkl --out output/cand_India.tsv --top-k 50 --batch 20000

# Matching → matching_results.tsv
python -m src.predict_phase5 --help
python -m src.phase8 --help          # assembles and audits the final TSVs
```

**Key modules:** `normalize.py` (script-agnostic normalisation) · `blocking.py` (the four
channels) · `rerank_vec.py` (vectorised sparse features) · `run_blocking.py` (streaming
production blocking) · `features.py` (23-column extension) · `model.py` (LightGBM +
calibration) · `decide.py` (exclusive assignment) · `phase8.py` (assembly and audit) ·
`scoring.py` (spec-exact macro-F0.5) · `diagnostics/` (every experiment behind the numbers).

### B. Additional Results

All measurements are preserved in `reports/`:

| Report | Contents |
|---|---|
| `eda.md` | Noise taxonomy, script census, per-country DF profiles, reachability by channel and script |
| `phase0.md` | Schema verification, environment, embedding-model benchmark with licences |
| `phase2.md` | Normalisation verified on all 24,229,173 records; zero crashes |
| `phase3_blocking.md` | The four defects found and what each cost; fusion comparison; depth and `max_df` sweeps |
| `phase3_production_accuracy.md` | Production-path recall against ground truth, and what the number does not say |
| `phase4_implementation.md` | 23-column feature extension and per-feature AUC methodology |

**Measured dead ends**, recorded so they are not re-tested: an embedding channel
(+0.12% recall) · LightGBM as *reranker* with a classification objective (54% @10 versus
94% for logistic regression) · tighter `max_df` (0.002 cost ~5 points versus 0.01) ·
shallower rerank pools (depth 400 costs 0.43–0.64%) · a per-query broadcast to avoid
duplicated sparse gathers (10% *slower* than one large vectorised gather) · multiprocessing
the retrieval matmul (1.45× only — memory-bandwidth bound, versus 4.9× from
`sparse_dot_topn`).

### C. Reproducibility notes

Developed on a 48-core Xeon with 125 GB RAM. CPU-only; no GPU required. Two artefacts ship
via Git LFS so the pipeline can be retrained without rebuilding from scratch:
`work/paircache/*.npz` (23.2M pre-featurised labelled pairs) and `trainsplit/*.tsv` (300k
labelled train candidates at production K=50).

The train candidates were generated with the **identical** configuration as the test
candidates — same reranker, same depth, same K. Configuration parity matters: a matcher
tuned on a different candidate distribution than it meets at inference is miscalibrated in
a way that appears on the leaderboard rather than in validation.
