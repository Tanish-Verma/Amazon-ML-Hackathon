# Business Entity Resolution — Amazon ML Challenge 2026

Matching business records across three independent, noisy sources — in English,
nine Indic scripts, and a country that never appears in the training data.

**Final leaderboard score: ~0.90 F0.5**

---

## The problem

Three companies each keep a list of businesses. Nobody agreed on spelling,
formatting, or even language, and there are no shared identifiers. These four rows
are the same shop:

```
Source 1 (reference)   Payne Enterprises      3315 Fremont Street, Peoria, IL
Source 2               Payne Énterprises      3315 FREMONT ST, PEORIA, IL
Source 2               PAYNE-ENRTPRMISES      3315 FREMONT SAINT, PEORIA, IL
Source 3               Payne Etrepndiels      3315 Fremont St, Peoria, Illinois
```

So are these two — where the name shares *no characters at all*:

```
Source 1   Raj Investments LLP             6(29), C.I.T. Colony, Mylapore, Chennai, Tamil Nadu
Source 2   ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி       6(29), C.I.T. COLONY, MYLAPORE, CHENNAI, Tamil Nadu
```

**The task:** for each of 1,732,544 Source-1 records, find all of its matches among
~10M Source-2/3 records. Many have none.

**Scored by macro-averaged F0.5** — precision counts double, and a false match on a
business that genuinely has no matches costs a full point. When unsure, predict nothing.

## Results

| | |
|---|---|
| **Final F0.5 (leaderboard)** | **~0.90** |
| Blocking recall @ K=50 | **0.9809** (0.9716 India · 0.9902 US) |
| Reduction ratio | 0.99999 — 4.1M candidates per query down to 50 |
| Predicted singleton rate | 6.57% (ground truth: 5.58%) |
| Mean matches per matched entity | 3.06 (ground truth: 3.67) |
| Official validator | **PASS** with `--check-ids` |

The slightly conservative singleton rate and match count are deliberate: under F0.5,
abstaining beats guessing.

## How it works

```
                 ┌──────────────────────── per country (US · India · France) ───┐
dataset/*.tsv    │                                                              │
      │          │  [1] NORMALISE      Unicode-category cleanup; preserves       │
      ▼          │                     Indic vowel marks, strips Latin           │
 [0] PREP ───────┤                     diacritics, extracts digit runs           │
 Parquet store   │                                                               │
 by country      │  [2] FOUR CHANNELS  name-trigrams │ address │ name-tokens │   │
                 │                     digit-keys — IDF-weighted sparse indexes  │
                 │                                                               │
                 │  ──── then stream query batches of 20,000 ────                │
                 │                                                               │
                 │  [3] RETRIEVE       top-1000 per channel (sp_matmul_topn)     │
                 │  [4] FUSE           reciprocal rank fusion → ~1000 pool       │
                 │  [5] RERANK         logistic regression, 16 cheap features    │
                 │  [6] CUT            top-50 → candidate_pairs.tsv              │
                 │                                                               │
                 │  [7] FEATURES       23 columns: + rank, margin, reverse rank  │
                 │  [8] MATCH          LightGBM, isotonic-calibrated             │
                 │  [9] DECIDE         threshold + globally exclusive assignment │
                 │                     → matching_results.tsv                   │
                 └───────────────────────────────────────────────────────────────┘
```

**Everything is CPU-only.** No GPU, no neural network, no embeddings — that was a
measured decision, not a constraint (see below).

## Repository layout

```
├── README.md                     ← you are here
├── Documentation_template.md     the submission methodology write-up
├── PLAN.md                       phase-by-phase plan, every decision with its evidence
│
├── code/business_entity_resolution/
│   ├── README.md                 architecture, folder guide, run instructions
│   ├── requirements.txt          pinned dependencies
│   ├── src/                      30 modules — see that README for the ★ ones
│   │   └── diagnostics/          the experiments behind every reported number
│   └── tests/                    11 test suites
│
├── reports/                      measurements, one per phase
├── docs/
│   ├── TASKS.md                  status board and onboarding
│   ├── DATA.md                   which artefacts ship, and how to rebuild the rest
│   ├── README-for-the-team.md    plain-English explainer
│   ├── SUBMISSION-CHECKLIST.md   final verification record
│   ├── phase-guides/             per-phase run guides
│   ├── handoffs/                 the prompts used to hand phases between people
│   └── challenge/                the original problem statement and guidelines
│
├── models/reranker.pkl           trained blocking reranker (2.2 KB)
├── trainsplit/                   300k labelled train candidates (Git LFS)
├── work/paircache/               23.2M pre-featurised labelled pairs (Git LFS)
├── output/                       candidate_pairs.tsv · matching_results.tsv (gitignored)
└── utils/validate_submission.py  organiser-provided validator
```

```bash
git lfs pull     # fetches trainsplit/ and work/paircache/
```

## Quick start

```bash
bash code/business_entity_resolution/src/bootstrap.sh ~/amlc
source ~/amlc/venv/bin/activate
cd code/business_entity_resolution

python -m src.cli --data-dir ../../dataset verify              # schema checks
python -m src.cli --data-dir ../../dataset --work-dir work prep # → parquet store
python -m pytest -q tests/                                     # test suite
```

Full run instructions: [`code/business_entity_resolution/README.md`](code/business_entity_resolution/README.md).

## Five things we learned the hard way

**1. The address channel solves transliteration — not embeddings.**
For all nine Indic scripts, name matching finds 0.2–1.0% of true matches. The union
with address matching finds 98–99%, because **street numbers survive a change of
script untouched**. A multilingual embedding channel was measured at **+0.12% recall**,
so the final system has no neural model and needs no GPU.

**2. Fusing ranked lists by summing scores cost 16 points of recall.**
Character-trigram cosines are an order of magnitude larger than address-token cosines,
so name candidates outranked and evicted address candidates — fatal for exactly the
transliterated records whose only route is the address. Reciprocal rank fusion is
scale-free and fixed it.

**3. "Reachability" is not "retrievability."**
We first measured that 99.88% of true pairs share *some* lexical signal and treated it
as the target. But sharing signal is not the same as ranking in the top 50 of 4M
records. The honest number is pool recall at a stated depth. Always quote the depth.

**4. Python objects don't survive `fork` at corpus scale.**
A per-record store of 40 GB of `frozenset`s plus forked workers exhausted a 125 GB
machine twice. CPython writes a refcount into every object a worker touches, so pages
are *copied*, not shared. Rewriting the same features as binary sparse matrices —
every set intersection becomes `(Q[qrows].multiply(C[crows])).sum(axis=1)` — brought
it to **0.68 GB** and removed the need to fork at all.

**5. Precision is worth ~7× recall here.**
At precision 0.9, four more points of blocking recall are worth about +0.008 F0.5;
seven more points of precision are worth about +0.055. Once blocking cleared ~98%,
every remaining hour belonged to the decision rule.

The single biggest precision lever was a structural fact in the data: **every
Source-2/3 record belongs to at most one Source-1 entity** — verified across all
7,638,365 ground-truth links. Enforcing that as a globally exclusive assignment
removes a whole class of false merges that no per-pair threshold can catch.
