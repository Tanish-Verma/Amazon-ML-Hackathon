# PLAN.md — Business Entity Resolution (Amazon ML Challenge 2026)

**Status:** **Phases 0-1 complete (2026-09-25).** Awaiting sign-off for Phase 2.
Environment live on `cmslab`, schema verified, R0 submission passes the validator.
**Author:** lead ML engineer · **Reviewer/PO:** Tanish
**Written:** 2026-09-25

---

## 0. What I read before writing this

- `6ab5628d5a817_amazon_ml_challenge_problem_statement.pdf` (8 pp) and `README.md` (same content, fuller).
- `6ab56657b4f1a_guidelines_and_key_instructions_amazon_ml_challenge_2026.pdf` (2 pp).
- `Documentation_template.md`, `requirements.txt`, `utils/validate_submission.py` (read in full).
- All 7 dataset TSVs (schemas, row counts, country distributions, ground-truth statistics, sampled match groups).
- Existing untracked prior work: `run_pipeline.py`, `src/{normalize,blocking,evaluate_recall,generate_synthetic_data}.py`, `output/candidate_pairs.tsv`.

Everything below is grounded in measurements from those files, not assumptions. Measurements are
reproduced in §1 so you can sanity-check my reasoning before approving.

---

## 1. Ground facts established up front

### 1.1 Scale

| File | Rows |
|---|---|
| train_source1 | 2,206,821 |
| train_source2 | 5,034,616 |
| train_source3 | 5,285,603 |
| train_ground_truth | 2,206,821 |
| test_source1 | **1,732,544** (every one needs a row in the submission) |
| test_source2 | 4,887,273 |
| test_source3 | 5,082,316 |

~10.3M records to index per split. ~20M across both splits.

### 1.2 Country distribution

| | US | India | France |
|---|---|---|---|
| train S1 | 1,323,633 | 883,188 | — |
| test S1 | 663,106 | 809,986 | **259,452 (15.0%)** |
| test S2 | 1,871,330 | 2,312,565 | 703,378 |
| test S3 | 1,945,701 | 2,405,000 | 731,615 |

France is 15% of the scored entities. It cannot be treated as a rounding error.

### 1.3 Ground-truth structure — three findings that drive the design

1. **Singleton rate is only 5.58%** (123,247 of 2,206,821). Lower than I expected. Predicting
   "everything is a singleton" scores ~0.056, so recall genuinely matters; but each of those
   123k entities is worth a full 1.0 and a single false merge zeroes it.
2. **Each S2/S3 record is claimed by exactly one S1 entity.** I verified this across all
   7,638,365 matched IDs — multiplicity is 1 for every single one, no exceptions. The relation is
   strictly one-to-many. **This is the single biggest precision lever available**: the final
   decision can be posed as a constrained assignment (each S2/S3 record goes to at most one S1),
   which structurally eliminates a whole class of false merges that a per-pair threshold makes.
3. **Match-list sizes are tightly bounded.** Mean 3.67 per non-singleton, max 11 overall, and per
   source max 5 from S2 and max 6 from S3. Distribution peaks at 3. A candidate top-K of ~25-40
   is therefore far above what is needed — the current `output/candidate_pairs.tsv` (2.5 GB,
   ~90 candidates/entity) is roughly 3x wider than useful.

Also: 73% of S2/S3 records are matched to some S1; ~27% are pure distractors. Address is empty in
3.3% of S2/S3 records and never empty in S1.

### 1.4 Country is a safe partition key

Across 188,806 sampled ground-truth groups, **zero** groups contained a cross-country match.
Partitioning candidate generation by the literal `country` string is therefore safe and — because
it partitions on whatever label is present rather than on a fixed set — generalizes to France and
to any future label for free. This is a partition, not a rule table; it is the one piece of
country-awareness I want in the pipeline.

### 1.5 What the noise actually looks like

From sampled real match groups:

- Indic **transliteration** of whole names: `Raj Investments LLP` ↔ `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி`,
  `Ss Food Private Limited` ↔ `एसएस फूड प्राइवेट लिमिटेड`. Zero character overlap with the Latin form.
- **Injected typos**: `Maure Wilblims`, `Payne Enterpires`, `PAYNE-ENRTPRMISES`, `Orellana Invsmbens`.
- **Injected diacritics** on Latin text: `Payne Énterprises`, `Lumay Bóral`, `Dréxkor`.
- **Token drop / addition / reorder**: `Hendricks and Flowers Inc` ↔ `Hendricks and Inc Flowers`.
- **Domain-style names**: `maurewilliamscolombier.com`.
- **Total name replacement**, recoverable only via address: S1 `Maure Williams Colombier Inc` ↔
  S3 `Dréxkor` at `85 Wanye Avenue, Ticonderoga Townshiip, New York`.
- **Address**: component reordering, `St`/`Street`/`SAINT`, state name ↔ 2-letter code ↔ regional
  script (`Tamil Nadu` / `TN` / `தமிழ்நாடு`), house-number mangling (`AF-684` / `AF-0684`,
  `1056` / `1056c` / `1056-1060`), and empty.

**Consequence:** name and address must be *independent* candidate channels whose union is taken —
a name-only pipeline loses the `Dréxkor` case, an address-only pipeline loses every empty-address
match. Neither channel alone has a recall ceiling worth having.

### 1.6 France, inspected directly

I read France test records. It is Latin script, structurally identical to US: `175 Boulevard du
Président Franklin Roosevelt, Bordeaux, Nouvelle-Aquitaine`. The three concrete risks:

1. **Unseen legal suffixes** — `SARL`, `SAS`, `SASU`, `EURL`, `SCI`, `SA`, `S.A.S`, `E.U.R.L.`.
   These are very high frequency; if they are not treated as suffixes they act as spurious
   name-overlap signal between unrelated French businesses.
2. **Unseen address abbreviations** — `R.`/`RUE`, `AV`/`AVENUE`, `BD`/`Boulevard`, `ALLÉE`, `IMPASSE`, `CHEMIN`.
3. **Region ↔ department alternation** — `Hauts-de-France` ↔ `Nord`, `Nouvelle-Aquitaine` ↔ `Gironde`.
   Structurally the same problem as `Illinois` ↔ `IL` and `Tamil Nadu` ↔ `TN`.

**Design response (this is the core principle of the whole plan):** every one of those three is
solved by *deriving* the stoplists and alias sets from corpus statistics computed at runtime on
whatever countries are present, rather than from a hand-written table. High-document-frequency
tokens within a country partition are down-weighted automatically by TF-IDF; region/department
aliases can be mined from co-occurrence within the provided data alone. Hand-written dictionaries
stay as a supplementary signal only, never as the gate. This is both the generalization strategy
and the answer to "don't build per-country rule tables".

### 1.7 Compute — university Slurm box `cmslab` is primary (decided 2026-09-25)

**AWS is dropped.** The GPU quota request was rejected, and the available free-tier instances
(`t3`/`t8i` micro/small, `m7i-flex.large`) top out at 2 vCPU / 8 GB — *worse than the laptop* for a
10M-record workload. There is no version of this pipeline that runs there.

**`cmslab` is a much better fit than what AWS was going to give us:**

```
2x Intel Xeon Gold 6240R  | 48 physical cores / 96 threads | AVX-512 + avx512_vnni
125 GB RAM (120 GB free)  | NVIDIA RTX A2000, 12 GB VRAM   | CUDA driver 535.309.01 (CUDA 12.2)
/home  222 GB free on vol | (/scratch root-owned, unusable) | Slurm 21.08, partition LocalQ, no time limit
```

**Access is live and verified (2026-09-25).** `10.2.4.21:22` is reachable directly from the laptop
(7.4 ms, no VPN or jump host), and **key-based auth already works** — no `ssh-copy-id` needed, so
there is no setup action outstanding on your side. I can drive the box non-interactively, which
keeps iteration fast. Also verified: **outbound internet works** (PyPI and huggingface.co both
return 200, no proxy), so pinned dependency installs and open-weight model downloads will work
directly on the box. Toolchain present: Python 3.11.9, gcc 11.4, git, rsync.

**The headline win is RAM and cores, not the GPU.** 125 GB against 7 GB removes the constraint that
was distorting the whole design: indexes can live in memory, no disk-backed streaming, no
per-partition gymnastics. 48 cores turns the lexical blocking pass — the part that actually
determines our score — from an overnight job into minutes. Rungs R1 and R2, the realistic target,
become comfortable rather than heroic.

**Five operational facts I will design around, each worth getting right the first time:**

1. **Storage: everything stays inside `/home/co24btech11023/`.** `/scratch` is root-owned and
   unwritable (verified). Per your instruction I will **not** use `/tmp`, `/dev/shm`, or anywhere
   else on the shared system — the project root is `/home/co24btech11023/amlc/` and nothing is
   written outside it. Space is treated as **limited** regardless of what the volume reports.

   | Artefact | Est. size | Notes |
   |---|---|---|
   | Dataset, decompressed | 2.4 GB | streamed in, no `.gz` ever lands |
   | Columnar store (zstd parquet) | ~2 GB | |
   | Test embeddings (10M x 384 fp16) | 7.7 GB | rung R3 only |
   | Train embeddings (subset) | ~5-10 GB | rung R3 only |
   | Candidate files (train val + test) | ~1.5 GB | |
   | Pair features (~52M pairs x 40 feats) | 4-8 GB | fp16 where safe |
   | HF model cache + checkpoints | ~2 GB | rung R3 only |
   | **Total, rungs R0-R2** | **~12-16 GB** | the baseline is cheap |
   | **Total, with R3** | **~35-40 GB** | |

   Two consequences I am designing around:

   - **I never delete anything on `cmslab`. You do.** So disk usage only ever grows during a
     session, which means I must avoid redundant copies rather than clean them up afterwards.
     Concretely: the dataset transfers by **stream-compressing over the wire**
     (`gzip -c | ssh … 'gunzip -c > dest'`), so neither end ever stores a `.gz` alongside the
     unpacked file. Same discipline for every intermediate.
   - **Disk is checked before every bulk write** and the job fails loudly rather than filling the
     volume. If we do get tight, I report what is consuming space and ask you to delete — I will
     not propose running the deletion myself.

2. **CUDA 12.2 driver.** Our local venv has `torch 2.14.0+cu130`, which **will not run there** — it
   needs a much newer driver. The server needs a CUDA 12.x-targeted build. CUDA minor-version
   compatibility should let a cu124/cu126 wheel run on a 535 driver, but that is exactly the kind of
   claim worth testing rather than trusting, so Phase 0 verifies it empirically before anything
   depends on it.
3. **GPU access via Slurm, but no GRES configured.** Access to the box is Slurm-gated, and you've
   established that holding 60+ cores for 4-5 hours is uncontroversial there — so sustained heavy
   use is not a concern and I'll stop treating it as one. One practical implementation detail:
   `scontrol` reports `Gres=(null)`, so **`--gres=gpu:1` would be rejected** — the A2000 is visible
   from inside an ordinary allocation without it. Since the GPU is therefore not tracked as a
   countable resource, **every embedding pass still checkpoints**, so any interruption costs
   minutes rather than the whole run. Usable VRAM is ~11.9 GB (the display is on the same card).

4. **Slurm sees 48 CPUs** (`CPUTot=48`, `ThreadsPerCore=1`) though the box has 96 threads.
   `srun --cpus-per-task=48 --mem=120G` gets the full machine. Worth going through Slurm even though
   the GPU isn't gated by it — it is a single shared node and the queue is what stops us being killed.
5. **Python 3.11.9 at `/usr/local/bin/python3`.** I will build an isolated venv under the project
   root and not touch anything system-wide.
**Revised embedding-model recommendation — reinstating my original downscale.** I withdrew it when
a 24 GB A10G looked likely; the A2000 is a different proposition. It is compute-comparable to the
laptop's 4060 (same fp16 class, ~1/3 of an A10G), so the argument that killed the large models on
the laptop applies again here. The 12 GB of VRAM does help — it holds the largest country partition
for exact top-K with room to spare (India S3 at 2.4M x 1024 dims fp16 is 4.9 GB) — but VRAM was
never the bottleneck; throughput was.

| Model | Params | Licence | Role |
|---|---|---|---|
| `multilingual-e5-small` | ~118M | MIT | **Default** for full-corpus blocking |
| `multilingual-e5-base` | ~278M | MIT | Stretch, if the Phase 0 benchmark allows |
| `multilingual-e5-large` / `Qwen3-Embedding-0.6B` | ~560M / ~600M | MIT / Apache-2.0 | Reranker over the *capped* candidate set only (~40x less work), not full corpus |

Parameter counts and licences above are from memory and are **not** the basis for the final
decision — Phase 0 verifies each against the live model card, as you required.

**Fair play:** `cmslab` is compute only. No external data source, API or lookup service is involved,
so this is nowhere near the external-lookup prohibition. Challenge data stays inside your own home directory.

### 1.8 Time budget

Challenge window closes **27 Sep 2026, 23:59 IST**. At time of writing that is **~57 hours**.
Max 5 leaderboard submissions per day.

The AWS quota gamble is off the table, which removes the schedule's main unknown: `cmslab` is
available now, so there is no queue to wait on and no fallback that depends on someone else's
approval. The remaining risks are ordinary engineering ones.

---

## 2. Target repository layout (set up in Phase 0, so packaging is never a scramble)

```
.
├── output/
│   ├── candidate_pairs.tsv          # blocking stage output (final candidate set)
│   └── matching_results.tsv         # scored submission
├── code/business_entity_resolution/
│   ├── src/
│   │   ├── io_utils.py              # streaming TSV read/write, columnar store, partitioning
│   │   ├── normalize.py             # script-agnostic normalization
│   │   ├── stats.py                 # corpus-derived stoplists / IDF / alias mining
│   │   ├── blocking_lexical.py      # char-ngram TF-IDF + address-key channels
│   │   ├── embed.py                 # embedding model wrapper, memmap writer
│   │   ├── blocking_ann.py          # GPU exact top-K over embedding partitions
│   │   ├── candidates.py            # channel fusion, top-K cap, candidate_pairs.tsv writer
│   │   ├── features.py              # pairwise feature computation
│   │   ├── model.py                 # LightGBM train / calibrate / predict
│   │   ├── decide.py                # score -> final match set (assignment + thresholds)
│   │   ├── scoring.py               # macro F0.5 scorer (spec-exact)
│   │   ├── split.py                 # validation + unseen-country splits
│   │   └── cli.py                   # subcommands: prep, block, features, train, match, score
│   ├── README.md
│   └── requirements.txt
├── PLAN.md
├── Documentation_template.md        # filled in at Phase 9
├── utils/validate_submission.py     # provided, unmodified
└── dataset/
```

Two independently runnable entry points, as you required:

```bash
python -m src.cli block  --split test   # -> output/candidate_pairs.tsv   (recall ceiling)
python -m src.cli match  --split test   # -> output/matching_results.tsv  (consumes the above)
```

`candidate_pairs.tsv` is written as the *final* candidate set — after top-K capping, exactly what
the classifier scores — per the problem statement's explicit definition.

---

## 3. Fallback ladder (risk management against the ~57-hour clock)

Each rung is a complete, validator-passing submission. We never end up with nothing.

| Rung | Approach | Runs on | Est. | Purpose |
|---|---|---|---|---|
| **R0** | All singletons (empty everywhere) | laptop | minutes | Proves format end-to-end. F0.5 ≈ 0.056. |
| **R1** | Lexical blocking + tuned similarity threshold | cmslab CPU | Phases 0-3 | Floor score, no ML. |
| **R2** | Lexical blocking + LightGBM + assignment constraint | cmslab CPU | Phases 0-5 | **The realistic target.** |
| **R3** | R2 + embedding channel for transliteration recall | cmslab GPU | + Phase 3 ch. D | Upside. |

R0-R2 need no GPU, so a contended or claimed A2000 cannot cost us a submission — at worst it costs
the transliteration recall that Phase 1 will already have quantified. I will reach R1 before
investing in embeddings, so embedding work is a strict upgrade on a working baseline rather than a
prerequisite for having one.

---

## 4. Phases

Each phase lists tasks, the files it touches, and its done criterion. I stop after each and wait.

---

### Phase 0 — Repo & environment audit

**Tasks**

*`cmslab` bootstrap (runs first)*

0a. ~~SSH access~~ — **done**, verified working (§1.7).
0b. Create `/home/co24btech11023/amlc/` as the project root and transfer the dataset by
    **stream-compressing over the wire**, so no `.gz` copy is ever left behind on either end.
0c. Build an isolated venv on Python 3.11.9 under the project root; **verify the torch/CUDA pairing
    actually works on the 535 driver** before anything depends on it (§1.7 fact 2). If GPU torch
    proves awkward, R0-R2 need only CPU torch or none at all — this must not block.
0d. **Benchmark embedding throughput on the A2000** (e5-small vs e5-base, fp16, realistic sequence
    lengths) and extrapolate to 20M records. This number, not my estimate, decides the model.
0e. Commit a bootstrap script to `src/` so the environment is reproducible and Phase 9 has it documented.
0f. Establish a disk watchdog: assert free space before each bulk write, fail loudly rather than
    filling the volume, and surface a "what's using space" report to you when it gets tight.
    **No deletion on `cmslab` happens without you doing it.**

*Repo and data plumbing*

1. Create the `code/business_entity_resolution/{src,README.md,requirements.txt}` skeleton per §2.
2. Migrate the useful parts of the existing `src/` (the normalization ideas in `normalize.py` are
   a reasonable starting point; `blocking.py`'s in-RAM index is not viable at 7 GB and will be
   replaced). Retire `run_pipeline.py` and `generate_synthetic_data.py` — the real data is here.
3. Delete the stale 2.5 GB `output/candidate_pairs.tsv` (superseded; it is ~3x wider than needed
   and was produced by the index that exhausted memory). **I will confirm with you before deleting.**
4. Build `io_utils.py`: streaming TSV reader, and a one-pass converter to a compact on-disk
   columnar store partitioned by `(split, source, country)`. Partition list is discovered from the
   data, never hardcoded.
5. Verify every schema assertion mechanically: exact column names/order, `entity_id` prefix
   integrity, uniqueness, `test_source1` ID set == the validator's `required` set.
6. Pin `requirements.txt` to what we actually import (the current root one is a full-environment
   dump including TensorFlow/CUDA, unusable as a reproducibility artefact).
7. Write the R0 all-singletons submission and run `utils/validate_submission.py --check-ids` on it.

**Files:** `code/business_entity_resolution/src/{io_utils.py,cli.py}`, `requirements.txt`, `output/`

**Done when:** the columnar store exists for all 6 source files, schema assertions pass, and the
R0 submission returns `PASS` from the validator. We know the plumbing is correct before any ML.

**RESULT — all done criteria met.** Store: 15 country partitions, 864 MB, 22.6s. Verify: PASS on
all 7 files, identical row counts on `cmslab` and laptop (confirms transfer integrity); one-to-many
property re-confirmed mechanically (0 multi-claimed IDs). R0: `PASS` with `--check-ids`, 1,732,544
rows. Env: torch 2.5.1+cu121 on driver 535, CUDA available, 12.44 GB VRAM, 21.1 TFLOPS fp16.
Total footprint on `cmslab`: 8.8 GB (5.6 GB of that is the venv, mostly torch).

**Embedding benchmark (task 0d), measured on the A2000 over 20k real test rows (mean 84 chars):**

| Model | Licence (from model card) | Params (counted) | Dim | texts/sec | 20M records |
|---|---|---|---|---|---|
| `intfloat/multilingual-e5-small` | MIT | 117.7M | 384 | 5,001 | **1.11 h** |
| `intfloat/multilingual-e5-base` | MIT | 278.0M | 768 | 1,951 | **2.85 h** |

Both satisfy the MIT/Apache + ≤8B constraint with enormous margin. Licence read from model-card
metadata and parameter counts summed from the loaded weights — neither is from memory.

**Open questions for you are collected in §5 — I'd like answers before/with sign-off.**

---

### Phase 1 — EDA

**Tasks**
1. Reproduce and extend §1.3 on the full ground truth (singleton rate, cardinality, per-source caps).
2. **Quantify the transliteration problem**, which decides how much embeddings are worth:
   for every true match, classify the name pair as {same-script-Latin, Latin↔Indic, other} and
   measure, per class, whether the *address* channel alone could recover it. The number that
   matters: *% of true matches that are non-Latin-name AND empty-or-unmatchable-address*. If that
   is ~0%, embeddings are an optimization; if it is several %, they are essential. I will report
   this number explicitly before we commit GPU hours.
3. Character-script census of names and addresses per country per source.
4. Token document-frequency profiles per country partition — the empirical basis for the derived
   stoplists in §1.6. Confirm that `sarl`/`sas`/`eurl` really do sit in the same DF band for
   France as `ltd`/`llc`/`inc` do for US/India (this is the test of whether the derived approach
   actually replaces the hand-written table).
5. Address anatomy: numeric-token presence and discriminativeness, since street/door numbers look
   like the strongest script-invariant signal available.
6. Distractor analysis: characterize the 27% of S2/S3 records matching nothing.

**Files:** `src/stats.py`, `notebooks/` or `reports/eda.md` (written output, not a live notebook)

**Done when:** `reports/eda.md` exists with the numbers above, and specifically answers: *how much
recall can a purely lexical pipeline reach?* That answer determines whether the embedding channel
is on the critical path or optional.

**RESULT — answered decisively. Lexical ceiling is 99.88%; only 0.12% of true matches are
unreachable lexically.** Name alone 90.43%, address alone 94.82%. Transliterated Indic names are
reachable 0.2-1.0% by name but 98-99% via address, so **the address channel solves transliteration,
not embeddings**. Distractors are 26.6% (S2) / 25.4% (S3) — precision, not recall, is where the
score is decided.

**France generalisation confirmed by measurement, not reasoning.** DF profiled on the test split
(a frequency count needs no labels): France `sarl` 28.3% / `sas` 20.1% sit in the same band as US
`llc` 26.9% / `inc` 18.0%. The same mechanism also catches French function words (`de`, `du`,
`des`) that a hardcoded English stoplist would have missed. No French-specific table needed.

**A bug was found and fixed in the EDA itself** — `[^\w\s]` shatters Indic scripts because their
vowel signs are categories Mn/Mc. Corrected to strip by Unicode category. The headline moved only
99.86% → 99.88% (the bug degraded the name channel only, so the original was a lower bound).
See `reports/eda.md` §7 — it is the worked example Phase 2's tokeniser must not repeat.

**PROPOSED PLAN CHANGE (needs your approval):** drop the embedding channel from Phase 3 blocking —
0.12% recall cannot repay the GPU hours and storage. Keep embeddings only as a candidate *matching
feature* in Phase 4, where the corrected per-script numbers show the classifier is blind for
transliterated records. Deferrable: it changes the feature matrix only, leaving
`candidate_pairs.tsv` untouched. See `reports/eda.md` §6.

---

### Phase 2 — Normalization

**Tasks**
1. Unicode NFKC → case fold → diacritic strip (NFKD + combining-mark removal), applied to Latin
   and Latin-adjacent ranges; non-Latin scripts passed through unharmed (the existing
   `_strip_diacritics_safe` has the right instinct here and I'll keep that idea).
2. Script-agnostic tokenization (Unicode word boundaries, not `[a-z]+`).
3. **Derived** legal-suffix and address-stopword handling: compute per-country token DF at runtime;
   tokens above a DF percentile are down-weighted rather than deleted. A small hand-written
   suffix list stays as a *supplementary* signal with a feature flag, and Phase 7 measures whether
   removing it changes unseen-country recall — that is the honest test of whether we have
   accidentally built a rule table.
4. Numeric-token extraction and canonicalization (`AF-0684` → `af684`, `1056-1060` → `{1056,1060}`),
   since these are the highest-signal, most script-invariant address features.
5. Name variants produced: `clean`, `tokens`, `core_tokens`, `joined` (de-spaced, for domain-style
   names), `char_ngrams`.
6. Unit tests over the real noisy examples in §1.5 — every one of them must normalize to something
   that makes its true match reachable.

**Files:** `src/normalize.py`, `src/stats.py`, `tests/test_normalize.py`

**Done when:** tests pass on the §1.5 examples plus a France set (SARL/R./BD), and normalization
throughput is measured (must sustain ≥100k records/sec/core or it becomes the bottleneck at 20M).

---

### Phase 3 — Blocking / candidate generation

The recall ceiling. Per §1.5 this must be a **union of independent channels**, computed per
country partition.

**Tasks**
1. **Channel A — name char-n-gram TF-IDF.** 3-5 char n-grams over the normalized name, IDF
   weighted per country partition, top-K by cosine via a sparse inverted index with a per-token
   posting cap. Character n-grams are inherently script-agnostic and typo-robust — this is what
   catches `Payne Enterpires` and `maurewilliamscolombier.com`.
2. **Channel B — address keys.** Blocking keys built from canonical numeric tokens x rare address
   word tokens. Targets the `Dréxkor` class (name useless, address decisive).
3. **Channel C — name token-set.** Cheap exact/near token overlap, catches reordering.
4. **Channel D — embedding ANN.** A second pass *within this phase*, gated on the GPU box.
   The channel interface is designed now so it drops in additively, without reworking fusion.
5. **Fusion & cap:** union channels, score each candidate by a cheap combined signal, keep top-K
   per S1 entity. K tuned on the recall/cost curve; §1.3 says true lists max at 11, so K in
   [25, 40] is the region to explore. **This capped set is what `candidate_pairs.tsv` contains.**
6. **Measure and report: macro recall, micro recall, reduction ratio, candidate-size distribution,
   and a sample of missed true matches with diagnosis.** Per-country breakdown mandatory.
7. Memory discipline: every step streams; peak RSS asserted under 4 GB in CI-style check.

**Files:** `src/{blocking_lexical.py,candidates.py}`, `src/cli.py`

**Done when:** `python -m src.cli block --split train-val` produces a capped candidate set with
measured macro recall (target ≥0.95 lexical-only, to be revised by the Phase 1 transliteration
number), reduction ratio reported, peak RSS under 4 GB, and full-test runtime extrapolated and
confirmed to fit the budget.

---

### Phase 4 — Feature engineering

**Tasks**
1. Name: char-n-gram cosine, token Jaccard, token-set containment, Levenshtein ratio,
   Jaro-Winkler, longest-common-subsequence ratio, ordered-vs-unordered delta (detects
   transposition), length ratio, glued-string similarity.
2. Address: token Jaccard, numeric-token exact/partial overlap (high value per §1.5), rare-token
   overlap, IDF-weighted cosine, `address_is_empty` flag (3.3% of S2/S3 — must be an explicit
   feature, not an imputed zero).
3. Cross-field: name-vs-address leakage (catches name components appearing in the address).
4. Contextual/rank features — these matter a lot for precision and are easy to miss:
   rank of this candidate within its S1 entity's list, score margin to the best candidate, ratio
   to the best score, number of candidates for that entity, and the **reverse-direction rank**
   (how this S1 ranks among all S1s competing for this S2/S3 record). The last one operationalizes
   the one-to-many structure from §1.3 at the feature level.
5. Source indicator (S2 vs S3) — legitimate, it is in the ID prefix, not a country rule.
6. `country` handled as a **partition context only**, never as a one-hot feature. This is a
   deliberate choice: one-hotting country is exactly what breaks on France. Country-conditional
   information enters only via per-partition IDF statistics, which are computed for whatever
   country is present.
7. Feature computation streams over `candidate_pairs.tsv`; output to disk-backed arrays.

**Files:** `src/features.py`, `tests/test_features.py`

**Done when:** features computed for the validation split, no NaN/inf, per-feature distributions
and label separation (AUC per single feature) reported, and throughput confirmed to fit the budget.

---

### Phase 5 — Matching model & decision rule

**Tasks**
1. Labels: a candidate pair is positive iff it is in that S1 entity's ground-truth list.
   Class balance measured (with K≈30, expect ~10% positive).
2. **Model: LightGBM binary classifier.** Justification for the methodology doc: this is a
   gradient-boosted decision tree over engineered similarity features, not a pretrained
   foundation model — it is trained from scratch on the provided training data only, ships as a
   few MB of tree structure, and is MIT licensed. The ≤8B / MIT-Apache constraint is satisfied
   trivially by it and applies substantively to the *embedding* model chosen in Phase 6.
3. Probability calibration (isotonic) so thresholds are interpretable.
4. **Decision rule — the precision lever.** Not a flat pairwise cutoff:
   - a. Apply a high-precision threshold `t` tuned directly for macro-F0.5.
   - b. **Enforce the one-to-many constraint from §1.3:** each S2/S3 record may be assigned to at
     most one S1 entity. Resolve contention by score (greedy by descending confidence, with a
     margin requirement before stealing). This removes false merges that no per-pair threshold can.
   - c. Per-entity relative rule: accept candidates within a margin of the entity's best score,
     so a confident single match does not drag in mediocre siblings.
   - d. Use the observed per-source cardinality caps (≤5 S2, ≤6 S3) as a **soft prior / tie-break
     only**, not a hard rule — they are estimated from train and I do not want to hardcode a bound
     that may differ in test.
   - e. Explicit singleton decision: if nothing clears `t`, emit empty. Since singletons are 5.6%
     and worth 1.0 each, tune the abstain threshold against macro-F0.5 directly.
5. Threshold/margin grid searched against the Phase 6 scorer, per-country reported to confirm we
   are not tuning a threshold that only works for one country.

**Files:** `src/{model.py,decide.py}`, `src/cli.py`

**Done when:** validation macro-F0.5 reported overall and per country, with an ablation table
showing the contribution of the assignment constraint and of each decision rule component.

---

### Phase 6 — Local validation harness

**Proposed change to your ordering — flagging it explicitly rather than doing it silently:** the
F0.5 scorer is needed to evaluate Phases 3-5, so I want to build `src/scoring.py` and
`src/split.py` *during Phase 0/1* and keep Phase 6 as the place where the harness is hardened and
the stress-test infrastructure is completed. Nothing else about the ordering changes. Please
confirm.

**Tasks**
1. `scoring.py` implementing macro-F0.5 exactly as specified, including every edge case:
   true-singleton + predicted-empty → 1.0; true-singleton + any prediction → 0.0; true non-empty +
   predicted empty → 0.0; standard formula otherwise.
2. **Sanity-check against the problem statement's worked example** (predicted 3, true 2,
   P=2/3, R=1.0 → **0.714**) as a unit test. Also assert the all-singletons baseline reproduces
   the 5.58% singleton rate as its score.
3. Validation splits: a stratified held-out slice of train S1 entities, where the S2/S3 corpus is
   the **full** train S2/S3 (not a subsample) so distractor density and therefore precision are
   realistic. A subsampled corpus would flatter every precision number we produce.
4. A fast dev split (~50k S1 entities) for iteration and a larger confirm split (~300k) for
   decisions.
5. Bootstrap confidence intervals on the macro-F0.5, so we can tell a real improvement from noise
   before spending one of 5 daily submissions on it.

**Files:** `src/{scoring.py,split.py}`, `tests/test_scoring.py`

**Done when:** the worked example passes to 3 decimals, splits are reproducible from a seed, and
the harness scores a full validation run end-to-end.

---

### Phase 7 — Generalization / unseen-country stress test

**Tasks**
1. **Primary test — hide India:** fit everything country-dependent (IDF stats, thresholds,
   classifier) on **US only**, evaluate on **India only**. Hardest version: unseen script.
2. **Secondary test — hide US:** fit on **India only**, evaluate on **US only**. This is the
   closer analogue to France: Latin script, but unseen legal suffixes and unseen state
   abbreviations. I want both, because France's actual failure mode is the second one.
3. Report the recall/precision/F0.5 delta vs the in-distribution number. A large collapse means
   hidden hardcoding, and I will hunt it down before touching test data.
4. **Ablate the hand-written dictionaries** (suffix list, abbreviation map) under the unseen-country
   condition. If removing them barely moves the score, the derived statistics are carrying the
   load and we are genuinely generalizing. If the score collapses, we have built a rule table with
   extra steps and I will say so.
5. Grep-level audit for literal `"US"` / `"India"` / `"France"` anywhere in the pipeline; the only
   permitted use of a country value is as an opaque partition key.
6. France-specific smoke test on test data *without labels*: candidate-set size distributions,
   score distributions, and predicted singleton rate for France vs US vs India. A France singleton
   rate wildly out of line with the others is the alarm bell that we can detect without labels.

**Files:** `src/split.py`, `reports/generalization.md`

**Done when:** both stress tests are reported with deltas, the dictionary ablation is reported, the
audit is clean, and the France distribution check looks sane. **This gates touching the test set.**

---

### Phase 8 — Inference & packaging

**Tasks**
1. Full-test run: prep → block → features → score → decide, per country partition, streaming.
2. Write `output/candidate_pairs.tsv` and `output/matching_results.tsv`.
3. **Explicitly assert every validator rule before running the validator:** exactly 1,732,544 rows
   in each file, one row per test S1 entity, no duplicate S1 rows, no duplicate IDs within a list,
   S2-/S3- prefixes only, matched ⊆ candidates, UTF-8, tab-separated, exact headers.
4. Run `python3 utils/validate_submission.py -m ... -c ... -t dataset/test --check-ids`.
   Note: `--check-ids` loads all S2/S3 IDs into memory — on 7 GB RAM this may need to be run
   against the matching file alone (the validator's own docstring anticipates exactly this).
   Iterate to `PASS`.
5. Record runtime, peak RSS, and predicted-singleton rate per country as a final sanity check.

**Files:** `src/cli.py`, `output/*`

**Done when:** validator prints `PASS` with `--check-ids` on, and the sanity metrics are recorded.

---

### Phase 9 — Reproducibility & submission package

**Tasks**
1. `code/business_entity_resolution/README.md`: exact end-to-end run steps, runtimes, hardware
   assumptions, and the 7 GB-RAM design notes.
2. `requirements.txt` pinned to actual imports with exact versions.
3. Fill `Documentation_template.md`: methodology, blocking strategy (with measured recall ceiling
   and reduction ratio), features, model architecture, threshold selection, results and error
   analysis, the generalization study from Phase 7, and the explicit model-license statement.
4. **Model license & size statement**, verified from live model cards at the time of selection
   (Phase 3), not from memory — exact license, exact parameter count, and how it was confirmed.
5. Build `<team_name>_submission.zip` in the exact structure from the problem statement and verify
   by extracting to a clean directory and checking the tree.

**Files:** `README.md`, `requirements.txt`, `Documentation_template.md`, packaging script

**Done when:** the zip extracts to the specified structure and a clean-room reading of the README
would let a reviewer reproduce both output files.

---

## 5. Open questions for you

**Resolved 2026-09-25:** the university Slurm box `cmslab` is the primary environment, laptop for
dev (§1.7). AWS is dropped — GPU quota rejected, and free-tier instances are smaller than the
laptop. The RAM question is closed: 125 GB settles it.

### 5.1 Action items on your side

**All previously blocking items are now resolved.** SSH key auth works, the box is reachable without
VPN, outbound internet works for model/dependency downloads, and your home directory has room for the
baseline pipeline even with `/scratch` locked. Nothing is outstanding on your side except the sign-off itself.

Shared-machine etiquette is settled: you've held 60+ cores for 4-5 hours without complaint, so
sustained heavy CPU use is fine, and access is Slurm-gated. Standing rules I'm working under:
**all writes stay inside `/home/co24btech11023/`**, and **I never delete anything there — you do.**

**Completed under your approval (2026-09-25):** deleted the stale 2.5 GB `output/candidate_pairs.tsv`;
retired `run_pipeline.py`, `generate_synthetic_data.py`, `blocking.py`, `evaluate_recall.py`. Kept
`src/normalize.py` for migration into Phase 2. All removed files are backed up in the session
scratchpad in case you want anything back.

### 5.2 Still open — decisions I need from you

4. ~~**Model downloads.**~~ **Approved 2026-09-25.** Verified that huggingface.co is
   reachable from `cmslab`, so weights can be fetched directly on the box. Inference stays local.

5. **Embedding model.** My recommendation has tracked the hardware twice and has now landed back
   where it started: **`multilingual-e5-small` as the default**, because the A2000 is
   compute-comparable to the laptop GPU rather than to the A10G I was briefly planning around
   (reasoning in §1.7). The larger models from your original list stay in play as a **reranker over
   the capped candidate set**, which is ~40x less work than full-corpus blocking.
   **Recommendation: let the Phase 0 benchmark (task 0d) decide rather than my estimate**, with
   licence and parameter count verified from the live model card before locking in. Confirm you're
   happy with benchmark-and-pick.

6. ~~**Deleting the stale candidate file / retiring old code.**~~ **Approved and done** — see §5.1.

7. **Submission metadata** for `Documentation_template.md` and the zip filename: team name and
   team member list.

8. **Leaderboard cadence.** 5 submissions/day, ~58 hours left. I suggest one early R1/R2 probe to
   confirm our local validation tracks the leaderboard, then spend the rest on genuine
   improvements. Confirm you want me to flag when a submission is worth spending.

---

## 6. What I am most and least confident about

**Most confident:** the one-to-many assignment constraint (§1.3) is a large, structural precision
win that a lot of teams will miss; the country partition is safe and free; a union of independent
name and address channels is required and sufficient for high blocking recall.

**Least confident, and where I'd want your challenge:**

- Whether the lexical channels alone reach acceptable recall on Indic transliterations. Phase 1
  task 2 is designed specifically to answer this before we spend GPU hours, and the answer may
  change the priority of Phase 6 substantially in either direction.
- **Whether the A2000 can embed 20M records inside the remaining window.** Phase 0 task 0d answers
  this with a measurement in the first hour rather than a guess, and §3 is built so the answer
  changes our score, never our ability to submit.

- The K for candidate capping: the F0.5 objective punishes recall-chasing, so the optimum K may be
  much smaller than the recall-maximizing K. I'll tune it against F0.5 end-to-end, not against
  blocking recall in isolation.
