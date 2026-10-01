# Handoff prompt — Phase 5 (Matching model and decision rule)

**How to use this:** open a new Claude Code session in the repo and paste
everything below the line. It is self-contained.

---

## ROLE

You are the lead ML engineer on a business entity-resolution project (Amazon ML
Challenge 2026). I am your reviewer. You implement **one phase at a time**, then
summarise what changed, tell me how to inspect it, and **stop and wait** for my
go-ahead. If you want to change a decision already recorded in `PLAN.md`, flag it
explicitly and ask — never revise it silently.

## FIRST, READ THESE (do not ask me to paste them)

1. `code/business_entity_resolution/README.md` — the pipeline, the folder
   structure, and the design decisions with their evidence. Start here.
2. `PLAN.md` — the phase plan and every decision made so far.
3. `reports/` — the measurements. `phase3_blocking.md` is the most important.
4. `code/business_entity_resolution/src/config.py` — every tuned constant, each
   annotated with the measurement that chose it.
5. The problem statement PDF — the source of truth. Verify anything I summarise.

## WHERE THINGS RUN

```bash
ssh co24btech11023@10.2.4.21        # key auth works, no VPN
cd ~/amlc && git pull
source ~/amlc/venv/bin/activate
cd ~/amlc/code/business_entity_resolution
```

48 cores, 125 GB RAM, RTX A2000 12 GB (**not needed** — the pipeline is CPU-only).
Data at `~/amlc/dataset/`, Parquet store at `~/amlc/work/store/`.

**Rules for this machine, non-negotiable:**
- **Never delete anything.** Ask the owner. This includes your own mistakes.
- Write only inside `/home/co24btech11023/`. Never `/tmp`, `/dev/shm`, `/scratch`.
- It is shared. **Check free RAM before any bulk allocation** — this pipeline has
  OOM'd the box before, and `src/run_blocking.py::_mem_guard` exists because of it.
- **Never `git push`.** Commit locally; the repo owner pushes.
- Run long jobs one country at a time, and report RAM/swap periodically.

## FACTS ALREADY ESTABLISHED — do not re-derive

- **Metric: macro-averaged F0.5.** Precision counts double. A false match on a true
  singleton costs a full point. When unsure, predict nothing.
- **5.58%** of Source-1 entities are true singletons.
- **Each S2/S3 record is claimed by exactly one S1 entity** — verified across all
  7,638,365 links. This is the single biggest precision lever available.
- **No true match crosses a country boundary**, so `country` is a partition key.
  Never hardcode `{US, India}`. The test set contains **France** (15% of entities),
  absent from training.
- **26% of S2/S3 records match nothing** (~2.7M distractors). Precision, not
  recall, decides the score.
- **Blocking is done and measured:** pool recall 98.8% India / 99.3% US at depth
  1000; after reranking, **97.4% / 99.3% at K=50**. `candidate_pairs.tsv` carries
  50 candidates per entity.
- **No embeddings, no GPU.** Measured worth of an embedding channel: 0.12% recall.
  The address channel solves transliteration because digits survive script change.
- Test set: **1,732,544** Source-1 entities. Every one needs exactly one output row.

## HOW I WANT YOU TO WORK

- Measure rather than assume, and show me the numbers.
- Smoke-test on the smallest partition before a full run, and **do not change
  memory-related parameters between the smoke test and the real run** — that
  mistake cost hours here.
- If the data contradicts the plan, say so and propose the change.
- Tell me plainly when something failed or you got something wrong.
- Don't over-engineer. A working baseline already exists; protect it.

## YOUR TASK: PHASE 5 — MATCHER + DECISION RULE

**This phase has more leverage on the final score than anything else.** At
precision 0.9, four extra points of blocking recall are worth ~+0.008 F0.5; seven
points of precision are worth ~+0.055. Spend your effort here.

Target files: `src/model.py`, `src/decide.py`.

1. **Labels:** a candidate pair is positive iff it is in that entity's ground-truth
   list. Expect ~7% positives at K=50.

2. **Model: LightGBM.** Note for the methodology doc — this is a gradient-boosted
   tree ensemble trained from scratch on the provided data, a few MB of tree
   structure, MIT licensed. The ≤8B-parameter / MIT-or-Apache constraint is
   satisfied trivially.
   **Caution from Phase 3:** a LightGBM *classifier* ranked badly there (54% @10 vs
   94% for logistic regression) because classification loss with 0.13% positives
   produces coarse, tied scores. If you use trees for ranking, use `LGBMRanker`
   with lambdarank **grouped by query**, or calibrate carefully. Compare against
   logistic regression as the baseline to beat.

3. **Calibrate probabilities** (isotonic) so thresholds are interpretable.

4. **The decision rule — not a flat cutoff.** This is the precision lever:
   - a. A high-precision threshold `t`, tuned directly for macro-F0.5.
   - b. **Enforce the one-to-many constraint:** each S2/S3 record may be assigned
     to at most one S1 entity. Resolve contention by score, requiring a margin
     before one entity steals a record from another. This removes false merges
     that no per-pair threshold can.
   - c. **Per-entity relative rule:** accept only candidates within a margin of
     that entity's best score, so one confident match doesn't drag in siblings.
   - d. Use the observed per-source caps (≤5 from S2, ≤6 from S3) as a **soft
     prior or tie-break only** — they are estimated from train and may differ in test.
   - e. **Explicit abstention:** if nothing clears `t`, emit an empty list.
     Singletons are 5.58% and worth 1.0 each.

5. **Report an ablation table** showing what each component of (4) contributes, and
   per-country F0.5 so you are not tuning a threshold that only works for one country.

**Done when:** validation macro-F0.5 is reported overall and per country, with the
ablation table, and the assignment constraint's contribution is quantified.

## WHEN YOU FINISH

1. Summarise what changed and how I can inspect it.
2. Commit locally with a clear message (**do not push**).
3. Update `PLAN.md` with the result, and add `reports/phase5.md` with the numbers.
4. **Write the next handoff prompt** to `docs/handoff-phase6.md` in this same
   shape, by re-reading `PLAN.md` and `reports/` so it carries current context.
   Carry this instruction forward.
5. Stop and wait for my review.
