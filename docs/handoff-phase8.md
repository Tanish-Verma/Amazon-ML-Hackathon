# Handoff prompt — Phase 8 (Inference and packaging)

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

## YOUR TASK: PHASE 8 — FULL TEST INFERENCE + PACKAGING

1. **Run the full test set** country by country: blocking is already done and
   `candidate_pairs.tsv` exists — apply features → model → decision rule to produce
   `output/matching_results.tsv`. Stream by batch; report RAM/swap periodically.

2. **Assert every validator rule yourself before running the validator:** exactly
   1,732,544 rows in each file, one row per test S1 entity, no duplicate S1 rows,
   no duplicate IDs within a list, S2-/S3- prefixes only, matched ⊆ candidates,
   UTF-8, tab-separated, exact headers. `tests/smoke_candidates.py` covers most of
   this for a candidate file; extend it for the matching file.

3. **Run the official validator to PASS:**
   ```
   python3 utils/validate_submission.py --matching output/matching_results.tsv \
       --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
   ```
   `--check-ids` loads all S2/S3 IDs into memory; if it struggles, run it against
   the matching file alone — that is the only scored file.

4. **Record sanity metrics:** predicted singleton rate per country, mean matches
   per entity, runtime, peak RSS. Compare the singleton rate against train's 5.58%
   — a large divergence means the threshold is miscalibrated.

5. **Build the submission zip** in exactly the structure the problem statement
   specifies: `output/`, `code/business_entity_resolution/{src,README.md,
   requirements.txt}`, and the filled `Documentation_template.md`. Verify by
   extracting to a clean directory.

**Done when:** the validator prints `PASS` with `--check-ids`, the sanity metrics
are recorded, and the zip extracts to the specified structure.

## WHEN YOU FINISH

1. Summarise what changed and how I can inspect it.
2. Commit locally with a clear message (**do not push**).
3. Update `PLAN.md` with the result, and add `reports/phase8.md` with the numbers.
4. **Write the next handoff prompt** to `docs/handoff-phase9.md` in this same
   shape, by re-reading `PLAN.md` and `reports/` so it carries current context.
   Carry this instruction forward.
5. Stop and wait for my review.
