# Handoff prompt — Phase 3 (Blocking / candidate generation)

**How to use this:** open a new Claude Code session in the repo and paste
everything below the line. It is self-contained; Claude does not need this
conversation's history.

---

## ROLE

You are the lead ML engineer on a business entity-resolution project (Amazon ML
Challenge 2026). I am your reviewer. You implement **one phase at a time**, then
summarise what changed, tell me how to inspect it, and **stop and wait** for my
go-ahead. If you want to change a decision already recorded in `PLAN.md`, flag it
explicitly and ask — never revise it silently.

## FIRST, READ THESE (do not ask me to paste them)

In the repo, in this order:

1. `PLAN.md` — the full phase plan, the hard constraints, and every decision made
   so far with its reasoning.
2. `reports/eda.md` — Phase 1 measurements. §6 is the important part.
3. `reports/phase0.md` — environment and verified dataset facts.
4. `code/business_entity_resolution/src/` — existing code. Read
   `io_utils.py`, `scoring.py`, `cli.py`, `eda.py` before writing anything.
5. `6ab5628d5a817_amazon_ml_challenge_problem_statement.pdf` — the source of
   truth for all rules. Verify anything I've summarised against it yourself.

## WHERE THINGS RUN

Everything runs on a shared university server:

```bash
ssh co24btech11023@10.2.4.21          # key auth already works, no VPN
cd ~/amlc && git pull
source ~/amlc/venv/bin/activate
cd ~/amlc/code/business_entity_resolution
```

Specs: 48 cores, 125 GB RAM, RTX A2000 12 GB, CUDA driver 535 (so torch must be
a **cu121** build). Data is at `~/amlc/dataset/`, the parquet store at
`~/amlc/work/store/`.

**Rules for this machine, non-negotiable:**
- **Never delete anything.** Ask the owner to do it. This includes cleaning up
  your own mistakes.
- Write only inside `/home/co24btech11023/`. Never `/tmp`, `/dev/shm`, `/scratch`.
- Treat disk as limited; check free space before bulk writes.
- It's shared. Don't assume you're the only user.
- **Never `git push`.** Commit locally; the repo owner pushes.
- Add anything large to `.gitignore` and tell the owner about it.

## STATE OF PLAY

Phases 0, 1 and 2 are complete and committed.

**Before you start, read `docs/handoff-phase3.md`'s predecessor context by
checking what Phase 2 actually produced** — `src/normalize.py`, its tests, and
the Phase 2 entry in `PLAN.md`. Phase 3 consumes its output directly, so if the
normaliser's interface differs from what is described below, trust the code.

**The task:** for each of 1,732,544 test Source-1 business records, find all
matching records among ~10M Source-2/Source-3 records. Output two TSVs in
`output/`: `matching_results.tsv` (scored) and `candidate_pairs.tsv` (the
blocking stage's final candidate set).

**Facts already established — do not re-derive:**

- Scored by **macro-averaged F0.5**: precision counts double. A false match on a
  true singleton costs a full point. When unsure, predict nothing.
- **5.58%** of Source-1 entities are true singletons.
- **Each Source-2/3 record is claimed by exactly one Source-1 entity** — verified
  across all 7,638,365 links. This becomes an assignment constraint in Phase 5.
- **No true match crosses a country boundary**, so `country` is a safe partition
  key. Partition on the literal label; never hardcode `{US, India}`. The test set
  contains **France**, which is absent from training and is 15% of the score.
- **A lexical pipeline reaches 99.86% of true matches.** Name-only 90.46%,
  address-only 94.78%. Only 0.14% need anything cleverer.
- **Transliterated Indic names are solved by the address channel**, not by
  embeddings: name-matching finds ~1% of them, but the union finds 98–99%,
  because street numbers survive transliteration.
- **26% of Source-2/3 records match nothing.** Precision, not recall, decides the
  score.
- Legal-form tokens dominate name document-frequency (India `limited` 59.1%,
  `private` 48.9%; US `llc` 26.9%, `inc` 18.0%), so a DF-threshold rule
  down-weights them — and will catch France's `SARL`/`SAS`/`EURL` by the same
  mechanism with no French-specific table.

## YOUR TASK: PHASE 3 — BLOCKING / CANDIDATE GENERATION

This phase sets the **recall ceiling** for the whole system: any true match not
retrieved here is unrecoverable no matter how good the later model is. The EDA
says a lexical approach can reach 99.86%, so the target is to actually realise
that in a real index at scale.

Target files: `src/blocking_lexical.py`, `src/candidates.py`, plus `src/cli.py`
subcommands.

1. **Work per country partition.** The parquet store is already partitioned that
   way. This is a partition on whatever label is present, not a rule — it must
   work unchanged when France appears.

2. **Build independent channels and take their union.** The EDA is unambiguous
   that neither channel alone is enough (name 90.46%, address 94.78%, union
   99.86%):
   - **Channel A — name character n-grams.** 3-5 char n-grams, IDF-weighted per
     country, retrieved via a sparse inverted index with a per-token posting cap.
     Character n-grams are script-agnostic and typo-robust.
   - **Channel B — address keys.** Canonical digit runs crossed with rare address
     word tokens. This is the channel that solves transliteration; give it real
     attention rather than treating it as a backstop.
   - **Channel C — name token sets.** Cheap, catches word reordering.
   - **No embedding channel.** Phase 1 measured its value at 0.14% additional
     recall, which cannot repay the GPU hours or storage. If you believe the
     evidence says otherwise, argue it with numbers before building it.

3. **Fuse and cap.** Union the channels, score each candidate with a cheap
   combined signal, keep the top K per Source-1 entity. True match lists max out
   at 11, so explore K in roughly 25-40. **Tune K against end-to-end F0.5, not
   against blocking recall alone** — F0.5 punishes recall-chasing, so the best K
   is probably smaller than the recall-maximising one.

4. **The capped set is what `candidate_pairs.tsv` contains.** The problem
   statement is explicit: it is the final candidate list the model scores, not an
   early pass you filter later.

5. **Measure and report**, per country: macro recall, micro recall, reduction
   ratio, candidate-set size distribution, and a sample of missed true matches
   with a diagnosis of *why* each was missed.

6. **Watch resources.** ~10M records per split. Report peak RSS and wall-clock,
   and extrapolate the full test run before committing to it. A previous attempt
   at this exhausted memory by holding tuples in a Python dict — prefer compact
   arrays and interned strings.

**Done when:** blocking runs on a train validation slice with measured macro
recall (target >=0.98, given the 99.86% ceiling), the reduction ratio and size
distribution are reported per country, peak RSS and runtime are recorded, and
the full-test runtime is extrapolated and shown to fit the deadline.

## WHEN YOU FINISH PHASE 3

1. Summarise what changed and how I can inspect it.
2. Commit locally with a clear message (**do not push**).
3. Update `PLAN.md` with the Phase 3 result, including the measured recall
   ceiling, since every later phase is bounded by it.
4. **Write the next handoff prompt** to `docs/handoff-phase4.md`, in the same
   shape as this document, by re-reading `PLAN.md` and `reports/` so it carries
   accurate, current context. Carry this instruction forward so Phase 4 produces
   `docs/handoff-phase5.md` in turn.
5. Stop and wait for my review.

## HOW I WANT YOU TO WORK

- Measure rather than assume, and show me the numbers.
- If the data contradicts the plan, say so and propose the change — don't just
  follow the plan off a cliff.
- Tell me plainly when something failed or you got something wrong.
- Don't over-engineer. There is a deadline and a working baseline already exists.
