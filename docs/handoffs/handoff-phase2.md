# Handoff prompt — Phase 2 (Normalisation)

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

Phases 0 and 1 are complete and committed. `main` is current.

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
- **A lexical pipeline reaches 99.88% of true matches.** Name-only 90.43%,
  address-only 94.82%. Only 0.12% need anything cleverer.
- **Transliterated Indic names are solved by the address channel**, not by
  embeddings: name-matching finds under 1% of them, but the union finds 98–99%,
  because street numbers survive transliteration.
- **France generalisation is already measured, not assumed.** Token DF on the
  test split: France `sarl` 28.3%, `sas` 20.1%, `eurl` 6.5%, `sasu` 4.1% — the
  same band as US `llc` 26.9% / `inc` 18.0%. The same DF rule also catches French
  function words (`de`, `du`, `des`) that an English stoplist would miss. This is
  why derived stoplists beat hand-written tables; don't undo it.
- **26% of Source-2/3 records match nothing.** Precision, not recall, decides the
  score.
- Legal-form tokens dominate name document-frequency (India `limited` 59.1%,
  `private` 48.9%; US `llc` 26.9%, `inc` 18.0%), so a DF-threshold rule
  down-weights them — and will catch France's `SARL`/`SAS`/`EURL` by the same
  mechanism with no French-specific table.

## YOUR TASK: PHASE 2 — NORMALISATION

Build the text normalisation layer that every later stage depends on. Target
file: `code/business_entity_resolution/src/normalize.py` (a first draft exists
from earlier work — **read it, then redesign it**; it was written before the EDA
and hardcodes per-country dictionaries, which is exactly what we must avoid).

1. **Unicode-correct cleanup.** NFKC, case folding, and diacritic stripping that
   applies to Latin text but **must not** decompose Indic scripts — their vowel
   signs are combining marks and stripping them destroys the text. `src/eda.py`
   has a working version of this guard; reuse the idea.
2. **Script-agnostic tokenisation — and we already know one way to get this
   wrong.** The Phase 1 EDA originally used `[^\w\s]` to strip punctuation. That
   looks correct and silently destroys every Indic script, because their vowel
   signs are Unicode categories `Mn`/`Mc` which `\w` does not match:

   ```
   'प्राइवेट'  --[^\w\s]-->  'प र इव ट'        # shattered into loose consonants
   ```

   It surfaced as single Devanagari characters topping the document-frequency
   table. The fix is to strip by Unicode *category* — remove `P*`/`S*`/`C*`, keep
   `L*`/`N*`/`M*`. See `reports/eda.md` §7 and the corrected `basic_norm` in
   `src/eda.py`. Your tokeniser must handle Latin, Devanagari, Tamil, Telugu,
   Kannada, Bengali, Gujarati, Malayalam, Oriya and Gurmukhi — all confirmed
   present — plus French accented Latin. Never `[a-z]+`, and never bare `\w`.
3. **Derived stoplists, not hand-written ones.** Compute per-country token
   document frequency at runtime and down-weight high-DF tokens. A small
   hand-written legal-suffix list may exist *only* as a supplementary signal
   behind a feature flag, so Phase 7 can measure whether removing it hurts an
   unseen country. This is the core generalisation requirement — a hardcoded
   per-country table is the failure mode we are actively designing against.
4. **Numeric/address key extraction.** Canonicalise digit runs (`AF-0684` →
   `af684`, `1056-1060` → `{1056, 1060}`). Per the EDA these are the strongest
   script-invariant signal we have; treat them as a first-class output.
5. **Outputs per record:** `clean`, `tokens`, `core_tokens` (high-DF removed),
   `joined` (de-spaced, for domain-style names like `bnpgroup.com`), and
   `char_ngrams`.
6. **Tests** in `tests/test_normalize.py` over real noisy examples: the
   transliteration pairs, `Payne Énterprises`/`PAYNE-ENRTPRMISES`, empty
   addresses, and **French examples** (`Thermal & Fils SASU`,
   `20 Rue Parmentier`, `63 R. DE DIEPPE`, `Établissements Dëleves EURL`).
7. **Measure throughput.** 20M records get normalised, so report records/sec. If
   it can't sustain ~100k/sec/core it becomes the bottleneck.

**Done when:** tests pass on real examples from all three countries, throughput
is measured and reported, and nothing in the module branches on a specific
country name.

## WHEN YOU FINISH PHASE 2

1. Summarise what changed and how I can inspect it.
2. Commit locally with a clear message (**do not push**).
3. Update `PLAN.md` with the Phase 2 result.
4. **Write the next handoff prompt** to `docs/handoff-phase3.md`, in the same
   shape as this document, by re-reading `PLAN.md` and `reports/` so it carries
   accurate, current context. A draft already exists — replace it with one that
   reflects what Phase 2 actually built, including any surprises, and carry
   forward this instruction so Phase 3 produces `docs/handoff-phase4.md` in turn.
5. Stop and wait for my review.

## HOW I WANT YOU TO WORK

- Measure rather than assume, and show me the numbers.
- If the data contradicts the plan, say so and propose the change — don't just
  follow the plan off a cliff.
- Tell me plainly when something failed or you got something wrong.
- Don't over-engineer. There is a deadline and a working baseline already exists.
