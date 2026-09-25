# What we're actually building — a plain guide

Read section 1 and stop if that's all you need. Each section after it goes a
layer deeper. No prior knowledge of this codebase is assumed.

---

## 1. The problem, in plain English

Three different companies each keep a list of businesses. Nobody agreed on
spelling, formatting, or even language. The same corner shop might appear as:

```
List 1 (the reference):  Payne Enterprises      3315 Fremont Street, Peoria, IL
List 2:                  Payne Énterprises      3315 FREMONT ST, PEORIA, IL
List 2:                  PAYNE-ENRTPRMISES      3315 FREMONT SAINT, PEORIA, IL
List 3:                  Payne Etrepndiels      3315 Fremont St, Peoria, Illinois
```

Those four rows are **the same business**. There is no shared ID connecting
them — no registration number, no phone number, nothing. All we get is a name,
an address, and a country.

**Our job:** for every business in List 1, find all of its copies in Lists 2
and 3. That's it. This is called **entity resolution**.

The output is one line per List-1 business:

```
S1-00001    S2-00047,S2-00193,S3-00812
S1-00002    S3-00004
S1-00003                                  <- this one has no matches anywhere
```

### Why it's hard

- **It's big.** 1.7 million businesses in List 1, and ~10 million rows to search
  through. Comparing everything to everything is 17 trillion comparisons. Not
  happening.
- **The noise is deliberate.** Typos, missing words, reordered words, dropped
  address pieces.
- **It's multilingual.** An Indian business often appears translated into
  Hindi/Tamil/Telugu script: `Raj Investments LLP` becomes
  `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி`. Zero letters in common.
- **The test set contains France, which never appears in training.** So
  anything we hardcode about US or Indian addresses will fail on 15% of the
  answer sheet.
- **Some businesses genuinely have no match** (5.6% of them). Saying "no match"
  correctly earns full marks. Guessing wrong loses everything for that business.

### How we're scored

A precision-weighted score called **F0.5**. Translation: **being wrong is twice
as bad as missing something.** If we're unsure, the correct move is to say
nothing. That single fact drives most of our design decisions.

---

## 2. The approach, in three steps

**Step 1 — Blocking (narrowing down).** We can't compare everything to
everything. So for each List-1 business we cheaply pull maybe 30 plausible
candidates out of the 10 million. This is a search-engine-style index. The rule
here: *be generous.* Anything we fail to pull at this stage is lost forever.

**Step 2 — Scoring.** For each of those ~30 candidate pairs we compute a few
dozen similarity measures — how alike are the names, the addresses, the street
numbers — and feed them to a trained model that outputs "how likely is this a
real match?"

**Step 3 — Deciding.** We turn those scores into a final yes/no list. The rule
here is the opposite of step 1: *be strict*, because wrong answers cost double.

We deliberately keep steps 1 and 2 separately runnable, so we can measure "did
blocking lose anything?" independently from "is the model accurate?"

---

## 3. The two findings that shaped everything

### Finding 1: each record belongs to exactly one business

We checked all 7,638,365 known matches. **Not one** List-2 or List-3 record
belongs to more than one List-1 business. Ever.

That's a gift. It means if two different businesses both want to claim the same
record, at most one of them is right — so we can make them compete and throw out
the loser. That kills a whole category of mistakes for free, which matters a lot
when mistakes cost double.

### Finding 2: we don't need AI embeddings for the multilingual problem

This was the scary one. How do you match `Raj Investments LLP` to
`ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி` when they share no characters? The obvious
answer is a multilingual neural network that maps both to the same "meaning".

We measured it instead of assuming. Result:

| Matching using... | Finds this % of true matches |
|---|---|
| Names only | 90.46% |
| Addresses only | 94.78% |
| **Either one** | **99.86%** |

Only **0.14%** of matches are invisible to both. And for the transliterated
Indian names specifically, name-matching finds ~1% — but the *address* finds
98–99% of them, because **street numbers don't change when you translate a
name.** `6(29), C.I.T. Colony, Chennai` looks the same in every script.

So the address channel solves the multilingual problem, and the expensive neural
approach would buy us an extra 0.14%. Given that the score punishes wrong answers
twice as hard as missing ones, that's not worth having.

**This is why we're not spending our GPU time on embeddings.** We measured, and
the simple thing wins.

---

## 4. Where the score will actually be won

About **26% of the records in Lists 2 and 3 match nothing at all** — roughly 2.7
million decoys sitting in the pool looking plausible. Combined with the 5.6% of
businesses that legitimately have no match, and a scoring rule that punishes
false matches double:

> Our problem is not finding matches. It's refusing the wrong ones.

Everything from here is about precision.

---

## 5. Current state

| Phase | What it does | Status |
|---|---|---|
| 0 | Environment, data checks, output plumbing | **Done** |
| 1 | EDA — understand the data | **Done** |
| 2 | Normalisation — clean up text across languages | Next |
| 3 | Blocking — build the candidate finder | |
| 4 | Features — similarity measures | |
| 5 | Matching model + decision rule | |
| 6 | Validation harness | |
| 7 | Unseen-country (France) stress test | |
| 8 | Run on test data, package | |
| 9 | Documentation, reproducibility | |

We already have a valid submission file that scores ~0.056 (it says "no match"
for everything). It's worthless as a score but it proves our output format is
accepted, which means we can never end up with nothing to submit.

---

## 6. Running it yourself

Everything lives on the lab server:

```bash
ssh co24btech11023@10.2.4.21
cd ~/amlc && git pull
source ~/amlc/venv/bin/activate
cd ~/amlc/code/business_entity_resolution

python -m src.cli --data-dir ~/amlc/dataset verify     # sanity-check the data (~17s)
python tests/test_scoring.py                           # check the scorer
```

**Heads-up:** it's one shared machine and one shared folder. Before a big run,
tell the others. Don't delete anything that isn't yours.

Deeper reading, in order: `reports/eda.md` (all the measurements),
`PLAN.md` (the full phase-by-phase plan and why each decision was made).
