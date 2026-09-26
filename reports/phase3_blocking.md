# Phase 3 — blocking: measurements and design decision

## Summary

The first implementation scored macro recall **0.7102**. Three defects and one
wrong assumption were found and measured. With those fixed, a broad candidate
pool reaches **97.93%** recall at a cost we can afford.

## Defect 1 — fusion by raw score sum (cost: ~16 points)

Channels were fused by a weighted sum of raw cosine scores. Character-trigram
cosines are far larger in magnitude than address-token cosines, so name
candidates systematically outranked and evicted address candidates during the
final top-K cap. That is catastrophic for transliterated records, whose *only*
retrieval route is the address.

India, identical retrieved candidates, recall@30:

| fusion | recall@30 |
|---|---|
| sum of raw scores (original) | 70.13% |
| **reciprocal rank fusion (RRF)** | **86.64%** |
| per-channel z-normalisation | 86.98% |
| round-robin quota | 85.34% |

RRF adopted: it is scale-free by construction — the property the raw sum lacked —
parameter-free, and it overtakes z-norm at K >= 50.

## Defect 2 — max_df pruned away the signal (cost: ~5 points)

`max_df_frac=0.002` dropped any feature present in >0.2% of the corpus. This was
chosen on the assumption that high-frequency features contribute nothing to
ranking. That assumption was wrong for character trigrams, whose distribution is
extremely skewed: the ~1,500 most common trigrams account for most occurrences,
so a tight threshold strips most of a name's trigrams. Measured effect: queries
carried only **2.3 usable features each**.

| max_df_frac | feats/query (ngram) | recall@30 |
|---|---|---|
| 0.002 (original) | 2.3 | 86.72% |
| **0.01** | **6.6** | **91.91%** |
| 0.05 | 12.6 | 92.94% |
| no pruning | 21.2 | 94.01% |

Every earlier sweep went *tighter*, i.e. further in the wrong direction.

## Defect 3 — single-threaded retrieval

Retrieval was 92% of an 79-minute run. Parallelising across 24 processes gave
only **1.45x** (sparse matmul is memory-bandwidth bound, and forking a multi-GB
address space per channel eats the rest). Replaced with `sparse_dot_topn`'s
`sp_matmul_topn`, a multithreaded C++ top-n sparse matmul: identical mathematics,
**4.9x** at 24 threads. 48 threads was slower than 24, confirming the bandwidth
ceiling.

## New channel — exact digit keys

Digit runs are the one script-invariant signal (reports/eda.md §5): a street
number reads identically in Devanagari, Tamil, French and Latin. Split into its
own channel rather than left inside the address channel, where it shared L2
normalisation with dozens of address word tokens and was diluted by unshared
street/city words. Its feature space is tiny and highly selective, so it costs
**0.1s** per partition-sample — effectively free.

## The decisive measurement: recall vs pool depth

India, 5,000 queries / 18,451 true pairs, four channels, RRF fusion:

| config | @30 | @50 | @100 | @200 | @500 | @1000 | union ceiling | est/partition |
|---|---|---|---|---|---|---|---|---|
| **D (max_df 0.01)** | 91.73 | 93.40 | 94.95 | 96.18 | **97.26** | **97.93** | 98.84% | **26m** |
| E (max_df 0.05) | 92.98 | 94.91 | 96.47 | 97.61 | 98.40 | 98.84 | 99.35% | 113m |

Per-channel solo recall@1000 under config D:

| channel | solo R@1000 | time |
|---|---|---|
| address | **92.64%** | 2.2s |
| name_ngram | 71.93% | 5.5s |
| name_token | 67.80% | 0.8s |
| digit_key | 43.93% | 0.1s |

## Decision: go broad on K, not loose on max_df

**D@1000 (97.93%, 26m) beats E@50 (94.91%, 113m) — 3 points better at a quarter
of the cost.** Depth is nearly free because it is the same matmul with a deeper
cut, whereas loosening `max_df` multiplies the matmul itself. This is why the
reviewer's broad-pool instinct was right and my top-30 framing was wrong.

Cost of config D: train-subset ~11m + full test ~67m = **~1.3h**.

## Consequence: a rerank stage is mandatory, not optional

`candidate_pairs.tsv` cannot carry K=1000: 1.73M test entities x 1000 is 1.73
**billion** pairs (~21 GB), and featurising that in Phase 4 is not feasible. So
the pipeline becomes:

```
4 cheap retrieval channels  (depth 1000 each)
        v
RRF fusion -> broad pool of 500-1000        [held in memory, never written]
        v
cheap feature rerank                         [part of blocking]
        v
top 30-50  ->  candidate_pairs.tsv
        v
strong matcher (Phase 5)
```

This is consistent with the problem statement, which defines
`candidate_pairs.tsv` as "the exact set of records you feed into your matching
model for inference" — so the post-rerank list is the correct thing to write.

## The open question

RRF@50 is 93.40%, but RRF ranks by rank-fusion rather than by actual pairwise
similarity. A cheap reranker scoring real features (char-trigram Jaccard, token
Jaccard, digit overlap, Jaro-Winkler) over the 1000-pool should rank far better at
the top. **If rerank@50 approaches the pool's 97.93%, we get broad-pool recall at
top-50 downstream cost.** That measurement decides the final K and is the next
step.
