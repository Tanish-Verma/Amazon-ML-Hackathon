# Data artefacts: what ships, what doesn't, and how to get it back

## Committed to the repo

| Path | Size | Why it ships |
|---|---|---|
| `models/reranker.pkl` | 2.2 KB | The trained reranker — 16 coefficients + a scaler. Tiny, and it makes the pipeline runnable without rebuilding anything. |
| `work/paircache/pairs_India.npz` | 243 MB | **Git LFS.** 11.4M labelled train pairs with all 16 features computed. |
| `work/paircache/pairs_US.npz` | 217 MB | **Git LFS.** 11.8M labelled train pairs. |
| `trainsplit/cand_train_India.tsv` | 99 MB | **Git LFS.** 150k train entities at K=50 — the labelled dev/confirm set. |
| `trainsplit/cand_train_US.tsv` | 99 MB | **Git LFS.** 150k train entities at K=50. |

### The two train-side artefacts, and when to use which

**`trainsplit/cand_train_*.tsv` is what Phases 4-6 should develop against.** 300,000
train entities with exactly the candidates the matcher will see in production — same
model, same depth 1000, same K=50 as the test run. Join against
`dataset/train/train_ground_truth.tsv` to get labels. Configuration parity with the test
run matters: a matcher tuned on a different candidate distribution than it meets at
inference is quietly miscalibrated, and that failure shows up on the leaderboard rather
than in validation.

**`work/paircache/*.npz` is for fast experiments.** 4,000 entities per country, but the
full depth-1000 pool with all 16 features *already computed* — so model comparisons,
threshold sweeps and depth studies run in seconds rather than hours. This is how the
depth-truncation sweep was measured in under a minute. Too small to train a final
matcher on, and it is the 1000-deep pool rather than the production top-50.

The pair caches ship because they are **not cheaply regenerable** — rebuilding them
means ~25 minutes of retrieval plus feature computation per country. With them you
can retrain the reranker, re-measure recall at any depth, or try a different model
class in seconds.

```bash
git lfs pull          # after cloning, to fetch the caches
```

## NOT committed (and why)

| Path | Size | Why not | How to regenerate |
|---|---|---|---|
| `dataset/` | 2.4 GB | The organisers' challenge data — not ours to redistribute. | Download from the challenge portal. |
| `work/store/` | 864 MB | Regenerable in ~23 seconds. | `python -m src.cli --data-dir dataset --work-dir work prep` |
| `prod/cand_*.tsv` | ~1.1 GB | Would exceed GitHub's 1 GB free LFS quota on its own, and it is a submission artefact rather than source. | The production run — see below (~3h). |
| `.venv/`, `venv/` | 5–7 GB | Environment. | `bash src/bootstrap.sh` |

**On the LFS quota:** GitHub's free tier is 1 GB storage and 1 GB/month bandwidth.
The pair caches use 460 MB of that. Adding the ~1.1 GB of candidate TSVs would
exceed it and require a paid data pack, which is why they are excluded.

## Regenerating the candidate files

```bash
# one country at a time, or all three concurrently (the box has 48 cores / 125 GB)
python -m src.run_blocking --store work/store --split test \
    --countries India --model models/reranker.pkl \
    --out prod/cand_India.tsv --top-k 50 --workers 8 --batch 20000
```

Then concatenate the three into `output/candidate_pairs.tsv` (header once).

Measured cost, three countries concurrently: **~3 hours**, peak ~45 GB RAM.
