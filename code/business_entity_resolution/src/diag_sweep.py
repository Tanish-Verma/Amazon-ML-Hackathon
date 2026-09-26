"""
Cost/recall sweep: find parameters that cut retrieval time without losing recall.

The first full run took 79 minutes, 92% of it in retrieval. Parallelising across
24 processes bought only 1.45x -- sparse matmul is memory-bandwidth bound and
forking a multi-GB address space per channel eats much of the gain. So the work
itself has to shrink.

Retrieval cost is dominated by posting-list length: for each query we touch every
corpus record sharing any of its kept features. Two levers control that directly:

  max_df_frac  -- features present in more than this fraction of the corpus are
                  dropped. Their IDF weight is near zero anyway, so they
                  contribute almost nothing to ranking while costing the most.
  query_terms  -- how many of the query's rarest features we keep. The earlier
                  diagnostic showed rarest-14 scores IDENTICALLY to using all
                  terms, which suggests headroom to cut further.

Recall is measured with RRF fusion, since the fusion experiment showed RRF beats
the raw-sum fusion by ~16 points at K=30.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import replace

import numpy as np
from sparse_dot_topn import sp_matmul_topn

sys.path.insert(0, ".")
from src.blocking import ChannelSpec, ChannelIndex, feats_address, feats_name_ngrams, feats_name_tokens  # noqa: E402
from src.candidates import extract_features_parallel, fuse_rrf, load_partition  # noqa: E402

DEPTH = 120       # per-channel retrieval depth fed to fusion
KS = (20, 30, 50)


def make_config(tag, ng_df, ng_qt, ad_df, ad_qt, tk_df, tk_qt):
    return tag, (
        ChannelSpec("name_ngram", feats_name_ngrams, 1.0, DEPTH, ng_df, ng_qt),
        ChannelSpec("address",    feats_address,     1.0, DEPTH, ad_df, ad_qt),
        ChannelSpec("name_token", feats_name_tokens, 0.7, DEPTH, tk_df, tk_qt),
    )


# The first sweep only went TIGHTER and lost recall. The sparse_dot_topn
# benchmark then revealed the real problem: with max_df_frac=0.002 the queries
# carry only ~2.3 features each. Character trigrams have a very skewed
# distribution -- the ~1,500 most common trigrams account for most occurrences --
# so a tight max_df strips most of a name's trigrams and leaves almost nothing to
# match on. That, not the fusion alone, is why the name channel only reached 34%.
# So this sweep goes the other way: LOOSER, which we can now afford at 4.9x.
CONFIGS = [
    make_config("A 0.002 (old)", 0.002, 14, 0.002, 14, 0.005, 10),
    make_config("D 0.01",        0.010, 20, 0.010, 20, 0.010, 14),
    make_config("E 0.05",        0.050, 26, 0.050, 26, 0.050, 18),
    make_config("F 0.20",        0.200, 32, 0.200, 32, 0.200, 24),
    make_config("G no prune",    1.000, 40, 1.000, 40, 1.000, 30),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--country", default="India")
    ap.add_argument("--sample", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    q_ids, q_names, q_addrs = load_partition(args.store, "train", "s1", args.country)
    c_ids, c_names, c_addrs = [], [], []
    for src in ("s2", "s3"):
        i, n, a = load_partition(args.store, "train", src, args.country)
        c_ids += i; c_names += n; c_addrs += a
    crow = {e: i for i, e in enumerate(c_ids)}
    print(f"{args.country}: queries={len(q_ids):,} corpus={len(c_ids):,}", flush=True)

    gt = {}
    with open(args.gt, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.partition("\t")
            rest = rest.rstrip("\n")
            if rest:
                gt[s1] = rest.split(",")

    qpos = {e: i for i, e in enumerate(q_ids)}
    rng = random.Random(0)
    pick = [e for e in q_ids if e in gt and any(m in crow for m in gt[e])]
    rng.shuffle(pick); pick = pick[:args.sample]
    sel = [qpos[e] for e in pick]
    truth = {j: {crow[m] for m in gt[q_ids[qi]] if m in crow} for j, qi in enumerate(sel)}
    n_pairs = sum(len(v) for v in truth.values())
    print(f"sample: {len(sel):,} queries, {n_pairs:,} true pairs\n", flush=True)

    # Features do not depend on max_df/query_terms, so extract once and reuse.
    base = CONFIGS[0][1]
    t0 = time.perf_counter()
    cfe = extract_features_parallel(c_names, c_addrs, base, args.workers)
    qfe = extract_features_parallel([q_names[i] for i in sel], [q_addrs[i] for i in sel],
                                   base, args.workers)
    print(f"features once in {time.perf_counter()-t0:.0f}s\n", flush=True)

    scale = len(q_ids) / len(sel)   # extrapolate sample timing to the partition
    hdr = ("  " + "config".ljust(14) + "".join(f"{f'R@{k}':>8}" for k in KS)
           + f"{'retr s':>9}{'est full':>10}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))

    for tag, specs in CONFIGS:
        per_docs, per_scores, t_retr = [], [], 0.0
        feat_per_q = []
        for spec, cf, qf in zip(specs, cfe, qfe):
            index = ChannelIndex(spec, cf)
            Q = index._build(qf, prune_to_rarest=spec.query_terms)
            feat_per_q.append(Q.nnz / max(1, Q.shape[0]))
            t1 = time.perf_counter()
            # Multithreaded C++ top-n sparse matmul: identical mathematics to the
            # chunked scipy loop it replaces, ~4.9x faster at 24 threads.
            R = sp_matmul_topn(Q, index.matrix, top_n=DEPTH, sort=True, n_threads=24)
            t_retr += time.perf_counter() - t1
            docs, scores = [], []
            for r in range(R.shape[0]):
                i, j = R.indptr[r], R.indptr[r + 1]
                docs.append(R.indices[i:j]); scores.append(R.data[i:j])
            per_docs.append(docs); per_scores.append(scores)
            del index, Q, R
        weights = [s.weight for s in specs]
        hits = {k: 0 for k in KS}
        for j in range(len(sel)):
            ranked = fuse_rrf([pd[j] for pd in per_docs], [ps[j] for ps in per_scores],
                              weights, max(KS))
            for k in KS:
                hits[k] += len(truth[j] & set(ranked[:k]))
        est = t_retr * scale / 60.0
        fpq = "/".join(f"{v:.1f}" for v in feat_per_q)
        print("  " + tag.ljust(14)
              + "".join(f"{100*hits[k]/n_pairs:>7.2f} " for k in KS)
              + f"{t_retr:>8.1f}{est:>9.1f}m   feats/query {fpq}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
