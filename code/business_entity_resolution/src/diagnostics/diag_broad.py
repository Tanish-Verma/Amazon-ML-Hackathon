"""
Broad-candidate experiment.

Question posed by the reviewer: rather than pushing blocking itself toward very
high precision at top-30, is it better to retrieve a BROAD pool and let a
downstream reranker/matcher do the hard ranking? Concretely -- is R@500 or R@1000
materially better than R@50?

If it is, the pipeline should become:

    cheap retrieval channels -> broad union/RRF -> top 100-1000
      -> cheap feature rerank -> top 30-50 -> strong matcher

This measures the recall curve out to K=1000 for the best affordable loose
configuration, with a NEW exact digit-key channel added (digit runs are the one
script-invariant signal per reports/eda.md, and they retrieve via a tiny,
selective feature space, so the channel is nearly free).

Also reports each channel alone at depth 1000, so we can see which channels are
actually contributing and which are only adding cost.
"""

from __future__ import annotations

import argparse
import random
import sys
import time

import numpy as np
from sparse_dot_topn import sp_matmul_topn

sys.path.insert(0, ".")
from src.blocking import (  # noqa: E402
    ChannelIndex, ChannelSpec, feats_address, feats_digit_keys,
    feats_name_ngrams, feats_name_tokens,
)
from src.candidates import extract_features_parallel, fuse_rrf, load_partition  # noqa: E402

DEPTH = 1000
KS = (30, 50, 100, 200, 500, 1000)
THREADS = 24


def config(tag, df, qt_ng, qt_ad, qt_tk):
    return tag, (
        ChannelSpec("name_ngram", feats_name_ngrams, 1.0, DEPTH, df,    qt_ng),
        ChannelSpec("address",    feats_address,     1.0, DEPTH, df,    qt_ad),
        ChannelSpec("name_token", feats_name_tokens, 0.7, DEPTH, df,    qt_tk),
        # Exact-key channel: tiny selective space, so a tight max_df is right --
        # a digit run in >0.2% of addresses ('1', '100') carries no signal.
        ChannelSpec("digit_key",  feats_digit_keys,  1.0, DEPTH, 0.002, 16),
    )


CONFIGS = [config("D 0.01", 0.01, 20, 20, 14),
           config("E 0.05", 0.05, 26, 26, 18)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--country", default="India")
    ap.add_argument("--sample", type=int, default=5000)
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
    print(f"sample: {len(sel):,} queries, {n_pairs:,} true pairs", flush=True)

    specs0 = CONFIGS[0][1]
    t0 = time.perf_counter()
    cfe = extract_features_parallel(c_names, c_addrs, specs0, args.workers)
    qfe = extract_features_parallel([q_names[i] for i in sel], [q_addrs[i] for i in sel],
                                   specs0, args.workers)
    print(f"features once in {time.perf_counter()-t0:.0f}s\n", flush=True)
    scale = len(q_ids) / len(sel)

    for tag, specs in CONFIGS:
        print(f"=== {tag} ===", flush=True)
        per_docs, per_scores, t_retr = [], [], 0.0
        for spec, cf, qf in zip(specs, cfe, qfe):
            index = ChannelIndex(spec, cf)
            Q = index._build(qf, prune_to_rarest=spec.query_terms)
            t1 = time.perf_counter()
            R = sp_matmul_topn(Q, index.matrix, top_n=DEPTH, sort=True, n_threads=THREADS)
            dt = time.perf_counter() - t1
            t_retr += dt
            docs = [R.indices[R.indptr[r]:R.indptr[r + 1]] for r in range(R.shape[0])]
            scores = [R.data[R.indptr[r]:R.indptr[r + 1]] for r in range(R.shape[0])]
            solo = sum(len(truth[j] & set(docs[j].tolist())) for j in range(len(sel)))
            print(f"  {spec.name:<11} vocab={len(index.vocab):>9,} "
                  f"feats/q={Q.nnz/max(1,Q.shape[0]):>5.1f} "
                  f"solo R@{DEPTH}={100*solo/n_pairs:>6.2f}%  {dt:>7.1f}s", flush=True)
            per_docs.append(docs); per_scores.append(scores)
            del index, Q, R

        union = sum(len(truth[j] & {int(d) for ch in per_docs for d in ch[j]})
                    for j in range(len(sel)))
        print(f"  UNION CEILING (all channels, depth {DEPTH}): {100*union/n_pairs:.2f}%")

        weights = [s.weight for s in specs]
        hits = {k: 0 for k in KS}
        for j in range(len(sel)):
            ranked = fuse_rrf([pd[j] for pd in per_docs], [ps[j] for ps in per_scores],
                              weights, max(KS))
            rset = set()
            prev = 0
            for k in KS:
                rset |= set(ranked[prev:k]); prev = k
                hits[k] += len(truth[j] & rset)
        print("  RRF fused recall: " + "  ".join(f"@{k}={100*hits[k]/n_pairs:.2f}%" for k in KS))
        print(f"  retrieval {t_retr:.1f}s on sample -> est {t_retr*scale/60:.0f}m for this partition\n",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
