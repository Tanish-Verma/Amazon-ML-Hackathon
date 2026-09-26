"""
Fusion experiment: why does combining three channels score WORSE than the best
channel alone?

Measured on India (reports/phase3_blocking.md + logs/diag_india.log):

    address channel alone, top-30 .......... 79.95% recall
    all three channels fused, K=30 ......... 66.51% recall   <-- fusion LOSES 13 pts

That is a fusion defect, not a retrieval one. The cause is score-scale mismatch:
the fused score was a weighted SUM of raw per-channel cosines, but character
trigram cosines are naturally much larger than address-token cosines, so name
candidates systematically outrank address candidates and evict them during the
final top-K cap. For a transliterated Indic record the name channel returns
nothing but noise, and that noise was displacing the address hits that are the
only way to find those matches.

This compares scale-free fusion strategies against the buggy one:

  sum_raw   -- current: weighted sum of raw cosines (the baseline to beat)
  rrf       -- reciprocal rank fusion, sum of w/(k0 + rank). Scale-free by
               construction, which is exactly the property the raw sum lacked.
  znorm     -- per-query per-channel score standardisation, then weighted sum.
  quota     -- round-robin interleave of the channels' ranked lists, which
               *guarantees* every channel places candidates regardless of scale.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, ".")
from src.blocking import DEFAULT_CHANNELS, ChannelIndex  # noqa: E402
from src.candidates import extract_features_parallel, load_partition  # noqa: E402

KS = (10, 20, 30, 50, 100)
RETRIEVE = 120          # per-channel depth fed into fusion
RRF_K0 = 20


def fuse_sum_raw(per_channel, weights):
    acc = defaultdict(float)
    for (docs, scores), w in zip(per_channel, weights):
        for d, s in zip(docs, scores):
            acc[d] += w * s
    return acc


def fuse_rrf(per_channel, weights):
    acc = defaultdict(float)
    for (docs, scores), w in zip(per_channel, weights):
        order = np.argsort(-scores)
        for rank, i in enumerate(order):
            acc[int(docs[i])] += w / (RRF_K0 + rank)
    return acc


def fuse_znorm(per_channel, weights):
    acc = defaultdict(float)
    for (docs, scores), w in zip(per_channel, weights):
        if scores.size == 0:
            continue
        mu, sd = scores.mean(), scores.std()
        z = (scores - mu) / sd if sd > 1e-9 else np.zeros_like(scores)
        for d, s in zip(docs, z):
            acc[int(d)] += w * float(s)
    return acc


def fuse_quota(per_channel, weights):
    """Round-robin interleave: every channel is guaranteed slots."""
    lists = []
    for docs, scores in per_channel:
        order = np.argsort(-scores)
        lists.append([int(docs[i]) for i in order])
    acc, seen, pos, rank = {}, set(), [0] * len(lists), 0
    while any(pos[i] < len(lists[i]) for i in range(len(lists))):
        for i in range(len(lists)):
            while pos[i] < len(lists[i]) and lists[i][pos[i]] in seen:
                pos[i] += 1
            if pos[i] < len(lists[i]):
                d = lists[i][pos[i]]; pos[i] += 1
                seen.add(d); rank += 1
                acc[d] = -rank      # higher = better
    return acc


STRATEGIES = {"sum_raw": fuse_sum_raw, "rrf": fuse_rrf,
              "znorm": fuse_znorm, "quota": fuse_quota}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--country", default="India")
    ap.add_argument("--sample", type=int, default=8000)
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
    cand = [e for e in q_ids if e in gt and any(m in crow for m in gt[e])]
    rng.shuffle(cand); cand = cand[:args.sample]
    sel = [qpos[e] for e in cand]
    truth = {qi: {crow[m] for m in gt[q_ids[qi]] if m in crow} for qi in sel}
    n_pairs = sum(len(v) for v in truth.values())
    print(f"sample: {len(sel):,} queries, {n_pairs:,} true pairs\n", flush=True)

    t0 = time.perf_counter()
    cfe = extract_features_parallel(c_names, c_addrs, DEFAULT_CHANNELS, args.workers)
    qfe = extract_features_parallel([q_names[i] for i in sel], [q_addrs[i] for i in sel],
                                   DEFAULT_CHANNELS, args.workers)
    print(f"features in {time.perf_counter()-t0:.0f}s", flush=True)

    # Retrieve RETRIEVE-deep per channel once, then replay every fusion strategy.
    retrieved = [[] for _ in sel]
    for spec, cf, qf in zip(DEFAULT_CHANNELS, cfe, qfe):
        idx = ChannelIndex(spec, cf)
        saved = spec.top_k
        object.__setattr__(idx.spec, "top_k", RETRIEVE) if False else None
        C_T, k = idx.matrix, RETRIEVE
        Q = idx._build(qf, prune_to_rarest=spec.query_terms)
        for start in range(0, Q.shape[0], 500):
            S = (Q[start:start + 500] @ C_T).tocsr()
            for r in range(S.shape[0]):
                a, b = S.indptr[r], S.indptr[r + 1]
                ix, vl = S.indices[a:b], S.data[a:b]
                if ix.size > k:
                    s = np.argpartition(-vl, k)[:k]
                    ix, vl = ix[s], vl[s]
                retrieved[start + r].append((ix, vl))
        print(f"  retrieved channel {spec.name}", flush=True)
        del idx, Q

    weights = [s.weight for s in DEFAULT_CHANNELS]
    # Ceiling: union of all channels at RETRIEVE depth.
    union_hit = sum(len(truth[qi] & {int(d) for ch in retrieved[j] for d in ch[0]})
                    for j, qi in enumerate(sel))
    print(f"\n  UNION CEILING at depth {RETRIEVE}/channel: {100*union_hit/n_pairs:.2f}%\n")

    hdr = "  " + "strategy".ljust(12) + "".join(f"{f'@{k}':>9}" for k in KS)
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for name, fn in STRATEGIES.items():
        hits = {k: 0 for k in KS}
        for j, qi in enumerate(sel):
            acc = fn(retrieved[j], weights)
            if not acc:
                continue
            order = sorted(acc, key=lambda d: -acc[d])
            for k in KS:
                hits[k] += len(truth[qi] & set(order[:k]))
        print("  " + name.ljust(12) + "".join(f"{100*hits[k]/n_pairs:>8.2f} " for k in KS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
