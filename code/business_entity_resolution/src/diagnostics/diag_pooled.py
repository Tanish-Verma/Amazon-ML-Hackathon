"""
Two questions, one experiment.

1. POOLING. The per-country diagnostic trained a separate model per country, but
   production cannot: France has no labels, so there is no France model to train.
   We need ONE model fitted on India+US and applied to all three. The per-country
   feature weights diverged sharply (name_trigram_jaccard +1.230 on India vs
   +0.042 on US), so pooling might cost real recall. This measures whether it does.

2. MODEL CLASS. Is logistic regression enough? Compared here against LightGBM
   (gradient-boosted trees) on identical features and identical splits. Trees can
   represent feature interactions and non-monotone effects that a linear model
   cannot -- e.g. "a shared digit key matters much more when the name is
   non-Latin" -- which is exactly the kind of structure the diverging per-country
   weights hint at.

Stage 1 (expensive: retrieval + pair features) caches to disk, so stage 2 can
compare any number of models without recomputing. That separation is what makes
answering the model question cheap.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sparse_dot_topn import sp_matmul_topn

sys.path.insert(0, ".")
from src.blocking import ChannelIndex  # noqa: E402
from src.candidates import extract_features_parallel, load_partition  # noqa: E402
from src.config import RRF_K0, THREADS, channel_specs  # noqa: E402
from src.rerank_vec import build_store_streaming  # noqa: E402
from src.rerank import FEATURE_NAMES, pair_features  # noqa: E402

DEPTH = 1000
KS = (10, 20, 30, 50, 100)


def build_cache(store, gt, country, sample, workers, cache_dir, log=print):
    path = os.path.join(cache_dir, f"pairs_{country}.npz")
    if os.path.isfile(path):
        log(f"  [{country}] cache hit: {path}")
        return path

    q_ids, q_names, q_addrs = load_partition(store, "train", "s1", country)
    c_ids, c_names, c_addrs, c_is_s3 = [], [], [], []
    for src in ("s2", "s3"):
        i, n, a = load_partition(store, "train", src, country)
        c_ids += i; c_names += n; c_addrs += a
        c_is_s3 += [1.0 if src == "s3" else 0.0] * len(i)
    crow = {e: i for i, e in enumerate(c_ids)}

    qpos = {e: i for i, e in enumerate(q_ids)}
    rng = random.Random(0)
    pick = [e for e in q_ids if e in gt and any(m in crow for m in gt[e])]
    rng.shuffle(pick); pick = pick[:sample]
    sel = [qpos[e] for e in pick]
    truth = {j: {crow[m] for m in gt[q_ids[qi]] if m in crow} for j, qi in enumerate(sel)}
    log(f"  [{country}] {len(sel):,} queries, {sum(len(v) for v in truth.values()):,} true pairs")

    specs = channel_specs()
    cfe = extract_features_parallel(c_names, c_addrs, specs, workers)
    qfe = extract_features_parallel([q_names[i] for i in sel], [q_addrs[i] for i in sel],
                                   specs, workers)

    pool_rank = [dict() for _ in sel]
    pool_hits = [dict() for _ in sel]
    for spec, cf, qf in zip(specs, cfe, qfe):
        index = ChannelIndex(spec, cf)
        Q = index._build(qf, prune_to_rarest=spec.query_terms)
        R = sp_matmul_topn(Q, index.matrix, top_n=DEPTH, sort=True, n_threads=THREADS)
        for r in range(R.shape[0]):
            a, b = R.indptr[r], R.indptr[r + 1]
            for rank, d in enumerate(R.indices[a:b].tolist()):
                pool_rank[r][d] = pool_rank[r].get(d, 0.0) + spec.weight / (RRF_K0 + rank)
                nh, br = pool_hits[r].get(d, (0, 10**6))
                pool_hits[r][d] = (nh + 1, min(br, rank))
        del index, Q, R
    del cfe, qfe

    needed = sorted({d for pr in pool_rank for d in pr})
    cforms = dict(zip(needed, forms_parallel([c_names[d] for d in needed],
                                            [c_addrs[d] for d in needed], workers)))
    qforms = forms_parallel([q_names[i] for i in sel], [q_addrs[i] for i in sel], workers)

    t0 = time.perf_counter()
    X, y, gid = [], [], []
    for j in range(len(sel)):
        qf_ = qforms[j]
        for d, rrf in pool_rank[j].items():
            nh, br = pool_hits[j][d]
            X.append(pair_features(qf_, cforms[d], rrf, nh, br, c_is_s3[d]))
            y.append(1 if d in truth[j] else 0)
            gid.append(j)
    X = np.asarray(X, dtype=np.float32)
    log(f"  [{country}] {len(y):,} pairs in {time.perf_counter()-t0:.0f}s")
    np.savez_compressed(path, X=X, y=np.asarray(y, dtype=np.int8),
                        gid=np.asarray(gid, dtype=np.int32),
                        n_queries=np.int32(len(sel)),
                        truth_sizes=np.asarray([len(truth[j]) for j in range(len(sel))],
                                               dtype=np.int32))
    return path


def recall_at_k(score, gid, y, n_queries, eval_mask_q, ks=KS):
    """Recall@K over held-out queries, ranking each query's pool by `score`."""
    hits = {k: 0 for k in ks}
    total = 0
    order_by_q = {}
    for j in np.unique(gid[eval_mask_q]):
        m = gid == j
        s, lab = score[m], y[m]
        total += int(lab.sum())
        top = np.argsort(-s)
        lab_sorted = lab[top]
        for k in ks:
            hits[k] += int(lab_sorted[:k].sum())
    return {k: 100 * hits[k] / total for k in ks}, total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--countries", nargs="+", default=["India", "US"])
    ap.add_argument("--sample", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--cache-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    gt = {}
    with open(args.gt, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.partition("\t")
            rest = rest.rstrip("\n")
            if rest:
                gt[s1] = rest.split(",")

    print("=== stage 1: build / load pair-feature caches ===", flush=True)
    data = {}
    for c in args.countries:
        p = build_cache(args.store, gt, c, args.sample, args.workers, args.cache_dir)
        d = np.load(p)
        data[c] = dict(X=d["X"], y=d["y"], gid=d["gid"], nq=int(d["n_queries"]))
        print(f"  [{c}] X={data[c]['X'].shape} positives={int(data[c]['y'].sum()):,}", flush=True)

    # 60/40 split BY QUERY within each country; pooled training uses both
    # countries' train halves, and each country is evaluated on its own held-out
    # half so per-country degradation is visible.
    tr_parts, ev = [], {}
    for c, d in data.items():
        cut = int(0.6 * d["nq"])
        tr = d["gid"] < cut
        tr_parts.append((d["X"][tr], d["y"][tr]))
        ev[c] = dict(X=d["X"][~tr], y=d["y"][~tr], gid=d["gid"][~tr])

    Xtr = np.vstack([p[0] for p in tr_parts])
    ytr = np.concatenate([p[1] for p in tr_parts])
    print(f"\npooled training set: {Xtr.shape}, {100*ytr.mean():.3f}% positive", flush=True)

    print("\n=== stage 2: models ===", flush=True)
    models = {}

    t0 = time.perf_counter()
    scaler = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(scaler.transform(Xtr), ytr)
    models["pooled logistic regression"] = (
        lambda X: lr.decision_function(scaler.transform(X)), time.perf_counter() - t0)

    try:
        import lightgbm as lgb
        t0 = time.perf_counter()
        gbm = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.1,
                                 min_child_samples=50, is_unbalance=True,
                                 n_jobs=args.workers, verbose=-1).fit(Xtr, ytr)
        models["pooled LightGBM (300 trees)"] = (
            lambda X: gbm.predict_proba(X)[:, 1], time.perf_counter() - t0)
    except ImportError:
        print("  (lightgbm unavailable)")

    hdr = "  " + "model / country".ljust(42) + "".join(f"{f'@{k}':>9}" for k in KS)
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for name, (fn, t_fit) in models.items():
        for c in args.countries:
            t1 = time.perf_counter()
            s = fn(ev[c]["X"])
            t_score = time.perf_counter() - t1
            r, tot = recall_at_k(s, ev[c]["gid"], ev[c]["y"], 0,
                                np.ones(len(ev[c]["gid"]), dtype=bool))
            rate = len(s) / t_score
            print("  " + f"{name} · {c}".ljust(42)
                  + "".join(f"{r[k]:>8.2f} " for k in KS)
                  + f"  [{rate/1e6:.2f}M pairs/s]", flush=True)
        print(f"  {'':42}  fit {t_fit:.1f}s")
    # Ceiling for reference
    for c in args.countries:
        tot = int(ev[c]["y"].sum())
        print(f"  pool ceiling {c}: {100.0:.2f}% of {tot:,} held-out true pairs "
              f"(all are in the pool by construction)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
