"""
Reranker diagnostic: can a cheap learned reranker recover the broad pool's
recall at top-50?

The pool (RRF over 4 channels, depth 1000) holds 97.93% of true pairs but RRF
only surfaces 93.40% by top-50, because it ranks by rank-fusion and never
inspects the pair. This measures what logistic regression over 16 cheap pairwise
features recovers, against RRF as the baseline.

Evaluated on MULTIPLE countries (both labelled ones), not India alone, since the
two behave very differently: India carries nine non-Latin scripts and leans on the
address channel, while US is all-Latin with far denser trigrams.

Train/eval are split BY QUERY, never by pair, so no query's candidates appear on
both sides. Splitting by pair would leak and flatter the result.

Also measures the pool-depth question: reranking the top 200 of the pool instead
of all 1000 is 5x cheaper, so the report shows whether that costs recall.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from multiprocessing import Pool

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sparse_dot_topn import sp_matmul_topn

sys.path.insert(0, ".")
from src.blocking import (  # noqa: E402
    ChannelIndex, ChannelSpec, feats_address, feats_digit_keys,
    feats_name_ngrams, feats_name_tokens,
)
from src.candidates import extract_features_parallel, load_partition  # noqa: E402
from src.rerank import FEATURE_NAMES, build_form, pair_features  # noqa: E402

DEPTH = 1000
RRF_K0 = 20
THREADS = 24
KS = (10, 20, 30, 50, 100)
RERANK_DEPTHS = (200, 1000)


def channel_specs(df=0.01):
    """Config D plus the exact digit-key channel (reports/phase3_blocking.md)."""
    return (
        ChannelSpec("name_ngram", feats_name_ngrams, 1.0, DEPTH, df,    20),
        ChannelSpec("address",    feats_address,     1.0, DEPTH, df,    20),
        ChannelSpec("name_token", feats_name_tokens, 0.7, DEPTH, df,    14),
        ChannelSpec("digit_key",  feats_digit_keys,  1.0, DEPTH, 0.002, 16),
    )


_FORMS = {}


def _form_chunk(args):
    names, addrs = args
    return [build_form(n, a) for n, a in zip(names, addrs)]


def forms_parallel(names, addrs, workers, chunk=20000):
    tasks = [(names[i:i + chunk], addrs[i:i + chunk]) for i in range(0, len(names), chunk)]
    out = []
    with Pool(workers) as p:
        for part in p.imap(_form_chunk, tasks, chunksize=1):
            out.extend(part)
    return out


def run_country(store, gt, country, sample, workers, log=print):
    q_ids, q_names, q_addrs = load_partition(store, "train", "s1", country)
    c_ids, c_names, c_addrs, c_is_s3 = [], [], [], []
    for src in ("s2", "s3"):
        i, n, a = load_partition(store, "train", src, country)
        c_ids += i; c_names += n; c_addrs += a
        c_is_s3 += [1.0 if src == "s3" else 0.0] * len(i)
    crow = {e: i for i, e in enumerate(c_ids)}
    log(f"\n===== {country}: queries={len(q_ids):,} corpus={len(c_ids):,} =====")

    qpos = {e: i for i, e in enumerate(q_ids)}
    rng = random.Random(0)
    pick = [e for e in q_ids if e in gt and any(m in crow for m in gt[e])]
    rng.shuffle(pick); pick = pick[:sample]
    sel = [qpos[e] for e in pick]
    truth = {j: {crow[m] for m in gt[q_ids[qi]] if m in crow} for j, qi in enumerate(sel)}
    n_pairs = sum(len(v) for v in truth.values())
    log(f"sample: {len(sel):,} queries, {n_pairs:,} true pairs")

    specs = channel_specs()
    t0 = time.perf_counter()
    cfe = extract_features_parallel(c_names, c_addrs, specs, workers)
    qfe = extract_features_parallel([q_names[i] for i in sel], [q_addrs[i] for i in sel],
                                   specs, workers)
    log(f"channel features in {time.perf_counter()-t0:.0f}s")

    # ---- retrieve + RRF fuse into the broad pool -------------------------
    t0 = time.perf_counter()
    pool_rank = [dict() for _ in sel]     # doc -> rrf score
    pool_hits = [dict() for _ in sel]     # doc -> (channels hit, best rank)
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
    log(f"retrieval+fusion in {time.perf_counter()-t0:.0f}s")

    pool_recall = sum(len(truth[j] & set(pool_rank[j])) for j in range(len(sel)))
    log(f"POOL recall (union, depth {DEPTH}): {100*pool_recall/n_pairs:.2f}%")

    # ---- normalised forms, once per record ------------------------------
    needed = sorted({d for pr in pool_rank for d in pr})
    t0 = time.perf_counter()
    cforms_list = forms_parallel([c_names[d] for d in needed], [c_addrs[d] for d in needed], workers)
    cforms = dict(zip(needed, cforms_list))
    qforms = forms_parallel([q_names[i] for i in sel], [q_addrs[i] for i in sel], workers)
    log(f"record forms for {len(needed):,} candidates in {time.perf_counter()-t0:.0f}s")

    # ---- pair features --------------------------------------------------
    t0 = time.perf_counter()
    X, y, gid, docid = [], [], [], []
    for j in range(len(sel)):
        qf_ = qforms[j]
        for d, rrf in pool_rank[j].items():
            nh, br = pool_hits[j][d]
            X.append(pair_features(qf_, cforms[d], rrf, nh, br, c_is_s3[d]))
            y.append(1 if d in truth[j] else 0)
            gid.append(j); docid.append(d)
    X = np.asarray(X, dtype=np.float32); y = np.asarray(y, dtype=np.int8)
    gid = np.asarray(gid); docid = np.asarray(docid)
    t_feat = time.perf_counter() - t0
    log(f"pair features: {len(y):,} pairs in {t_feat:.0f}s "
        f"({len(y)/t_feat:,.0f} pairs/sec, {100*y.mean():.2f}% positive)")

    # ---- fit logistic regression, split BY QUERY ------------------------
    nq = len(sel); cut = int(0.6 * nq)
    tr = gid < cut; ev = ~tr
    scaler = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(scaler.transform(X[tr]), y[tr])
    score = clf.decision_function(scaler.transform(X))
    log(f"trained on {cut:,} queries, evaluating on {nq-cut:,}")

    log("\n  feature weights (standardised, descending |w|):")
    for name, w in sorted(zip(FEATURE_NAMES, clf.coef_[0]), key=lambda t: -abs(t[1])):
        log(f"    {name:<26}{w:>8.3f}")

    # ---- recall@K: RRF vs reranker, on eval queries only ---------------
    eval_qs = [j for j in range(cut, nq)]
    eval_pairs = sum(len(truth[j]) for j in eval_qs)
    log(f"\n  recall@K on {len(eval_qs):,} held-out queries ({eval_pairs:,} true pairs)")
    hdr = "    " + "ranking".ljust(30) + "".join(f"{f'@{k}':>9}" for k in KS)
    log(hdr); log("    " + "-" * (len(hdr) - 4))

    by_q = {}
    for j in eval_qs:
        m = gid == j
        by_q[j] = (docid[m], score[m], X[m, FEATURE_NAMES.index("rrf_score")])

    def recall_curve(key_fn, label):
        hits = {k: 0 for k in KS}
        for j in eval_qs:
            docs, sc, rrf = by_q[j]
            order = np.argsort(-key_fn(sc, rrf))
            ranked = docs[order]
            for k in KS:
                hits[k] += len(truth[j] & set(ranked[:k].tolist()))
        log("    " + label.ljust(30) + "".join(f"{100*hits[k]/eval_pairs:>8.2f} " for k in KS))
        return {k: 100 * hits[k] / eval_pairs for k in KS}

    rrf_curve = recall_curve(lambda sc, rrf: rrf, "RRF (baseline)")
    rr_curve = recall_curve(lambda sc, rrf: sc, "reranked (logistic reg)")

    # Reranking only the top-N of the pool is cheaper; does it cost recall?
    for depth in RERANK_DEPTHS:
        if depth >= DEPTH:
            continue
        hits = {k: 0 for k in KS}
        for j in eval_qs:
            docs, sc, rrf = by_q[j]
            keep = np.argsort(-rrf)[:depth]
            d2, s2 = docs[keep], sc[keep]
            ranked = d2[np.argsort(-s2)]
            for k in KS:
                hits[k] += len(truth[j] & set(ranked[:k].tolist()))
        log("    " + f"reranked (top-{depth} of pool)".ljust(30)
            + "".join(f"{100*hits[k]/eval_pairs:>8.2f} " for k in KS))

    pr = 100 * sum(len(truth[j] & set(by_q[j][0].tolist())) for j in eval_qs) / eval_pairs
    log(f"\n  pool ceiling on eval queries: {pr:.2f}%")
    log(f"  rerank@50 recovers {100*rr_curve[50]/pr:.1f}% of the pool ceiling "
        f"(RRF@50 recovered {100*rrf_curve[50]/pr:.1f}%)")
    return clf, scaler


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--countries", nargs="+", default=["India", "US"])
    ap.add_argument("--sample", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    gt = {}
    with open(args.gt, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.partition("\t")
            rest = rest.rstrip("\n")
            if rest:
                gt[s1] = rest.split(",")

    for country in args.countries:
        run_country(args.store, gt, country, args.sample, args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
