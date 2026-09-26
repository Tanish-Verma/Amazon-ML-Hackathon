"""
Production blocking run: retrieve -> RRF -> rerank -> top-K -> candidate_pairs.tsv

Pipeline (settled by the measurements in reports/phase3_blocking.md):

    4 cheap channels, depth 1000   name trigrams | address | name tokens | digit keys
              v
    RRF fusion  ->  broad pool (~1000/query, 98.8-99.4% of true pairs)
              v
    learned reranker over 16 cheap pairwise features (vectorised)
              v
    top-K  ->  candidate_pairs.tsv        (97.4% India / 99.3% US at K=50)

Two things this file exists to get right, both learned by failing at them:

1. STREAMING BY QUERY. 1.73M test queries x 1000 candidates is ~1.7 BILLION pool
   entries. Channel indexes and the corpus store are built once per country, then
   queries are walked in batches: retrieve -> fuse -> rerank -> emit -> discard.

2. NO FORKED WORKERS IN THE FEATURE STAGE. The first version held per-record
   Python objects (frozensets of strings) for the whole corpus -- 40 GB on India --
   and forked workers to score pairs. CPython writes refcounts into every object a
   worker touches, so those pages were COPIED, not shared: 6 workers added 68 GB
   and exhausted a 125 GB machine plus swap, twice. Lowering the worker count did
   not fix it and could not. Features are now computed by vectorised scipy and
   multithreaded rapidfuzz over binary sparse matrices (~4 GB, no per-element
   refcounts), so there is nothing to fork. See src/rerank_vec.py.

The written top-K is deliberately the FINAL candidate set the matcher will score,
which is what the problem statement defines candidate_pairs.tsv to be.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import numpy as np
import psutil
from sparse_dot_topn import sp_matmul_topn

from src.blocking import ChannelIndex
from src.candidates import discover_countries, extract_features_parallel, load_partition
from src.io_utils import assert_free_space, write_id_list_tsv
from src.rerank_vec import build_store_streaming, pair_features_vec

from src.config import DEPTH, RRF_K0, THREADS  # noqa: E402  (config, not constants)

FEAT_CHUNK = 1_000_000     # pairs per vectorised feature sub-chunk

def _mem_guard(stage: str, need_gb: float, log=print) -> None:
    """Abort with a clear message rather than exhausting a shared machine.

    Kept because the earlier design OOM'd this box twice. The sparse rewrite
    removed the cause, but a cheap tripwire on a shared machine is still right.
    """
    avail = psutil.virtual_memory().available / 1e9
    swap = psutil.swap_memory().used / 1e9
    log(f"    [mem] {stage}: {avail:.0f} GB available, swap {swap:.1f} GB")
    if avail < need_gb:
        raise MemoryError(
            f"Refusing to continue at {stage}: {avail:.0f} GB available, need "
            f"~{need_gb:.0f} GB. Lower --batch and re-run. Nothing was deleted.")



def _accumulate_pools(indexes, specs, qfe, n_rows: int, n_corpus: int):
    """Retrieve every channel and aggregate into flat numpy arrays.

    Replaces a list of per-query dicts. For a 20k batch those dicts held ~20M
    Python tuples (3-4 GB) and had to be walked in Python before scoring. Here
    each channel's hits go straight into int32/float32 arrays, and the
    (query, candidate) aggregation is a sort plus two bincounts.

    Returns (qi, ci, rrf, n_hit, best_rank) sorted by qi then ci, which is also
    exactly the grouping the feature stage wants.
    """
    q_parts, c_parts, w_parts, r_parts = [], [], [], []
    for spec, index, qf in zip(specs, indexes, qfe):
        Q = index._build(qf, prune_to_rarest=spec.query_terms)
        R = sp_matmul_topn(Q, index.matrix, top_n=DEPTH, sort=True, n_threads=THREADS)
        counts = np.diff(R.indptr)
        q_parts.append(np.repeat(np.arange(R.shape[0], dtype=np.int64), counts))
        c_parts.append(R.indices.astype(np.int64))
        # sort=True means indices are already in descending-score order per row,
        # so position within the row IS the channel rank.
        rank = np.arange(len(R.indices), dtype=np.int64) - np.repeat(R.indptr[:-1], counts)
        r_parts.append(rank)
        w_parts.append((spec.weight / (RRF_K0 + rank)).astype(np.float32))
        del Q, R

    qi = np.concatenate(q_parts); ci = np.concatenate(c_parts)
    rrf_w = np.concatenate(w_parts); rank = np.concatenate(r_parts)
    del q_parts, c_parts, w_parts, r_parts

    # One key per (query, candidate); int64 is ample (20k x 4.7M << 2^63).
    key = qi * np.int64(n_corpus) + ci
    order = np.lexsort((rank, key))          # ties broken by best rank first
    key_s = key[order]
    first = np.empty(len(key_s), dtype=bool)
    first[0] = True
    np.not_equal(key_s[1:], key_s[:-1], out=first[1:])
    gid = np.cumsum(first) - 1
    n_groups = int(gid[-1]) + 1

    rrf_sum = np.bincount(gid, weights=rrf_w[order], minlength=n_groups).astype(np.float32)
    n_hit = np.bincount(gid, minlength=n_groups).astype(np.float32)
    idx0 = np.flatnonzero(first)
    best_rank = rank[order][idx0].astype(np.float32)
    uniq = key_s[idx0]
    return (uniq // n_corpus).astype(np.int64), (uniq % n_corpus).astype(np.int64), \
        rrf_sum, n_hit, best_rank


def _score_pools(qstore, cstore, qi, ci, rrf, nh, br, is_s3, model, scaler,
                 top_k, n_rows, workers):
    """Vectorised features -> linear score -> top-K per query."""
    if len(qi) == 0:
        return [np.empty(0, dtype=np.int64) for _ in range(n_rows)]

    coef = model.coef_[0].astype(np.float32)
    intercept = np.float32(model.intercept_[0])
    mean = scaler.mean_.astype(np.float32)
    sd = scaler.scale_.astype(np.float32)

    scores = np.empty(len(qi), dtype=np.float32)
    # Sub-chunk on query boundaries so each chunk stays whole groups -- the
    # feature stage relies on qi being grouped.
    bounds = [0]
    step = FEAT_CHUNK
    while bounds[-1] < len(qi):
        nxt = min(bounds[-1] + step, len(qi))
        while nxt < len(qi) and qi[nxt] == qi[nxt - 1]:
            nxt += 1
        bounds.append(nxt)
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        X = pair_features_vec(qstore, cstore, qi[lo:hi], ci[lo:hi],
                              rrf[lo:hi], nh[lo:hi], br[lo:hi],
                              is_s3[ci[lo:hi]], workers=workers)
        scores[lo:hi] = ((X - mean) / sd) @ coef + intercept
        del X

    order = np.lexsort((-scores, qi))
    qi_sorted = qi[order]
    qs = np.searchsorted(qi_sorted, np.arange(n_rows), side="left")
    qe = np.searchsorted(qi_sorted, np.arange(n_rows), side="right")
    return [ci[order[qs[r]:qe[r]][:top_k]] for r in range(n_rows)]


def run_country(store, split, country, model, scaler, specs, *, top_k, workers,
                batch=20000, max_queries=0, log=print):
    """Yield (s1_id, [candidate ids]) for one country, streaming by query batch."""
    q_ids, q_names, q_addrs = load_partition(store, split, "s1", country)
    if not q_ids:
        return
    if max_queries:
        q_ids, q_names, q_addrs = (q_ids[:max_queries], q_names[:max_queries],
                                   q_addrs[:max_queries])
    c_ids, c_names, c_addrs, c_is_s3 = [], [], [], []
    for src in ("s2", "s3"):
        i, n, a = load_partition(store, split, src, country)
        c_ids += i; c_names += n; c_addrs += a
        c_is_s3 += [1.0 if src == "s3" else 0.0] * len(i)
    c_is_s3 = np.asarray(c_is_s3, dtype=np.float32)
    log(f"  [{country}] queries={len(q_ids):,} corpus={len(c_ids):,}")

    t0 = time.perf_counter()
    corpus_feats = extract_features_parallel(c_names, c_addrs, specs, workers)
    indexes = [ChannelIndex(spec, cf) for spec, cf in zip(specs, corpus_feats)]
    del corpus_feats
    log(f"  [{country}] channel indexes in {time.perf_counter()-t0:.0f}s")

    _mem_guard(f"{country} before corpus store", 20.0, log)
    t0 = time.perf_counter()
    cstore = build_store_streaming(c_names, c_addrs, workers, log=log)
    log(f"  [{country}] corpus sparse store in {time.perf_counter()-t0:.0f}s")
    _mem_guard(f"{country} after corpus store", 15.0, log)

    t_retr = t_rank = 0.0
    for start in range(0, len(q_ids), batch):
        end = min(start + batch, len(q_ids))
        bn, ba = q_names[start:end], q_addrs[start:end]

        t1 = time.perf_counter()
        qfe = extract_features_parallel(bn, ba, specs, workers)
        qi, ci, rrf, nh, br = _accumulate_pools(indexes, specs, qfe, end - start,
                                                len(c_ids))
        del qfe
        t_retr += time.perf_counter() - t1

        t1 = time.perf_counter()
        qstore = build_store_streaming(bn, ba, workers, vocabs=cstore.vocabs)
        ranked = _score_pools(qstore, cstore, qi, ci, rrf, nh, br, c_is_s3,
                              model, scaler, top_k, end - start, workers)
        t_rank += time.perf_counter() - t1

        for r, docs in enumerate(ranked):
            yield q_ids[start + r], [c_ids[d] for d in docs.tolist()]
        del qi, ci, rrf, nh, br, qstore, ranked
        log(f"  [{country}] {end:,}/{len(q_ids):,}  retr {t_retr:.0f}s  rank {t_rank:.0f}s  "
            f"avail {psutil.virtual_memory().available/1e9:.0f}GB", flush=True)

    del indexes, cstore


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", required=True, help="pickle with model/scaler/specs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--batch", type=int, default=20000)
    ap.add_argument("--countries", nargs="+", default=None)
    ap.add_argument("--max-queries", type=int, default=0,
                    help="cap queries per country (smoke testing only; 0 = all)")
    args = ap.parse_args()

    with open(args.model, "rb") as f:
        bundle = pickle.load(f)
    model, scaler, specs = bundle["model"], bundle["scaler"], bundle["specs"]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    assert_free_space(os.path.dirname(args.out) or ".", need_gb=3.0)

    countries = args.countries or discover_countries(args.store, args.split)
    print(f"split={args.split} countries={countries} K={args.top_k} "
          f"workers={args.workers} batch={args.batch}")
    t0 = time.time()

    def rows():
        for c in countries:
            yield from run_country(args.store, args.split, c, model, scaler, specs,
                                   top_k=args.top_k, workers=args.workers,
                                   batch=args.batch,
                                   max_queries=args.max_queries)

    n = write_id_list_tsv(args.out, ["source1_entity_id", "candidate_entity_ids"], rows())
    print(f"\nwrote {n:,} rows to {args.out} in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
