"""
Phase 3 orchestration: run the blocking channels per country, fuse them, cap to
top-K, and write `candidate_pairs.tsv`.

Parallelism. Feature extraction is the only Python-level hot loop -- it calls the
Phase 2 normaliser once per record, which measured 37k records/sec/core, so ~11
minutes single-threaded over the full corpus. It is embarrassingly parallel, so
it runs in a process pool. Retrieval itself is a sparse matmul and already runs
at C speed. Workers default to a modest fraction of the box rather than all 48,
since it is a shared machine; override with --workers.

The output of this stage is the FINAL candidate set -- after top-K capping --
because the problem statement defines candidate_pairs.tsv as exactly what the
matcher runs inference over, not an earlier pass that gets filtered later.
"""

from __future__ import annotations

import os
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq

from src.blocking import DEFAULT_CHANNELS, ChannelIndex, ChannelSpec

# Set in each worker by the initialiser; avoids re-pickling the extractor.
_WORKER_SPECS: Tuple[ChannelSpec, ...] = ()

# Retrieval state shared with worker processes. Populated in the PARENT before
# the pool is created, so on Linux (fork) the children inherit the sparse
# matrices copy-on-write instead of having them pickled per task -- the matrices
# are gigabytes, so pickling them would cost more than the work itself.
_SHARED: dict = {}


def _retrieve_range(rng):
    """Score one contiguous block of queries and return its top-K per row.

    scipy's sparse matmul is single-threaded C, so the only way to use the other
    47 cores is to fan chunks out across processes. Returns compact int32/float32
    arrays so what crosses the pipe back is O(chunk x K), not the score matrix.
    """
    a, b = rng
    C_T, Q, k = _SHARED["ct"], _SHARED["q"], _SHARED["k"]
    S = (Q[a:b] @ C_T).tocsr()
    docs, scores = [], []
    for r in range(S.shape[0]):
        i, j = S.indptr[r], S.indptr[r + 1]
        ix, vl = S.indices[i:j], S.data[i:j]
        if ix.size > k:
            sel = np.argpartition(-vl, k)[:k]
            ix, vl = ix[sel], vl[sel]
        docs.append(ix.astype(np.int32))
        scores.append(vl.astype(np.float32))
    return a, docs, scores


def retrieve_parallel(index, query_feats, workers: int, chunk: int = 400):
    """Per-query (docs, scores) for one channel, fanned across processes."""
    Q = index._build(query_feats, prune_to_rarest=index.spec.query_terms)
    n = Q.shape[0]
    out_docs: List[np.ndarray] = [None] * n       # type: ignore[list-item]
    out_scores: List[np.ndarray] = [None] * n     # type: ignore[list-item]
    ranges = [(i, min(i + chunk, n)) for i in range(0, n, chunk)]

    _SHARED["ct"], _SHARED["q"], _SHARED["k"] = index.matrix, Q, index.spec.top_k
    try:
        if workers <= 1:
            results = map(_retrieve_range, ranges)
            for a, d, sc in results:
                out_docs[a:a + len(d)] = d
                out_scores[a:a + len(sc)] = sc
        else:
            with Pool(workers) as pool:        # fork: inherits _SHARED
                for a, d, sc in pool.imap_unordered(_retrieve_range, ranges, chunksize=4):
                    out_docs[a:a + len(d)] = d
                    out_scores[a:a + len(sc)] = sc
    finally:
        _SHARED.clear()
    return out_docs, out_scores


# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------
# The first full run fused channels by summing raw cosines and scored 13 points
# WORSE than its best single channel (66.51% vs 79.95% on India). Character
# trigram cosines are far larger in magnitude than address-token cosines, so name
# candidates outranked and evicted address candidates during the final cap --
# catastrophic for transliterated records, whose ONLY route is the address.
# Reciprocal rank fusion is scale-free by construction, which is exactly the
# property the raw sum lacked.
RRF_K0 = 20


def fuse_rrf(per_channel_docs, per_channel_scores, weights, final_k: int):
    """Reciprocal rank fusion -> the final capped candidate list (doc indices)."""
    acc: dict = {}
    for docs, scores, w in zip(per_channel_docs, per_channel_scores, weights):
        if docs is None or docs.size == 0:
            continue
        order = np.argsort(-scores)
        for rank, i in enumerate(order):
            d = int(docs[i])
            acc[d] = acc.get(d, 0.0) + w / (RRF_K0 + rank)
    if not acc:
        return []
    if len(acc) <= final_k:
        return sorted(acc, key=lambda d: -acc[d])
    ds = np.fromiter(acc.keys(), dtype=np.int64, count=len(acc))
    sc = np.fromiter(acc.values(), dtype=np.float32, count=len(acc))
    keep = np.argpartition(-sc, final_k)[:final_k]
    return ds[keep[np.argsort(-sc[keep])]].tolist()


def _init_worker(specs):
    global _WORKER_SPECS
    _WORKER_SPECS = specs


def _extract_chunk(args):
    """Extract every channel's features for a chunk of records, in one pass.

    One pass over the chunk rather than one per channel: normalisation is the
    expensive part and all three channels share it.
    """
    names, addrs = args
    out = [[] for _ in _WORKER_SPECS]
    for name, addr in zip(names, addrs):
        for i, spec in enumerate(_WORKER_SPECS):
            out[i].append(spec.extract(name, addr))
    return out


def extract_features_parallel(names: Sequence[str], addrs: Sequence[str],
                             specs: Tuple[ChannelSpec, ...], workers: int,
                             chunk: int = 20_000) -> List[List[List[str]]]:
    """Returns per-channel lists of per-record feature lists."""
    tasks = [(names[i:i + chunk], addrs[i:i + chunk]) for i in range(0, len(names), chunk)]
    per_channel: List[List[List[str]]] = [[] for _ in specs]
    if workers <= 1:
        _init_worker(specs)
        results = map(_extract_chunk, tasks)
    else:
        pool = Pool(workers, initializer=_init_worker, initargs=(specs,))
        results = pool.imap(_extract_chunk, tasks, chunksize=1)
    for res in results:
        for i, part in enumerate(res):
            per_channel[i].extend(part)
    if workers > 1:
        pool.close(); pool.join()
    return per_channel


def load_partition(store: str, split: str, source: str, country: str):
    """Ids/names/addresses for one (split, source, country) parquet partition."""
    safe = "".join(ch if ch.isalnum() else "_" for ch in country)
    path = os.path.join(store, f"{split}_{source}_{safe}.parquet")
    if not os.path.isfile(path):
        return [], [], []
    t = pq.read_table(path)
    return (t.column("entity_id").to_pylist(),
            t.column("business_name").to_pylist(),
            t.column("business_address").to_pylist())


def discover_countries(store: str, split: str) -> List[str]:
    """Country labels present, read from the data rather than hardcoded."""
    out = set()
    for fn in os.listdir(store):
        if fn.startswith(f"{split}_s1_") and fn.endswith(".parquet"):
            out.add(fn[len(f"{split}_s1_"):-len(".parquet")])
    return sorted(out)


def block_country(store: str, split: str, country: str, *, final_k: int,
                  specs: Tuple[ChannelSpec, ...], workers: int,
                  log=print) -> Dict[str, List[str]]:
    """Blocking for one country partition. Returns {s1_id: [candidate ids]}."""
    q_ids, q_names, q_addrs = load_partition(store, split, "s1", country)
    if not q_ids:
        return {}

    c_ids: List[str] = []
    c_names: List[str] = []
    c_addrs: List[str] = []
    for src in ("s2", "s3"):
        i, n, a = load_partition(store, split, src, country)
        c_ids += i; c_names += n; c_addrs += a

    log(f"  [{country}] queries={len(q_ids):,} corpus={len(c_ids):,}")

    t0 = time.perf_counter()
    corpus_feats = extract_features_parallel(c_names, c_addrs, specs, workers)
    query_feats = extract_features_parallel(q_names, q_addrs, specs, workers)
    log(f"  [{country}] features extracted in {time.perf_counter()-t0:.1f}s "
        f"({workers} workers)")

    per_docs, per_scores = [], []
    for spec, cf, qf in zip(specs, corpus_feats, query_feats):
        t1 = time.perf_counter()
        index = ChannelIndex(spec, cf)
        d, sc = retrieve_parallel(index, qf, workers)
        per_docs.append(d); per_scores.append(sc)
        log(f"  [{country}] channel {spec.name:<11} vocab={len(index.vocab):,} "
            f"pruned={index.n_pruned:,} in {time.perf_counter()-t1:.1f}s")
        del index

    weights = [s.weight for s in specs]
    out: Dict[str, List[str]] = {}
    for qi, qid in enumerate(q_ids):
        rows = fuse_rrf([pd[qi] for pd in per_docs],
                        [ps[qi] for ps in per_scores], weights, final_k)
        out[qid] = [c_ids[d] for d in rows]
    return out
