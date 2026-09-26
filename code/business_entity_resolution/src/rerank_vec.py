"""
Vectorised reranker feature computation.

Replaces the object-per-record design in src/rerank.py, which could not survive
production scale. Measured on India (4.72M corpus records): the Python
`frozenset`-based record forms cost **40 GB**, and because CPython writes
refcounts into every object a worker touches, forked workers COPY those pages
rather than sharing them -- 6 workers consumed a further 68 GB and exhausted a
125 GB machine plus its swap, twice.

The insight that fixes it: every set-based feature is an intersection count, and
intersection counts over binary sparse matrices are a vectorised scipy operation.

    |A n B|  ==  (Q[qrows].multiply(C[crows])).sum(axis=1)

with |A| and |B| precomputed as row nnz. So the corpus store becomes five binary
CSR matrices plus two plain string lists (only Jaro-Winkler needs real text):

    frozensets of strings  ~40 GB      ->     binary CSR + strings  ~4 GB

Numpy buffers carry no per-element refcounts, so nothing is copied on fork. And
because the arithmetic is now vectorised and rapidfuzz.process.cpdist is itself
multithreaded C++, the feature stage needs no worker processes at all -- which is
what actually removes the failure mode rather than tuning around it.

Feature order is identical to src.rerank.FEATURE_NAMES, so a model trained on the
old implementation applies unchanged. tests/test_rerank_parity.py asserts that.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from src.normalize import char_ngrams, clean_text, extract_numeric_keys, tokenize

FIELDS = ("name_tri", "name_tok", "addr_tri", "addr_tok", "digits")


def record_fields(name: str, addr: str) -> Tuple[str, str, List[str], List[str],
                                                 List[str], List[str], List[str]]:
    """Cleaned strings plus the five feature families for one record."""
    nc = clean_text(name)
    ac = clean_text(addr) if addr and addr.strip() else ""
    return (nc, ac,
            char_ngrams(nc, 3), tokenize(nc),
            char_ngrams(ac, 3), tokenize(ac),
            sorted(extract_numeric_keys(addr)) if addr else [])


class SparseStore:
    """Binary CSR matrices for the five fields, plus cleaned text."""

    __slots__ = ("mats", "counts", "name_clean", "addr_clean", "vocabs", "n")

    def __init__(self, rows: Sequence[Tuple], vocabs: Dict[str, Dict[str, int]] | None = None):
        self.n = len(rows)
        self.name_clean = [r[0] for r in rows]
        self.addr_clean = [r[1] for r in rows]
        per_field = [[r[2 + i] for r in rows] for i in range(5)]

        build_vocab = vocabs is None
        self.vocabs = {} if build_vocab else vocabs
        self.mats, self.counts = {}, {}
        for fname, lists in zip(FIELDS, per_field):
            if build_vocab:
                vocab: Dict[str, int] = {}
                for fs in lists:
                    for f in fs:
                        if f not in vocab:
                            vocab[f] = len(vocab)
                self.vocabs[fname] = vocab
            else:
                vocab = self.vocabs[fname]
            indptr = np.zeros(self.n + 1, dtype=np.int64)
            indices: List[int] = []
            # `counts` must be the TRUE number of distinct features, not the
            # number that happen to be in this vocabulary. A query store shares
            # the corpus vocabulary, so any query feature the corpus never uses
            # is absent from the matrix -- harmless for the intersection (it
            # cannot match anything) but fatal for |A| in the Jaccard
            # denominator, which would shrink and inflate every similarity.
            true_counts = np.zeros(self.n, dtype=np.float32)
            for i, fs in enumerate(lists):
                uniq = set(fs)
                true_counts[i] = len(uniq)
                indices.extend(vocab[f] for f in uniq if f in vocab)
                indptr[i + 1] = len(indices)
            idx = np.asarray(indices, dtype=np.int32)
            mat = sp.csr_matrix(
                (np.ones(len(idx), dtype=np.float32), idx, indptr),
                shape=(self.n, max(1, len(vocab))))
            self.mats[fname] = mat
            self.counts[fname] = true_counts

    def nbytes(self) -> float:
        total = sum(m.data.nbytes + m.indices.nbytes + m.indptr.nbytes
                    for m in self.mats.values())
        return total / 1e9


def _rowwise_inter(Qm: sp.csr_matrix, Cm: sp.csr_matrix,
                   qrows: np.ndarray, crows: np.ndarray,
                   groups: np.ndarray | None = None) -> np.ndarray:
    """|A n B| for aligned row pairs, vectorised.

    NOTE on a rejected optimisation. ``qrows`` holds ~1M entries with only ~20k
    distinct values (each query repeats once per candidate), so this materialises
    each query row many times over. Walking the query groups and broadcasting one
    query row per group removes that duplication -- and MEASURED 10% SLOWER
    (194s vs 176s per 20k batch on France), because it replaces one large
    vectorised gather with ~100k small scipy calls whose per-call overhead
    dominates. scipy's fancy-index gather is fast C; the redundancy is cheaper
    than the Python loop. Keeping the simple form.

    ``groups`` is accepted and ignored, so callers need not change.
    """
    if Qm.shape[1] == 0 or Cm.shape[1] == 0 or len(qrows) == 0:
        return np.zeros(len(qrows), dtype=np.float32)
    prod = Qm[qrows].multiply(Cm[crows])
    return np.asarray(prod.sum(axis=1)).ravel().astype(np.float32)


def _query_groups(qrows: np.ndarray) -> np.ndarray:
    """Start/end boundaries of each run of equal values in a grouped array."""
    if len(qrows) == 0:
        return np.empty((0, 2), dtype=np.int64)
    change = np.flatnonzero(np.diff(qrows)) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(qrows)]))
    return np.stack((starts, ends), axis=1)


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.zeros_like(num, dtype=np.float32)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def pair_features_vec(qstore: SparseStore, cstore: SparseStore,
                      qrows: np.ndarray, crows: np.ndarray,
                      rrf: np.ndarray, n_hit: np.ndarray, best_rank: np.ndarray,
                      is_s3: np.ndarray, workers: int = 8) -> np.ndarray:
    """Feature matrix for aligned (query row, corpus row) pairs.

    Columns follow src.rerank.FEATURE_NAMES exactly.
    """
    n = len(qrows)
    X = np.zeros((n, 16), dtype=np.float32)

    # qrows arrives grouped by query (the caller sorts), so the group boundaries
    # are computed once and reused across all five fields.
    groups = _query_groups(qrows)
    inter, qc, cc = {}, {}, {}
    for f in FIELDS:
        inter[f] = _rowwise_inter(qstore.mats[f], cstore.mats[f], qrows, crows, groups)
        qc[f] = qstore.counts[f][qrows]
        cc[f] = cstore.counts[f][crows]

    def jac(f):
        return _safe_div(inter[f], qc[f] + cc[f] - inter[f])

    X[:, 0] = jac("name_tri")
    X[:, 1] = jac("name_tok")
    X[:, 2] = _safe_div(inter["name_tok"], np.minimum(qc["name_tok"], cc["name_tok"]))

    qn = [qstore.name_clean[i] for i in qrows]
    cn = [cstore.name_clean[i] for i in crows]
    qa = [qstore.addr_clean[i] for i in qrows]
    ca = [cstore.addr_clean[i] for i in crows]

    X[:, 3] = process.cpdist(qn, cn, scorer=JaroWinkler.similarity,
                             workers=workers, dtype=np.float32)
    X[:, 4] = process.cpdist(qn, cn, scorer=fuzz.token_set_ratio,
                             workers=workers, dtype=np.float32) / 100.0

    lq = np.fromiter((len(s) for s in qn), dtype=np.float32, count=n)
    lc = np.fromiter((len(s) for s in cn), dtype=np.float32, count=n)
    X[:, 5] = _safe_div(np.minimum(lq, lc), np.maximum(lq, lc))

    X[:, 6] = jac("addr_tri")
    X[:, 7] = jac("addr_tok")

    # Original semantics: 0 when either address is empty, not a JW of "".
    jw_a = process.cpdist(qa, ca, scorer=JaroWinkler.similarity,
                          workers=workers, dtype=np.float32)
    la = np.fromiter((len(s) for s in qa), dtype=np.float32, count=n)
    lb = np.fromiter((len(s) for s in ca), dtype=np.float32, count=n)
    both_addr = (la > 0) & (lb > 0)
    X[:, 8] = np.where(both_addr, jw_a, 0.0)

    X[:, 9] = jac("digits")
    X[:, 10] = inter["digits"]
    X[:, 11] = np.where(both_addr, 0.0, 1.0)

    X[:, 12] = rrf
    X[:, 13] = n_hit
    X[:, 14] = best_rank
    X[:, 15] = is_s3
    return X


# --------------------------------------------------------------------------
# Streaming builder
# --------------------------------------------------------------------------
# Building a SparseStore from a materialised list of per-record field lists would
# recreate the very problem this module exists to solve: 4.72M records x five
# lists of strings is tens of GB held at once. So the builder consumes records in
# chunks, folds each chunk into the growing CSR arrays, and drops it. Peak memory
# is then the final matrices (~4 GB) rather than the intermediate Python objects.

_EXTRACT_CHUNK_SIZE = 20000


def _extract_chunk(args):
    names, addrs = args
    return [record_fields(n, a) for n, a in zip(names, addrs)]


class StreamingStoreBuilder:
    """Accumulate records into a SparseStore without holding them all."""

    def __init__(self, vocabs: Dict[str, Dict[str, int]] | None = None):
        self.fixed_vocab = vocabs is not None
        self.vocabs = vocabs if vocabs is not None else {f: {} for f in FIELDS}
        self._indices = {f: [] for f in FIELDS}
        self._indptr = {f: [0] for f in FIELDS}
        self._counts = {f: [] for f in FIELDS}
        self.name_clean: List[str] = []
        self.addr_clean: List[str] = []
        self.n = 0

    def add_rows(self, rows: Sequence[Tuple]) -> None:
        for r in rows:
            self.name_clean.append(r[0])
            self.addr_clean.append(r[1])
            for k, fname in enumerate(FIELDS):
                fs = r[2 + k]
                uniq = set(fs)
                self._counts[fname].append(len(uniq))
                vocab = self.vocabs[fname]
                if self.fixed_vocab:
                    cols = [vocab[f] for f in uniq if f in vocab]
                else:
                    cols = []
                    for f in uniq:
                        j = vocab.get(f)
                        if j is None:
                            j = len(vocab); vocab[f] = j
                        cols.append(j)
                self._indices[fname].extend(cols)
                self._indptr[fname].append(len(self._indices[fname]))
            self.n += 1

    def finish(self) -> SparseStore:
        store = SparseStore.__new__(SparseStore)
        store.n = self.n
        store.name_clean = self.name_clean
        store.addr_clean = self.addr_clean
        store.vocabs = self.vocabs
        store.mats, store.counts = {}, {}
        for fname in FIELDS:
            idx = np.asarray(self._indices[fname], dtype=np.int32)
            indptr = np.asarray(self._indptr[fname], dtype=np.int64)
            store.mats[fname] = sp.csr_matrix(
                (np.ones(len(idx), dtype=np.float32), idx, indptr),
                shape=(self.n, max(1, len(self.vocabs[fname]))))
            store.counts[fname] = np.asarray(self._counts[fname], dtype=np.float32)
            self._indices[fname] = []; self._indptr[fname] = [0]; self._counts[fname] = []
        return store


def build_store_streaming(names: Sequence[str], addrs: Sequence[str], workers: int,
                          vocabs=None, log=None) -> SparseStore:
    """Extract in a pool, fold chunk-by-chunk in the parent, discard as we go."""
    from multiprocessing import Pool

    builder = StreamingStoreBuilder(vocabs)
    tasks = [(names[i:i + _EXTRACT_CHUNK_SIZE], addrs[i:i + _EXTRACT_CHUNK_SIZE])
             for i in range(0, len(names), _EXTRACT_CHUNK_SIZE)]
    if workers <= 1:
        for t in tasks:
            builder.add_rows(_extract_chunk(t))
    else:
        with Pool(workers) as pool:
            for k, rows in enumerate(pool.imap(_extract_chunk, tasks, chunksize=1)):
                builder.add_rows(rows)
                del rows
                if log and (k + 1) % 50 == 0:
                    log(f"      store: {builder.n:,}/{len(names):,} records")
    store = builder.finish()
    if log:
        log(f"      store built: {store.n:,} records, matrices {store.nbytes():.2f} GB")
    return store
