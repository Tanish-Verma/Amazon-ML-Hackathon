"""
Cheap feature-based reranker over the broad candidate pool.

Blocking retrieves a deep pool (RRF over four channels, ~1000 per query) which
holds 97.93% of true pairs. RRF ranks by *rank fusion*, so it only reaches 93.40%
at top-50 — it never looks at the actual pair. This module scores real pairwise
similarity so the pool can be cut to top-30/50 while keeping most of its recall.

Deliberately cheap: set intersections plus rapidfuzz (C++) similarities, scored
by logistic regression. No neural model. It has to run over ~1.7 billion pairs at
full test scale, so per-pair cost is the binding constraint, and a linear model
over ~16 features is both fast and easy to inspect.

Design note on cost: per-record normalised forms (token sets, trigram sets, digit
keys) are computed ONCE per record and reused across every pair that record
appears in. Normalising inside the pair loop would repeat the same work hundreds
of times per record and dominate everything else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from src.normalize import char_ngrams, clean_text, extract_numeric_keys, tokenize

FEATURE_NAMES = [
    "name_trigram_jaccard",
    "name_token_jaccard",
    "name_token_containment",
    "name_jaro_winkler",
    "name_token_set_ratio",
    "name_len_ratio",
    "addr_trigram_jaccard",
    "addr_token_jaccard",
    "addr_jaro_winkler",
    "digit_jaccard",
    "digit_shared_count",
    "addr_missing",
    "rrf_score",
    "n_channels_hit",
    "best_channel_rank",
    "is_source3",
]


@dataclass
class RecordForm:
    """Normalised forms of one record, computed once and reused."""
    __slots__ = ("name_clean", "name_tokens", "name_tri", "addr_clean",
                 "addr_tokens", "addr_tri", "digits")
    name_clean: str
    name_tokens: frozenset
    name_tri: frozenset
    addr_clean: str
    addr_tokens: frozenset
    addr_tri: frozenset
    digits: frozenset


def build_form(name: str, addr: str) -> RecordForm:
    nc = clean_text(name)
    ac = clean_text(addr) if addr and addr.strip() else ""
    return RecordForm(
        name_clean=nc,
        name_tokens=frozenset(tokenize(nc)),
        name_tri=frozenset(char_ngrams(nc, 3)),
        addr_clean=ac,
        addr_tokens=frozenset(tokenize(ac)),
        addr_tri=frozenset(char_ngrams(ac, 3)),
        digits=frozenset(extract_numeric_keys(addr)) if addr else frozenset(),
    )


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def pair_features(q: RecordForm, c: RecordForm, rrf: float,
                  n_hit: int, best_rank: int, is_s3: float) -> List[float]:
    """The 16 features for one query-candidate pair."""
    n_inter = len(q.name_tokens & c.name_tokens)
    containment = n_inter / min(len(q.name_tokens), len(c.name_tokens)) \
        if q.name_tokens and c.name_tokens else 0.0
    lq, lc = len(q.name_clean), len(c.name_clean)
    len_ratio = min(lq, lc) / max(lq, lc) if lq and lc else 0.0
    addr_missing = 1.0 if not c.addr_clean or not q.addr_clean else 0.0
    d_inter = len(q.digits & c.digits)

    return [
        _jaccard(q.name_tri, c.name_tri),
        _jaccard(q.name_tokens, c.name_tokens),
        containment,
        JaroWinkler.similarity(q.name_clean, c.name_clean),
        fuzz.token_set_ratio(q.name_clean, c.name_clean) / 100.0,
        len_ratio,
        _jaccard(q.addr_tri, c.addr_tri),
        _jaccard(q.addr_tokens, c.addr_tokens),
        JaroWinkler.similarity(q.addr_clean, c.addr_clean) if q.addr_clean and c.addr_clean else 0.0,
        _jaccard(q.digits, c.digits),
        float(d_inter),
        addr_missing,
        rrf,
        float(n_hit),
        float(best_rank),
        is_s3,
    ]


# --------------------------------------------------------------------------
# Parallel scoring for production scale
# --------------------------------------------------------------------------
# The diagnostic measured 66,466 pairs/sec single-threaded. At full test scale
# (1.73M queries x 1000 candidates = ~1.73 billion pairs) that is ~7.2 hours on
# one core, which is the single largest cost left in the pipeline. The loop is
# embarrassingly parallel over queries, so it is fanned across processes.
#
# Workers inherit the record forms through fork (Linux) rather than having them
# pickled per task: the forms for a 10M-record corpus are several GB, so shipping
# them per task would cost far more than the scoring itself.

_SCORE_STATE: dict = {}


def _score_chunk(args):
    """Score one contiguous run of queries; return top-K doc ids per query."""
    lo, hi = args
    qforms = _SCORE_STATE["qforms"]
    cforms = _SCORE_STATE["cforms"]
    pools = _SCORE_STATE["pools"]
    is_s3 = _SCORE_STATE["is_s3"]
    coef = _SCORE_STATE["coef"]
    mean = _SCORE_STATE["mean"]
    scale = _SCORE_STATE["scale"]
    intercept = _SCORE_STATE["intercept"]
    top_k = _SCORE_STATE["top_k"]

    out = []
    for qi in range(lo, hi):
        pool = pools[qi]
        if not pool:
            out.append(np.empty(0, dtype=np.int64))
            continue
        qf = qforms[qi]
        docs = np.fromiter(pool.keys(), dtype=np.int64, count=len(pool))
        feats = np.empty((len(pool), len(coef)), dtype=np.float32)
        for r, d in enumerate(docs):
            rrf, nh, br = pool[int(d)]
            feats[r] = pair_features(qf, cforms[int(d)], rrf, nh, br, is_s3[int(d)])
        # Inline the StandardScaler + linear decision function: one matmul beats
        # round-tripping through sklearn objects for a billion rows.
        z = (feats - mean) / scale
        score = z @ coef + intercept
        if docs.size > top_k:
            keep = np.argpartition(-score, top_k)[:top_k]
            docs, score = docs[keep], score[keep]
        out.append(docs[np.argsort(-score)])
    return lo, out


def rerank_parallel(qforms, cforms, pools, is_s3, model, scaler, top_k: int,
                    workers: int, chunk: int = 2000):
    """Rerank every query's pool down to top_k. Returns list of doc-id arrays."""
    from multiprocessing import Pool as _Pool

    n = len(qforms)
    _SCORE_STATE.update(
        qforms=qforms, cforms=cforms, pools=pools, is_s3=is_s3,
        coef=model.coef_[0].astype(np.float32),
        intercept=float(model.intercept_[0]),
        mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32),
        top_k=top_k,
    )
    ranges = [(i, min(i + chunk, n)) for i in range(0, n, chunk)]
    result: List = [None] * n
    try:
        if workers <= 1:
            for lo, part in map(_score_chunk, ranges):
                result[lo:lo + len(part)] = part
        else:
            with _Pool(workers) as pool:      # fork: inherits _SCORE_STATE
                for lo, part in pool.imap_unordered(_score_chunk, ranges, chunksize=1):
                    result[lo:lo + len(part)] = part
    finally:
        _SCORE_STATE.clear()
    return result
