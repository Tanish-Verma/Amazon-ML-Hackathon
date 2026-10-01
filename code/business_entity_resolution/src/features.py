"""Phase 4 features. The first 16 columns are the exact Phase 3 features.

One country at a time; country is NEVER a model feature. For exact reverse
ranks, use the complete country's contenders, not one query batch.
"""
from __future__ import annotations

import time
from collections import Counter
import numpy as np
from src.normalize import tokenize
from src.rerank import FEATURE_NAMES as BASE_NAMES
from src.rerank_vec import pair_features_vec

EXTRA_NAMES = ["candidate_rank", "score_margin_to_best", "score_ratio_to_best",
               "candidate_count", "reverse_rank", "shared_rare_address_tokens",
               "city_region_agreement"]
FEATURE_NAMES = list(BASE_NAMES) + EXTRA_NAMES


def reverse_ranks(query_ids, candidate_ids, scores):
    """Complete partition's reverse ranks, deterministic ties by query ID.

    Candidate IDs must identify S2/S3 records uniquely; IDs from distinct
    sources must be source-prefixed if their namespaces overlap.
    """
    q, c, s = np.asarray(query_ids), np.asarray(candidate_ids), np.asarray(scores, dtype=np.float64)
    if len(q) != len(c) or len(c) != len(s) or not np.isfinite(s).all():
        raise ValueError("Aligned IDs and finite scores required")
    if not len(s):
        return np.empty(0, dtype=np.float32)
    order = np.lexsort((q, -s, c))
    sc = c[order]
    boundaries = np.r_[0, np.flatnonzero(sc[1:] != sc[:-1]) + 1]
    lengths = np.diff(np.r_[boundaries, len(s)])
    rank = np.arange(len(s), dtype=np.int64) - np.repeat(boundaries, lengths) + 1
    out = np.empty(len(s), dtype=np.float32)
    out[order] = rank
    return out


def address_document_frequencies(cstore):
    """DF from this country's S2/S3 address corpus (never validation labels)."""
    mat = cstore.mats["addr_tok"].tocsc()
    df = np.diff(mat.indptr)
    return {token: int(df[idx]) for token, idx in cstore.vocabs["addr_tok"].items()}


def _location_tokens(raw_address):
    """Extract trailing address components, not a country-specific gazetteer.

    Location components are only *proxies*: addresses without commas yield no
    location evidence; do not confuse this with verified city/region parsing.
    """
    parts = str(raw_address or "").split(",")
    if len(parts) < 2:
        return set()
    return set(tokenize(" ".join(parts[-2:])))


def address_context(qstore, cstore, qrows, crows, df,
                    q_raw_addrs=None, c_raw_addrs=None,
                    rare_fraction=0.01):
    """Rare address-token overlap and IDF-weighted trailing-component agreement.

    Raw comma-delimited addresses are required for the location proxy; when
    unavailable it returns zero (unknown), never guesses from shared street text.
    """
    n = len(qrows)
    out = np.zeros((n, 2), dtype=np.float32)
    rare_cut = max(1, int(cstore.n * rare_fraction))
    for i, (qi, ci) in enumerate(zip(qrows, crows)):
        qi, ci = int(qi), int(ci)
        shared = set(tokenize(qstore.addr_clean[qi])) & set(tokenize(cstore.addr_clean[ci]))
        out[i, 0] = sum(0 < df.get(t, 0) <= rare_cut for t in shared)
        if q_raw_addrs is None or c_raw_addrs is None:
            continue
        qt = _location_tokens(q_raw_addrs[qi])
        ct = _location_tokens(c_raw_addrs[ci])
        if not qt or not ct:
            continue
        # A weighted Jaccard score: rarer shared city/region tokens count more.
        union = qt | ct
        weights = {t: np.log1p(cstore.n / (1 + df.get(t, cstore.n))) for t in union}
        denom = sum(weights.values())
        if denom:
            out[i, 1] = sum(weights[t] for t in qt & ct) / denom
    return out


def rank_context(qrows, crows, scores, reverse_rank):
    """All candidates of each S1 must be included in this call."""
    q, c, s, rr = (np.asarray(qrows), np.asarray(crows),
                   np.asarray(scores, dtype=np.float32),
                   np.asarray(reverse_rank, dtype=np.float32))
    n = len(s)
    if any(len(v) != n for v in (q, c, rr)) or not np.isfinite(s).all() or not np.isfinite(rr).all():
        raise ValueError("Aligned finite ranking inputs required")
    out = np.zeros((n, 5), dtype=np.float32)
    if not n:
        return out
    order = np.lexsort((c, -s, q))
    sorted_q = q[order]
    starts = np.r_[0, np.flatnonzero(sorted_q[1:] != sorted_q[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    for start, end in zip(starts, ends):
        ix = order[start:end]
        best = float(s[ix[0]])
        out[ix, 0] = np.arange(1, len(ix) + 1)
        out[ix, 1] = best - s[ix]
        out[ix, 2] = s[ix] / best if abs(best) > 1e-12 else 0
        out[ix, 3] = len(ix)
    out[:, 4] = rr
    return out


def build_features(qstore, cstore, qrows, crows, rrf, n_hit, best_rank,
                   is_s3, scores, reverse_rank, address_df, workers=8,
                   q_raw_addrs=None, c_raw_addrs=None):
    """Return (n_pairs,23) finite float32 matrix, retaining all 16 base features."""
    n = len(qrows)
    if any(len(v) != n for v in (crows, rrf, n_hit, best_rank, is_s3, scores, reverse_rank)):
        raise ValueError("Pair columns must have equal lengths")
    base = pair_features_vec(qstore, cstore, np.asarray(qrows), np.asarray(crows),
                             np.asarray(rrf), np.asarray(n_hit), np.asarray(best_rank),
                             np.asarray(is_s3), workers=workers)
    ranks = rank_context(qrows, crows, scores, reverse_rank)
    address = address_context(qstore, cstore, qrows, crows, address_df,
                              q_raw_addrs, c_raw_addrs)
    X = np.column_stack((base, ranks, address)).astype(np.float32)
    if X.shape != (n, len(FEATURE_NAMES)) or not np.isfinite(X).all():
        raise ValueError("Feature matrix has incorrect shape or NaN/inf")
    return X


def feature_auc(X, y):
    """Return (name, AUC, direction-neutral AUC); no label leakage."""
    from sklearn.metrics import roc_auc_score
    X, y = np.asarray(X), np.asarray(y)
    if X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES) or len(y) != len(X):
        raise ValueError("Incorrect feature/label shapes")
    if len(np.unique(y)) != 2:
        raise ValueError("AUC needs both classes")
    out = []
    for i, name in enumerate(FEATURE_NAMES):
        auc = float(roc_auc_score(y, X[:, i]))
        out.append((name, auc, max(auc, 1.0 - auc)))
    return out
