"""Phase 5 precision-first, globally exclusive assignment."""
from __future__ import annotations
from collections import defaultdict


def decide(rows, threshold: float, relative_margin: float, exclusive: bool = True):
    """rows: (query_id, candidate_id, calibrated_probability) for one country.

    Accept candidates with p >= threshold and p >= best_per_query - margin.
    Highest probability wins shared candidates; ties deterministic by IDs.
    Return {query_id: set(candidate_id)}; missing queries mean empty prediction.
    The caller must provide the COMPLETE country's scored pairs, not a chunk.
    """
    if not 0 <= threshold <= 1 or not 0 <= relative_margin <= 1:
        raise ValueError('threshold and margin must be in [0,1]')
    best = {}
    data = []
    seen = set()
    for q, c, p in rows:
        q, c, p = str(q), str(c), float(p)
        if not 0 <= p <= 1:
            raise ValueError('Invalid calibrated probability')
        if (q,c) in seen:
            raise ValueError('Duplicate candidate pair')
        seen.add((q,c))
        data.append((q,c,p))
        best[q] = max(p, best.get(q, -1.0))
    eligible = [(q,c,p) for q,c,p in data
                if p >= threshold and p >= best[q] - relative_margin]
    eligible.sort(key=lambda x: (-x[2], x[1], x[0]))
    output = defaultdict(set)
    used = set()
    for q,c,p in eligible:
        if exclusive and c in used:
            continue
        output[q].add(c)
        used.add(c)
    return dict(output)


def score_decisions(rows, truth, threshold, margin, exclusive=True):
    from src.scoring import macro_f05
    return macro_f05(decide(rows, threshold, margin, exclusive), truth)
