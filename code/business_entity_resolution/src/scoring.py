"""
The competition metric: macro-averaged F-beta with beta = 0.5.

Built early (ahead of blocking and matching) because every downstream decision
-- candidate cap K, classifier threshold, assignment margin -- is tuned against
this number, and a scorer that disagrees with the official one in any edge case
would silently mistune all of them.

Specification, from the problem statement:

    F_0.5 = (1.25 * P * R) / (0.25 * P + R)

computed PER Source-1 entity and then averaged over all Source-1 entities in
the evaluation set, singletons included. The stated singleton convention is
that an entity with no true matches scores 1.0 when an empty list is predicted
and 0.0 when anything is predicted.

The edge cases are where a re-implementation usually drifts, so they are named
explicitly here and pinned by tests:

  true empty, pred empty      -> 1.0   (correctly identified singleton)
  true empty, pred non-empty  -> 0.0   (false merge onto a singleton)
  true non-empty, pred empty  -> 0.0   (P undefined; no credit)
  no overlap at all           -> 0.0   (P = R = 0; formula would divide by zero)
"""

from __future__ import annotations

from typing import Iterable, Mapping

BETA = 0.5
BETA_SQ = BETA * BETA  # 0.25


def f_beta(precision: float, recall: float, beta_sq: float = BETA_SQ) -> float:
    """F-beta from precision and recall, guarding the 0/0 case."""
    denom = beta_sq * precision + recall
    if denom == 0.0:
        return 0.0
    return ((1.0 + beta_sq) * precision * recall) / denom


def entity_score(predicted: set[str], truth: set[str]) -> float:
    """F_0.5 for a single Source-1 entity, following the stated conventions."""
    if not truth:
        # A true singleton: full credit only for predicting nothing.
        return 1.0 if not predicted else 0.0
    if not predicted:
        # Missed every match; precision is undefined and earns no credit.
        return 0.0
    hits = len(predicted & truth)
    if hits == 0:
        return 0.0
    precision = hits / len(predicted)
    recall = hits / len(truth)
    return f_beta(precision, recall)


def macro_f05(
    predictions: Mapping[str, Iterable[str]],
    truth: Mapping[str, Iterable[str]],
) -> dict:
    """Macro-average F_0.5 over every entity in ``truth``.

    ``truth`` defines the evaluation set: an entity present in truth but absent
    from ``predictions`` is scored as an empty prediction (which is what the
    scorer effectively does to a missing row), rather than being skipped --
    skipping it would inflate the average.

    Returns the headline score plus the diagnostic breakdown we actually steer
    on: how we do on singletons versus matched entities, and whether losses are
    coming from false merges or from misses.
    """
    total = 0.0
    n = 0
    sing_n = sing_correct = 0
    match_n = 0
    match_score = 0.0
    tp = fp = fn = 0

    for s1_id, true_iter in truth.items():
        true_set = set(true_iter)
        pred_set = set(predictions.get(s1_id, ()))
        score = entity_score(pred_set, true_set)
        total += score
        n += 1

        if not true_set:
            sing_n += 1
            if not pred_set:
                sing_correct += 1
        else:
            match_n += 1
            match_score += score

        tp += len(pred_set & true_set)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)

    return {
        "macro_f05": total / n if n else 0.0,
        "n_entities": n,
        "n_singletons": sing_n,
        "singleton_accuracy": sing_correct / sing_n if sing_n else float("nan"),
        "n_matched_entities": match_n,
        "macro_f05_matched_only": match_score / match_n if match_n else float("nan"),
        "micro_precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "micro_recall": tp / (tp + fn) if (tp + fn) else float("nan"),
        "tp": tp, "fp": fp, "fn": fn,
    }
