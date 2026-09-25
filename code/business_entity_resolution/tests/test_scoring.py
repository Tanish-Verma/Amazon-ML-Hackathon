"""Pins the metric against the problem statement's own worked example."""
import math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.scoring import entity_score, macro_f05, f_beta


def test_worked_example_from_problem_statement():
    # Predicted [S2-00047, S2-00193, S3-00812], truth [S2-00047, S3-00812].
    # P = 2/3, R = 1.0  ->  the PDF states 0.714.
    got = entity_score({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"})
    assert abs(got - 0.714) < 5e-4, got


def test_singleton_conventions():
    assert entity_score(set(), set()) == 1.0            # correct singleton
    assert entity_score({"S2-1"}, set()) == 0.0          # false merge on singleton
    assert entity_score(set(), {"S2-1"}) == 0.0          # missed everything
    assert entity_score({"S2-9"}, {"S2-1"}) == 0.0       # no overlap


def test_perfect_and_precision_weighting():
    assert entity_score({"S2-1", "S3-2"}, {"S2-1", "S3-2"}) == 1.0
    # beta=0.5 weights precision above recall: losing precision must hurt more
    # than losing the same amount of recall.
    lost_precision = entity_score({"S2-1", "S2-2"}, {"S2-1"})          # P=.5 R=1
    lost_recall    = entity_score({"S2-1"}, {"S2-1", "S2-2"})          # P=1  R=.5
    assert lost_precision < lost_recall


def test_missing_prediction_row_counts_as_empty():
    truth = {"S1-a": {"S2-1"}, "S1-b": set()}
    res = macro_f05({}, truth)
    assert res["macro_f05"] == 0.5   # 0.0 for the matched entity, 1.0 for the singleton
    assert res["n_entities"] == 2


def test_all_singleton_baseline_equals_singleton_rate():
    truth = {f"S1-{i}": (set() if i < 3 else {"S2-1"}) for i in range(10)}
    res = macro_f05({}, truth)
    assert math.isclose(res["macro_f05"], 0.3)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"  PASS {name}")
    print("all scoring tests passed")
