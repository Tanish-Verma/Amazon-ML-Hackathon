import numpy as np
from src.features import (FEATURE_NAMES, reverse_ranks, build_features,
                          address_document_frequencies, feature_auc, rank_context)
from src.rerank_vec import SparseStore, record_fields


def test_reverse_rank_across_queries_and_ties():
    got = reverse_ranks([1, 2, 3, 1], [10, 10, 10, 20], [.8, .9, .9, .2])
    assert got.tolist() == [3, 1, 2, 1]


def test_features_and_location():
    c_raw = ["12 Main St, Paris, Ile de France", "77 Elm St, Lyon, Auvergne"]
    q_raw = ["12 Main St, Paris, Ile de France", "77 Elm St, Lyon, Auvergne"]
    c = SparseStore([record_fields("Coffee House", c_raw[0]),
                     record_fields("Tea House", c_raw[1])])
    q = SparseStore([record_fields("Coffee House", q_raw[0]),
                     record_fields("Tea House", q_raw[1])], vocabs=c.vocabs)
    qi = np.array([0, 0, 1]); ci = np.array([0, 1, 1]); sc = np.array([.9, .4, .8])
    X = build_features(q, c, qi, ci, np.ones(3), np.ones(3), np.ones(3),
                       np.zeros(3), sc, reverse_ranks(qi, ci, sc),
                       address_document_frequencies(c), workers=1,
                       q_raw_addrs=q_raw, c_raw_addrs=c_raw)
    assert X.shape == (3, 23) and np.isfinite(X).all()
    assert X[:, 16].tolist() == [1, 2, 1]
    assert X[:, 19].tolist() == [2, 2, 1]
    assert X[:, 20].tolist() == [1, 2, 1]
    assert X[0, 22] > X[1, 22]
    assert not any("country" in name for name in FEATURE_NAMES)


def test_zero_and_negative_score_handling():
    X = rank_context([0, 0, 1], [0, 1, 2], [0, -1, -2], [1, 1, 1])
    assert np.isfinite(X).all() and X[0, 2] == 0


def test_auc():
    X = np.tile(np.array([[0.], [1.]], dtype=np.float32), (1, 23))
    assert len(feature_auc(X, [0, 1])) == 23
