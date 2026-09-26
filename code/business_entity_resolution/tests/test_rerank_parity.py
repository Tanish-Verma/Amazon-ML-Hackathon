"""The vectorised reranker must produce features IDENTICAL to the original.

The production model was trained on features from src.rerank.pair_features. If
src.rerank_vec computes anything differently -- a different empty-set convention,
a different column order -- the model silently scores garbage. So this asserts
column-by-column equality on real noisy records rather than trusting the rewrite.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.rerank import FEATURE_NAMES, build_form, pair_features  # noqa: E402
from src.rerank_vec import SparseStore, pair_features_vec, record_fields  # noqa: E402

# Real records drawn from reports/eda.md, chosen to exercise the edge cases:
# transliteration, injected typos, diacritics, empty addresses, French suffixes.
QUERIES = [
    ("Ss Food Private Limited", "Af-684, Nandgram, Ghaziabad, Uttar Pradesh"),
    ("Payne Enterprises", "3315 Fremont Street, Peoria, IL"),
    ("Maure Williams Colombier Inc", "85 Wayne Avenue, Ticonderoga, NY"),
    ("Thermal & Fils SASU", "20 Rue Parmentier, Dunkerque, Hauts-de-France"),
    ("Orellana Investments LLC", ""),
]
CANDIDATES = [
    ("एसएस फूड प्राइवेट लिमिटेड", "AF-0684, GHAZIABAD, उत्तर प्रदेश"),
    ("Payne Énterprises", "3315 FREMONT ST, PEORIA, IL"),
    ("PAYNE-ENRTPRMISES", "3315 FREMONT SAINT, PEORIA, IL"),
    ("Dréxkor", "85 Wanye Avenue, Ticonderoga Townshiip, New York"),
    ("maurewilliamscolombier.com", ""),
    ("Marina Ecole France Sarl", "63 R. DE DIEPPE, LILLE, Hauts-de-France"),
    ("LLC Orellana Invsmbens", "728 A Quail Avenue, Geneva, Iowa"),
]


def test_vectorised_features_match_original():
    qrows_src = [record_fields(n, a) for n, a in QUERIES]
    crows_src = [record_fields(n, a) for n, a in CANDIDATES]
    # Query store must share the corpus vocabularies, or column ids won't align.
    cstore = SparseStore(crows_src)
    qstore = SparseStore(qrows_src, vocabs=cstore.vocabs)

    qi, ci, rrf, nh, br, s3 = [], [], [], [], [], []
    for a in range(len(QUERIES)):
        for b in range(len(CANDIDATES)):
            qi.append(a); ci.append(b)
            rrf.append(0.05 * (a + 1)); nh.append((a + b) % 4 + 1)
            br.append(3 * b); s3.append(float(b % 2))

    X_vec = pair_features_vec(
        qstore, cstore, np.asarray(qi), np.asarray(ci),
        np.asarray(rrf, dtype=np.float32), np.asarray(nh, dtype=np.float32),
        np.asarray(br, dtype=np.float32), np.asarray(s3, dtype=np.float32), workers=2)

    qforms = [build_form(n, a) for n, a in QUERIES]
    cforms = [build_form(n, a) for n, a in CANDIDATES]
    X_ref = np.asarray([
        pair_features(qforms[a], cforms[b], rrf[k], nh[k], br[k], s3[k])
        for k, (a, b) in enumerate(zip(qi, ci))], dtype=np.float32)

    assert X_vec.shape == X_ref.shape, (X_vec.shape, X_ref.shape)
    bad = []
    for j, fname in enumerate(FEATURE_NAMES):
        if not np.allclose(X_vec[:, j], X_ref[:, j], atol=2e-4):
            d = np.abs(X_vec[:, j] - X_ref[:, j])
            bad.append(f"{fname}: max|delta|={d.max():.5f} at row {int(d.argmax())} "
                       f"(vec={X_vec[d.argmax(), j]:.5f} ref={X_ref[d.argmax(), j]:.5f})")
    assert not bad, "feature mismatch:\n  " + "\n  ".join(bad)


def test_empty_address_and_empty_name_do_not_crash():
    rows = [record_fields("", ""), record_fields("Some Co", "")]
    store = SparseStore(rows)
    q = SparseStore([record_fields("Some Co", "1 Main St")], vocabs=store.vocabs)
    X = pair_features_vec(q, store, np.array([0, 0]), np.array([0, 1]),
                          np.zeros(2, np.float32), np.ones(2, np.float32),
                          np.zeros(2, np.float32), np.zeros(2, np.float32), workers=1)
    assert np.isfinite(X).all()
    assert X[:, FEATURE_NAMES.index("addr_missing")].tolist() == [1.0, 1.0]


if __name__ == "__main__":
    fails = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn(); print(f"  PASS {name}")
            except AssertionError as e:
                fails.append(name); print(f"  FAIL {name}:\n{e}")
    print(f"\n{len(fails)} failure(s)." if fails else "\nparity confirmed")
    sys.exit(1 if fails else 0)
