"""
Train and persist the production reranker.

Reads the cached pair features built by src.diag_pooled (retrieval + feature
computation is the expensive part, so it is not repeated here), fits ONE pooled
logistic regression across all labelled countries, and writes a bundle holding
the model, the scaler and the channel specs so inference cannot drift from
training.

Pooled rather than per-country because France carries no labels: a per-country
model is impossible for it, and country-conditional models are exactly the
hardcoding the brief rules out. Measured cost of pooling is -0.04 points on
India and -0.33 on US (reports/phase3_blocking.md).

Logistic regression rather than trees or a neural model because the measured
headroom does not justify anything larger: the pool holds 98.8-99.3% of true
pairs and this model already recovers 98.5-99.6% of what is in the pool.
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from src.config import channel_specs  # noqa: E402
from src.rerank import FEATURE_NAMES  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout-frac", type=float, default=0.2,
                    help="queries held back per country to sanity-check the fit")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.cache_dir, "pairs_*.npz")))
    if not files:
        raise SystemExit(f"no pair caches in {args.cache_dir}")

    Xs, ys, holds = [], [], {}
    for path in files:
        country = os.path.basename(path)[len("pairs_"):-len(".npz")]
        d = np.load(path)
        X, y, gid, nq = d["X"], d["y"], d["gid"], int(d["n_queries"])
        cut = int((1.0 - args.holdout_frac) * nq)
        tr = gid < cut
        Xs.append(X[tr]); ys.append(y[tr])
        holds[country] = (X[~tr], y[~tr], gid[~tr])
        print(f"  {country}: {X.shape[0]:,} pairs, train {int(tr.sum()):,}, "
              f"holdout {int((~tr).sum()):,}, positives {int(y.sum()):,}")

    X = np.vstack(Xs); y = np.concatenate(ys)
    del Xs, ys
    print(f"\npooled training set: {X.shape}, {100*y.mean():.3f}% positive")

    t0 = time.perf_counter()
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(scaler.transform(X), y)
    print(f"fitted in {time.perf_counter()-t0:.1f}s")

    # Sanity check: recall@K per held-out country, so a bad fit is caught here
    # rather than after a two-hour production run.
    print("\nholdout recall@K (of positives present in the pool):")
    print("  " + "country".ljust(10) + "".join(f"{f'@{k}':>9}" for k in (10, 20, 30, 50)))
    for country, (Xh, yh, gh) in holds.items():
        s = model.decision_function(scaler.transform(Xh))
        hits = {k: 0 for k in (10, 20, 30, 50)}
        tot = 0
        for q in np.unique(gh):
            m = gh == q
            lab = yh[m][np.argsort(-s[m])]
            tot += int(lab.sum())
            for k in hits:
                hits[k] += int(lab[:k].sum())
        print("  " + country.ljust(10) + "".join(f"{100*hits[k]/tot:>8.2f} " for k in hits))

    bundle = {"model": model, "scaler": scaler, "specs": channel_specs(),
              "feature_names": FEATURE_NAMES}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nwrote {args.out} ({os.path.getsize(args.out)/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
