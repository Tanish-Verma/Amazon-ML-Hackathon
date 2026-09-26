"""
How much recall do we lose by reranking a shallower slice of the pool?

Rerank cost is linear in pool depth and rerank is ~82% of the runtime, so cutting
depth is the biggest algorithmic lever left. This needs no retrieval: the cached
pair features already hold every pool pair together with its RRF score, so we can
truncate to the top-N by RRF and re-rank exactly as production would.

Reports recall@50 at each depth, per country, plus the delta against full depth.
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, ".")
from src.rerank import FEATURE_NAMES  # noqa: E402

DEPTHS = (150, 200, 300, 400, 500, 700, 1000)
K = 50
RRF_COL = FEATURE_NAMES.index("rrf_score")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    with open(args.model, "rb") as f:
        b = pickle.load(f)
    model, scaler = b["model"], b["scaler"]

    print(f"  recall@{K} of pool positives, by rerank depth\n")
    hdr = "  " + "country".ljust(10) + "".join(f"{d:>9}" for d in DEPTHS)
    print(hdr); print("  " + "-" * (len(hdr) - 2))

    for path in sorted(glob.glob(os.path.join(args.cache_dir, "pairs_*.npz"))):
        country = os.path.basename(path)[len("pairs_"):-len(".npz")]
        d = np.load(path)
        X, y, gid, nq = d["X"], d["y"], d["gid"], int(d["n_queries"])
        # Evaluate on the same held-out 40% of queries the model was not fit on.
        cut = int(0.6 * nq)
        m = gid >= cut
        X, y, gid = X[m], y[m], gid[m]
        score = model.decision_function(scaler.transform(X))
        rrf = X[:, RRF_COL]

        total = int(y.sum())
        row = {}
        for depth in DEPTHS:
            hits = 0
            for q in np.unique(gid):
                sel = gid == q
                s, r, lab = score[sel], rrf[sel], y[sel]
                if len(s) > depth:                      # keep top-`depth` by RRF
                    keep = np.argpartition(-r, depth)[:depth]
                    s, lab = s[keep], lab[keep]
                hits += int(lab[np.argsort(-s)[:K]].sum())
            row[depth] = 100 * hits / total
        print("  " + country.ljust(10) + "".join(f"{row[d]:>8.2f} " for d in DEPTHS))
        full = row[DEPTHS[-1]]
        print("  " + "delta".ljust(10) + "".join(f"{row[d]-full:>+8.2f} " for d in DEPTHS))
    print(f"\n  (delta is against depth {DEPTHS[-1]}; negative = recall lost)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
