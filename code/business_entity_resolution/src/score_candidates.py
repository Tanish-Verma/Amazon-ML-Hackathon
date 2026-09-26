"""
Score a candidate TSV against ground truth.

This is the accuracy check for the PRODUCTION path. Every recall number quoted so
far came from diagnostics operating on cached features; this one reads the actual
file `src/run_blocking.py` wrote, so it validates the production code end to end --
including the array-based pool refactor, which no earlier measurement covered.

The test set has no labels, so accuracy is measured by running the identical
pipeline on a train slice. Reports per-country macro and micro recall, candidate
set sizes and the reduction ratio.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.candidates import load_partition  # noqa: E402
from src.eval_blocking import evaluate  # noqa: E402


def load_id_lists(path):
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(r)
        for row in r:
            out[row[0]] = row[1].split(",") if len(row) > 1 and row[1] else []
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="+", required=True,
                    help="one or more candidate TSVs (they are merged)")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    cands = {}
    for p in args.candidates:
        part = load_id_lists(p)
        cands.update(part)
        print(f"  loaded {len(part):,} rows from {os.path.basename(p)}")

    # Country of each S1 entity, and corpus size per country for reduction ratio.
    country_of, corpus_sizes = {}, {}
    for fn in sorted(os.listdir(args.store)):
        if not fn.startswith(f"{args.split}_s1_") or not fn.endswith(".parquet"):
            continue
        ctry = fn[len(f"{args.split}_s1_"):-len(".parquet")]
        ids, _, _ = load_partition(args.store, args.split, "s1", ctry)
        for e in ids:
            country_of[e] = ctry
        corpus_sizes[ctry] = sum(
            len(load_partition(args.store, args.split, s, ctry)[0]) for s in ("s2", "s3"))

    # Evaluate only over entities we actually produced candidates for, so a
    # partial run (e.g. --max-queries) is scored honestly rather than being
    # penalised for entities it was never asked to process.
    truth = {}
    with open(args.gt, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.partition("\t")
            if s1 in cands:
                rest = rest.rstrip("\n")
                truth[s1] = rest.split(",") if rest else []
    print(f"  scoring {len(truth):,} entities with ground truth\n")

    report = evaluate(cands, truth, country_of, corpus_sizes)
    print(report)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(f"# Production-path candidate accuracy ({args.split})\n\n{report}\n")
        print(f"\n  wrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
