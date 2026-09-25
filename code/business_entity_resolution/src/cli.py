"""
Command-line entry points for the pipeline.

Each stage is separately runnable so blocking quality can be inspected
independently of final matching precision, as the challenge asks:

    python -m src.cli verify   --data-dir DATA            # schema assertions
    python -m src.cli prep     --data-dir DATA            # TSV -> partitioned parquet
    python -m src.cli baseline --data-dir DATA --out OUT  # R0 all-singleton submission
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyarrow.compute as pc  # noqa: E402

from src.io_utils import (  # noqa: E402
    SOURCE_COLUMNS, GROUND_TRUTH_COLUMNS, assert_free_space, build_columnar_store,
    free_bytes, read_source_table, stream_tsv, write_id_list_tsv, pa_unique,
)

SPLIT_FILES = {
    "train": ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv"],
    "test": ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"],
}


def cmd_verify(args) -> int:
    """Assert every structural claim the pipeline later relies on.

    Cheap to run and worth running first: a wrong assumption about ID prefixes
    or duplicate entity IDs would not surface until the submission is rejected.
    """
    problems = []
    for split, files in SPLIT_FILES.items():
        for idx, fname in enumerate(files, start=1):
            path = os.path.join(args.data_dir, split, fname)
            if not os.path.isfile(path):
                print(f"  SKIP {split}/{fname} (not present)")
                continue
            t0 = time.time()
            table = read_source_table(path)
            # Arrow compute rather than .to_pylist(): a 5M-row source would
            # otherwise materialise millions of Python str objects and cost
            # several GB for what are one-pass aggregations.
            eid = table.column("entity_id")
            prefix = f"S{idx}-"
            n_bad_prefix = len(eid) - pc.sum(pc.starts_with(eid, prefix)).as_py()
            n_dupes = len(eid) - len(pc.unique(eid))
            countries = pa_unique(table, "country")
            n_empty_name = pc.sum(pc.equal(pc.utf8_trim_whitespace(table.column("business_name")), "")).as_py()
            n_empty_addr = pc.sum(pc.equal(pc.utf8_trim_whitespace(table.column("business_address")), "")).as_py()
            print(f"  {split}/{fname}: {len(eid):,} rows, countries={countries}, "
                  f"empty_name={n_empty_name:,}, empty_addr={n_empty_addr:,} ({time.time()-t0:.1f}s)")
            if n_bad_prefix:
                problems.append(f"{fname}: {n_bad_prefix} ids lack the {prefix} prefix")
            if n_dupes:
                problems.append(f"{fname}: {n_dupes} duplicate entity_ids")

    gt_path = os.path.join(args.data_dir, "train", "train_ground_truth.tsv")
    if os.path.isfile(gt_path):
        s1_path = os.path.join(args.data_dir, "train", "train_source1.tsv")
        s1_ids = set(read_source_table(s1_path).column("entity_id").to_pylist())
        gt_ids, n_sing, n_matched, claimed = set(), 0, 0, {}
        multi_claimed = 0
        for row in stream_tsv(gt_path, GROUND_TRUTH_COLUMNS):
            gt_ids.add(row["source1_entity_id"])
            raw = row["matched_entity_ids"]
            if not raw:
                n_sing += 1
                continue
            for mid in raw.split(","):
                n_matched += 1
                if mid in claimed:
                    multi_claimed += 1
                else:
                    claimed[mid] = 1
        print(f"  ground truth: {len(gt_ids):,} entities, {n_sing:,} singletons "
              f"({100*n_sing/len(gt_ids):.2f}%), {n_matched:,} matched ids")
        # The one-to-many property the decision rule will exploit.
        print(f"  one-to-many check: {multi_claimed} S2/S3 ids claimed by >1 S1 entity "
              f"({'HOLDS' if multi_claimed == 0 else 'VIOLATED'})")
        if gt_ids != s1_ids:
            problems.append(f"ground truth covers {len(gt_ids)} ids but source1 has {len(s1_ids)}")

    print()
    if problems:
        print(f"FAIL — {len(problems)} problem(s):")
        for i, p in enumerate(problems, 1):
            print(f"  {i}. {p}")
        return 1
    print("PASS — all schema assertions hold.")
    return 0


def cmd_prep(args) -> int:
    """Convert every source TSV into country-partitioned Parquet."""
    out_dir = os.path.join(args.work_dir, "store")
    # Create first, then check: the space assertion stats the filesystem via the
    # path, so it needs the directory to exist.
    os.makedirs(out_dir, exist_ok=True)
    assert_free_space(args.work_dir, need_gb=6.0)
    for split, files in SPLIT_FILES.items():
        for idx, fname in enumerate(files, start=1):
            path = os.path.join(args.data_dir, split, fname)
            if not os.path.isfile(path):
                continue
            t0 = time.time()
            written = build_columnar_store(path, out_dir, split, f"s{idx}")
            sizes = sum(os.path.getsize(p) for p in written) / 1e6
            print(f"  {split}/s{idx}: {len(written)} partitions, {sizes:.0f} MB ({time.time()-t0:.1f}s)")
    print(f"\nstore at {out_dir}; {free_bytes(args.work_dir)/1e9:.1f} GB free")
    return 0


def cmd_baseline(args) -> int:
    """Rung R0: predict every test Source-1 entity as a singleton.

    Scores only ~the singleton rate, but it exercises the entire output path --
    row count, ordering, headers, encoding -- so a formatting defect is caught
    now rather than against a real submission.
    """
    s1_path = os.path.join(args.data_dir, "test", "test_source1.tsv")
    os.makedirs(args.out, exist_ok=True)
    ids = [r["entity_id"] for r in stream_tsv(s1_path, SOURCE_COLUMNS)]
    n1 = write_id_list_tsv(os.path.join(args.out, "matching_results.tsv"),
                           ["source1_entity_id", "matched_entity_ids"],
                           ((i, ()) for i in ids))
    n2 = write_id_list_tsv(os.path.join(args.out, "candidate_pairs.tsv"),
                           ["source1_entity_id", "candidate_entity_ids"],
                           ((i, ()) for i in ids))
    print(f"  wrote {n1:,} matching rows and {n2:,} candidate rows to {args.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="src.cli")
    ap.add_argument("--data-dir", default="dataset", help="folder containing train/ and test/")
    ap.add_argument("--work-dir", default="work", help="scratch area for derived artefacts")
    ap.add_argument("--out", default="output", help="submission output folder")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("verify", cmd_verify), ("prep", cmd_prep), ("baseline", cmd_baseline)):
        sub.add_parser(name).set_defaults(func=fn)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
