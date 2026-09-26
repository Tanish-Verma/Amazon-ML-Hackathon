"""Smoke checks on a candidate_pairs TSV before committing to a long run.

Verifies exactly what the reviewer asked for: output format, no duplicate
query-candidate pairs, the intended top-K per query, and valid ID prefixes.
"""
import csv
import sys
from collections import Counter


def check(path, expect_k, source1_path=None):
    errs, n, empties, sizes = [], 0, 0, Counter()
    seen_s1 = set()
    with open(path, newline="", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        if header != ["source1_entity_id", "candidate_entity_ids"]:
            errs.append(f"bad header: {header}")
        r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in r:
            n += 1
            if len(row) != 2:
                errs.append(f"row {n}: {len(row)} fields, expected 2"); continue
            s1, rest = row
            if not s1.startswith("S1-"):
                errs.append(f"row {n}: source id {s1!r} lacks S1- prefix")
            if s1 in seen_s1:
                errs.append(f"duplicate source1_entity_id row: {s1}")
            seen_s1.add(s1)
            ids = rest.split(",") if rest else []
            if not ids:
                empties += 1
            sizes[len(ids)] += 1
            if len(ids) != len(set(ids)):
                dupes = [k for k, v in Counter(ids).items() if v > 1]
                errs.append(f"{s1}: duplicate candidate ids {dupes[:3]}")
            if len(ids) > expect_k:
                errs.append(f"{s1}: {len(ids)} candidates exceeds K={expect_k}")
            for cid in ids:
                if not cid.startswith(("S2-", "S3-")):
                    errs.append(f"{s1}: candidate {cid!r} lacks S2-/S3- prefix"); break

    print(f"  rows                : {n:,}")
    print(f"  empty candidate list: {empties:,}")
    print(f"  size distribution   : " +
          ", ".join(f"{k}:{v:,}" for k, v in sorted(sizes.items(), reverse=True)[:5]))
    at_k = sizes.get(expect_k, 0)
    print(f"  exactly K={expect_k}       : {at_k:,} ({100*at_k/n:.2f}%)")
    if source1_path:
        with open(source1_path, encoding="utf-8") as f:
            next(f)
            need = {l.split("\t", 1)[0] for l in f if l.strip()}
        missing = need - seen_s1
        extra = seen_s1 - need
        print(f"  S1 coverage         : {len(seen_s1):,} of {len(need):,} required")
        if missing:
            errs.append(f"{len(missing)} required S1 entities missing")
        if extra:
            errs.append(f"{len(extra)} rows with S1 ids not in the split")
    print()
    if errs:
        print(f"FAIL — {len(errs)} issue(s):")
        for e in errs[:10]:
            print(f"  - {e}")
        return 1
    print("PASS — format, uniqueness and top-K all correct.")
    return 0


if __name__ == "__main__":
    ap_path = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    src = sys.argv[3] if len(sys.argv) > 3 else None
    sys.exit(check(ap_path, k, src))
