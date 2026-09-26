"""
Phase 2 verification against real data on real hardware.

The notebook's own 19 checks all passed, but every one used invented strings. The
questions that actually decide whether Phase 2 is usable are:

  1. Does it survive 12.5M real records without crashing or mangling anything?
  2. What throughput does it sustain on the real Xeon (PLAN.md wants ~100k/s/core)?
  3. Do DF-derived stoplists behave on REAL per-country corpora -- in particular,
     does France's sarl/sas actually get caught, and does the default 2%
     threshold throw away words we need?
  4. REGRESSION: does using this normaliser preserve the 99.88% lexical
     reachability ceiling the whole plan is built on? If normalisation quietly
     lowers that, every later phase inherits the loss.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from collections import Counter

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.normalize import (  # noqa: E402
    DocumentFrequencyModel, char_ngrams, clean_text, extract_numeric_keys,
    normalize_record, tokenize,
)

NAME_REACHABLE = 0.15   # identical thresholds to reports/eda.md, for comparability
ADDR_REACHABLE = 0.20


def jac(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def load(store, split, source):
    recs = {}
    for fn in sorted(os.listdir(store)):
        if fn.startswith(f"{split}_{source}_") and fn.endswith(".parquet"):
            t = pq.read_table(os.path.join(store, fn))
            recs.update(zip(t.column("entity_id").to_pylist(),
                            zip(t.column("business_name").to_pylist(),
                                t.column("business_address").to_pylist(),
                                t.column("country").to_pylist())))
    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--sample-entities", type=int, default=150_000)
    args = ap.parse_args()

    print("=" * 72)
    print("1. CRASH / INTEGRITY SWEEP over every real record")
    print("=" * 72)
    swept = crashes = 0
    empty_clean = 0
    t0 = time.perf_counter()
    for split in ("train", "test"):
        for src in ("s1", "s2", "s3"):
            recs = load(args.store, split, src)
            if not recs:
                continue
            for name, addr, _c in recs.values():
                try:
                    c = clean_text(name)
                    extract_numeric_keys(addr)
                    if not c and name.strip():
                        empty_clean += 1
                except Exception as e:      # noqa: BLE001
                    crashes += 1
                    if crashes <= 3:
                        print(f"  CRASH {name!r}: {e}")
                swept += 1
            print(f"  {split}/{src}: {len(recs):,} swept")
            del recs
    el = time.perf_counter() - t0
    print(f"\n  total {swept:,} records, {crashes} crashes, "
          f"{empty_clean:,} non-empty names normalising to empty "
          f"({100*empty_clean/swept:.4f}%)")
    print(f"  single-core throughput: {swept/el:,.0f} records/sec over the full sweep")

    print()
    print("=" * 72)
    print("2. DF STOPLISTS FROM REAL PER-COUNTRY CORPORA (test split, incl. France)")
    print("=" * 72)
    dfm = DocumentFrequencyModel()
    s1t = load(args.store, "test", "s1")
    for name, _a, ctry in s1t.values():
        dfm.update(ctry, tokenize(clean_text(name)))
    for thresh in (0.02, 0.05, 0.10):
        sl = dfm.build_stoplists(min_df_fraction=thresh)
        print(f"\n  min_df_fraction={thresh}:")
        for ctry in sorted(sl):
            toks = sorted(sl[ctry], key=lambda t: -dfm.df_fraction(ctry, t))
            shown = ", ".join(f"{t}({100*dfm.df_fraction(ctry,t):.0f}%)" for t in toks[:12])
            print(f"    {ctry:8s} {len(sl[ctry]):3d} tokens: {shown}")

    print()
    print("=" * 72)
    print("3. REACHABILITY REGRESSION vs reports/eda.md (99.88% ceiling)")
    print("=" * 72)
    gt = {}
    with open(args.gt, encoding="utf-8") as f:
        next(f)
        for line in f:
            a, _, b = line.partition("\t")
            b = b.rstrip("\n")
            if b:
                gt[a] = b.split(",")
    s1 = load(args.store, "train", "s1")
    s23 = load(args.store, "train", "s2")
    s23.update(load(args.store, "train", "s3"))

    rng = random.Random(0)
    keys = [k for k in gt if k in s1]
    rng.shuffle(keys)
    keys = keys[:args.sample_entities]

    st = Counter()
    for s1id in keys:
        n1, a1, _c = s1[s1id]
        g1 = set(char_ngrams(clean_text(n1)))
        t1 = set(tokenize(clean_text(a1)))
        k1 = extract_numeric_keys(a1)
        for mid in gt[s1id]:
            r = s23.get(mid)
            if r is None:
                continue
            n2, a2, _ = r
            name_ok = jac(g1, set(char_ngrams(clean_text(n2)))) >= NAME_REACHABLE
            a2c = clean_text(a2)
            addr_ok = (jac(t1, set(tokenize(a2c))) >= ADDR_REACHABLE
                       or bool(k1 & extract_numeric_keys(a2))) if a2.strip() else False
            st["pairs"] += 1
            st["name"] += name_ok
            st["addr"] += addr_ok
            st["either"] += (name_ok or addr_ok)
    p = st["pairs"]
    print(f"  sample: {len(keys):,} entities, {p:,} true pairs\n")
    print(f"  {'channel':<28}{'this run':>12}{'eda.md':>12}{'delta':>10}")
    for label, key, base in (("name", "name", 90.43), ("address", "addr", 94.82),
                             ("EITHER (ceiling)", "either", 99.88)):
        got = 100 * st[key] / p
        print(f"  {label:<28}{got:>11.2f}%{base:>11.2f}%{got-base:>+9.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
