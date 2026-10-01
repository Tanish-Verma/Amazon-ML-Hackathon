"""Phase 6: reproducible stratified entity holdouts; NEVER subsample S2/S3 corpus."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def stable_hash(value: str, seed: int) -> int:
    return int.from_bytes(hashlib.blake2b(f'{seed}:{value}'.encode(), digest_size=8).digest(), 'big')


def stratified_splits(records, dev_size=50_000, confirm_size=300_000, seed=42):
    """records: iterable (s1_id,country,is_singleton). Disjoint, stratified by (country,singleton).
    Deterministic independent of input ordering. Returns {dev:[ids],confirm:[ids]}.
    """
    groups = defaultdict(list)
    seen = set()
    for q, country, singleton in records:
        if q in seen:
            raise ValueError(f'Duplicate Source-1 ID: {q}')
        seen.add(q)
        groups[(country, bool(singleton))].append(str(q))
    total = len(seen)
    if dev_size < 0 or confirm_size < 0 or dev_size + confirm_size > total:
        raise ValueError(f'Requested {dev_size}+{confirm_size} holdout entities from {total}')
    for key, ids in groups.items():
        ids.sort(key=lambda q: (stable_hash(q, seed), q))
    # Largest-remainder apportionment preserves strata without rounding drift.
    def allocate(n, capacities):
        available = sum(capacities.values())
        if n > available:
            raise ValueError('Not enough entities')
        ideal = {k: n * v / available for k, v in capacities.items()}
        counts = {k: int(x) for k, x in ideal.items()}
        remaining = n - sum(counts.values())
        order = sorted(capacities, key=lambda k: (-(ideal[k] - counts[k]), str(k)))
        for k in order[:remaining]:
            counts[k] += 1
        return counts
    dev_n = allocate(dev_size, {k: len(v) for k,v in groups.items()})
    confirm_n = allocate(confirm_size, {k: len(v)-dev_n[k] for k,v in groups.items()})
    dev, confirm = [], []
    for k in sorted(groups):
        ids = groups[k]
        dev.extend(ids[:dev_n[k]])
        confirm.extend(ids[dev_n[k]:dev_n[k]+confirm_n[k]])
    dev.sort(key=lambda q: (stable_hash(q,seed+1),q))
    confirm.sort(key=lambda q: (stable_hash(q,seed+2),q))
    assert len(dev)==dev_size and len(confirm)==confirm_size and not set(dev)&set(confirm)
    return {'dev':dev,'confirm':confirm}


def load_records(gt_path, source1_path):
    from src.io_utils import stream_tsv, GROUND_TRUTH_COLUMNS, SOURCE_COLUMNS
    truth = {}
    for row in stream_tsv(str(gt_path), GROUND_TRUTH_COLUMNS):
        q=row['source1_entity_id']
        if q in truth: raise ValueError(f'Duplicate truth ID {q}')
        truth[q] = not bool(row['matched_entity_ids'])
    seen=set()
    for row in stream_tsv(str(source1_path), SOURCE_COLUMNS):
        q=row['entity_id']
        if q in seen: raise ValueError(f'Duplicate Source-1 ID {q}')
        seen.add(q)
        if q not in truth: raise ValueError(f'Missing truth for {q}')
        yield q,row['country'],truth[q]
    if len(seen)!=len(truth): raise ValueError('Truth has IDs absent from Source-1')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gt',required=True)
    p.add_argument('--source1',required=True,help='FULL training Source-1 TSV')
    p.add_argument('--out',required=True,help='NEW output directory under home')
    p.add_argument('--dev-size',type=int,default=50_000)
    p.add_argument('--confirm-size',type=int,default=300_000)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    out=Path(a.out).expanduser().resolve()
    if not out.is_relative_to(Path.home().resolve()): raise ValueError('Output must be under home')
    if out.exists(): raise FileExistsError(out)
    result=stratified_splits(load_records(a.gt,a.source1),a.dev_size,a.confirm_size,a.seed)
    out.mkdir(parents=True,exist_ok=False)
    for name,ids in result.items():
        with (out/f'{name}_ids.tsv').open('x',encoding='utf-8',newline='') as f:
            w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['source1_entity_id']);w.writerows((q,) for q in ids)
    (out/'manifest.json').write_text(json.dumps({'seed':a.seed,'dev_size':len(result['dev']),'confirm_size':len(result['confirm']),'source1':a.source1,'gt':a.gt,'IMPORTANT':'Use full training S2/S3 corpus for both holdouts. Keep these IDs entirely out of model fitting, calibration and threshold tuning.'},indent=2))
    print('Created',out, {k:len(v) for k,v in result.items()})
if __name__=='__main__':main()
