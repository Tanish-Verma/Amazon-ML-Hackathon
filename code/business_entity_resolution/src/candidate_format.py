"""Inspect, expand and cross-check official two-column candidate_pairs.tsv.

Never invent retrieval scores. Optional enriched sidecar must contain the exact
per-pair Phase 3 metadata required by Phase 4.
"""
from __future__ import annotations
import argparse
import csv
import json
import sqlite3
from collections import Counter
from pathlib import Path

CANDIDATE_HEADER = ['source1_entity_id', 'candidate_entity_ids']
ENRICHED_HEADER = ['source1_entity_id', 'candidate_entity_id', 'rrf_score',
                   'n_channels_hit', 'best_channel_rank', 'reranker_score']


def candidate_rows(path):
    with open(path, encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        header = next(reader, None)
        if header != CANDIDATE_HEADER:
            raise ValueError(f'{path}: expected {CANDIDATE_HEADER}, got {header}')
        for line, row in enumerate(reader, 2):
            if len(row) != 2 or not row[0].startswith('S1-'):
                raise ValueError(f'{path}:{line}: invalid two-column S1 row')
            ids = row[1].split(',') if row[1] else []
            if any(not x.startswith(('S2-', 'S3-')) or x != x.strip() for x in ids):
                raise ValueError(f'{path}:{line}: invalid candidate ID')
            if len(ids) != len(set(ids)):
                raise ValueError(f'{path}:{line}: duplicate candidate ID')
            yield row[0], ids


def inspect(path, expected=None):
    seen = set()
    counts = Counter()
    n_pairs = 0
    for q, candidates in candidate_rows(path):
        if q in seen:
            raise ValueError(f'Duplicate S1 entity: {q}')
        seen.add(q)
        n_pairs += len(candidates)
        counts[len(candidates)] += 1
    if expected is not None and len(seen) != expected:
        raise ValueError(f'Expected {expected} S1 rows, got {len(seen)}')
    return {'s1_rows': len(seen), 'candidate_pairs': n_pairs,
            'min_candidates': min(counts) if counts else 0,
            'max_candidates': max(counts) if counts else 0,
            'empty_candidate_lists': counts[0]}


def expand(input_path, output_path):
    """Export ID-only long form. NOT a Phase 4 enriched feature input."""
    out = Path(output_path)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        writer.writerow(['source1_entity_id', 'candidate_entity_id', 'candidate_position'])
        for q, candidates in candidate_rows(input_path):
            for rank, cid in enumerate(candidates, 1):
                writer.writerow([q, cid, rank])


def verify_enriched(candidate_path, enriched_path, db_path):
    """Verify exact pair-set equality using disk-backed SQLite; no pair-list RAM blowup.

    Requires unique q,c in the enriched file; its score columns are validated as
    numeric/finite. Database path must be new. This does NOT generate metadata.
    """
    import math
    db_path = Path(db_path)
    if db_path.exists():
        raise FileExistsError(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(db_path)
    try:
        db.execute('CREATE TABLE pairs (q TEXT NOT NULL, c TEXT NOT NULL, matched INTEGER DEFAULT 0, PRIMARY KEY(q,c)) WITHOUT ROWID')
        for q, cs in candidate_rows(candidate_path):
            db.executemany('INSERT INTO pairs(q,c) VALUES (?,?)', ((q, c) for c in cs))
        db.commit()
        with open(enriched_path, encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            if reader.fieldnames != ENRICHED_HEADER:
                raise ValueError(f'Expected enriched header {ENRICHED_HEADER}; got {reader.fieldnames}')
            count = 0
            for row in reader:
                q, c = row['source1_entity_id'], row['candidate_entity_id']
                for col in ENRICHED_HEADER[2:]:
                    if not math.isfinite(float(row[col])):
                        raise ValueError(f'Nonfinite {col} for {q}/{c}')
                result = db.execute('UPDATE pairs SET matched=matched+1 WHERE q=? AND c=?', (q, c))
                if result.rowcount != 1:
                    raise ValueError(f'Enriched pair absent from candidate file: {q}/{c}')
                count += 1
                if count % 100_000 == 0:
                    db.commit()
        db.commit()
        missing = db.execute('SELECT COUNT(*) FROM pairs WHERE matched=0').fetchone()[0]
        duplicate = db.execute('SELECT COUNT(*) FROM pairs WHERE matched>1').fetchone()[0]
        if missing or duplicate:
            raise ValueError(f'Sidecar mismatch: {missing} missing pairs, {duplicate} duplicated pairs')
        return {'verified_pairs': count, 'exact_pair_set': True}
    finally:
        db.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('inspect'); p.add_argument('--candidates', required=True); p.add_argument('--expected', type=int)
    p = sub.add_parser('expand'); p.add_argument('--candidates', required=True); p.add_argument('--out', required=True)
    p = sub.add_parser('verify-sidecar'); p.add_argument('--candidates', required=True)
    p.add_argument('--enriched', required=True); p.add_argument('--db', required=True)
    a = ap.parse_args()
    if a.command == 'inspect': result = inspect(a.candidates, a.expected)
    elif a.command == 'expand': expand(a.candidates, a.out); result = {'expanded_to': a.out}
    else: result = verify_enriched(a.candidates, a.enriched, a.db)
    print(json.dumps(result, indent=2))

if __name__ == '__main__': main()
