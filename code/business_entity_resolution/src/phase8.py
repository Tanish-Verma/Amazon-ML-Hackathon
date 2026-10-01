"""Phase 8: run country inference, assemble exact TSVs and audit submission.

No overwrites/deletes. All writable paths must be under the invoking user's home.
Run from code/business_entity_resolution: python -m src.phase8 --help
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

CAND_HEADER = ['source1_entity_id', 'candidate_entity_ids']
MATCH_HEADER = ['source1_entity_id', 'matched_entity_ids']
EXPECTED = 1_732_544


def home_path(value):
    p = Path(value).expanduser().resolve()
    home = Path.home().resolve()
    if p == home or home not in p.parents:
        raise ValueError(f'Writable path must be BELOW home {home}: {p}')
    return p


def rows(path, header):
    with Path(path).open('r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        got = next(reader, None)
        if got != header:
            raise ValueError(f'{path}: expected header {header}, got {got}')
        for line, row in enumerate(reader, 2):
            if len(row) != 2:
                raise ValueError(f'{path}:{line}: expected 2 tab-separated fields')
            yield row


def ids(value):
    if not value:
        return []
    parts = value.split(',')
    if any(not p or p != p.strip() or not p.startswith(('S2-', 'S3-')) for p in parts):
        raise ValueError(f'Invalid candidate/match ID list: {value[:120]!r}')
    if len(parts) != len(set(parts)):
        raise ValueError(f'Duplicate ID within list: {value[:120]!r}')
    return parts


def test_ids(store):
    import pyarrow.parquet as pq
    paths = sorted(Path(store).glob('test_s1_*.parquet'))
    if not paths:
        raise FileNotFoundError(f'No test_s1_*.parquet in {store}')
    for path in paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=50_000, columns=['entity_id']):
            for eid in batch.column(0).to_pylist():
                if not isinstance(eid, str) or not eid.startswith('S1-'):
                    raise ValueError(f'Invalid S1 ID {eid!r} in {path}')
                yield eid


def candidate_db(conn, paths):
    conn.execute('CREATE TABLE candidate (qid TEXT PRIMARY KEY, vals TEXT NOT NULL) WITHOUT ROWID')
    count = 0
    for path in paths:
        for q, value in rows(path, CAND_HEADER):
            if not q.startswith('S1-'):
                raise ValueError(f'Bad S1 ID {q}')
            ids(value)
            conn.execute('INSERT INTO candidate VALUES (?,?)', (q, value))
            count += 1
        conn.commit()
    return count


def match_db(conn, paths):
    conn.execute('CREATE TABLE match (qid TEXT PRIMARY KEY, vals TEXT NOT NULL) WITHOUT ROWID')
    for path in paths:
        for q, value in rows(path, MATCH_HEADER):
            if not q.startswith('S1-'):
                raise ValueError(f'Bad S1 ID {q}')
            ids(value)
            conn.execute('INSERT INTO match VALUES (?,?)', (q, value))
        conn.commit()


def assemble(store, candidates, matches, out, expected=EXPECTED):
    """SQLite index bounds RAM; outputs exactly one row per test S1, incl. singletons.

    All country candidate TSVs must contain the COMPLETE set of S1 IDs for their
    partition. matches_nonempty.tsv may omit singleton rows.
    """
    import sqlite3
    out = home_path(out)
    if out.exists():
        raise FileExistsError(out)
    if not candidates or not matches:
        raise ValueError('Supply country candidate and match files')
    out.mkdir(parents=True, exist_ok=False)
    os.environ['TMPDIR'] = str(out)
    conn = sqlite3.connect(out / 'assembly.sqlite')
    conn.execute('PRAGMA temp_store=FILE')
    conn.execute('PRAGMA journal_mode=DELETE')
    t0 = time.monotonic()
    candidate_db(conn, candidates)
    match_db(conn, matches)
    conn.execute('CREATE TABLE seen (qid TEXT PRIMARY KEY) WITHOUT ROWID')
    n = n_single = n_matched = 0
    with (out / 'candidate_pairs.tsv').open('x', encoding='utf-8', newline='') as cf, \
         (out / 'matching_results.tsv').open('x', encoding='utf-8', newline='') as mf:
        cw = csv.writer(cf, delimiter='\t', quoting=csv.QUOTE_NONE, lineterminator='\n')
        mw = csv.writer(mf, delimiter='\t', quoting=csv.QUOTE_NONE, lineterminator='\n')
        cw.writerow(CAND_HEADER)
        mw.writerow(MATCH_HEADER)
        for q in test_ids(store):
            conn.execute('INSERT INTO seen VALUES (?)', (q,))
            c = conn.execute('SELECT vals FROM candidate WHERE qid=?', (q,)).fetchone()
            if c is None:
                raise ValueError(f'Missing candidate row for test entity {q}; do not fabricate an empty list')
            m = conn.execute('SELECT vals FROM match WHERE qid=?', (q,)).fetchone()
            cset = set(ids(c[0]))
            mlist = ids(m[0]) if m else []
            if not set(mlist) <= cset:
                raise ValueError(f'Matches not in candidate list for {q}: {set(mlist)-cset}')
            cw.writerow((q, c[0]))
            mw.writerow((q, ','.join(mlist)))
            n += 1
            n_single += not bool(mlist)
            n_matched += len(mlist)
            if n % 50_000 == 0:
                conn.commit()
                print(f'Assembled {n:,} entities', flush=True)
        conn.commit()
    for table in ('candidate', 'match'):
        unknown = conn.execute(f'SELECT COUNT(*) FROM {table} WHERE qid NOT IN (SELECT qid FROM seen)').fetchone()[0]
        if unknown:
            raise ValueError(f'{unknown} unknown S1 IDs in {table} files')
    if n != expected:
        raise ValueError(f'Expected {expected:,} test S1 entities, found {n:,}')
    metrics = {'entities': n, 'predicted_singletons': n_single,
               'singleton_rate': n_single / n, 'matched_ids': n_matched,
               'assembly_seconds': round(time.monotonic() - t0, 2)}
    (out / 'assembly_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
    conn.close()
    print(json.dumps(metrics, indent=2))
    return metrics


def audit(candidate_file, match_file, store=None, expected=EXPECTED, check_ids=False):
    """Streaming synchronized audit, O(max candidates for one S1) RAM.

    If store is provided, also verifies test S1 coverage; with --check-ids,
    checks membership in the partitioned test S2/S3 corpus (higher memory).
    """
    import itertools
    from collections import defaultdict
    from pathlib import Path
    from tempfile import TemporaryDirectory
    import sqlite3
    if check_ids and not store:
        raise ValueError('--check-ids requires --store')
    # SQLite disk index is inside home; avoids storing 1.7M query IDs in Python.
    import tempfile
    with tempfile.TemporaryDirectory(prefix='phase8_audit_', dir=str(Path.home())) as td:
        db = sqlite3.connect(Path(td) / 'audit.sqlite')
        db.execute('CREATE TABLE seen (qid TEXT PRIMARY KEY) WITHOUT ROWID')
        db.execute('CREATE TABLE owners (cid TEXT PRIMARY KEY, qid TEXT) WITHOUT ROWID')
        valid = None
        if check_ids:
            import pyarrow.parquet as pq
            db.execute('CREATE TABLE valid (cid TEXT PRIMARY KEY) WITHOUT ROWID')
            files = sorted(Path(store).glob('test_s[23]_*.parquet'))
            if not files:
                raise FileNotFoundError('No test_s2/test_s3 parquet files')
            for file in files:
                for batch in pq.ParquetFile(file).iter_batches(batch_size=50_000, columns=['entity_id']):
                    db.executemany('INSERT INTO valid VALUES (?)', ((x,) for x in batch.column(0).to_pylist()))
                db.commit()
        n = single = pairs = 0
        crows = rows(candidate_file, CAND_HEADER)
        mrows = rows(match_file, MATCH_HEADER)
        for cr, mr in itertools.zip_longest(crows, mrows):
            if cr is None or mr is None:
                raise ValueError('Candidate and matching files have different row counts')
            q, cv = cr
            mq, mv = mr
            if q != mq or not q.startswith('S1-'):
                raise ValueError(f'Rows out of sync or invalid: {q!r}, {mq!r}')
            db.execute('INSERT INTO seen VALUES (?)', (q,))
            cids = ids(cv)
            mids = ids(mv)
            if not set(mids).issubset(cids):
                raise ValueError(f'{q}: matched ID missing from candidate list')
            for mid in mids:
                try:
                    db.execute('INSERT INTO owners VALUES (?,?)', (mid, q))
                except sqlite3.IntegrityError:
                    raise ValueError(f'Candidate assigned to multiple S1 entities: {mid}')
            if check_ids:
                for cid in cids:
                    if db.execute('SELECT 1 FROM valid WHERE cid=?', (cid,)).fetchone() is None:
                        raise ValueError(f'Unknown S2/S3 ID: {cid}')
            n += 1
            single += not bool(mids)
            pairs += len(cids)
            if n % 50_000 == 0:
                db.commit()
        if n != expected:
            raise ValueError(f'Expected {expected:,} rows, found {n:,}')
        if store:
            for q in test_ids(store):
                if db.execute('SELECT 1 FROM seen WHERE qid=?', (q,)).fetchone() is None:
                    raise ValueError(f'Missing test S1 entity {q}')
            db.commit()
        print(json.dumps({'PASS': True, 'entities': n, 'candidate_pairs': pairs,
                          'singleton_rate': single/n}, indent=2))


def run(args):
    """End-to-end test inference; one country at a time, no silent reuse."""
    import psutil
    out = home_path(args.out)
    if out.exists():
        raise FileExistsError(out)
    if psutil.virtual_memory().available < args.min_free_gb * 1e9:
        raise RuntimeError('Insufficient free RAM before starting')
    if not Path(args.store).is_dir():
        raise FileNotFoundError(args.store)
    out.mkdir(parents=True, exist_ok=False)
    os.environ['TMPDIR'] = str(out)
    log = []
    for country in args.countries:
        safe = country.replace('/', '_').replace(' ', '_')
        cdir = out / safe
        cdir.mkdir(exist_ok=False)
        candidate = cdir / 'candidates.tsv'
        enriched = cdir / 'enriched.tsv'
        features = cdir / 'features'
        predicted = cdir / 'predicted'
        steps = [
            [sys.executable, '-m', 'src.run_blocking_enriched', '--store', args.store,
             '--split', 'test', '--countries', country, '--model', args.blocking_model,
             '--out', str(candidate), '--metadata-out', str(enriched),
             '--top-k', str(args.top_k), '--workers', str(args.workers),
             '--batch', str(args.block_batch)],
            [sys.executable, '-m', 'src.run_features', '--pairs', str(enriched),
             '--store', args.store, '--split', 'test', '--country', country,
             '--out', str(features), '--workers', str(args.workers),
             '--query-batch', str(args.feature_batch), '--min-free-gb', str(args.min_free_gb)],
            [sys.executable, '-m', 'src.predict_phase5', '--features', str(features),
             '--model', args.matching_model, '--out', str(predicted),
             '--batch', str(args.score_batch), '--min-free-gb', str(args.min_free_gb)],
        ]
        for step in steps:
            if psutil.virtual_memory().available < args.min_free_gb * 1e9:
                raise RuntimeError('RAM guard before stage')
            print('RUN:', ' '.join(step), flush=True)
            start = time.monotonic()
            subprocess.run(step, check=True, env={**os.environ, 'TMPDIR': str(cdir)})
            log.append({'country': country, 'stage': step[2],
                        'seconds': round(time.monotonic() - start, 1),
                        'rss_peak_gb': None})
            (out / 'runtime.json').write_text(json.dumps(log, indent=2), encoding='utf-8')
    assemble(args.store,
             [out / c.replace('/', '_').replace(' ', '_') / 'candidates.tsv' for c in args.countries],
             [out / c.replace('/', '_').replace(' ', '_') / 'predicted' / 'matches_nonempty.tsv' for c in args.countries],
             out / 'submission', args.expected)
    audit(out / 'submission' / 'candidate_pairs.tsv',
          out / 'submission' / 'matching_results.tsv', args.store, args.expected,
          check_ids=args.check_ids)
    print('Internal audit passed. Run the OFFICIAL validator separately; only its PASS completes Phase 8.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('assemble', help='assemble existing per-country outputs')
    a.add_argument('--store', required=True)
    a.add_argument('--candidates', nargs='+', required=True)
    a.add_argument('--matches', nargs='+', required=True)
    a.add_argument('--out', required=True)
    a.add_argument('--expected', type=int, default=EXPECTED)
    v = sub.add_parser('audit', help='validate TSVs without running inference')
    v.add_argument('--candidates', required=True)
    v.add_argument('--matches', required=True)
    v.add_argument('--store')
    v.add_argument('--expected', type=int, default=EXPECTED)
    v.add_argument('--check-ids', action='store_true')
    r = sub.add_parser('run', help='run phases 3-5 per country then assemble')
    r.add_argument('--store', required=True)
    r.add_argument('--blocking-model', required=True)
    r.add_argument('--matching-model', required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--countries', nargs='+', required=True)
    r.add_argument('--top-k', type=int, default=50)
    r.add_argument('--workers', type=int, default=8)
    r.add_argument('--block-batch', type=int, default=20_000)
    r.add_argument('--feature-batch', type=int, default=100)
    r.add_argument('--score-batch', type=int, default=50_000)
    r.add_argument('--min-free-gb', type=float, default=25)
    r.add_argument('--expected', type=int, default=EXPECTED)
    r.add_argument('--check-ids', action='store_true')
    args = p.parse_args()
    if args.command == 'assemble':
        assemble(args.store, args.candidates, args.matches, args.out, args.expected)
    elif args.command == 'audit':
        audit(args.candidates, args.matches, args.store, args.expected, args.check_ids)
    else:
        run(args)


if __name__ == '__main__':
    main()
