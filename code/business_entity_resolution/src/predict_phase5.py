"""Disk-backed Phase 5 inference, ONE country per run; no full pair matrix in RAM."""
from __future__ import annotations
import argparse
import csv
import os
import pickle
import sqlite3
import time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from src.features import FEATURE_NAMES


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--features',required=True,help='ONE country Phase 4 test feature folder')
    ap.add_argument('--model',required=True,help='Phase 5 model.pkl')
    ap.add_argument('--out',required=True,help='NEW country output directory under home')
    ap.add_argument('--min-free-gb',type=float,default=25)
    ap.add_argument('--batch',type=int,default=50000)
    args=ap.parse_args()
    import psutil
    print('RAM available GB',psutil.virtual_memory().available/1e9,'swap used GB',psutil.swap_memory().used/1e9,flush=True)
    if psutil.virtual_memory().available < args.min_free_gb*1e9:
        raise RuntimeError('Insufficient available RAM')
    out=Path(args.out).resolve()
    if not str(out).startswith(str(Path.home().resolve())+os.sep):
        raise ValueError('Output must be inside your home directory')
    if out.exists():raise FileExistsError(out)
    paths=sorted(Path(args.features).glob('features_*.parquet'))
    if not paths:raise FileNotFoundError('No Phase 4 feature chunks found')
    with open(args.model,'rb') as f: bundle=pickle.load(f)
    if bundle['features']!=FEATURE_NAMES:raise ValueError('Phase 4/5 feature schema mismatch')
    threshold=float(bundle['threshold']);margin=float(bundle['margin'])
    out.mkdir(parents=True,exist_ok=False)
    # Explicitly keep SQLite temporary files in user's home directory.
    os.environ['TMPDIR']=str(out)
    db=sqlite3.connect(str(out/'scores.sqlite'))
    db.execute('PRAGMA temp_store=FILE')
    db.execute('CREATE TABLE eligible (qid TEXT,cid TEXT,p REAL,PRIMARY KEY(qid,cid)) WITHOUT ROWID')
    db.execute('CREATE TABLE best (qid TEXT PRIMARY KEY,p REAL) WITHOUT ROWID')
    n=0;t0=time.monotonic()
    for path in paths:
        pf=pq.ParquetFile(path)
        missing=set(FEATURE_NAMES+['source1_entity_id','candidate_entity_id'])-set(pf.schema_arrow.names)
        if missing:raise ValueError(f'{path} missing {missing}')
        for batch in pf.iter_batches(batch_size=args.batch,columns=FEATURE_NAMES+['source1_entity_id','candidate_entity_id']):
            d=batch.to_pydict()
            X=np.column_stack([d[name] for name in FEATURE_NAMES]).astype('float32')
            if not np.isfinite(X).all():raise ValueError(f'Nonfinite feature in {path}')
            p=bundle['calibrator'].predict(bundle['model'].predict_proba(X)[:,1])
            # Only p >= threshold can ever be accepted. Best among those is identical
            # to best over ALL pairs for any entity with an accepted candidate.
            rows=[(str(q),str(c),float(prob)) for q,c,prob in zip(d['source1_entity_id'],d['candidate_entity_id'],p) if prob>=threshold]
            db.executemany('INSERT INTO eligible VALUES(?,?,?)',rows)
            db.executemany('INSERT INTO best VALUES(?,?) ON CONFLICT(qid) DO UPDATE SET p=max(p,excluded.p)',[(q,v) for q,_,v in rows])
            db.commit()
            n+=len(p)
            if n%500000 < args.batch:
                print(f'{n:,} pairs scored; {n/max(time.monotonic()-t0,1):,.0f} pairs/sec; '
                      f'free RAM {psutil.virtual_memory().available/1e9:.1f}GB; '
                      f'swap {psutil.swap_memory().used/1e9:.1f}GB',flush=True)
                if psutil.virtual_memory().available < args.min_free_gb*1e9:
                    raise RuntimeError('RAM guard: stopping before next batch')
    db.execute('CREATE INDEX idx_score ON eligible(p DESC,cid,qid)')
    db.commit()
    db.execute('CREATE TABLE owners (cid TEXT PRIMARY KEY,qid TEXT) WITHOUT ROWID')
    db.execute('CREATE TABLE accepted (qid TEXT,cid TEXT,PRIMARY KEY(qid,cid)) WITHOUT ROWID')
    # Global descending score across the WHOLE country, not each Parquet chunk.
    cursor=db.execute('SELECT e.qid,e.cid,e.p FROM eligible e JOIN best b USING(qid) '
                      'WHERE e.p >= b.p - ? ORDER BY e.p DESC,e.cid,e.qid',(margin,))
    accepted=0
    for q,c,p in cursor:
        if db.execute('INSERT OR IGNORE INTO owners VALUES(?,?)',(c,q)).rowcount:
            db.execute('INSERT INTO accepted VALUES(?,?)',(q,c));accepted+=1
        if accepted%100000==0 and accepted:db.commit()
    db.commit()
    # Only nonempty query rows here; final assembly must add all missing S1 IDs.
    with (out/'matches_nonempty.tsv').open('x',newline='') as f:
        w=csv.writer(f,delimiter='\t',quoting=csv.QUOTE_NONE)
        w.writerow(['source1_entity_id','matched_entity_ids'])
        qprev=None;ids=[]
        for q,c in db.execute('SELECT qid,cid FROM accepted ORDER BY qid,cid'):
            if qprev is not None and q!=qprev:
                w.writerow([qprev,','.join(ids)]);ids=[]
            qprev=q;ids.append(c)
        if qprev is not None:w.writerow([qprev,','.join(ids)])
    print(f'Finished {n:,} pairs; {accepted:,} accepted matches. Output: {out}/matches_nonempty.tsv',flush=True)
    db.close()
if __name__=='__main__':main()
