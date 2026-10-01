"""Phase 5 LightGBM model, entity-level split, isotonic calibration, validation."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from src.features import FEATURE_NAMES
from src.decide import score_decisions
from src.io_utils import read_ground_truth


def split_entity(qid: str) -> int:
    """Stable entity split: 0 fit (70%), 1 calibration (15%), 2 tuning (15%)."""
    v = int.from_bytes(hashlib.blake2b(str(qid).encode(), digest_size=8).digest(), 'big') % 100
    return 0 if v < 70 else (1 if v < 85 else 2)


def feature_files(root):
    paths = sorted(Path(root).glob('*/features_*.parquet'))
    if not paths:
        paths = sorted(Path(root).glob('features_*.parquet'))
    if not paths:
        raise FileNotFoundError(f'No Phase 4 Parquet chunks under {root}')
    return paths


def collect_train(root, max_rows=2_000_000, batch_size=65536, seed=42):
    """Bounded per-split reservoir, retaining all three split populations.

    Reservoir sampling is over candidate PAIRS; holdout assignment is by ENTITY.
    For exact macro-F0.5 tuning, collect ALL pairs for the sampled tuning entities
    in a subsequent pass; never score truncated entity candidate sets.
    """
    rng = np.random.default_rng(seed)
    per_split = {0: [], 1: [], 2: []}
    seen = [0,0,0]
    caps = [int(max_rows*.7), int(max_rows*.15), max_rows-int(max_rows*.85)]
    for path in feature_files(root):
        pf = pq.ParquetFile(path)
        missing = set(FEATURE_NAMES + ['source1_entity_id','candidate_entity_id','label']) - set(pf.schema_arrow.names)
        if missing:
            raise ValueError(f'{path}: missing {missing}; run labelled Phase 4 first')
        for b in pf.iter_batches(batch_size=batch_size, columns=FEATURE_NAMES+['source1_entity_id','candidate_entity_id','label']):
            d = b.to_pydict()
            X = np.column_stack([d[n] for n in FEATURE_NAMES]).astype('float32')
            if not np.isfinite(X).all():
                raise ValueError(f'Nonfinite features in {path}')
            for i,q in enumerate(d['source1_entity_id']):
                k = split_entity(q)
                seen[k] += 1
                row = (str(q),str(d['candidate_entity_id'][i]), X[i],int(d['label'][i]),str(path.parent.name))
                if len(per_split[k]) < caps[k]:
                    per_split[k].append(row)
                else:
                    j = int(rng.integers(seen[k]))
                    if j < caps[k]:
                        per_split[k][j] = row
    return per_split, seen


def full_entities(root, selected_qids):
    """Second pass retrieves COMPLETE candidate lists for sampled tune entities."""
    rows=[]
    for path in feature_files(root):
        pf=pq.ParquetFile(path)
        for b in pf.iter_batches(batch_size=65536,columns=FEATURE_NAMES+['source1_entity_id','candidate_entity_id','label']):
            d=b.to_pydict()
            for i,q in enumerate(d['source1_entity_id']):
                if q in selected_qids:
                    rows.append((q,str(d['candidate_entity_id'][i]),
                                 np.asarray([d[n][i] for n in FEATURE_NAMES],dtype='float32'),
                                 int(d['label'][i]),path.parent.name))
    return rows


def fit(args):
    import lightgbm as lgb
    from src.scoring import macro_f05
    if Path(args.out).exists():
        raise FileExistsError('Choose a NEW --out path; existing files are never overwritten')
    if not str(Path(args.out).resolve()).startswith(str(Path.home().resolve())+os.sep):
        raise ValueError('--out must be under your home directory')
    import psutil
    free=psutil.virtual_memory().available/1e9
    print(f'Free RAM {free:.1f} GB, swap used {psutil.swap_memory().used/1e9:.1f} GB',flush=True)
    if free < args.min_free_gb:
        raise RuntimeError('Insufficient free RAM')
    groups,counts=collect_train(args.features,args.max_rows)
    print('Available pairs per split:',counts,'sampled:',{k:len(v) for k,v in groups.items()},flush=True)
    if not all(groups.values()):
        raise ValueError('Empty split: need more labelled training pairs')
    fit_rows=groups[0]
    cal_rows=groups[1]
    X=np.stack([r[2] for r in fit_rows]); y=np.array([r[3] for r in fit_rows],dtype='uint8')
    Xc=np.stack([r[2] for r in cal_rows]); yc=np.array([r[3] for r in cal_rows],dtype='uint8')
    if len(np.unique(y))<2 or len(np.unique(yc))<2:
        raise ValueError('Both fit and calibration need positive and negative pairs')
    model=lgb.LGBMClassifier(n_estimators=450,learning_rate=.045,num_leaves=31,
                              max_depth=-1,min_child_samples=80,colsample_bytree=.9,
                              subsample=.9,subsample_freq=1,reg_lambda=2.,n_jobs=args.threads,
                              random_state=42,verbosity=-1)
    model.fit(X,y,feature_name=FEATURE_NAMES)
    raw=model.predict_proba(Xc)[:,1]
    calibrator=IsotonicRegression(out_of_bounds='clip',y_min=0,y_max=1)
    calibrator.fit(raw,yc)
    # Select complete entities, not a random subset of individual pairs.
    tune_entities=sorted({r[0] for r in groups[2]})[:args.tune_entities]
    tune=full_entities(args.features,set(tune_entities))
    Xt=np.stack([r[2] for r in tune]); yt=np.array([r[3] for r in tune],dtype='uint8')
    prob=calibrator.predict(model.predict_proba(Xt)[:,1])
    print('fit positive rate',float(y.mean()),'cal positive rate',float(yc.mean()),flush=True)
    print('tune pair AUC',float(roc_auc_score(yt,prob)) if len(np.unique(yt))==2 else None,flush=True)
    print('tune average precision',float(average_precision_score(yt,prob)),flush=True)
    rows_by_country={}
    for (q,c,_,_,country),p in zip(tune,prob):
        rows_by_country.setdefault(country,[]).append((q,c,float(p)))
    gt=read_ground_truth(args.gt)
    truth={q:gt[q] for q in tune_entities if q in gt}
    if len(truth)!=len(tune_entities):
        raise ValueError('Missing truth rows for tune entities')
    # Overall tuning across all country partitions; competition never crosses countries.
    best=None; grid=[]
    for threshold in (0.35,0.50,0.65,0.75,0.82,0.88,0.92,0.95,0.97,0.985):
        for margin in (0.0,0.03,0.07,0.12,0.20,0.35,1.0):
            for exclusive in (False,True):
                from src.decide import decide
                predictions={}
                for country, rows in rows_by_country.items():
                    predictions.update(decide(rows,threshold,margin,exclusive))
                metrics=macro_f05(predictions,truth)
                entry=dict(threshold=threshold,margin=margin,exclusive=exclusive,**metrics)
                grid.append(entry)
                if exclusive and (best is None or metrics['macro_f05']>best['macro_f05']):
                    best=entry
    # Report per-country performance using the same global chosen parameters.
    from src.decide import decide
    by_country={}
    for country,rows in rows_by_country.items():
        qs={q for q,_,_ in rows}
        t={q:truth[q] for q in qs}
        by_country[country]=macro_f05(decide(rows,best['threshold'],best['margin'],True),t)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    with (out/'model.pkl').open('xb') as f:
        pickle.dump({'model':model,'calibrator':calibrator,'features':FEATURE_NAMES,
                     'threshold':best['threshold'],'margin':best['margin']},f)
    report={'sample_counts':counts,'fit_positive_rate':float(y.mean()),
            'cal_positive_rate':float(yc.mean()),'tune_entities':len(tune_entities),
            'best':best,'per_country':by_country,
            'ablation':sorted(grid,key=lambda x:-x['macro_f05'])[:20],
            'note':'Only sampled validation; verify against official scorer and larger holdout.'}
    (out/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=True))
    print(json.dumps({'best':best,'per_country':by_country},indent=2),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--features',required=True,help='labelled train Phase 4 Parquet directory')
    p.add_argument('--gt',required=True,help='train ground truth TSV')
    p.add_argument('--out',required=True,help='NEW output directory under home')
    p.add_argument('--max-rows',type=int,default=800000)
    p.add_argument('--tune-entities',type=int,default=3000)
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--min-free-gb',type=float,default=25)
    fit(p.parse_args())
if __name__=='__main__':main()
