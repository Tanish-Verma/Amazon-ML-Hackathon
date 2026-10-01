"""Score a held-out prediction TSV, with stratified bootstrap CI and per-country metrics."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from src.scoring import entity_score, macro_f05


def bootstrap_ci(pred,truth,countries,n_boot=1000,seed=42,alpha=.05):
    """Entity-level, country-stratified bootstrap of the macro-F0.5."""
    by_country={}
    for q,t in truth.items():
        country=countries[q]
        by_country.setdefault(country,[]).append(entity_score(set(pred.get(q,())),set(t)))
    rng=np.random.default_rng(seed)
    arrays=[np.asarray(v,dtype=np.float64) for v in by_country.values()]
    n=sum(map(len,arrays))
    if not n: raise ValueError('Empty evaluation')
    if n_boot < 1: raise ValueError('n_boot must be positive')
    samples=np.empty(n_boot)
    for i in range(n_boot):
        samples[i]=sum(float(a[rng.integers(0,len(a),len(a))].sum()) for a in arrays)/n
    return {'point':sum(a.sum() for a in arrays)/n,'ci_low':float(np.quantile(samples,alpha/2)), 'ci_high':float(np.quantile(samples,1-alpha/2)), 'bootstrap_replicates':n_boot,'seed':seed,'method':'stratified entity bootstrap (fixed per-country counts)'}


def read_predictions(path):
    from src.io_utils import stream_tsv
    out={}
    for row in stream_tsv(str(path),['source1_entity_id','matched_entity_ids']):
        q=row['source1_entity_id']
        if q in out: raise ValueError(f'Duplicate prediction {q}')
        raw=row['matched_entity_ids']
        ids=raw.split(',') if raw else []
        if len(ids)!=len(set(ids)): raise ValueError(f'Duplicate matched ID for {q}')
        out[q]=set(ids)
    return out


def read_ids(path):
    from src.io_utils import stream_tsv
    out=[]
    for row in stream_tsv(str(path),['source1_entity_id']):
        out.append(row['source1_entity_id'])
    if len(out)!=len(set(out)): raise ValueError('Duplicate holdout IDs')
    return set(out)


def evaluate(gt,source1,ids,predictions,n_boot=1000,seed=42,strict=True):
    from src.io_utils import stream_tsv, GROUND_TRUTH_COLUMNS, SOURCE_COLUMNS
    selected=read_ids(ids)
    truth={}
    for row in stream_tsv(str(gt),GROUND_TRUTH_COLUMNS):
        q=row['source1_entity_id']
        if q in selected:
            if q in truth: raise ValueError(f'Duplicate truth ID {q}')
            truth[q]=set(row['matched_entity_ids'].split(',')) if row['matched_entity_ids'] else set()
    if selected!=set(truth): raise ValueError(f'Missing truth for {len(selected-set(truth))} holdout IDs')
    countries={}
    for row in stream_tsv(str(source1),SOURCE_COLUMNS):
        q=row['entity_id']
        if q in selected:
            if q in countries: raise ValueError(f'Duplicate source1 ID {q}')
            countries[q]=row['country']
    if selected!=set(countries): raise ValueError('Missing source1 country for holdout IDs')
    pred=read_predictions(predictions)
    extra=set(pred)-selected
    if extra and strict: raise ValueError(f'Predictions include {len(extra)} IDs outside holdout')
    if strict and selected-set(pred): raise ValueError(f'Missing {len(selected-set(pred))} prediction rows; explicit empty rows required')
    pred={q:v for q,v in pred.items() if q in selected}
    overall=macro_f05(pred,truth)
    overall['confidence_interval']=bootstrap_ci(pred,truth,countries,n_boot,seed)
    by_country={}
    for country in sorted(set(countries.values())):
        t={q:truth[q] for q in truth if countries[q]==country}
        by_country[country]=macro_f05(pred,t)
        by_country[country]['confidence_interval']=bootstrap_ci(pred,t,countries,n_boot,seed)
    return {'overall':overall,'by_country':by_country,'holdout_entities':len(selected),'predicted_rows':len(pred)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for x in ('gt','source1','ids','predictions','out'):p.add_argument('--'+x,required=True)
    p.add_argument('--bootstrap',type=int,default=1000)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--allow-missing',action='store_true')
    a=p.parse_args()
    out=Path(a.out).expanduser().resolve()
    if not out.is_relative_to(Path.home().resolve()):raise ValueError('Output must be under home')
    if out.exists():raise FileExistsError(out)
    report=evaluate(a.gt,a.source1,a.ids,a.predictions,a.bootstrap,a.seed,not a.allow_missing)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
