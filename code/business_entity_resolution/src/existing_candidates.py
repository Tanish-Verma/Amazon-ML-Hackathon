"""Recompute a documented, *new* lexical score for existing official 2-column candidates.

The missing Phase-3 RRF/channel metadata cannot be recovered: zero denotes
unavailable for those three feature slots. Training and test use the SAME
lexical surrogate; do not combine these features with an old enriched model.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import psutil
from src.candidates import load_partition
from src.rerank_vec import build_store_streaming, pair_features_vec

HEADER=['source1_entity_id','candidate_entity_ids']
OUT=['source1_entity_id','candidate_entity_id','rrf_score','n_channels_hit','best_channel_rank','reranker_score']

def iter_lists(path):
    with open(path,encoding='utf8',newline='') as f:
        r=csv.reader(f,delimiter='\t',quoting=csv.QUOTE_NONE)
        if next(r,None)!=HEADER:raise ValueError('Expected exact official two-column header')
        for row in r:
            if len(row)!=2:raise ValueError(f'Invalid candidate row: {row[:2]}')
            q,raw=row
            ids=raw.split(',') if raw else []
            if len(ids)!=len(set(ids)) or any(not c.startswith(('S2-','S3-')) for c in ids):
                raise ValueError(f'Invalid or duplicate candidates for {q}')
            yield q,ids

def lexical_score(X):
    # Only real lexical/digit similarities. Positive [0,1] surrogate, NOT Phase-3 score.
    # columns 0 name trigram, 1 token, 3 Jaro-Winkler, 6 address trigram,
    # 7 address token, 8 address Jaro-Winkler, 9 digit Jaccard.
    return np.clip(.14*X[:,0]+.10*X[:,1]+.10*X[:,3]+.20*X[:,6]+.20*X[:,7]+.16*X[:,8]+.10*X[:,9],0,1)

def convert(path,store,split,country,out,workers=4,min_free_gb=25,max_queries=0):
    out=Path(out).expanduser().resolve()
    if not out.is_relative_to(Path.home().resolve()):raise ValueError('Output must be under home')
    if out.exists():raise FileExistsError(out)
    if psutil.virtual_memory().available<min_free_gb*1e9:raise RuntimeError('RAM guard')
    qids,qn,qa=load_partition(store,split,'s1',country)
    cids,cn,ca=[],[],[]
    for source in ('s2','s3'):
        ids,names,addrs=load_partition(store,split,source,country)
        cids+=ids;cn+=names;ca+=addrs
    if not qids or not cids:raise ValueError('Missing country partition')
    qm={q:i for i,q in enumerate(qids)};cm={c:i for i,c in enumerate(cids)}
    if len(qm)!=len(qids) or len(cm)!=len(cids):raise ValueError('Duplicate record IDs')
    corpus=build_store_streaming(cn,ca,workers)
    out.parent.mkdir(parents=True,exist_ok=True)
    n=0;nq=0
    with out.open('x',encoding='utf8',newline='') as f:
        w=csv.writer(f,delimiter='\t',quoting=csv.QUOTE_NONE,lineterminator='\n');w.writerow(OUT)
        for q,ids in iter_lists(path):
            if q not in qm:continue
            if max_queries and nq>=max_queries:break
            nq+=1
            if not ids:continue
            missing=[c for c in ids if c not in cm]
            if missing:raise ValueError(f'{q}: unknown or cross-country candidate {missing[:2]}')
            query=build_store_streaming([qn[qm[q]]],[qa[qm[q]]],workers,vocabs=corpus.vocabs)
            ci=np.asarray([cm[c] for c in ids],dtype=np.int32)
            z=np.zeros(len(ci),dtype=np.float32)
            X=pair_features_vec(query,corpus,np.zeros(len(ci),dtype=np.int32),ci,z,z,z,
                                np.asarray([c.startswith('S3-') for c in ids],dtype=np.float32),workers=workers)
            scores=lexical_score(X)
            for c,s in zip(ids,scores):w.writerow([q,c,0,0,0,f'{float(s):.8f}'])
            n+=len(ids)
            if nq%1000==0:
                print(f'{country}: {nq:,} queries / {n:,} pairs; free RAM {psutil.virtual_memory().available/1e9:.1f}GB',flush=True)
                if psutil.virtual_memory().available<min_free_gb*1e9:raise RuntimeError('RAM guard')
    if not n:raise ValueError('No pairs produced; verify input split/country')
    print(f'Wrote {n:,} pairs to {out}; RRF/channel features unavailable and set to zero',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for x in ('candidates','store','country','out'):p.add_argument('--'+x,required=True)
    p.add_argument('--split',choices=['train','test'],required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--min-free-gb',type=float,default=25)
    p.add_argument('--max-queries',type=int,default=0)
    a=p.parse_args();convert(a.candidates,a.store,a.split,a.country,a.out,a.workers,a.min_free_gb,a.max_queries)
if __name__=='__main__':main()
