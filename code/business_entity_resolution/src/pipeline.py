"""Integrated orchestration using existing candidate_pairs.tsv, without rerunning Phase 3.

Requires a SEPARATE labeled TRAIN candidate TSV generated from training S1 records;
test candidate_pairs.tsv has no labels and must never be used for training.
Country runs are sequential. Every stage uses new paths and aborts on errors.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import psutil
from src.phase8 import assemble,audit

def run(*argv):
    print('\nRUN', ' '.join(map(str,argv)),flush=True)
    subprocess.run([sys.executable,'-m',*map(str,argv)],check=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for x in ('store','train-candidates','test-candidates','gt','out'):p.add_argument('--'+x,required=True)
    p.add_argument('--blocking-model',help='Optional: generate TRAIN candidates only if --train-candidates is missing')
    p.add_argument('--train-countries',nargs='+',default=['US','India'])
    p.add_argument('--test-countries',nargs='+',default=['US','India','France'])
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--min-free-gb',type=float,default=25)
    p.add_argument('--query-batch',type=int,default=100)
    p.add_argument('--max-queries',type=int,default=0,help='Smoke-test only; do NOT submit truncated results')
    p.add_argument('--max-train-rows',type=int,default=800000)
    p.add_argument('--expected',type=int,default=1732544)
    p.add_argument('--skip-audit',action='store_true',help='Smoke-test only')
    p.add_argument('--manifest',help='Official-layout Phase 9 manifest, optional')
    a=p.parse_args()
    out=Path(a.out).expanduser().resolve();home=Path.home().resolve()
    if not out.is_relative_to(home) or out==home:raise ValueError('Output must be inside home')
    if out.exists():raise FileExistsError(out)
    for f in (a.test_candidates,a.gt):
        if not Path(f).is_file():raise FileNotFoundError(f)
    if not Path(a.train_candidates).is_file() and not a.blocking_model:
        raise FileNotFoundError('Missing training candidate TSV; provide --blocking-model to generate it from TRAIN only')
    if not Path(a.store).is_dir():raise FileNotFoundError(a.store)
    if a.max_queries and not a.skip_audit:raise ValueError('Smoke-test requires --skip-audit')
    if psutil.virtual_memory().available<a.min_free_gb*1e9:raise RuntimeError('Insufficient RAM')
    out.mkdir(parents=True,exist_ok=False)
    os.environ['TMPDIR']=str(out)
    log=[]
    def stage(label,*cmd):
        if psutil.virtual_memory().available<a.min_free_gb*1e9:raise RuntimeError('RAM guard')
        run(*cmd);log.append(label);(out/'completed_stages.json').write_text(json.dumps(log,indent=2))
    if not Path(a.train_candidates).is_file():
        if not Path(a.blocking_model).is_file():raise FileNotFoundError(a.blocking_model)
        a.train_candidates=str(out/'generated_train_candidates.tsv')
        stage('generate TRAIN candidates only','src.run_blocking','--store',a.store,'--split','train','--model',a.blocking_model,'--out',a.train_candidates,'--top-k','50','--workers',a.workers,'--countries',*a.train_countries,'--max-queries',a.max_queries)
    trainroot=out/'phase4_train'
    for country in a.train_countries:
        cdir=trainroot/country;cdir.mkdir(parents=True,exist_ok=False)
        side=cdir/'recomputed_enriched.tsv'
        stage(f'phase4 train adapter {country}','src.existing_candidates','--candidates',a.train_candidates,'--store',a.store,'--split','train','--country',country,'--out',side,'--workers',a.workers,'--min-free-gb',a.min_free_gb,'--max-queries',a.max_queries)
        stage(f'phase4 train features {country}','src.run_features','--pairs',side,'--store',a.store,'--split','train','--country',country,'--gt',a.gt,'--out',cdir/'features','--workers',a.workers,'--query-batch',a.query_batch,'--min-free-gb',a.min_free_gb)
    # model.feature_files expects country/features_*.parquet: supply separate flat links
    flat=out/'phase4_train_flat';flat.mkdir()
    import shutil
    for country in a.train_countries:
        dest=flat/country;dest.mkdir()
        for f in sorted((trainroot/country/'features'/country).glob('features_*.parquet')):
            os.symlink(f,dest/f.name)
    modeldir=out/'phase5_model'
    stage('phase5 train','src.model','--features',flat,'--gt',a.gt,'--out',modeldir,'--threads',a.workers,'--max-rows',a.max_train_rows,'--min-free-gb',a.min_free_gb)
    candidates=[];matches=[]
    for country in a.test_countries:
        cdir=out/'test'/country;cdir.mkdir(parents=True,exist_ok=False)
        side=cdir/'recomputed_enriched.tsv'
        stage(f'phase4 test adapter {country}','src.existing_candidates','--candidates',a.test_candidates,'--store',a.store,'--split','test','--country',country,'--out',side,'--workers',a.workers,'--min-free-gb',a.min_free_gb,'--max-queries',a.max_queries)
        feat=cdir/'features'
        stage(f'phase4 test features {country}','src.run_features','--pairs',side,'--store',a.store,'--split','test','--country',country,'--out',feat,'--workers',a.workers,'--query-batch',a.query_batch,'--min-free-gb',a.min_free_gb)
        pred=cdir/'predicted'
        stage(f'phase5 predict {country}','src.predict_phase5','--features',feat/country,'--model',modeldir/'model.pkl','--out',pred,'--min-free-gb',a.min_free_gb)
        candidates.append(a.test_candidates) # original 2-column file is assembled ONCE below
        matches.append(pred/'matches_nonempty.tsv')
    if a.max_queries:
        print('SMOKE TEST FINISHED: intentionally no full submission assembly',flush=True)
        return
    # Existing candidate file already covers ALL countries; assemble only matches
    from src.phase8 import home_path,rows,MATCH_HEADER,CAND_HEADER
    import csv,sqlite3
    submission=out/'submission';submission.mkdir()
    db=sqlite3.connect(str(submission/'matches.sqlite'))
    db.execute('CREATE TABLE matches(qid TEXT PRIMARY KEY, vals TEXT NOT NULL)')
    for file in matches:
        for q,ids in rows(file,MATCH_HEADER):db.execute('INSERT INTO matches VALUES(?,?)',(q,ids))
        db.commit()
    target=submission/'matching_results.tsv'
    with target.open('x',newline='') as f:
        w=csv.writer(f,delimiter='\t',quoting=csv.QUOTE_NONE,lineterminator='\n');w.writerow(MATCH_HEADER)
        for q,_ in rows(a.test_candidates,CAND_HEADER):
            r=db.execute('SELECT vals FROM matches WHERE qid=?',(q,)).fetchone()
            w.writerow([q,r[0] if r else ''])
    db.close()
    # Preserve the exact original candidate file, not a regenerated substitute.
    original=submission/'candidate_pairs.tsv';os.symlink(Path(a.test_candidates).resolve(),original)
    if not a.skip_audit:
        audit(original,target,a.store,a.expected,check_ids=True)
        log.append('phase8 internal audit passed');(out/'completed_stages.json').write_text(json.dumps(log,indent=2))
    if a.manifest:
        print('Phase 9 manifest must reference the new submission files:',original,target,flush=True)
        stage('phase9 package','src.phase9','build','--manifest',a.manifest,'--out',out/'submission.zip')
    print('DONE',submission,flush=True)
if __name__=='__main__':main()
