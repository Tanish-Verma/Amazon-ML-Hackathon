"""Merge per-country nonempty matches with ALL test S1 IDs into submission TSV."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import pyarrow.parquet as pq

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--store',required=True)
    p.add_argument('--country-results',nargs='+',required=True,help='all countries matches_nonempty.tsv')
    p.add_argument('--out',required=True,help='NEW output TSV; never overwrites')
    args=p.parse_args()
    # Expected result rows fit in RAM more readily than 87M candidate pairs.
    matches={}
    for path in args.country_results:
        with open(path,newline='') as f:
            r=csv.DictReader(f,delimiter='\t',quoting=csv.QUOTE_NONE)
            if r.fieldnames!=['source1_entity_id','matched_entity_ids']:
                raise ValueError(f'Unexpected header: {path}')
            for row in r:
                q=row['source1_entity_id']
                if q in matches:raise ValueError(f'Duplicate query {q}')
                matches[q]=row['matched_entity_ids']
    files=sorted(Path(args.store).glob('test_s1_*.parquet'))
    if not files:raise FileNotFoundError('No test_s1_*.parquet in store')
    seen=set();out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    with out.open('x',newline='') as f:
        w=csv.writer(f,delimiter='\t',quoting=csv.QUOTE_NONE)
        w.writerow(['source1_entity_id','matched_entity_ids'])
        for path in files:
            pf=pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=100000,columns=['entity_id']):
                for q in batch.column(0).to_pylist():
                    if q in seen:raise ValueError(f'Duplicate test S1 ID {q}')
                    seen.add(q)
                    w.writerow([q,matches.get(q,'')])
    if set(matches)-seen:raise ValueError(f'{len(set(matches)-seen)} predictions have unknown S1 IDs')
    print(f'Wrote {len(seen):,} test entities; {len(matches):,} with predicted matches to {out}')
if __name__=='__main__':main()
