import csv
import json
from pathlib import Path
import pytest
from src.generalization import compare, check_manifest, drift, audit_code, score_block


def fake(country, n=10, p=.7, r=.6, f=.65):
    return {'holdout_entities': n, 'by_country': {country: {'micro_precision':p,'micro_recall':r,'macro_f05':f}}}


def test_compare():
    out=compare(fake('India'),fake('India',p=.5),fake('India',p=.4),'India')
    assert out['transfer_minus_in_distribution']['micro_precision']==pytest.approx(-.2)
    assert out['ablation_minus_transfer']['micro_precision']==pytest.approx(-.1)


def test_wrong_country_rejected():
    with pytest.raises(ValueError): score_block(fake('US'),'India')


def test_different_holdout_rejected():
    with pytest.raises(ValueError): compare(fake('India'),fake('India',n=11),fake('India'),'India')


def test_manifest_requires_provenance():
    with pytest.raises((ValueError,KeyError)): check_manifest({'runs':{}})


def write_tsv(path, field, rows):
    with path.open('w') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(['source1_entity_id',field]);w.writerows(rows)


def test_drift(tmp_path):
    c=tmp_path/'c.tsv';p=tmp_path/'p.tsv'
    write_tsv(c,'candidate_entity_ids',[('a','x,y'),('b','z')])
    write_tsv(p,'matched_entity_ids',[('a','x'),('b','')])
    out=drift(c,p)
    assert out['n']==2 and out['predicted_singleton_rate']==.5


def test_drift_rejects_unretrieved(tmp_path):
    c=tmp_path/'c.tsv';p=tmp_path/'p.tsv'
    write_tsv(c,'candidate_entity_ids',[('a','x')]);write_tsv(p,'matched_entity_ids',[('a','y')])
    with pytest.raises(ValueError):drift(c,p)


def test_audit(tmp_path):
    (tmp_path/'bad.py').write_text('X="France"\n')
    result=audit_code(tmp_path)
    assert result['count']==1 and result['hits'][0]['line']==1
