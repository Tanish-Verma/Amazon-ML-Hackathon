"""Phase 7: leakage-checked country transfer, ablation, code audit and unlabeled drift.

This is an evaluator, not a substitute for training independent Phase 3-5 pipelines.
See README_PHASE7.md for the required run matrix and provenance manifest.
"""
from __future__ import annotations
import argparse
import ast
import csv
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path

COUNTRIES = ('US', 'India')
METRICS = ('micro_precision', 'micro_recall', 'macro_f05')


def read_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def write_new(path, value):
    p = Path(path).expanduser().resolve()
    if not p.is_relative_to(Path.home().resolve()):
        raise ValueError('Output must be inside home directory')
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(value, indent=2, ensure_ascii=False))


def score_block(report, country):
    if report.get('holdout_entities', 0) <= 0:
        raise ValueError('Empty holdout')
    by = report['by_country']
    if set(by) != {country}:
        raise ValueError(f'Expected ONLY {country} in report; found {list(by)}')
    obj = by[country]
    # src.scoring.macro_f05 names must be verified against real Phase 6 JSON.
    if not all(k in obj for k in METRICS):
        raise ValueError(f'Missing expected metric keys {METRICS}; found {list(obj)}')
    out = {k: float(obj[k]) for k in METRICS}
    if any((math.isfinite(v) and not 0 <= v <= 1) for v in out.values()):
        raise ValueError('Finite metrics must be in [0,1]')
    # The official scorer uses NaN for undefined micro precision/recall
    # (e.g. no predicted positives); encode as null for strict JSON.
    out = {k: (v if math.isfinite(v) else None) for k,v in out.items()}
    out['n'] = report['holdout_entities']
    if 'confidence_interval' in obj:
        out['confidence_interval'] = obj['confidence_interval']
    return out


def compare(baseline, transfer, ablated, country):
    b, t, a = (score_block(r, country) for r in (baseline, transfer, ablated))
    if len({b['n'], t['n'], a['n']}) != 1:
        raise ValueError('Holdout sizes differ: evaluate exactly the same IDs')
    return {'target': country, 'entities': b['n'],
            'in_distribution': b, 'transfer': t, 'transfer_without_dictionaries': a,
            'transfer_minus_in_distribution': {k: (t[k]-b[k] if t[k] is not None and b[k] is not None else None) for k in METRICS},
            'ablation_minus_transfer': {k: (a[k]-t[k] if a[k] is not None and t[k] is not None else None) for k in METRICS}}


def check_manifest(manifest):
    """Fail closed if provenance or split identity is missing.

    Manifest describes six independent training/evaluation runs; all IDs are opaque.
    """
    runs = manifest['runs']
    expected = {
        'us_baseline': ('US', 'US', True),
        'us_to_india': ('US', 'India', True),
        'us_to_india_ablated': ('US', 'India', False),
        'india_baseline': ('India', 'India', True),
        'india_to_us': ('India', 'US', True),
        'india_to_us_ablated': ('India', 'US', False),
    }
    if set(runs) != set(expected):
        raise ValueError(f'Expected exactly these runs: {sorted(expected)}')
    for key, (train, target, dictionaries) in expected.items():
        run = runs[key]
        if (run.get('fit_country'), run.get('eval_country'), run.get('dictionaries_enabled')) != (train, target, dictionaries):
            raise ValueError(f'Invalid training/evaluation/dictionary provenance: {key}')
        for required in ('fit_ids_sha256', 'eval_ids_sha256', 'model_sha256',
                         'prediction_path', 'phase6_report_path', 'blocking_config_sha256',
                         'normalization_config_sha256', 'feature_config_sha256',
                         'threshold_config_sha256', 'corpus_stats_provenance'):
            if not run.get(required):
                raise ValueError(f'{key}: missing {required}')
        if run['corpus_stats_provenance'] not in ('target_unlabeled_only', 'training_only'):
            raise ValueError(f'{key}: corpus statistics provenance invalid')
    # Exact same target IDs across baseline, transfer and ablation.
    for trio in (('us_baseline','india_to_us','india_to_us_ablated'),
                 ('india_baseline','us_to_india','us_to_india_ablated')):
        if len({runs[k]['eval_ids_sha256'] for k in trio}) != 1:
            raise ValueError(f'Target holdout IDs differ for {trio}')
    # Strong check that transfer model is not target-fitted; model/config hashes
    # should be the same as corresponding baseline except ablation normalization.
    for base, transfer in (('us_baseline','us_to_india'),('india_baseline','india_to_us')):
        if runs[base]['model_sha256'] != runs[transfer]['model_sha256']:
            raise ValueError(f'Model changed during country transfer: {base} -> {transfer}')
        if runs[base]['threshold_config_sha256'] != runs[transfer]['threshold_config_sha256']:
            raise ValueError('Thresholds were retuned on target country')
    return runs


def build_report(manifest):
    runs = check_manifest(manifest)
    reports = {key: read_json(value['phase6_report_path']) for key, value in runs.items()}
    result = {
        'hide_india': compare(reports['india_baseline'], reports['us_to_india'],
                              reports['us_to_india_ablated'], 'India'),
        'hide_us': compare(reports['us_baseline'], reports['india_to_us'],
                           reports['india_to_us_ablated'], 'US'),
        'warning': ('Descriptive holdout results, not France accuracy. '
                    'All six prediction files must come from independently verified pipeline runs.'),
    }
    return result


COUNTRY_RE = re.compile(r'(?<![\w])(?:US|India|France)(?![\w])', re.I)


def audit_code(root):
    """Report country literals in Python source; reviewer must inspect every hit."""
    hits = []
    for path in sorted(Path(root).rglob('*.py')):
        if any(p in {'.venv','venv','.git','__pycache__'} for p in path.parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        except (UnicodeError, SyntaxError) as exc:
            hits.append({'file': str(path), 'error': str(exc)})
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for match in COUNTRY_RE.finditer(node.value):
                    hits.append({'file': str(path), 'line': node.lineno,
                                 'literal': match.group(0), 'context': node.value[:160]})
    return {'hits': hits, 'count': len(hits),
            'note': 'Review each hit: opaque partition labels and CLI examples may be permitted; country-specific feature/model rules are not.'}


def read_result_tsv(path, field):
    result = {}
    with open(path, encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        if not reader.fieldnames or not {'source1_entity_id',field} <= set(reader.fieldnames):
            raise ValueError(f'Expected source1_entity_id and {field}: {path}')
        for row in reader:
            q = row['source1_entity_id']
            if q in result:
                raise ValueError(f'Duplicate S1 ID {q}')
            raw = row[field]
            ids = [x.strip() for x in raw.split(',') if x.strip()]
            if len(ids) != len(set(ids)):
                raise ValueError(f'Duplicate candidate/matched ID for {q}')
            result[q] = ids
    return result


def percentile(values, q):
    if not values: return None
    a = sorted(values)
    i = (len(a)-1)*q
    lo, hi = math.floor(i), math.ceil(i)
    return a[lo]*(hi-i)+a[hi]*(i-lo) if hi != lo else float(a[lo])


def drift(candidates, predictions):
    c = read_result_tsv(candidates, 'candidate_entity_ids')
    p = read_result_tsv(predictions, 'matched_entity_ids')
    if set(c) != set(p):
        raise ValueError(f'Candidate/prediction S1 mismatch: {len(set(c)^set(p))}')
    for q in c:
        if not set(p[q]) <= set(c[q]):
            raise ValueError(f'Matches not contained in candidates: {q}')
    counts = [len(x) for x in c.values()]
    matched = [len(x) for x in p.values()]
    n = len(counts)
    if not n: raise ValueError('Empty results')
    return {'n': n, 'candidate_count': {'mean': statistics.fmean(counts),
            'p50': percentile(counts,.5),'p90': percentile(counts,.9),
            'p99': percentile(counts,.99), 'empty_rate': counts.count(0)/n},
            'predicted_singleton_rate': matched.count(0)/n,
            'predicted_match_count_mean': statistics.fmean(matched)}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='command',required=True)
    report=sub.add_parser('report')
    report.add_argument('--manifest',required=True);report.add_argument('--out',required=True)
    audit=sub.add_parser('audit')
    audit.add_argument('--root',required=True);audit.add_argument('--out',required=True)
    drift_p=sub.add_parser('drift')
    drift_p.add_argument('--country',required=True)
    drift_p.add_argument('--candidates',required=True)
    drift_p.add_argument('--predictions',required=True)
    drift_p.add_argument('--out',required=True)
    a=ap.parse_args()
    if a.command=='report': result=build_report(read_json(a.manifest))
    elif a.command=='audit': result=audit_code(a.root)
    else: result={'country':a.country, **drift(a.candidates,a.predictions)}
    write_new(a.out,result)

if __name__=='__main__': main()
