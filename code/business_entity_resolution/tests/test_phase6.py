from src.split import stratified_splits
from src.validate_phase6 import bootstrap_ci
from src.scoring import entity_score


def test_splits_repeatable_disjoint_and_stratified():
    rows=[(f'S1-{i}', 'A' if i%3 else 'B', i%19==0) for i in range(1000)]
    a=stratified_splits(rows,100,200,42)
    b=stratified_splits(reversed(rows),100,200,42)
    assert a==b
    assert len(a['dev'])==100 and len(a['confirm'])==200
    assert not set(a['dev'])&set(a['confirm'])
    assert a!=stratified_splits(rows,100,200,43)


def test_invalid_split_and_duplicate():
    import pytest
    with pytest.raises(ValueError):stratified_splits([('a','X',False)]*2,1,0)
    with pytest.raises(ValueError):stratified_splits([('a','X',False)],1,1)


def test_scoring_examples():
    assert round(entity_score({'a','b','c'},{'a','b'}),3)==0.714
    assert entity_score(set(),set())==1
    assert entity_score({'a'},set())==0
    assert entity_score(set(),{'a'})==0


def test_bootstrap_reproducible():
    truth={str(i):({'x'} if i%5 else set()) for i in range(100)}
    pred={str(i):({'x'} if i%3 else set()) for i in range(100)}
    countries={str(i):('A' if i%2 else 'B') for i in range(100)}
    a=bootstrap_ci(pred,truth,countries,200,42)
    assert a==bootstrap_ci(pred,truth,countries,200,42)
    assert a['ci_low']<=a['point']<=a['ci_high']


def test_all_singleton_baseline():
    truth={str(i):set() if i<558 else {'x'} for i in range(10000)}
    countries={q:'X' for q in truth}
    pred={q:set() for q in truth}
    assert abs(bootstrap_ci(pred,truth,countries,100,1)['point']-.0558)<1e-12
