from src.decide import decide

def test_competition():
    rows=[('a','x',.91),('b','x',.90),('b','y',.89)]
    assert decide(rows,.8,.1)=={'a':{'x'},'b':{'y'}}

def test_singleton_and_margin():
    rows=[('a','x',.95),('a','y',.80),('b','z',.3)]
    assert decide(rows,.8,.05)=={'a':{'x'}}

def test_duplicate_rejected():
    import pytest
    with pytest.raises(ValueError):decide([('a','x',.9),('a','x',.8)],.5,.1)
