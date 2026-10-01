import csv
import sqlite3
import pytest
from src.phase8 import ids, rows, candidate_db, match_db


def test_ids():
    assert ids('') == []
    assert ids('S2-a,S3-b') == ['S2-a', 'S3-b']
    for value in ['S2-a,S2-a', 'S1-a', 'S2-a,', ' S2-a']:
        with pytest.raises(ValueError):
            ids(value)


def test_header_and_duplicates(tmp_path):
    f = tmp_path / 'c.tsv'
    f.write_text('source1_entity_id\tcandidate_entity_ids\nS1-a\tS2-x\nS1-a\tS2-y\n')
    with sqlite3.connect(':memory:') as conn:
        with pytest.raises(sqlite3.IntegrityError):
            candidate_db(conn, [f])
    f.write_text('wrong\theader\n')
    with pytest.raises(ValueError):
        list(rows(f, ['source1_entity_id', 'candidate_entity_ids']))


def test_match_singletons(tmp_path):
    f = tmp_path / 'm.tsv'
    f.write_text('source1_entity_id\tmatched_entity_ids\nS1-a\t\nS1-b\tS3-y\n')
    with sqlite3.connect(':memory:') as conn:
        match_db(conn, [f])
        assert conn.execute('SELECT vals FROM match WHERE qid="S1-a"').fetchone()[0] == ''


def test_audit_end_to_end(tmp_path):
    from src.phase8 import audit
    c = tmp_path / 'c.tsv'
    m = tmp_path / 'm.tsv'
    c.write_text('source1_entity_id\tcandidate_entity_ids\nS1-a\tS2-x,S3-y\nS1-b\t\n')
    m.write_text('source1_entity_id\tmatched_entity_ids\nS1-a\tS3-y\nS1-b\t\n')
    audit(c, m, expected=2)
    m.write_text('source1_entity_id\tmatched_entity_ids\nS1-a\tS2-other\nS1-b\t\n')
    with pytest.raises(ValueError, match='missing from candidate'):
        audit(c, m, expected=2)
