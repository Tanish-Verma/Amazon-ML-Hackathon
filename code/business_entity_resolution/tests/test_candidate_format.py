import csv
import pytest
from src.candidate_format import inspect, expand, verify_enriched


def test_inspect_expand(tmp_path):
    src = tmp_path / 'candidate_pairs.tsv'
    src.write_text('source1_entity_id\tcandidate_entity_ids\nS1-1\tS3-2,S2-3\nS1-4\t\n')
    assert inspect(src) == {'s1_rows': 2, 'candidate_pairs': 2, 'min_candidates': 0,
                            'max_candidates': 2, 'empty_candidate_lists': 1}
    dst = tmp_path / 'pairs.tsv'
    expand(src, dst)
    assert dst.read_text().splitlines()[1:] == ['S1-1\tS3-2\t1', 'S1-1\tS2-3\t2']


def test_duplicate_rejected(tmp_path):
    src = tmp_path / 'c.tsv'
    src.write_text('source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-2,S2-2\n')
    with pytest.raises(ValueError, match='duplicate'):
        inspect(src)


def test_verify_sidecar(tmp_path):
    src = tmp_path / 'c.tsv'
    src.write_text('source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-2,S3-3\n')
    rich = tmp_path / 'rich.tsv'
    rich.write_text('source1_entity_id\tcandidate_entity_id\trrf_score\tn_channels_hit\tbest_channel_rank\treranker_score\n'
                    'S1-1\tS2-2\t0.1\t2\t1\t0.9\nS1-1\tS3-3\t0.2\t1\t3\t0.8\n')
    assert verify_enriched(src, rich, tmp_path / 'v.sqlite')['verified_pairs'] == 2


def test_missing_sidecar_pair(tmp_path):
    src = tmp_path / 'c.tsv'
    src.write_text('source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-2,S3-3\n')
    rich = tmp_path / 'rich.tsv'
    rich.write_text('source1_entity_id\tcandidate_entity_id\trrf_score\tn_channels_hit\tbest_channel_rank\treranker_score\n'
                    'S1-1\tS2-2\t0.1\t2\t1\t0.9\n')
    with pytest.raises(ValueError, match='missing'):
        verify_enriched(src, rich, tmp_path / 'v.sqlite')
