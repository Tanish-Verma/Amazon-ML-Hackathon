import hashlib
import json
import zipfile
import pytest
from src.phase9 import check_archive_name, load_manifest, sha256, verify


def test_reject_traversal():
    for name in ('../x', '/x', 'x/../y', 'x\\y', 'x//y', 'x/'):
        with pytest.raises(ValueError): check_archive_name(name)


def test_manifest_duplicate(tmp_path):
    f = tmp_path / 'data'; f.write_text('hello')
    m = tmp_path / 'manifest.json'
    m.write_text(json.dumps({'files': [{'source': str(f), 'archive_path': 'a'}, {'source': str(f), 'archive_path': 'a'}]}))
    with pytest.raises(ValueError): load_manifest(m)


def test_zip_verify(tmp_path):
    p = tmp_path / 'test.zip'
    with zipfile.ZipFile(p, 'w') as z: z.writestr('output/candidate_pairs.tsv', 'x')
    verify(p, [{'archive_path': 'output/candidate_pairs.tsv', 'sha256': hashlib.sha256(b'x').hexdigest()}])


def test_zip_reject_unsafe(tmp_path):
    p = tmp_path / 'test.zip'
    with zipfile.ZipFile(p, 'w') as z: z.writestr('../escape', 'x')
    with pytest.raises(ValueError): verify(p)


def test_hash(tmp_path):
    f = tmp_path / 'a'; f.write_bytes(b'abc')
    assert sha256(f) == hashlib.sha256(b'abc').hexdigest()
