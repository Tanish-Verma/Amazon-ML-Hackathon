"""Phase 9: conservative, manifest-driven submission packager.

Does not guess the competition's archive layout. Requires a user-reviewed JSON
manifest mapping source files to exact archive paths from the official statement.
All writes must be in a fresh directory below the user's home.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import zipfile


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def checked_out(path):
    p = Path(path).expanduser().resolve()
    home = Path.home().resolve()
    if p == home or home not in p.parents:
        raise ValueError(f'Output must be inside your home directory: {p}')
    if p.exists():
        raise FileExistsError(f'Refusing to overwrite: {p}')
    return p


def check_archive_name(name):
    if not isinstance(name, str) or not name or '\\' in name:
        raise ValueError(f'Invalid archive path: {name!r}')
    p = PurePosixPath(name)
    if p.is_absolute() or any(s in ('', '.', '..') for s in name.split('/')) or name.endswith('/'):
        raise ValueError(f'Unsafe archive path: {name!r}')
    return name


def load_manifest(path):
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    if set(obj) != {'files'} or not isinstance(obj['files'], list) or not obj['files']:
        raise ValueError('Manifest must contain a nonempty files array only')
    names = set()
    for item in obj['files']:
        if set(item) != {'source', 'archive_path'}:
            raise ValueError('Each file needs source and archive_path')
        name = check_archive_name(item['archive_path'])
        if name in names:
            raise ValueError(f'Duplicate archive path: {name}')
        names.add(name)
        p = Path(item['source']).expanduser().resolve()
        if not p.is_file() or p.is_symlink():
            raise FileNotFoundError(f'Missing or symlink source: {p}')
        item['source'] = str(p)
    return obj


def build(manifest, out):
    entries = load_manifest(manifest)['files']
    out = checked_out(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    # Exclusive creation prevents accidental replacement of existing submissions.
    with out.open('xb') as stream:
        with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6, allowZip64=True) as z:
            for item in entries:
                src = Path(item['source'])
                name = item['archive_path']
                before = sha256(src)
                with src.open('rb') as f, z.open(name, 'w', force_zip64=True) as dest:
                    for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
                        dest.write(block)
                after = sha256(src)
                if before != after:
                    raise RuntimeError(f'Source changed during packaging: {src}')
                records.append({'archive_path': name, 'source': str(src),
                                'bytes': src.stat().st_size, 'sha256': after})
    verify(out, records)
    print(json.dumps({'archive': str(out), 'files': records}, indent=2))


def verify(archive, records=None):
    archive = Path(archive)
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive entries')
        for name in names:
            check_archive_name(name)
        if z.testzip() is not None:
            raise ValueError('Corrupt archive')
        if records is not None:
            if set(names) != {r['archive_path'] for r in records}:
                raise ValueError('Archive tree differs from manifest')
            for record in records:
                h = hashlib.sha256()
                with z.open(record['archive_path']) as f:
                    for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
                        h.update(block)
                if h.hexdigest() != record['sha256']:
                    raise ValueError(f'Checksum mismatch: {record["archive_path"]}')
    print(f'Archive structure and CRC verified: {archive} ({len(names)} files)')


def environment(out):
    out = checked_out(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {'python': sys.version, 'executable': sys.executable,
            'pip_freeze': subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines()}
    with out.open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
        f.write('\n')
    print(f'Environment recorded: {out}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build'); b.add_argument('--manifest', required=True); b.add_argument('--out', required=True)
    v = sub.add_parser('verify'); v.add_argument('--archive', required=True)
    e = sub.add_parser('environment'); e.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    if args.cmd == 'build': build(args.manifest, args.out)
    elif args.cmd == 'verify': verify(args.archive)
    else: environment(args.out)

if __name__ == '__main__':
    main()
