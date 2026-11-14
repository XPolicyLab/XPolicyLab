"""Download the selected original RobotWin3D files without changing their layout."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = 'RAW_DATA_DOWNLOAD_RECEIPT.json'
REPLICA = 'RAW_MEDIA_REPLICA_SHA256.json'
REPOSITORIES = {'flex-pi/robotwin_3d', 'liumouliu/cr-wm-posttrain-assets'}


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def relative(value):
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or '..' in path.parts or '\\' in value:
        raise ValueError('Invalid relative data path: ' + value)
    return path


def local_path(root, name):
    path = root.joinpath(*relative(name).parts)
    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError('Data path escapes its output directory: ' + name)
    return path


def atomic_json(path, value):
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix='.' + path.name + '.', delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, separators=(',', ':'))
        stream.write('\n')
    temporary.replace(path)


def load_manifest(path):
    value = json.loads(path.read_text())
    if value.get('version') != 'cloudrobo_wm.robotwin_clean_download.v1':
        raise ValueError('Unsupported data download manifest')
    rows = value['files']
    if not isinstance(rows, dict) or not rows:
        raise ValueError('Empty data download manifest')
    for name, row in rows.items():
        relative(name)
        relative(row.get('source', name))
        repo = row.get('repo_id', value['repo_id'])
        revision = row.get('revision', value['revision'])
        if repo not in REPOSITORIES or re.fullmatch('[0-9a-f]{40}', revision) is None:
            raise ValueError('Data sources must use the pinned original dataset repositories')
        if row.get('sha256') and re.fullmatch('[0-9a-f]{64}', row['sha256']) is None:
            raise ValueError('Invalid SHA256: ' + name)
        if 'bytes' in row and (type(row['bytes']) is not int or row['bytes'] < 0):
            raise ValueError('Invalid byte size: ' + name)
    return value, rows


def verify(path, row, expected_sha=None, expected_size=None, git_etag=None):
    if not path.is_file():
        raise FileNotFoundError('Missing data file: ' + str(path))
    size = path.stat().st_size
    if row.get('bytes', size) != size or (expected_size is not None and expected_size != size):
        raise ValueError('Data size differs; file preserved: ' + str(path))
    actual = digest(path)
    if row.get('sha256', actual) != actual or (expected_sha and expected_sha != actual):
        raise ValueError('Data SHA256 differs; file preserved: ' + str(path))
    if git_etag:
        git = hashlib.sha1(('blob ' + str(size) + '\0').encode())
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                git.update(chunk)
        if git.hexdigest() != git_etag:
            raise ValueError('Hub Git blob identity differs: ' + str(path))
    return {'bytes': size, 'sha256': actual}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'posttrain/data/robotwin_download_manifest.json')
    parser.add_argument('--workers', type=int, default=4)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', help='Inspect the selection without network access or file writes')
    mode.add_argument('--verify-only', action='store_true', help='Verify an existing download using its saved SHA256 receipt')
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error('--workers must be in 1..16')
    value, rows = load_manifest(args.manifest)
    if args.manifest.resolve() == (ROOT / 'posttrain/data/robotwin_download_manifest.json').resolve():
        listed = (ROOT / 'posttrain/data/clean_files.txt').read_text().splitlines()
        if len(listed) != len(set(listed)) or set(listed) != set(rows):
            raise ValueError('clean_files.txt and the data download manifest disagree')
    root = args.output_dir.expanduser().resolve()
    binding = digest(args.manifest)
    if args.dry_run:
        print(json.dumps({'files': len(rows), 'selection': value.get('selection'),
                          'repo_id': value['repo_id'], 'revision': value['revision'],
                          'manifest_sha256': binding, 'output_dir': str(root),
                          'network_used': False, 'files_written': False}, indent=2))
        return
    previous = {}
    receipt_path = local_path(root, RECEIPT)
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get('manifest_sha256') == binding and receipt.get('complete'):
            previous = receipt['files']
    if args.verify_only:
        def one(item):
            name, row = item
            old = previous.get(name, {})
            expected = row.get('sha256') or old.get('sha256')
            if not expected:
                raise ValueError('No verified SHA256 receipt for ' + name + '; run the pinned Hub download first')
            return name, verify(local_path(root, name), row, expected, old.get('bytes'))
    else:
        try:
            from huggingface_hub import hf_hub_download, hf_hub_url, get_hf_file_metadata
        except ImportError as error:
            raise RuntimeError('Install the download dependency: python3 -m pip install huggingface_hub') from error
        root.mkdir(parents=True, exist_ok=True)
        def one(item):
            name, row = item
            repo, revision = row.get('repo_id', value['repo_id']), row.get('revision', value['revision'])
            source = row.get('source', name)
            metadata = get_hf_file_metadata(hf_hub_url(repo, source, repo_type='dataset', revision=revision), token=False)
            if metadata.commit_hash != revision:
                raise ValueError('Hub resolved a different commit: ' + name)
            etag = (metadata.etag or '').strip('"')
            if re.fullmatch('[0-9a-f]{64}|[0-9a-f]{40}', etag) is None:
                raise ValueError('Hub returned no supported content hash: ' + name)
            expected_sha = etag if len(etag) == 64 else None
            git_etag = etag if len(etag) == 40 else None
            path = local_path(root, name)
            if path.exists():
                return name, verify(path, row, expected_sha, metadata.size, git_etag)
            # Cache download keeps the published source immutable; copy to a new destination only.
            cached = Path(hf_hub_download(repo_id=repo, filename=source, repo_type='dataset', revision=revision, token=False))
            record = verify(cached, row, expected_sha, metadata.size, git_etag)
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.cloudrobo-data-', delete=False) as output:
                temporary = Path(output.name)
                try:
                    with cached.open('rb') as stream:
                        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                            output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                    # Validate independent output bytes before publishing without overwrite.
                    verify(temporary, row, record['sha256'], record['bytes'])
                    temporary.chmod(0o644)
                    os.link(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
            return name, record
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = dict(pool.map(one, rows.items()))
    if not args.verify_only:
        replica = {}
        for name, row in rows.items():
            if 'original_mtime_ns' in row:
                path = local_path(root, name)
                observed = path.stat()
                replica[str(path)] = {'expected': [row['bytes'], row['original_mtime_ns']],
                                      'observed': [observed.st_size, observed.st_mtime_ns],
                                      'sha256': records[name]['sha256']}
        atomic_json(local_path(root, REPLICA), {'schema': 'cloudrobo.media_replica.sha256.v1', 'files': replica})
        atomic_json(receipt_path, {'version': 'cloudrobo_wm.raw_data_download.v1', 'complete': True,
                                  'manifest_sha256': binding, 'files': records})
    print(json.dumps({'raw_data_verified': True, 'files': len(records),
                      'verify_only': args.verify_only, 'output_dir': str(root)}))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
