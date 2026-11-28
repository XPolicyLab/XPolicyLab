"""Download the official, immutable Qwen revision and verify every runtime file."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.request

MANIFEST = Path(__file__).resolve().parent / 'environment/QWEN_SOURCES.json'
ORIGIN = 'https://huggingface.co'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _fetch_whole(url, target, size, attempts):
    """Small Hub files may be served by a cache API without Range support."""
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url, headers={'User-Agent': 'VAP-public-qwen-fetch/1'},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read(size + 1)
            if len(data) != size:
                raise ValueError(f'Qwen file length differs: {len(data)} != {size}')
            with target.open('xb') as output:
                output.write(data)
            return
        except (OSError, ValueError):
            if target.exists():
                target.unlink()
            if attempt + 1 == attempts:
                raise


def _fetch_block(url, target, start, stop, attempts):
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url, headers={
                    'User-Agent': 'VAP-public-qwen-fetch/1',
                    'Range': f'bytes={start}-{stop}',
                },
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                expected = stop - start + 1
                content_range = response.headers.get('Content-Range')
                if response.status != 206 or not (
                    content_range and content_range.startswith(f'bytes {start}-{stop}/')
                ):
                    raise ValueError(
                        f'Pinned byte range not honored: {start}-{stop}, '
                        f'status={response.status}, Content-Range={content_range!r}, '
                        f'url={response.url}'
                    )
                data = response.read(expected + 1)
                if len(data) != expected:
                    raise ValueError(f'Qwen block length differs: {start}-{stop}')
            with target.open('r+b') as output:
                output.seek(start)
                output.write(data)
            return
        except (OSError, ValueError):
            if attempt + 1 == attempts:
                raise


def download_file(url, target, size, *, connections, chunk_size, attempts):
    if connections < 1 or chunk_size < 1024 * 1024:
        raise ValueError('Invalid Qwen download concurrency or chunk size')
    ranges = [(start, min(start + chunk_size, size) - 1)
              for start in range(0, size, chunk_size)]
    with target.open('xb') as output:
        output.truncate(size)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=connections, thread_name_prefix='qwen-fetch'
    ) as pool:
        futures = [pool.submit(_fetch_block, url, target, start, stop, attempts)
                   for start, stop in ranges]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def verify(root, manifest):
    actual = {p.name for p in root.iterdir() if p.is_file() and p.suffix in {
        '.json', '.safetensors', '.model', '.txt', '.jinja'
    }}
    if actual != set(manifest['files']):
        raise ValueError(f'Qwen file inventory differs: {sorted(actual ^ set(manifest["files"]))}')
    hashes = {}
    for name, record in manifest['files'].items():
        path = root / name
        if path.stat().st_size != record['size']:
            raise ValueError(f'Qwen file size mismatch: {name}')
        hashes[name] = digest(path)
        if hashes[name] != record['sha256']:
            raise ValueError(f'Qwen SHA256 mismatch: {name}')
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    if identity != manifest['model_id']:
        raise ValueError('Qwen identity differs from the VAP training checkpoint')
    return {'repository': manifest['repository'], 'revision': manifest['revision'],
            'model_id': identity, 'files_sha256': hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--base-url', default=ORIGIN,
                        help='Optional HTTPS mirror; payload hashes remain authoritative')
    parser.add_argument('--transport-revision', default=None,
                        help='Optional mirror-only revision alias; official hashes still bind every byte')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--connections', type=int, default=4,
                        help='Bounded byte-range concurrency for large files')
    parser.add_argument('--chunk-mib', type=int, default=128)
    parser.add_argument('--attempts', type=int, default=5)
    args = parser.parse_args()
    if not args.base_url.startswith('https://'):
        parser.error('HTTPS is required')
    manifest = json.loads(MANIFEST.read_text())
    transport_revision = args.transport_revision or manifest['revision']
    if not transport_revision or '/' in transport_revision or transport_revision.startswith('.'):
        parser.error('Unsafe transport revision')
    root = args.output.expanduser().resolve()
    if args.verify_only:
        report = verify(root, manifest)
        provenance = json.loads(root.with_name(root.name + '.provenance.json').read_text())
        if provenance != report:
            raise ValueError('Qwen provenance differs from the verified runtime payload')
        print(json.dumps(report, indent=2))
        return
    if root.exists():
        raise FileExistsError(f'Refusing to use an existing download directory: {root}')
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=root.name + '.partial-', dir=root.parent))
    try:
        for name, record in manifest['files'].items():
            url = f'{args.base_url.rstrip("/")}/{manifest["repository"]}/resolve/{transport_revision}/{name}'
            target = temporary / name
            if record['size'] <= args.chunk_mib * 1024 * 1024:
                _fetch_whole(url, target, record['size'], attempts=args.attempts)
            else:
                download_file(
                    url, target, record['size'], connections=args.connections,
                    chunk_size=args.chunk_mib * 1024 * 1024, attempts=args.attempts,
                )
            if target.stat().st_size != record['size'] or digest(target) != record['sha256']:
                raise ValueError(f'Qwen download failed verification: {name}')
            print('verified', name, flush=True)
        report = verify(temporary, manifest)
        # Runtime Qwen identity hashes every JSON in the model directory.
        # Keep download provenance outside that payload so it cannot alter it.
        provenance = root.with_name(root.name + '.provenance.json')
        if provenance.exists():
            raise FileExistsError(f'Refusing to overwrite provenance: {provenance}')
        with provenance.open('x') as stream:
            json.dump(report, stream, indent=2)
            stream.write('\n')
        os.rename(temporary, root)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    print('QWEN_READY', root, flush=True)


if __name__ == '__main__':
    main()
