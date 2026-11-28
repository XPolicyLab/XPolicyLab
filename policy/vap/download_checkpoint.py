"""Download the current VAP checkpoint from the public Hugging Face main branch."""
import argparse
import concurrent.futures
import hashlib
import http.client
import json
import os
from pathlib import Path
import shutil
import ssl
import sys
import tempfile
import time
import urllib.error
import urllib.request

MANIFEST = Path(__file__).resolve().parent / 'environment/CHECKPOINT_SOURCES.json'
ORIGIN = 'https://huggingface.co'
CHUNK_SIZE = 64 * 1024 * 1024
MAX_ATTEMPTS = 8


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def request(url, method='GET', headers=None):
    return urllib.request.Request(url, method=method, headers={
        'User-Agent': 'VAP-public-checkpoint-fetch/1', **(headers or {})})


def remote_metadata(url, expected_size=None):
    for attempt in range(MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(request(url, method='HEAD'), timeout=120) as response:
                if response.status != 200:
                    raise ValueError(f'Checkpoint metadata request failed: HTTP {response.status}')
                length = response.headers.get('Content-Length')
                if length is None and expected_size is not None:
                    # A chunked HEAD response has no length. The repository API
                    # already binds this required file to its current-main size.
                    size = expected_size
                else:
                    try:
                        size = int(length)
                    except (TypeError, ValueError) as error:
                        raise ValueError('Checkpoint server did not provide a valid Content-Length') from error
                etag = response.headers.get('ETag')
            if size <= 0:
                raise ValueError(f'Checkpoint file is empty: {url.rsplit("/", 1)[-1]}')
            if not etag:
                raise ValueError('Checkpoint server must provide an ETag for resumable downloads')
            return size, etag
        except (urllib.error.URLError, http.client.HTTPException,
                ssl.SSLError, TimeoutError, ConnectionError, ValueError):
            if attempt + 1 == MAX_ATTEMPTS:
                raise
            time.sleep(min(2 ** attempt, 10))
    raise AssertionError('unreachable')


def save_metadata(path, url, start, end, total, etag, sha256):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('x') as stream:
        json.dump({'url': url, 'start': start, 'end': end, 'total': total,
                   'etag': etag, 'sha256': sha256}, stream)
        stream.write('\n')
    os.rename(temporary, path)


def save_completed(path, url, total, etag, sha256):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('x') as stream:
        json.dump({'url': url, 'total': total, 'etag': etag, 'sha256': sha256}, stream)
        stream.write('\n')
    os.rename(temporary, path)


def save_download_state(path, files):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('x') as stream:
        json.dump({'schema': 1, 'files': files}, stream)
        stream.write('\n')
    os.rename(temporary, path)


def remote_lfs_metadata(base_url, repository, revision, required_files):
    """Identify current public files; an Xet ETag is not a file SHA256."""
    url = f'{base_url}/api/models/{repository}/tree/{revision}?recursive=true&expand=true'
    with urllib.request.urlopen(request(url), timeout=120) as response:
        if response.status != 200:
            raise ValueError(f'Checkpoint metadata request failed: HTTP {response.status}')
        rows = json.load(response)
    if not isinstance(rows, list):
        raise ValueError('Checkpoint repository metadata is not a file list')
    required = set(required_files)
    result = {}
    for row in rows:
        if not isinstance(row, dict) or row.get('type') != 'file' or row.get('path') not in required:
            continue
        name = row['path']
        if name in result or type(row.get('size')) is not int or row['size'] <= 0:
            raise ValueError(f'Invalid public checkpoint metadata: {name}')
        lfs = row.get('lfs')
        if lfs is None:
            # A regular Git file has no LFS oid. Its range/ETag checks still apply.
            result[name] = {'size': row['size'], 'sha256': None}
            continue
        if (not isinstance(lfs, dict) or not isinstance(lfs.get('oid'), str)
                or len(lfs['oid']) != 64
                or any(c not in '0123456789abcdef' for c in lfs['oid'])
                or type(lfs.get('size')) is not int or lfs['size'] != row['size']):
            raise ValueError(f'Invalid public LFS checkpoint metadata: {name}')
        result[name] = {'size': lfs['size'], 'sha256': lfs['oid']}
    missing = required - result.keys()
    if missing:
        raise ValueError(f'Checkpoint metadata is missing required files: {sorted(missing)}')
    return result


def remote_sha256(etag):
    """Legacy LFS-compatible ETag fallback; an Xet hash is not a file SHA256."""
    normalized = etag.removeprefix('W/').strip('"').lower()
    if etag.startswith('W/') or len(normalized) != 64 \
            or any(c not in '0123456789abcdef' for c in normalized):
        return None
    return normalized


def fetch_range(url, target, start, end, total, etag):
    """Download one exact range; retain verified chunks and incomplete bytes."""
    expected = end - start + 1
    metadata = target.with_name(target.name + '.json')
    recorded = None
    if metadata.is_file():
        recorded = json.loads(metadata.read_text())
        expected_record = {'url': url, 'start': start, 'end': end, 'total': total, 'etag': etag}
        if any(recorded.get(key) != value for key, value in expected_record.items()):
            raise ValueError(f'Incomplete chunk does not match the current main payload: {target.name}')
    if target.exists():
        if target.is_symlink():
            raise ValueError(f'Resume chunk must not be a symlink: {target.name}')
        size = target.stat().st_size
        if size > expected or (size == 0 and recorded is not None):
            raise ValueError(f'Incomplete chunk has an invalid size: {target.name}')
        if size == expected:
            if recorded is not None:
                if digest(target) != recorded.get('sha256'):
                    raise ValueError(f'Completed chunk differs from its recorded SHA256: {target.name}')
                return 0
            normalized = remote_sha256(etag)
            actual = digest(target)
            if normalized is not None and actual == normalized:
                save_metadata(metadata, url, start, end, total, etag, actual)
                return 0
            # No metadata survived the last interruption. Rename this unproven
            # block aside; any later failure leaves evidence, never silent loss.
            unverified = target.with_name(target.name + '.unverified')
            if unverified.exists() or unverified.is_symlink():
                raise ValueError(f'Refusing to overwrite retained unverified chunk: {unverified.name}')
            target.rename(unverified)
        elif size < expected and etag.startswith('W/'):
            # Ordinary Git files on the mirror may not implement Range. Never
            # append a second whole-file response to an interrupted download.
            unverified = target.with_name(target.name + '.unverified')
            if unverified.exists() or unverified.is_symlink():
                raise ValueError(f'Refusing to overwrite retained unverified chunk: {unverified.name}')
            target.rename(unverified)
    elif metadata.exists():
        raise ValueError(f'Chunk metadata has no payload: {metadata}')

    for attempt in range(MAX_ATTEMPTS):
        offset = target.stat().st_size if target.exists() else 0
        weak_whole = etag.startswith('W/') and start == 0 and offset == 0 and end == total - 1
        headers = {} if weak_whole else {'Range': f'bytes={start + offset}-{end}'}
        if not etag.startswith('W/'):
            headers['If-Range'] = etag
        try:
            with urllib.request.urlopen(request(url, headers=headers), timeout=120) as response:
                expected_range = f'bytes {start + offset}-{end}/{total}'
                valid_framing = (response.status == 200 and response.headers.get('Content-Range') is None
                                 if weak_whole else response.status == 206
                                 and response.headers.get('Content-Range') == expected_range)
                supplied_length = response.headers.get('Content-Length')
                if (not valid_framing or response.headers.get('ETag') != etag
                        or supplied_length is not None and int(supplied_length) != expected - offset):
                    raise ValueError(f'Unsafe range response for {target.name}')
                with target.open('ab') as output:
                    remaining = expected - offset
                    while remaining:
                        block = response.read(min(8 * 1024 * 1024, remaining))
                        if not block:
                            raise ValueError(f'Incomplete response for {target.name}')
                        output.write(block)
                        remaining -= len(block)
                    if response.read(1):
                        raise ValueError(f'Response exceeds the expected file size: {target.name}')
            save_metadata(metadata, url, start, end, total, etag, digest(target))
            return offset
        except (urllib.error.URLError, http.client.HTTPException,
                ssl.SSLError, TimeoutError, ConnectionError, ValueError) as error:
            if isinstance(error, urllib.error.HTTPError) and error.code not in (408, 429) and error.code < 500:
                raise
            if attempt + 1 == MAX_ATTEMPTS:
                raise
            time.sleep(min(2 ** attempt, 10))
    raise AssertionError('unreachable')


def fetch(url, target, workers=6, completed_record=None, expected_file=None):
    """Fetch a stable remote snapshot; retain its resume record outside the payload."""
    total, etag = remote_metadata(url, expected_file['size'] if expected_file is not None else None)
    if expected_file is not None and total != expected_file['size']:
        raise ValueError(f'Remote LFS size differs from the range response: {target.name}')
    # Sibling metadata cannot violate the exact three-file runtime inventory.
    record_path = target.parent.with_name(target.parent.name + '.' + target.name + '.complete.json')
    legacy_record = target.with_name(target.name + '.complete.json')
    parts = target.with_name(target.name + '.parts')
    if target.exists() and target.is_file() and not target.is_symlink() \
            and target.stat().st_size < total and parts.is_dir():
        # A killed assembly is not a downloaded payload. The verified chunks
        # below are the source of truth for rebuilding this own partial file.
        target.unlink()
    if target.exists():
        if not target.is_file() or target.is_symlink() or target.stat().st_size != total:
            raise ValueError(f'Retained assembled file does not match current main: {target.name}')
        try:
            record = json.loads(record_path.read_text())
        except FileNotFoundError:
            try:
                record = json.loads(legacy_record.read_text())
            except FileNotFoundError:
                record = completed_record
                if record is None:
                    # Compatibility with an older interrupted transfer: public LFS
                    # metadata identifies the plain file even when the ETag is Xet.
                    normalized = (expected_file['sha256'] if expected_file is not None
                                  else remote_sha256(etag))
                    if normalized is None:
                        raise ValueError(f'Cannot safely reuse an unrecorded assembled file: {target.name}')
                    record = {'url': url, 'total': total, 'etag': etag, 'sha256': normalized}
        if (record.get('url') != url or record.get('total') != total or record.get('etag') != etag):
            raise ValueError(f'Assembled file does not match the current main payload: {target.name}')
        if not isinstance(record.get('sha256'), str) or len(record['sha256']) != 64:
            raise ValueError(f'Invalid assembled-file SHA256 record: {target.name}')
        if digest(target) != record['sha256']:
            raise ValueError(f'Assembled file differs from its recorded SHA256: {target.name}')
        if expected_file is not None and expected_file['sha256'] is not None \
                and record['sha256'] != expected_file['sha256']:
            raise ValueError(f'Assembled file differs from the remote LFS SHA256: {target.name}')
        save_completed(record_path, url, total, etag, record['sha256'])
        legacy_record.unlink(missing_ok=True)
        # Once the assembled file matches the current remote binding, the old
        # chunks are no longer part of the strict three-file payload inventory.
        if parts.exists() or parts.is_symlink():
            shutil.rmtree(parts)
        print(f'reusing assembled {target.name}', flush=True)
        return
    parts.mkdir(exist_ok=True)
    ranges = [(index, start, min(start + CHUNK_SIZE, total) - 1)
              for index, start in enumerate(range(0, total, CHUNK_SIZE))]

    def download_chunk(item):
        index, start, end = item
        return fetch_range(url, parts / f'{index:05d}.part', start, end, total, etag)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(download_chunk, item) for item in ranges]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            if completed == len(futures) or completed % 10 == 0:
                print(f'transferred {target.name}: {completed}/{len(futures)} chunks', flush=True)
    with target.open('xb') as output:
        for index, start, end in ranges:
            part = parts / f'{index:05d}.part'
            record = json.loads(part.with_name(part.name + '.json').read_text())
            if (part.stat().st_size != end - start + 1
                    or digest(part) != record['sha256']):
                raise ValueError(f'Chunk verification failed: {part.name}')
            with part.open('rb') as source:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
    actual = digest(target)
    expected = expected_file['sha256'] if expected_file is not None else remote_sha256(etag)
    if expected is not None and actual != expected:
        raise ValueError(f'Assembled file differs from the remote LFS SHA256: {target.name}')
    save_completed(record_path, url, total, etag, actual)
    legacy_record.unlink(missing_ok=True)
    shutil.rmtree(parts)


def verify(root, manifest):
    actual = {path.name for path in root.iterdir()}
    if actual != set(manifest['files']):
        raise ValueError(f'Checkpoint file inventory differs: {sorted(actual ^ set(manifest["files"]))}')
    files = {}
    for name in manifest['files']:
        path = root / name
        size = path.stat().st_size
        if size == 0:
            raise ValueError(f'Checkpoint file is empty: {name}')
        files[name] = {'size': size, 'sha256': digest(path)}
    return {'repository': manifest['repository'], 'revision': manifest['revision'],
            'files': files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--base-url', default=ORIGIN,
                        help='Optional HTTPS mirror for the same main-branch payloads')
    parser.add_argument('--workers', type=int, default=6,
                        help='Concurrent bounded range requests per file (default: 6)')
    parser.add_argument('--resume', type=Path,
                        help='Incomplete sibling directory retained by an earlier failed invocation')
    parser.add_argument('--verify-only', action='store_true',
                        help='Recheck required files and their recorded size/SHA256')
    args = parser.parse_args()
    if not args.base_url.startswith('https://'):
        parser.error('HTTPS is required')
    if not 1 <= args.workers <= 16:
        parser.error('--workers must be between 1 and 16')
    manifest = json.loads(MANIFEST.read_text())
    root = args.output.expanduser().resolve()
    if args.verify_only:
        if args.resume:
            parser.error('--resume cannot be combined with --verify-only')
        report = verify(root, manifest)
        provenance = json.loads(root.with_name(root.name + '.provenance.json').read_text())
        if provenance != report:
            raise ValueError('Checkpoint provenance differs from the verified runtime payload')
        print(json.dumps(report, indent=2))
        return
    if args.output.expanduser().exists() or args.output.expanduser().is_symlink():
        raise FileExistsError(f'Refusing to use an existing download directory: {args.output}')
    root.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        temporary = args.resume.expanduser().resolve()
        expected_parent = root.parent.resolve()
        if (not temporary.is_dir() or temporary.parent != expected_parent
                or not temporary.name.startswith(root.name + '.partial-')):
            raise ValueError('--resume must be a retained partial directory for this output')
    else:
        temporary = Path(tempfile.mkdtemp(prefix=root.name + '.partial-', dir=root.parent))
    base_url = args.base_url.rstrip('/')
    lfs_files = remote_lfs_metadata(base_url, manifest['repository'], manifest['revision'],
                                    manifest['files'])
    recorded_base = temporary.with_name(temporary.name + '.source_base_url')
    legacy_base = temporary / '.source_base_url'
    if args.resume:
        records = [path for path in (recorded_base, legacy_base) if path.is_file()]
        if not records:
            raise ValueError('Resume source record is missing')
        if any(path.read_text().strip() != base_url for path in records):
            raise ValueError('Resume source differs from the original download')
        if legacy_base.is_file():
            if not recorded_base.exists() and not recorded_base.is_symlink():
                recorded_base.write_text(base_url + '\n')
            legacy_base.unlink()
    else:
        recorded_base.write_text(base_url + '\n', encoding='utf-8')
    resume_state_path = temporary.with_name(temporary.name + '.download_state.json')
    resume_state = json.loads(resume_state_path.read_text()) if resume_state_path.is_file() else {}
    if resume_state and (resume_state.get('schema') != 1 or not isinstance(resume_state.get('files'), dict)):
        raise ValueError('Invalid retained download state')
    provenance = root.with_name(root.name + '.provenance.json')
    retained_provenance = None
    if args.resume and provenance.is_file() and not provenance.is_symlink():
        # A hard kill can occur after this record but before atomic publication.
        # It is reusable only after independently matching the verified payload.
        try:
            retained_provenance = json.loads(provenance.read_text())
        except ValueError as error:
            raise ValueError('Refusing to overwrite an unreadable retained provenance') from error
    if provenance.exists() or provenance.is_symlink():
        if retained_provenance is None:
            raise FileExistsError(f'Refusing to overwrite provenance: {provenance}')
    try:
        for name in manifest['files']:
            url = f'{args.base_url.rstrip("/")}/{manifest["repository"]}/resolve/{manifest["revision"]}/{name}'
            fetch(url, temporary / name, workers=args.workers,
                  completed_record=resume_state.get('files', {}).get(name),
                  expected_file=lfs_files[name])
            print('downloaded', name, flush=True)
        completed = {name: json.loads(temporary.with_name(
            temporary.name + '.' + name + '.complete.json').read_text()) for name in manifest['files']}
        report = verify(temporary, manifest)
        if any(completed[name]['sha256'] != report['files'][name]['sha256'] for name in manifest['files']):
            raise ValueError('Assembled-file provenance differs from verified payload')
        # Keep the remote binding until publication succeeds. The payload itself
        # remains exactly the three runtime files throughout every kill window.
        save_download_state(resume_state_path, completed)
        legacy_base.unlink(missing_ok=True)
        created_provenance = False
        try:
            if retained_provenance is not None and retained_provenance != report:
                raise ValueError('Retained provenance differs from verified payload')
            if retained_provenance is None:
                with provenance.open('x') as stream:
                    created_provenance = True
                    json.dump(report, stream, indent=2)
                    stream.write('\n')
            os.rename(temporary, root)
        except BaseException:
            if created_provenance:
                provenance.unlink()
            raise
    except BaseException:
        print(f'CHECKPOINT_INCOMPLETE {temporary}', file=sys.stderr, flush=True)
        raise
    recorded_base.unlink(missing_ok=True)
    resume_state_path.unlink(missing_ok=True)
    for name in manifest['files']:
        temporary.with_name(temporary.name + '.' + name + '.complete.json').unlink()
    print('CHECKPOINT_READY', root, flush=True)


if __name__ == '__main__':
    main()
