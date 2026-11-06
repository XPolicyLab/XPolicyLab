"""Download only explicitly supplied, hash-bound model files; missing URLs are not invented."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit
import urllib.request


def validated_files(manifest, urls):
    records = manifest.get('files') if isinstance(manifest, dict) else None
    if not isinstance(records, list) or not records or not isinstance(urls, dict):
        raise ValueError('A nonempty exact model manifest and URL mapping are required')
    identifiers, paths = set(), set()
    for item in records:
        if (not isinstance(item, dict) or not isinstance(item.get('id'), str) or not item['id']
                or not isinstance(item.get('path'), str)
                or not isinstance(item.get('sha256'), str)
                or re.fullmatch('[0-9a-f]{64}', item['sha256']) is None
                or type(item.get('bytes')) is not int or item['bytes'] < 0):
            raise ValueError('Every model file requires an exact ID, path, SHA256 and byte count')
        relative = Path(item['path'])
        if (relative.is_absolute() or '..' in relative.parts or not relative.parts
                or relative.as_posix() != item['path']):
            raise ValueError('Asset path must stay within its destination')
        if item['id'] in identifiers or item['path'] in paths:
            raise ValueError('Model IDs and destination paths must be unique')
        identifiers.add(item['id'])
        paths.add(item['path'])
        url = urls.get(item['id']) or item.get('url')
        if url is not None:
            parsed = urlsplit(url) if isinstance(url, str) else None
            if (parsed is None or parsed.scheme != 'https' or not parsed.hostname
                    or parsed.username is not None or parsed.password is not None):
                raise ValueError('An authorized HTTPS URL without embedded credentials is required')
    if set(urls) - identifiers:
        raise ValueError('URL overrides may name only exact declared model IDs')
    return records


def download(manifest, destination, urls, *, check_only=False, opener=None):
    records = validated_files(manifest, urls)
    destination = Path(destination).resolve()
    pending, planned = [], []
    for item in records:
        filename = destination / item['path']
        if not filename.resolve().is_relative_to(destination):
            raise ValueError('Asset destination must not escape through a symlink')
        if filename.is_symlink():
            raise ValueError('An existing model destination must be a regular file')
        if filename.exists():
            if not filename.is_file():
                raise ValueError('An existing model destination must be a regular file')
            with filename.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != item['sha256'] or filename.stat().st_size != item['bytes']:
                raise ValueError('Preserve the existing nonmatching model file: ' + item['id'])
            continue
        partial = filename.with_suffix(filename.suffix + '.part')
        if partial.exists() or partial.is_symlink():
            raise FileExistsError('Preserve the existing partial model download: ' + item['id'])
        url = urls.get(item['id']) or item.get('url')
        pending.append({'id': item['id'], 'url_missing': not bool(url), 'digest_missing': False,
            'download_required': True})
        planned.append((item, filename, partial, url))
    if check_only or any(item['url_missing'] for item in pending):
        return {'complete': not pending, 'pending': pending, 'downloads_started': False}
    opener = urllib.request.urlopen if opener is None else opener
    for item, filename, partial, url in planned:
        filename.parent.mkdir(parents=True, exist_ok=True)
        with partial.open('xb') as stream, opener(url, timeout=60) as response:
            while chunk := response.read(8 * 1024 * 1024):
                stream.write(chunk)
        with partial.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != item['sha256'] or partial.stat().st_size != item['bytes']:
            raise ValueError('Downloaded asset does not match its frozen identity: ' + item['id'])
        filename.hardlink_to(partial)
        partial.unlink()
    return {'complete': True, 'pending': [], 'downloads_started': bool(planned)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/models/downloads.json')
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--urls', type=Path, help='JSON mapping asset id to an authorized HTTPS URL')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--prepare', action='store_true', help='Also prepare templates, SAM runtime and protocol code')
    parser.add_argument('--official-assets', type=Path, help='Installed official Assets directory; never modified')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    urls = json.loads(args.urls.read_text()) if args.urls else {}
    result = download(manifest, args.destination, urls, check_only=args.check_only)
    if args.prepare:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from rsi.resource_setup import prepare
        from rsi.deployment import locations
        assets = args.official_assets or Path(locations()['official_root']) / 'Assets'
        prepared = prepare(args.destination, official_assets=assets, check_only=args.check_only)
        result = dict(downloads=result, resources=prepared, complete=result['complete'] and prepared['complete'])
    print(json.dumps(result, indent=2))
    return 0 if result['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
