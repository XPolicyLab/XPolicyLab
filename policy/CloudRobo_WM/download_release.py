"""Fetch the pinned CR-WM release, or verify/copy an offline release."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import quote

from download_checkpoint import Asset, AssetError, check_file, copy_checked, relative_asset_path, safe_path, write_asset


def load_release(path: Path, group: str):
    data = json.loads(path.read_text(encoding='utf-8'))
    if data['repo_id'] != 'liumouliu/cr-wm-posttrain-assets' or data['repo_type'] != 'dataset':
        raise AssetError('Unexpected release repository')
    revision = data['revision']
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise AssetError('The release must use a pinned commit')
    rows, seen = [], set()
    for row in data['files']:
        if row['group'] != group:
            continue
        source = relative_asset_path(row['source'])
        target = relative_asset_path(row['path'])
        if target in seen:
            raise AssetError(f'Duplicate destination: {target}')
        seen.add(target)
        size, digest = row['bytes'], row['sha256']
        if type(size) is not int or size < 0:
            raise AssetError(f'Invalid file size: {target}')
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise AssetError(f'Invalid checksum: {target}')
        url = f"https://huggingface.co/datasets/{data['repo_id']}/resolve/{revision}/{quote(source, safe='/')}"
        rows.append((source, Asset(target, size, digest, url)))
    if not rows:
        raise AssetError(f'No release files for {group}')
    return data, rows


def run(args):
    data, rows = load_release(args.manifest, args.group)
    root = args.output_dir.expanduser().resolve()
    local = args.source_dir.expanduser().resolve() if args.source_dir else None
    pending = []
    for source, asset in rows:
        destination = safe_path(root, asset.path)
        if destination.exists():
            check_file(destination, asset)
        elif args.verify_only:
            raise AssetError(f'Missing file: {destination}')
        else:
            pending.append((source, asset))
    if local is not None:
        for source, asset in pending:
            check_file(safe_path(local, source), asset)
    for source, asset in pending:
        if local is None:
            write_asset(asset, root, source_dir=None, base_url=None, replace=False, timeout=args.timeout)
            print(f'[verified] {asset.path}')
            continue
        origin = safe_path(local, source)
        destination = safe_path(root, asset.path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix='.crwm-release-', dir=destination.parent)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, 'wb') as output, origin.open('rb') as stream:
                copy_checked(stream, output, asset)
                output.flush()
                os.fsync(output.fileno())
            temporary.chmod(0o644)
            safe_path(root, asset.path)
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        print(f'[verified] {asset.path}')
    print(f'Verified {len(rows)} {args.group} files; downloaded/copied {len(pending)}.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', choices=('inference', 'training', 'metadata'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, help='Offline source using the HF repository layout')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--timeout', type=float, default=60.0)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).with_name('release_manifest.json'))
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    try:
        run(args)
    except (AssetError, OSError, ValueError) as error:
        parser.exit(2, f'{error}\n')


if __name__ == '__main__':
    main()
