"""Explicit asset staging/download. No upload or publication is implemented."""
from pathlib import Path, PurePosixPath
import argparse
import json
import os
import urllib.request
from delivery import WORKSPACE, asset_entries, sha256, verify_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('verify', 'download-public', 'link-local'))
    parser.add_argument('--root', type=Path, default=WORKSPACE)
    parser.add_argument('--base-source', type=Path)
    parser.add_argument('--jam-source', type=Path)
    parser.add_argument('--fastwam-source', type=Path)
    parser.add_argument('--method', choices=('jam', 'fastwam'))
    parser.add_argument('--dry-run', action='store_true', help='List public downloads without network or writes')
    args = parser.parse_args()
    if args.dry_run and args.operation != 'download-public':
        parser.error('--dry-run is supported for download-public only')
    if args.operation == 'verify':
        print(json.dumps({'verified': verify_assets(args.root, hashes=True, method=args.method)}))
        return
    for item in asset_entries():
        if item['kind'] == 'checkpoint' and args.method and item.get('method') != args.method:
            continue
        dest = args.root / item['path']
        if args.operation == 'download-public':
            if not item.get('url'):
                raise ValueError(f'No public source recorded: {item["path"]}')
            if args.dry_run:
                print(json.dumps({'path': str(dest), 'url': item['url'], 'bytes': item['bytes'],
                                  'sha256': item['sha256']}))
                continue
            if dest.exists() and sha256(dest) == item['sha256']:
                continue
            if dest.exists() or dest.is_symlink():
                raise FileExistsError(f'Refusing to overwrite asset: {dest}')
            dest.parent.mkdir(parents=True, exist_ok=True)
            if item.get('source_repository') and item.get('source_revision') and item.get('source_file'):
                # local_dir keeps the actual file at its final relative path;
                # HF's adjacent metadata/incomplete file supports resume without
                # duplicating a 12 GB checkpoint in a second global cache.
                from huggingface_hub import hf_hub_download
                source = PurePosixPath(item['source_file'])
                if source.is_absolute() or '..' in source.parts:
                    raise ValueError('Invalid source path')
                local_dir = dest.parents[len(source.parts) - 1]
                if dest.relative_to(local_dir).as_posix() != str(source):
                    raise ValueError(f'Source/destination layout mismatch: {dest}')
                downloaded = Path(hf_hub_download(repo_id=item['source_repository'],
                    revision=item['source_revision'], filename=str(source), local_dir=local_dir, token=False))
                if downloaded.resolve() != dest.resolve() or dest.stat().st_size != item['bytes'] or sha256(dest) != item['sha256']:
                    raise ValueError(f'Invalid download: {dest}')
                continue
            temp = dest.with_suffix(dest.suffix + '.part')
            with urllib.request.urlopen(item['url'], timeout=120) as response, temp.open('wb') as out:
                while block := response.read(8 << 20):
                    out.write(block)
            if temp.stat().st_size != item['bytes'] or sha256(temp) != item['sha256']:
                raise ValueError(f'Invalid download: {temp}')
            temp.replace(dest)
        else:
            if item['kind'] == 'base':
                if not args.base_source:
                    continue
                source = args.base_source / Path(item['path']).relative_to('assets/base')
            else:
                source = getattr(args, item['method'] + '_source')
                if not source:
                    continue
            source = source.expanduser().resolve()
            if not source.is_file() or source.stat().st_size != item['bytes']:
                raise ValueError(f'Missing or wrong-sized source: {source}')
            if dest.exists() or dest.is_symlink():
                if dest.resolve() != source:
                    raise FileExistsError(f'Refusing to overwrite: {dest}')
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.symlink_to(os.path.relpath(source, dest.parent))
    print('Staging finished. Run verify before launch. Local symlinks are not a transport format.')


if __name__ == '__main__':
    main()
