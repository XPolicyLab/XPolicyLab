#!/usr/bin/env python3
"""Download the pinned six-task Piper X real dataset, with LFS SHA256 checks."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

TASKS = ('cap_pen', 'classify_objects', 'disassemble_LEGO', 'hang_mugs',
         'pack_objects_into_backpack', 'sweep_blocks')
REPO = 'RoboDojo-Benchmark/Robodojo'
REVISION = '35efbc7dedfdbeeb6e95fb749bd885d73d483e41'
ROOT = Path('/original-workspace/robodojo_piperx_20261013')


def json_url(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            sha.update(block)
    return sha.hexdigest()


def download(item, root):
    path = root / item['path']
    expected = item['lfs']['oid']
    if path.exists() and path.stat().st_size == item['size'] and digest(path) == expected:
        return str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.part')
    url = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{item["path"]}?download=true'
    for attempt in range(4):
        try:
            sha = hashlib.sha256()
            with urllib.request.urlopen(url, timeout=180) as response, temporary.open('wb') as out:
                while block := response.read(4 << 20):
                    sha.update(block)
                    out.write(block)
            if temporary.stat().st_size != item['size'] or sha.hexdigest() != expected:
                raise ValueError(f'Invalid size or SHA256: {path}')
            os.replace(temporary, path)
            return str(path)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--manifest-only', action='store_true')
    parser.add_argument('--limit-per-task', type=int, default=0)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.root / 'download_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        assert manifest['revision'] == REVISION and manifest['tasks'] == list(TASKS)
    else:
        files = []
        for task in TASKS:
            prefix = f'data/RoboDojo_real/{task}/piper_x/data'
            items = json_url(f'https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/{prefix}?limit=1000')
            episodes = [x for x in items if x['type'] == 'file' and x['path'].endswith('.hdf5')]
            if len(episodes) != 100:
                raise ValueError(f'{task}: expected 100 demos, got {len(episodes)}')
            files.extend(sorted(episodes, key=lambda x: x['path']))
        manifest = dict(repo=REPO, revision=REVISION, tasks=list(TASKS), files=files,
                        total_bytes=sum(x['size'] for x in files))
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'files'}), flush=True)
    if args.manifest_only:
        return
    files = manifest['files']
    if args.limit_per_task:
        files = [x for task in TASKS for x in
                 [f for f in files if f'/'+task+'/' in f['path']][:args.limit_per_task]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download, item, args.root) for item in files]
        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            path = future.result()
            if count <= 6 or count % 20 == 0:
                print(f'Downloaded and verified {count}/{len(files)} {path}', flush=True)
    if len(files) == 600:
        (args.root / 'download_complete.json').write_text(json.dumps(dict(
            revision=REVISION, verified_files=600, total_bytes=manifest['total_bytes'])) + '\n')


if __name__ == '__main__':
    main()
