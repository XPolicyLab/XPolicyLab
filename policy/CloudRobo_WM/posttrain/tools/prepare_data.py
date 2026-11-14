"""Prepare the published Clean index and validate the native training resources."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tempfile

POST = Path(__file__).resolve().parents[1]
POLICY = POST.parent


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def safe(root, relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or '..' in path.parts or '\\' in relative:
        raise ValueError('Invalid training resource path: ' + relative)
    target = root.joinpath(*path.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError('Training resource escapes its root: ' + relative)
    return target


def write_same(path, data):
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise FileExistsError('Prepared file differs; use a new bundle: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.prepare-', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def copy_same(source, target):
    if not source.is_file():
        raise FileNotFoundError('Missing standard training resource: ' + str(source))
    if target.exists():
        if not target.is_file() or sha(target) != sha(source):
            raise FileExistsError('Existing resource differs; file preserved: ' + str(target))
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.prepare-', delete=False) as stream:
        temporary = Path(stream.name)
        with source.open('rb') as incoming:
            shutil.copyfileobj(incoming, stream, 8 * 1024 * 1024)
    try:
        if sha(temporary) != sha(source):
            raise ValueError('Resource copy differs: ' + str(source))
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def metadata():
    seal = json.loads((POST / 'data/training_metadata_manifest.json').read_text())
    compressed = POST / 'data' / seal['index']['compressed_path']
    stats = POST / 'data' / seal['stats']['path']
    if sha(compressed) != seal['index']['compressed_sha256'] or sha(stats) != seal['stats']['sha256']:
        raise ValueError('Published Clean metadata changed')
    index_bytes = gzip.decompress(compressed.read_bytes())
    if len(index_bytes) != seal['index']['bytes'] or hashlib.sha256(index_bytes).hexdigest() != seal['index']['sha256']:
        raise ValueError('Clean index decompression identity differs')
    index, statistics = json.loads(index_bytes), json.loads(stats.read_text())
    episodes = index['episodes']
    counts = {}
    for episode in episodes:
        if episode['split'] != 'train' or episode['collection_regime'] != 'demo_clean':
            raise ValueError('Clean index contains a different split/regime')
        task = episode['clean_task']
        counts[task] = counts.get(task, 0) + 1
    digest = fingerprint(dict(manifest=index, split='train', window_stride=1))
    if (len({e['episode_id'] for e in episodes}) != seal['episodes']
            or sum(e['length'] for e in episodes) != seal['frames']
            or counts != seal['task_counts'] or len(counts) != 50 or set(counts.values()) != {50}
            or digest != seal['window_index_digest'] or statistics['contract']['manifests'] != [digest]):
        raise ValueError('Clean selection/statistics contract differs')
    return seal, index, index_bytes, stats.read_bytes(), statistics



def stage_prompt_resources(source, data, config):
    """Preserve the existing augmented prompt cache; never generate embeddings."""
    prompt = config['prompt_augmentation']
    if not prompt.get('enabled'):
        return
    source = source / 'prompt_clean'
    target = data / 'prompt_clean'
    manifest_path = source / 'manifest.json'
    if sha(manifest_path) != prompt['manifest_sha256']:
        raise ValueError('Augmented prompt manifest differs from the training configuration')
    manifest = json.loads(manifest_path.read_text())
    copy_same(manifest_path, target / 'manifest.json')
    for name, record in manifest['files'].items():
        original = safe(source, name)
        if original.stat().st_size != record['bytes'] or sha(original) != record['SHA256']:
            raise ValueError('Augmented prompt index differs: ' + name)
        copy_same(original, safe(target, name))
    vectors = sorted((source / 'vectors').iterdir())
    if len(vectors) != manifest['unique_prompts']:
        raise ValueError('Augmented prompt vector coverage differs')
    for original in vectors:
        if (original.is_symlink() or not original.is_file()
                or re.fullmatch(r'[0-9a-f]{64}\.t5_len128\.wan22ti2v5b\.pt', original.name) is None):
            raise ValueError('Unexpected augmented prompt cache entry: ' + original.name)
        copy_same(original, target / 'vectors' / original.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True, help='Downloaded original RobotWin3D directory')
    parser.add_argument('--bundle-root', type=Path, required=True, help='Separate writable training bundle directory')
    parser.add_argument('--training-resources', type=Path, help='Standard resource release containing eef/ and instruction_overlay/')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--index-only', action='store_true', help='Prepare/check the supplied index and statistics only; no READY publication')
    mode.add_argument('--verify-only', action='store_true', help='Validate an already prepared bundle without changing it')
    parser.add_argument('--prepare-permissions', action='store_true', help='Apply the supplied Ascend recipe 0777 data contract to user-owned input/bundle files')
    args = parser.parse_args()
    raw, bundle = args.data_root.expanduser().resolve(), args.bundle_root.expanduser().resolve()
    if not raw.is_dir() or raw == bundle or bundle.is_relative_to(raw):
        raise ValueError('Use an existing raw data directory and a separate bundle root')
    if args.verify_only and args.prepare_permissions:
        raise ValueError('--verify-only does not change permissions')
    ready = bundle / 'READY.json'
    if ready.exists() and not (args.index_only or args.verify_only):
        raise FileExistsError('Bundle already has READY.json; use --verify-only or a new bundle')
    seal, index, index_bytes, stats_bytes, statistics = metadata()
    data = bundle / 'data'
    if not args.verify_only:
        data.mkdir(parents=True, exist_ok=True)
        alias = data / 'robotwin_3d'
        if alias.exists() or alias.is_symlink():
            if alias.resolve() != raw:
                raise ValueError('Existing raw-data link targets another dataset')
        else:
            alias.symlink_to(raw, target_is_directory=True)
        write_same(data / 'indices/train_manifest_0000.json', index_bytes)
        write_same(data / 'indices/train_stats.json', stats_bytes)
    elif sha(data / 'indices/train_manifest_0000.json') != seal['index']['sha256'] or sha(data / 'indices/train_stats.json') != seal['stats']['sha256']:
        raise ValueError('Prepared index/statistics differ')
    if args.index_only:
        print(json.dumps({'Clean_index_prepared': True, 'episodes': seal['episodes'], 'windows': seal['frames'],
                          'tasks': 50, 'training_READY_written': False, 'models_loaded': False,
                          'index': str(data / 'indices/train_manifest_0000.json')}))
        return
    if args.training_resources and not args.verify_only:
        source = args.training_resources.expanduser().resolve()
        side = json.loads((source / 'eef/manifest.json').read_text())
        copy_same(source / 'eef/manifest.json', data / 'eef/manifest.json')
        for ep in index['episodes']:
            relative = side['episodes'][str(ep['episode_id'])]['pose_path']
            copy_same(safe(source / 'eef', relative), safe(data / 'eef', relative))
        for name in ['metadata/selected_training_reader_rows.json', 'cache/cache_jobs.json']:
            copy_same(source / 'instruction_overlay' / name, data / 'instruction_overlay' / name)
        jobs = json.loads((source / 'instruction_overlay/cache/cache_jobs.json').read_text())
        for job in jobs['jobs']:
            copy_same(safe(source / 'instruction_overlay/cache', job['filename']),
                      safe(data / 'instruction_overlay/cache', job['filename']))
        stage_prompt_resources(source, data, json.loads((POST / 'configs/data.json').read_text()))
    raw_manifest = json.loads((POST / 'data/robotwin_download_manifest.json').read_text())
    raw_before = {name: (safe(raw, name).stat().st_dev, safe(raw, name).stat().st_ino,
                         safe(raw, name).stat().st_size, safe(raw, name).stat().st_mtime_ns)
                  for name in raw_manifest['files']}
    # Verify pinned original raw bytes before constructing the native reader.
    subprocess.run([sys.executable, '-B', str(POLICY / 'tools/download_data.py'), '--output-dir', str(raw), '--verify-only'], check=True)
    for name, expected in seal['raw_metadata_sha256'].items():
        if sha(safe(raw, name)) != expected:
            raise ValueError('Original dataset metadata differs: ' + name)
    os.environ['CLOUDROBO_WM_DATA_ROOT'] = str(data)
    os.environ['CLOUDROBO_WM_MEDIA_REPLICA_RECEIPT'] = str(raw / 'RAW_MEDIA_REPLICA_SHA256.json')
    sys.path.insert(0, str(POST / 'src'))
    from cloudrobo_wm.datasets.multisource.history import HistoryRichPromptDataset
    cfg = json.loads((POST / 'configs/data.json').read_text())
    cfg['manifests'] = [str(data / x) for x in cfg['manifests']]
    cfg['stats_path'] = str(data / cfg['stats_path'])
    for key in ['instruction_overlay', 'prompt_augmentation']:
        cfg[key]['root'] = str(data / cfg[key]['root'])
    dataset = HistoryRichPromptDataset(**cfg)
    if len(dataset) != seal['frames'] or len(dataset.manifests[0]['episodes']) != seal['episodes']:
        raise ValueError('Native reader coverage differs')
    side = data / 'eef/manifest.json'
    group = next(iter(statistics['groups'].values()))
    if sha(side) != group['sidecar_manifest_sha256']:
        raise ValueError('Standard sidecar manifest differs from the published statistics')
    for adapter in dataset.adapters:
        for ep in adapter.manifest['episodes']:
            adapter.episode(ep)  # Checks original parquet, joint alignment and selected pose bytes.
            dataset._context(adapter.source, ep, anchor=0, occurrence=0)
    files = [safe(raw, name) for name in raw_manifest['files']]
    files += [f for f in data.rglob('*') if f.is_file() and not f.is_symlink() and not f.is_relative_to(data / 'robotwin_3d')]
    directories = set()
    for f in files:
        current = f.parent
        base = raw if f.is_relative_to(raw) else bundle
        while current.is_relative_to(base):
            directories.add(current)
            if current == base:
                break
            current = current.parent
    if args.prepare_permissions:
        for f in [*files, *directories]:
            if f.stat().st_uid != os.getuid():
                raise PermissionError('Refusing to change another owner\'s data permissions: ' + str(f))
        for f in [*files, *directories]:
            f.chmod(0o777)
    permissions = all(stat.S_IMODE(f.stat().st_mode) == 0o777 for f in [*files, *directories])
    if not permissions:
        raise ValueError('Ascend data permission contract not met; use --prepare-permissions on your own writable data/bundle')
    raw_receipt = json.loads((raw / 'RAW_DATA_DOWNLOAD_RECEIPT.json').read_text())['files']
    verified_raw = {}
    for name, previous in raw_before.items():
        f = safe(raw, name)
        current = f.stat()
        if previous != (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns):
            raise ValueError('Raw data changed during preparation: ' + name)
        verified_raw[str(f)] = raw_receipt[name]['sha256']
    inventory = {'version': 'cloudrobo_wm.prepared_data_files.v1', 'files':
                 [{'path': str(f), 'bytes': f.stat().st_size,
                   'sha256': verified_raw[str(f)] if str(f) in verified_raw else sha(f)}
                  for f in sorted(set(files))]}
    if not args.verify_only:
        write_same(bundle / 'PREPARED_DATA_FILES.json', (json.dumps(inventory, sort_keys=True, separators=(',', ':')) + '\n').encode())
        proof = dict(complete=True, target_sha256_verified=True, permissions_0777_verified=True,
                     data_manifest_sha256=sha(bundle / 'PREPARED_DATA_FILES.json'), episodes=2500,
                     windows=560900, tasks=50, native_reader_verified=True,
                     initialization_and_model_validation_included=False)
        write_same(ready, (json.dumps(proof, indent=2) + '\n').encode())
    elif not ready.is_file() or not json.loads(ready.read_text()).get('complete'):
        raise ValueError('Prepared data has no complete READY.json')
    print(json.dumps({'data_preparation_verified': True, 'episodes':2500, 'windows':560900,
                      'verify_only':args.verify_only, 'training_or_model_started':False}))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, ImportError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
