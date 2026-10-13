"""Checked-in source/configuration and separately downloaded Joint96 weights."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile

POLICY_DIR = Path(__file__).resolve().parent
TOKENIZER_PATH = 'base_models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(root, relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError(f'Expected a relative release path: {relative}')
    return Path(root) / path


def read_source(policy_dir=POLICY_DIR):
    root = Path(policy_dir) / 'joint96_source'
    manifest = json.loads((root / 'source_manifest.json').read_text())
    if manifest.get('schema') != 'phoenix-joint96-source-v1' or not manifest.get('sha256'):
        raise ValueError('Missing checked-in Joint96 source manifest')
    for name, expected in manifest['sha256'].items():
        if sha256(relative_path(root, name)) != expected:
            raise ValueError(f'Checked-in source checksum mismatch: {name}')
    return manifest, {key: relative_path(root, value) for key, value in manifest['paths'].items()}


def prepare_tokenizer(root, source):
    """Copy the checked-in vocabulary to the model loader's standard layout."""
    target = Path(root) / TOKENIZER_PATH
    target.mkdir(parents=True, exist_ok=True)
    for path in sorted(Path(source).iterdir()):
        if not path.is_file():
            continue
        destination = target / path.name
        if destination.exists():
            if sha256(destination) != sha256(path):
                raise ValueError(f'Tokenizer copy differs from repository: {destination}')
        else:
            shutil.copy2(path, destination)


def read_release(policy_dir=POLICY_DIR):
    release = json.loads((Path(policy_dir) / 'phoenix_release.json').read_text())
    if release.get('schema') != 'phoenix-joint96-release-v2':
        raise ValueError('Expected robot-specific Phoenix_WAM release manifest')
    return release


def release_weights(release, robot_type=None):
    """Select one robot checkpoint, or all checkpoints for a shared download."""
    from .robot_profiles import SUPPORTED_ROBOTS
    robots = SUPPORTED_ROBOTS if robot_type is None else (robot_type,)
    selected = {}
    for robot in robots:
        if robot not in SUPPORTED_ROBOTS or robot not in release['checkpoints']:
            raise ValueError(f'Unsupported checkpoint robot: {robot}')
        key = 'checkpoint' if robot_type is not None else f'checkpoint_{robot}'
        selected[key] = release['checkpoints'][robot]
    if any(key.startswith('checkpoint') for key in release['weights']):
        raise ValueError('Shared weights must not override robot checkpoints')
    return {**selected, **release['weights']}


def load_assets(root, *, policy_dir=POLICY_DIR, robot_type="arx_x5"):
    root = Path(root).resolve(strict=True)
    release = read_release(policy_dir)
    specs = release_weights(release, robot_type)
    _, paths = read_source(policy_dir)
    from .robot_profiles import select_robot_assets
    paths = select_robot_assets(paths, robot_type)
    weights = {}
    for key, spec in specs.items():
        path = relative_path(root, spec['path'])
        if not path.is_file():
            raise FileNotFoundError(f'Missing {robot_type} {key}: {path}. '
                                    f'Run download_checkpoint.sh {root} --robot-type {robot_type}.')
        if sha256(path) != spec['sha256']:
            raise ValueError(f'Weight checksum mismatch: {key}')
        weights[key] = path
    # Vocabulary/config are code assets; only model weights come from ModelScope.
    prepare_tokenizer(root, paths.pop('tokenizer_root'))
    paths.update(weights)
    paths['base_model_root'] = root / 'base_models'
    paths['tokenizer_root'] = root / TOKENIZER_PATH
    paths['checkpoint_spec'] = specs['checkpoint']
    return dict(step=specs['checkpoint']['step'],
                checkpoint_sha256=specs['checkpoint']['sha256'], robot_type=robot_type), paths


def verify_remote_weights(*, policy_dir=POLICY_DIR, robot_type=None, api=None):
    """Verify the public repository contains every exact weight used by inference."""
    if api is None:
        from modelscope.hub.api import HubApi
        api = HubApi()
    specs = release_weights(read_release(policy_dir), robot_type)
    listings, result = {}, {}
    for key, spec in specs.items():
        identity = (spec['repo_id'], spec['revision'])
        if identity not in listings:
            listings[identity] = {item['Path']: item for item in api.get_model_files(
                model_id=identity[0], revision=identity[1], recursive=True) if item.get('Type') == 'blob'}
        entry = listings[identity].get(spec['filename'])
        if entry is None:
            raise FileNotFoundError(f'Missing published weight: {spec["repo_id"]}/{spec["filename"]}')
        if str(entry.get('Sha256', '')).lower() != spec['sha256']:
            raise ValueError(f'Published weight differs from pinned checksum: {key}')
        if 'size' in spec and int(entry.get('Size', -1)) != spec['size']:
            raise ValueError(f'Published weight size differs from release: {key}')
        result[key] = dict(repo_id=spec['repo_id'], revision=spec['revision'],
            filename=spec['filename'], sha256=spec['sha256'], published=True)
    return result


def download_weights(root, *, policy_dir=POLICY_DIR, robot_type=None, api=None, download_file=None, cache_dir=None):
    """Download individual pinned weight files; never download source or archives."""
    if api is None or download_file is None:
        from modelscope.hub.api import HubApi
        from modelscope.hub.file_download import model_file_download
        api = api or HubApi()
        download_file = download_file or model_file_download
    root, policy_dir = Path(root).resolve(), Path(policy_dir)
    specs = release_weights(read_release(policy_dir), robot_type)
    _, source = read_source(policy_dir)
    missing, listings = [], {}
    for key, spec in specs.items():
        path = relative_path(root, spec['path'])
        if path.exists():
            if not path.is_file() or sha256(path) != spec['sha256']:
                raise ValueError(f'Refusing to replace an existing mismatched weight: {path}')
            continue
        identity = (spec['repo_id'], spec['revision'])
        if identity not in listings:
            listings[identity] = api.get_model_files(model_id=identity[0], revision=identity[1], recursive=True)
        entry = next((item for item in listings[identity] if item.get('Type') == 'blob'
                      and item.get('Path') == spec['filename']), None)
        if entry is None:
            raise FileNotFoundError(f'Missing published weight: {spec["repo_id"]}/{spec["filename"]}. '
                                    'Upload this weight using the filename in phoenix_release.json.')
        if str(entry.get('Sha256', '')).lower() != spec['sha256']:
            raise ValueError(f'Published weight differs from pinned checksum: {key}')
        missing.append(spec)
    # Check that all required files are published before downloading large files.
    root.mkdir(parents=True, exist_ok=True)
    for spec in missing:
        print(f'Downloading weight: {spec["repo_id"]}/{spec["filename"]}', flush=True)
        cached = Path(download_file(model_id=spec['repo_id'], file_path=spec['filename'],
                                    revision=spec['revision'], cache_dir=str(cache_dir or policy_dir / '.cache/modelscope')))
        if sha256(cached) != spec['sha256']:
            raise ValueError(f'Downloaded weight checksum mismatch: {spec["filename"]}')
        target = relative_path(root, spec['path'])
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.weight-', delete=False) as stream:
            temporary = Path(stream.name)
        try:
            shutil.copyfile(cached, temporary)
            if sha256(temporary) != spec['sha256']:
                raise ValueError('Copied weight checksum mismatch')
            temporary.rename(target)
        finally:
            temporary.unlink(missing_ok=True)
    prepare_tokenizer(root, source['tokenizer_root'])
    (root / 'WEIGHTS_READY.json').write_text(json.dumps(dict(robot_type=robot_type or 'all',
        sha256={key: spec['sha256'] for key, spec in specs.items()},
        source='checked-in joint96_source', model_loaded=False), indent=2) + '\n')
    print(f'Individual weights verified: {root}', flush=True)


def preflight(root, *, expected_step=None, robot_type="arx_x5"):
    from .model import _activate_runtime
    manifest, paths = load_assets(root, robot_type=robot_type)
    if expected_step is not None and int(expected_step) != manifest['step']:
        raise ValueError('Requested step differs from pinned checkpoint')
    _activate_runtime(paths['runtime_root'])
    import torch
    from .joint96_backend import load_inference_config, validate_contract
    from wam.model.modules.codecs.joint_visual_encoder import encoder_source_sha256
    payload = torch.load(paths['checkpoint'], mmap=True, weights_only=True, map_location='cpu')
    from .robot_profiles import checkpoint_robot_support
    support = checkpoint_robot_support(payload, paths['robot_profile'])
    cfg = load_inference_config(paths['inference_config'], paths)
    identity = validate_contract(cfg, payload, paths['dataset_stats'], manifest['step'])
    if encoder_source_sha256(paths['encoder_source_root']) != identity['source_sha256']:
        raise ValueError('Checked-in encoder source differs from the trained source')
    revision = paths['encoder_source_root'] / 'third_party/dino90k_runtime/dinov3/.source-commit'
    if revision.read_text().strip() != identity['dinov3_source_commit']:
        raise ValueError('Checked-in DINO revision differs from checkpoint')
    return dict(step=manifest['step'], model_loaded=False, identity=identity,
                checkpoint_sha256=manifest['checkpoint_sha256'], robot_support=support)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('weights', type=Path)
    parser.add_argument('--expected-step', type=int)
    parser.add_argument('--robot-type', choices=('arx_x5', 'piper', 'piper_x'),
                        help='Robot to download, verify or preflight; download/verify defaults to all robots')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--download', action='store_true')
    mode.add_argument('--verify-remote', action='store_true')
    parser.add_argument('--cache-dir', type=Path, help='ModelScope download cache directory')
    args = parser.parse_args()
    if args.download:
        download_weights(args.weights, robot_type=args.robot_type, cache_dir=args.cache_dir)
    elif args.verify_remote:
        print(json.dumps(verify_remote_weights(robot_type=args.robot_type), indent=2))
    else:
        if args.robot_type is None:
            parser.error('--robot-type is required for checkpoint preflight')
        print(json.dumps(preflight(args.weights, expected_step=args.expected_step, robot_type=args.robot_type), indent=2))
