"""Path relocation and fixed-registration checks; no model math is changed here."""
from pathlib import Path
import hashlib
import json
import os

POLICY_DIR = Path(__file__).resolve().parent
WORKSPACE = POLICY_DIR.parents[2]
TASKS = ('cap_pen', 'classify_objects', 'disassemble_LEGO', 'hang_mugs',
         'pack_objects_into_backpack', 'sweep_blocks')
CHECKPOINTS = {
    'jam': '9b8e4bbe1f8093d34886313ad2dd8a355276c660014bec84723c08d6d68c85a6',
    'fastwam': '383863b3c0f6d387e6a6cafcb34e2a65e52f11ebaf637d4ddf71b25cc9a86511',
}
STATS_SHA = 'f2aaab5cda6908fe278f1252d181c41c23998147f69f1fd76947d62fe64f2d1b'
FIXED = {'robot': 'piper_x', 'action_type': 'joint', 'action_dim': 14, 'fps': 25,
         'steps': 28630, 'action_horizon': 32, 'replan_steps': 24,
         'num_inference_steps': 10, 'seed': 42, 'mixed_precision': 'bf16',
         'text_cfg_scale': 1.0, 'rand_device': 'cpu', 'tiled': False, 'sigma_shift': None,
         'dataset_revision': '35efbc7dedfdbeeb6e95fb749bd885d73d483e41'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_deployment(path, *, verify_checkpoint=True):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / 'deployment.json'
    if path.name != 'deployment.json':
        raise ValueError('Expected a bundle directory or deployment.json')
    manifest = json.loads(path.read_text())
    method = manifest.get('method')
    if method not in CHECKPOINTS or manifest.get('checkpoint_sha256') != CHECKPOINTS[method]:
        raise ValueError('Registration must use its fixed step_028630 checkpoint')
    for key, value in FIXED.items():
        if manifest.get(key) != value:
            raise ValueError(f'Fixed inference/training identity mismatch: {key}')
    if manifest.get('tasks') != list(TASKS) or manifest.get('stats_sha256') != STATS_SHA:
        raise ValueError('Unexpected task subset or normalization identity')
    for key in ('code_root', 'checkpoint', 'dataset_stats'):
        manifest[key] = str((path.parent / manifest[key]).resolve())
    if Path(manifest['code_root']) != POLICY_DIR / 'runtime':
        raise ValueError('This candidate uses its bundled, versioned runtime only')
    external = os.environ.get('FASTWAM_ROOT')
    if external and Path(external).resolve() != Path(manifest['code_root']):
        raise ValueError('Unset FASTWAM_ROOT: a different runtime would change the registered policy')
    if sha256(manifest['dataset_stats']) != STATS_SHA:
        raise ValueError('Normalization checksum mismatch')
    if verify_checkpoint and sha256(manifest['checkpoint']) != CHECKPOINTS[method]:
        raise ValueError('Checkpoint checksum mismatch')
    return manifest


def validate_overrides(cfg, manifest):
    for key in ('seed', 'replan_steps', 'num_inference_steps', 'action_horizon'):
        if cfg.get(key) is not None and cfg[key] != manifest[key]:
            raise ValueError(f'{key} is fixed at {manifest[key]} for this registration')
    if cfg.get('task_name') not in TASKS:
        raise ValueError('task_name must be one of the six registered Piper X tasks')


def asset_entries():
    return json.loads((POLICY_DIR / 'assets_manifest.json').read_text())['files']


def verify_assets(root=WORKSPACE, *, hashes=False, method=None):
    checked = []
    for item in asset_entries():
        if item.get('method') and method and item['method'] != method:
            continue
        path = Path(root) / item['path']
        if not path.is_file() or path.stat().st_size != item['bytes']:
            raise ValueError(f'Missing or wrong-sized asset: {path}')
        if hashes and sha256(path) != item['sha256']:
            raise ValueError(f'Asset checksum mismatch: {path}')
        checked.append(item['path'])
    return checked
