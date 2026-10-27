"""Official run naming and links to externally stored data/checkpoints."""
import hashlib
import json
import os
from pathlib import Path
import uuid

POLICY_ROOT = Path(__file__).resolve().parents[2]


def standard_name(args, *, include_seed=True):
    cfg = {key: getattr(args, key, None) for key in
           ('bench_name', 'ckpt_name', 'env_cfg_type', 'action_type', 'seed')}
    fields = ('bench_name', 'ckpt_name', 'env_cfg_type', 'action_type')
    if not any(cfg[key] is not None for key in fields):
        return None
    from XPolicyLab.utils.checkpoint_resolver import build_run_dir_name
    if any(not cfg[key] for key in fields):
        raise ValueError('Supply all four benchmark/checkpoint/profile/action fields')
    if cfg['bench_name'].lower() != 'robotwin' or cfg['action_type'] != 'joint':
        raise ValueError('Supported: RoboTwin / joint')
    for key in fields:
        value = cfg[key]
        if value in ('.', '..') or '/' in value or '\\' in value:
            raise ValueError(f'{key} must be a simple name, not a path')
    return build_run_dir_name(cfg, include_seed=include_seed)


def add_standard_arguments(parser):
    for flag in ('bench-name', 'ckpt-name', 'env-cfg-type', 'action-type'):
        parser.add_argument('--' + flag)


def check_link(link, target, *, checkpoint_output=None):
    """Refuse to replace real paths or links belonging to another run."""
    link, target = Path(link), Path(target).resolve()
    if not os.path.lexists(link):
        return
    if not link.is_symlink():
        raise FileExistsError(f'Refusing to replace existing path: {link}')
    current = link.resolve()
    if current == target:
        return
    if checkpoint_output is not None:
        owner = Path(checkpoint_output).resolve()
        if current.parent == owner and current.name.startswith('step_') and current.name[5:].isdigit():
            return
    raise FileExistsError(f'{link} already points to another output: {current}; use a new run name')


def publish_link(link, target, *, checkpoint_output=None):
    link, target = Path(link), Path(target).resolve(strict=True)
    link.parent.mkdir(parents=True, exist_ok=True)
    # Serialize cooperating writers and keep readers on the previous complete target.
    import fcntl
    with (link.parent / f'.{link.name}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        check_link(link, target, checkpoint_output=checkpoint_output)
        temporary = link.with_name(f'.{link.name}.{uuid.uuid4().hex}.tmp')
        try:
            temporary.symlink_to(target, target_is_directory=True)
            os.replace(temporary, link)
        finally:
            temporary.unlink(missing_ok=True)


def prepared_training_paths(name):
    directory = POLICY_ROOT / 'data' / name
    metadata = directory / 'train.jsonl'
    report_file = directory / 'validation.json'
    if not metadata.is_file() or not report_file.is_file():
        raise FileNotFoundError(f'Run process_data.sh with the same benchmark/checkpoint/profile/action first: {directory}')
    report = json.loads(report_file.read_text())
    root = Path(report['data_root'])
    if not root.is_absolute() or not root.is_dir():
        raise FileNotFoundError(f'Prepared data root is unavailable: {root}')
    if hashlib.sha256(metadata.read_bytes()).hexdigest() != report['jsonl_sha256']:
        raise ValueError(f'Prepared metadata changed after validation: {metadata}')
    return {'SFT_JSONL_PATH': str(metadata.resolve()), 'SFT_DATA_ROOT': str(root)}
