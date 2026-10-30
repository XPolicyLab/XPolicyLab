"""Apply external machine resources without changing task routing."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .paths import ROOT, normalize_config, normalize_environment, policy_path


def _settings(filename, active=()):
    filename = Path(filename)
    identity = filename.resolve()
    if identity in active:
        raise ValueError('Runtime configuration includes contain a cycle')
    settings = json.loads(filename.read_text())
    includes = settings.get('includes', [])
    if not isinstance(includes, list) or not all(isinstance(item, str) for item in includes):
        raise ValueError('Runtime configuration includes must be a list of paths')
    for included in includes:
        path = Path(included).expanduser()
        if not path.is_absolute():
            path = filename.parent / path
        yield from _settings(path, (*active, identity))
    yield settings


def apply(config):
    cfg = dict(config)
    filename = os.environ.get('ME_BRAIN_RUNTIME_CONFIG')
    blocks = []
    if filename:
        for settings in _settings(policy_path(filename)):
            blocks.extend([settings.get('common', {}), settings.get('tasks', {}).get(cfg.get('task_name'), {})])
    for block in blocks:
        values = block.get('config', {})
        forbidden = {'task_name', 'action_type', 'eval_batch', 'policy_name', 'me_brain_backend'} & values.keys()
        if forbidden:
            raise ValueError('Resource settings cannot override routing: ' + str(sorted(forbidden)))
    for block in blocks:
        cfg.update(block.get('config', {}))
        for key, value in block.get('environment', {}).items():
            os.environ[key] = normalize_environment(key, value)
    for key in tuple(os.environ):
        if key in {'ME_BRAIN_RUNTIME_CONFIG'}:
            continue
        if key.endswith(('_PATH', '_FILE', '_ROOT', '_DIR', '_SOURCE', '_ASSETS', '_MODELS')) and os.environ[key]:
            os.environ[key] = normalize_environment(key, os.environ[key])
    return normalize_config(cfg)


def _same_content(first, second):
    if not first.is_file() or not second.is_file() or first.stat().st_size != second.stat().st_size:
        return False
    def checksum(path):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        return digest.digest()
    return checksum(first) == checksum(second)


def prepare(config, route):
    evidence = policy_path(config.get('strategy_evidence_dir') or os.environ.get('ME_BRAIN_EVIDENCE_DIR') or 'runtime')
    evidence.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=route['canonical_task'] + '-', dir=evidence))
    task = route['canonical_task']
    for prefix in ('B88R42WQ', 'COAP'):
        os.environ.update({prefix + '_TASK': route.get('policy_task', task), prefix + '_EVIDENCE_DIR': str(run), prefix + '_NUMERIC_IK': '1' if route.get('numeric_ik') else '0', prefix + '_PROBE_ONLY': '0', prefix + '_DEVELOPMENT_TRUTH': '0', prefix + '_PHYSICALRSI_NATIVE': '0', prefix + '_LXM6WUEC_NATIVE': '0'})
        os.environ.pop(prefix + '_DIAGNOSTIC_MODE', None)
    cfg = dict(config, task_name=task)
    for relative, external in cfg.get('source_assets', {}).items():
        relative = Path(relative)
        if relative.is_absolute() or '..' in relative.parts or 'assets' not in relative.parts:
            raise ValueError('source_assets must use relative asset paths')
        target = ROOT / 'resources' / task / relative
        external = policy_path(external)
        if not external.exists():
            raise FileNotFoundError('Configured source asset is unavailable')
        if target.exists() and not target.samefile(external):
            if not _same_content(target, external):
                raise ValueError('Conflicting task asset: ' + str(relative))
            continue
        if target.is_symlink():
            target.unlink()
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(os.path.relpath(external, target.parent), target_is_directory=external.is_dir())
    if route.get('calibrated'):
        cfg.update(calibration_path=str(ROOT / 'resources' / task / 'configs/calibration.json'), trace_path=str(run / 'trace.jsonl'))
    return cfg, run
