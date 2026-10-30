"""Bind the frozen RGB coin package to its original runtime conditions."""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys

from ..core.paths import ROOT, policy_path

TASK_ROOT = ROOT / 'tasks/deposit_coin'
PACKAGE_ROOT = ROOT / 'resources/deposit_coin/coap/vendor/code-skills/implementations/rdj_v1_deposit_coin/12fa351b9c54f64f/rdj_v1_deposit_coin'
PACKAGE_HASHES = json.loads((TASK_ROOT / 'package_hashes.json').read_text())
MODEL_SHA256 = '89e8340a0f31c5d5ed2c7be21a03bf22a36f4b8b3e535e9d5811bf178003c53f'
DEPENDENCY_PATH = Path('dependencies/curobo/9077c9afff1c3fda/curobo')


def verify_package():
    """Check every upstream source, calibration, mesh, and provenance file."""
    errors = []
    actual = {str(path.relative_to(PACKAGE_ROOT)) for path in PACKAGE_ROOT.rglob('*')
              if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc'}
    if actual != set(PACKAGE_HASHES):
        errors.append('Coin package file set differs from the R12 archive')
    for relative, expected in PACKAGE_HASHES.items():
        path = PACKAGE_ROOT / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append('Coin package checksum mismatch: ' + relative)
    if PACKAGE_HASHES.get('model.py') != MODEL_SHA256:
        errors.append('Coin model identity differs from the declared candidate')
    return errors


def load_descriptor(seed):
    if isinstance(seed, bool) or str(seed) not in ('0', '1', '2'):
        raise ValueError('Coin R12 provides descriptors for seed 0, 1, or 2')
    descriptor = json.loads((TASK_ROOT / f'descriptors/seed{seed}/deposit_coin.json').read_text())
    identity = descriptor['implementation']
    if identity['module'] != 'rdj_v1_deposit_coin.model' or identity['class'] != 'Model' or identity['sha256'] != MODEL_SHA256:
        raise ValueError('Coin descriptor does not identify the frozen R12 model')
    if descriptor['model_config']['seed'] != int(seed) or descriptor['environment']['RDJ_SEED'] != str(seed):
        raise ValueError('Coin descriptor seed mismatch')
    return descriptor


def _dependency_path(config):
    assets = config.get('coin_assets_root') or os.environ.get('ME_BRAIN_COIN_ASSETS')
    if not assets:
        raise ValueError('coin_assets_root or ME_BRAIN_COIN_ASSETS is required')
    return policy_path(assets) / DEPENDENCY_PATH


def dependency_issues(config):
    issues = []
    try:
        dependency = _dependency_path(config)
    except ValueError as error:
        issues.append(str(error))
    else:
        locked = json.loads((TASK_ROOT / 'dependencies.json').read_text())
        for relative, expected in locked.items():
            path = dependency / relative
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                issues.append('Coin CuRobo dependency mismatch: ' + relative)
        loaded = sys.modules.get('curobo')
        if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(dependency.resolve()):
            issues.append('Another CuRobo implementation is already imported')
    try:
        if importlib.metadata.version('websockets') != '14.2':
            issues.append('Coin policy environment requires websockets==14.2')
    except importlib.metadata.PackageNotFoundError:
        issues.append('Coin policy environment requires websockets==14.2')
    if importlib.util.find_spec('torch') is None:
        issues.append('Coin policy environment requires CUDA-enabled torch')
    return issues


def _validate_dependencies(config):
    issues = dependency_issues(config)
    if issues:
        raise ValueError('\n'.join(issues))


def create(config, route, scope, run):
    errors = verify_package()
    if errors:
        raise ValueError('\n'.join(errors))
    descriptor = load_descriptor(config.get('seed', 0))
    checkpoint = config.get('ckpt_name')
    if checkpoint not in (None, '', 'none', 'null', MODEL_SHA256):
        raise ValueError('Coin R12 has no weights; ckpt_name must identify its model.py SHA256')
    _validate_dependencies(config)
    output = Path(run).resolve()
    environment = {name: str(value).replace('{output}', str(output))
                   for name, value in descriptor['environment'].items()}
    trace = policy_path(os.environ['COIN_TRACE_DIR']) if os.environ.get('COIN_TRACE_DIR') else output / 'coin_trace'
    trace.mkdir(parents=True, exist_ok=True)
    environment['COIN_TRACE_DIR'] = str(trace)
    os.environ.update(environment)
    dependency = str(_dependency_path(config))
    while dependency in sys.path:
        sys.path.remove(dependency)
    sys.path.insert(0, dependency)
    model_config = dict(descriptor['model_config'])
    for key in ('host', 'port', 'gpu_id', 'action_dim', 'policy_name', 'bench_name'):
        if config.get(key) is not None:
            model_config[key] = config[key]
    model_config['ckpt_name'] = MODEL_SHA256
    return scope.resolve(route['model'])(model_config)
