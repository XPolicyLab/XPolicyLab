"""Bind the frozen RGB coin package to its original runtime conditions."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from ..core.curobo_runtime import require_curobo
from ..core.paths import ROOT, policy_path

TASK_ROOT = ROOT / 'tasks/deposit_coin'
PACKAGE_ROOT = ROOT / 'resources/deposit_coin/coap/vendor/code-skills/implementations/rdj_v1_deposit_coin/12fa351b9c54f64f/rdj_v1_deposit_coin'
PACKAGE_HASHES = json.loads((TASK_ROOT / 'package_hashes.json').read_text())
MODEL_SHA256 = '89e8340a0f31c5d5ed2c7be21a03bf22a36f4b8b3e535e9d5811bf178003c53f'


def verify_package():
    """Check every upstream source, calibration, mesh, and provenance file."""
    errors = []
    actual = {str(path.relative_to(PACKAGE_ROOT)) for path in PACKAGE_ROOT.rglob('*')
              if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc'}
    if actual != set(PACKAGE_HASHES):
        errors.append('Coin package file set differs from the package manifest')
    for relative, expected in PACKAGE_HASHES.items():
        path = PACKAGE_ROOT / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append('Coin package checksum mismatch: ' + relative)
    if PACKAGE_HASHES.get('model.py') != MODEL_SHA256:
        errors.append('Coin model identity differs from the declared candidate')
    return errors


def load_descriptor(seed):
    seed = int(str(seed))
    descriptor = json.loads((TASK_ROOT / 'descriptors/deposit_coin.json').read_text())
    identity = descriptor['implementation']
    if identity['module'] != 'rdj_v1_deposit_coin.model' or identity['class'] != 'Model' or identity['sha256'] != MODEL_SHA256:
        raise ValueError('Coin descriptor does not identify the frozen R12 model')
    descriptor['model_config']['seed'] = seed
    descriptor['environment']['RDJ_SEED'] = str(seed)
    return descriptor


def create(config, route, scope, run):
    errors = verify_package()
    if errors:
        raise ValueError('\n'.join(errors))
    descriptor = load_descriptor(config.get('seed', 0))
    checkpoint = config.get('ckpt_name')
    if checkpoint not in (None, '', 'none', 'null', MODEL_SHA256):
        raise ValueError('Coin R12 has no weights; ckpt_name must identify its model.py SHA256')
    require_curobo()
    output = Path(run).resolve()
    environment = {name: str(value).replace('{output}', str(output))
                   for name, value in descriptor['environment'].items()}
    trace = policy_path(os.environ['COIN_TRACE_DIR']) if os.environ.get('COIN_TRACE_DIR') else output / 'coin_trace'
    trace.mkdir(parents=True, exist_ok=True)
    environment['COIN_TRACE_DIR'] = str(trace)
    os.environ.update(environment)
    model_config = dict(descriptor['model_config'])
    for key in ('host', 'port', 'gpu_id', 'action_dim', 'policy_name', 'bench_name'):
        if config.get(key) is not None:
            model_config[key] = config[key]
    model_config['ckpt_name'] = MODEL_SHA256
    return scope.resolve(route['model'])(model_config)
