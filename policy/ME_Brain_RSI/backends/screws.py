"""Bind the frozen RGB screws package to its declared runtime conditions."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from ..core.curobo_runtime import require_curobo
from ..core.paths import ROOT, policy_path

TASK_ROOT = ROOT / 'tasks/fasten_screws'
PACKAGE_ROOT = ROOT / 'resources/fasten_screws/coap/vendor/code-skills/implementations/rdj_v1_fasten_screws/770a6c1b7de4b67b/rdj_v1_fasten_screws'
PACKAGE_HASHES = json.loads((TASK_ROOT / 'package_hashes.json').read_text())
MODEL_SHA256 = 'e9f5934437e891614e140a6c78b8b5bda3136e23779551aa068a90a54ed55a3f'


def verify_package():
    """Check all frozen source, calibration, mesh, and provenance bytes."""
    errors = []
    actual = {str(path.relative_to(PACKAGE_ROOT)) for path in PACKAGE_ROOT.rglob('*')
              if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc'}
    if actual != set(PACKAGE_HASHES):
        errors.append('Screws package file set differs from the package manifest')
    for relative, expected in PACKAGE_HASHES.items():
        path = PACKAGE_ROOT / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append('Screws package checksum mismatch: ' + relative)
    if PACKAGE_HASHES.get('model.py') != MODEL_SHA256:
        errors.append('Screws model identity differs from the declared candidate')
    return errors


def load_descriptor(seed):
    seed = int(str(seed))
    descriptor = json.loads((TASK_ROOT / 'descriptors/fasten_screws.json').read_text())
    identity = descriptor['implementation']
    if (identity['module'] != 'rdj_v1_fasten_screws.model'
            or identity['class'] != 'Model' or identity['sha256'] != MODEL_SHA256):
        raise ValueError('Screws descriptor does not identify the frozen R16 model')
    model = descriptor['model_config']
    environment = descriptor['environment']
    model['seed'] = seed
    environment['RDJ_SEED'] = str(seed)
    fixed = {'task_name': 'fasten_screws', 'env_cfg_type': 'arx_x5',
             'action_type': 'joint', 'eval_batch': True, 'protocol': 'ws',
             'ckpt_name': MODEL_SHA256}
    if any(model.get(key) != value for key, value in fixed.items()):
        raise ValueError('Screws descriptor model contract differs from R16')
    if environment.get('RDJ_CONFIG') != 'fasten_screws' or not environment.get('SCREWS_TRACE_DIR'):
        raise ValueError('Screws descriptor must declare its task and SCREWS_TRACE_DIR')
    return descriptor


def create(config, route, scope, run):
    errors = verify_package()
    if errors:
        raise ValueError('\n'.join(errors))
    descriptor = load_descriptor(config.get('seed', 0))
    checkpoint = config.get('ckpt_name')
    if checkpoint not in (None, '', 'none', 'null', MODEL_SHA256):
        raise ValueError('Screws R16 has no weights; ckpt_name must identify its model.py SHA256')
    require_curobo()
    output = Path(run).resolve()
    environment = {name: str(value).replace('{output}', str(output))
                   for name, value in descriptor['environment'].items()}
    trace = policy_path(os.environ.get('SCREWS_TRACE_DIR') or environment['SCREWS_TRACE_DIR'])
    trace.mkdir(parents=True, exist_ok=True)
    environment['SCREWS_TRACE_DIR'] = str(trace)
    os.environ.update(environment)
    model_config = dict(descriptor['model_config'])
    for key in ('host', 'port', 'gpu_id', 'action_dim', 'policy_name', 'bench_name'):
        if config.get(key) is not None:
            model_config[key] = config[key]
    model_config['ckpt_name'] = MODEL_SHA256
    return scope.resolve(route['model'])(model_config)
