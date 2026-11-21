"""Install the pinned Simate inference environment/assets without loading models."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from simate_assets import install_assets, load_metadata, sha256_file, validate_source

POLICY = Path(__file__).resolve().parent
FILES = ('model.py', 'hybrid_backend.py', 'preflight.py', 'deploy.yml',
         'simate_bridge.py', 'simate_server.py', 'simate_runtime.py', 'simate_backend.json')
PROBE = '''import json,platform,torch,sys
assert sys.version_info[:2]==(3,11), 'Simate requires Python 3.11'
assert tuple(int(x) for x in torch.__version__.split('.')[:2]) in {(2,6),(2,7),(2,8)}, 'Use CUDA PyTorch 2.6-2.8'
assert torch.version.cuda is not None, 'Simate requires a CUDA PyTorch build'
print(json.dumps(dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda)))
'''


def write(path, value):
    temporary = path.with_name(path.name + '.new-' + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def run(command, env, *, capture=False):
    result = subprocess.run([str(arg) for arg in command], check=True, env=env,
                            capture_output=capture, text=True)
    return result.stdout.strip() if capture else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-input', action='store_true', help='verify source/manifests only, no install/download')
    args = parser.parse_args()
    metadata = load_metadata(POLICY)
    source = validate_source(POLICY)
    profile = json.loads((POLICY / 'submission.json').read_text())['hybrid']
    if (profile.get('variant'), profile.get('approved_steps_min'), profile.get('approved_steps_max'),
            profile.get('vla_horizon')) != ('adaptive_2_50', 2, 50, 50):
        raise ValueError('Simate submission requires the authenticated adaptive_2_50 profile')
    if args.check_input:
        print(json.dumps({'source': source, 'backend': 'simate_beta', 'check': 'offline inputs only'}))
        return
    state = POLICY / 'runtime/simate-install'
    for name in ('tmp', 'cache'):
        (state / name).mkdir(parents=True, exist_ok=True)
    env = {name: value for name, value in os.environ.items()
           if not any(marker in name.upper() for marker in ('API_KEY', 'ACCESS_TOKEN', 'SECRET'))}
    env.update(TMPDIR=str(state / 'tmp'), TMP=str(state / 'tmp'), TEMP=str(state / 'tmp'),
               UV_CACHE_DIR=os.environ.get('UV_CACHE_DIR', str(state / 'cache/uv')),
               PIP_CACHE_DIR=str(state / 'cache/pip'), PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
    env.pop('PYTHONHOME', None)
    env.pop('PYTHONPATH', None)
    explicit = os.environ.get('ROBORSI_SIM_MODEL_PYTHON')
    python = Path(os.path.abspath(os.path.expanduser(explicit))) if explicit else POLICY / 'runtime/simate-env/bin/python'
    if not python.is_file():
        if explicit:
            raise ValueError('Explicit ROBORSI_SIM_MODEL_PYTHON is missing; no fallback was selected')
        base = os.environ.get('ROBORSI_SIM_BASE_PYTHON') or os.environ.get('ROBORSI_BOOTSTRAP_PYTHON')
        if not base:
            base = shutil.which('python3.11') or shutil.which('python3')
        if not base:
            raise ValueError('Activate the official RoboDojo Python 3.11 environment before installing')
        run([base, '-B', '-c', PROBE], env)
        destination = python.parent.parent
        if destination.exists() or destination.is_symlink():
            raise ValueError('Existing default Simate environment is incomplete; preserve it before reinstalling')
        run([base, '-m', 'venv', '--system-site-packages', destination], env)
    # Reuse the official CUDA PyTorch, preserving simulator and sibling-policy files.
    # All non-Torch dependencies below are fixed by the committed SHA-256 lock.
    torch_identity = json.loads(run([python, '-B', '-c', PROBE], env, capture=True))
    uv = shutil.which('uv', path=env.get('PATH'))
    if not uv:
        candidate = POLICY / 'runtime/bootstrap/uv/bin/uv'
        if candidate.is_file():
            uv = str(candidate)
    if not uv:
        raise ValueError('Run bash install.sh first to prepare the authenticated uv installer')
    run([uv, 'pip', 'install', '--python', python, '--require-hashes', '--no-deps',
         '--requirement', POLICY / 'simate/requirements.lock'], env)
    imports = ('import cv2,numpy,sentencepiece,safetensors,websockets,msgpack,yaml; '
               'from websockets.asyncio.server import serve; print("Simate dependency imports passed")')
    run([python, '-B', '-c', imports], env)
    checkpoint = Path(os.environ.get('ROBORSI_SIM_CHECKPOINT',
                      str(POLICY / 'checkpoints/simate-beta-robodojo-eval'))).expanduser().resolve()
    assets = install_assets(checkpoint, POLICY)
    alias = POLICY / 'checkpoints/simate-beta-robodojo-eval'
    alias.parent.mkdir(exist_ok=True)
    if alias.resolve() != checkpoint:
        if alias.exists() or alias.is_symlink():
            raise ValueError('Preserve existing checkpoint alias; ROBORSI_SIM_CHECKPOINT selects another directory')
        alias.symlink_to(checkpoint, target_is_directory=True)
    settings = {'schema_version': 1, 'backend': 'simate_beta', 'model_python': str(python),
                'checkpoint_dir': str(checkpoint), 'controller': 'adaptive_2_50', 'exec_chunk_size': 50}
    write(POLICY / 'simate_backend.json', settings)
    installed = {name: sha256_file(POLICY / name) for name in FILES}
    receipt = {'schema_version': 1, 'installed_at': datetime.now(timezone.utc).isoformat(),
               'settings': settings, 'model_python': str(python), 'checkpoint': str(checkpoint),
               'vla_backend': 'simate_beta', 'adaptive_variant': 'adaptive_2_50',
               'installed_sha256': installed,
               'files': {name: {'installed_sha256': digest} for name, digest in installed.items()},
               'source': source, 'torch': torch_identity, 'assets': assets,
               'requirements_sha256': metadata['resources']['requirements.lock'],
               'actions': 'installed fixed Simate dependencies/assets; no model, simulator or model API started'}
    write(POLICY / 'simate-install.json', receipt)
    print(json.dumps({'status': 'ready', 'backend': 'simate_beta', 'receipt': str(POLICY / 'simate-install.json')}))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise SystemExit('Simate installation failed: ' + str(error)) from None
