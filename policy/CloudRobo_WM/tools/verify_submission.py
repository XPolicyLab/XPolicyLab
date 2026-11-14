"""Check adapter interfaces, source syntax and packaged file integrity without loading models."""
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys


root = Path(sys.argv[1]).resolve()


def check_manifest(base, name):
    value = json.loads((base / name).read_text())['files']
    rows = value.items() if isinstance(value, dict) else ((row['path'], row) for row in value)
    for relative, row in rows:
        path = PurePosixPath(relative)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Invalid checksum path: ' + relative)
        target = base.joinpath(*path.parts)
        if not target.is_file() or not target.resolve().is_relative_to(base.resolve()):
            raise ValueError('Missing packaged file: ' + relative)
        if target.stat().st_size != row['bytes'] or hashlib.sha256(target.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Packaged file checksum differs: ' + relative)


required = ['README.md', '__init__.py', 'install.sh', 'eval.sh', 'model.py', 'deploy.py', 'deploy.yml',
            'setup_eval_policy_server.sh', 'setup_eval_env_client.sh', 'download_data.sh', 'process_data.sh',
            'train.sh', 'posttrain/FULL_TRAIN.sh', 'posttrain/SMOKE_RESUME.sh', 'posttrain/runtime/launch.py',
            'posttrain/tools/prepare_data.py']
for name in required:
    if not (root / name).is_file():
        raise FileNotFoundError(name)
deploy = {line.split(':', 1)[0]: line.split(':', 1)[1].strip()
          for line in (root / 'deploy.yml').read_text().splitlines() if ':' in line}
if deploy['policy_name'] != root.name or deploy['protocol'] != 'ws' or deploy['eval_batch'] != 'false':
    raise ValueError('Adapter deployment settings differ')
tree = ast.parse((root / 'model.py').read_text())
model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Model')
methods = {node.name for node in model.body if isinstance(node, ast.FunctionDef)}
if not {'__init__', 'update_obs', 'update_obs_batch', 'get_action', 'get_action_batch', 'reset'} <= methods:
    raise ValueError('Incomplete adapter interface')
for path in root.rglob('*'):
    if not path.is_file() or any(part in {'.venv', '__pycache__', '.git'} for part in path.relative_to(root).parts):
        continue
    if path.suffix == '.py':
        compile(path.read_text(), str(path), 'exec')
    elif path.suffix == '.sh':
        subprocess.run(['bash', '-n', str(path)], check=True)
check_manifest(root, 'CHECKSUMS.json')
check_manifest(root / 'posttrain', 'FILES_SHA256.json')
print(json.dumps({'static_check_passed': True,
                  'checks': ['adapter interface', 'deployment settings', 'Python and shell syntax',
                             'packaged file integrity', 'training source integrity'],
                  'models_loaded': False}, indent=2))
