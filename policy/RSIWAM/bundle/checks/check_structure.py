"""CPU-only syntax, package identity, and standard adapter contract checks."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime'))
from rsiwam_loader import check_package, BASE_ASSETS, require_hash


def main():
    parsed = []
    for path in sorted(ROOT.rglob('*.py')):
        ast.parse(path.read_text(), filename=str(path.relative_to(ROOT)))
        parsed.append(str(path.relative_to(ROOT)))
    shell = sorted(ROOT.rglob('*.sh'))
    for path in shell:
        subprocess.run(['bash', '-n', str(path)], check=True)
    artifacts = check_package()
    require_hash(ROOT / 'base_weights.json', BASE_ASSETS)
    required = ['model.py', 'deploy.py', 'deploy.yml', '__init__.py', 'install.sh',
                'eval.sh', 'setup_eval_policy_server.sh', 'setup_eval_env_client.sh']
    for name in required:
        if not (ROOT / 'policy/RSIWAM' / name).is_file():
            raise ValueError('Missing standard adapter file: ' + name)
    result = dict(status='CPU_STRUCTURE_AND_IDENTITY_PASSED', python_files=len(parsed),
                  shell_files=len(shell), standard_adapter_files=required,
                  candidate_id=artifacts['commit']['identity']['candidate_id'],
                  optimizer_step=artifacts['commit']['step'],
                  gpu_used=False, clean_environment_validated=False,
                  simulator_evaluation_run=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
