"""CPU argument/environment tests with fake conda/XPolicyLab paths; no servers."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix='launch-fixture-', dir=ROOT / 'checks') as folder:
        tmp = Path(folder)
        xpl = tmp / 'XPolicyLab'
        (xpl / 'policy').mkdir(parents=True)
        (xpl / 'setup_policy_server.py').write_text("raise RuntimeError('Fixture must never execute')\n")
        (xpl / 'policy/RSIWAM').symlink_to(ROOT / 'policy/RSIWAM', target_is_directory=True)
        command = [sys.executable, str(ROOT / 'policy/RSIWAM/launch_policy.py'),
                   '--task', 'stack_bowls', '--xpl-root', str(xpl), '--dry-run']
        receipt = json.loads(subprocess.check_output(command, text=True))
        assert receipt['dry_run'] and receipt['cublas_workspace_config'] == ':4096:8'
        assert 'policy_name=RSIWAM' in receipt['command']
        assert not any(value.startswith('seed=') for value in receipt['command'])
        (xpl / 'policy/RSIWAM').unlink()
        failed = subprocess.run(command, capture_output=True, text=True)
        assert failed.returncode != 0 and 'do not copy the policy folder alone' in failed.stderr
        fake = tmp / 'fake_conda'
        (fake / 'bin').mkdir(parents=True)
        (fake / 'etc/profile.d').mkdir(parents=True)
        conda = fake / 'bin/conda'
        conda.write_text('#!/bin/bash\nprintf "%s\\n" "$FAKE_CONDA_ROOT"\n')
        conda.chmod(0o755)
        (fake / 'etc/profile.d/conda.sh').write_text('conda() { export CONDA_PREFIX="$FAKE_CONDA_ROOT"; }\n')
        python = fake / 'bin/python'
        python.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
        python.chmod(0o755)
        env = dict(os.environ, PATH=str(fake / 'bin') + os.pathsep + os.environ['PATH'],
                   FAKE_CONDA_ROOT=str(fake), RSIWAM_XPL_ROOT=str(xpl),
                   RSIWAM_BASE_WEIGHTS=str(tmp / 'missing_external_weights'))
        server = ROOT / 'policy/RSIWAM/setup_eval_policy_server.sh'
        for seed in ['1', '7', '42']:
            args = ['bash', str(server), 'RoboDojo', 'stack_bowls',
                    'rsiwam-original32-lora33-step400', 'arx_x5', 'ee', seed,
                    '0', 'FAKE_ENV', '19960', '127.0.0.1']
            output = subprocess.check_output(args, env=env, text=True).splitlines()
            assert output[0] == '-u' and output[2:4] == ['--task', 'stack_bowls']
            assert '--seed' not in output
    print(json.dumps(dict(status='CPU_LAUNCH_CONTRACT_PASSED',
                         cublas_workspace_config=':4096:8',
                         complete_package_symlink_required=True,
                         environment_seeds_accepted=[1, 7, 42],
                         model_seed_override_forwarded=False,
                         fake_conda_used=True, fake_xpolicylab_used=True,
                         gpu_used=False, server_started=False,
                         clean_environment_validated=False), indent=2))


if __name__ == '__main__':
    main()
