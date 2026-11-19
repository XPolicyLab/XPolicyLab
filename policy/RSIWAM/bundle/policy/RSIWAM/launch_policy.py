"""Prepare the standard XPolicyLab server command; --dry-run does not load weights."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=19960)
    parser.add_argument('--xpl-root', default=os.environ.get('RSIWAM_XPL_ROOT'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not args.xpl_root:
        parser.error('Set RSIWAM_XPL_ROOT or --xpl-root')
    root = Path(args.xpl_root).expanduser().resolve()
    if not (root / 'setup_policy_server.py').is_file():
        parser.error('XPolicyLab setup_policy_server.py was not found')
    if not 1024 <= args.port <= 65535:
        parser.error('Use a port in 1024..65535')
    package = Path(__file__).resolve().parents[2]
    if (root / 'policy/RSIWAM').resolve() != package / 'policy/RSIWAM':
        parser.error('Link XPolicyLab/policy/RSIWAM to this complete package/policy/RSIWAM; do not copy the policy folder alone')
    overrides = dict(policy_name='RSIWAM', task_name=args.task, host=args.host, port=args.port)
    command = [sys.executable, '-u', str(root / 'setup_policy_server.py'),
               '--config_path', str(package / 'policy/RSIWAM/deploy.yml'), '--overrides']
    command += [f'{key}={value}' for key, value in overrides.items()]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu,
               CUBLAS_WORKSPACE_CONFIG=':4096:8',
               PYTHONPATH=os.pathsep.join([str(root.parent), str(root),
                                          str(package / 'src'), str(package / 'runtime'),
                                          os.environ.get('PYTHONPATH', '')]))
    print(json.dumps(dict(display_name='RSIWAM', command=command, dry_run=args.dry_run,
                          cublas_workspace_config=env['CUBLAS_WORKSPACE_CONFIG'],
                          clean_environment_validated=False), indent=2), flush=True)
    if not args.dry_run:
        os.chdir(root.parent)
        os.execvpe(command[0], command, env)


if __name__ == '__main__':
    main()
