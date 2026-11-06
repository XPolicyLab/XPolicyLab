"""One authoritative runtime/evaluation configuration for every Rule mode."""
import json
import argparse
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'configs/official-rule-evaluation.json'


def settings(mode=None, *, root=ROOT):
    root = Path(root)
    data = json.loads((root / CONFIG).read_text())
    if (data.get('official_eval') is not True or data.get('official_reward_only') is not True
            or data.get('policy_name') != 'Riemann-RSI-Preview'):
        raise ValueError('Rule modes require the common official evaluation configuration')
    if mode is not None:
        if mode not in data['tasks']:
            raise ValueError('Mode is outside the common Rule configuration: ' + mode)
        task = json.loads((root / 'tasks' / mode / 'task.json').read_text())
        if task.get('runtime_config') != CONFIG:
            raise ValueError('Rule task differs from the common runtime configuration: ' + mode)
    from .deployment import locations
    resolved = locations(root)
    data.update(official_root=resolved['official_root'], python=resolved['python'])
    data['deployment'] = {'cluster': resolved['cluster']}
    return data


def deployment_settings(*, root=ROOT):
    """Current platform locations, derived from the same official configuration."""
    common = settings(root=root)
    return dict(common['deployment'], python=common['python'], official_root=common['official_root'])


def require_python():
    config = settings()
    if (Path(sys.prefix).resolve() != Path(config['python']).parent.parent.resolve()
            or sys.version.split()[0] != config['official_python_version']):
        raise RuntimeError('Use ' + config['python'])


def official_root():
    return Path(settings()['official_root'])


def python_prefix():
    return Path(settings()['python']).parent.parent


def requested_mode(arguments):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--task_name')
    parser.add_argument('--policy_name')
    parser.add_argument('--protocol', choices=('ws',))
    parsed, _ = parser.parse_known_args(arguments)
    if parsed.task_name is not None:
        from .policy_catalog import resolve_task
        resolve_task(None, parsed.task_name)
    return parsed


def run_client(mode):
    """Every task's command-line entry delegates to the same official client."""
    from .policy_catalog import resolve_task
    resolve_task(None, mode)
    configuration = settings()
    arguments = sys.argv[1:]
    parsed = requested_mode(arguments)
    if parsed.task_name is not None and parsed.task_name != mode:
        raise ValueError('Task entry and requested official task differ')
    if parsed.task_name is None:
        arguments = [*arguments, '--task_name', mode]
    if parsed.policy_name is None:
        arguments = [*arguments, '--policy_name', configuration['policy_name']]
    target = ROOT / 'tools/run_official_eval.py'
    sys.argv = [str(target), *arguments]
    runpy.run_path(str(target), run_name='__main__')
    return 0
