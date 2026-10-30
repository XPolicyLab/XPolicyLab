"""Task-only routing, usable by shell entrypoints without ML dependencies."""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REGISTRY = json.loads((ROOT / 'tasks/registry.json').read_text())['tasks']


def resolve(task, backend='auto'):
    if not isinstance(task, str) or not task.strip():
        raise ValueError('task_name is required')
    if backend not in ('auto', 'vla'):
        raise ValueError('me_brain_backend must be auto or vla')
    canonical = next((name for name, item in REGISTRY.items()
                      if task == name or task in item['aliases']), task)
    if backend == 'vla' or canonical not in REGISTRY:
        return dict(task_name=task, canonical_task=canonical, driver='vla',
                    action_type='joint', eval_batch=True, max_envs=None)
    return dict(REGISTRY[canonical], task_name=task, canonical_task=canonical)


def configure(config):
    cfg = dict(config)
    route = resolve(cfg.get('task_name'), cfg.get('me_brain_backend', 'auto'))
    action = cfg.get('action_type')
    if action not in (None, 'auto', route['action_type']):
        raise ValueError(f"{route['task_name']} requires {route['action_type']}, got {action}")
    if route['driver'] != 'vla' and cfg.get('env_cfg_type') != 'arx_x5':
        raise ValueError('Selected scripted policies require env_cfg_type=arx_x5')
    cfg.update(action_type=route['action_type'], eval_batch=route['eval_batch'])
    if 'request_timeout_s' in route:
        cfg['request_timeout_s'] = route['request_timeout_s']
    return cfg, route


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--backend', default='auto')
    parser.add_argument('--action', default='auto')
    parser.add_argument('--write-config', type=Path)
    parser.add_argument('--field', choices=('action_type', 'eval_batch', 'driver'))
    args = parser.parse_args()
    cfg, route = configure(dict(task_name=args.task, me_brain_backend=args.backend,
                                action_type=args.action, env_cfg_type='arx_x5'))
    if args.write_config:
        content = (ROOT / 'deploy.yml').read_text()
        overrides = dict(action_type=route['action_type'], eval_batch=route['eval_batch'], me_brain_backend=args.backend)
        if 'request_timeout_s' in route:
            overrides['request_timeout_s'] = route['request_timeout_s']
        for key, value in overrides.items():
            line = key + ': ' + json.dumps(value)
            if re.search(r'^' + key + r':.*$', content, re.M):
                content = re.sub(r'^' + key + r':.*$', line, content, flags=re.M)
            else:
                content += '\n' + line + '\n'
        args.write_config.write_text(content)
    value = route.get(args.field) if args.field else route
    print(json.dumps(value) if not isinstance(value, str) else value)
