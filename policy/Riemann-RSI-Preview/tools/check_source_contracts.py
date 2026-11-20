"""Standard-library source/declaration checks; no SDK, model or GPU startup."""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check(root=ROOT):
    loaded_before = set(sys.modules)
    from rsi.policy_catalog import catalog
    from rsi.task_contract import verify
    root = Path(root).resolve()
    names = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z']).decode().split('\0')
    python_count = json_count = 0
    for relative in filter(None, names):
        path = root / relative
        if path.suffix not in ('.py', '.json'):
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError('Tracked source/configuration must be an ordinary file: ' + relative)
        if path.suffix == '.py':
            ast.parse(path.read_bytes(), filename=relative)
            python_count += 1
        else:
            json.loads(path.read_bytes())
            json_count += 1
    modes = catalog(root)
    if len(modes) != 54:
        raise ValueError('The current official catalog must cover 54 modes')
    roots = {mode: verify(root, mode) for mode in modes}
    registry = json.loads((root / 'configs/skill-registry.json').read_text())
    learned_modes = {mode for mode, row in modes.items() if row['backend'] == 'learned'}
    if set(registry['modes']) != learned_modes:
        raise ValueError('The callable registry must contain exactly the current learned modes')
    selected = set()
    for mode in learned_modes:
        plan = json.loads((root / registry['modes'][mode]['composition']).read_text())
        selected.update(plan['bindings'].values())
        for module in plan['modules']:
            for slot in module['slots']:
                if not set(slot['allowed']).issubset(registry['skills']):
                    raise ValueError('A learned slot still declares a retired implementation')
    if set(registry['skills']) != selected:
        raise ValueError('The callable registry must not retain unused implementations')
    runtime_packages = ('numpy', 'torch', 'isaacsim', 'isaaclab', 'XPolicyLab')
    imported_runtime = [name for name in set(sys.modules) - loaded_before
                        if any(name == prefix or name.startswith(prefix + '.') for prefix in runtime_packages)]
    if imported_runtime:
        raise RuntimeError('A source-only check imported runtime packages: ' + ', '.join(sorted(imported_runtime)))
    return {'scope': 'source syntax and current executable declarations only',
            'python_files': python_count, 'json_files': json_count,
            'verified_modes': len(roots), 'sdk_or_model_loaded': False,
            'gpu_or_official_episode_launched': False,
            'external_resource_bytes_verified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(check(args.root), indent=2))


if __name__ == '__main__':
    main()
