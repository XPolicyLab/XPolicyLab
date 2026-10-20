"""Install project dependencies while preserving an existing device runtime."""
from __future__ import annotations

import argparse
from importlib import metadata, util
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib


def framework_constraints(required: dict[str, str]) -> list[str]:
    constraints = []
    for name, expected in required.items():
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            raise RuntimeError(
                f'Install {name} {expected} for your selected device runtime first.'
            ) from None
        if installed.partition('+')[0] != expected:
            raise RuntimeError(f'{name} must have base version {expected}.')
        if '\n' in installed or '\r' in installed:
            raise RuntimeError(f'Invalid installed metadata for {name}.')
        constraints.append(f'{name}==={installed}')
    return constraints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Check prerequisites without installing packages.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    runtime = tomllib.loads((root / 'runtime-requirements.toml').read_text(encoding='utf-8'))
    try:
        constraints = framework_constraints(runtime['framework'])
        if util.find_spec('decord') is None:
            raise RuntimeError('Install a compatible CPU decoder exposing the decord module first; see ENVIRONMENT.md.')
    except RuntimeError as error:
        parser.exit(1, f'{error}\n')
    if args.check_only:
        print('Runtime prerequisites found; validate device and decoder behavior with the data precheck.')
        return
    requirements = [f'{name}=={version}' for name, version in runtime['packages'].items()]
    with tempfile.TemporaryDirectory(prefix='crwm-install-') as temporary:
        path = Path(temporary) / 'runtime-constraints.txt'
        path.write_text('\n'.join(constraints) + '\n', encoding='utf-8')
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '--constraint', str(path), *requirements, '--editable', str(root)],
            check=True,
        )
    if framework_constraints(runtime['framework']) != constraints:
        raise RuntimeError('Framework packages changed during dependency installation.')


if __name__ == '__main__':
    main()
