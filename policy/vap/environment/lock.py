"""Lock a complete public wheelhouse without modifying the Python environment.

Run with CPython 3.11. The download phase resolves the requirements using public
PyPI and direct PyTorch wheel URLs. The wheelhouse is only a reproducible build
input, never a runtime path; install.py independently enforces every lock hash.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import unquote, urlsplit
import urllib.request

HERE = Path(__file__).resolve().parent


def public_environ():
    env = dict(os.environ)
    for name in list(env):
        if name.startswith('PIP_') or name in ('PYTHONHOME', 'PYTHONPATH', 'PYTHONSTARTUP'):
            env.pop(name, None)
    env.update(PIP_DISABLE_PIP_VERSION_CHECK='1', PIP_ROOT_USER_ACTION='ignore',
               PIP_CONFIG_FILE=os.devnull, PYTHONNOUSERSITE='1')
    return env


def direct_requirements():
    requirements = {}
    for line in (HERE / 'requirements.in').read_text().splitlines():
        if ' @ ' in line:
            name, url = line.split(' @ ', 1)
            filename = unquote(urlsplit(url.split('#', 1)[0]).path.rsplit('/', 1)[-1])
            digest = url.rsplit('#sha256=', 1)[1]
            requirements[filename] = (name, digest)
    return requirements


def pypi_metadata(name, version, filename):
    with urllib.request.urlopen(f'https://pypi.org/pypi/{name}/{version}/json', timeout=60) as response:
        metadata = json.load(response)
    matches = [item for item in metadata['urls'] if item['filename'] == filename
               and item['packagetype'] == 'bdist_wheel']
    if len(matches) != 1:
        raise ValueError(f'Cannot identify public PyPI wheel: {name}=={version} {filename}')
    return matches[0]


def lock_wheelhouse(target):
    direct = direct_requirements()
    wheels = sorted(target.glob('*.whl'))
    if not wheels:
        raise ValueError('Public dependency resolution produced no wheels')
    lines = ['# Complete public CPython 3.11 Linux x86_64 runtime lock.',
             '# Generate with python3.11 -I environment/lock.py WHEELHOUSE.',
             '# Every artifact is binary-only and independently checked by install.py.']
    for wheel in wheels:
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
        if wheel.name in direct:
            name, expected = direct.pop(wheel.name)
            if digest != expected:
                raise ValueError(f'Public direct wheel hash mismatch: {wheel.name}')
            requirement = next(line for line in (HERE / 'requirements.in').read_text().splitlines()
                               if line.startswith(name + ' @ '))
        else:
            name, version = wheel.name.split('-')[:2]
            artifact = pypi_metadata(name, version, wheel.name)
            if artifact['digests']['sha256'] != digest:
                raise ValueError(f'Public PyPI wheel hash mismatch: {wheel.name}')
            requirement = f'{name}=={version}'
        lines.extend((requirement + ' \\', f'    --hash=sha256:{digest}'))
    if direct:
        raise ValueError(f'Missing locked direct wheels: {sorted(direct)}')
    lock = HERE / 'requirements.lock'
    if lock.exists() or lock.is_symlink():
        raise FileExistsError(f'Refusing to overwrite lock: {lock}')
    lock.write_text('\n'.join(lines) + '\n')
    print(f'Locked {len(wheels)} public wheels: {lock}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wheelhouse', type=Path,
                        help='New empty directory for resolved public wheels')
    parser.add_argument('--index-url', default='https://pypi.org/simple',
                        help='Public PyPI transport; direct wheel hashes remain mandatory')
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        raise SystemExit('Run lock.py with CPython 3.11')
    target = args.wheelhouse.expanduser().absolute()
    if target.exists() or target.is_symlink():
        raise SystemExit(f'Refusing to overwrite a wheelhouse: {target}')
    target.mkdir(parents=True)
    command = [sys.executable, '-I', '-m', 'pip', 'download', '--dest', target,
               '--index-url', args.index_url, '--only-binary', ':all:',
               '--retries', '2', '--timeout', '60', '--requirement', HERE / 'requirements.in']
    print('+', ' '.join(str(arg) for arg in command), flush=True)
    subprocess.run(command, check=True, env=public_environ())
    lock_wheelhouse(target)


if __name__ == '__main__':
    main()
