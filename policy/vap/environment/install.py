"""Install or check the isolated public VAP policy runtime.

The default install target is the adapter's .venv. Only CPython 3.11 on
Linux x86_64 is supported by the locked binary wheels. No Conda environment,
private wheel, editable installation, model checkpoint or simulator is used.
"""
import argparse
import hashlib
import json
import os
import fcntl
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from types import SimpleNamespace
import urllib.request
from urllib.parse import unquote

HERE = Path(__file__).resolve().parent
POLICY_DIR = HERE.parent
LOCK = HERE / 'requirements.lock'


def run(args, **kwargs):
    print('+', ' '.join(str(arg) for arg in args), flush=True)
    subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def target_python(target):
    return target / 'bin' / 'python'


def install(args):
    python = Path(args.python).resolve() if args.python else Path(sys.executable).resolve()
    if not python.is_file():
        raise FileNotFoundError(f'Python 3.11 interpreter not found: {python}')
    check_interpreter(python)
    for path in (LOCK, HERE / 'bootstrap.lock'):
        if not path.is_file():
            raise FileNotFoundError(f'Required public hash lock is missing: {path}')
    versions = locked_versions()
    if set(versions) != set(locked_hashes(LOCK)):
        raise ValueError('Every locked runtime package must have a SHA256 hash')
    target = args.target.expanduser().absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f'Refusing to overwrite existing target: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    env = public_environ()
    lock_path = target.parent / f'.{target.name}.install.lock'
    staging = None
    published = False
    # Reuse the lock file after failures; flock, not file existence, is the lock.
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if target.exists() or target.is_symlink():
                raise FileExistsError(f'Refusing to overwrite existing target: {target}')
            staging = Path(tempfile.mkdtemp(prefix=f'{target.name}-installing-', dir=target.parent))
            # Install with the final venv identity but stage package payloads.
            # Relocating an ordinary venv would leave bin/pip and activate
            # pointing to the deleted staging path.
            run([python, '-I', '-m', 'venv', '--without-pip', target], env=env)
            published = True
            bootstrap_pip(python, target, env)
            venv_python = target_python(target)
            packages = staging / 'packages'
            run([venv_python, '-I', '-m', 'pip', '--python', venv_python, 'install',
                 '--ignore-installed', '--prefix', packages,
                 '--require-hashes', '--only-binary', ':all:',
                 '--index-url', args.index_url, '--retries', '2', '--timeout', '60',
                 '--requirement', LOCK], env=env)
            site_packages = target / 'lib' / 'python3.11' / 'site-packages'
            for child in (packages / 'lib' / 'python3.11' / 'site-packages').iterdir():
                child.rename(site_packages / child.name)
            scripts = packages / 'bin'
            if scripts.is_dir():
                for child in scripts.iterdir():
                    child.rename(target / 'bin' / child.name)
            report(SimpleNamespace(target=target, gpu=False, report=None))
            shutil.rmtree(staging)
        except BaseException:
            if staging is not None:
                shutil.rmtree(staging, ignore_errors=True)
            if published:
                # This invocation created the target while holding the lock.
                shutil.rmtree(target, ignore_errors=True)
            raise
    print(f'Installed public policy environment: {target}', flush=True)


def public_environ():
    env = dict(os.environ)
    # Inherited index/find-links/editable paths must not silently satisfy a
    # dependency. A mirror is an optional transport, not a trusted package source.
    for name in list(env):
        if name.startswith('PIP_') or name in ('PYTHONHOME', 'PYTHONPATH', 'PYTHONSTARTUP'):
            env.pop(name, None)
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               PIP_DISABLE_PIP_VERSION_CHECK='1', PIP_ROOT_USER_ACTION='ignore',
               PIP_CONFIG_FILE=os.devnull,
               PIP_CACHE_DIR=str(Path(env.get('VAP_INSTALL_CACHE',
                                               POLICY_DIR / '.cache' / 'pip')).resolve()))
    Path(env['PIP_CACHE_DIR']).mkdir(parents=True, exist_ok=True)
    for name in ('VAP_QWEN_MODEL', 'WALLX_MODEL_ROOT', 'VAP_CACHE_ROOT'):
        if name in os.environ:
            env[name] = os.environ[name]
    return env


def check_interpreter(python):
    run([python, '-I', '-c',
         'import platform,sys; assert sys.version_info[:2] == (3,11), sys.version; '
         'assert sys.platform == "linux" and platform.machine() == "x86_64", '
         '(sys.platform,platform.machine())'])


def bootstrap_pip(python, target, env):
    """Bootstrap only pip; runtime wheels do not need setuptools or wheel."""
    version = locked_versions(HERE / 'bootstrap.lock')['pip']
    name = f'pip-{version}-py3-none-any.whl'
    with urllib.request.urlopen(f'https://pypi.org/pypi/pip/{version}/json', timeout=60) as response:
        metadata = json.load(response)
    wheels = [item for item in metadata['urls']
              if item['packagetype'] == 'bdist_wheel' and item['filename'] == name]
    if len(wheels) != 1:
        raise ValueError(f'Cannot identify the public {name} artifact')
    wheel = wheels[0]
    expected = locked_hashes(HERE / 'bootstrap.lock')['pip']
    if wheel['digests']['sha256'] != expected:
        raise ValueError(f'PyPI metadata changed for {name}: {expected}')
    destination = Path(env['PIP_CACHE_DIR']) / name
    if destination.exists():
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if digest == expected:
            print(f'Using hash-verified cached {name}', flush=True)
        else:
            destination.unlink()
    if not destination.exists():
        partial = destination.with_suffix('.part')
        offset = partial.stat().st_size if partial.is_file() else 0
        if offset and hashlib.sha256(partial.read_bytes()).hexdigest() == expected:
            partial.replace(destination)
        elif offset >= wheel['size']:
            partial.unlink()
            offset = 0
        request = urllib.request.Request(wheel['url'])
        if offset:
            request.add_header('Range', f'bytes={offset}-')
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != (206 if offset else 200):
                offset = 0
            mode = 'ab' if offset and response.status == 206 else 'wb'
            if mode == 'wb':
                offset = 0
            with partial.open(mode) as output:
                while True:
                    try:
                        block = response.read(1024 * 1024)
                    except TimeoutError:
                        print(f'pip download paused at {offset} bytes; rerun to resume', file=sys.stderr)
                        raise
                    if not block:
                        break
                    output.write(block)
                    offset += len(block)
        if hashlib.sha256(partial.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Download incomplete or failed SHA256 verification; rerun to resume: {name}')
        partial.replace(destination)
    print(f'Bootstrapping verified public pip {version}', flush=True)
    run([python, '-I', '-c',
         'import sys; sys.path.insert(0, sys.argv[1]); '
         'from pip._internal.cli.main import main; raise SystemExit(main(sys.argv[2:]))',
         destination, '--python', target_python(target), 'install',
         '--no-deps', '--no-index', destination], env=env)
    run([target_python(target), '-I', '-m', 'pip', 'install',
         '--no-deps', '--require-hashes', '--index-url', 'https://pypi.org/simple',
         '--retries', '2', '--timeout', '60',
         '--requirement', HERE / 'bootstrap.lock'], env=env)
    for package in ('setuptools', 'wheel'):
        run([target_python(target), '-I', '-m', 'pip', 'uninstall', '--yes', package], env=env)


def requirement_lines(lock):
    # pip-tools places hashes on indented backslash-continuation lines.
    return [line for line in textwrap.dedent(lock.read_text()).replace('\\\n', '').splitlines()
            if not line.lstrip().startswith('#')]


def locked_hashes(lock):
    hashes = {}
    for line in requirement_lines(lock):
        match = re.match(r'^([A-Za-z0-9_.-]+)(?:==[^ ]+| @ \S+).*?--hash=sha256:([0-9a-f]{64})', line)
        if match:
            name, digest = match.groups()
            hashes[re.sub(r'[-_.]+', '-', name).lower()] = digest
    if not hashes:
        raise ValueError(f'No SHA256 hashes found in {lock}')
    return hashes


def locked_versions(lock=None):
    lock = LOCK if lock is None else lock
    versions = {}
    pattern = re.compile(r'^([A-Za-z0-9_.-]+)(?:==([^ ]+)| @ (\S+))')
    for line in requirement_lines(lock):
        match = pattern.match(line)
        if not match:
            continue
        name, version, url = match.groups()
        if url:
            from urllib.parse import urlsplit
            fragment = dict(part.split('=', 1) for part in urlsplit(url).fragment.split('&') if '=' in part)
            filename = unquote(urlsplit(url).path.rsplit('/', 1)[-1])
            version = filename.split('-')[1]
            if 'sha256' not in fragment:
                raise ValueError(f'Direct wheel lacks an immutable SHA256 fragment: {name}')
            if f'--hash=sha256:{fragment["sha256"]}' not in line:
                raise ValueError(f'Direct wheel fragment/hash mismatch: {name}')
        versions[re.sub(r'[-_.]+', '-', name).lower()] = version
    if not versions:
        raise ValueError(f'No packages found in {lock}')
    return versions


def module_report(python):
    root = str(POLICY_DIR.parents[2])
    xpl = str(POLICY_DIR.parents[1])
    repo = str(POLICY_DIR / 'upstream')
    code = """import importlib,importlib.metadata as md,json,os,platform,sys
sys.path[:0] = sys.argv[1:]
names = ['torch','torchvision','triton','transformers','numpy','scipy','yaml','safetensors',
         'einops','PIL','tqdm','ftfy','regex','imageio','websockets','msgpack','msgpack_numpy',
         'pydantic','cv2','accelerate','h5py','XPolicyLab.model_template']
r = {name: importlib.import_module(name).__file__ for name in names}
import torch,wall_x.utils.constant,wall_x.model.vap.serving,wall_x.model.vapv2.serving
r.update(sys_executable=sys.executable,resolved_executable=os.path.realpath(sys.executable),
         sys_path=sys.path,python_prefix=sys.prefix,
         wall_x_constant=wall_x.utils.constant.__file__,
         vap_serving=wall_x.model.vap.serving.__file__,
         vapv2_serving=wall_x.model.vapv2.serving.__file__,
         python=platform.python_version(),torch_version=torch.__version__,cuda=torch.version.cuda,
         packages={d.metadata['Name']:d.version for d in md.distributions()})
print(json.dumps(r,sort_keys=True))"""
    env = public_environ()
    env['PYTHONPATH'] = os.pathsep.join((root, xpl, repo))
    result = subprocess.run([str(python), '-I', '-c', code, root, xpl, repo], env=env, check=True,
                            capture_output=True, text=True)
    return json.loads(result.stdout)


def report(args):
    target = args.target.expanduser().absolute()
    python = target_python(target)
    if not python.is_file():
        raise FileNotFoundError(f'No public VAP environment at {target}; run install.sh install first')
    check_interpreter(python)
    data = module_report(python)
    if target.is_symlink():
        raise ValueError(f'Policy environment must not be a symlink: {target}')
    prefix = Path(data['python_prefix']).resolve()
    if prefix != target:
        raise ValueError(f'Python prefix is not the new environment: {prefix}')
    if Path(data['sys_executable']).absolute() != python.absolute():
        raise ValueError(f'Python executable differs from target: {data["sys_executable"]}')
    if Path(data['resolved_executable']) != python.resolve():
        raise ValueError(f'Resolved Python executable differs from target: {data["resolved_executable"]}')
    for link in (target / 'lib').rglob('*'):
        if link.is_symlink():
            resolved = str(link.resolve())
            if ('/site-packages/' in resolved or '/dist-packages/' in resolved) and not link.resolve().is_relative_to(target):
                raise ValueError(f'Environment links to external packages: {link} -> {resolved}')
    allowed = (target / 'lib', POLICY_DIR.parents[2])
    external = {}
    for name, path in data.items():
        if name in {'python_prefix', 'sys_executable', 'resolved_executable', 'sys_path', 'packages'}:
            continue
        if isinstance(path, str) and path.startswith('/'):
            source = Path(path).resolve()
            if not any(source.is_relative_to(parent) for parent in allowed):
                external[name] = path
    if external:
        raise ValueError(f'Modules resolve outside the new environment or adapter: {external}')
    expected = locked_versions()
    actual = {re.sub(r'[-_.]+', '-', name).lower(): version
              for name, version in data['packages'].items()}
    drift = {name: {'expected': version, 'actual': actual.get(name)}
             for name, version in expected.items() if actual.get(name) != version}
    if drift:
        raise ValueError(f'Installed packages differ from public lock: {drift}')
    if data['cuda'] != '13.0':
        raise ValueError(f'Expected the locked CUDA 13.0 runtime, got {data["cuda"]}')
    run([python, '-I', '-m', 'pip', 'check'], env=public_environ())
    data['status'] = 'imports_passed'
    data['environment'] = str(target)
    data['real_checkpoint_tested'] = False
    if args.gpu:
        code = ('import json,torch; assert torch.cuda.is_available(), "CUDA is not available"; '
                'x=torch.ones((16,16),device="cuda"); assert (x@x).sum().item()==4096; '
                'torch.cuda.synchronize(); print(json.dumps({"gpu":torch.cuda.get_device_name(0)}))')
        result = subprocess.run([str(python), '-I', '-c', code], env=public_environ(),
                                check=True, capture_output=True, text=True)
        data['gpu'] = json.loads(result.stdout)['gpu']
    if args.report:
        path = args.report.resolve()
        if path.exists() or path.is_symlink():
            raise FileExistsError(f'Refusing to overwrite report: {path}')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: data[key] for key in ('status', 'environment', 'python', 'torch_version', 'cuda')},
                     indent=2), flush=True)
    if args.gpu:
        print('GPU:', data['gpu'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('install', 'check'), nargs='?', default='install')
    parser.add_argument('target', nargs='?', type=Path, default=POLICY_DIR / '.venv')
    parser.add_argument('--python', help='External CPython 3.11 used to create the venv')
    parser.add_argument('--index-url', default='https://pypi.org/simple',
                        help='Public package mirror; all wheel hashes remain mandatory')
    parser.add_argument('--gpu', action='store_true', help='Also perform a small CUDA matrix multiplication')
    parser.add_argument('--report', type=Path, help='Write a new JSON verification report')
    args = parser.parse_args()
    if args.mode == 'install' and (args.gpu or args.report):
        parser.error('--gpu/--report are check options; run install.sh check after installation')
    if args.mode == 'install':
        install(args)
    else:
        report(args)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
