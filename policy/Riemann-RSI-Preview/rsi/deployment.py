"""Machine-local locations for the single official policy implementation."""
from functools import lru_cache
import json
import os
import shutil
import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def absolute(value):
    path = Path(value).expanduser()
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('Deployment locations must be absolute paths without parent traversal')
    return path


@lru_cache(maxsize=16)
def _read(path, modified, size):
    data = json.loads(Path(path).read_text())
    allowed = {'official_runtime', 'wam_checkpoint', 'encoder_path', 'policy_python',
               'resources_base', 'resource_roots', 'resource_aliases', 'slurm_bin', 'native_library_dirs', 'cluster'}
    if not isinstance(data, dict) or set(data) - allowed:
        raise ValueError('Unknown local deployment fields')
    runtime = data.get('official_runtime', {})
    if not isinstance(runtime, dict) or set(runtime) - {'official_root', 'python'}:
        raise ValueError('Local runtime binding may only relocate SDK and Python')
    for value in runtime.values(): absolute(value)
    native = data.get('native_library_dirs', [])
    if not isinstance(native, list): raise ValueError('Native library directories must be a location list')
    for value in native: absolute(value)
    for key in ('wam_checkpoint', 'encoder_path', 'policy_python', 'resources_base', 'slurm_bin'):
        if key in data: absolute(data[key])
    if data.get('cluster', 'local') not in ('local', 'nx'):
        raise ValueError('Unknown GPU lease namespace')
    roots = data.get('resource_roots', {})
    if not isinstance(roots, dict): raise ValueError('Resource roots must be a location mapping')
    for value in roots.values(): absolute(value)
    aliases = data.get('resource_aliases', {})
    if not isinstance(aliases, dict): raise ValueError('Resource aliases must be a location mapping')
    for values in aliases.values():
        if not isinstance(values, list): raise ValueError('Resource aliases must be path lists')
        for value in values: absolute(value)
    return data


def local(root=ROOT):
    path = Path(root) / 'deploy.local.json'
    if not path.is_file(): return {}
    stat = path.stat()
    return _read(str(path), stat.st_mtime_ns, stat.st_size)


def discover_sdk(root=ROOT):
    """Inspect only the policy's ancestors and the already imported SDK package."""
    candidates = list(Path(root).resolve().parents)
    package = sys.modules.get('XPolicyLab')
    filename = getattr(package, '__file__', None)
    if filename: candidates.extend(Path(filename).resolve().parents)
    for parent in dict.fromkeys(candidates):
        if (parent / 'src/eval_client/main.py').is_file() and (parent / 'env_cfg').is_dir():
            return parent
    # Source-only CI needs a deterministic location, not an installed simulator.
    # Runtime admission verifies existence and the pinned external identity.
    return Path(root).resolve().parent / 'RoboDojo'


def locations(root=ROOT, *, environment=None):
    environment = os.environ if environment is None else environment
    data = local(root); runtime = data.get('official_runtime', {})
    sdk = absolute(environment.get('RIEMANN_OFFICIAL_ROOT') or runtime.get('official_root') or discover_sdk(root)).resolve()
    python = absolute(environment.get('RIEMANN_EVAL_PYTHON') or runtime.get('python') or sys.executable)
    resources = absolute(environment.get('RIEMANN_RESOURCES_BASE') or data.get('resources_base')
                         or Path.home() / '.cache/riemann-rsi/resources')
    policy_python = absolute(environment.get('RIEMANN_POLICY_PYTHON') or data.get('policy_python') or python)
    encoder = absolute(environment.get('RIEMANN_WAM_ENCODER') or data.get('encoder_path') or resources / 'encoder')
    return dict(official_root=str(sdk), python=str(python), eval_prefix=str(python.parent.parent.resolve()),
                policy_python=str(policy_python), policy_prefix=str(policy_python.parent.parent.resolve()),
                encoder_path=str(encoder), resources_base=str(resources), cluster=data.get('cluster', 'local'))


def model_options(root=ROOT, *, environment=None):
    environment = os.environ if environment is None else environment
    data = local(root)
    options = {key:data[key] for key in ('wam_checkpoint', 'encoder_path') if key in data}
    for key, variable in (('wam_checkpoint','RIEMANN_WAM_CHECKPOINT'), ('encoder_path','RIEMANN_WAM_ENCODER')):
        if environment.get(variable): options[key] = str(absolute(environment[variable]))
    return options


def slurm_srun(root=ROOT):
    directory = os.environ.get('RIEMANN_SLURM_BIN') or local(root).get('slurm_bin')
    executable = str(absolute(directory) / 'srun') if directory else shutil.which('srun')
    if not executable or not os.access(executable, os.X_OK):
        raise ValueError('Slurm srun is missing; set PATH or RIEMANN_SLURM_BIN')
    return executable


def native_library_dirs(root=ROOT, *, environment=None):
    environment = os.environ if environment is None else environment
    explicit = environment.get('RIEMANN_NATIVE_LIBRARY_DIRS')
    values = explicit.split(os.pathsep) if explicit else local(root).get('native_library_dirs', [])
    return list(dict.fromkeys(str(absolute(value)) for value in values))


def apply_native_libraries(environment, root=ROOT):
    """Bind an existing runtime's libraries in child processes only."""
    values = native_library_dirs(root, environment=environment)
    if values:
        for value in values:
            if not Path(value).is_dir(): raise ValueError('Missing declared native library directory: ' + value)
        existing = environment.get('LD_LIBRARY_PATH', '').split(os.pathsep)
        environment['LD_LIBRARY_PATH'] = os.pathsep.join(dict.fromkeys([*values, *(p for p in existing if p)]))
    return values


def interpreter_python(value):
    """Resolve the standard path/directory/Conda argument without loading models."""
    path = Path(value).expanduser()
    if path.is_file(): return str(path.absolute())
    if (path / 'bin/python').is_file(): return str((path / 'bin/python').absolute())
    code = 'import sys; print(sys.executable)'
    output = subprocess.check_output(['conda','run','--no-capture-output','-n',str(value),
                                      'python','-I','-S','-c',code],text=True,timeout=30)
    return str(absolute(output.strip()))


def evaluator_path(environment, executable, *, require_ffmpeg=True):
    """Give both client entry paths the admitted evaluator's executables."""
    prefix = str(Path(executable).parent)
    parts = [prefix, *environment.get('PATH', '').split(os.pathsep)]
    path = os.pathsep.join(dict.fromkeys(value for value in parts if value))
    if require_ffmpeg and shutil.which('ffmpeg', path=path) is None:
        raise RuntimeError('The declared evaluator environment must provide ffmpeg before GPU startup')
    environment['PATH'] = path
    return path
