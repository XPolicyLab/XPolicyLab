"""Standard physical GPU leases and cleanup of owned process groups."""
from contextlib import contextmanager, ExitStack
import csv
import fcntl
import io
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def cuda_visibility(gpus, inventory=None, *, query=subprocess.check_output):
    """Map explicit physical indices to UUIDs before CUDA or Kit imports.

    The same mapping serves the owned launcher and standard split scripts.
    This only resolves identity; the caller retains the GPU lifetime leases.
    """
    if isinstance(gpus, str):
        values = gpus.split(',')
        if not values or any(not value.isdecimal() for value in values):
            raise ValueError('Supply explicit comma-separated physical GPU numbers')
        gpus = [int(value) for value in values]
    else:
        gpus = list(gpus)
    if (not gpus or any(type(gpu) is not int or gpu < 0 for gpu in gpus)
            or len(set(gpus)) != len(gpus)):
        raise ValueError('Distinct physical GPU indices are required')
    if inventory is None:
        output = query(['nvidia-smi', '--query-gpu=index,uuid',
                        '--format=csv,noheader,nounits'], text=True, timeout=30)
        inventory = []
        for row in csv.reader(io.StringIO(output)):
            if len(row) != 2:
                raise ValueError('Physical GPU query identity mismatch')
            index, identity = [value.strip() for value in row]
            if not index.isdecimal():
                raise ValueError('Physical GPU query identity mismatch')
            inventory.append({'index': int(index), 'uuid': identity})
    identities = {}
    for row in inventory:
        index, identity = row['index'], row['uuid']
        if (type(index) is not int or index < 0 or index in identities
                or not isinstance(identity, str) or not identity.startswith('GPU-')
                or ',' in identity or identity in identities.values()):
            raise ValueError('Physical GPU query identity mismatch')
        identities[index] = identity
    if any(gpu not in identities for gpu in gpus):
        raise ValueError('Requested physical GPU is unavailable')
    return ','.join(identities[gpu] for gpu in gpus)


def configure_split_devices(value, *, renderer=False, environment=None):
    """Bind a split entry's devices without changing the shared runtime."""
    environment = os.environ if environment is None else environment
    visible = cuda_visibility(value)
    if renderer and ',' in visible:
        raise ValueError('One simulator GPU is required per episode')
    environment['CUDA_VISIBLE_DEVICES'] = visible
    if renderer:
        environment['RIEMANN_PHYSICAL_ENV_GPU'] = str(int(value))
    from rsi.deployment import apply_native_libraries
    apply_native_libraries(environment)
    if renderer and environment.get('EVAL_ENV_TYPE') != 'debug':
        from rsi.graphics_runtime import configure
        configure(environment, visible)
    return visible


def shared_lock_root(root=ROOT):
    from rsi.rule_configuration import deployment_settings
    deployment = deployment_settings(root=root)
    directory = Path.home() / '.cache/robodojo-shared-gpu-locks'
    return directory / socket.gethostname() if deployment.get('cluster') == 'nx' else directory


def idle_inventory(gpu):
    output = subprocess.check_output(['nvidia-smi', f'--id={gpu}',
        '--query-gpu=index,uuid,memory.used,memory.free', '--format=csv,noheader,nounits'],
        text=True, timeout=30)
    rows = list(csv.reader(io.StringIO(output)))
    if len(rows) != 1 or len(rows[0]) != 4:
        raise RuntimeError('Physical GPU query identity mismatch')
    index, identity, used, free = [value.strip() for value in rows[0]]
    if int(index) != gpu or not identity.startswith('GPU-') or min(int(used), int(free)) < 0:
        raise RuntimeError('Physical GPU query identity mismatch')
    applications = subprocess.check_output(['nvidia-smi', f'--id={gpu}',
        '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True, timeout=30).strip()
    if int(used) >= 512 or applications:
        raise RuntimeError('GPU occupied; no workload stopped')
    return {'index': gpu, 'uuid': identity, 'memory_used_mib': int(used), 'memory_free_mib': int(free)}


@contextmanager
def gpu_leases(gpus, directory=None, inventory=idle_inventory, lease_descriptors=None):
    selected = sorted(set(gpus))
    if not selected or any(type(gpu) is not int or gpu < 0 for gpu in selected):
        raise ValueError('Distinct physical GPU indices are required')
    directory = shared_lock_root() if directory is None else Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for gpu in selected:
            descriptor = os.open(directory / f'gpu{gpu}.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            stream = stack.enter_context(os.fdopen(descriptor, 'a'))
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f'GPU {gpu} already leased; no workload stopped') from error
            if lease_descriptors is not None:
                lease_descriptors.append(descriptor)
        yield [inventory(gpu) for gpu in selected]


def _process_identity(pid):
    path = Path('/proc') / str(pid)
    try:
        fields = (path / 'stat').read_text().rsplit(') ', 1)[1].split()
        return dict(pid=pid, uid=path.stat().st_uid, state=fields[0],
                    pgid=int(fields[2]), sid=int(fields[3]), start=fields[19])
    except FileNotFoundError:
        return None


def remember_owned_group(process):
    """Capture the newly created session before poll() can reap its leader."""
    identity = _process_identity(process.pid)
    if (identity is None or identity['uid'] != os.getuid()
            or identity['pgid'] != process.pid or identity['sid'] != process.pid):
        raise RuntimeError('New child is not the expected owned session leader')
    process._riemann_owned_group = identity


def _live_owned_group(identity):
    # Exact group lookup, no filesystem walk or name-based process signalling.
    result = subprocess.run(['pgrep', '-g', str(identity['pgid'])],
                            text=True, capture_output=True, check=False)
    if result.returncode not in (0, 1):
        raise RuntimeError('Cannot verify owned process group')
    live = []
    for word in result.stdout.split():
        member = _process_identity(int(word))
        if member is None:
            continue
        if (member['uid'] != identity['uid'] or member['sid'] != identity['sid']
                or member['pgid'] != identity['pgid']
                or member['pid'] == identity['pid'] and member['start'] != identity['start']):
            raise RuntimeError('Owned process group identity changed; no signal permitted')
        if member['state'] != 'Z':
            live.append(member['pid'])
    return live


def _stop_tracked_group(process, identity):
    # A crashed/reaped leader does not imply its workers have released the GPU
    # descriptors. Retain the parent's lease while stopping the verified group.
    for sig, grace in ((signal.SIGTERM, 20), (signal.SIGKILL, 10)):
        if not _live_owned_group(identity):
            break
        try:
            os.killpg(identity['pgid'], sig)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + grace
        while _live_owned_group(identity) and time.monotonic() < deadline:
            time.sleep(.05)
    if _live_owned_group(identity):
        raise RuntimeError('Owned workers remain alive; cleanup has not completed')
    process.wait(timeout=10)


def stop_owned(process):
    if process is not None and hasattr(process, '_riemann_owned_group'):
        return _stop_tracked_group(process, process._riemann_owned_group)
    if process is None or process.poll() is not None:
        return
    try:
        if os.getpgid(process.pid) != process.pid:
            raise RuntimeError('Refusing to signal a process outside its owned session')
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    except ProcessLookupError:
        process.wait(timeout=10)
