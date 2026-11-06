"""OS-level isolation ("layer 2") for the model-code worker.

The in-process audit hook in guard.py is a soft limit: native library calls
(cv2.imread, torch.from_file, a cffi/ctypes open+read, _posixsubprocess.fork_exec)
reach the kernel without a Python audit event and so slip past it. This module
asks the Linux kernel to enforce the same rules with Landlock, applied in the
worker's serve() before the Python guard and before any model code runs. A
Landlock restriction is inherited across fork and execve, so ffmpeg/ffprobe and
any other child of the worker inherit it too.

What the kernel enforces once apply() succeeds:
  files   read only the read roots (which include the system and interpreter
          paths needed to run, with /etc cut down to ETC_ENTRIES) and the write
          roots; write only the write roots (plus the knowledge root when the
          session's knowledge is writable); execute only the interpreter and the
          allowed executables.
  network TCP connect only to the robot service's port(s); no TCP bind.

Limits the kernel does not cover here:
  - Landlock has no UDP rule and there is no network namespace, so native UDP,
    including to the outside, is stopped only by the Python guard's hook.
  - Landlock network rules match by port only, not by address, so native code
    could reach the robot service's port number on another host.
  - stat() is not governed: the existence, size and times of files outside the
    roots stay visible, though not their contents.

On a kernel without Landlock, or on a non-Linux host, apply() does not
restrict anything. The caller decides whether that is fatal: the worker runs on the
Python guard alone where the kernel has no Landlock (as the README
says), and refuses to start where Landlock is there but cannot be applied, unless a
test-only override is set.
"""
from __future__ import annotations

import ctypes
import os
import site
import stat
import sys
import sysconfig

# syscall numbers (x86_64 and arm64 both use these)
SYS_landlock_create_ruleset = 444
SYS_landlock_add_rule = 445
SYS_landlock_restrict_self = 446
PR_SET_NO_NEW_PRIVS = 38

LANDLOCK_CREATE_RULESET_VERSION = 1 << 0
LANDLOCK_RULE_PATH_BENEATH = 1
LANDLOCK_RULE_NET_PORT = 2

# Filesystem access rights (bit positions; the ABI that first offered each).
FS = {
    'EXECUTE': 1 << 0, 'WRITE_FILE': 1 << 1, 'READ_FILE': 1 << 2, 'READ_DIR': 1 << 3,
    'REMOVE_DIR': 1 << 4, 'REMOVE_FILE': 1 << 5, 'MAKE_CHAR': 1 << 6, 'MAKE_DIR': 1 << 7,
    'MAKE_REG': 1 << 8, 'MAKE_SOCK': 1 << 9, 'MAKE_FIFO': 1 << 10, 'MAKE_BLOCK': 1 << 11,
    'MAKE_SYM': 1 << 12, 'REFER': 1 << 13, 'TRUNCATE': 1 << 14,
}
# Network access rights (ABI 4).
NET_BIND_TCP = 1 << 0
NET_CONNECT_TCP = 1 << 1

# Rights that apply to a regular file (a rule on a file fd must carry only these).
FILE_RIGHTS = FS['EXECUTE'] | FS['WRITE_FILE'] | FS['READ_FILE'] | FS['TRUNCATE']
READ_ACCESS = FS['READ_FILE'] | FS['READ_DIR']
EXEC_DIR_ACCESS = READ_ACCESS | FS['EXECUTE']
EXEC_FILE_ACCESS = FS['READ_FILE'] | FS['EXECUTE']
# A writable directory: read, create, remove, rename within the tree and truncate — but never execute,
# so model code cannot drop a binary in its workspace and run it.
WRITE_ACCESS = (FS['READ_FILE'] | FS['READ_DIR'] | FS['WRITE_FILE'] | FS['REMOVE_DIR'] | FS['REMOVE_FILE'] |
                FS['MAKE_DIR'] | FS['MAKE_REG'] | FS['MAKE_SOCK'] | FS['MAKE_FIFO'] | FS['MAKE_SYM'] |
                FS['REFER'] | FS['TRUNCATE'])


class SandboxError(RuntimeError):
    """Landlock could not be applied on a host where it was required."""


class _RulesetAttr(ctypes.Structure):
    _fields_ = [('handled_access_fs', ctypes.c_uint64), ('handled_access_net', ctypes.c_uint64)]


class _PathBeneathAttr(ctypes.Structure):
    _pack_ = 1  # the kernel struct is __attribute__((packed))
    _fields_ = [('allowed_access', ctypes.c_uint64), ('parent_fd', ctypes.c_int32)]


class _NetPortAttr(ctypes.Structure):
    _fields_ = [('allowed_access', ctypes.c_uint64), ('port', ctypes.c_uint64)]


def _libc():
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    return libc


def abi_version():
    """The kernel's Landlock ABI, 0 when Landlock is absent or disabled."""
    if not sys.platform.startswith('linux'):
        return 0
    try:
        libc = _libc()
        result = libc.syscall(SYS_landlock_create_ruleset, None, ctypes.c_size_t(0),
                              ctypes.c_uint32(LANDLOCK_CREATE_RULESET_VERSION))
    except (OSError, AttributeError):
        return 0
    return int(result) if result > 0 else 0


ISOLATION_FULL = ('python audit hook; Linux Landlock in the worker (files, TCP, exec; not MPTCP, UDP or Unix '
                  'sockets); see worker.stderr "v2 sandbox:"')
ISOLATION_FILES = ('python audit hook; Linux Landlock in the worker (files, exec; kernel ABI < 4, so no TCP rules; '
                   'not UDP or Unix sockets); see worker.stderr "v2 sandbox:"')
ISOLATION_NONE = ('python audit hook only: this kernel has no Landlock, so native-library file, network and exec '
                  'calls are not confined by the kernel; see worker.stderr "v2 sandbox:"')


def isolation_summary(abi=None):
    """The worker isolation this host gives, for run.json and the evaluator console.

    A worker runs on the Python audit hook alone where the kernel has no Landlock (see worker.serve).
    """
    abi = abi_version() if abi is None else abi
    return ISOLATION_FULL if abi >= 4 else ISOLATION_FILES if abi >= 1 else ISOLATION_NONE


def _handled(abi):
    """The access rights to govern, bounded to what this ABI supports."""
    fs = (FS['EXECUTE'] | FS['WRITE_FILE'] | FS['READ_FILE'] | FS['READ_DIR'] | FS['REMOVE_DIR'] |
          FS['REMOVE_FILE'] | FS['MAKE_CHAR'] | FS['MAKE_DIR'] | FS['MAKE_REG'] | FS['MAKE_SOCK'] |
          FS['MAKE_FIFO'] | FS['MAKE_BLOCK'] | FS['MAKE_SYM'])
    if abi >= 2:
        fs |= FS['REFER']
    if abi >= 3:
        fs |= FS['TRUNCATE']
    # IOCTL_DEV (ABI 5) is left out on purpose: governing it would block the ioctls CUDA makes on
    # /dev/nvidia*; ioctl is not a file-content leak we need to stop here.
    net = (NET_BIND_TCP | NET_CONNECT_TCP) if abi >= 4 else 0
    return fs, net


def _library_dirs():
    """Directories holding shared objects the interpreter and its packages load (read, for dlopen/mmap)."""
    dirs = set()
    for prefix in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix):
        dirs.add(os.path.join(prefix, 'lib'))
        dirs.add(os.path.join(prefix, 'lib64'))
    for name in ('stdlib', 'platstdlib', 'purelib', 'platlib'):
        path = sysconfig.get_paths().get(name)
        if path:
            dirs.add(path)
    try:
        dirs.update(site.getsitepackages())
    except AttributeError:
        pass
    # Not the user site-packages: the worker's HOME is inside its writable workspace, and nothing writable
    # may be executable.
    dirs.update(('/usr/lib', '/usr/lib64', '/usr/local/lib', '/lib', '/lib64'))
    return {d for d in dirs if d}


def _executable_files(executables, write_reals):
    """Absolute paths of the interpreter and the allowed helper executables (ffmpeg, ffprobe).

    An executable that resolves under a write root is dropped: a model-writable file must never be
    executable (an empty PATH entry could otherwise make shutil.which find one in the workspace).
    """
    import shutil
    files = {sys.executable, os.path.realpath(sys.executable)}
    for name in executables or ():
        name = os.path.basename(str(name))
        found = shutil.which(name)
        if found:
            files.add(found)
            files.add(os.path.realpath(found))
    return {f for f in files if f and not _under(os.path.realpath(f), write_reals)}


def _under(path, roots):
    return any(path == root or path.startswith(root.rstrip(os.sep) + os.sep) for root in roots)


# The /etc entries the interpreter, numpy/PIL/cv2/torch, CUDA and ffmpeg read (found by tracing the worker
# tests and a session doing ordinary image work): the dynamic linker cache, time zone, name service and
# MIME tables, account names, libgcrypt (ffmpeg) and the NVIDIA profiles. Nothing else under /etc is readable.
ETC_ENTRIES = ('ld.so.cache', 'ld.so.conf', 'ld.so.conf.d', 'localtime', 'timezone', 'nsswitch.conf', 'hosts',
               'host.conf', 'resolv.conf', 'gai.conf', 'passwd', 'group', 'mime.types', 'gcrypt', 'nvidia',
               'fonts', 'os-release', 'lsb-release')


def _etc_entries():
    return [os.path.join('/etc', name) for name in ETC_ENTRIES]


def _plan(read_roots, write_roots, executables, write_files=()):
    """Merge every source into one access mask per path (OR-ing where a path has more than one role)."""
    plan = {}

    def grant(path, access):
        if path:
            plan[path] = plan.get(path, 0) | access

    for root in read_roots:
        for path in _etc_entries() if os.path.realpath(str(root)) == '/etc' else [str(root)]:
            grant(path, READ_ACCESS)
    for root in write_roots:
        grant(str(root), WRITE_ACCESS)
    # Single existing files that may be appended to (the guard's own log): no right to create anything beside them.
    for path in write_files:
        grant(str(path), FS['READ_FILE'] | FS['WRITE_FILE'])
    # Library directories get EXECUTE: execve of a dynamically linked program (python, ffmpeg) needs it on
    # the dynamic loader (ld-linux) that lives here, and the loader can map a shared object even where
    # userspace mmap(PROT_EXEC) alone would not need it. A consequence is that any ELF under these dirs, or
    # any readable file run through the loader, can be execve'd — but every such process inherits this same
    # sandbox (no new file or network reach), so execve is a hardening layer, not the data/network boundary.
    for directory in _library_dirs():
        grant(directory, EXEC_DIR_ACCESS)
    # /dev holds /dev/null, /dev/urandom and /dev/shm (torch shared memory): read and write, never execute.
    grant('/dev', WRITE_ACCESS)
    # This process's own /proc entries may be written as well as read: the CUDA driver names its threads through
    # /proc/self/task/<tid>/comm and fails to initialise (error 304) when that write is refused. Other
    # processes' entries (the parent's environ) stay unreadable.
    grant(f'/proc/{os.getpid()}', READ_ACCESS | FS['WRITE_FILE'] | FS['TRUNCATE'])
    write_reals = [os.path.realpath(str(root)) for root in write_roots]
    for path in _executable_files(executables, write_reals):
        grant(path, EXEC_FILE_ACCESS)
    return plan


def _refuse_symlinked(paths):
    """A write root or writable file that is itself a symlink is refused (it fails closed).

    Writable roots and files sit inside the run or knowledge tree. If model code from an earlier worker
    swapped one for a symlink before this worker started, following it would grant write elsewhere. System
    read roots such as /lib are legitimately symlinks and are not checked.
    """
    for path in paths:
        path = str(path)
        if os.path.lexists(path) and os.path.islink(path):
            raise OSError(f'refusing to sandbox: writable path {path} is a symlink')


def apply(read_roots, write_roots, executables=(), endpoints=(), write_files=(), require=True):
    """Restrict this thread (and its future children) with Landlock.

    read_roots/write_roots are the guard's roots already merged with the system read roots.
    endpoints are the (host, port) pairs of the robot service; only their TCP ports may be connected.
    write_files are existing single files outside the write roots that may be written (the guard's log).
    When require is True, a Linux host that cannot apply Landlock raises SandboxError; otherwise, and
    on any non-Linux host, apply() returns a status dict with applied=False and restricts nothing.
    """
    status = {'applied': False, 'abi': 0, 'fs': False, 'net': False, 'reason': ''}
    if not sys.platform.startswith('linux'):
        status['reason'] = f'Landlock is unavailable on {sys.platform}'
        return status
    abi = abi_version()
    status['abi'] = abi
    if abi < 1:
        status['reason'] = 'the kernel has no Landlock support'
        if require:
            raise SandboxError(status['reason'])
        return status
    # restrict_self binds only the calling thread and the threads it starts afterwards, so a thread that
    # already exists (e.g. a BLAS pool from importing numpy) would stay unsandboxed. Apply before any such
    # import and refuse if the worker is somehow already multi-threaded.
    threads = len(os.listdir('/proc/self/task'))
    if threads != 1:
        status['reason'] = f'the worker already has {threads} threads; Landlock must be applied single-threaded'
        if require:
            raise SandboxError(status['reason'])
        return status
    try:
        _refuse_symlinked(list(write_roots) + list(write_files))
        _restrict(abi, read_roots, write_roots, executables, endpoints, write_files)
    except (OSError, ValueError, TypeError) as exc:
        status['reason'] = f'landlock could not be applied: {exc}'
        if require:
            raise SandboxError(status['reason']) from exc
        return status
    status['applied'] = True
    status['net'] = abi >= 4
    status['fs'] = True
    if abi < 4:
        status['reason'] = 'kernel ABI < 4: network rules not enforced (files only)'
    return status


def _restrict(abi, read_roots, write_roots, executables, endpoints, write_files):
    libc = _libc()
    handled_fs, handled_net = _handled(abi)
    attr = _RulesetAttr(handled_access_fs=handled_fs, handled_access_net=handled_net)
    fd = libc.syscall(SYS_landlock_create_ruleset, ctypes.byref(attr), ctypes.c_size_t(ctypes.sizeof(attr)),
                      ctypes.c_uint32(0))
    if fd < 0:
        raise OSError(ctypes.get_errno(), 'landlock_create_ruleset')
    try:
        for path, access in _plan(read_roots, write_roots, executables, write_files).items():
            _add_path(libc, fd, path, access, handled_fs)
        if handled_net:
            for _host, port in endpoints:
                if port is not None:
                    _add_net_port(libc, fd, int(port), NET_CONNECT_TCP)
        _no_new_privs(libc)
        if libc.syscall(SYS_landlock_restrict_self, ctypes.c_int(fd), ctypes.c_uint32(0)) != 0:
            raise OSError(ctypes.get_errno(), 'landlock_restrict_self')
    finally:
        os.close(fd)


def _no_new_privs(libc):
    # prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0): required before restrict_self without CAP_SYS_ADMIN, and
    # inherited across execve so helper processes cannot regain privileges.
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(ctypes.c_int(PR_SET_NO_NEW_PRIVS), ctypes.c_ulong(1),
                  ctypes.c_ulong(0), ctypes.c_ulong(0), ctypes.c_ulong(0)) != 0:
        raise OSError(ctypes.get_errno(), 'prctl(PR_SET_NO_NEW_PRIVS)')


def _add_path(libc, fd, path, access, handled_fs):
    try:
        path_fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
    except OSError:
        return  # a path that does not exist on this host (e.g. /lib64 on some distros) is simply skipped
    try:
        is_dir = stat.S_ISDIR(os.fstat(path_fd).st_mode)  # the opened fd, not a second name lookup
        allowed = access & handled_fs
        if not is_dir:
            allowed &= FILE_RIGHTS  # dir-only rights are invalid on a file fd
        if not allowed:
            return
        attr = _PathBeneathAttr(allowed_access=allowed, parent_fd=path_fd)
        if libc.syscall(SYS_landlock_add_rule, ctypes.c_int(fd), ctypes.c_int(LANDLOCK_RULE_PATH_BENEATH),
                        ctypes.byref(attr), ctypes.c_uint32(0)) != 0:
            raise OSError(ctypes.get_errno(), f'landlock_add_rule path {path}')
    finally:
        os.close(path_fd)


def _add_net_port(libc, fd, port, access):
    attr = _NetPortAttr(allowed_access=access, port=port)
    if libc.syscall(SYS_landlock_add_rule, ctypes.c_int(fd), ctypes.c_int(LANDLOCK_RULE_NET_PORT),
                    ctypes.byref(attr), ctypes.c_uint32(0)) != 0:
        raise OSError(ctypes.get_errno(), f'landlock_add_rule net {port}')
