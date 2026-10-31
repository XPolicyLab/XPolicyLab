"""Confinement of the worker process that runs the model's cells.

Three layers:

* Landlock (Linux >= 5.13, unprivileged): after the worker has imported what it needs, it gives
  up file access outside a short list of read-only trees (the Python installation, the hexashell
  package, system libraries) and its own workspace (read-write), and - with Landlock ABI >= 4 -
  every TCP connect and bind.  This is the real boundary: the benchmark's files (layouts, task
  code) and the network are out of reach whatever the cell does.
* seccomp: no network sockets (on kernels whose Landlock cannot block TCP), no new programs or
  processes, no signals to or tracing of other processes.
* Python level: the cell's builtins carry an ``__import__`` that only admits a list of modules,
  and an ``open`` limited to the workspace.  These give clear error messages; they are not the
  boundary.
"""

from __future__ import annotations

import builtins
import ctypes
import ctypes.util
import os
import platform
from pathlib import Path

# -- Landlock ----------------------------------------------------------------------------------------
_SYSCALLS = {"x86_64": (444, 445, 446), "aarch64": (444, 445, 446)}
_CREATE_RULESET_VERSION = 1
_RULE_PATH_BENEATH = 1
_PR_SET_NO_NEW_PRIVS = 38

FS_EXECUTE = 1 << 0
FS_WRITE_FILE = 1 << 1
FS_READ_FILE = 1 << 2
FS_READ_DIR = 1 << 3
FS_REMOVE_DIR = 1 << 4
FS_REMOVE_FILE = 1 << 5
FS_MAKE_CHAR = 1 << 6
FS_MAKE_DIR = 1 << 7
FS_MAKE_REG = 1 << 8
FS_MAKE_SOCK = 1 << 9
FS_MAKE_FIFO = 1 << 10
FS_MAKE_BLOCK = 1 << 11
FS_MAKE_SYM = 1 << 12
FS_REFER = 1 << 13  # ABI 2
FS_TRUNCATE = 1 << 14  # ABI 3
NET_BIND_TCP = 1 << 0  # ABI 4
NET_CONNECT_TCP = 1 << 1

_FILE_ONLY = FS_EXECUTE | FS_WRITE_FILE | FS_READ_FILE | FS_TRUNCATE


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64), ("handled_access_net", ctypes.c_uint64)]


class _PathBeneath(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def landlock(read_only: list[str], read_write: list[str]) -> str:
    """Confine this process; returns a one-line description of what was applied."""
    calls = _SYSCALLS.get(platform.machine())
    if calls is None:
        return f"landlock: unsupported machine {platform.machine()}"
    create, add_rule, restrict = calls
    libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
    libc.syscall.restype = ctypes.c_long
    abi = libc.syscall(create, None, ctypes.c_size_t(0), ctypes.c_uint32(_CREATE_RULESET_VERSION))
    if abi < 1:
        return "landlock: not available on this kernel"
    fs = (FS_EXECUTE | FS_WRITE_FILE | FS_READ_FILE | FS_READ_DIR | FS_REMOVE_DIR | FS_REMOVE_FILE
          | FS_MAKE_CHAR | FS_MAKE_DIR | FS_MAKE_REG | FS_MAKE_SOCK | FS_MAKE_FIFO | FS_MAKE_BLOCK
          | FS_MAKE_SYM)
    if abi >= 2:
        fs |= FS_REFER
    if abi >= 3:
        fs |= FS_TRUNCATE
    net = (NET_BIND_TCP | NET_CONNECT_TCP) if abi >= 4 else 0
    attr = _RulesetAttr(fs, net)
    size = ctypes.sizeof(_RulesetAttr) if abi >= 4 else 8
    ruleset = libc.syscall(create, ctypes.byref(attr), ctypes.c_size_t(size), ctypes.c_uint32(0))
    if ruleset < 0:
        return f"landlock: create_ruleset failed (errno {ctypes.get_errno()})"
    read = FS_EXECUTE | FS_READ_FILE | FS_READ_DIR
    added = []
    try:
        for paths, access in ((read_only, read), (read_write, fs)):
            for path in paths:
                if not os.path.exists(path):
                    continue
                fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
                try:
                    allowed = access if os.path.isdir(path) else access & _FILE_ONLY
                    rule = _PathBeneath(allowed & fs, fd)
                    if libc.syscall(add_rule, ctypes.c_int(ruleset), ctypes.c_int(_RULE_PATH_BENEATH),
                                    ctypes.byref(rule), ctypes.c_uint32(0)) == 0:
                        added.append(path)
                finally:
                    os.close(fd)
        if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            return f"landlock: no_new_privs failed (errno {ctypes.get_errno()})"
        if libc.syscall(restrict, ctypes.c_int(ruleset), ctypes.c_uint32(0)) != 0:
            return f"landlock: restrict_self failed (errno {ctypes.get_errno()})"
    finally:
        os.close(ruleset)
    return f"landlock ABI {abi}: files limited to {len(added)} trees" + (", TCP blocked" if net else "")


# -- seccomp ---------------------------------------------------------------------------------------
# A second layer for what Landlock leaves open on older kernels (TCP needs ABI 4, Linux 6.7): the
# worker may not open network sockets, start programs or processes, signal or trace other
# processes.  Threads stay possible (numpy and OpenCV use them).  x86_64 only.
_AUDIT_ARCH_X86_64 = 0xC000003E
_X32_BIT = 0x40000000
_SECCOMP_RET_ALLOW = 0x7FFF0000
_SECCOMP_RET_ERRNO = 0x00050000
_EPERM, _ENOSYS = 1, 38
_CLONE_THREAD = 0x00010000
_AF_UNIX = 1
_NR = {"socket": 41, "clone": 56, "fork": 57, "vfork": 58, "execve": 59, "kill": 62, "ptrace": 101,
       "tkill": 200, "tgkill": 234, "process_vm_readv": 310, "process_vm_writev": 311, "execveat": 322,
       "pidfd_send_signal": 424, "io_uring_setup": 425, "io_uring_enter": 426, "io_uring_register": 427,
       "pidfd_open": 434, "clone3": 435}
_DENIED = ("fork", "vfork", "execve", "execveat", "kill", "tkill", "tgkill", "ptrace", "process_vm_readv",
           "process_vm_writev", "pidfd_send_signal", "pidfd_open", "io_uring_setup", "io_uring_enter",
           "io_uring_register")


class _SockFilter(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint16), ("jt", ctypes.c_uint8), ("jf", ctypes.c_uint8), ("k", ctypes.c_uint32)]


class _SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_uint16), ("filter", ctypes.POINTER(_SockFilter))]


def _seccomp_program() -> list[tuple[int, int, int, int]]:
    load, jeq, jge, jset, ret = 0x20, 0x15, 0x35, 0x45, 0x06
    deny = _SECCOMP_RET_ERRNO | _EPERM
    prog = [(load, 0, 0, 4),  # A = arch
            (jeq, 1, 0, _AUDIT_ARCH_X86_64),
            (ret, 0, 0, deny),  # another ABI (i386, ...): nothing
            (load, 0, 0, 0),  # A = syscall number
            (jge, 0, 1, _X32_BIT),
            (ret, 0, 0, deny)]  # x32 calls
    for name in _DENIED:
        prog += [(jeq, 0, 1, _NR[name]), (ret, 0, 0, deny)]
    # clone3 hides its flags in memory: answer "not implemented" so the C library falls back to clone
    prog += [(jeq, 0, 1, _NR["clone3"]), (ret, 0, 0, _SECCOMP_RET_ERRNO | _ENOSYS)]
    # clone: threads only
    prog += [(jeq, 0, 3, _NR["clone"]), (load, 0, 0, 16), (jset, 1, 0, _CLONE_THREAD), (ret, 0, 0, deny)]
    # the clone check jumps here (allow) when CLONE_THREAD is set; reload the number for the next test
    prog += [(load, 0, 0, 0)]
    # socket: Unix sockets only
    prog += [(jeq, 0, 3, _NR["socket"]), (load, 0, 0, 16), (jeq, 1, 0, _AF_UNIX), (ret, 0, 0, deny)]
    prog += [(ret, 0, 0, _SECCOMP_RET_ALLOW)]
    return prog


def seccomp_lockdown() -> str:
    """Install the seccomp filter on this process (and everything it would start)."""
    if platform.machine() != "x86_64":
        return f"seccomp: skipped on {platform.machine()}"
    libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
    prog = _seccomp_program()
    filters = (_SockFilter * len(prog))(*[_SockFilter(*ins) for ins in prog])
    fprog = _SockFprog(len(prog), filters)
    if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        return f"seccomp: no_new_privs failed (errno {ctypes.get_errno()})"
    if libc.prctl(22, 2, ctypes.byref(fprog), 0, 0) != 0:  # PR_SET_SECCOMP, SECCOMP_MODE_FILTER
        return f"seccomp: filter refused (errno {ctypes.get_errno()})"
    return f"seccomp: {len(prog)} instructions (no network, programs, processes or signals)"


# -- Python-level guards ---------------------------------------------------------------------------
ALLOWED_MODULES = frozenset({
    "numpy", "math", "cmath", "json", "re", "itertools", "functools", "collections", "statistics",
    "random", "dataclasses", "typing", "copy", "time", "heapq", "bisect", "operator", "string",
    "textwrap", "fractions", "decimal", "enum", "pprint", "cv2", "colorsys", "abc", "numbers",
    "contextlib", "datetime",
})


class SandboxError(ImportError):
    pass


def guarded_import(real_import):
    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        top = name.split(".")[0]
        if level != 0 or top not in ALLOWED_MODULES:
            raise SandboxError(
                f"import of '{name}' is not available in cells; available: "
                + ", ".join(sorted(ALLOWED_MODULES)))
        return real_import(name, globals, locals, fromlist, level)

    return _import


def guarded_open(workspace: Path, real_open):
    root = workspace.resolve()

    def _open(file, mode="r", *args, **kwargs):
        if isinstance(file, int):
            raise PermissionError("cells may open files only by name, inside the workspace")
        path = (root / os.fspath(file)).resolve()
        if path != root and root not in path.parents:
            raise PermissionError(f"cells may only open files inside the workspace ({root})")
        return real_open(path, mode, *args, **kwargs)

    return _open


def cell_builtins(workspace: Path) -> dict:
    out = dict(vars(builtins))
    out["__import__"] = guarded_import(builtins.__import__)
    out["open"] = guarded_open(workspace, builtins.open)
    for name in ("exit", "quit", "help", "breakpoint", "input", "copyright", "credits", "license"):
        out.pop(name, None)
    return out
