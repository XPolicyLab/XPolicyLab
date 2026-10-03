"""Trusted Linux launcher. Re-exec Python only after jail and syscall setup."""

import ctypes
import ctypes.util
import errno
import os
import resource
import signal
import sys
import termios


def enter(root, executable, cpu_seconds, memory_bytes, output_bytes):
    if sys.platform != "linux" or os.geteuid() != 0:
        raise RuntimeError("This isolation backend requires Linux chroot privilege")
    parent = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    seccomp = ctypes.CDLL(ctypes.util.find_library("seccomp"), use_errno=True)
    seccomp.seccomp_init.argtypes = [ctypes.c_uint32]
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    seccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int
    seccomp.seccomp_rule_add.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    seccomp.seccomp_load.argtypes = [ctypes.c_void_p]
    context = seccomp.seccomp_init(0x50000 | errno.EPERM)
    if not context:
        raise RuntimeError("Cannot initialize syscall filter")
    allowed = """read write readv writev close close_range lseek pread64 pwrite64
        open openat access faccessat faccessat2 stat lstat fstat newfstatat statx
        readlink readlinkat getdents64 fcntl dup dup2 dup3
        brk mmap mprotect munmap mremap madvise
        rt_sigaction rt_sigprocmask rt_sigreturn sigaltstack
        futex set_tid_address set_robust_list rseq arch_prctl
        clock_gettime clock_getres gettimeofday time nanosleep clock_nanosleep
        getpid getppid gettid getuid geteuid getgid getegid getgroups
        uname getrandom prlimit64 getrlimit getrusage sched_getaffinity sched_yield
        getcwd chdir fchdir umask fsync fdatasync ftruncate
        execve exit exit_group""".split()
    for name in allowed:
        syscall = seccomp.seccomp_syscall_resolve_name(name.encode())
        if syscall >= 0 and seccomp.seccomp_rule_add(context, 0x7FFF0000, syscall, 0):
            raise RuntimeError("Cannot configure syscall filter: " + name)

    # CPython's script fopen uses FIOCLEX. Do not expose arbitrary device ioctls.
    class Comparison(ctypes.Structure):
        _fields_ = [
            ("arg", ctypes.c_uint),
            ("op", ctypes.c_uint),
            ("a", ctypes.c_uint64),
            ("b", ctypes.c_uint64),
        ]

    seccomp.seccomp_rule_add_array.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.POINTER(Comparison),
    ]
    comparison = Comparison(1, 4, termios.FIOCLEX, 0)  # SCMP_CMP_EQ
    if seccomp.seccomp_rule_add_array(
        context,
        0x7FFF0000,
        seccomp.seccomp_syscall_resolve_name(b"ioctl"),
        1,
        ctypes.byref(comparison),
    ):
        raise RuntimeError("Cannot configure close-on-exec ioctl")
    os.closerange(3, resource.getrlimit(resource.RLIMIT_NOFILE)[0])
    os.chroot(root)
    os.chdir("/")
    os.setgroups([])
    os.setgid(65534)
    os.setuid(65534)
    # Changing credentials clears PDEATHSIG, so install it after dropping UID.
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) or libc.prctl(38, 1, 0, 0, 0):
        raise RuntimeError("Cannot install parent-death/no-new-privileges controls")
    if os.getppid() != parent:
        raise RuntimeError("Isolation controller exited during setup")
    for limit, value in (
        (resource.RLIMIT_CPU, cpu_seconds),
        (resource.RLIMIT_AS, memory_bytes),
        (resource.RLIMIT_FSIZE, output_bytes),
        (resource.RLIMIT_NOFILE, 64),
        (resource.RLIMIT_CORE, 0),
        (resource.RLIMIT_NPROC, 0),
    ):
        resource.setrlimit(limit, (value, value))
    if seccomp.seccomp_load(context):
        raise RuntimeError("Cannot load syscall filter")
    os.execve(
        executable,
        [executable, "-I", "-S", "-B", "/policy.py"],
        {
            "LANG": "C.UTF-8",
            "PYTHONHASHSEED": "0",
            "LD_LIBRARY_PATH": "/usr/local/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:/lib64:/lib",
        },
    )


if __name__ == "__main__":
    enter(sys.argv[1], sys.argv[2], *(int(value) for value in sys.argv[3:]))
