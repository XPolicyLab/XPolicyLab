"""Client side of XPolicyLab policy servers: a machine-wide lock per server."""

import contextlib
import os
import tempfile


# Shared by every process on the machine, isolated kernels included (see policy_server_lock):
# /dev/shm on Linux, else the system's temporary directory (macOS, where there is no
# isolation; on Linux without /dev/shm an isolated kernel's /tmp is its own and the lock
# would not be shared).
POLICY_LOCK_DIR = os.path.join("/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir(),
                               "harness-policy-locks")


def policy_lock_path(url):
    """The lock file of one policy server, however its URL is written (localhost or
    127.0.0.1, trailing slash)."""
    import re

    server = re.sub(r"^\w+://", "", str(url).strip().lower()).rstrip("/").replace("localhost", "127.0.0.1")
    return os.path.join(POLICY_LOCK_DIR, re.sub(r"[^a-z0-9._-]+", "_", server) + ".lock")


def _open_lock_file(path):
    """A read-only descriptor of the lock file (flock needs no more), created 0666 if missing.
    An existing file is opened without O_CREAT: in a sticky directory the kernel refuses
    O_CREAT on another account's file (fs.protected_regular), even one that exists."""
    for _ in range(10):
        try:
            return os.open(path, os.O_RDONLY)
        except FileNotFoundError:
            pass
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_CREAT | os.O_EXCL, 0o666)
        except FileExistsError:
            continue  # another process created it in between
        with contextlib.suppress(OSError):
            os.fchmod(descriptor, 0o666)  # the umask narrowed the mode
        return descriptor
    # Neither opens nor can be created: a dangling symlink, say.
    raise OSError(f"cannot open or create the policy server lock {path}")


@contextlib.contextmanager
def policy_server_lock(url):
    """Hold a machine-wide lock on one XPolicyLab policy server.

    An XPolicyLab server keeps a single current observation: update_obs replaces it and
    get_action answers from it. Two episodes sharing a server (parallel suite slots) can
    otherwise interleave, so one gets the action computed from the other's scene. The
    lock is a file lock, so it holds across threads and processes on this machine only. It
    does not cover clients that do not take it, such as RoboDojo's own eval client, or
    harness runs on another machine reaching the same server. A call that waits on a
    restarting server keeps the lock, so the other episodes wait with it.

    The lock files live in POLICY_LOCK_DIR, which every episode's isolated kernel sees as
    the same directory: the isolation gives each kernel its own empty home and /tmp
    (runtime/isolation.py), so a lock there would be private to one episode. On Linux the
    directory and files are open to every account on the machine (sticky, like /tmp), so
    whichever account creates them, the others can still take the lock."""
    import fcntl

    os.makedirs(POLICY_LOCK_DIR, exist_ok=True)
    with contextlib.suppress(OSError):  # another account created it
        os.chmod(POLICY_LOCK_DIR, 0o1777)
    try:
        descriptor = _open_lock_file(policy_lock_path(url))
    except PermissionError as error:
        raise PermissionError(f"cannot open the policy server lock in {POLICY_LOCK_DIR}; it must be writable "
                              f"by every account that runs policies (chmod 1777): {error}") from error
    with os.fdopen(descriptor, "rb") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
