"""Reserve native evaluator logging capacity before simulator startup."""

import errno
import fcntl
import os
import time

from orchestrator.files import append_jsonl


def reserve_monitor_pipe(read_fd, audit_path):
    """Wait for logging capacity before native startup can block in C++."""
    started = time.monotonic()
    waiting = False
    while fcntl.fcntl(read_fd, fcntl.F_GETPIPE_SZ) < 64 * 1024:
        try:
            fcntl.fcntl(read_fd, fcntl.F_SETPIPE_SZ, 64 * 1024)
        except OSError as exc:
            if exc.errno not in (errno.EPERM, errno.ENOMEM):
                raise
            if not waiting:
                append_jsonl(audit_path, {"pid": os.getpid(), "fd": read_fd,
                                         "status": "waiting", "errno": exc.errno})
                waiting = True
            time.sleep(5)
    append_jsonl(audit_path, {"pid": os.getpid(), "fd": read_fd, "status": "reserved",
                             "bytes": fcntl.fcntl(read_fd, fcntl.F_GETPIPE_SZ),
                             "wait_seconds": time.monotonic() - started})
