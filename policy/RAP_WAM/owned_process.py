"""Run one command in an owned session and clean it up without PID-based commands from callers."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def session_members():
    """Read only enough process metadata to check this still-live supervisor's group."""
    group = os.getpgrp()
    result = []
    proc = Path("/proc")
    if proc.is_dir():
        for path in proc.iterdir():
            if not path.name.isdigit() or int(path.name) == os.getpid():
                continue
            try:
                # comm may contain spaces or parentheses; fields after the last ')' start at state.
                fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
                if int(fields[2]) == group and fields[0] not in ("Z", "X"):
                    result.append(int(path.name))
            except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, IndexError):
                continue
    else:
        listing = subprocess.Popen(["ps", "-axo", "pid=,pgid=,stat="], stdout=subprocess.PIPE, text=True)
        stdout, _ = listing.communicate()
        if listing.returncode:
            raise RuntimeError("Unable to inspect the owned session for cleanup")
        for line in stdout.splitlines():
            pid, pgid, state = line.split()
            if int(pgid) == group and int(pid) not in (os.getpid(), listing.pid) and not state.startswith("Z"):
                result.append(int(pid))
    return result


def save_status(control, record):
    pending = control / "status.pending.json"
    pending.write_text(json.dumps(record) + "\n")
    pending.replace(control / "status.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-dir", required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required")
    root = Path(os.environ["RAP_WAM_WORK_DIR"]).resolve()
    control = Path(args.control_dir).resolve()
    control.relative_to(root)
    if not control.is_dir():
        parser.error("control directory must already exist under RAP_WAM_WORK_DIR")

    # The supervisor remains the session/group leader throughout cleanup.
    # No other process can reuse its numeric group while this process is alive.
    os.setsid()
    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    child = None
    exit_code = 1
    record = {"supervisor_pid": os.getpid(), "session_id": os.getsid(0)}
    try:
        if (control / "stop").exists() or os.getppid() != args.parent_pid:
            exit_code = 143
        else:
            child = subprocess.Popen(command)
            record["worker_pid"] = child.pid
            (control / "started.json").write_text(json.dumps(record) + "\n")
            while (child.poll() is None and not stopping and not (control / "stop").exists()
                   and os.getppid() == args.parent_pid):
                time.sleep(0.05)
            exit_code = child.returncode if child.returncode is not None else 143
    except BaseException as exc:
        record["error_type"] = type(exc).__name__
        print(f"[SUPERVISOR] {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        # This call can only address our own live group, never a previously recorded PGID.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        os.killpg(os.getpgrp(), signal.SIGTERM)
        deadline = time.monotonic() + 3.0
        members = []
        while True:
            if child is not None:
                child.poll()
            try:
                members = session_members()
            except Exception as exc:
                record["cleanup_inspection_error"] = type(exc).__name__
                # Unknown membership calls for a final signal to our still-owned
                # group, never a scan for processes to terminate elsewhere.
                members = [-1]
                break
            if not members or time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        record["exit_code"] = 128 - exit_code if exit_code < 0 else exit_code
        record["cleanup"] = "group_sigkill_requested" if members else "complete"
        try:
            save_status(control, record)
        finally:
            if members:
                # Includes this supervisor: parent waits for its exit and reads the recorded
                # worker status. Group identity stays reserved until this final syscall.
                os.killpg(os.getpgrp(), signal.SIGKILL)
    return record["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
