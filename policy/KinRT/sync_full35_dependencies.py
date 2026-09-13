"""Run the pinned uv sync through an equivalent package mirror."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import tomllib


LOCK_SHA256 = "4a02069e2398769cb45974d189f9f77011bc5937809137d3698b150bb5d50a1d"
TUNA = "https://pypi.tuna.tsinghua.edu.cn"
TERMINATE_GRACE_SECONDS = 10
KILL_GRACE_SECONDS = 5


def mirrored_lock(original: bytes, mirror: str) -> bytes:
    if hashlib.sha256(original).hexdigest() != LOCK_SHA256:
        raise ValueError("uv.lock does not match the pinned KinRT source; no files were changed")
    if mirror not in ("pypi", "tencent", "original"):
        raise ValueError(f"Unsupported package mirror: {mirror}")
    if mirror == "original":
        return original
    replacements = {}

    def transform(value):
        if isinstance(value, dict):
            return {key: transform(item) for key, item in value.items()}
        if isinstance(value, list):
            return [transform(item) for item in value]
        if not isinstance(value, str) or not value.startswith(TUNA + "/"):
            return value
        suffix = value[len(TUNA):]
        if suffix != "/simple" and not suffix.startswith("/packages/"):
            raise ValueError(f"Unexpected locked TUNA URL: {value}")
        if mirror == "tencent":
            target = "https://mirrors.tencent.com/pypi" + suffix
        elif suffix == "/simple":
            target = "https://pypi.org/simple"
        else:
            target = "https://files.pythonhosted.org" + suffix
        replacements[value] = target
        return target

    text = original.decode("utf-8")
    expected = transform(tomllib.loads(text))
    for old, new in replacements.items():
        text = text.replace(json.dumps(old), json.dumps(new))
    if tomllib.loads(text) != expected:
        raise ValueError("Mirror transformation changed lock content beyond download URLs")
    return text.encode("utf-8")


def sync(project: Path, uv: str, mirror: str) -> int:
    project = project.resolve()
    lock = project / "uv.lock"
    original = lock.read_bytes()
    mirrored = mirrored_lock(original, mirror)
    print(f"[KinRT] Frozen lock SHA-256: {LOCK_SHA256}; download mirror: {mirror}", flush=True)
    process = None
    termination_signal = None
    returncode = 0

    def request_termination(signum, _frame):
        nonlocal termination_signal
        termination_signal = signum

    previous_handler = signal.signal(signal.SIGTERM, request_termination)
    # Stop and reap uv before restoring a lock that its child processes may read.
    try:
        if mirrored != original:
            lock.write_bytes(mirrored)
        if termination_signal is None:
            process = subprocess.Popen(
                [uv, "sync", "--frozen", "--no-default-groups", "--python", str(project / ".venv/bin/python")],
                cwd=project,
                start_new_session=os.name == "posix",
            )
            while termination_signal is None:
                try:
                    returncode = process.wait(timeout=0.2)
                    break
                except subprocess.TimeoutExpired:
                    continue
    finally:
        try:
            if process is not None and (termination_signal is not None or process.poll() is None):
                stop_uv(process)
            if lock.read_bytes() != original:
                lock.write_bytes(original)
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
    return 128 + termination_signal if termination_signal is not None else returncode


def stop_uv(process: subprocess.Popen) -> None:
    if os.name == "posix":
        # uv may exit before a build subprocess that ignores SIGTERM.
        for signum, grace in ((signal.SIGTERM, TERMINATE_GRACE_SECONDS), (signal.SIGKILL, KILL_GRACE_SECONDS)):
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                process.wait(timeout=KILL_GRACE_SECONDS)
                return
            deadline = time.monotonic() + grace
            while True:
                process.poll()
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    process.wait(timeout=KILL_GRACE_SECONDS)
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(0.1, remaining))
        raise RuntimeError(f"uv process group {process.pid} still exists after SIGKILL; lock was not restored")

    try:
        process.terminate()
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        process.wait(timeout=KILL_GRACE_SECONDS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("--uv", required=True)
    parser.add_argument("--mirror", choices=("pypi", "tencent", "original"), default="pypi")
    args = parser.parse_args()
    sys.exit(sync(args.project, args.uv, args.mirror))
