"""Start, stop and check the services of a deploy (`harness up`, `down`, `status`).

Each service runs as its own process group with its log in <state>/logs/<name>.log
and a record in <state>/<name>.json (pid, port, GPU, command). A service whose port
already answers is left alone: on a shared machine it may have been started by hand
or by another checkout, and starting a second copy would only fail on the port.
"""

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from . import deploy as deploys


def listening(port: int, host: str = "127.0.0.1", timeout: float = 0.5) -> bool:
    """Whether something accepts TCP connections on `host`:`port`."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _alive(pid: int | None) -> bool:
    """Whether the process runs (an exited child of this process is reaped, not alive)."""
    if not pid:
        return False
    try:
        if os.waitpid(pid, os.WNOHANG)[0] == pid:
            return False
    except ChildProcessError:
        pass  # not our child
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1][:1] != "Z"
    except (OSError, IndexError):
        return True


def _record_path(deploy, name: str) -> Path:
    return Path(deploy.state) / f"{name}.json"


def record(deploy, name: str) -> dict | None:
    """The record of a service started here (`<state>/<name>.json`: pid, port, GPU, argv,
    cwd, env, start time), or None when there is none or it is unreadable."""
    path = _record_path(deploy, name)
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def status(deploy, names=None) -> list[dict]:
    """One row per service the deploy places (or `names`): mode, GPU, URL, whether
    its port answers and the process harness up started for it."""
    rows = []
    names = list(names) if names else sorted(name for name, p in deploy.placements.items() if p.mode != "off")
    for name in names:
        placement = deploy.placement(name)
        spec = deploy.services[name]
        row = {"service": name, "kind": spec.kind, "mode": placement.mode, "gpu": placement.gpu,
               "url": deploys.url(deploy, name)}
        if placement.mode in deploys.LOCAL:
            port = deploys.port(deploy, name)
            started = record(deploy, name)
            pid = (started or {}).get("pid")
            row.update(port=port, up=listening(port), pid=pid if _alive(pid) else None,
                       started_by_harness=bool(started and _alive(pid)), vram_gb=spec.vram_gb)
        elif placement.mode == "external":
            address = row["url"].split("://", 1)[-1].split("/", 1)[0]
            host, _, port = address.rpartition(":")
            row["up"] = listening(int(port or 80), host or address) if port.isdigit() else None
        rows.append(row)
    return rows


def _spawn(deploy, name: str, log=print, *, allow_on_demand: bool = False):
    """Start one resident service (or, for the broker, an on-demand one) unless its port
    already answers; the process or None."""
    placement = deploy.placement(name)
    allowed = ("resident", "on_demand") if allow_on_demand else ("resident",)
    if placement.mode not in allowed:
        raise ValueError(f"{name} is {placement.mode} in deploy {deploy.name}; only resident services are started "
                         "(the broker starts on-demand ones)")
    port = deploys.port(deploy, name)
    if listening(port):
        log(f"{name}: port {port} already answers; left as it is")
        return None
    missing = deploys.missing_files(deploy, name)
    if missing:
        raise RuntimeError(f"{name}: missing files {missing}; run `harness fetch {name} --deploy {deploy.name}`")
    argv, environment, cwd = deploys.launch(deploy, name)
    logs = Path(deploy.state) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with open(logs / f"{name}.log", "ab") as handle:
        handle.write(f"\n=== {time.strftime('%F %T')} start {' '.join(argv)}\n".encode())
        handle.flush()
        process = subprocess.Popen(argv, cwd=cwd, env={**os.environ, **environment}, stdout=handle,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    _record_path(deploy, name).write_text(json.dumps({
        "pid": process.pid, "port": port, "gpu": placement.gpu, "argv": argv, "cwd": cwd,
        "env": {k: v for k, v in environment.items() if k in ("CUDA_VISIBLE_DEVICES", "PYTHONPATH")},
        "started": time.time()}, indent=1))
    log(f"{name}: started pid {process.pid} on GPU {placement.gpu}, port {port}; log {logs / (name + '.log')}")
    return process


def start(deploy, name: str, *, wait: bool = True, log=print) -> dict:
    """Start one resident service unless its port already answers. Returns its status."""
    process = _spawn(deploy, name, log)
    if wait and process is not None:
        wait_ready(deploy, name, process)
    return status(deploy, [name])[0]


def wait_ready(deploy, name: str, process=None, timeout: float | None = None) -> None:
    """Wait until service `name` answers on its port. Raises RuntimeError when `process`
    (or the pid of its record) exits first, and TimeoutError after `timeout` seconds
    (default three times its load_seconds, at least 60)."""
    spec = deploy.services[name]
    port = deploys.port(deploy, name)
    deadline = time.monotonic() + (timeout if timeout is not None else max(60.0, 3 * spec.load_seconds))
    pid = None if process is not None else (record(deploy, name) or {}).get("pid")
    while time.monotonic() < deadline:
        if listening(port):
            return
        exited = process.poll() is not None if process is not None else (pid is not None and not _alive(pid))
        if exited:
            raise RuntimeError(f"{name} exited before it answered; "
                               f"see {Path(deploy.state) / 'logs' / (name + '.log')}")
        time.sleep(1.0)
    raise TimeoutError(f"{name} did not answer on port {port} within {deadline:.0f} s")


def stop(deploy, name: str, *, grace: float = 15.0, log=print) -> bool:
    """Stop a service harness up started (SIGINT, then SIGTERM, then SIGKILL to its
    process group). Services started otherwise are not touched."""
    started = record(deploy, name)
    pid = (started or {}).get("pid")
    if not _alive(pid):
        log(f"{name}: not started by harness up here")
        _record_path(deploy, name).unlink(missing_ok=True)
        return False
    _terminate(pid, (signal.SIGINT, signal.SIGTERM, signal.SIGKILL), grace)
    _record_path(deploy, name).unlink(missing_ok=True)
    log(f"{name}: stopped pid {pid}")
    return True


def _terminate(pid: int, signals, grace: float) -> None:
    """Signal a process group until its leader is gone, each signal `grace` seconds
    apart. A process started from a background job may ignore SIGINT; the next signal
    still ends it."""
    for sig in signals:
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline and _alive(pid):
            time.sleep(0.2)
        if not _alive(pid):
            return


def up(deploy, names=None, *, log=print) -> list[dict]:
    """Start the resident services (all, or `names`) at once, then wait until each
    answers; start the broker when the deploy has on-demand services."""
    names = list(names) if names else sorted(n for n, p in deploy.placements.items() if p.mode == "resident")
    names = [name for name in names if deploy.placement(name).mode == "resident"]
    processes = {name: _spawn(deploy, name, log) for name in names}
    for name, process in processes.items():
        if process is not None:
            wait_ready(deploy, name, process)
    start_broker(deploy, log=log)
    return status(deploy, names)


def start_broker(deploy, log=print):
    """Start the broker of the deploy's on-demand services, unless there are none or it runs."""
    from .broker import broker_port

    if not any(p.mode == "on_demand" for p in deploy.placements.values()):
        return None
    port = broker_port(deploy)
    if listening(port):
        return None
    logs = Path(deploy.state) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    source = str(deploys.REPO / "src")
    with open(logs / "broker.log", "ab") as handle:
        process = subprocess.Popen([sys.executable, "-u", "-m", "harness_policy.services.broker", "--deploy",
                                    str(deploy.path)], cwd=str(deploys.REPO), stdout=handle,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
                                   env={**os.environ, "PYTHONPATH": os.pathsep.join(
                                       [source, str(deploys.REPO), os.environ.get("PYTHONPATH", "")])})
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not listening(port):
        if process.poll() is not None:
            raise RuntimeError(f"the broker exited; see {logs / 'broker.log'}")
        time.sleep(0.2)
    log(f"broker: pid {process.pid} on port {port} for the on-demand services")
    return process


def down(deploy, names=None, *, log=print) -> None:
    """Stop the services harness up (or the broker) started; without names also the broker."""
    everything = not names
    names = list(names) if names else sorted(n for n, p in deploy.placements.items() if p.mode in deploys.LOCAL)
    if everything:
        broker = Path(deploy.state) / "broker.json"
        try:
            pid = json.loads(broker.read_text())["pid"]
        except (OSError, ValueError, KeyError):
            pid = None
        if _alive(pid):
            _terminate(pid, (signal.SIGTERM, signal.SIGKILL), 10.0)
            log(f"broker: stopped pid {pid}")
        broker.unlink(missing_ok=True)
    for name in names:
        if record(deploy, name) or deploy.placement(name).mode == "resident":
            stop(deploy, name, log=log)
