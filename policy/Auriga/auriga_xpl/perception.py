"""Perception services (VGGT-Omega + SAM 3.1) owned by the Auriga policy server.

Auriga's external-perception features (vggt_height, object_table, scene3d, object_table_observe,
scene3d_observe, held_perception) call two local HTTP services:

* the VGGT service (``/health``, ``/height``, ``/object_table``, ``/points``, ``/held``), which the
  configuration reaches through ``features.vggt_endpoint``;
* the SAM 3.1 text-prompt service (``/health``, ``/segment``), which only the VGGT service calls.

``start_services(cfg, gpu_id)`` starts both as child processes of the policy server, on loopback
ports it checks for conflicts, waits for ``/health`` with a timeout, verifies the version strings,
and returns a handle whose ``urls["vggt"]`` is injected into the Auriga configuration as
``features.vggt_endpoint``. The evaluator does nothing extra: the processes live and die with the
policy server (a small shell guard kills a service when the server process disappears, even after
SIGKILL), and ``handle.stop()`` stops them and frees the ports.

The service code is ``services/vggt_height`` of the pinned Auriga checkout (point and text masks
through SAM 3.1), started with the arguments of its ``start.sh``.

Crash handling matches the v5 runs: Auriga's clients never fail an episode on a service error
(object_table / scene3d answer ``status: "unknown"``, obs3d attachments report ``"unavailable"``,
pixel_plane keeps the caller plane). A watchdog restarts a crashed service once on the same port;
after a second crash it is left down and marked unavailable, and Auriga degrades as above.

Standard library only (the policy environment needs nothing extra for this module).
"""
from __future__ import annotations

import atexit
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ADAPTER_DIR = Path(__file__).resolve().parents[1]

# Version strings the services report on GET /health.
EXPECTED_VGGT = {"version": "vggt-height-v2", "object_table": "object-table-v5-hp1",
                 "points": "scene3d-points-v1", "held": "held-v1"}
EXPECTED_SAM31 = {"version": "sam31-text-v1"}

# The guard runs the service in the background and kills it when the policy server process is gone
# (checked every 2 s) or when the guard itself is terminated; its exit status is the service's.
_GUARD = r'''
p=$1; shift
"$@" &
c=$!
trap 'kill -TERM "$c" 2>/dev/null; wait "$c"; exit $?' TERM INT HUP
while kill -0 "$p" 2>/dev/null && kill -0 "$c" 2>/dev/null; do sleep 2 & wait $!; done
if kill -0 "$c" 2>/dev/null; then kill -TERM "$c" 2>/dev/null; fi
wait "$c"
'''


class PerceptionError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------- helpers
def _resolve(value, default: str) -> Path:
    path = Path(str(value)) if value not in (None, "") else Path(default)
    return path if path.is_absolute() else (ADAPTER_DIR / path).resolve()


def _config_env() -> dict:
    """KEY=VALUE lines of the adapter's config.env (setup_eval_policy_server.sh exports the same file;
    read here too so the module works when the server is started another way). No shell expansion."""
    values: dict[str, str] = {}
    path = ADAPTER_DIR / "config.env"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def _setting(cfg: dict, key: str, env: str, default):
    """deploy.yml ``perception`` section > process environment > config.env > default."""
    section = cfg.get("perception") if isinstance(cfg.get("perception"), dict) else {}
    if section.get(key) not in (None, ""):
        return section[key]
    if os.environ.get(env, "") != "":
        return os.environ[env]
    if _config_env().get(env, "") != "":
        return _config_env()[env]
    return default


def _truthy(value) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def port_free(port: int, host: str = "127.0.0.1") -> bool:
    """True when nothing listens on host:port. Bind test with SO_REUSEADDR, as the services bind
    (http.server sets allow_reuse_address), so TIME_WAIT sockets of a stopped service do not count."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, int(port)))
        except OSError:
            return False
    return True


def _pick_port(preferred: int, taken: set[int]) -> int:
    if preferred not in taken and port_free(preferred):
        return preferred
    for port in range(preferred + 10, preferred + 400):
        if port not in taken and port_free(port):
            return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def get_health(url: str, timeout_s: float = 3.0) -> dict | None:
    try:
        with urlopen(url.rstrip("/") + "/health", timeout=timeout_s) as response:
            answer = json.loads(response.read())
    except (URLError, OSError, ValueError):
        return None
    return answer if isinstance(answer, dict) and answer.get("status") == "ok" else None


def _version_mismatch(answer: dict, expected: dict) -> dict:
    return {key: answer.get(key) for key, value in expected.items() if answer.get(key) != value}


def healthy(target=None, timeout_s: float = 3.0) -> bool:
    """Health of a handle, a VGGT URL, or (no argument) the default http://127.0.0.1:8331."""
    if isinstance(target, PerceptionHandle):
        return target.healthy(timeout_s)
    answer = get_health(str(target or "http://127.0.0.1:8331"), timeout_s)
    return bool(answer) and not _version_mismatch(answer, EXPECTED_VGGT) and answer.get("object_table_backend") == "sam31"


def _resolve_gpu(cfg: dict, gpu_id) -> str:
    """Physical GPU index for the services: explicit setting > gpu_id (relative to the server's
    CUDA_VISIBLE_DEVICES when that is set) > the server's first visible GPU > 0."""
    explicit = _setting(cfg, "gpu", "VGGT_GPU", None)
    if explicit not in (None, ""):
        return str(explicit)
    visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if gpu_id not in (None, "", "null"):
        index = int(gpu_id)
        if visible:
            return visible[index] if 0 <= index < len(visible) else visible[0]
        return str(index)
    return visible[0] if visible else "0"


def _service_env(home: Path, gpu: str) -> dict:
    env = {key: value for key, value in os.environ.items()
           if key not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "VIRTUAL_ENV", "CONDA_PREFIX",
                          "CONDA_DEFAULT_ENV", "LD_PRELOAD")}
    env.update(HF_HOME=str(home / "cache" / "hf"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               CUDA_VISIBLE_DEVICES=gpu,
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", PYTHONUNBUFFERED="1",
               PYTHONNOUSERSITE="1")
    return env


# ---------------------------------------------------------------------------------------- service
class _Service:
    def __init__(self, name: str, port: int, cmd_for_port, env: dict, log_dir: Path, expected: dict):
        self.name, self.port, self.cmd_for_port, self.env, self.expected = name, port, cmd_for_port, env, expected
        self.log_path = log_dir / f"{name}.log"
        self.status_path = log_dir / f"{name}.STATUS"
        self.proc: subprocess.Popen | None = None
        self.starts = 0
        self.state = "new"            # new | starting | up | restarting | unavailable | stopped
        self.last_health: dict | None = None
        self.started_s: float | None = None
        self.ready_s: float | None = None
        self.exit_codes: list[int] = []

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def launch(self) -> None:
        cmd = self.cmd_for_port(self.port) + ["--status", str(self.status_path)]
        log = open(self.log_path, "ab")
        log.write(f"\n=== {time.strftime('%F %T %Z')} start #{self.starts + 1} port={self.port} "
                  f"gpu={self.env.get('CUDA_VISIBLE_DEVICES')}\n".encode())
        log.flush()
        self.proc = subprocess.Popen(["/bin/sh", "-c", _GUARD, "auriga-perception-guard", str(os.getpid()), *cmd],
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     cwd=str(self.log_path.parent), env=self.env, start_new_session=True)
        log.close()
        self.starts += 1
        self.started_s = time.monotonic()
        self.state = "starting"

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def wait_healthy(self, timeout_s: float) -> dict:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not self.alive():
                code = self.proc.returncode if self.proc else None
                raise PerceptionError(f"{self.name} exited with code {code} while loading; log tail:\n{self.tail()}")
            answer = get_health(self.url, 2.0)
            if answer is not None:
                bad = _version_mismatch(answer, self.expected)
                if bad:
                    raise PerceptionError(f"{self.name} on {self.url} reports unexpected versions {bad}")
                self.last_health, self.state = answer, "up"
                self.ready_s = time.monotonic() - (self.started_s or time.monotonic())
                return answer
            time.sleep(1.0)
        raise PerceptionError(f"{self.name} not healthy after {timeout_s:.0f} s; log tail:\n{self.tail()}")

    def terminate(self, timeout_s: float = 20.0) -> None:
        proc = self.proc
        if proc is None:
            return
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout_s)
            except subprocess.TimeoutExpired:
                pass
        try:  # whatever is left in the group (service child of the guard)
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            pass

    def tail(self, lines: int = 25) -> str:
        try:
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-lines:])
        except OSError:
            return ""

    def describe(self) -> dict:
        health = self.last_health or {}
        return {"url": self.url, "state": self.state, "starts": self.starts, "exit_codes": self.exit_codes,
                "pid": self.proc.pid if self.proc else None, "ready_s": self.ready_s, "log": str(self.log_path),
                "health": {key: health.get(key) for key in ("version", "object_table", "points", "held",
                                                             "object_table_backend", "sam31", "device",
                                                             "cuda_visible_devices", "load_s", "allocated_mb",
                                                             "reserved_mb") if key in health}}


# ---------------------------------------------------------------------------------------- handle
class PerceptionHandle:
    """Owned services (or a reused external one). ``urls``: {"vggt": ..., "sam31": ...}."""

    def __init__(self, services: list[_Service], *, variant: str, gpu: str | None, owned: bool,
                 restart_limit: int, start_timeout_s: float, urls: dict | None = None):
        self.services = services
        self.variant, self.gpu, self.owned = variant, gpu, owned
        self.restart_limit, self.start_timeout_s = restart_limit, start_timeout_s
        self._urls = urls
        self.events: list[dict] = []
        self._stopping = threading.Event()
        self._lock = threading.Lock()
        self._watchdog: threading.Thread | None = None
        self.startup_s: float | None = None

    @property
    def urls(self) -> dict:
        if self._urls is not None:
            return dict(self._urls)
        return {service.name: service.url for service in self.services}

    def healthy(self, timeout_s: float = 3.0) -> bool:
        vggt = get_health(self.urls["vggt"], timeout_s)
        if not vggt or _version_mismatch(vggt, EXPECTED_VGGT) or vggt.get("object_table_backend") != "sam31":
            return False
        sam31 = self.urls.get("sam31")
        return sam31 is None or bool(get_health(sam31, timeout_s))

    def status(self) -> dict:
        return {"variant": self.variant, "gpu": self.gpu, "owned": self.owned, "urls": self.urls,
                "healthy": self.healthy(), "services": [service.describe() for service in self.services],
                "events": list(self.events)}

    # -- watchdog: restart a crashed service once, then leave it down (Auriga degrades per request)
    def _start_watchdog(self) -> None:
        self._watchdog = threading.Thread(target=self._watch, name="auriga-perception-watchdog", daemon=True)
        self._watchdog.start()

    def _watch(self) -> None:
        while not self._stopping.wait(5.0):
            for service in self.services:
                with self._lock:
                    if self._stopping.is_set() or service.state != "up" or service.alive():
                        continue
                    code = service.proc.returncode if service.proc else None
                    service.exit_codes.append(code)
                    event = {"time": time.strftime("%F %T %Z"), "service": service.name, "exit_code": code}
                    if service.starts - 1 >= self.restart_limit:
                        service.state = "unavailable"
                        event["action"] = "left down (restart limit reached); Auriga degrades per request"
                        self.events.append(event)
                        self._log(event)
                        continue
                    event["action"] = "restart"
                    self.events.append(event)
                    self._log(event)
                    if not port_free(service.port):
                        time.sleep(2.0)
                    service.launch()
                    service.state = "restarting"
                try:  # outside the lock, so stop() is never blocked by a model load
                    service.wait_healthy(self.start_timeout_s)
                    self._log({"service": service.name, "action": "restarted", "ready_s": service.ready_s})
                except PerceptionError as error:
                    if not self._stopping.is_set():
                        service.state = "unavailable"
                        self._log({"service": service.name, "action": "restart failed", "error": str(error)[:500]})

    def _log(self, event: dict) -> None:
        print("[perception] " + json.dumps(event, default=str), file=sys.stderr, flush=True)

    def stop(self, timeout_s: float = 20.0) -> dict:
        """Stop owned services (VGGT first, then SAM 3.1) and report whether their ports are free."""
        if self._stopping.is_set():
            return {"stopped": True, "repeat": True}
        self._stopping.set()
        with self._lock:
            for service in reversed(self.services):
                if self.owned:
                    service.terminate(timeout_s)
                    service.state = "stopped"
        if self._watchdog is not None:
            self._watchdog.join(10.0)
        ports = {service.name: port_free(service.port) for service in self.services} if self.owned else {}
        return {"stopped": True, "ports_free": ports}


# ---------------------------------------------------------------------------------------- start
def _commands(variant: str, home: Path, auriga_src: Path, sam31_url_for):
    weights = home / "weights"
    vggt_py, sam31_py = home / "venv" / "bin" / "python", home / "venv_sam31" / "bin" / "python"
    vggt_ckpt = weights / "vggt_omega_1b_512.pt"
    sam31_ckpt = weights / "sam3.1" / "sam3.1_multiplex.pt"
    if variant == "auriga":
        base = auriga_src / "services" / "vggt_height"
        vggt_script, sam31_script = base / "vggt_height_service.py", base / "sam31" / "sam31_service.py"
        for script in (vggt_script, sam31_script):
            if not script.is_file():
                raise PerceptionError(f"missing Auriga service file {script} (run install.sh)")
        # Same arguments as services/vggt_height/start.sh.
        sam31_cmd = lambda port: [str(sam31_py), str(sam31_script), "--host", "127.0.0.1", "--port", str(port),
                                  "--ckpt", str(sam31_ckpt), "--cache", "64"]
        vggt_cmd = lambda port: [str(vggt_py), str(vggt_script), "--host", "127.0.0.1", "--port", str(port),
                                 "--ckpt", str(vggt_ckpt), "--sam31", sam31_url_for(), "--cache", "128",
                                 "--max_queue", "256"]
    else:
        raise PerceptionError(f"unknown perception variant {variant!r} (auriga)")
    needed = [vggt_py, sam31_py, vggt_ckpt, sam31_ckpt]
    missing = [str(path) for path in needed if not path.exists()]
    if missing:
        raise PerceptionError("perception not installed: missing " + ", ".join(missing)
                              + " (run install_perception.sh and download_assets.sh)")
    return sam31_cmd, vggt_cmd


def start_services(cfg: dict | None = None, gpu_id=None) -> PerceptionHandle:
    """Start SAM 3.1 then VGGT on loopback, wait for /health, return the handle.

    cfg is the deploy.yml dict; an optional ``perception`` section overrides the environment
    (config.env): home (VGGT_HOME), variant (PERCEPTION_VARIANT), vggt_port (VGGT_PORT),
    sam31_port (SAM31_PORT), gpu (VGGT_GPU, physical index), start_timeout_s
    (PERCEPTION_START_TIMEOUT_S), restart_limit (PERCEPTION_RESTART_LIMIT), reuse (PERCEPTION_REUSE),
    log_dir (PERCEPTION_LOG_DIR). Raises PerceptionError when the services cannot be started.
    """
    cfg = dict(cfg or {})
    variant = str(_setting(cfg, "variant", "PERCEPTION_VARIANT", "auriga"))
    home = _resolve(_setting(cfg, "home", "VGGT_HOME", ""), "third_party/vggt_home")
    auriga_src = _resolve(os.environ.get("AURIGA_SRC") or _config_env().get("AURIGA_SRC") or cfg.get("auriga_src"),
                          "third_party/Auriga")
    vggt_port = int(_setting(cfg, "vggt_port", "VGGT_PORT", 8331))
    sam31_port = int(_setting(cfg, "sam31_port", "SAM31_PORT", 8333))
    timeout_s = float(_setting(cfg, "start_timeout_s", "PERCEPTION_START_TIMEOUT_S", 600))
    restart_limit = int(_setting(cfg, "restart_limit", "PERCEPTION_RESTART_LIMIT", 1))
    reuse = _truthy(_setting(cfg, "reuse", "PERCEPTION_REUSE", "0"))

    if reuse:  # an already running service with the expected versions (not owned, never stopped here)
        url = f"http://127.0.0.1:{vggt_port}"
        if healthy(url):
            answer = get_health(url) or {}
            handle = PerceptionHandle([], variant="external", gpu=answer.get("cuda_visible_devices"), owned=False,
                                      restart_limit=0, start_timeout_s=timeout_s,
                                      urls={"vggt": url, "sam31": answer.get("sam31_endpoint")})
            handle._log({"action": "reuse", "url": url})
            return handle

    gpu = _resolve_gpu(cfg, gpu_id)
    result_dir = _resolve(cfg.get("result_dir"), "results")
    log_dir = _resolve(_setting(cfg, "log_dir", "PERCEPTION_LOG_DIR", ""),
                       str(result_dir / "perception" / f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"))
    log_dir.mkdir(parents=True, exist_ok=True)
    env = _service_env(home, gpu)

    ports: dict[str, int] = {}
    ports["sam31"] = _pick_port(sam31_port, set())
    ports["vggt"] = _pick_port(vggt_port, {ports["sam31"]})
    sam31_cmd, vggt_cmd = _commands(variant, home, auriga_src, lambda: f"http://127.0.0.1:{ports['sam31']}")
    sam31 = _Service("sam31", ports["sam31"], sam31_cmd, env, log_dir, EXPECTED_SAM31)
    vggt = _Service("vggt", ports["vggt"], vggt_cmd, env, log_dir, EXPECTED_VGGT)
    handle = PerceptionHandle([sam31, vggt], variant=variant, gpu=gpu, owned=True,
                              restart_limit=restart_limit, start_timeout_s=timeout_s)
    atexit.register(handle.stop)
    started = time.monotonic()
    try:
        for service in (sam31, vggt):
            for attempt in range(2):  # a port taken between the check and the bind: pick another once
                service.launch()
                try:
                    service.wait_healthy(timeout_s)
                    break
                except PerceptionError:
                    if attempt == 0 and "Address already in use" in service.tail(60):
                        service.terminate()
                        service.starts -= 1
                        service.port = _pick_port(service.port + 1, set(ports.values()))
                        ports[service.name] = service.port
                        continue
                    raise
        if vggt.last_health.get("object_table_backend") != "sam31":
            raise PerceptionError(f"VGGT service does not see SAM 3.1 at {sam31.url}: {vggt.last_health}")
    except BaseException:
        handle.stop()
        raise
    handle.startup_s = time.monotonic() - started
    handle._log({"action": "started", "variant": variant, "gpu": gpu, "urls": handle.urls,
                 "startup_s": round(handle.startup_s, 1), "log_dir": str(log_dir)})
    handle._start_watchdog()
    return handle


# ---------------------------------------------------------------------------------------- CLI
def _main(argv: list[str]) -> int:
    """python -m auriga_xpl.perception [--hold SECONDS] [--variant auriga] [--gpu N]
    Starts the services, prints the status JSON, keeps them up for --hold seconds (0: stop at once),
    stops them and prints whether the ports are free."""
    import argparse
    parser = argparse.ArgumentParser(prog="auriga_xpl.perception")
    parser.add_argument("--hold", type=float, default=0.0)
    parser.add_argument("--variant")
    parser.add_argument("--gpu")
    parser.add_argument("--home")
    parser.add_argument("--vggt-port", type=int)
    parser.add_argument("--sam31-port", type=int)
    parser.add_argument("--log-dir")
    args = parser.parse_args(argv)
    section = {key: value for key, value in dict(variant=args.variant, gpu=args.gpu, home=args.home,
                                                  vggt_port=args.vggt_port, sam31_port=args.sam31_port,
                                                  log_dir=args.log_dir).items() if value not in (None, "")}
    handle = start_services({"perception": section})
    print(json.dumps(handle.status(), indent=1, default=str), flush=True)
    try:
        deadline = time.monotonic() + args.hold
        while time.monotonic() < deadline:
            time.sleep(min(5.0, max(0.0, deadline - time.monotonic())))
    except KeyboardInterrupt:
        pass
    print(json.dumps(handle.stop(), default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
