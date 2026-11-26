"""On-demand services: started when a skill first calls them, swapped on a shared GPU.

A deploy places a service with mode = "on_demand" (and optionally swap = "<group>",
idle_seconds = N). The broker, one per deploy and machine, starts such a service the
first time a client asks for it (POST /ensure), keeps at most one service of a swap
group loaded, and stops one that nobody used for idle_seconds. Clients ask before
each call (services/broker.py:ensure, throttled), so a service in use is never
swapped out: a request for another member of its group waits until it has been idle
for SWAP_GRACE seconds.

The simulator does not advance while a skill waits, so loading a model on demand
costs wall time only, not episode steps; results are those of a resident deploy. A
policy with memory (scoped = true) that is swapped out starts its memory afresh.

    python -m harness_policy.services.broker --deploy two-gpu      (harness up starts it)
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import signal
import threading
import time
from urllib.request import Request, urlopen

from . import deploy as deploys
from . import supervisor


SWAP_GRACE = 10.0       # seconds a service must be unused before a swap takes its GPU
DEFAULT_IDLE = 900.0    # seconds unused before an on-demand service is stopped
CLIENT_INTERVAL = 2.0   # a client asks the broker at most this often per service


class Broker:
    """The broker of one deploy's on-demand services: it starts them when asked, swaps
    members of a swap group on their GPU, and stops services left idle."""

    def __init__(self, deploy, log=print, swap_grace: float = SWAP_GRACE):
        self.deploy = deploy
        self.log = log
        self.swap_grace = swap_grace
        self.lock = threading.Condition()
        self.processes = {}     # service -> Popen (started by this broker)
        self.last_used = {}     # service -> monotonic time of the last ensure
        self.starting = set()
        self.stopped = False

    def on_demand(self) -> list[str]:
        """The deploy's on-demand services, sorted by name."""
        return sorted(name for name, p in self.deploy.placements.items() if p.mode == "on_demand")

    def _peers(self, name):
        group = self.deploy.placement(name).swap
        return [] if not group else [other for other in self.on_demand()
                                     if other != name and self.deploy.placement(other).swap == group]

    def _running(self, name) -> bool:
        return supervisor.listening(deploys.port(self.deploy, name))

    def ensure(self, name: str, timeout: float = 1800.0) -> dict:
        """Make `name` answer, swapping out idle peers of its group; returns when it does."""
        if self.deploy.placement(name).mode != "on_demand":
            raise ValueError(f"{name} is not on demand in deploy {self.deploy.name}")
        started = time.monotonic()
        with self.lock:
            while True:
                self.last_used[name] = time.monotonic()
                if self._running(name) and name not in self.starting:
                    return {"service": name, "url": deploys.url(self.deploy, name), "waited": time.monotonic() - started}
                busy = [peer for peer in self._peers(name) if (self._running(peer) or peer in self.starting)
                        and time.monotonic() - self.last_used.get(peer, 0.0) < self.swap_grace]
                if not busy and name not in self.starting:
                    break
                if time.monotonic() - started > timeout:
                    raise TimeoutError(f"{name}: its swap group stayed busy ({busy}) for {timeout:.0f} s")
                self.lock.wait(timeout=1.0)
            self.starting.add(name)
        try:
            for peer in self._peers(name):
                if self._running(peer) or peer in self.processes:
                    self.log(f"broker: swapping out {peer} for {name}")
                    supervisor.stop(self.deploy, peer, log=self.log)
                    self.processes.pop(peer, None)
            process = supervisor._spawn(self.deploy, name, self.log, allow_on_demand=True)
            if process is not None:
                self.processes[name] = process
                supervisor.wait_ready(self.deploy, name, process)
        finally:
            with self.lock:
                self.starting.discard(name)
                self.last_used[name] = time.monotonic()
                self.lock.notify_all()
        return {"service": name, "url": deploys.url(self.deploy, name), "waited": time.monotonic() - started}

    def reap(self):
        """Stop on-demand services unused for their idle_seconds."""
        while not self.stopped:
            time.sleep(5.0)
            for name in self.on_demand():
                idle = float(self.deploy.placement(name).idle_seconds or DEFAULT_IDLE)
                with self.lock:
                    unused = time.monotonic() - self.last_used.get(name, time.monotonic())
                    if name in self.starting or unused < idle or not self._running(name):
                        continue
                self.log(f"broker: {name} unused for {unused:.0f} s; stopping it")
                supervisor.stop(self.deploy, name, log=self.log)
                self.processes.pop(name, None)

    def status(self) -> list[dict]:
        """One row per on-demand service: its swap group, whether it answers, and seconds
        since it was last asked for (None if never)."""
        now = time.monotonic()
        return [{"service": name, "swap": self.deploy.placement(name).swap, "up": self._running(name),
                 "idle_seconds": round(now - self.last_used[name], 1) if name in self.last_used else None}
                for name in self.on_demand()]

    def serve(self, host="127.0.0.1", port=None):
        """The broker's HTTP server (GET: status; POST /ensure {"service"}: ensure), bound
        but not yet serving, on the deploy's broker port by default. Starts the idle reaper."""
        broker = self
        port = int(port or broker_port(self.deploy))

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def _reply(self, status, body):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._reply(200, {"deploy": broker.deploy.name, "services": broker.status()})

            def do_POST(self):
                try:
                    body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
                    if self.path != "/ensure":
                        raise ValueError(f"unknown route {self.path}")
                    self._reply(200, broker.ensure(str(body["service"])))
                except Exception as error:
                    self._reply(400, {"error": f"{type(error).__name__}: {error}"})

        server = ThreadingHTTPServer((host, port), Handler)
        threading.Thread(target=self.reap, daemon=True).start()
        return server


def broker_port(deploy) -> int:
    """The port of the deploy's broker (default 8899)."""
    return int(getattr(deploy, "broker_port", None) or 8899)


def broker_url(deploy) -> str:
    """The URL of the deploy's broker on this machine."""
    return f"http://127.0.0.1:{broker_port(deploy)}"


_LAST: dict[tuple, float] = {}
_GUARD = threading.Lock()


def ensure(config: dict | None, *, timeout: float = 1800.0) -> None:
    """Before calling a service: when its client config is on demand, ask the broker to
    make it answer (at most every CLIENT_INTERVAL seconds while it is in use)."""
    on_demand = (config or {}).get("on_demand") if isinstance(config, dict) else None
    if not on_demand:
        return
    key = (on_demand["url"], on_demand["service"])
    with _GUARD:
        if time.monotonic() - _LAST.get(key, -1e9) < CLIENT_INTERVAL:
            return
    request = Request(on_demand["url"].rstrip("/") + "/ensure", data=json.dumps({"service": on_demand["service"]}).encode(),
                      headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            json.load(response)
    except Exception as error:
        detail = getattr(error, "read", lambda: b"")().decode(errors="replace") if hasattr(error, "read") else ""
        raise RuntimeError(f"service {on_demand['service']} could not be started on demand: {error} {detail}") from error
    with _GUARD:
        _LAST[key] = time.monotonic()


def without_on_demand(config: dict | None) -> dict | None:
    """The client config without the broker entry (for clients that pass it on)."""
    if not isinstance(config, dict) or "on_demand" not in config:
        return config
    return {key: value for key, value in config.items() if key != "on_demand"}


def main(argv=None):
    """Run the broker of `--deploy` until interrupted, with its pid and port in
    `<state>/broker.json` while it runs."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deploy", required=True)
    args = parser.parse_args(argv)
    deploy = deploys.load_deploy(args.deploy)

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    # Started from a background job, a process inherits SIGINT ignored; and SIGTERM
    # should run the cleanup below too.
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    broker = Broker(deploy, log=lambda message: print(f"{time.strftime('%F %T')} {message}", flush=True))
    server = broker.serve()
    record = Path(deploy.state) / "broker.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({"pid": os.getpid(), "port": server.server_port}))
    print(f"broker for deploy {deploy.name} on port {server.server_port}: {broker.on_demand()}", flush=True)
    try:
        server.serve_forever()
    finally:
        broker.stopped = True
        record.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
