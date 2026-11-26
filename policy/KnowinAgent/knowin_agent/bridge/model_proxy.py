"""Allow-list CONNECT proxy for contained agents: only the model endpoints, through the host's upstream proxy.

Same policy as the H4/H5 L20 runtime (setup/l20_codex_runtime.py): a random Basic credential per proxy, CONNECT
only to ALLOWED_HOSTS:443, no direct fallback, idle upstream legs closed. The agent container's only network
route is this proxy, so a shell command inside it cannot reach anything but the model API.
"""

from __future__ import annotations

import base64
import hmac
import secrets
import select
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ALLOWED_HOSTS = {"api.openai.com", "auth.openai.com", "chatgpt.com", "ab.chatgpt.com", "wham.chatgpt.com"}
UPSTREAM_SILENCE_SEC = 300
RELAY_PORT = 3128

# A plain TCP relay (stdlib only) run in a small container attached to both the agents' internal network and
# an outbound one. Agents on an --internal network have no route anywhere else; the relay only forwards to
# the host's ModelProxy, which keeps the credential check and the host allow-list.
RELAY_SCRIPT = r"""
import socket, sys, threading
target = (sys.argv[1], int(sys.argv[2]))
server = socket.socket()
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("0.0.0.0", int(sys.argv[3])))
server.listen(64)
def pump(src, dst):
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        for s in (src, dst):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
def serve(client):
    try:
        upstream = socket.create_connection(target, timeout=10)
    except OSError:
        client.close()
        return
    upstream.settimeout(None)
    threading.Thread(target=pump, args=(client, upstream), daemon=True).start()
    pump(upstream, client)
while True:
    conn, _ = server.accept()
    threading.Thread(target=serve, args=(conn,), daemon=True).start()
"""


def start_relay(name: str, *, image: str, internal: str, outbound: str, target: tuple[str, int],
                labels: dict | None = None) -> str:
    """Start the relay container `name` on `outbound`, attach it to `internal`; returns its host:port there."""
    import subprocess

    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    label_args = [x for key, value in (labels or {}).items() for x in ("--label", f"{key}={value}")]
    started = subprocess.run(
        ["docker", "run", "-d", "--rm", "--name", name, *label_args, "--network", outbound, "--read-only",
         "--cap-drop", "ALL",
         "--security-opt", "no-new-privileges", "--pids-limit", "64", "--memory", "256m", "--cpus", "1",
         image, "python3", "-c", RELAY_SCRIPT, target[0], str(target[1]), str(RELAY_PORT)],
        capture_output=True, text=True)
    if started.returncode != 0:
        raise RuntimeError(f"cannot start relay {name}: {started.stderr.strip()[-300:]}")
    attached = subprocess.run(["docker", "network", "connect", internal, name], capture_output=True, text=True)
    if attached.returncode != 0:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        raise RuntimeError(f"cannot attach relay {name} to {internal}: {attached.stderr.strip()[-300:]}")
    return f"{name}:{RELAY_PORT}"


def stop_relay(name: str) -> None:
    import subprocess

    subprocess.run(["docker", "rm", "-f", name], capture_output=True)


class ModelProxy(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, bind: str, upstream: tuple[str, int] = ("127.0.0.1", 17890), allowed=ALLOWED_HOSTS):
        self.upstream = upstream
        self.allowed = {f"{host}:443" for host in allowed}
        token = secrets.token_urlsafe(32)
        self.authorization = "Basic " + base64.b64encode(f"codex:{token}".encode()).decode()
        super().__init__((bind, 0), _Tunnel)
        self.url = f"http://codex:{token}@{bind}:{self.server_port}"
        threading.Thread(target=self.serve_forever, daemon=True).start()


class _Tunnel(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass  # never log credentials or URLs

    def reject(self, status: int) -> None:
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def do_CONNECT(self):
        self.close_connection = True
        if not hmac.compare_digest(self.headers.get("Proxy-Authorization", ""), self.server.authorization):
            return self.reject(407)
        if self.path not in self.server.allowed:
            return self.reject(403)
        try:
            upstream = socket.create_connection(self.server.upstream, timeout=10)
        except OSError:
            return self.reject(502)
        with upstream:
            upstream.sendall(f"CONNECT {self.path} HTTP/1.1\r\nHost: {self.path}\r\n\r\n".encode("ascii"))
            header = bytearray()
            while not header.endswith(b"\r\n\r\n") and len(header) < 16384:
                chunk = upstream.recv(1)
                if not chunk:
                    break
                header.extend(chunk)
            if not bytes(header).split(b"\r\n", 1)[0].startswith(b"HTTP/1.1 200"):
                return self.reject(502)
            self.send_response(200, "Connection established")
            self.end_headers()
            upstream.settimeout(None)
            sockets = [self.connection, upstream]
            last_down = time.monotonic()
            while True:
                ready, _, _ = select.select(sockets, [], [], 30)
                if time.monotonic() - last_down > UPSTREAM_SILENCE_SEC:
                    return
                for current in ready:
                    data = current.recv(65536)
                    if not data:
                        return
                    if current is upstream:
                        last_down = time.monotonic()
                    (upstream if current is self.connection else self.connection).sendall(data)

    def do_GET(self):
        return self.reject(405)

    def do_POST(self):
        return self.reject(405)
