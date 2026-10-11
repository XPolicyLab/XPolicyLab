"""HTTP interface of robo-server: the agent routes and the operator routes, on two ports."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from roboshell.contract import v0 as C


def start_http(session, host, port, admin_port):
    """session.submit(kind, payload) -> reply dict; session.obs_cache() -> {file name: bytes}."""

    class Handler(BaseHTTPRequestHandler):
        admin = False

        def log_message(self, *_):
            pass

        def reply(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def payload(self):
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                return json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                return None

        def do_GET(self):
            if self.admin and self.path == C.ROUTE_ADMIN_RESULT:
                return self.reply(session.submit("result", {}))
            if not self.admin and self.path == C.ROUTE_TOOLS:
                return self.reply(session.tool_schema())
            if not self.admin and self.path.startswith(C.ROUTE_OBS):
                name = self.path.rsplit("/", 1)[-1]
                data = session.obs_cache().get(name)
                if data is not None and name in C.OBS_FILES + C.OBS_DEPTH_FILES:
                    self.send_response(200)
                    kind = "application/json" if name.endswith(".json") else "application/octet-stream" if name.endswith(".npy") else "image/png"
                    self.send_header("Content-Type", kind)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    return self.wfile.write(data)
            self.reply({"error": "not found"}, 404)

        def do_POST(self):
            payload = self.payload()
            if payload is None:
                return self.reply({"error": "invalid JSON", "exit_code": C.EXIT_BAD_ARGS}, 400)
            if self.admin and self.path == C.ROUTE_ADMIN_RESET:
                return self.reply(session.submit("reset", payload))
            if self.admin and self.path == C.ROUTE_ADMIN_REPLAY:
                return self.reply(session.submit("replay", payload))
            if self.admin and self.path == C.ROUTE_ADMIN_FINISH:
                return self.reply(session.submit("finish", payload))
            if self.admin and self.path == C.ROUTE_ADMIN_SHUTDOWN:
                return self.reply(session.submit("shutdown", payload))
            if not self.admin and self.path == C.ROUTE_CMD:
                payload = {k: v for k, v in payload.items() if not k.startswith("_")}
                return self.reply(session.submit("cmd", payload))
            self.reply({"error": "not found"}, 404)

    class AdminHandler(Handler):
        admin = True

    servers = []
    for bind_host, bind_port, handler in ((host, port, Handler), ("127.0.0.1", admin_port, AdminHandler)):
        if bind_port is None:
            continue
        http = ThreadingHTTPServer((bind_host, bind_port), handler)
        http.daemon_threads = True
        threading.Thread(target=http.serve_forever, daemon=True).start()
        servers.append(http)
    return servers


def dispatch(session, episode_getter, kind, payload):
    """Shared handling of agent commands; reset/replay/result belong to the session."""
    episode = episode_getter()
    cmd = payload.get("cmd")
    if episode is None:
        return {"error": "no episode has been started", "exit_code": C.EXIT_EPISODE_OVER}
    if cmd == "status":
        return dict(episode.last_feedback or {"note": "no command yet"}, exit_code=C.EXIT_OK)
    if episode.over:
        return {"error": "the episode is over", "episode_over": True, "exit_code": C.EXIT_EPISODE_OVER}
    if cmd not in C.BUDGETED + C.FREE and cmd not in episode.registry:
        return {"error": f"unknown command {cmd!r}", "exit_code": C.EXIT_BAD_ARGS}
    return episode.execute(payload)
