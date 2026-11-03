"""A serial worker keeps all simulator operations on the serving thread."""

from http.server import BaseHTTPRequestHandler, HTTPServer
import hmac
import json

from .contracts import Action


class WorkerAPI:
    """The worker's RPC methods over one environment: describe, reset, execute and call.
    It tracks the current observation so a stale or replayed action is never executed, and
    passes results through the observation `contract` when one is given."""

    def __init__(self, environment, contract=None):
        self.environment = environment
        # What the clients may see (core/contract.py); None reports everything.
        self.contract = contract
        self.current = None
        self.done = False

    def _filtered(self, method, arguments, result):
        return self.contract.apply(method, arguments, result) if self.contract is not None else result

    def dispatch(self, method: str, arguments: dict):
        """The result of one RPC: `describe` (protocol version, name, tools, methods),
        `reset`, `execute` (only on the current episode and step) or `call` (one of the
        environment's own methods). An unknown method is an error."""
        if method == "describe":
            return {"protocol_version": 1, "name": self.environment.name,
                    "tools": self.environment.tools(), "snapshot_restore": False,
                    "methods": sorted(getattr(self.environment, "methods", ()))}
        if method == "call":
            # Environment-specific extensions (e.g. per-tick control for the
            # runtime). The environment enforces its own episode/step checks.
            name = arguments["name"]
            if name not in getattr(self.environment, "methods", ()):
                raise ValueError(f"Unknown environment method: {name}")
            inner = arguments.get("arguments", {})
            return self._filtered(name, inner, self.environment.call(name, inner))
        if method == "reset":
            self.current = self.environment.reset(**arguments)
            self.done = False
            return self._filtered("reset", arguments, self.current.to_dict())
        if method == "execute":
            if self.current is None or self.done:
                raise ValueError("Reset the worker before executing an action")
            if (arguments["episode_id"], arguments["step"]) != (
                self.current.episode_id, self.current.step,
            ):
                raise ValueError("Stale observation; action was not executed")
            action = Action(**arguments["action"])
            # Mark this observation consumed before entering the simulator. If it
            # throws after applying an action, reset is required; never replay it.
            previous = self.current
            self.current = None
            transition = self.environment.execute(action, previous.episode_id, previous.step)
            self.current = transition.observation
            self.done = transition.terminated or transition.truncated
            return transition.to_dict()
        raise ValueError(f"Unknown worker method: {method}")


def create_server(environment, host="127.0.0.1", port=8765, token=None, contract=None):
    """An HTTP server (not yet serving) that answers POST /rpc with WorkerAPI, one request
    at a time. With a `token` (required on a non-loopback host) requests must carry it as a
    bearer token; errors come back as {"error": ...} with status 400."""
    if host not in ("127.0.0.1", "localhost", "::1") and not token:
        raise ValueError("Set HARNESS_WORKER_TOKEN before binding a non-loopback interface")
    api = WorkerAPI(environment, contract)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            if self.path != "/rpc":
                self.send_error(404)
                return
            if token and not hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {token}"):
                self.send_error(401)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 32 * 1024 * 1024:
                    raise ValueError("Invalid request size")
                request = json.loads(self.rfile.read(size))
                result = {"result": api.dispatch(request["method"], request.get("arguments", {}))}
                status = 200
            except Exception as error:
                result = {"error": f"{type(error).__name__}: {error}"}
                status = 400
            body = json.dumps(result, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return HTTPServer((host, port), Handler)
