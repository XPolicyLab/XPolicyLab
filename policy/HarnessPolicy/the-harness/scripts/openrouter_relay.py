#!/usr/bin/env python3
"""Relay OpenRouter calls through a machine that holds the API key.

Run on the machine with the key (standard library only):

    OPENROUTER_API_KEY=... OPENROUTER_RELAY_TOKEN=<shared secret> \\
        python3 scripts/openrouter_relay.py --port 8850

and reach it from the machine running the harness over an SSH tunnel
(`ssh -N -L 8850:127.0.0.1:8850 <key machine>`). There, export the same token and
the relay URL; every OpenRouter model (the main model, helper VLM, verifier) then goes
through the relay with no change to the configuration files:

    export OPENROUTER_RELAY_TOKEN=<shared secret>
    export OPENROUTER_RELAY_URL=http://127.0.0.1:8850/api/v1
    python3 scripts/harness program <experiment>

See docs/how-to/openrouter-relay.md.

Each request must carry `Authorization: Bearer <relay token>`. The relay replaces
it with the OpenRouter key and forwards the request unchanged to
https://openrouter.ai/api/v1/<path>, then streams the reply back (SSE included) with
OpenRouter's status. Only paths under /api/v1/ are forwarded. It logs one line per
request (path, model, status, seconds), never prompts or keys. The relay listens on
127.0.0.1 by default: it speaks plain HTTP, so the token and prompts are only
protected by the tunnel. It does not retry; the harness client decides that.
"""

import argparse
import hmac
import json
import os
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PREFIX = "/api/v1/"
# Hop-by-hop or relay-owned request headers; everything else goes to OpenRouter as sent.
_DROPPED = {"authorization", "host", "connection", "content-length", "accept-encoding",
            "keep-alive", "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade"}
_REPLY_HEADERS = {"content-type", "x-generation-id", "retry-after"}


def make_handler(token, api_key, upstream="https://openrouter.ai", timeout=900.0):
    class Relay(BaseHTTPRequestHandler):
        # HTTP/1.0: the reply ends when the connection closes, so a stream needs no framing.
        protocol_version = "HTTP/1.0"

        def do_GET(self):
            self._relay("GET")

        def do_POST(self):
            self._relay("POST")

        def _relay(self, method):
            started = time.perf_counter()
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if not self.path.startswith(PREFIX):
                return self._error(404, "the relay forwards /api/v1/* only", started)
            sent = self.headers.get("Authorization", "")
            if not hmac.compare_digest(sent.encode(), f"Bearer {token}".encode()):
                return self._error(401, "wrong or missing relay token", started)
            headers = {name: value for name, value in self.headers.items() if name.lower() not in _DROPPED}
            headers["Authorization"] = f"Bearer {api_key}"
            request = urllib.request.Request(upstream + self.path, data=body if method == "POST" else None,
                                             headers=headers, method=method)
            try:
                reply = urllib.request.urlopen(request, timeout=timeout)
            except urllib.error.HTTPError as error:
                reply = error  # OpenRouter's own error status and body go back as they are
            except (urllib.error.URLError, OSError) as error:
                return self._error(502, f"OpenRouter unreachable: {error}", started, body)
            with reply:
                self.send_response(reply.status)
                for name, value in reply.headers.items():
                    if name.lower() in _REPLY_HEADERS:
                        self.send_header(name, value)
                self.end_headers()
                try:
                    while chunk := reply.read1(65536):
                        self.wfile.write(chunk)
                        self.wfile.flush()
                except OSError:
                    pass  # the client hung up, or OpenRouter dropped the stream mid-reply
            self._log(reply.status, started, body)

        def _error(self, status, message, started, body=b""):
            payload = json.dumps({"error": {"code": status, "message": message, "relay": True}}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            self._log(status, started, body)

        def _log(self, status, started, body):
            try:
                model = json.loads(body).get("model", "-") if body else "-"
            except (ValueError, AttributeError):
                model = "-"
            print(f"{time.strftime('%H:%M:%S')} {self.client_address[0]} {self.command} {self.path} "
                  f"{model} {status} {time.perf_counter() - started:.1f}s", file=sys.stderr, flush=True)

        def log_message(self, format, *args):
            pass  # _log writes the one line per request

    return Relay


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1",
                        help="listen address; 0.0.0.0 exposes plain HTTP to the network (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8850)
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY", help="variable holding the OpenRouter key")
    parser.add_argument("--token-env", default="OPENROUTER_RELAY_TOKEN",
                        help="variable holding the secret clients must send as their bearer token")
    parser.add_argument("--timeout", type=float, default=900.0, help="seconds to wait on OpenRouter")
    args = parser.parse_args()
    api_key, token = os.environ.get(args.api_key_env), os.environ.get(args.token_env)
    if not api_key:
        parser.error(f"set {args.api_key_env}")
    if not token or len(token) < 16:
        parser.error(f"set {args.token_env} to a secret of at least 16 characters "
                     "(e.g. python3 -c 'import secrets; print(secrets.token_urlsafe(32))')")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(token, api_key, timeout=args.timeout))
    print(f"relaying http://{args.host}:{args.port}{PREFIX} -> https://openrouter.ai{PREFIX}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
