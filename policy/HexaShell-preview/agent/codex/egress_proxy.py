#!/usr/bin/env python3
"""Egress proxy: the only way out of the agent container. Standard library only.

Two listeners:
  MODEL_PORT  -> MODEL_UPSTREAM, path prefix MODEL_PREFIX only; the API key is
                 added here, so the agent container never holds it.
  ROBO_PORT   -> ROBO_UPSTREAM, agent routes of robo-server only (no /admin).

Responses are streamed, so server-sent events pass through unbuffered.
"""

import http.client
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
              "transfer-encoding", "upgrade", "host", "content-length"}


def log(message):
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


class Proxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    upstream = None  # urlsplit result
    allowed_prefixes = ()
    key_file = None
    timeout_s = 900

    def log_message(self, *_):
        pass

    def read_body(self):
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            body = b""
            while True:
                size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                if size == 0:
                    while self.rfile.readline().strip():
                        pass
                    return body
                body += self.rfile.read(size)
                self.rfile.readline()
        length = int(self.headers.get("Content-Length", 0) or 0)
        return self.rfile.read(length) if length else b""

    def refuse(self, code, text):
        body = text.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def forward(self):
        if not self.path.startswith(self.allowed_prefixes):
            log(f"DENY {self.command} {self.path.split('?')[0]}")
            return self.refuse(403, "path not allowed\n")
        body = self.read_body()
        dump_dir = os.environ.get("DUMP_DIR")
        if dump_dir and self.command == "POST":
            # audit aid: keep the first model request so the offered tools can be checked
            target = os.path.join(dump_dir, "first_request.bin")
            if not os.path.exists(target):
                with open(target, "wb") as handle:
                    handle.write(f"{self.headers.get('Content-Encoding', 'identity')}\n".encode() + body)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}
        if self.key_file:
            headers = {k: v for k, v in headers.items() if k.lower() not in ("authorization", "api-key", "x-api-key")}
            with open(self.key_file) as handle:
                headers["Authorization"] = "Bearer " + handle.read().strip()
        headers["Host"] = self.upstream.netloc
        connection = http.client.HTTPConnection(self.upstream.hostname, self.upstream.port or 80, timeout=self.timeout_s)
        try:
            connection.request(self.command, self.path, body=body or None, headers=headers)
            response = connection.getresponse()
        except OSError as err:
            log(f"FAIL {self.command} {self.path.split('?')[0]}: {err}")
            return self.refuse(502, f"upstream error: {err}\n")
        self.send_response(response.status, response.reason)
        for key, value in response.getheaders():
            if key.lower() not in HOP_BY_HOP:
                self.send_header(key, value)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        sent = 0
        try:
            while True:
                chunk = response.read1(65536)
                if not chunk:
                    break
                self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                self.wfile.flush()
                sent += len(chunk)
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except OSError as err:
            log(f"ABORT {self.path.split('?')[0]}: {err}")
            self.close_connection = True
        finally:
            connection.close()
        log(f"{response.status} {self.command} {self.path.split('?')[0]} in={len(body)} out={sent}")

    do_GET = do_POST = do_PUT = do_DELETE = forward


def serve(port, upstream, prefixes, key_file=None):
    handler = type("Handler", (Proxy,), {"upstream": urlsplit(upstream), "allowed_prefixes": tuple(prefixes),
                                          "key_file": key_file})
    server = ThreadingHTTPServer(("0.0.0.0", port), handler)
    server.daemon_threads = True
    log(f"listen :{port} -> {upstream} {list(prefixes)}")
    server.serve_forever()


def main():
    model = threading.Thread(target=serve, daemon=True, args=(
        int(os.environ.get("MODEL_PORT", "8080")),
        os.environ["MODEL_UPSTREAM"],
        [os.environ.get("MODEL_PREFIX", "/openai/")],
        os.environ.get("MODEL_KEY_FILE"),
    ))
    model.start()
    serve(int(os.environ.get("ROBO_PORT", "18700")), os.environ["ROBO_UPSTREAM"], ["/v1/cmd", "/v1/obs/", "/v1/tools"])


if __name__ == "__main__":
    main()
