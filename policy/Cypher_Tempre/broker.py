"""Narrow HTTP bridge for official remote clients; no file-read or shell RPC."""
from __future__ import annotations
import argparse
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import re
import threading
from .client import ContinualClient


class Broker(ThreadingHTTPServer):
    def __init__(self, address, env, token):
        super().__init__(address, Handler)
        self.env = dict(env)
        self.token = token
        self.clients = {}
        self.active = None
        self.guard = threading.Lock()
        self.busy = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Never log headers, tokens, prompts or camera payloads.
        return

    def respond(self, code, value):
        data = json.dumps(value, allow_nan=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        supplied = self.headers.get('Authorization', '')
        if not hmac.compare_digest(supplied, 'Bearer ' + self.server.token):
            self.respond(401, {'error': 'unauthorized'})
            return
        match = re.fullmatch(r'/sessions/([0-9a-f]{32})(?:/(complete|finish))?', self.path)
        if not match:
            self.respond(404, {'error': 'unknown endpoint'})
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            self.respond(400, {'error': 'invalid body length'})
            return
        if not 0 < length <= 32 * 1024 * 1024:
            self.respond(413, {'error': 'body outside permitted size'})
            return
        session, operation = match.groups()
        try:
            value = json.loads(self.rfile.read(length))
            if not isinstance(value, dict):
                raise ValueError('Expected object')
            if operation is None:
                with self.server.guard:
                    if self.server.active is not None or session in self.server.clients:
                        self.respond(409, {'error': 'another session is active or this ID already exists'})
                        return
                    client = ContinualClient(self.server.env)
                    self.server.clients[session] = client
                    self.server.active = session
                self.respond(200, {'session': session, 'same_persistent_agent': True})
                return
            with self.server.guard:
                if self.server.active != session:
                    self.respond(409, {'error': 'session is not active'})
                    return
                client = self.server.clients[session]
            if not self.server.busy.acquire(blocking=False):
                self.respond(409, {'error': 'a controller operation is already in flight'})
                return
            try:
                if operation == 'complete':
                    if set(value) != {'messages', 'tools'} or not isinstance(value['messages'], list) or not isinstance(value['tools'], list):
                        raise ValueError('Invalid observation packet')
                    self.respond(200, client.complete(value['messages'], value['tools']))
                else:
                    client.finish()
                    with self.server.guard:
                        self.server.active = None
                    self.respond(200, {'session': session, 'ended': True, 'persistent_memory_retained': True})
            finally:
                self.server.busy.release()
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            self.respond(400, {'error': 'invalid packet'})
        except Exception as error:
            self.respond(500, {'error': 'controller transport failed', 'category': type(error).__name__})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8767)
    a = p.parse_args()
    token = os.environ.get('CYPHER_TEMPRE_BROKER_TOKEN', '')
    if len(token) < 32:
        p.error('Set CYPHER_TEMPRE_BROKER_TOKEN to at least32 random characters.')
    for key in ['CYPHER_TEMPRE_SKILL_ROOT', 'CYPHER_TEMPRE_STREAM_ROOT']:
        if not os.environ.get(key):
            p.error('Missing ' + key)
    server = Broker((a.host, a.port), os.environ, token)
    print(json.dumps({'status': 'listening', 'host': a.host, 'port': server.server_port,
                      'tls': 'Terminate HTTPS in a trusted reverse proxy for non-loopback access.'}), flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
