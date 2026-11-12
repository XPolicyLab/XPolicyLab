"""Local authenticated RPC; simulation work stays on the simulator's main thread."""
import hashlib
import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class CommandServer:
    def __init__(self, token, port=0):
        self.token, self.jobs, self.records = token, queue.Queue(), {}
        self.lock = threading.RLock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_POST(self):
                try:
                    if self.headers.get('Authorization') != 'Bearer ' + outer.token:
                        self.send_error(403); return
                    size = int(self.headers.get('Content-Length', '0'))
                    if not 0 < size <= 2_000_000: raise ValueError('Invalid request size')
                    payload = json.loads(self.rfile.read(size))
                    result = outer.dispatch(self.path, payload)
                    data = json.dumps(result, allow_nan=False).encode()
                    self.send_response(200)
                except Exception as e:
                    data = json.dumps({'error': str(e)}).encode()
                    self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                try: self.wfile.write(data)
                except BrokenPipeError: pass

        self.http = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)

    def start(self):
        self.thread.start()
        return 'http://127.0.0.1:' + str(self.http.server_port)

    def dispatch(self, path, payload):
        cid = payload.get('id')
        if not isinstance(cid, str) or not 1 <= len(cid) <= 128: raise ValueError('Invalid command ID')
        with self.lock:
            if path == '/submit':
                digest = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
                if cid in self.records:
                    if digest != self.records[cid]['digest']: raise ValueError('ID reused with different payload')
                else:
                    self.records[cid] = {'id': cid, 'status': 'queued', 'digest': digest,
                        'cancel': threading.Event(), 'request': payload}
                    self.jobs.put(cid)
            elif path == '/cancel':
                if cid not in self.records: raise ValueError('Unknown command ID')
                self.records[cid]['cancel'].set()
            elif path != '/status': raise ValueError('Unknown endpoint')
            if cid not in self.records: raise ValueError('Unknown command ID; no command was submitted under this ID')
            return {k: v for k, v in self.records[cid].items() if k in ('id', 'status', 'result')}

    def next(self, timeout=.2):
        cid = self.jobs.get(timeout=timeout)
        record = self.records[cid]
        record['status'] = 'running'
        return record

    def complete(self, record, result, status='done'):
        with self.lock:
            record['result'], record['status'] = result, status

    def close(self):
        self.http.shutdown()
        self.http.server_close()
