"""Cancellable HTTP/SSE transport. Only a complete message becomes response.json."""
import copy
import hashlib
import json
import os
import sys
import time
import urllib.request
import urllib.error
import ssl
import certifi
from pathlib import Path


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False))
    temporary.replace(path)


class StreamError(RuntimeError):
    def __init__(self, kind, message):
        self.kind = kind
        super().__init__(message)


class MessageStream:
    """Reconstruct wire blocks without changing signed thinking content."""
    def __init__(self):
        self.message = None
        self.blocks = {}
        self.json_parts = {}
        self.invalid_tool_inputs = {}
        self.closed = set()
        self.complete = False

    def feed(self, event):
        kind = event.get('type')
        if kind == 'ping':
            return
        if kind == 'error':
            error = event.get('error') or {}
            raise StreamError(error.get('type', 'stream_error'), error.get('message', 'stream error'))
        if kind == 'message_start':
            if self.message is not None:
                raise ValueError('duplicate message_start')
            self.message = copy.deepcopy(event['message'])
            self.message['content'] = []
            return
        if self.message is None:
            raise ValueError('stream event before message_start')
        if kind == 'content_block_start':
            index = event['index']
            if index in self.blocks:
                raise ValueError('duplicate block index')
            self.blocks[index] = copy.deepcopy(event['content_block'])
        elif kind == 'content_block_delta':
            index, delta = event['index'], event['delta']
            if index in self.closed:
                raise ValueError('delta after block closed')
            block = self.blocks[index]
            dtype = delta.get('type')
            fields = {'text_delta': 'text', 'thinking_delta': 'thinking', 'signature_delta': 'signature'}
            if dtype in fields:
                field = fields[dtype]
                block[field] = block.get(field, '') + delta[field]
            elif dtype == 'input_json_delta':
                self.json_parts[index] = self.json_parts.get(index, '') + delta['partial_json']
                if self.json_parts[index].strip():
                    # The initial {} is a placeholder, not the generated tool input.
                    block.pop('input', None)
            elif dtype == 'citations_delta':
                block.setdefault('citations', []).append(delta['citation'])
            else:
                raise ValueError('unsupported content delta: '+str(dtype))
        elif kind == 'content_block_stop':
            index = event['index']
            if index not in self.blocks or index in self.closed:
                raise ValueError('invalid block stop')
            # Empty deltas on a no-argument tool preserve its initial input {}.
            if index in self.json_parts and self.json_parts[index].strip():
                try:
                    self.blocks[index]['input'] = json.loads(self.json_parts[index])
                except ValueError as error:
                    # Keep reading so a trailing max_tokens stop and usage survive.
                    self.invalid_tool_inputs[index] = str(error)
            self.closed.add(index)
        elif kind == 'message_delta':
            self.message.update(event.get('delta', {}))
            # Usage values are cumulative, not per-chunk increments.
            self.message.setdefault('usage', {}).update(event.get('usage', {}))
        elif kind == 'message_stop':
            if set(self.blocks) != self.closed or not self.message.get('stop_reason'):
                raise StreamError('generation_incomplete', 'message stopped before all blocks and stop reason')
            if self.invalid_tool_inputs:
                raise StreamError('generation_incomplete', 'message stopped with invalid tool JSON')
            if sorted(self.blocks) != list(range(len(self.blocks))):
                raise ValueError('noncontiguous stream blocks')
            self.complete = True
        # Unknown envelope events stay in raw SSE; unknown content deltas fail closed.

    def generation_incomplete(self):
        return bool(self.invalid_tool_inputs or
                    (self.message or {}).get('stop_reason') == 'max_tokens')

    def snapshot(self):
        value = copy.deepcopy(self.message or {})
        value['content'] = [copy.deepcopy(self.blocks[i]) for i in sorted(self.blocks)]
        return value


class WireCapture:
    """Write each received chunk immediately; mask echoed credentials across chunk boundaries."""
    def __init__(self, path, credential=''):
        self.path, self.key = path, credential.encode()
        self.raw = path.open('w+b', buffering=0)
        self.original_tail = self.saved_tail = b''
        self.received, self.redactions = 0, 0
        self.digest = hashlib.sha256()

    def append(self, chunk):
        self.digest.update(chunk)
        self.received += len(chunk)
        original = self.original_tail + chunk
        output = bytearray(self.saved_tail + chunk)
        if self.key:
            start = original.find(self.key)
            while start >= 0:
                output[start:start+len(self.key)] = b'*' * len(self.key)
                self.redactions += 1
                start = original.find(self.key, start+len(self.key))
        if self.saved_tail:
            self.raw.seek(-len(self.saved_tail), os.SEEK_CUR)
        remaining = memoryview(output)
        while remaining:
            written = self.raw.write(remaining)
            if not written:
                raise OSError('short archive write')
            remaining = remaining[written:]
        keep = max(0, len(self.key)-1)
        self.original_tail = original[-keep:] if keep else b''
        self.saved_tail = bytes(output[-keep:]) if keep else b''
        # A killed worker still leaves both received bytes and an explicit incomplete record.
        save(self.path.parent/'wire_capture.json', {
            'file': self.path.name, 'received_bytes': self.received,
            'received_sha256': self.digest.hexdigest(), 'credential_redactions': self.redactions})

    def close(self):
        self.raw.flush()
        os.fsync(self.raw.fileno())
        self.raw.close()


def received_chunks(response, capture):
    # read1 returns available data without waiting for a full body or an SSE newline.
    # HTTPResponse and BytesIO both implement it.
    while True:
        chunk = response.read1(16384)
        if not chunk:
            return
        capture.append(chunk)
        yield chunk


def consume_sse(response, directory, redact, credential=''):
    parser, data, count, wire_bytes = MessageStream(), [], 0, 0
    capture = WireCapture(directory/'response.sse', credential)
    try:
        def dispatch(eof=False):
            nonlocal count
            if data:
                try:
                    event = json.loads('\n'.join(data))
                except ValueError as error:
                    if eof:
                        raise StreamError('incomplete_stream', 'EOF during SSE event') from error
                    raise
                parser.feed(event)
                count += 1
                data.clear()
                save(directory/'partial_response.json', json.loads(redact(
                    json.dumps(parser.snapshot(), ensure_ascii=False).encode())))
                save(directory/'progress.json', {'events': count, 'wire_bytes': wire_bytes,
                     'last_event_unix_s': time.time(), 'message_complete': parser.complete})
        try:
            pending = b''
            for chunk in received_chunks(response, capture):
                wire_bytes += len(chunk)
                pending += chunk
                while b'\n' in pending:
                    line, pending = pending.split(b'\n', 1)
                    text = line.decode('utf-8').rstrip('\r')
                    if not text:
                        dispatch()
                        if parser.complete:
                            return parser.snapshot()
                    elif text.startswith('data:'):
                        data.append(text[5:].lstrip(' '))
            if pending.startswith(b'data:'):
                try:
                    data.append(pending[5:].decode('utf-8').lstrip(' ').rstrip('\r'))
                except UnicodeDecodeError as error:
                    raise StreamError('incomplete_stream', 'EOF during SSE UTF-8 character') from error
            dispatch(eof=True)
            if not parser.complete:
                raise StreamError('incomplete_stream', 'EOF before message_stop')
            return parser.snapshot()
        except Exception as error:
            if parser.generation_incomplete() and not parser.complete:
                raise StreamError('generation_incomplete', 'incomplete generated response: '+str(error)) from error
            raise
        finally:
            save(directory/'partial_response.json', json.loads(redact(
                json.dumps(parser.snapshot(), ensure_ascii=False).encode())))
            if parser.json_parts:
                save(directory/'partial_tool_inputs.json', json.loads(redact(
                    json.dumps(parser.json_parts, ensure_ascii=False).encode())))
    finally:
        capture.close()


def consume_body(response, directory, credential):
    capture = WireCapture(directory/'response.wire', credential)
    parts = []
    try:
        for chunk in received_chunks(response, capture):
            parts.append(chunk)
        body = b''.join(parts)
        length = response.headers.get('Content-Length')
        if length is not None and len(body) != int(length):
            raise StreamError('incomplete_body', 'EOF before Content-Length bytes arrived')
        return body
    finally:
        capture.close()


def send_once(directory, config, key, request_bytes):
    redact = lambda data: data.replace(key.encode(), b'[REDACTED]')
    headers = dict(config.get('request_headers', {'Content-Type': 'application/json'}))
    headers['Authorization'] = 'Bearer '+key
    if config['provider'] == 'anthropic':
        headers['anthropic-version'] = '2023-06-01'
    request = urllib.request.Request(config['url'], data=request_bytes, headers=headers)
    started = time.monotonic()
    status, body, response_headers, error, stream_error, mode = None, b'', {}, None, None, None
    complete = body_complete = False
    def http_state():
        return {'http_status': status, 'seconds': time.monotonic()-started,
            'headers': response_headers, 'error': error, 'stream_error_type': stream_error,
            'transport_mode': mode, 'body_complete': body_complete, 'response_complete': complete}
    try:
        try:
            r = urllib.request.urlopen(request, timeout=config.get('idle_timeout', config['timeout']),
                                       context=ssl.create_default_context(cafile=certifi.where()))
        except urllib.error.HTTPError as e:
            r = e  # Error bodies use the same incremental archival path.
        with r:
            status = r.status
            response_headers = {k: v.replace(key, '[REDACTED]') for k, v in r.headers.items()
                                if k.lower() in ('request-id','x-request-id','retry-after','content-type','content-length')}
            mode = ('sse' if config['provider'] == 'anthropic' and
                    'text/event-stream' in r.headers.get('Content-Type', '') else 'json')
            save(directory/'http.json', http_state())
            if mode == 'sse':
                result = consume_sse(r, directory, redact, key)
                body = json.dumps(result, ensure_ascii=False).encode()
            else:
                body = consume_body(r, directory, key)
            # Preserve complete non-JSON error bodies as raw wire, never as fake JSON.
            body_complete = True
            json.loads(body)
            temporary = directory/'response.json.tmp'
            temporary.write_bytes(redact(body))
            temporary.replace(directory/'response.json')
            complete = True
    except Exception as e:
        error = str(e).replace(key, '[REDACTED]')
        stream_error = getattr(e, 'kind', None)
    # Partial bodies are retained on disk but never accepted as complete model responses.
    http = http_state()
    save(directory/'http.json', http)
    return http


def main():
    # Exactly one wire request per worker. Retry/accounting/admission belong to API.
    directory = Path(sys.argv[1]).resolve()
    config = json.loads((directory/'transport.json').read_text())
    key = os.environ.get(config['key_env'], '')
    if not key:
        raise RuntimeError('API credential missing')
    body = (directory/'request.json').read_bytes()
    save(directory/'transport_worker.json', {
        'format': 'incremental-wire-archive', 'started_unix_s': time.time(),
        'request_sha256': hashlib.sha256(body).hexdigest()})
    send_once(directory, config, key, body)


if __name__ == '__main__':
    main()
