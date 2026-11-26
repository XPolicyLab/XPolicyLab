"""Process-local replay ledger. Never evict execution evidence to admit work.

No disk durability is claimed: a different boot ID always invalidates recovery.
The private protocol is enabled on the internal network by default; clients
negotiate it over HELLO. A configured token adds an optional authentication gate.
"""

import hashlib
import math
import os
import struct
import time
from collections.abc import Mapping
from collections import deque

import numpy as np

PROTOCOL = "r2si-replay-v1"
_log_handler = None
_log_listener = None


def configure_logging():
    """Keep slow log sinks off the event loop; replay evidence lives in the ledger."""
    import logging
    from logging.handlers import QueueHandler, QueueListener
    import queue
    global _log_handler, _log_listener
    if _log_handler is not None:
        return
    class BoundedHandler(QueueHandler):
        dropped = 0
        def enqueue(self, record):
            try:
                self.queue.put_nowait(record)
            except queue.Full:
                self.dropped += 1
    pending = queue.Queue(maxsize=4096)
    sink = logging.StreamHandler()
    sink.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s'))
    _log_handler = BoundedHandler(pending)
    root = logging.getLogger()
    root.handlers = [_log_handler]
    root.setLevel(logging.INFO)
    _log_listener = QueueListener(pending, sink)
    _log_listener.start()  # Daemon: a stuck external sink must not pin shutdown.


def dropped_log_count():
    return _log_handler.dropped if _log_handler is not None else 0


def canonical(value):
    """Tagged, length-delimited encoding; see doc/WEBSOCKET_REPLAY.md."""
    def pack(tag, data):
        return tag + struct.pack(">Q", len(data)) + data
    if value is None:
        return b"n"
    if isinstance(value, (bool, np.bool_)):
        return b"t" if value else b"f"
    if isinstance(value, np.ndarray) or isinstance(value, np.generic):
        array = np.asarray(value)
        if array.dtype.hasobject or array.dtype.fields is not None:
            raise ValueError("Object/structured arrays are not supported")
        return pack(b"a", canonical(array.dtype.str) + canonical(list(array.shape))
                    + pack(b"b", array.tobytes(order="C")))
    if isinstance(value, int):
        return pack(b"i", str(value).encode("ascii"))
    if isinstance(value, float):
        return b"d" + struct.pack(">d", value)
    if isinstance(value, str):
        return pack(b"s", value.encode("utf-8"))
    if isinstance(value, (bytes, bytearray, memoryview)):
        return pack(b"b", bytes(value))
    if isinstance(value, (list, tuple)):
        return pack(b"l", b"".join(canonical(item) for item in value))
    if isinstance(value, Mapping):
        pairs = sorted((canonical(k), canonical(v)) for k, v in value.items())
        return pack(b"m", b"".join(k + v for k, v in pairs))
    raise ValueError("Unsupported fingerprint value: " + type(value).__name__)


def fingerprint(frame):
    envelope = frame.to_wire_dict()
    envelope.pop("sent_at", None)
    # The binding is transport metadata, checked independently on every send.
    envelope["payload"] = {k: v for k, v in frame.payload.items() if k != "_replay"}
    return hashlib.sha256(canonical(envelope)).hexdigest()


class ReplayError(Exception):
    def __init__(self, reason, *, admitted=False):
        self.reason, self.admitted = reason, admitted
        super().__init__(reason)


class Ledger:
    def __init__(self, *, retention=600, max_records=200000, max_bytes=512*1024*1024,
                 response_reserve=256*1024, max_pending=128, per_session_pending=16):
        if not math.isfinite(retention) or retention <= 0 or min(
                max_records, max_bytes, response_reserve, max_pending, per_session_pending) < 1:
            raise ValueError("Replay limits must be positive and finite")
        self.retention = retention
        self.max_records, self.max_bytes = max_records, max_bytes
        self.response_reserve = response_reserve
        self.max_pending, self.per_session_pending = max_pending, per_session_pending
        self.records = {}
        self.bytes = 0
        self.completed = deque()
        self.pending_count = 0
        self.pending_by_session = {}

    def expire(self):
        now = time.monotonic()
        while self.completed and now >= self.completed[0]['completed'] + self.retention:
            record = self.completed.popleft()
            if record['state'] == 'done':
                self.bytes -= record['charge'] - record['metadata_charge']
                record['charge'] = record['metadata_charge']
                record['response'] = None
                record['state'] = 'expired'

    def register(self, key, digest, input_bytes):
        self.expire()
        metadata = 1024 + sum(len(str(part).encode()) for part in key)
        # Budget decoded input, codec/canonical scratch and dispatch snapshot.
        charge = metadata + 3*input_bytes + self.response_reserve
        if (len(self.records) >= self.max_records or self.bytes + charge > self.max_bytes
                or self.pending_count >= self.max_pending
                or self.pending_by_session.get(key[:2], 0) >= self.per_session_pending):
            raise ReplayError('NOT_ADMITTED')
        record = dict(state='pending', digest=digest, response=None, task=None,
                      received=time.monotonic(), completed=None, charge=charge,
                      metadata_charge=metadata, input_bytes=input_bytes, key=key, counted=True)
        self.records[key] = record
        self.bytes += charge
        self.pending_count += 1
        self.pending_by_session[key[:2]] = self.pending_by_session.get(key[:2], 0) + 1
        return record

    def finish_pending(self, record):
        if record['counted']:
            record['counted'] = False
            self.pending_count -= 1
            key = record['key'][:2]
            self.pending_by_session[key] -= 1
            if not self.pending_by_session[key]:
                self.pending_by_session.pop(key)

    def complete(self, record, wire):
        # All supported action shapes are far below the reserved reply size.
        # An unexpected oversized/unencodable result is uncertain, never replayed.
        if len(wire) > self.response_reserve:
            raise ReplayError('REQUEST_UNCERTAIN', admitted=True)
        record['response'] = wire
        record['completed'] = time.monotonic()
        self.bytes -= record['charge'] - record['metadata_charge'] - len(wire)
        record['charge'] = record['metadata_charge'] + len(wire)
        record['state'] = 'done'
        self.finish_pending(record)
        self.completed.append(record)


def add_replay_arguments(parser):
    parser.add_argument('--disconnect-grace-seconds', type=float, default=180)
    parser.add_argument('--replay-retention-seconds', type=float, default=600)
    parser.add_argument('--replay-max-records', type=int, default=200000)
    parser.add_argument('--replay-max-mib', type=int, default=512)
    parser.add_argument('--replay-max-pending', type=int, default=128)
    parser.add_argument('--replay-per-session-pending', type=int, default=16)


def replay_options(args):
    if (not math.isfinite(args.disconnect_grace_seconds) or args.disconnect_grace_seconds <= 0
            or not math.isfinite(args.replay_retention_seconds)
            or args.replay_retention_seconds < max(600, args.disconnect_grace_seconds)):
        raise ValueError('Replay retention must be >=600s and cover the finite positive disconnect grace')
    return dict(disconnect_grace=args.disconnect_grace_seconds,
                replay_token=os.environ.get('POLICY_REPLAY_TOKEN'),
                ledger=Ledger(retention=args.replay_retention_seconds,
                              max_records=args.replay_max_records,
                              max_bytes=args.replay_max_mib*1024*1024,
                              max_pending=args.replay_max_pending,
                              per_session_pending=args.replay_per_session_pending))


def authenticate_inspection(client):
    """Authenticate existing read-only launcher's identity checks, if configured."""
    token = os.environ.get('POLICY_REPLAY_TOKEN')
    if token:
        from client_server.ws.protocol.messages import MessageType
        response = client._run(client._client.request(MessageType.HELLO, {
            'replay': {'protocol': PROTOCOL, 'token': token}}))
        if response.payload.get('replay_protocol') != PROTOCOL:
            raise RuntimeError('Endpoint has no authenticated replay capability')


def replay_settings(options):
    """Non-secret startup receipt; never serialize the options/token directly."""
    ledger = options['ledger']
    return dict(protocol=PROTOCOL, recovery_enabled=True, token_required=bool(options['replay_token']),
                access_mode='token' if options['replay_token'] else 'internal_network',
                durability='process_memory', disconnect_grace_s=options['disconnect_grace'],
                retention_s=ledger.retention, max_records=ledger.max_records, max_bytes=ledger.max_bytes,
                max_pending=ledger.max_pending, per_session_pending=ledger.per_session_pending)
