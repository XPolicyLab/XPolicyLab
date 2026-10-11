"""Private local IPC: commit bytes before publishing a verified identity marker."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

def atomic(path, value):
    path = Path(path)
    data = json.dumps(value, ensure_ascii=False).encode('utf-8')
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)
    return data

def commit(directory, packet):
    directory = Path(directory)
    if (directory/'accepted.json').exists() or (directory/'response.ready.json').exists():
        raise ValueError('Response already committed or accepted')
    data = atomic(directory/'response.json', packet)
    atomic(directory/'response.ready.json', {'request_id':packet['request_id'],
        'bytes':len(data), 'sha256':hashlib.sha256(data).hexdigest()})

def read_committed(directory, request_id):
    directory = Path(directory)
    ready = directory/'response.ready.json'
    if not ready.exists(): return None
    marker = json.loads(ready.read_bytes())
    data = (directory/'response.json').read_bytes()
    if marker.get('request_id') != request_id:
        raise ValueError('Commit request identity mismatch')
    if not data or len(data) != marker.get('bytes') or hashlib.sha256(data).hexdigest() != marker.get('sha256'):
        raise ValueError('Committed response integrity mismatch')
    packet = json.loads(data)
    if not isinstance(packet, dict) or packet.get('request_id') != request_id:
        raise ValueError('Response request identity mismatch')
    if not isinstance(packet.get('response'), dict):
        raise ValueError('Response must be an object')
    return packet
