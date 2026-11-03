"""Durable history records and offline integrity checks; never resubmit robot actions."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from ..runtime.io_utils import atomic, read_json


def file_record(path):
    digest = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
            size += len(chunk)
    return {'bytes': size, 'sha256': digest.hexdigest()}


class HistoryJournal:
    """One immutable, fsynced record per message/event; survives a missing final snapshot."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if any(self.directory.iterdir()):
            raise RuntimeError('Existing history journal; use a new session, never overwrite an interrupted run')
        self.index = 0

    def append(self, kind, value):
        record = {'index': self.index, 'kind': kind, **value}
        path = self.directory/f'{self.index:08d}.json'
        temporary = path.with_suffix('.pending')
        with temporary.open('xb') as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False).encode())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        # Persist the rename as well as its content on filesystems supporting fsync.
        fd = os.open(self.directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        self.index += 1


def read_history_journal(directory):
    """Reject gaps; .pending files are uncommitted records, not complete messages."""
    messages, rounds = [], []
    for index, path in enumerate(sorted(Path(directory).glob('*.json'))):
        record = json.loads(path.read_text())
        if path.name != f'{index:08d}.json' or record['index'] != index:
            raise ValueError('history journal sequence gap')
        if record['kind'] == 'message':
            if record['message_id'] != len(messages):
                raise ValueError('history message sequence gap')
            messages.append(record['message'])
            rounds.append(record['round_id'])
    return messages, rounds


def finalize_attempt(folder):
    """Fingerprint closed artifacts even on errors/cancellation; no claim of unseen bytes."""
    folder = Path(folder)
    http = read_json(folder/'http.json', {})
    worker = read_json(folder/'transport_worker.json', {})
    wire = next((folder/name for name in ('response.sse', 'response.wire')
                 if (folder/name).exists()), None)
    complete = http.get('body_complete', http.get('response_complete')) is True
    status = ('complete' if complete else 'partial' if wire and wire.stat().st_size
              else 'no_response_received' if worker else 'not_started')
    files = {p.name: file_record(p) for p in sorted(folder.iterdir())
             if p.is_file() and p.name != 'archive_manifest.json'}
    atomic(folder/'archive_manifest.json', {
        'version': 1, 'response_status': status, 'files': files,
        'complete_json_response': http.get('response_complete') is True,
        'request_matches_worker': (files.get('request.json', {}).get('sha256') == worker['request_sha256']
                                   if worker.get('request_sha256') else None),
        'credential_redaction': 'Only an echoed transport credential is masked; see wire_capture.json.',
        'note': 'request.json is the exact submitted body. Partial/no-response records do not imply '
                'that the provider did not process the request. Replaying inputs does not guarantee identical output.'})


def verify_attempt(folder):
    folder = Path(folder)
    manifest = json.loads((folder/'archive_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if Path(name).name != name:
            raise ValueError('archive manifest requires local filenames')
        path = folder/name
        if not path.is_file() or file_record(path) != expected:
            raise ValueError('archive integrity mismatch: '+str(path))
    return manifest
