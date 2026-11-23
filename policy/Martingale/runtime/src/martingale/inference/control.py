# Copyright © 2026 Martingale. All Rights Reserved.
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import tempfile
import time
import uuid


PROVIDER_INTERRUPTS = frozenset({
    'provider_quota', 'provider_auth', 'provider_network', 'provider_error',
    'account_busy',
})


def atomic_json(path, value):
    """Publish a complete private status document, including concurrent writers."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def process_identity(pid=None):
    """A PID alone can identify a different process after host/process loss."""
    pid = os.getpid() if pid is None else int(pid)
    try:
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        start = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError, ValueError) as exc:
        raise RuntimeError('Cannot verify the live Linux process identity') from exc
    return {'hostname': socket.gethostname(), 'boot_id': boot, 'pid': pid,
            'process_start': start}


def classify_provider_failure(faults, stderr=''):
    """Recognize exhausted quota, not generic 429/auth/transport failures."""
    messages = []
    for fault in faults:
        if isinstance(fault, str):
            messages.append(fault)
        elif isinstance(fault, dict):
            messages.append(json.dumps(fault, ensure_ascii=False))
    message = '\n'.join(messages + [stderr]).lower().replace('\u2019', "'")
    if 'thread_identity_mismatch' in message:
        return 'provider_protocol'
    if any(marker in message for marker in (
        'usage_limit_reached', 'insufficient_quota', 'usage limit reached',
        "you've hit your usage limit", 'you have hit your usage limit',
    )):
        return 'provider_quota'
    if any(marker in message for marker in (
        'unauthorized', 'authentication', 'invalid_token', 'token_expired',
        'refresh_token', '401', '403', 'login required', 'not logged in',
    )):
        return 'provider_auth'
    if any(marker in message for marker in (
        'connection', 'network', 'timed out', 'timeout', 'dns', 'tls',
        'stream disconnected', '502', '503', '504',
    )):
        return 'provider_network'
    return 'provider_error'


def read_status(episode, *, require_live=False):
    episode = Path(episode).expanduser().resolve()
    try:
        status = json.loads((episode / 'status.json').read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError('Cannot read episode status.json') from exc
    if not isinstance(status, dict):
        raise RuntimeError('Invalid episode status')
    if require_live:
        recorded = status.get('process_identity')
        if not isinstance(recorded, dict) or recorded != process_identity(status.get('pid')):
            raise RuntimeError('Episode belongs to another host or process; live resume is unavailable')
        os.kill(status['pid'], 0)
    return status


def request_resume(episode, account=None, registry_path=None):
    """Queue one compare-and-checked request for a particular pause of a live session.

    The live session checks everything again before applying it. The CLI never
    changes credentials, replaces a worker, or executes a robot action.
    """
    episode = Path(episode).expanduser().resolve()
    status = read_status(episode, require_live=True)
    if status.get('status') != 'interrupted' or not status.get('pause_id'):
        raise RuntimeError('Only an interrupted live episode can resume')
    if not status.get('resume_allowed', False):
        raise RuntimeError('Uncertain cell execution requires operator reconciliation before resume')
    if registry_path is not None:
        configured = status.get('account_registry')
        if not configured or Path(registry_path).resolve() != Path(configured).resolve():
            raise ValueError('Resume must use the registry already assigned to this episode')
    if account is not None:
        raise RuntimeError('Mid-episode account handoff is disabled')
    request = {
        'version': 1, 'request_id': uuid.uuid4().hex, 'pause_id': status['pause_id'],
        'thread_id': status.get('thread_id'), 'account': account,
        'process_identity': status['process_identity'], 'requested_at': time.time(),
    }
    fd, temporary = tempfile.mkstemp(prefix='.resume-', dir=episode)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(request, stream)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, episode / 'resume.request')
        except FileExistsError as exc:
            raise RuntimeError('A resume request is already pending') from exc
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {'request_id': request['request_id'], 'status': 'queued',
            'episode': str(episode), 'account': account,
            'note': 'Inspect status.json for acceptance; no actions were replayed.'}
