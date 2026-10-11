"""Desktop transport for the unmodified RoboDawn agent prompt/command loop."""
import base64
import json
import os
from pathlib import Path
import time
import uuid
from datetime import datetime, timezone
from .vlm_client import VLMError

def atomic(path, value):
    atomic_bytes(path, json.dumps(value, ensure_ascii=False).encode('utf-8'))

def atomic_bytes(path, blob):
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with tmp.open('xb') as f:
        f.write(blob)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    fd = os.open(str(path.parent), os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)

class DesktopClient:
    model = 'codex-desktop-session'

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_id = os.environ.get('ROBODAWN_DESKTOP_RUN_ID', self.root.parent.name)
        lock = self.root / 'client.lock.json'
        # An existing lock is an unknown restart state, never silently resume.
        with lock.open('x', encoding='utf-8') as f:
            json.dump({'run_id': self.run_id, 'pid': os.getpid()}, f)
            f.flush(); os.fsync(f.fileno())
        # Small mutable control records need POSIX atomic replacement. Images and
        # large message/trajectory payloads remain in the configured media root.
        control = os.environ.get('ROBODAWN_DESKTOP_CONTROL_ROOT')
        self.control_root = Path(control) if control else self.root
        self.control_root.mkdir(parents=True, exist_ok=True)
        if self.control_root != self.root:
            (self.root / 'pending.json').symlink_to(self.control_root / 'pending.json')

    def complete(self, messages, *, deadline, **kwargs):
        started = time.monotonic()
        created_at = datetime.now(timezone.utc).isoformat()
        request_id = uuid.uuid4().hex
        directory = self.root / request_id
        directory.mkdir()
        control_directory = self.control_root / request_id
        if control_directory != directory:
            control_directory.mkdir()
            for name in ('telemetry.json', 'response.json', 'accepted.json'):
                (directory / name).symlink_to(control_directory / name)
        atomic(directory / 'messages.json', messages)
        readable = []
        text_chars = 0
        image_count = 0
        image_bytes = 0
        for mi, msg in enumerate(messages):
            readable.append(f'ROLE {msg["role"]}')
            content = msg['content']
            if isinstance(content, str):
                text_chars += len(content)
                readable.append(content)
                continue
            for pi, part in enumerate(content):
                if part.get('type') == 'text':
                    text_chars += len(part['text'])
                    readable.append(part['text'])
                elif part.get('type') == 'image_url':
                    url = part['image_url']['url']
                    if not url.startswith('data:image/'):
                        raise ValueError('Only embedded images are supported')
                    filename = f'm{mi:03d}_p{pi:03d}.jpg'
                    blob = base64.b64decode(url.split(',', 1)[1])
                    atomic_bytes(directory / filename, blob)
                    image_count += 1
                    image_bytes += len(blob)
                    readable.append(f'IMAGE {filename}')
        atomic_bytes(directory / 'prompt.txt', '\n\n'.join(readable).encode('utf-8'))
        metrics = {
            'schema_version': 2, 'run_id': self.run_id, 'request_id': request_id, 'model_transport': self.model,
            'created_at_utc': created_at, 'status': 'pending',
            'constructed_prompt_text_chars': text_chars,
            'constructed_prompt_image_count': image_count,
            'constructed_prompt_image_bytes': image_bytes,
            'actual_model_context_verified': False,
            'input_tokens': None, 'output_tokens': None, 'reasoning_tokens': None,
            'token_usage_source': 'unavailable_desktop_transport',
            'provider_inference_s': None, 'cost_usd': None,
            'timing_note': 'Bridge elapsed time includes file exchange, tools, and desktop waiting; not pure model inference.',
        }
        atomic(control_directory / 'telemetry.json', metrics)
        atomic(self.control_root / 'pending.json', {'run_id': self.run_id, 'request_id': request_id, 'directory': str(directory), 'response_directory': str(control_directory)})
        response = control_directory / 'response.json'
        while time.monotonic() < deadline:
            if response.exists():
                # Re-read only this response, for at most 1 second and never
                # beyond the original deadline. No action is resent.
                for bad_read in range(6):
                    try:
                        packet = json.loads(response.read_text(encoding='utf-8'))
                        break
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        if bad_read == 5 or time.monotonic() + .2 >= deadline:
                            metrics.update(status='invalid_response')
                            atomic(control_directory / 'telemetry.json', metrics)
                            raise VLMError('Malformed desktop response after bounded reread', retryable=False)
                        time.sleep(.2)
                if packet.get('run_id') != self.run_id:
                    raise ValueError('Stale desktop run identity')
                if (control_directory / 'accepted.json').exists():
                    raise ValueError('Response already consumed')
                if not isinstance(packet.get('response'), dict):
                    raise ValueError('Invalid desktop response payload')
                if packet.get('request_id') != request_id:
                    raise ValueError('Stale desktop response')
                text = json.dumps(packet['response'], ensure_ascii=False)
                elapsed = time.monotonic() - started
                metrics.update(status='accepted', accepted_at_utc=datetime.now(timezone.utc).isoformat(),
                               bridge_elapsed_s=elapsed, response_chars=len(text))
                atomic(control_directory / 'telemetry.json', metrics)
                atomic(control_directory / 'accepted.json', {'run_id': self.run_id, 'request_id': request_id, 'accepted_at_utc': metrics['accepted_at_utc']})
                return {'text': text, 'usage': {}, 'finish_reason': 'desktop_response',
                        'latency_s': elapsed, 'attempts': 1, 'retry_history': []}
            time.sleep(.2)
        metrics.update(status='timeout', ended_at_utc=datetime.now(timezone.utc).isoformat(),
                       bridge_elapsed_s=time.monotonic()-started)
        atomic(control_directory / 'telemetry.json', metrics)
        raise VLMError('Desktop decision timed out', retryable=False)
