"""Desktop transport for the unmodified RoboDawn agent prompt/command loop."""
import base64
import json
import os
from pathlib import Path
import time
import uuid
from datetime import datetime, timezone
from .vlm_client import VLMError

def atomic_bytes(path, data):
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with tmp.open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)

def atomic(path, value):
    atomic_bytes(path, json.dumps(value, ensure_ascii=False).encode('utf-8'))

class DesktopClient:
    model = 'codex-desktop-session'

    def __init__(self, root):
        self.root = Path(root)
        self.run_id = str(self.root.parent)
        self.root.mkdir(parents=True, exist_ok=True)
        self.control_root = self.root.parent / 'control' if self.root.name == 'bridge' else self.root
        self.control_root.mkdir(parents=True, exist_ok=True)

    def complete(self, messages, *, deadline, **kwargs):
        started = time.monotonic()
        created_at = datetime.now(timezone.utc).isoformat()
        request_id = uuid.uuid4().hex
        directory = self.root / request_id
        directory.mkdir()
        response_directory = self.control_root / request_id
        response_directory.mkdir(exist_ok=True)
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
            'schema_version': 1, 'request_id': request_id, 'run_id': self.run_id, 'model_transport': self.model,
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
        atomic(directory / 'telemetry.json', metrics)
        atomic(self.root / 'pending.json', {'request_id': request_id, 'run_id': self.run_id, 'directory': str(directory), 'response_directory': str(response_directory)})
        response = response_directory / 'response.json'
        unreadable_since = None
        visibility_retries = 0
        while time.monotonic() < deadline:
            if response.exists():
                try:
                    packet = json.loads(response.read_text(encoding='utf-8'))
                except (json.JSONDecodeError, UnicodeDecodeError, FileNotFoundError) as exc:
                    # Object-store rename visibility is not POSIX atomic: a name can
                    # precede readable contents. Never execute a partial response.
                    now = time.monotonic()
                    unreadable_since = now if unreadable_since is None else unreadable_since
                    visibility_retries += 1
                    if now - unreadable_since >= 10:
                        raise VLMError('Desktop response remained unreadable for10s', retryable=False) from exc
                    time.sleep(.2)
                    continue
                if not isinstance(packet, dict) or not isinstance(packet.get('response'), dict):
                    raise ValueError('Invalid desktop response envelope')
                if packet.get('request_id') != request_id or packet.get('run_id') != self.run_id:
                    raise ValueError('Stale desktop response')
                if (directory / 'accepted.json').exists():
                    raise ValueError('Response already consumed; inspect execution before resuming')
                text = json.dumps(packet['response'], ensure_ascii=False)
                elapsed = time.monotonic() - started
                metrics.update(status='accepted', accepted_at_utc=datetime.now(timezone.utc).isoformat(),
                               bridge_elapsed_s=elapsed, response_chars=len(text), visibility_retries=visibility_retries)
                atomic(directory / 'telemetry.json', metrics)
                atomic(directory / 'accepted.json', {'request_id': request_id, 'run_id': self.run_id, 'accepted_at_utc': metrics['accepted_at_utc']})
                return {'text': text, 'usage': {}, 'finish_reason': 'desktop_response',
                        'latency_s': elapsed, 'attempts': 1, 'retry_history': []}
            time.sleep(.2)
        metrics.update(status='timeout', ended_at_utc=datetime.now(timezone.utc).isoformat(),
                       bridge_elapsed_s=time.monotonic()-started)
        atomic(directory / 'telemetry.json', metrics)
        raise VLMError('Desktop decision timed out', retryable=False)
