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
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)

class DesktopClient:
    model = 'codex-desktop-session'

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def complete(self, messages, *, deadline, **kwargs):
        started = time.monotonic()
        created_at = datetime.now(timezone.utc).isoformat()
        request_id = uuid.uuid4().hex
        directory = self.root / request_id
        directory.mkdir()
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
                    (directory / filename).write_bytes(blob)
                    image_count += 1
                    image_bytes += len(blob)
                    readable.append(f'IMAGE {filename}')
        (directory / 'prompt.txt').write_text('\n\n'.join(readable), encoding='utf-8')
        metrics = {
            'schema_version': 1, 'request_id': request_id, 'model_transport': self.model,
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
        atomic(self.root / 'pending.json', {'request_id': request_id, 'directory': str(directory)})
        response = directory / 'response.json'
        while time.monotonic() < deadline:
            if response.exists():
                packet = json.loads(response.read_text(encoding='utf-8'))
                if packet.get('request_id') != request_id:
                    raise ValueError('Stale desktop response')
                text = json.dumps(packet['response'], ensure_ascii=False)
                elapsed = time.monotonic() - started
                metrics.update(status='accepted', accepted_at_utc=datetime.now(timezone.utc).isoformat(),
                               bridge_elapsed_s=elapsed, response_chars=len(text))
                atomic(directory / 'telemetry.json', metrics)
                atomic(directory / 'accepted.json', {'request_id': request_id, 'accepted_at_utc': metrics['accepted_at_utc']})
                return {'text': text, 'usage': {}, 'finish_reason': 'desktop_response',
                        'latency_s': elapsed, 'attempts': 1, 'retry_history': []}
            time.sleep(.2)
        metrics.update(status='timeout', ended_at_utc=datetime.now(timezone.utc).isoformat(),
                       bridge_elapsed_s=time.monotonic()-started)
        atomic(directory / 'telemetry.json', metrics)
        raise VLMError('Desktop decision timed out', retryable=False)
