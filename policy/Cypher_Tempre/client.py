"""Observation-only transport; never makes robot decisions or resets memory."""
from __future__ import annotations
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def chain_tip(root):
    path = Path(root) / 'chain/rings.jsonl'
    raw = path.read_bytes()
    ring = json.loads(raw.splitlines()[-1])
    return {'ring': ring['index'], 'ring_hash': ring['ring_hash'],
            'ledger_sha256': hashlib.sha256(raw).hexdigest()}


def validate_reply(reply, tools):
    if not isinstance(reply, dict) or set(reply) != {'name', 'arguments'}:
        raise ValueError('Reply must contain exactly name and arguments.')
    allowed = {x['function']['name'] for x in tools}
    if reply['name'] not in allowed or not isinstance(reply['arguments'], dict):
        raise ValueError('Unknown tool or invalid argument object.')
    if reply['name'] == 'move_eef':
        arguments = reply['arguments']
        targets = arguments.get('targets')
        if not isinstance(targets, dict) or not targets:
            raise ValueError('A nonempty target map is required.')
        schema = next(x['function']['parameters'] for x in tools if x['function']['name'] == 'move_eef')
        properties = schema.get('properties', {}).get('targets', {}).get('properties', {})
        allowed_keys = set(properties) if properties else {
            f'{arm}_{axis}' for arm in ('left', 'right')
            for axis in ('x', 'y', 'z', 'pitch_deg', 'roll_deg', 'yaw_deg', 'gripper')}
        for key, value in targets.items():
            if key not in allowed_keys or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('Targets must use declared dimensions and finite numbers.')
        if not isinstance(arguments.get('note'), str):
            raise ValueError('A model-authored observation/action note is required.')
    return reply


class ContinualClient:
    """One session per episode, one shared skill state across every episode."""
    def __init__(self, env):
        self.env = dict(env)
        self.skill = Path(self.env['CYPHER_TEMPRE_SKILL_ROOT']).resolve()
        # Missing memory is an error. An empty chain is never silently substituted.
        chain_tip(self.skill)
        self.root = Path(self.env['CYPHER_TEMPRE_STREAM_ROOT']).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.session = uuid.uuid4().hex
        self.folder = self.root / 'sessions' / self.session
        self.folder.mkdir(parents=True, exist_ok=False)
        self.turn = 0
        self.started = False

    def complete(self, messages, tools):
        if not self.started:
            atomic_json(self.folder / 'session.json', {
                'session': self.session, 'created_at': time.time(),
                'skill_root': str(self.skill), 'memory_before': chain_tip(self.skill),
                'model': self.env.get('CYPHER_TEMPRE_MODEL', 'gpt-6.1-sol'),
                'reasoning_effort': self.env.get('CYPHER_TEMPRE_REASONING_EFFORT', 'xhigh')})
            self.started = True
        self.turn += 1
        texts, images = [], []
        current = next(m for m in reversed(messages) if isinstance(m.get('content'), list))
        for part in current['content']:
            if part['type'] == 'text':
                texts.append(part['text'])
            elif part['type'] == 'image_url':
                prefix, data = part['image_url']['url'].split(',', 1)
                if prefix != 'data:image/jpeg;base64':
                    raise ValueError('Expected RGB JPEG observations from the upstream encoder.')
                name = f'frame-{self.turn:03}-{len(images)}.jpg'
                (self.folder / name).write_bytes(base64.b64decode(data, validate=True))
                images.append(name)
        atomic_json(self.folder / f'request-{self.turn:03}.json', {
            'session': self.session, 'turn': self.turn,
            'instructions': messages[0]['content'], 'goal': messages[1]['content'],
            'state_and_cameras': texts, 'images': images,
            'recent_feedback': [m['content'] for m in messages if m['role'] == 'tool'][-2:],
            'tools': tools})
        timeout = float(self.env.get('CYPHER_TEMPRE_REPLY_TIMEOUT', '600'))
        if not math.isfinite(timeout) or not 1 <= timeout <= 7200:
            raise ValueError('Reply timeout must be finite and within1..7200seconds.')
        reply_path = self.folder / f'reply-{self.turn:03}.json'
        deadline = time.monotonic() + timeout
        while not reply_path.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Controller reply timed out: session{self.session}/turn{self.turn}')
            time.sleep(.1)
        reply = validate_reply(json.loads(reply_path.read_text(encoding='utf-8-sig')), tools)
        atomic_json(self.folder / 'memory-last-reply.json', chain_tip(self.skill))
        return {'choices': [{'finish_reason': 'tool_calls', 'message': {
            'role': 'assistant', 'content': None, 'tool_calls': [{
                'id': f'cypher-{self.session}-{self.turn}', 'type': 'function',
                'function': {'name': reply['name'], 'arguments': json.dumps(reply['arguments'], allow_nan=False)}}]}}]}

    def finish(self):
        atomic_json(self.folder / 'episode-ended.json', {
            'session': self.session, 'ended_at': time.time(), 'native_replies': self.turn,
            'memory_after': chain_tip(self.skill),
            'scope': 'Boundary receipt only; official outcome is supplied by the evaluator.'})
