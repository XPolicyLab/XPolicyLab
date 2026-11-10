"""Local stage reviews and request settings shared by the digit identity checker."""
import os
from pathlib import Path
import uuid

import numpy as np


MODEL = os.getenv('ME_BRAIN_API_MODEL', '')
URL = os.getenv('ME_BRAIN_API_URL', '')
AUTH_HEADER = os.getenv('ME_BRAIN_API_AUTH_HEADER') or 'Authorization'
REQUEST_ID_HEADER = os.getenv('ME_BRAIN_API_REQUEST_ID_HEADER') or 'X-Request-ID'
TOKEN_FILE = os.getenv('ME_BRAIN_API_TOKEN_FILE', '')
TOKEN_PATH = Path(TOKEN_FILE).expanduser() if TOKEN_FILE else None
if TOKEN_PATH is not None and not TOKEN_PATH.is_absolute():
    TOKEN_PATH = Path(__policy_root__) / TOKEN_PATH


def request_headers(token):
    headers = {'Content-Type': 'application/json', REQUEST_ID_HEADER: str(uuid.uuid4())}
    if AUTH_HEADER != 'Authorization':
        headers[AUTH_HEADER] = token
    headers['Authorization'] = 'Bearer ' + token
    return headers


def scalar(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


class OfflineSupervisor:
    def __init__(self):
        self.calls = 0
        self.elapsed = 0.
        self.records = []
        self.episode = uuid.uuid4().hex[:12]

    def review(self, stage, step, context, frames, allowed):
        request_id = f'{self.episode}-{self.calls+1}-{step}'
        local = (context.get('local_evidence') or {}) if isinstance(context, dict) else {}
        status = local.get('local_status')
        contact = bool(local.get('coarse_stable_contact')) or any(
            isinstance(item, dict) and item.get('contact')
            for item in local.get('wrist_observations') or []
        )
        confidence = .9 if contact else .5

        if stage in ('post_lift', 'pre_place', 'transport_stall'):
            if status == 'empty' and 'skip_empty' in allowed:
                action, possession, placement = 'skip_empty', 'empty', 'not_applicable'
            else:
                action, possession, placement = 'continue', 'held_clear', 'aligned'
        elif stage == 'post_release':
            action, possession, placement = 'continue', 'empty', 'unknown'
        else:
            action, possession, placement = 'continue', 'not_applicable', 'not_applicable'

        if action not in allowed:
            action = next((option for option in ('continue', 'observe') if option in allowed), 'hold')

        self.records.append(dict(
            request_id=request_id, stage=stage, step=int(step), allowed_actions=list(allowed),
            frame_sha256={}, status='offline_local', elapsed_seconds=0.,
            result=dict(action=action, possession=possession, placement=placement,
                        local_status=status, contact=contact),
        ))
        return dict(
            request_id=request_id, stage=stage, action=action, possession=possession,
            placement=placement, confidence=confidence if action == 'continue' else .9,
            evidence=f'offline_local_evidence:{status}:contact={contact}', error=None,
        )
