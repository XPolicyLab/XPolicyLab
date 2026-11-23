# Copyright © 2026 Martingale. All Rights Reserved.
from dataclasses import asdict, dataclass
import hashlib
import json

@dataclass(frozen=True)
class InferenceConfig:
    model: str = 'gpt-6.2-astra'
    effort: str = 'xhigh'
    provider: str = 'codex_subscription'
    serving: str = 'default'
    sam3_sha256: str | None = None
    def __post_init__(self):
        if self.provider != 'codex_subscription' or self.serving != 'default':
            raise ValueError('This deployment implements subscription/default only; no fallback')
        if not self.model or self.effort not in ('low', 'medium', 'high', 'xhigh'):
            raise ValueError('Explicit model and supported effort required')
        if self.sam3_sha256 is not None and (len(self.sam3_sha256)!=64 or any(c not in '0123456789abcdef' for c in self.sam3_sha256)):
            raise ValueError('Pin the shared segmentation checkpoint SHA-256')
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()
    @classmethod
    def load(cls, path):
        with open(path) as stream: return cls(**json.load(stream))
