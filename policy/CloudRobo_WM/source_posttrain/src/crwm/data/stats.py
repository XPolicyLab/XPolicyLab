from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .schema import STATE, ACTION, STATE_DIM, ACTION_DIM, SCHEMA_VERSION, CODEC_VERSION, atomic_json, fingerprint

def default_contract(**kwargs):
    return dict(schema=SCHEMA_VERSION, codec=CODEC_VERSION, **kwargs)

class StatsRegistry:

    def __init__(self, groups, contract):
        self.groups, self.contract = (groups, contract)
        for key, group in groups.items():
            if group.get('profile'):
                from .robotwin_joint14 import validate_group
                validate_group(group)
            if group['mode'] not in {'quantile', 'z-score'}:
                raise ValueError(f'{key}: unknown normalization mode')
            if group['mode'] == 'quantile' and any((group[k].get('quantiles_computed') is False for k in ('state', 'action'))):
                raise ValueError(f'{key}: quantile normalization requires computed quantiles')
            for kind, dim in (('state', STATE_DIM), ('action', ACTION_DIM)):
                for stat in ('count', 'mean', 'std', 'q01', 'q99'):
                    x = np.asarray(group[kind][stat])
                    if x.shape != (dim,) or not np.isfinite(x).all():
                        raise ValueError(f'{key}/{kind}/{stat}: invalid statistics')

    @property
    def digest(self):
        return fingerprint(self.to_dict())

    def to_dict(self):
        return dict(version=1, contract=self.contract, groups=self.groups)

    @classmethod
    def load(cls, path, expected_contract=None):
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        if payload.get('version') != 1:
            raise ValueError('Unsupported statistics version')
        if expected_contract is not None and payload['contract'] != expected_contract:
            raise ValueError('Statistics contract mismatch; regenerate from the training split')
        return cls(payload['groups'], payload['contract'])

    def normalize(self, key, kind, values, valid, *, report=False):
        from .robotwin_eef_cleanonly import normalize
        return normalize(self.groups[key], kind, values, valid, report=report)
