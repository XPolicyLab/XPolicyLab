from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import numpy as np
from .schema import STATE_DIM, ACTION_DIM, file_sha256, fingerprint, within
from .robotwin_joint14 import INDICES, native_fields
PROFILE = 'robotwin_joint14_cleanonly_v1'
STATS_KEY = 'robotwin_aloha_joint14_cleanonly_v1'
PROTOCOL = 'robotwin.clean50.c2c_c2r.v1'

def population(episodes):
    return [dict(episode_id=e['episode_id'], task=e['clean_task'], length=e['length'], data_sha256=e['data_sha256']) for e in episodes]

def validate_manifest(manifest):
    contract = manifest['source']['clean_contract']
    if contract.get('protocol') != PROTOCOL or contract.get('statistics_population') != 'all_selected_raw_frames_once':
        raise ValueError('Unknown clean-only population contract')
    counts = dict.fromkeys(contract['task_counts'], 0)
    seen = set()
    for ep in manifest['episodes']:
        if ep['episode_id'] in seen or ep.get('collection_regime') != 'demo_clean' or ep['split'] != 'train' or (ep.get('clean_task') not in counts) or (ep['intervals'] != [[0, ep['length']]]):
            raise ValueError('Invalid or contaminated clean-only manifest')
        seen.add(ep['episode_id'])
        counts[ep['clean_task']] += 1
    if counts != contract['task_counts'] or not counts or any((n != contract['episodes_per_task'] for n in counts.values())) or (manifest['validation_fraction'] != 0) or (sum((e['length'] for e in manifest['episodes'])) != contract['raw_frames']) or (fingerprint(population(manifest['episodes'])) != contract['population_sha256']):
        raise ValueError('Clean-only population fingerprint/count mismatch')

def validate_group(group):
    from .robotwin_joint14 import NATIVE_STATS_SHA256
    payload = {k: v for k, v in group.items() if k != 'native_stats_sha256'}
    if group.get('profile') != PROFILE or group.get('mode') != 'z-score' or group.get('native_indices') != INDICES or (group.get('native_stats_sha256') in (None, NATIVE_STATS_SHA256)) or (group['native_stats_sha256'] != fingerprint(payload)) or (group.get('clean_contract', {}).get('protocol') != PROTOCOL):
        raise ValueError('Clean-only statistics profile/fingerprint mismatch')
    for kind, dim in [('state', STATE_DIM), ('action', ACTION_DIM)]:
        active = np.zeros(dim, bool)
        active[INDICES[kind]] = True
        count = np.asarray(group[kind]['count'])
        std = np.asarray(group[kind]['std'])
        if count.shape != (dim,) or not np.array_equal(count > 0, active) or (std < 0).any() or (not np.all(count[active] == group['clean_contract']['raw_frames'])):
            raise ValueError('Clean-only statistics require exactly the native 14 slots')
