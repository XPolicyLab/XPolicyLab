"""Saved scene-seed protocol; usable without simulator or model dependencies."""
import hashlib
import json
import operator
from pathlib import Path

SEED_LIST_PROTOCOL = 'robocasa_scene_seed_list_v1'
SEED_MODULUS = 2**32


def validate_seed(seed):
    if isinstance(seed, bool):
        raise ValueError('Scene seed must be an integer in [0, 2**32)')
    try:
        seed = operator.index(seed)
    except TypeError as exc:
        raise ValueError('Scene seed must be an integer in [0, 2**32)') from exc
    if not 0 <= seed < SEED_MODULUS:
        raise ValueError('Scene seed must be an integer in [0, 2**32)')
    return seed


def load_scene_seed_list(path, n_episodes):
    raw = Path(path).read_bytes()
    manifest = json.loads(raw)
    if not isinstance(manifest, dict) or manifest.get('protocol') != SEED_LIST_PROTOCOL:
        raise ValueError('Invalid scene seed list protocol')
    values = manifest.get('episode_seeds')
    if not isinstance(values, list) or not values:
        raise ValueError('Scene seed list must be a nonempty list')
    seeds = [validate_seed(value) for value in values]
    if not 1 <= n_episodes <= len(seeds):
        raise ValueError('Scene seed list is too short for requested episodes')
    return {'protocol': SEED_LIST_PROTOCOL, 'manifest': manifest,
            'manifest_sha256': hashlib.sha256(raw).hexdigest(),
            'episode_seeds': seeds[:n_episodes],
            'schedule': 'saved episode_seeds[episode_index], identical across tasks/checkpoints'}
