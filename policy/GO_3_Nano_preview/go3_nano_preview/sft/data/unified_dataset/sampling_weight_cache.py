from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import numpy as np
CACHE_SCHEMA = 2
CACHE_DIRNAME = 'sampling_weights'

class SamplingWeightCacheError(ValueError):
    pass

def canonical_identity_bytes(identity: dict[str, Any]) -> bytes:
    return json.dumps(identity, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')

def fingerprint_identity(identity: dict[str, Any]) -> str:
    return hashlib.blake2b(canonical_identity_bytes(identity), digest_size=16).hexdigest()

def build_cache_identity(*, snapshot_id: str, episode_count: int, source_names: tuple[str, ...] | list[str], policy: dict[str, Any]) -> dict[str, Any]:
    return {'schema': CACHE_SCHEMA, 'metadata': {'storage': 'indexed', 'snapshot_id': str(snapshot_id), 'episode_count': int(episode_count), 'source_names': [str(name) for name in source_names]}, 'policy': policy}

def cache_path_for_identity(cache_dir: str | os.PathLike[str], identity: dict[str, Any]) -> Path:
    return Path(cache_dir) / f'{fingerprint_identity(identity)}.npz'

def save_sampling_weights(path: str | os.PathLike[str], *, identity: dict[str, Any], weights: np.ndarray) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    array = np.ascontiguousarray(np.asarray(weights, dtype=np.float64))
    if array.ndim != 1:
        raise SamplingWeightCacheError('sampling weights must be a 1-D float64 array')
    identity_bytes = canonical_identity_bytes(identity)
    expected_count = int(identity['metadata']['episode_count'])
    if array.shape[0] != expected_count:
        raise SamplingWeightCacheError(f'sampling weight length does not match identity: {array.shape[0]} != {expected_count}')
    temporary = destination.with_name(destination.name + '.tmp')
    with temporary.open('wb') as handle:
        np.savez(handle, schema=np.int64(CACHE_SCHEMA), weights=array, identity_json=np.frombuffer(identity_bytes, dtype=np.uint8))
    os.replace(temporary, destination)
    return destination

def load_sampling_weights(path: str | os.PathLike[str], *, identity: dict[str, Any]) -> np.ndarray:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    expected_bytes = canonical_identity_bytes(identity)
    expected_fingerprint = fingerprint_identity(identity)
    if source.stem != expected_fingerprint:
        raise SamplingWeightCacheError(f'sampling weight cache filename does not match current identity: file={source.stem}, expected={expected_fingerprint}')
    with np.load(source, allow_pickle=False) as payload:
        schema = int(payload['schema'])
        if schema != CACHE_SCHEMA:
            raise SamplingWeightCacheError(f'unsupported sampling weight cache schema {schema}, expected {CACHE_SCHEMA}')
        stored_identity = json.loads(bytes(payload['identity_json']).decode('utf-8'))
        stored_bytes = canonical_identity_bytes(stored_identity)
        if stored_bytes != expected_bytes:
            raise SamplingWeightCacheError('sampling weight cache identity does not match current Dataset policy; refusing to use stale weights')
        weights = np.ascontiguousarray(payload['weights'], dtype=np.float64)
    expected_count = int(identity['metadata']['episode_count'])
    if weights.ndim != 1 or weights.shape[0] != expected_count:
        raise SamplingWeightCacheError(f"sampling weight cache length does not match current metadata: {getattr(weights, 'shape', None)} != ({expected_count},)")
    if not np.isfinite(weights).all():
        raise SamplingWeightCacheError('sampling weight cache contains non-finite values')
    return weights
