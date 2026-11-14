"""P10-v1: optional image-space lighting gains on distinct past RGB times.

The current/future clip owns all geometry and base augmentation parameters.
This module samples independent streams without advancing any global RNG.
"""
from __future__ import annotations
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import numpy as np
import torch
VERSION = 'history_temporal_light_p10_v1'
DEFAULTS = dict(enabled=False, probability=0.1, seed=42, exposure_stops=[-0.25, 0.25], red_gain=[0.9, 1.1], blue_gain=[0.9, 1.1], eligible_positions=[0, 1, 2, 3])

def resolve_history_light_augmentation(value=None):
    """Disabled and probability-zero configurations are exact no-op paths."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError('history_light_augmentation must be a mapping')
    unknown = set(value) - set(DEFAULTS)
    if unknown:
        raise ValueError('Unknown history lighting configuration: ' + str(sorted(unknown)))
    cfg = {**DEFAULTS, **dict(value)}
    if type(cfg['enabled']) is not bool:
        raise ValueError('history_light_augmentation.enabled must be Boolean')
    if not cfg['enabled']:
        return None
    if isinstance(cfg['probability'], bool):
        raise ValueError('History lighting probability must be numeric')
    cfg['probability'] = float(cfg['probability'])
    if not math.isfinite(cfg['probability']) or not 0 <= cfg['probability'] <= 1:
        raise ValueError('History lighting probability must be in [0,1]')
    if isinstance(cfg['seed'], bool) or not isinstance(cfg['seed'], (int, np.integer)):
        raise ValueError('History lighting seed must be an integer')
    cfg['seed'] = int(cfg['seed'])
    for key in ('exposure_stops', 'red_gain', 'blue_gain'):
        pair = cfg[key]
        if isinstance(pair, (str, bytes)) or not isinstance(pair, Sequence) or len(pair) != 2:
            raise ValueError('History lighting range requires two values: ' + key)
        lo, hi = map(float, pair)
        if not (math.isfinite(lo) and math.isfinite(hi) and (lo <= hi)):
            raise ValueError('Invalid history lighting range: ' + key)
        if key != 'exposure_stops' and lo <= 0:
            raise ValueError('History lighting gains must be positive')
        cfg[key] = [lo, hi]
    positions = cfg['eligible_positions']
    if isinstance(positions, (str, bytes)) or not isinstance(positions, Sequence) or (not positions) or any((type(p) is not int or p not in range(6) for p in positions)) or (len(set(positions)) != len(positions)):
        raise ValueError('History lighting positions must be unique integers in [0,5]')
    cfg['eligible_positions'] = list(positions)
    try:
        minimum = 2.0 ** cfg['exposure_stops'][0] * min(1.0, cfg['red_gain'][0], cfg['blue_gain'][0])
        maximum = 2.0 ** cfg['exposure_stops'][1] * max(1.0, cfg['red_gain'][1], cfg['blue_gain'][1])
    except OverflowError as exc:
        raise ValueError('History lighting gain is outside FP32 range') from exc
    limits = np.finfo(np.float32)
    if minimum < float(limits.tiny) or not math.isfinite(maximum) or maximum > float(limits.max):
        raise ValueError('History lighting gain is outside FP32 range')
    return cfg if cfg['probability'] > 0 else None

def history_light_contract(value=None):
    cfg = resolve_history_light_augmentation(value)
    if cfg is None:
        return {'enabled': False}
    contract = dict(enabled=True, version=VERSION, parameters=cfg, probability_scope='one_independent_Bernoulli_per_sample_occurrence', eligible='configured_positions_with_native_index_strictly_before_anchor', temporal='independent_per_distinct_native_frame_index', cameras='same_temporal_RGB_gain_tuple_for_all_three_cameras', repeated_times='reuse_parameters_by_actual_native_frame_index', current='copy_actual_chunk_current_pixels_after_resize', operation='FP32_clamp(I*2**EV*[red_gain,1,blue_gain],0,1)', order='base_geometry_then_base_photometric_then_history_light_then_valid_fill_then_original_resize', RNG='SHA256(seed,window_index,sample_position,version,stream,native_frame_index); no_global_RNG', geometry_depth_K_masks_actions_targets='unchanged', approximation='image_space_gain; not_physical_relighting')
    contract['sha256'] = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return contract

def _rng(cfg, index, occurrence, stream, frame=None):
    key = [cfg['seed'], int(index), int(occurrence), VERSION, stream]
    if frame is not None:
        key.append(int(frame))
    digest = hashlib.sha256(json.dumps(key, separators=(',', ':')).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:16], 'little'))

def sample_history_light(cfg, *, index, occurrence, frame_indices, anchor):
    """Return an auditable plan; the gate is independent of base augmentation."""
    cfg = resolve_history_light_augmentation(cfg)
    result = dict(gate_selected=False, by_frame={})
    if cfg is None:
        return result
    indices = np.asarray(frame_indices)
    if indices.shape not in [(5,), (6,)] or not np.issubdtype(indices.dtype, np.integer) or anchor < 0 or (indices < 0).any() or (indices > anchor).any() or (np.diff(indices) < 0).any() or (max(cfg['eligible_positions']) >= len(indices)):
        raise ValueError('History lighting requires five or six causal native frame indices')
    if cfg['eligible_positions'] == [0, 1, 2, 3] and indices[-1] != anchor:
        raise ValueError('Original P10-v1 layout must end at anchor')
    if _rng(cfg, index, occurrence, 'gate').random() >= cfg['probability']:
        return result
    result['gate_selected'] = True
    for frame in sorted({int(indices[p]) for p in cfg['eligible_positions'] if indices[p] < anchor}):
        rng = _rng(cfg, index, occurrence, 'params', frame)
        result['by_frame'][frame] = dict(exposure_stops=float(rng.uniform(*cfg['exposure_stops'])), red_gain=float(rng.uniform(*cfg['red_gain'])), blue_gain=float(rng.uniform(*cfg['blue_gain'])))
    return result

def apply_history_light(image, frame_indices, anchor, by_frame, *, record=None):
    """Apply gains without mutating the source or touching current RGB."""
    if not by_frame:
        return image
    indices = np.asarray(frame_indices)
    if indices.ndim != 1 or len(indices) not in (5, 6) or image.ndim != 4 or (image.shape[:2] != (len(indices), 3)) or (image.dtype != torch.float32):
        raise ValueError('History lighting expects aligned FP32 RGB [T,3,H,W]')
    if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError('History lighting RGB must be finite in [0,1]')
    eligible = {int(t) for t in indices if t < anchor}
    if not set(by_frame) <= eligible:
        raise ValueError('History lighting cannot modify current or unselected timestamps')
    output = image.clone()
    clipped, total = (0, 0)
    for frame, params in by_frame.items():
        positions = np.flatnonzero(indices == frame).tolist()
        gain = image.new_tensor([params['red_gain'], 1.0, params['blue_gain']])
        gain = gain * 2.0 ** params['exposure_stops']
        scaled = image[positions] * gain[None, :, None, None]
        if record is not None:
            clipped += int(((scaled < 0) | (scaled > 1)).sum())
            total += scaled.numel()
        output[positions] = scaled.clamp(0, 1)
    if record is not None:
        record.update(clipped_values=clipped, affected_values=total, clipped_value_fraction=clipped / total if total else 0.0)
    return output
