"""Load canonical normalization statistics for the selected pretraining sources."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def read_stats(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f'Pretraining normalization statistics not found: {path}')
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict) or not isinstance(payload.get('groups'), dict):
        raise ValueError(f'Pretraining statistics must contain a groups mapping: {path}')
    return payload


def merge_stats(payloads: Iterable[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    merged: dict[str, Any] = {'groups': {}}
    owners: dict[str, Path] = {}
    for path, payload in payloads:
        for key in ('action_dim', 'state_dim', 'norm_mode'):
            if key not in payload:
                continue
            if key in merged and merged[key] != payload[key]:
                raise ValueError(f'Inconsistent {key} in normalization statistics: {path}')
            merged[key] = payload[key]
        for group, values in payload['groups'].items():
            if group in owners and merged['groups'][group] != values:
                raise ValueError(
                    f'Conflicting shared statistics group {group!r}: {owners[group]} and {path}. '
                    'Recompute sources sharing this group together.'
                )
            merged['groups'][group] = values
            owners[group] = path
    return merged


def load_pretrain_stats(specs, config, override=None):
    """Read each selected file once, retaining only groups used by active specs."""
    specs = [spec for spec in specs if spec.dataset_weight > 0]
    if not specs:
        raise ValueError('No active pretraining datasets selected')
    settings = config.get('stats') or {}
    override = override or settings.get('path') or settings.get('grouped_stats_path')
    requested: dict[Path, set[str]] = defaultdict(set)
    configured = [spec.normalization_stats for spec in specs]
    if not override and not any(configured):
        return None, []
    for spec in specs:
        path = Path(str(override)).expanduser() if override else None
        if path is None or path.is_dir():
            if not spec.normalization_stats:
                raise ValueError(f'Dataset {spec.name!r} has no normalization_stats file')
            source_path = Path(spec.normalization_stats).expanduser()
            path = source_path if path is None else path / source_path.name
        requested[path].add(spec.stats_group or spec.name)
    payloads = []
    for path, groups in sorted(requested.items()):
        payload = read_stats(path)
        missing = groups - payload['groups'].keys()
        if missing:
            raise KeyError(f'Missing normalization groups in {path}: {sorted(missing)}')
        payloads.append((path, {**payload, 'groups': {g: payload['groups'][g] for g in sorted(groups)}}))
    return merge_stats(payloads), [path for path, _ in payloads]


def split_stats(payload, specs):
    """Produce one portable JSON payload per configured source statistics file."""
    groups_by_file: dict[str, set[str]] = defaultdict(set)
    file_paths = {}
    for spec in specs:
        group = spec.stats_group or spec.name
        if group not in payload['groups']:
            continue
        if not spec.normalization_stats:
            raise ValueError(f'Dataset {spec.name!r} needs normalization_stats for split output')
        path = Path(spec.normalization_stats)
        filename = path.name
        if filename in file_paths and file_paths[filename] != path:
            raise ValueError(f'Duplicate statistics filename {filename!r}; use distinct filenames')
        file_paths[filename] = path
        groups_by_file[filename].add(group)
    covered = set().union(*groups_by_file.values()) if groups_by_file else set()
    if not covered:
        raise ValueError('No computed statistics match the configured sources')
    result = {}
    for filename, groups in sorted(groups_by_file.items()):
        result[filename] = {
            'format': 'internw0.grouped_stats.v1',
            **{k: payload[k] for k in ('action_dim', 'state_dim', 'norm_mode', 'passthrough_dims') if k in payload},
            'groups': {g: payload['groups'][g] for g in sorted(groups)},
        }
    return result
