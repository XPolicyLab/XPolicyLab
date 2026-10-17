#!/usr/bin/env python3
"""Freeze a per-task 90/10 episode split and joint training-only normalization."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from robodojo_piperx_download import TASKS, REVISION, ROOT
from build_aloha_stationary_fastwam_stats import (
    _fixed_list_tensor, _episode_path, _episode_stats, _sliding_window_with_replication,
    _combine, _atomic_json, _flatten_names,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    torch.set_num_threads(2)
    action_stats, state_stats = [], []
    audit = {'revision': REVISION, 'seed': 42, 'val_proportion': 0.1,
             'fps': 25, 'action_window': 32, 'global_batch': 128, 'epochs': 10,
             'tasks': {}, 'train_frames': 0, 'val_frames': 0,
             'training_episodes': 540, 'validation_episodes': 60,
             'checkpoint_selection': 'fixed epoch-10 endpoint, no test selection'}
    names = [f'{side}_joint_{i}' for side in ['left', 'right'] for i in range(7)]
    for task in TASKS:
        root = args.root / 'ready' / task
        info_path = root / 'meta/info.json'
        info = json.loads(info_path.read_text())
        converted = json.loads((root / 'conversion_audit.json').read_text())
        assert info['fps'] == 25 and info['total_episodes'] == 100
        assert converted['revision'] == REVISION
        for feature in ['action', 'observation.state']:
            assert _flatten_names(info['features'][feature]['names']) == names
            assert info['features'][feature]['shape'] == [14]
        indices = np.arange(100)
        np.random.default_rng(42).shuffle(indices)
        train, val = sorted(indices[:90].tolist()), sorted(indices[90:].tolist())
        lengths = converted['episode_lengths']
        for episode in train:
            table = pq.read_table(_episode_path(root, info, episode), columns=['action', 'observation.state'])
            action = _fixed_list_tensor(table, 'action', 14)
            state = _fixed_list_tensor(table, 'observation.state', 14)
            assert len(action) == lengths[episode]
            assert torch.isfinite(action).all() and torch.isfinite(state).all()
            action_stats.append(_episode_stats(_sliding_window_with_replication(action, 32)))
            state_stats.append(_episode_stats(state.unsqueeze(1)))
        n_train = sum(lengths[i] for i in train)
        n_val = sum(lengths[i] for i in val)
        audit['tasks'][task] = {'dataset_root': str(root), 'train_episode_indices': train,
            'val_episode_indices': val, 'episode_lengths': lengths,
            'train_frames': n_train, 'val_frames': n_val,
            'instructions': converted['instructions'],
            'info_sha256': hashlib.sha256(info_path.read_bytes()).hexdigest()}
        audit['train_frames'] += n_train
        audit['val_frames'] += n_val
        print(f'{task}: 90 train / 10 validation, {n_train} / {n_val} frames', flush=True)
    payload = {'state': {'default': _combine(state_stats)},
               'action': {'default': _combine(action_stats)},
               'num_episodes': 540, 'num_transition': audit['train_frames']}
    audit['steps_per_epoch'] = math.ceil(audit['train_frames'] / 128)
    audit['optimizer_steps'] = 10 * audit['steps_per_epoch']
    _atomic_json(args.root / 'stats/train90_seed42.json', payload)
    _atomic_json(args.root / 'stats/split_audit.json', audit)
    print(f'Frozen training length: {audit["optimizer_steps"]} updates', flush=True)


if __name__ == '__main__':
    main()
