#!/usr/bin/env python3
"""Offline action validation for Piper X; never reports physical success rates."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from robodojo_piperx_download import ROOT, TASKS
from robodojo_piperx_prepare import atomic_json


def load_policy(manifest):
    import torch
    from infer_aloha_stationary import AlohaFastWAMPolicy
    # Both methods deploy the same backbone/action-expert graph. The training-
    # only foresight head is not constructed during action inference.
    checkpoint = Path(manifest['checkpoint'])
    payload = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    if not {'mot', 'proprio_encoder'}.issubset(payload):
        raise ValueError('A complete joint-policy checkpoint is required')
    has_foresight = 'foresight_projectors' in payload
    if has_foresight != (manifest['method'] == 'jam'):
        raise ValueError('Checkpoint method does not match deployment manifest')
    previous_cwd = Path.cwd()
    code_root = Path(os.environ.get('FASTWAM_ROOT', manifest['code_root'])).resolve()
    os.environ['DIFFSYNTH_MODEL_BASE_PATH'] = str(code_root / 'checkpoints')
    os.environ['DIFFSYNTH_SKIP_DOWNLOAD'] = 'true'
    try:
        os.chdir(code_root)
        policy = AlohaFastWAMPolicy(
            task_config='robodojo_piperx_fastwam', checkpoint=checkpoint,
            dataset_stats=Path(manifest['dataset_stats']), device='cuda', mixed_precision='bf16',
            num_inference_steps=manifest.get('num_inference_steps', 10), action_horizon=32,
            sigma_shift=None, seed=42, text_cfg_scale=1.0, rand_device='cpu', tiled=False)
    finally:
        os.chdir(previous_cwd)
    expected = policy.model.mot.state_dict()
    if set(payload['mot']) != set(expected):
        raise ValueError('Checkpoint MoT keys do not exactly match the deployment graph')
    for name, value in payload['mot'].items():
        if value.shape != expected[name].shape:
            raise ValueError(f'Checkpoint shape mismatch: {name}')
    del payload, expected
    if policy.model.foresight_projectors is not None:
        raise RuntimeError('Training-only foresight branch remained in deployment')
    return policy


def evaluate(manifest, root, output, episodes_per_task=10, positions=4):
    import pyarrow.parquet as pq
    from infer_aloha_stationary import _load_lerobot_observation, _episode_parquet
    policy = load_policy(manifest)
    split = json.loads((root / 'stats/split_audit.json').read_text())
    latencies, predictions = [], {}
    report = {'method': manifest['method'], 'checkpoint': manifest['checkpoint'],
              'checkpoint_sha256': manifest['checkpoint_sha256'],
              'evaluation_type': 'held-out demonstration action prediction; not robot rollouts',
              'robot_success_rate': None, 'real_robot_status': 'awaiting_official_evaluation',
              'seed': 42, 'inference_steps': manifest.get('num_inference_steps', 10), 'tasks': {}}
    for task in TASKS:
        dataset = root / 'ready' / task
        info = json.loads((dataset / 'meta/info.json').read_text())
        meta = split['tasks'][task]
        task_errors, static_errors = [], []
        for episode in meta['val_episode_indices'][:episodes_per_task]:
            table = pq.read_table(_episode_parquet(dataset, info, episode))
            targets = np.asarray(table['action'].to_pylist(), dtype=np.float32)
            task_index = int(table['task_index'][0].as_py())
            task_names = [json.loads(line) for line in (dataset / 'meta/tasks.jsonl').read_text().splitlines()]
            instruction = next(x['task'] for x in task_names if x['task_index'] == task_index)
            for frame in np.unique(np.linspace(0, max(0, len(targets) - 32), positions, dtype=int)):
                observation = _load_lerobot_observation(dataset, episode, int(frame))
                action, latency = policy.predict(observation, instruction)
                target = targets[frame:frame + 32]
                if len(target) != 32:
                    raise ValueError('Evaluation sample crosses episode boundary')
                task_errors.append(np.abs(action - target).mean(axis=0))
                static_errors.append(np.abs(observation['qpos'][None] - target).mean(axis=0))
                latencies.append(latency)
                key = f'{task}_ep{episode:03d}_frame{frame:05d}'
                predictions[key] = np.stack([action, target])
        error = np.mean(task_errors, axis=0)
        static = np.mean(static_errors, axis=0)
        joint_indices = [i for i in range(14) if i not in (6, 13)]
        report['tasks'][task] = {'episodes': min(episodes_per_task, len(meta['val_episode_indices'])),
            'chunks': len(task_errors), 'mae_per_dimension_native_units': error.tolist(),
            'joint_mae_rad': float(error[joint_indices].mean()),
            'gripper_mae_native_units': float(error[[6, 13]].mean()),
            'hold_current_position_joint_mae_rad': float(static[joint_indices].mean()),
            'hold_current_position_gripper_mae_native_units': float(static[[6, 13]].mean())}
        print(task, json.dumps(report['tasks'][task]), flush=True)
        atomic_json(output, report)
    report['latency_s'] = {'cold': latencies[0], 'warm_median': float(np.median(latencies[1:])),
                           'warm_p95': float(np.quantile(latencies[1:], 0.95)), 'max': max(latencies)}
    report['completed_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    atomic_json(output, report)
    np.savez_compressed(output.with_suffix('.predictions.npz'), **predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes-per-task', type=int, default=10)
    parser.add_argument('--positions', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.episodes_per_task <= 10 or args.positions < 1:
        raise ValueError('Invalid held-out evaluation scope')
    manifest = json.loads((args.bundle / 'deployment.json').read_text())
    evaluate(manifest, args.root, args.output, args.episodes_per_task, args.positions)


if __name__ == '__main__':
    main()
