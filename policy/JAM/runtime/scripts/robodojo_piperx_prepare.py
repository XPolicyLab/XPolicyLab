#!/usr/bin/env python3
"""Prepare the six real Piper X tasks through XPolicyLab's official converter.

The only converter adaptation is a CPU H.264 encoder; data extraction and RGB
decoding remain upstream. Conversion is atomic per task and never overwrites a
completed dataset. Run heavy conversion on an allocated Miyabi-G node.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

from robodojo_piperx_download import TASKS, REVISION

RUN_ROOT = Path('/original-workspace/robodojo_piperx_20261013')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def encoder(imgs_dir, video_path, fps, overwrite=False, **kwargs):
    """LeRobot encoder hook. Inputs are upstream RGB PNGs, never raw JPEGs."""
    import av
    from PIL import Image
    imgs = sorted(Path(imgs_dir).glob('frame_*.png'))
    if not imgs:
        raise ValueError(f'No PNG frames: {imgs_dir}')
    destination = Path(video_path)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(imgs[0]) as im:
        width, height = im.size
    with av.open(str(destination), 'w') as out:
        stream = out.add_stream('libx264', rate=int(fps), options={
            'crf': '18', 'preset': 'fast', 'g': '2', 'bf': '0', 'threads': '2'})
        stream.width, stream.height, stream.pix_fmt = width, height, 'yuv420p'
        stream.codec_context.thread_count = 2
        for path in imgs:
            with Image.open(path) as im:
                frame = av.VideoFrame.from_image(im.convert('RGB'))
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


def prepare_task(run_root, task, limit=100, tag='ready'):
    import av
    import h5py
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    torch.set_num_threads(2)
    random.seed(42)
    np.random.seed(42)
    run_root = Path(run_root)
    output = run_root / tag / task
    marker = output / 'conversion_audit.json'
    if marker.exists():
        old = json.loads(marker.read_text())
        if old['revision'] != REVISION or old['episodes'] != limit:
            raise RuntimeError(f'Existing dataset has different provenance: {output}')
        print(f'Already verified: {task}', flush=True)
        return old
    if output.exists():
        raise RuntimeError(f'Unverified existing final dataset: {output}')

    sources = sorted((run_root / 'data/RoboDojo_real' / task / 'piper_x/data').glob('*.hdf5'))
    if len(sources) < limit or (limit == 100 and len(sources) != 100):
        raise RuntimeError(f'{task}: expected {limit} complete files, found {len(sources)}')
    sources = sources[:limit]
    staging_parent = run_root / 'conversion_staging' / f'{task}_{os.getpid()}_{time.time_ns()}'
    os.environ['HF_LEROBOT_HOME'] = str(staging_parent)
    os.environ['HF_DATASETS_CACHE'] = str(run_root / 'hf_cache/datasets')
    sys.path.insert(0, str(run_root))
    converter_path = run_root / 'XPolicyLab/scripts/transform_lerobot_v21_format.py'
    spec = importlib.util.spec_from_file_location('xpl_official_converter', converter_path)
    converter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = converter
    spec.loader.exec_module(converter)
    from XPolicyLab.utils.process_data import decode_image_bit
    converter.lerobot_dataset_mod.encode_video_frames = encoder
    encoder._vcodec_patched = True
    dataset = converter.create_empty_dataset(
        repo_id=task, robot_type='piper_x',
        motors=converter._build_motor_names_from_dims([7, 7]), fps=25,
        image_height=480, image_width=640,
        dataset_config=converter.DatasetConfig(image_writer_threads=4, video_backend='pyav'))
    audit = {'task': task, 'revision': REVISION, 'episodes': limit, 'fps': 25,
             'fps_source': 'RoboDojo paper real-world dataset protocol (25 Hz)',
             'converter_sha256': hashlib.sha256(converter_path.read_bytes()).hexdigest(),
             'encoder': 'libx264 crf18 g2 bf0 yuv420p', 'vector_order': 'left6, left_gripper, right6, right_gripper',
             'image_order': 'RGB via XPolicyLab.decode_image_bit',
             'action_alignment': 'raw action[t], no additional time shift',
             'episode_lengths': [], 'instructions': [], 'camera_first_frame_mae_max': 0.0}
    cameras = {'cam_high': 'cam_head', 'cam_left_wrist': 'cam_left_wrist', 'cam_right_wrist': 'cam_right_wrist'}
    parts = ['left_arm_joint_states', 'left_ee_joint_states', 'right_arm_joint_states', 'right_ee_joint_states']
    for episode, source in enumerate(sources):
        with h5py.File(source, 'r') as f:
            action = np.concatenate([f['action/' + part][:] for part in parts], axis=-1).astype('float32')
            state = np.concatenate([f['state/' + part][:] for part in parts], axis=-1).astype('float32')
            n = len(action)
            if action.shape != (n, 14) or state.shape != action.shape or n < 33:
                raise ValueError(f'{source}: invalid state/action shape')
            if not np.isfinite(action).all() or not np.isfinite(state).all():
                raise ValueError(f'{source}: non-finite joint vector')
            text = f['instruction'][()]
            text = text.decode('utf-8') if isinstance(text, bytes) else str(text)
            if not text.strip():
                raise ValueError(f'{source}: missing task instruction')
            first_frames = {}
            for key, source_key in cameras.items():
                colors = f[f'vision/{source_key}/colors']
                if colors.shape[0] != n:
                    raise ValueError(f'{source}: unsynchronized {source_key}')
                image = np.asarray(decode_image_bit(colors[0]))
                if image.shape != (480, 640, 3) or image.dtype != np.uint8:
                    raise ValueError(f'{source}: unexpected decoded image {image.shape}')
                first_frames[key] = image
        converter.convert_one(source, dataset, 'RoboDojo', 'v1.0', [7, 7], [7, 7], 480, 640)
        root = staging_parent / task
        table = pq.read_table(root / f'data/chunk-000/episode_{episode:06d}.parquet')
        for key, expected in [('action', action), ('observation.state', state)]:
            np.testing.assert_array_equal(np.asarray(table[key].to_pylist(), dtype=np.float32), expected)
        np.testing.assert_allclose(table['timestamp'].to_numpy(), np.arange(n) / 25, atol=1e-4)
        for camera, expected in first_frames.items():
            video_path = root / f'videos/chunk-000/observation.images.{camera}/episode_{episode:06d}.mp4'
            with av.open(str(video_path)) as video:
                stream = video.streams.video[0]
                if stream.frames != n or float(stream.average_rate) != 25:
                    raise ValueError(f'{video_path}: wrong frame count/rate {stream.frames}/{stream.average_rate}')
                actual = next(video.decode(video=0)).to_ndarray(format='rgb24')
            mae = float(np.abs(actual.astype(float) - expected.astype(float)).mean())
            if mae > 12:
                raise ValueError(f'{video_path}: decoded RGB differs excessively ({mae})')
            audit['camera_first_frame_mae_max'] = max(audit['camera_first_frame_mae_max'], mae)
        audit['episode_lengths'].append(n)
        if text not in audit['instructions']:
            audit['instructions'].append(text)
        print(f'{task}: {episode + 1}/{limit}, {n} frames, RGB/action/timestamp checked', flush=True)
    if dataset.image_writer is not None:
        dataset.stop_image_writer()
    root = staging_parent / task
    info = json.loads((root / 'meta/info.json').read_text())
    assert info['total_episodes'] == limit and info['total_frames'] == sum(audit['episode_lengths'])
    audit['total_frames'] = info['total_frames']
    atomic_json(root / 'conversion_audit.json', audit)
    output.parent.mkdir(parents=True, exist_ok=True)
    root.rename(output)
    print(f'Completed {task}: {output}', flush=True)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=RUN_ROOT)
    parser.add_argument('--task', choices=TASKS)
    parser.add_argument('--limit-per-task', type=int, default=100)
    parser.add_argument('--tag', default='ready')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if args.limit_per_task != 100 and args.tag == 'ready':
        raise ValueError('Use a distinct --tag for partial conversion')
    tasks = [args.task] if args.task else TASKS
    # Spawn prevents imported LeRobot cache constants leaking between tasks.
    import multiprocessing
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(args.workers, len(tasks)),
            mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(prepare_task, args.root, task, args.limit_per_task, args.tag) for task in tasks]
        results = [future.result() for future in futures]
    if len(results) == 6 and args.limit_per_task == 100:
        atomic_json(args.root / 'conversion_complete.json', {'revision': REVISION, 'tasks': results,
                    'episodes': sum(x['episodes'] for x in results), 'frames': sum(x['total_frames'] for x in results)})


if __name__ == '__main__':
    main()
