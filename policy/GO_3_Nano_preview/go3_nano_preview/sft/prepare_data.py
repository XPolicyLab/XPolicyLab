"""Validate and stage the processed Robotwin JSONL/MP4/action format."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import av
import numpy as np
import torch

from go3_nano_preview.sft.paths import POLICY_ROOT, add_standard_arguments, standard_name, check_link, publish_link


def validate_record(record, root):
    clip = Path(record['clip'])
    if clip.is_absolute() or '..' in clip.parts:
        raise ValueError('clip must be a relative path within data-root')
    directory = (root / clip).resolve()
    if not directory.is_relative_to(root):
        raise ValueError('clip resolves outside data-root')
    if not isinstance(record['task_caption'], str):
        raise ValueError('task_caption must be text')
    frames = int(record['video_frames'])
    if frames < 2 or int(record['action_frames']) < frames:
        raise ValueError('invalid frame counts')
    for step in record['step_caption']:
        if not (0 <= int(step['start_frame']) < int(step['end_frame']) <= frames):
            raise ValueError('step_caption interval exceeds video frames')
        if not isinstance(step['action_text'], str):
            raise ValueError('step caption must be text')
    for camera in ('head', 'left_hand', 'right_hand'):
        file = directory / (camera + '.mp4')
        with av.open(str(file)) as container:
            stream = container.streams.video[0]
            decoded = next(container.decode(stream)).to_ndarray(format='rgb24')
            if decoded.ndim != 3 or decoded.shape[2] != 3:
                raise ValueError(f'Invalid RGB video: {camera}')
            if stream.frames and stream.frames < frames:
                raise ValueError(f'{camera} is shorter than metadata')
    # Only use trusted local processed files: the source format may contain numpy objects.
    actions = torch.load(directory/'actions.pt', map_location='cpu', weights_only=False)
    if isinstance(actions, dict):
        raise ValueError('actions.pt must be a tensor/array of absolute joints, not a dictionary')
    arrays = [actions]
    for value in arrays:
        array = np.asarray(value)
        if array.ndim != 2 or array.shape[0] < frames or array.shape[1] != 14:
            raise ValueError(f'Expected at least {frames} rows of 14 joint/gripper values, got {array.shape}')
        if not np.isfinite(array).all():
            raise ValueError('Nonfinite action/state values')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jsonl', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output')
    parser.add_argument('--expected-episodes', type=int)
    add_standard_arguments(parser)
    args = parser.parse_args()
    source = Path(args.jsonl).resolve()
    root = Path(args.data_root).resolve()
    name = standard_name(args, include_seed=False)
    if not args.output and not name:
        parser.error('Set SFT_PREPARED_DIR or supply the four standard arguments')
    run_root = Path(os.environ.get('RUN_ROOT') or str(Path.home() / 'robotwin-runs')).expanduser()
    output = Path(args.output).expanduser().resolve() if args.output else (run_root / 'data' / name).resolve()
    link = POLICY_ROOT / 'data' / name if name else None
    if link is not None:
        check_link(link, output)
    checkout = Path(__file__).resolve().parents[2]
    if output.is_relative_to(checkout):
        raise ValueError('Prepared data must be outside the policy checkout')
    if output.exists():
        raise FileExistsError('Use a new output directory; no existing dataset will be overwritten')
    content = source.read_bytes()
    records = [json.loads(line) for line in content.splitlines() if line.strip()]
    if args.expected_episodes is not None and len(records) != args.expected_episodes:
        raise ValueError(f'Expected {args.expected_episodes} episodes, found {len(records)}')
    seen = set()
    tasks = Counter()
    for index, record in enumerate(records):
        if record['clip'] in seen:
            raise ValueError(f'Duplicate clip at record {index+1}')
        seen.add(record['clip'])
        try:
            validate_record(record, root)
        except Exception as exc:
            raise ValueError(f'Invalid record {index+1}: {record["clip"]}: {exc}') from exc
        tasks[Path(record['clip']).name.rsplit('_episode', 1)[0]] += 1
        if (index + 1) % 100 == 0:
            print(f'Checked {index+1}/{len(records)} episodes', flush=True)
    if not records:
        raise ValueError('Empty dataset')
    output.mkdir(parents=True)
    (output/'train.jsonl').write_bytes(content)
    report = dict(episodes=len(records), tasks=dict(sorted(tasks.items())), data_root=str(root),
                  jsonl_sha256=hashlib.sha256(content).hexdigest(),
                  validation='all actions finite; each camera first frame and declared frame count checked',
                  leaderboard_count_match=len(tasks)==50 and all(n==50 for n in tasks.values()))
    (output/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    if link is not None:
        publish_link(link, output)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
