import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from XPolicyLab.policy.Liber_0_lite.liber0.preprocessing import pack_images
from XPolicyLab.policy.Liber_0_lite.liber0.training.data import BatchStream, EpochBatchSampler, RoboDojoDataset, transform_frames

COLOR = dict(probability=.5, brightness=.3, contrast=.4, saturation=.5, hue=.05)


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    import torchcodec.decoders
    (tmp_path / 'meta').mkdir()
    (tmp_path / 'data').mkdir()
    features = {key: dict(shape=[14]) for key in ('action', 'observation.state')}
    features.update({f'observation.images.{camera}': dict(dtype='video') for camera in
                     ('cam_high', 'cam_left_wrist', 'cam_right_wrist')})
    info = dict(codebase_version='v2.1', total_frames=10, chunks_size=1000, features=features,
                data_path='data/{episode_index}.parquet', video_path='videos/{video_key}/{episode_index}.mp4')
    (tmp_path / 'meta/info.json').write_text(json.dumps(info))
    episodes = [dict(episode_index=i, length=length) for i, length in enumerate((4, 6))]
    (tmp_path / 'meta/episodes.jsonl').write_text('\n'.join(map(json.dumps, episodes)))
    (tmp_path / 'meta/tasks.jsonl').write_text(json.dumps(dict(task_index=0, task='Move the cup.')))
    for episode in episodes:
        n, index = episode['length'], episode['episode_index']
        pq.write_table(pa.table({'action': [[float(i)] * 14 for i in range(n)],
                                'observation.state': [[float(i + 1)] * 14 for i in range(n)],
                                'frame_index': list(range(n)), 'episode_index': [index] * n,
                                'task_index': [0] * n}), tmp_path / f'data/{index}.parquet')

    class Decoder:
        def __init__(self, path, **kwargs):
            self.length = episodes[int(Path(path).stem)]['length']

        def __len__(self):
            return self.length

        def get_frames_at(self, indices):
            return SimpleNamespace(data=torch.stack([torch.full((3, 24, 32), 10 * i, dtype=torch.uint8) for i in indices]))

    monkeypatch.setattr(torchcodec.decoders, 'VideoDecoder', Decoder)
    stats = {key: dict(default=dict(global_mean=[0] * 14, global_std=[1] * 14)) for key in ('action', 'state')}
    inference = dict(robotwin_camera_layout='four_grid_288x384', image_crop_scale=.95, action_horizon=32,
                     processor=dict(norm_default_mode='z-score', use_stepwise_action_norm=False,
                                    val_transforms=[{}, dict(size=[240, 320])]))
    return RoboDojoDataset(tmp_path, stats, inference, seed=42, training=False, color=COLOR)


def test_window_padding_and_episode_boundary(dataset):
    first, last, second = dataset[0], dataset[3], dataset[4]
    assert first['action_is_pad'].sum() == 28
    assert last['action_is_pad'].sum() == 31
    assert second['frame_index'] == 0 and second['episode_index'] == 1
    assert last['target_frame_index'] == 3
    assert torch.equal(first['action'][:4, 0], torch.arange(4).float())
    assert torch.equal(last['action'][:, 0], torch.full((32,), 3.))
    assert torch.equal(last['video'][:, 0, :144, 192:], last['video'][:, 1, :144, 192:])
    assert (last['video'][:, :, :144, 192:] == -1).all()
    assert first['prompt'].endswith('Move the cup.')


def test_image_processing_matches_inference():
    generator = torch.Generator().manual_seed(20)
    frames = torch.randint(0, 256, (3, 2, 3, 48, 64), dtype=torch.uint8, generator=generator)
    cue = torch.randint(0, 256, (3, 48, 64), dtype=torch.uint8, generator=generator)
    result = transform_frames(frames, cue, [240, 320], .95, False, COLOR, 42)
    for time in range(2):
        images = [value.permute(1, 2, 0).numpy() for value in frames[:, time]] + [cue.permute(1, 2, 0).numpy()]
        assert torch.equal(result[:, time], pack_images(images, [240, 320], .95)[0])


def test_augmentation_repeat_and_rng_isolation(dataset):
    dataset.training = True
    before = torch.get_rng_state()
    first = dataset[(2, 0)]
    second = dataset[(2, 0)]
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(first['video'], second['video'])


@pytest.mark.parametrize('size,world,batch', [(10, 3, 2), (1, 2, 4), (24, 2, 3)])
def test_sampler_resume(size, world, batch):
    all_indices = []
    for rank in range(world):
        sampler = EpochBatchSampler(size, batch, rank, world, 42, 3)
        rows = list(sampler)
        assert list(EpochBatchSampler(size, batch, rank, world, 42, 3, 1)) == rows[1:]
        assert all(epoch == 3 for row in rows for _, epoch in row)
        all_indices.extend(index for row in rows for index, _ in row)
    assert set(all_indices) == set(range(size))
    assert len(all_indices) % (world * batch) == 0


def test_worker_count_does_not_change_samples(dataset):
    from torch.utils.data import DataLoader
    dataset.training = True
    rows = []
    for workers in (0, 2):
        loader = DataLoader(dataset, batch_sampler=EpochBatchSampler(len(dataset), 2, 0, 1, 42), num_workers=workers,
                            generator=torch.Generator().manual_seed(10))
        rows.append([batch['video'] for batch in loader])
    assert all(torch.equal(left, right) for left, right in zip(*rows))


@pytest.mark.parametrize('consumed', [2, 5, 7])
def test_stream_resume_at_and_across_epoch_boundary(dataset, consumed):
    dataset.training = True
    progress = dict(step=0, epoch=0, batch=0)
    stream = BatchStream(dataset, batch_size=2, rank=0, world_size=1, seed=42, workers=0, progress=progress)
    for _ in range(consumed):
        next(stream)
    cursor = dict(progress)
    expected = [next(stream) for _ in range(4)]
    restored = BatchStream(dataset, batch_size=2, rank=0, world_size=1, seed=42, workers=0, progress=cursor)
    actual = [next(restored) for _ in range(4)]
    for left, right in zip(expected, actual):
        assert torch.equal(left['video'], right['video'])
        assert torch.equal(left['frame_index'], right['frame_index'])
    assert cursor == progress
