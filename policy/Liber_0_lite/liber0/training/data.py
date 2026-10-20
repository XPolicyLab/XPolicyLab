import json
import math
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.transforms import functional as F

from ..preprocessing import build_norm

CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')
PROMPT = "A video recorded from a robot's point of view executing the following instruction: "


def transform_frames(frames, cue, resize, crop_scale, training, color, seed):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        frames = F.resize(frames.flatten(0, 1).float() / 255, resize, antialias=True).reshape(3, 2, 3, *resize)
        cue = F.resize(cue.float() / 255, resize, antialias=True)
        height, width = resize
        crop_h, crop_w = max(1, round(height * crop_scale)), max(1, round(width * crop_scale))
        cropped = []
        for index in range(3):
            top = int(torch.randint(height - crop_h + 1, ()).item()) if training else (height - crop_h) // 2
            left = int(torch.randint(width - crop_w + 1, ()).item()) if training else (width - crop_w) // 2
            cropped.append(F.resized_crop(frames[index], top, left, crop_h, crop_w, resize, antialias=True))
            if index == 0:
                cue = F.resized_crop(cue, top, left, crop_h, crop_w, resize, antialias=True)
        frames = torch.stack(cropped)
        if training and torch.rand(()).item() < color['probability']:
            torch.randint(0, 1, ())
            values = torch.cat([frames.flatten(0, 1), cue.unsqueeze(0)])
            operations = []
            for key, function in [('brightness', F.adjust_brightness), ('contrast', F.adjust_contrast),
                                  ('saturation', F.adjust_saturation), ('hue', F.adjust_hue)]:
                extent = color[key]
                if extent > 0:
                    low, high = (-extent, extent) if key == 'hue' else (max(0., 1 - extent), 1 + extent)
                    factor = float(torch.empty(()).uniform_(low, high).item())
                    operations.append((function, factor))
            for index in torch.randperm(len(operations)).tolist():
                function, factor = operations[index]
                values = function(values, factor)
            values = values.clamp(0, 1)
            frames, cue = values[:6].reshape_as(frames), values[6]
        frames = F.resize(frames.flatten(0, 1), [144, 192], antialias=True).reshape(3, 2, 3, 144, 192)
        cue = F.resize(cue, [144, 192], antialias=True).expand_as(frames[0])
        mosaic = torch.cat([torch.cat([frames[0], cue], -1), torch.cat([frames[1], frames[2]], -1)], -2)
        return ((mosaic - .5) / .5).permute(1, 0, 2, 3)


class RoboDojoDataset(Dataset):
    def __init__(self, root, stats, inference, *, seed, training, color):
        self.root = Path(root)
        self.info = json.loads((self.root / 'meta/info.json').read_text())
        if self.info['codebase_version'] != 'v2.1':
            raise ValueError('Expected LeRobot v2.1 video data.')
        self.episodes = [json.loads(row) for row in (self.root / 'meta/episodes.jsonl').read_text().splitlines()]
        self.tasks = {row['task_index']: row['task'] for row in map(json.loads, (self.root / 'meta/tasks.jsonl').read_text().splitlines())}
        if len({row['episode_index'] for row in self.episodes}) != len(self.episodes):
            raise ValueError('Duplicate episode indices.')
        self.ends = np.cumsum([row['length'] for row in self.episodes]).tolist()
        if not self.ends or self.ends[-1] != self.info['total_frames'] or any(row['length'] <= 0 for row in self.episodes):
            raise ValueError('Invalid episode lengths.')
        for key in ('action', 'observation.state'):
            if self.info['features'][key]['shape'] != [14]:
                raise ValueError(f'Expected 14-D {key}.')
        for camera in CAMERAS:
            if self.info['features'][f'observation.images.{camera}']['dtype'] != 'video':
                raise ValueError('Expected RGB video observations.')
        self.norm = build_norm(stats)
        processor = inference['processor']
        if processor['norm_default_mode'] != 'z-score' or processor['use_stepwise_action_norm']:
            raise ValueError('Expected global z-score normalization.')
        if inference['robotwin_camera_layout'] != 'four_grid_288x384':
            raise ValueError('Expected four_grid_288x384.')
        self.resize = processor['val_transforms'][1]['size']
        self.crop_scale = inference['image_crop_scale']
        if not 0 < self.crop_scale <= 1:
            raise ValueError('image_crop_scale must be in (0, 1].')
        self.horizon = int(inference['action_horizon'])
        self.seed, self.training, self.color = seed, training, color
        self._cache = OrderedDict()

    def __len__(self):
        return self.ends[-1]

    def episode_paths(self, episode):
        values = dict(episode_index=episode, episode_chunk=episode // self.info['chunks_size'])
        data = self.root / self.info['data_path'].format(**values)
        videos = [self.root / self.info['video_path'].format(**values, video_key=f'observation.images.{camera}') for camera in CAMERAS]
        return data, videos

    def _load(self, episode, length):
        if episode in self._cache:
            self._cache.move_to_end(episode)
            return self._cache[episode]
        from torchcodec.decoders import VideoDecoder
        data, paths = self.episode_paths(episode)
        table = pq.read_table(data).to_pydict()
        if table['frame_index'] != list(range(length)) or table['episode_index'] != [episode] * length:
            raise ValueError(f'Invalid episode indexing: {data}')
        arrays = {key: torch.tensor(table[key], dtype=torch.float32) for key in ('action', 'observation.state')}
        if any(value.shape != (length, 14) or not torch.isfinite(value).all() for value in arrays.values()):
            raise ValueError(f'Invalid state/action values: {data}')
        videos = [VideoDecoder(str(path), device='cpu', num_ffmpeg_threads=1) for path in paths]
        if any(len(video) != length for video in videos):
            raise ValueError(f'Video/frame count mismatch: {data}')
        record = arrays, table['task_index'], videos
        self._cache[episode] = record
        if len(self._cache) > 2:
            self._cache.popitem(last=False)
        return record

    def __getitem__(self, item):
        index, epoch = item if isinstance(item, tuple) else (item, 0)
        if not 0 <= index < len(self):
            raise IndexError(index)
        slot = bisect_right(self.ends, index)
        record = self.episodes[slot]
        frame = index - (self.ends[slot - 1] if slot else 0)
        length = record['length']
        values, tasks, videos = self._load(record['episode_index'], length)
        future = min(frame + self.horizon, length - 1)
        images = torch.stack([video.get_frames_at([frame, future]).data for video in videos])
        cue = videos[0].get_frames_at([0]).data[0]
        video = transform_frames(images, cue, self.resize, self.crop_scale, self.training, self.color,
                                 self.seed + epoch * len(self) + index)
        indices = torch.arange(frame, frame + self.horizon)
        action = values['action'][indices.clamp_max(length - 1)]
        action = torch.clamp(action * self.norm['action_scale'] + self.norm['action_offset'], -5, 5)
        state = values['observation.state'][frame]
        state = torch.clamp(state * self.norm['state_scale'] + self.norm['state_offset'], -5, 5)
        return dict(video=video, action=action, proprio=state,
                    action_is_pad=indices >= length, prompt=PROMPT + self.tasks[tasks[frame]],
                    episode_index=record['episode_index'], frame_index=frame, target_frame_index=future)


class EpochBatchSampler(Sampler):
    def __init__(self, size, batch_size, rank, world_size, seed, epoch=0, start_batch=0):
        if size <= 0 or batch_size <= 0 or not 0 <= rank < world_size:
            raise ValueError('Invalid sampler dimensions.')
        self.size, self.batch_size, self.rank, self.world_size = size, batch_size, rank, world_size
        self.seed, self.epoch, self.start_batch = seed, epoch, start_batch
        self.batches = math.ceil(size / (batch_size * world_size))
        if not 0 <= start_batch <= self.batches:
            raise ValueError('Invalid resume batch cursor.')

    def __len__(self):
        return self.batches - self.start_batch

    def __iter__(self):
        order = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed + self.epoch))
        total = self.batches * self.batch_size * self.world_size
        order = order.repeat(math.ceil(total / self.size))[:total].reshape(self.batches, self.world_size, self.batch_size)
        for batch in order[self.start_batch:, self.rank]:
            yield [(int(index), self.epoch) for index in batch]


class BatchStream:
    def __init__(self, data, *, batch_size, rank, world_size, seed, workers, progress, pin_memory=False):
        self.data, self.batch_size, self.rank, self.world = data, batch_size, rank, world_size
        self.seed, self.workers, self.progress = seed, workers, progress
        self.pin_memory = pin_memory
        self.batches = EpochBatchSampler(len(data), batch_size, rank, world_size, seed).batches
        self.loader = None

    def __next__(self):
        if self.progress['batch'] == self.batches:
            self.progress['epoch'] += 1
            self.progress['batch'] = 0
            self.loader = None
        if self.loader is None:
            sampler = EpochBatchSampler(len(self.data), self.batch_size, self.rank, self.world,
                                        self.seed, self.progress['epoch'], self.progress['batch'])
            self.loader = DataLoader(self.data, batch_sampler=sampler, num_workers=self.workers,
                                     generator=torch.Generator().manual_seed(self.seed + self.progress['epoch']),
                                     pin_memory=self.pin_memory, persistent_workers=False)
            self.iterator = iter(self.loader)
        sample = next(self.iterator)
        self.progress['batch'] += 1
        return sample
