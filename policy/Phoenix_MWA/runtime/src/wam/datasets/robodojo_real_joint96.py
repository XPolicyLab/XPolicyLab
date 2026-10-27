"""Native LeRobot v3 real collection, joint14 labels and Robot37 prompt IDs.

No dataset conversion, FPS resampling, video cache or random error replacement.
The current-only Joint96 output contract and native WAMProcessor are reused.
"""
from bisect import bisect_right
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from .joint96_posttrain import Joint96PosttrainDataset
from .robotwin_joint96_io import FramewiseResize, official_text_padding
from .dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from .lerobot.robot_video_dataset import DEFAULT_PROMPT
from .lerobot.utils.normalizer import load_dataset_stats_from_json
from .lerobot.lerobot.datasets.video_utils import decode_video_frames

ROBOT_IDS = {'arx_x5': 2, 'piper': 4, 'piper_x': 15}
CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')
CONTRACT = 'robodojo.real.v3.joint14.robot37.v1'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def episode_arrays(root, info, episode, table_cache=None):
    path = Path(root) / info['data_path'].format(
        chunk_index=int(episode['data/chunk_index']), file_index=int(episode['data/file_index']))
    key = str(path)
    if table_cache is None or key not in table_cache:
        table = pq.read_table(path, columns=['action.joint', 'observation.joint_state',
                                            'episode_index', 'frame_index', 'timestamp'])
        if table_cache is not None:
            table_cache[key] = table
            while len(table_cache) > 2:
                table_cache.popitem(last=False)
    else:
        table = table_cache[key]
        table_cache.move_to_end(key)
    rows = table.filter(__import__('pyarrow').compute.equal(table['episode_index'], int(episode['episode_index'])))
    result = {k: np.asarray(rows[k].to_pylist(), dtype=np.float32 if k != 'frame_index' else np.int64)
              for k in ('action.joint', 'observation.joint_state', 'frame_index', 'timestamp')}
    n = int(episode['length'])
    if (len(rows) != n or result['action.joint'].shape != (n, 14)
            or result['observation.joint_state'].shape != (n, 14)
            or not np.array_equal(result['frame_index'], np.arange(n))
            or not all(np.isfinite(v).all() for v in result.values())):
        raise ValueError(f'Corrupt native joint14 episode: {root}:{episode["episode_index"]}')
    if not np.allclose(np.diff(result['timestamp']), 1.0 / info['fps'], atol=2e-4, rtol=0):
        raise ValueError('Episode timestamps differ from declared native FPS')
    return result


class RoboDojoRealJoint96Dataset(Joint96PosttrainDataset):
    """Separate v3 reader; no modifications to the existing v2 reader."""
    def __init__(self, *, package_root, data_root, domain, camera_slots, shape_meta, processor,
                 video_size, is_training_set, image_color_jitter=None, validation_panel_size=128,
                 **kwargs):
        torch.utils.data.Dataset.__init__(self)
        if domain != 'robodojo' or list(camera_slots) != [0, 1, 2] or list(video_size) != [256, 256]:
            raise ValueError('Expected RoboDojo real Joint96 three-view geometry')
        for key, expected in {'num_frames': 33, 'action_video_freq_ratio': 4,
                              'memory_video_anchor_size': 0, 'memory_recent_frame_offset': 0,
                              'global_sample_stride': 1, 'val_set_proportion': 0}.items():
            if kwargs.get(key, expected) != expected:
                raise ValueError(f'Unexpected real-data sampling: {key}')
        self.package_root, self.data_root = Path(package_root), Path(data_root)
        self.manifest = json.loads((self.package_root / 'manifest.json').read_text())
        if self.manifest['contract'] != CONTRACT or self.manifest['prompt_template'] != DEFAULT_PROMPT:
            raise ValueError('Wrong real-data package contract')
        self.is_training_set = bool(is_training_set)
        self.domain = domain
        self.camera_slots = tuple(camera_slots)
        self.video_view_names = list(CAMERAS)
        self.num_frames, self.action_video_freq_ratio = 33, 4
        self.global_sample_stride = 1
        self.memory_video_anchor_size = self.memory_recent_frame_offset = 0
        self.concat_multi_camera, self.single_canvas = 'latent_horizontal', False
        self.sync_vlm_color_jitter = self.sync_vlm_camera_jitter = False
        self.return_vlm_current_images = False
        self.proprio_noise_std, self.override_instruction = 0., None
        self.image_color_jitter = self._parse_image_color_jitter(image_color_jitter)
        self.image_camera_jitter = self.image_sensor_noise = None
        self.resize_transform = FramewiseResize(ResizeSmallestSideAspectPreserving(args={'img_w':256,'img_h':256}))
        self.crop_transform = CenterCrop(args={'img_w':256,'img_h':256})
        self.normalize_transform = Normalize(args={'mean':.5,'std':.5})
        self.processors = {}
        for eid in (2, 4, 15):
            p = instantiate(processor) if OmegaConf.is_config(processor) or isinstance(processor, dict) else deepcopy(processor)
            p.train() if self.is_training_set else p.eval()
            p.set_normalizer_from_stats(load_dataset_stats_from_json(self.package_root / f'stats_slot{eid}.json'))
            self.processors[eid] = p
        self._episodes, self._valid_ranges, self._valid_ends = [], [], []
        cursor = 0
        for task in self.manifest['tasks']:
            root = self.data_root / task['name']
            info = json.loads((root / 'meta/info.json').read_text())
            if sha256(root/'meta/info.json') != task['info_sha256'] or sha256(root/'meta/tasks.parquet') != task['tasks_sha256']:
                raise ValueError('Source metadata changed after package preparation')
            if info['codebase_version'] != 'v3.0' or ROBOT_IDS[info['robot_type']] != task['embodiment_id']:
                raise ValueError('Source robot identity mismatch')
            for ep in task['episodes']:
                count = max(0, int(ep['length']) - 4)
                if count:
                    self._episodes.append((root, info, task, ep))
                    self._valid_ranges.append((cursor, cursor+count))
                    cursor += count
                    self._valid_ends.append(cursor)
        self._parquet_cache = OrderedDict()
        self._last_episode = None
        self._last_arrays = None
        self._text = None
        if not self._valid_ends:
            raise ValueError('Empty real dataset')
        if not self.is_training_set:
            count = min(int(validation_panel_size), len(self))
            self.fixed_validation_panel = [dict(index=int((i+.5)*len(self)/count),
                source='robodojo_real', seed=20261009+i,
                embodiment_id=self._episodes[bisect_right(self._valid_ends, int((i+.5)*len(self)/count))][2]['embodiment_id'])
                for i in range(count)]

    def _raw_index(self, index):
        if not 0 <= int(index) < len(self):
            raise IndexError(index)
        return int(index)

    def locate(self, index):
        e = bisect_right(self._valid_ends, int(index))
        return e, int(index) - self._valid_ranges[e][0]

    def _read_window(self, raw_index):
        e, start = self.locate(raw_index)
        root, info, task, ep = self._episodes[e]
        if self._last_episode != e:
            self._last_arrays = episode_arrays(root, info, ep, self._parquet_cache)
            self._last_episode = e
        rows = self._last_arrays
        length = int(ep['length'])
        image_idx = start + np.arange(0, 33, 4)
        action_idx = start + np.arange(32)
        state_idx = start + np.arange(33)
        def bounded(x): return np.minimum(x, length-1)
        images = {}
        for camera in CAMERAS:
            key = 'observation.images.' + camera
            prefix = 'videos/' + key
            path = root / info['video_path'].format(video_key=key,
                chunk_index=int(ep[prefix+'/chunk_index']), file_index=int(ep[prefix+'/file_index']))
            times = (float(ep[prefix+'/from_timestamp']) +
                     rows['timestamp'][bounded(image_idx)].astype(np.float64) - float(rows['timestamp'][0]))
            frames = decode_video_frames(path, times.tolist(), tolerance_s=.51 / info['fps'])
            if tuple(frames.shape) != (9, 3, 256, 256):
                raise ValueError(f'Wrong source video dimensions: {path}')
            images[camera] = (frames*255).to(torch.uint8)
        raw = dict(idx=raw_index, task=task['instruction'], images=images,
            action={'default':torch.from_numpy(rows['action.joint'][bounded(action_idx)].copy())},
            state={'default':torch.from_numpy(rows['observation.joint_state'][bounded(state_idx)].copy())},
            action_is_pad=torch.from_numpy(action_idx>=length),
            state_is_pad=torch.from_numpy(state_idx>=length),
            image_is_pad=torch.from_numpy(image_idx>=length))
        return self.processors[task['embodiment_id']].preprocess(raw)

    def _format_video_window(self, *args, **kwargs):
        value = super()._format_video_window(*args, **kwargs)
        if not torch.isfinite(value).all():
            raise ValueError('Nonfinite augmented RGB')
        return value.clamp(-1, 1)

    def _get_cached_text_context(self, prompt):
        if self._text is None:
            self._text = torch.load(self.package_root/'text.pt', map_location='cpu', weights_only=True)
            if self._text['prompt_template'] != DEFAULT_PROMPT:
                raise ValueError('Wrong T5 template')
        row = self._text['rows'][prompt]
        return official_text_padding(row['context'], row['mask'])

    def __getitem__(self, index):
        result = super().__getitem__(index)
        e, _ = self.locate(index)
        result['embodiment_id'] = torch.tensor(self._episodes[e][2]['embodiment_id'], dtype=torch.long)
        return result
