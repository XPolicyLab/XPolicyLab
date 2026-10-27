import torch
import os, random
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from einops import rearrange
from decord import VideoReader
import decord
decord.bridge.set_bridge('torch')
from torchvision.io import read_video
import av
import json
import h5py
from scipy.spatial.transform import Rotation as R
import torchvision.transforms as transforms
import threading
from threading import Lock
from typing import Any
import fsspec
import importlib.util
import importlib.metadata
import os
import time
import cv2
import traceback
from torch.utils.data import get_worker_info
import resource
import gc
import collections
from time import perf_counter
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import time
import numpy as np
from torch.utils.data import Dataset as BaseDataset
from go3_nano_preview.sft.data.unified_dataset.action_utils import get_processed_actions
from go3_nano_preview.sft.data.unified_dataset.data_utils import *
from go3_nano_preview.sft.data.unified_dataset.embodiment_cue import compose_embodiment_cue, condition_caption
from go3_nano_preview.sft.data.unified_dataset.dp_bbox_conditioning import resolve_dense_prompt
from go3_nano_preview.sft.data.unified_dataset.episode_metadata import load_episode_metadata
from go3_nano_preview.sft.data.unified_dataset.sampling_weight_cache import CACHE_DIRNAME, SamplingWeightCacheError, build_cache_identity, cache_path_for_identity, load_sampling_weights, save_sampling_weights
from policy_core.utils.WAM_policy.body_embedding import DEFAULT_BODY_EMBEDDING_MAPPING, DEFAULT_NUM_BODY_EMBEDDINGS, body_type_for_emb_id, infer_body_type_from_dataset_name, resolve_body_emb_id, validate_body_embedding_mapping

def timeit_block(results_dict, key):

    class Timer:

        def __enter__(self):
            self.start = perf_counter()

        def __exit__(self, *args):
            results_dict[f'time_{key}'] = perf_counter() - self.start
    return Timer()

def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    return bool(value)

@dataclass(frozen=True)
class EpisodeSampleSelection:
    end_frames: int
    end_frames_action: int
    video_slices: tuple[int, ...]
    action_slices: tuple[int, ...]
    sample_metadata: dict[str, Any]
from go3_nano_preview.sft.data.dataloader.worker import accv_worker_init_fn
from policy_core.settings import settings
from go3_nano_preview.sft.data.transforms import Compose, NormalizeVideo, RandomVideoCrop, RandomVideoPadding, ResizeVideo
from go3_nano_preview.sft.data.utils.WAM_policy_camera import infer_camera_view_ids
class VideoDecoderCache:

    def __init__(self):
        self._cache: dict[str, tuple[Any, Any]] = {}
        self._lock = Lock()

    def get_decoder(self, video_path: str):
        if importlib.util.find_spec('torchcodec'):
            from torchcodec.decoders import VideoDecoder
        else:
            raise ImportError('torchcodec is required but not available.')
        video_path = str(video_path)
        with self._lock:
            try:
                if video_path not in self._cache:
                    decoder = VideoDecoder(video_path, seek_mode='approximate')
                    self._cache[video_path] = (decoder, None)
            except Exception as e:
                raise RuntimeError(f'torchcodec_DECODE_ERROR|{video_path}')
            return self._cache[video_path][0]

    def clear(self):
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._cache)

class AccvDecoderContractError(RuntimeError):
    pass

def _validate_accv_max_resolution(max_width, max_height):
    if max_width is None and max_height is None:
        return None
    if type(max_width) is not int or type(max_height) is not int or max_width <= 0 or (max_height <= 0):
        raise ValueError('accv_max_width and accv_max_height must both be positive integers, or both be omitted')
    return (max_width, max_height)

def _load_accv_reconfigure_backend():
    required_version = '0.2.0.post1.dev67+hotfix.g426d7fff5c'
    try:
        installed_version = importlib.metadata.version('accvlab-on-demand-video-decoder')
    except importlib.metadata.PackageNotFoundError as error:
        raise AccvDecoderContractError(f'ACCV reconfiguration requires accvlab-on-demand-video-decoder=={required_version}') from error
    if installed_version != required_version:
        raise AccvDecoderContractError(f'ACCV reconfiguration requires {required_version}; found {installed_version}')
    import accvlab.on_demand_video_decoder as nvc
    return nvc

def _probe_accv_streams(file_paths):
    stream_infos = []
    for path in file_paths:
        with av.open(path) as container:
            stream = container.streams.best('video')
            if stream is None:
                raise AccvDecoderContractError('ACCV request contains a file without a video stream')
            codec = stream.codec_context
            if codec.format is None:
                raise AccvDecoderContractError('Cannot determine the video pixel format for ACCV reuse')
            stream_infos.append((codec.width, codec.height, codec.name, codec.format.name))
    return stream_infos

class SmartDecoderManager:

    def __init__(self, gpuid, max_instances_per_worker=8, max_width=None, max_height=None):
        self.gpuid = gpuid
        self.max_resolution = _validate_accv_max_resolution(max_width, max_height)
        self._reconfigure_backend = _load_accv_reconfigure_backend() if self.max_resolution is not None else None
        self.max_instances_per_worker = max(1, int(max_instances_per_worker))
        self._cache = collections.OrderedDict()
        self.pid = os.getpid()
        self.log_file = 'decoder_debug.log'
        self.history_keys = set()

    def _log(self, message):
        timestamp = time.strftime('%H:%M:%S', time.localtime())
        current_active_keys = list(self._cache.keys())
        log_entry = f"[{timestamp}] [PID:{self.pid}] [GPU:{self.gpuid}]\n  - Message: {message}\n  - Active Keys ({len(current_active_keys)}/{self.max_instances_per_worker}): {current_active_keys}\n{'-' * 80}"

    def get_key(self, dataset_name, cam_names, resolutions, video_path=''):
        resolutions = resolutions or {}
        search_str = f'{dataset_name}_{video_path}'
        if 'tmp' in search_str:
            group = 'tmp'
        elif 'acone' in search_str:
            group = 'acone'
        elif 'agilex' in search_str:
            group = 'agilex'
        elif 'arxvr' in search_str:
            group = 'arxvr'
        elif 'processed' in search_str:
            group = 'opensource'
        elif 'sim' in search_str:
            group = 'sim'
        else:
            group = 'COMM'
        sorted_cams = sorted(cam_names)
        res_list = []
        for cam in sorted_cams:
            res = resolutions.get(cam, [640, 480])
            res_list.append(f'{res[0]}x{res[1]}')
        res_key = f"{group}_n{len(cam_names)}_{'_'.join(res_list)}"
        return res_key

    def get_reconfigure_request(self, file_paths):
        if self._reconfigure_backend is None:
            raise AccvDecoderContractError('Decoder reconfiguration is not enabled')
        stream_infos = _probe_accv_streams(file_paths)
        if not file_paths or not isinstance(stream_infos, (tuple, list)) or len(stream_infos) != len(file_paths):
            raise AccvDecoderContractError('ACCV stream information must cover every requested file')
        (max_width, max_height) = self.max_resolution
        slot_formats = []
        for info in stream_infos:
            try:
                (width, height, codec_name, pixel_format) = info
            except (TypeError, ValueError) as error:
                raise AccvDecoderContractError('ACCV returned incomplete stream information') from error
            if type(width) is not int or width <= 0 or type(height) is not int or (height <= 0) or (not isinstance(codec_name, str)) or (not codec_name) or (not isinstance(pixel_format, str)) or (not pixel_format):
                raise AccvDecoderContractError('ACCV returned invalid stream dimensions or pixel format')
            if width > max_width or height > max_height:
                raise AccvDecoderContractError(f'Video {width}x{height} exceeds ACCV decoder ceiling {max_width}x{max_height}')
            slot_formats.append((codec_name, pixel_format))
        cache_key = ('reconfigure', self.max_resolution, tuple(slot_formats))
        return (cache_key, stream_infos)

    def get_decoder(self, res_key, maxfiles):
        self.history_keys.add(res_key)
        if res_key in self._cache:
            item = self._cache[res_key]
            if item['pid'] == self.pid:
                self._cache.move_to_end(res_key)
                self._log(f'HIT: {res_key}')
                return item['instance']
        if len(self._cache) >= self.max_instances_per_worker:
            (old_key, old_item) = self._cache.popitem(last=False)
            old_decoder = old_item.get('instance')
            self._release_decoder_instance(old_decoder, release_pool=True)
            del old_item['instance']
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        try:
            if self._reconfigure_backend is None:
                import accvlab.on_demand_video_decoder as nvc
                decoder = nvc.CreateGopDecoder(maxfiles=maxfiles, iGpu=self.gpuid)
            else:
                (max_width, max_height) = self.max_resolution
                decoder = self._reconfigure_backend.CreateGopDecoder(maxfiles=maxfiles, iGpu=self.gpuid, maxWidth=max_width, maxHeight=max_height)
            self._cache[res_key] = {'instance': decoder, 'pid': self.pid}
            return decoder
        except Exception as e:
            raise e

    def purge_decoder(self, res_key):
        if res_key in self._cache:
            item = self._cache.pop(res_key)
            decoder = item.get('instance')
            self._release_decoder_instance(decoder, release_pool=True)
            del item
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _release_decoder_instance(self, decoder, release_pool=True):
        if decoder is None:
            return
        try:
            if hasattr(decoder, 'release_decoder'):
                decoder.release_decoder()
        except Exception as e:
            self._log(f'release_decoder failed: {e}')
        if release_pool:
            try:
                if hasattr(decoder, 'release_device_memory'):
                    decoder.release_device_memory()
            except Exception as e:
                self._log(f'release_device_memory failed: {e}')

    def release_device_memory(self, res_key):
        item = self._cache.get(res_key)
        if item is None:
            return
        if item.get('pid') != self.pid:
            return
        decoder = item.get('instance')
        if decoder is None:
            return
        try:
            decoder.release_device_memory()
        except Exception as e:
            self._log(f'release_device_memory failed for {res_key}: {e}')

class MixBase(BaseDataset):

    def __init__(self, jsonl_path_list, video_folder_list, dataset_source_list, dataset_name_list, fps_list=None, cam_use=[['head', 'left_hand', 'right_hand']], view_ids=None, train_modes=None, n_view=3, sample_size=(192, 256), repeat=None, preprocess='resize', fix_timestep=False, ignore_seek=False, use_adc=True, random_crop=False, random_padding=False, decode_type='cpu', gpuid=0, codec='h265', return_caption=True, use_step_caption=False, use_dense_prompt=False, dense_prompt_dropout_prob=0.6, dense_prompt_field='dense_prompt', use_recaption=False, task_recap_file=None, step_recap_file=None, return_action=False, return_video=True, n_previous_action=1, filter_action=False, norm_action=False, action_space_list=[], stat_file_list=[], action_key='action', state_key='state', jsonl_repeat_times=1, mode='train', use_sampling_weights=False, worker_gc_interval=10000, **kwargs):
        super().__init__()
        self._mode = mode
        zero_rank_print(f'loading annotations...')
        self._use_sampling_weights = _as_bool(use_sampling_weights)
        self._sampling_weights_resolved = False
        self._sampling_weights = None
        metadata_index_manifest = kwargs.get('metadata_index_manifest')
        self.metadata_mmap_cache_size = int(kwargs.get('metadata_mmap_cache_size', 16))
        self.source_weights = kwargs.get('source_weights')
        self.episode_metadata_cache_size = int(kwargs.get('episode_metadata_cache_size', 128))
        self.recap_cache_size = int(kwargs.get('recap_cache_size', 512))
        if self.episode_metadata_cache_size < 0:
            raise ValueError('episode_metadata_cache_size must be non-negative')
        if self.recap_cache_size < 0:
            raise ValueError('recap_cache_size must be non-negative')
        if metadata_index_manifest and self.metadata_mmap_cache_size <= 0:
            raise ValueError('metadata_mmap_cache_size must be positive when indexed metadata is enabled')
        self.random_crop = random_crop
        self.n_view = n_view
        self.return_action = return_action
        self.return_video = return_video
        self.dataset = []
        self.video_folder = []
        self.cam_name = []
        self.view_ids = []
        self.sample_source_indices = []
        self.sample_fps = []
        self.mix_captions = _as_bool(kwargs.get('mix_captions', False))
        self.step_goal_prob = kwargs.get('step_goal_prob', 0.0)
        self.speed_cue_prob = float(kwargs.get('speed_cue_prob', 0.0))
        _thr = kwargs.get('speed_cue_thresholds', None)
        self.speed_cue_thresholds = tuple((float(x) for x in _thr)) if _thr else None
        if self.speed_cue_prob > 0.0:
            print(f'Dataset __init__ speed cue ENABLED prob={self.speed_cue_prob} thresholds={self.speed_cue_thresholds}')
        self.embodiment_cue_prob = float(kwargs.get('embodiment_cue_prob', 0.0))
        self.embodiment_sim_drop_prob = float(kwargs.get('embodiment_sim_drop_prob', 0.2))
        if self.embodiment_cue_prob > 0.0:
            print(f'Dataset __init__ embodiment cue ENABLED prob={self.embodiment_cue_prob} sim_drop={self.embodiment_sim_drop_prob}')
        self.p_hand_drop = float(kwargs.get('p_hand_drop', 0.0))
        self.p_color_drop = float(kwargs.get('p_color_drop', 0.0))
        self.cue_drop_enabled = self.p_hand_drop > 0.0 or self.p_color_drop > 0.0
        self._drop_cues_fn = None
        if self.cue_drop_enabled:
            from go3_nano_preview.sft.data.unified_dataset.caption_cue_drop import drop_cues
            self._drop_cues_fn = drop_cues
            print(f'Dataset __init__ cue drop ENABLED p_hand_drop={self.p_hand_drop} p_color_drop={self.p_color_drop}')
        _empty_vals = kwargs.get('empty_caption_values', None)
        if _empty_vals is None:
            _empty_vals = ['', 'none', 'n/a', 'null', 'nan']
        self._empty_caption_values = {str(v).strip().lower() for v in _empty_vals}
        self.target_dim = kwargs.get('target_dim')
        self._use_step_goal = False
        self._step_goal_speed_value = None
        self._recap_trace = []
        self.dataset_name_list = []
        self.dataset_source_list = []
        self.action_space_list = []
        self.body_type_list = []
        self.body_emb_ids = []
        self.stat_info = {}
        self.domain_list = []
        self.fix_timestep = fix_timestep
        self.use_step_caption = use_step_caption
        self.use_dense_prompt = _as_bool(use_dense_prompt)
        self.dense_prompt_dropout_prob = float(dense_prompt_dropout_prob)
        self.dense_prompt_field = str(dense_prompt_field)
        if not 0.0 <= self.dense_prompt_dropout_prob <= 1.0:
            raise ValueError('dense_prompt_dropout_prob must be in [0, 1]')
        if self.use_dense_prompt and (not self.use_step_caption):
            raise ValueError('use_dense_prompt requires use_step_caption=True')
        if self.use_dense_prompt and (not self.dense_prompt_field):
            raise ValueError('dense_prompt_field must be non-empty')
        self.ignore_seek = ignore_seek
        if self.ignore_seek:
            print('Only seek memory videos')
        self.random_padding = random_padding
        self.use_adc = use_adc
        self.use_recaption = use_recaption
        self.video_suffix = '.mp4'
        self.gpuid = gpuid
        self.codec = codec
        self.decode_type = decode_type
        assert self.decode_type in {'cpu', 'torchcodec', 'accv'}, 'Invalid decode type'
        self.accv_fallback_to_cpu = _as_bool(kwargs.get('accv_fallback_to_cpu', True))
        self.accv_max_decoders_per_worker = int(kwargs.get('accv_max_decoders_per_worker', 8))
        accv_max_resolution = _validate_accv_max_resolution(kwargs.get('accv_max_width'), kwargs.get('accv_max_height'))
        if accv_max_resolution is None:
            self.accv_max_width = None
            self.accv_max_height = None
        else:
            (self.accv_max_width, self.accv_max_height) = accv_max_resolution
            if self.decode_type != 'accv':
                raise ValueError("accv_max_width/accv_max_height require decode_type='accv'")
            _load_accv_reconfigure_backend()
        self.accv_decode_all_configured_views = _as_bool(kwargs.get('accv_decode_all_configured_views', False))
        self.accv_disable_key_after_failures = int(kwargs.get('accv_disable_key_after_failures', 0) or 0)
        accv_cpu_keys = kwargs.get('accv_cpu_keys', [])
        if isinstance(accv_cpu_keys, str):
            accv_cpu_keys = [x.strip() for x in accv_cpu_keys.split(',')]
        self.accv_cpu_keys = {str(x).strip() for x in accv_cpu_keys if str(x).strip()}
        self.accv_log_bad_videos = _as_bool(kwargs.get('accv_log_bad_videos', True))
        self.drop_log_every_n = int(kwargs.get('drop_log_every_n', 100) or 0)
        self.drop_log_tracebacks = _as_bool(kwargs.get('drop_log_tracebacks', True))
        self.drop_log_paths = _as_bool(kwargs.get('drop_log_paths', False))
        self._accv_failure_counts = collections.Counter()
        self._accv_disabled_keys = set()
        self._sample_drop_counts = collections.Counter()
        self._accv_decoder = None
        self._accv_decoder_maxfiles = None
        self._accv_decoder_pid = None
        self.filter_action = filter_action
        self.norm_action = norm_action
        self.return_caption = return_caption
        self.repeat = repeat
        max_samples_per_source = kwargs.get('max_samples_per_source', None)
        if max_samples_per_source is not None:
            max_samples_per_source = int(max_samples_per_source)
        self.episode_metadata = load_episode_metadata(jsonl_paths=jsonl_path_list, video_folders=video_folder_list, dataset_sources=dataset_source_list, dataset_names=dataset_name_list, fps_values=fps_list, camera_names=cam_use, view_ids=view_ids, action_spaces=action_space_list, stat_files=stat_file_list, repeats=repeat, manifest_path=metadata_index_manifest, episode_cache_size=self.episode_metadata_cache_size, mmap_cache_size=self.metadata_mmap_cache_size, max_samples_per_source=max_samples_per_source)
        cache_flag = kwargs.get('sampling_weight_cache')
        if cache_flag is None:
            cache_enabled = metadata_index_manifest is not None
        else:
            cache_enabled = _as_bool(cache_flag)
        cache_dir = kwargs.get('sampling_weight_cache_dir')
        if cache_enabled:
            if cache_dir:
                self._sampling_weight_cache_dir = Path(cache_dir)
            elif self.episode_metadata.manifest_path:
                self._sampling_weight_cache_dir = Path(self.episode_metadata.manifest_path).parent / CACHE_DIRNAME
            else:
                raise ValueError('sampling_weight_cache requires sampling_weight_cache_dir or indexed metadata')
        else:
            self._sampling_weight_cache_dir = None
        self.dataset = self.episode_metadata
        self.repeat = list(self.episode_metadata.repeats)
        source_configs = self.episode_metadata.source_configs
        dataset_name_list = [str(config['dataset_name']) for config in source_configs]
        video_folder_list = [str(config['video_folder']) for config in source_configs]
        dataset_source_list = [str(config['dataset_source']) for config in source_configs]
        fps_list = [float(config['fps']) for config in source_configs]
        cam_use = [list(config['cam_use']) for config in source_configs]
        view_ids = [list(config.get('view_ids') or infer_camera_view_ids(cam_use[source_id])) for (source_id, config) in enumerate(source_configs)]
        action_space_list = [config.get('action_space', '') for config in source_configs]
        stat_file_list = [config.get('stat_file', '') for config in source_configs]
        for dataset_source in dataset_source_list:
            assert dataset_source == 'processed', 'Only processed Robotwin data is supported'
        if self.use_recaption:
            self.task_recap_map = self.episode_metadata.open_recap('task', task_recap_file, cache_size=self.recap_cache_size)
            self.step_recap_map = self.episode_metadata.open_recap('step', step_recap_file, cache_size=self.recap_cache_size)
        self.train_modes = self._normalize_train_modes(train_modes)
        camera_sampling_cfg = kwargs.get('camera_sampling_rules', None)
        if camera_sampling_cfg is None:
            camera_sampling_cfg = kwargs.get('camera_sampling', None)
        self.camera_sampling_rules = self._normalize_camera_sampling_rules(camera_sampling_cfg)
        body_mapping = kwargs.get('body_embedding_mapping') or DEFAULT_BODY_EMBEDDING_MAPPING
        num_body_embeddings = int(kwargs.get('num_body_embeddings', DEFAULT_NUM_BODY_EMBEDDINGS))
        body_mapping = validate_body_embedding_mapping(body_mapping, num_body_embeddings)
        body_type_by_source = kwargs.get('body_type_list')
        body_id_by_source = kwargs.get('body_emb_id_list')
        metadata_body_types = [config.get('body_type') for config in source_configs]
        metadata_body_ids = [config.get('body_emb_id') for config in source_configs]
        if any((value is not None for value in metadata_body_types)):
            if not all((value is not None for value in metadata_body_types)):
                raise ValueError('metadata body_type must be set for every source or none')
            body_type_by_source = metadata_body_types
        if any((value is not None for value in metadata_body_ids)):
            if not all((value is not None for value in metadata_body_ids)):
                raise ValueError('metadata body_emb_id must be set for every source or none')
            body_id_by_source = metadata_body_ids
        source_count = self.episode_metadata.source_count
        if self.source_weights is None:
            self.source_weights = np.ones(source_count, dtype=np.float64)
        else:
            self.source_weights = np.asarray(self.source_weights, dtype=np.float64)
            if len(self.source_weights) != source_count:
                raise ValueError('source_weights must match jsonl_path_list/source count')
            if not np.isfinite(self.source_weights).all() or (self.source_weights < 0).any() or (not (self.source_weights > 0).any()):
                raise ValueError('source_weights must be finite, non-negative, and contain at least one positive value')
        if body_type_by_source is not None and len(body_type_by_source) != source_count:
            raise ValueError(f'body_type_list must have the same length as metadata sources ({len(body_type_by_source)} != {source_count})')
        if body_id_by_source is not None and len(body_id_by_source) != source_count:
            raise ValueError(f'body_emb_id_list must have the same length as metadata sources ({len(body_id_by_source)} != {source_count})')
        body_source_debug_rows = []
        for (i, dataset_name) in enumerate(dataset_name_list):
            if len(view_ids[i]) != len(cam_use[i]):
                raise ValueError(f'view_ids length {len(view_ids[i])} does not match cam_use length {len(cam_use[i])} for dataset index {i}')
            body_type = body_type_by_source[i] if body_type_by_source is not None else infer_body_type_from_dataset_name(dataset_name)
            if body_id_by_source is not None:
                body_emb_id = int(body_id_by_source[i])
                if body_type is not None and resolve_body_emb_id(body_type, body_mapping) != body_emb_id:
                    raise ValueError(f'body_type_list[{i}]={body_type!r} maps to {resolve_body_emb_id(body_type, body_mapping)}, but body_emb_id_list[{i}]={body_emb_id}')
                if body_type is None:
                    body_type = body_type_for_emb_id(body_emb_id, body_mapping)
            else:
                if body_type is None:
                    raise ValueError(f'Cannot infer body type from dataset_name={dataset_name!r}. Extend policy_core.utils.WAM_policy.body_embedding (mapping + name inference) or set data.*.body_type_list explicitly.')
                body_emb_id = resolve_body_emb_id(body_type, body_mapping)
            if not 0 <= body_emb_id < num_body_embeddings:
                raise ValueError(f'body_emb_id={body_emb_id} for dataset {dataset_name!r} is outside [0, {num_body_embeddings - 1}]')
            body_source_debug_rows.append((dataset_name, body_type, body_emb_id))
            if self.return_action:
                if self.norm_action or self.filter_action:
                    with open(stat_file_list[i], 'r') as f:
                        self.stat_info[dataset_name] = json.load(f)
        sample_dataset_names = [dataset_name.replace('_ori', '') if dataset_source_list[source_id] in {'original_video', 'opensource_video', 'original_image'} else dataset_name for (source_id, dataset_name) in enumerate(dataset_name_list)]
        self.dataset_name_list = self.episode_metadata.project(sample_dataset_names)
        self.video_folder = self.episode_metadata.project(video_folder_list)
        self.cam_name = self.episode_metadata.project(cam_use)
        self.view_ids = self.episode_metadata.project(view_ids)
        self.sample_source_indices = self.episode_metadata.project(list(range(source_count)))
        self.sample_fps = self.episode_metadata.project(fps_list)
        self.dataset_source_list = self.episode_metadata.project(dataset_source_list)
        self.body_type_list = self.episode_metadata.project([row[1] for row in body_source_debug_rows])
        self.body_emb_ids = self.episode_metadata.project([row[2] for row in body_source_debug_rows])
        if self.return_action:
            self.action_space_list = self.episode_metadata.project(action_space_list)
        if self.episode_metadata.identity is not None:
            zero_rank_print(f'indexed metadata enabled: identity={self.episode_metadata.identity} physical_episodes={self.episode_metadata.physical_episode_count} logical_episodes={len(self.episode_metadata)} mmap_cache={self.metadata_mmap_cache_size} episode_cache={self.episode_metadata_cache_size}')
        if body_source_debug_rows:
            source_mode = 'explicit body_emb_id_list' if body_id_by_source is not None else 'explicit body_type_list' if body_type_by_source is not None else 'auto from dataset_name_list'
            rows = ', '.join((f'[{source_idx}] {dataset_name}->{body_type}:{body_emb_id}' for (source_idx, (dataset_name, body_type, body_emb_id)) in enumerate(body_source_debug_rows)))
            zero_rank_print(f'[body_embedding] source mapping ({source_mode}): {rows}')
        self.jsonl_repeat_times = int(jsonl_repeat_times)
        if isinstance(jsonl_repeat_times, bool) or self.jsonl_repeat_times <= 0 or self.jsonl_repeat_times != jsonl_repeat_times:
            raise ValueError('jsonl_repeat_times must be a positive integer')
        self.true_length = len(self.dataset)
        self.length = self.true_length * self.jsonl_repeat_times
        zero_rank_print(f'data scale: {self.length}')
        self.sample_size = sample_size
        if preprocess == 'center_crop_resize':
            self.pixel_transforms_resize = transforms.Compose([transforms.Resize(min(sample_size)), transforms.CenterCrop(sample_size)])
        if preprocess == 'resize':
            self.pixel_transforms_resize = transforms.Compose([transforms.Resize(sample_size)])
        self.pixel_transforms_norm = transforms.Compose([transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)])
        self.preprocess = preprocess
        self.n_previous_action = n_previous_action
        self.action_key = action_key
        self.state_key = state_key
        self.seek_key_frames_only = kwargs.get('seek_key_frames_only', False)
        self.worker_gc_interval = int(worker_gc_interval or 0)
        self._worker_getitem_count = 0

    def sampling_weight_policy_identity(self) -> dict[str, Any]:
        empty_caption_values = getattr(self, '_empty_caption_values', ())
        return {'weight_hook': type(self).compute_episode_sampling_weight.__qualname__, 'min_gap': int(getattr(self, 'min_gap', 0) or 0), 'mix_captions': bool(getattr(self, 'mix_captions', False)), 'use_adc': bool(getattr(self, 'use_adc', False)), 'empty_caption_values': sorted((str(value) for value in empty_caption_values))}

    def _sampling_weight_cache_identity(self) -> dict[str, Any]:
        metadata = self.episode_metadata
        if metadata.identity is None:
            raise SamplingWeightCacheError('sampling weight cache requires indexed metadata with a snapshot_id')
        policy = self.sampling_weight_policy_identity()
        source_weights = getattr(self, 'source_weights', None)
        if source_weights is not None:
            policy['source_weights'] = [float(value) for value in source_weights]
        policy['repeats'] = [int(value) for value in metadata.repeats]
        return build_cache_identity(snapshot_id=metadata.identity, episode_count=self.true_length, source_names=metadata.source_names, policy=policy)

    def _compute_episode_sampling_weights(self) -> np.ndarray:
        metadata = getattr(self, 'episode_metadata', None)
        source_weights = getattr(self, 'source_weights', None)
        if metadata is not None and source_weights is not None:
            return np.fromiter((float(self.compute_episode_sampling_weight(index, episode)) * source_weights[metadata.source_id(index)] for (index, episode) in enumerate(self.dataset)), dtype=np.float64, count=self.true_length)
        return np.fromiter((float(self.compute_episode_sampling_weight(index, episode)) for (index, episode) in enumerate(self.dataset)), dtype=np.float64, count=self.true_length)

    def get_sampling_weights(self) -> np.ndarray | None:
        if self._sampling_weights_resolved:
            return self._sampling_weights
        weights = None
        if self._use_sampling_weights and self.dataset:
            cache_dir = getattr(self, '_sampling_weight_cache_dir', None)
            per_episode = None
            if cache_dir is not None:
                identity = self._sampling_weight_cache_identity()
                cache_path = cache_path_for_identity(cache_dir, identity)
                if cache_path.is_file():
                    per_episode = load_sampling_weights(cache_path, identity=identity)
                    zero_rank_print(f'loaded sampling weights from {cache_path}')
                if per_episode is None:
                    per_episode = self._compute_episode_sampling_weights()
                    save_sampling_weights(cache_path, identity=identity, weights=per_episode)
                    zero_rank_print(f'wrote sampling weights to {cache_path}')
            else:
                per_episode = self._compute_episode_sampling_weights()
            weights = per_episode if self.jsonl_repeat_times == 1 else np.tile(per_episode, self.jsonl_repeat_times)
            print('Successfully use the sampling weights')
            zero_rank_print(f"[sampling_weights] enabled={self._use_sampling_weights}, source_weights={getattr(self, 'source_weights', None)}, count={len(weights)}, sum={weights.sum():.3f}")
        self._sampling_weights = weights
        self._sampling_weights_resolved = True
        return weights

    @property
    def uses_sampling_weights(self) -> bool:
        if not self._use_sampling_weights:
            return False
        weights = self.get_sampling_weights()
        return weights is not None and len(weights) > 0

    def get_sample_weight(self) -> np.ndarray:
        if not self._use_sampling_weights:
            return np.ones(len(self), dtype=np.float64)
        weights = self.get_sampling_weights()
        if weights is None or len(weights) == 0:
            return np.ones(len(self), dtype=np.float64)
        if len(weights) != len(self):
            raise ValueError(f'WAM_policy sampling weights must match the dataset length: {len(weights)} != {len(self)}')
        return np.asarray(weights, dtype=np.float64)

    def compute_episode_sampling_weight(self, episode_index: int, episode: dict[str, Any]) -> float:
        raise NotImplementedError(f'{type(self).__name__} must implement compute_episode_sampling_weight() when sampling weights are enabled')

    def normalize_cam_names(self, cam_names, data_source=None, dataset_name=None):
        return list(cam_names)

    def _normalize_train_modes(self, train_modes):
        if train_modes is None:
            return None
        normalized_train_modes = []
        for (mode_idx, mode_cfg) in enumerate(train_modes):
            if not isinstance(mode_cfg, dict):
                raise TypeError(f'train_modes[{mode_idx}] must be a dict')
            mode_positions = mode_cfg.get('positions', mode_cfg.get('keep_positions'))
            if mode_positions is not None:
                if not isinstance(mode_positions, (list, tuple)) or len(mode_positions) == 0:
                    raise ValueError(f'train_modes[{mode_idx}].positions must be a non-empty list')
                positions = [int(position) for position in mode_positions]
                if any((position < 0 for position in positions)):
                    raise ValueError(f'train_modes[{mode_idx}].positions must be non-negative')
                n_view = int(mode_cfg.get('n_view', len(positions)))
                if n_view != len(positions):
                    raise ValueError(f'train_modes[{mode_idx}] n_view={n_view} does not match positions length {len(positions)}')
                normalized_train_modes.append({'positions': positions, 'n_view': n_view})
                continue
            if 'cam_names' not in mode_cfg:
                raise ValueError(f'train_modes[{mode_idx}] is missing cam_names')
            cam_names = mode_cfg['cam_names']
            if not isinstance(cam_names, (list, tuple)) or len(cam_names) == 0:
                raise ValueError(f'train_modes[{mode_idx}].cam_names must be a non-empty list')
            normalized_mode = {'cam_names': self._clone_train_mode_value(cam_names)}
            view_ids = mode_cfg.get('view_ids')
            if view_ids is not None:
                normalized_mode['view_ids'] = list(view_ids)
            n_view = mode_cfg.get('n_view')
            if n_view is None:
                if isinstance(cam_names[0], (list, tuple)):
                    n_view = len(cam_names[0])
                else:
                    n_view = len(cam_names)
            normalized_mode['n_view'] = int(n_view)
            normalized_train_modes.append(normalized_mode)
        return normalized_train_modes

    def _clone_train_mode_value(self, value):
        if isinstance(value, (list, tuple)):
            cloned = []
            for item in value:
                if isinstance(item, (list, tuple)):
                    cloned.append(list(item))
                else:
                    cloned.append(item)
            return cloned
        return value

    def _select_train_mode_value_for_sample(self, value, source_idx, field_name, mode_idx):
        if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
            if source_idx >= len(value):
                raise IndexError(f'train_modes[{mode_idx}].{field_name} does not cover source index {source_idx}')
            return list(value[source_idx])
        return list(value)

    def _get_train_mode_overrides(self, idx, mode_idx):
        if self.train_modes is None:
            raise ValueError('Tagged training indices require train_modes to be configured')
        if mode_idx < 0 or mode_idx >= len(self.train_modes):
            raise IndexError(f'train mode index {mode_idx} is out of range')
        mode_cfg = self.train_modes[mode_idx]
        if 'positions' in mode_cfg:
            return {'positions': list(mode_cfg['positions'])}
        sample_idx = idx % self.true_length
        source_idx = self.sample_source_indices[sample_idx]
        cam_names = self._select_train_mode_value_for_sample(mode_cfg['cam_names'], source_idx, 'cam_names', mode_idx)
        view_ids = mode_cfg.get('view_ids')
        if view_ids is None:
            view_ids = list(range(len(cam_names)))
        else:
            view_ids = self._select_train_mode_value_for_sample(view_ids, source_idx, 'view_ids', mode_idx)
        if len(view_ids) != len(cam_names):
            raise ValueError(f'train_modes[{mode_idx}] view_ids length {len(view_ids)} does not match cam_names length {len(cam_names)}')
        n_view = int(mode_cfg.get('n_view', len(cam_names)))
        if n_view != len(cam_names):
            raise ValueError(f'train_modes[{mode_idx}] n_view={n_view} does not match cam_names length {len(cam_names)}')
        return {'cam_names': cam_names, 'view_ids': view_ids, 'n_view': n_view}

    def _as_list(self, value):
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return [value]

    def _normalize_camera_sampling_rules(self, rules):
        if not rules:
            return []
        if isinstance(rules, dict):
            rules = rules.get('rules', [rules])
        normalized = []
        for (rule_idx, rule) in enumerate(rules):
            if not isinstance(rule, dict):
                raise TypeError(f'camera_sampling_rules[{rule_idx}] must be a dict')
            if not _as_bool(rule.get('enabled', True)):
                continue
            groups = rule.get('groups')
            if groups is None and 'external_cams' in rule:
                external_count = int(rule.get('external_count', 2))
                wrist_cam = rule.get('wrist_cam', 'waist')
                external_view_ids = rule.get('external_view_ids', [3, 5])
                wrist_view_id = rule.get('wrist_view_id', 4)
                groups = [{'candidates': rule['external_cams'], 'count': external_count, 'replace': False, 'preserve_order': rule.get('preserve_order', True), 'available_only': _as_bool(rule.get('available_external_only', False)), 'view_ids': external_view_ids}, {'candidates': [wrist_cam], 'count': 1, 'replace': False, 'preserve_order': True, 'available_only': _as_bool(rule.get('available_wrist_only', False)), 'view_ids': [wrist_view_id]}]
            if not groups:
                raise ValueError(f'camera_sampling_rules[{rule_idx}] must define groups or external_cams')
            norm_groups = []
            total_count = 0
            for (group_idx, group) in enumerate(groups):
                if not isinstance(group, dict):
                    raise TypeError(f'camera_sampling_rules[{rule_idx}].groups[{group_idx}] must be a dict')
                candidates = group.get('candidates', group.get('cams'))
                candidates = [str(c) for c in self._as_list(candidates)]
                if not candidates:
                    raise ValueError(f'camera_sampling_rules[{rule_idx}].groups[{group_idx}] has no candidates')
                count = int(group.get('count', len(candidates)))
                if count <= 0:
                    raise ValueError(f'camera_sampling_rules[{rule_idx}].groups[{group_idx}].count must be > 0')
                replace = _as_bool(group.get('replace', False))
                if count > len(candidates) and (not replace):
                    raise ValueError(f'camera_sampling_rules[{rule_idx}].groups[{group_idx}] requests {count} cameras from {len(candidates)} candidates without replacement')
                group_view_ids = group.get('view_ids', group.get('view_id'))
                if group_view_ids is None and rule.get('view_ids') is None:
                    if count == len(candidates):
                        group_view_ids = infer_camera_view_ids(candidates)
                    else:
                        raise ValueError(f'camera_sampling_rules[{rule_idx}].groups[{group_idx}] must define view_ids when the rule does not define view_ids')
                if group_view_ids is not None:
                    group_view_ids = [int(v) for v in self._as_list(group_view_ids)]
                    if len(group_view_ids) != count:
                        raise ValueError(f'camera_sampling_rules[{rule_idx}].groups[{group_idx}].view_ids length {len(group_view_ids)} does not match count {count}')
                total_count += count
                norm_groups.append({'candidates': candidates, 'count': count, 'replace': replace, 'preserve_order': _as_bool(group.get('preserve_order', True)), 'available_only': _as_bool(group.get('available_only', False)), 'view_ids': group_view_ids})
            view_ids = rule.get('view_ids')
            if view_ids is not None:
                view_ids = [int(v) for v in self._as_list(view_ids)]
                if len(view_ids) != total_count:
                    raise ValueError(f'camera_sampling_rules[{rule_idx}].view_ids length {len(view_ids)} does not match sampled camera count {total_count}')
            normalized.append({'name': str(rule.get('name', f'rule_{rule_idx}')), 'source_indices': {int(x) for x in self._as_list(rule.get('source_indices'))}, 'dataset_sources': {str(x) for x in self._as_list(rule.get('dataset_sources', rule.get('dataset_source')))}, 'dataset_names': {str(x) for x in self._as_list(rule.get('dataset_names', rule.get('dataset_name')))}, 'dataset_name_contains': [str(x) for x in self._as_list(rule.get('dataset_name_contains'))], 'clip_contains': [str(x) for x in self._as_list(rule.get('clip_contains'))], 'groups': norm_groups, 'view_ids': view_ids, 'deterministic': rule.get('deterministic', None)})
        return normalized

    def _camera_rule_matches(self, rule, idx, dataset_name, data_source, clip):
        sample_source_idx = self.sample_source_indices[idx]
        if rule['source_indices'] and sample_source_idx not in rule['source_indices']:
            return False
        if rule['dataset_sources'] and data_source not in rule['dataset_sources']:
            return False
        if rule['dataset_names'] and dataset_name not in rule['dataset_names']:
            return False
        if rule['dataset_name_contains'] and (not any((x in dataset_name for x in rule['dataset_name_contains']))):
            return False
        if rule['clip_contains'] and (not any((x in clip for x in rule['clip_contains']))):
            return False
        return True

    def _select_camera_sampling_rule(self, idx, dataset_name, data_source, clip):
        for rule in self.camera_sampling_rules:
            if self._camera_rule_matches(rule, idx, dataset_name, data_source, clip):
                return rule
        return None

    def _apply_positions_override(self, cam_names, view_ids, positions):
        cam_names = list(cam_names)
        view_ids = list(view_ids)
        if len(view_ids) != len(cam_names):
            raise ValueError(f'view_ids length {len(view_ids)} does not match camera count {len(cam_names)}')
        positions = [int(position) for position in positions]
        selected_cam_names = []
        selected_view_ids = []
        for position in positions:
            if position < 0 or position >= len(cam_names):
                raise ValueError(f'multi-view mode position {position} is out of range for {len(cam_names)} active cameras')
            selected_cam_names.append(cam_names[position])
            selected_view_ids.append(view_ids[position])
        return (selected_cam_names, selected_view_ids, len(selected_cam_names))

    def get_available_cams(self, video_path, cam_names, idx, video_sync=True, n_view=None):
        cam_names = self.normalize_cam_names(cam_names)
        active_n_view = self.n_view if n_view is None else int(n_view)
        if len(cam_names) != active_n_view:
            raise ValueError('Robotwin camera count does not match n_view')
        return cam_names

    def _accv_video_path_exists(self, video_path, cam_name):
        if getattr(self, 'data_source', None) == 'lerobot':
            try:
                return os.path.exists(video_path.format(cam_name))
            except Exception:
                return False
        try:
            base_path = video_path.format(cam_name)
        except Exception:
            return False
        return os.path.exists(base_path) or os.path.exists(base_path + self.video_suffix) or os.path.exists(base_path + '.mp4')

    def _accv_unique_decode_cams(self, cam_names, video_path):
        decode_cam_names = []
        cam_to_decode_idx = {}
        for cam_name in cam_names:
            if cam_name in cam_to_decode_idx:
                continue
            if not self._accv_video_path_exists(video_path, cam_name):
                continue
            cam_to_decode_idx[cam_name] = len(decode_cam_names)
            decode_cam_names.append(cam_name)
        return (decode_cam_names, cam_to_decode_idx)

    def _filter_resolutions_for_cams(self, resolutions, cam_names):
        if not resolutions:
            return resolutions
        return {cam: resolutions[cam] for cam in cam_names if cam in resolutions}

    def _get_accv_decode_plan(self, video_path, active_cam_names, configured_cam_names, resolutions=None):
        active_cam_names = list(active_cam_names)
        if not self.accv_decode_all_configured_views:
            return (active_cam_names, list(range(len(active_cam_names))), resolutions)
        configured_cam_names = self.normalize_cam_names(configured_cam_names, data_source=getattr(self, 'data_source', None), dataset_name=getattr(self, 'dataset_name', None))
        (decode_cam_names, cam_to_decode_idx) = self._accv_unique_decode_cams(configured_cam_names, video_path)
        if not decode_cam_names or any((cam not in cam_to_decode_idx for cam in active_cam_names)):
            (decode_cam_names, cam_to_decode_idx) = self._accv_unique_decode_cams(active_cam_names, video_path)
        if any((cam not in cam_to_decode_idx for cam in active_cam_names)):
            return (active_cam_names, list(range(len(active_cam_names))), resolutions)
        select_indices = [cam_to_decode_idx[cam] for cam in active_cam_names]
        return (decode_cam_names, select_indices, self._filter_resolutions_for_cams(resolutions, decode_cam_names))

    def get_total_timesteps(self, video_path, action_path, cam_names, video_sync=True):
        if action_path.endswith('.pt'):
            action_length = len(torch.load(action_path))
        elif action_path.endswith('.parquet'):
            try:
                action_length = pq.ParquetFile(action_path).metadata.num_rows
            except Exception:
                try:
                    action_length = len(pd.read_parquet(action_path))
                except Exception:
                    raise NotImplementedError('parquet action files are not supported in this environment (install pyarrow or pandas)')
        elif action_path.endswith('.h5'):
            with h5py.File(action_path, 'r') as fid:
                if 'state' in fid:
                    action_length = len(fid['state/joint/position'])
                else:
                    action_length = len(fid)
        else:
            raise NotImplementedError
        if video_sync:
            if self.data_source == 'lerobot':
                _video_path = video_path.format(cam_names[0])
            else:
                base_path = video_path.format(cam_names[0])
                if base_path.endswith('.mp4'):
                    _video_path = base_path
                elif os.path.exists(base_path + self.video_suffix):
                    _video_path = base_path + self.video_suffix
                else:
                    _video_path = base_path + '.mp4'
            (codec, t) = get_video_info(_video_path)
        else:
            frame_idxes = os.listdir(os.path.dirname(os.path.dirname(video_path)))
            frame_idxes = [_ for _ in frame_idxes if _.isdigit()]
            t = len(frame_idxes)
            codec = 'img'
        if self.return_action:
            assert action_length == t
        return (codec, t)

    def seek_img(self, image_path, cam_names, end_frames, slices):
        videos = []
        if end_frames == 0:
            idxs = slices
        else:
            idxs = slices[:-end_frames]
        for cam_name in cam_names:
            video = []
            for idx in idxs:
                image_file = image_path.format(idx, cam_name)
                img = cv2.imread(image_file)[:, :, ::-1].copy()
                video.append(torch.from_numpy(img))
            video = torch.stack(video, dim=0).float().permute(3, 0, 1, 2).contiguous() / 255.0
            videos.append(video)
        return videos

    def _setup_device(self):
        if dist.is_initialized():
            local_rank = dist.get_rank() % torch.cuda.device_count()
        else:
            local_rank = settings.local_rank
        torch.cuda.set_device(local_rank)
        print(f'[INFO] Worker Process PID {os.getpid()} binding to GPU: {local_rank}')
        return local_rank

    def _seek_mp4_accv_impl(self, video_path, cam_names, end_frames, slices, resolutions=None, full_cam_names=None):
        if not hasattr(self, 'decoder_manager'):
            self.decoder_manager = SmartDecoderManager(self.gpuid, max_instances_per_worker=getattr(self, 'accv_max_decoders_per_worker', 8), max_width=self.accv_max_width, max_height=self.accv_max_height)
        requested_cams = list(cam_names)
        decode_cams = list(full_cam_names) if full_cam_names else list(requested_cams)
        for cam in requested_cams:
            if cam not in decode_cams:
                decode_cams.append(cam)
        idxs = slices if end_frames == 0 else slices[:-end_frames]
        is_lerobot = getattr(self, 'data_source', None) == 'lerobot'
        kept_cams = []
        file_path_list = []
        for cam_name in decode_cams:
            base_path = video_path.format(cam_name)
            if is_lerobot:
                video_file = base_path
            elif base_path.endswith('.mp4'):
                video_file = base_path
            elif os.path.exists(base_path + self.video_suffix):
                video_file = base_path + self.video_suffix
            else:
                video_file = base_path + '.mp4'
            if cam_name in requested_cams:
                if not os.path.exists(video_file):
                    raise FileNotFoundError(f'Missing video file: {video_file}')
            elif not os.path.exists(video_file):
                continue
            kept_cams.append(cam_name)
            file_path_list.append(video_file)
        decode_cams = kept_cams
        out_indices = [decode_cams.index(cam) for cam in requested_cams]
        keep_set = set(out_indices)
        res_key = self.decoder_manager.get_key(self.dataset_name, decode_cams, resolutions)
        cache_key = res_key
        if self.accv_fallback_to_cpu and res_key in self.accv_cpu_keys:
            return self.seek_mp4(video_path, cam_names, end_frames, slices)
        try:
            if self.accv_fallback_to_cpu and self.accv_disable_key_after_failures > 0 and (res_key in self._accv_disabled_keys):
                return self.seek_mp4(video_path, cam_names, end_frames, slices)
            stream_infos = None
            if self.accv_max_width is not None:
                (cache_key, stream_infos) = self.decoder_manager.get_reconfigure_request(file_path_list)
            nv_gop_dec = self.decoder_manager.get_decoder(cache_key, len(file_path_list))
            frame_positions = {}
            for (output_frame_idx, idx) in enumerate(idxs):
                frame_positions.setdefault(int(idx), []).append(output_frame_idx)
            output_slots_by_cam = [[] for _ in file_path_list]
            for (output_slot, cam_idx) in enumerate(out_indices):
                output_slots_by_cam[cam_idx].append(output_slot)
            videos = [None] * len(out_indices)
            for (frame_id, output_frame_indices) in frame_positions.items():
                frame_ids = [frame_id] * len(file_path_list)
                decoded_frames = nv_gop_dec.DecodeN12ToRGB(file_path_list, frame_ids, False)
                if stream_infos is not None and len(decoded_frames) != len(file_path_list):
                    raise AccvDecoderContractError('ACCV returned an unexpected number of camera frames')
                for (cam_idx, frame) in enumerate(decoded_frames):
                    if cam_idx not in keep_set:
                        continue
                    frame_tensor = torch.as_tensor(frame)
                    if stream_infos is not None:
                        (width, height, _, _) = stream_infos[cam_idx]
                        if tuple(frame_tensor.shape) != (height, width, 3) or frame_tensor.dtype != torch.uint8:
                            raise AccvDecoderContractError('ACCV output shape or dtype does not match the requested stream')
                    if frame_tensor.is_cuda:
                        frame_tensor = frame_tensor.cpu()
                    frame_chw = frame_tensor.permute(2, 0, 1)
                    for output_slot in output_slots_by_cam[cam_idx]:
                        video = videos[output_slot]
                        if video is None:
                            video = frame_tensor.new_empty((frame_chw.shape[0], len(idxs), frame_chw.shape[1], frame_chw.shape[2]))
                            videos[output_slot] = video
                        for output_frame_idx in output_frame_indices:
                            video[:, output_frame_idx].copy_(frame_chw)
                del decoded_frames
            keep_uint8_for_resize = not self.random_padding
            for (output_slot, video) in enumerate(videos):
                if video is None:
                    raise RuntimeError('ACCV did not return frames for every requested camera')
                if keep_uint8_for_resize:
                    if video.dtype != torch.uint8:
                        raise TypeError(f'ACCV resize-before-float expects uint8 decoder output, got {video.dtype}')
                else:
                    video = video.float() / 255.0
                videos[output_slot] = video
            self.decoder_manager.release_device_memory(cache_key)
            return videos
        except Exception as e:
            self.decoder_manager.purge_decoder(cache_key)
            if isinstance(e, AccvDecoderContractError):
                raise
            if self.accv_disable_key_after_failures > 0:
                self._accv_failure_counts[res_key] += 1
                if self._accv_failure_counts[res_key] >= self.accv_disable_key_after_failures:
                    self._accv_disabled_keys.add(res_key)
            if self.accv_log_bad_videos:
                error_info = f'[{res_key}] Paths:{file_path_list} Error: {str(e)}'
                with open('bad_accv_videos.txt', 'a') as f_err:
                    f_err.write(f'{error_info}\n')
                    f_err.flush()
            if self.accv_fallback_to_cpu:
                return self.seek_mp4(video_path, cam_names, end_frames, slices)
            raise RuntimeError(f'ACCV_DECODE_ERROR|{file_path_list}|{e}')

    def _decode_and_resize_full_clip(self, video_file):
        video_reader = VideoReader(video_file, num_threads=0)
        total = len(video_reader)
        frames_np = video_reader.get_batch(list(range(total))).detach().cpu().numpy()
        del video_reader
        t = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255.0
        t = self.pixel_transforms_resize(t)
        out = (t * 255.0).round().clamp(0, 255).to(torch.uint8)
        return out.permute(0, 2, 3, 1).contiguous().numpy()

    def _seek_mp4_cached(self, video_path, cam_names, end_frames, slices):
        idxs = slices if end_frames == 0 else slices[:-end_frames]
        videos = []
        for cam_name in cam_names:
            if self.data_source == 'lerobot':
                video_file = video_path.format(cam_name)
            else:
                video_file = video_path.format(cam_name) + self.video_suffix
                if not os.path.exists(video_file):
                    video_file = video_path.format(cam_name) + '.mp4'
            clip = self.frame_cache.load(video_file, self.sample_size)
            if clip is None:
                clip = self._decode_and_resize_full_clip(video_file)
                self.frame_cache.store(video_file, self.sample_size, clip)
            frames = np.asarray(clip[list(idxs)])
            video = torch.from_numpy(frames).permute(3, 0, 1, 2).contiguous().float() / 255.0
            videos.append(video)
        return videos

    def seek_mp4_accv(self, video_path, cam_names, end_frames, slices, resolutions=None, **kwargs):
        return self._seek_mp4_accv_impl(video_path, cam_names, end_frames, slices, resolutions=resolutions)

    def seek_mp4(self, video_path, cam_names, end_frames, slices):
        if getattr(self, 'use_frame_cache', False) and getattr(self, 'frame_cache', None) is not None:
            return self._seek_mp4_cached(video_path, cam_names, end_frames, slices)
        videos = []
        idxs = slices if end_frames == 0 else slices[:-end_frames]
        if self.data_source == 'lerobot':
            pass
        for cam_name in cam_names:
            if self.data_source == 'lerobot':
                video_file = video_path.format(cam_name)
            else:
                video_file = video_path.format(cam_name) + self.video_suffix
                if not os.path.exists(video_file):
                    video_file = video_path.format(cam_name) + '.mp4'
            video_reader = VideoReader(video_file, num_threads=0)
            if self.seek_key_frames_only:
                key_idxs = np.asarray(video_reader.get_key_indices())
                idxs = np.asarray(idxs)
                idxs = key_idxs[np.clip(np.searchsorted(key_idxs, idxs, side='right') - 1, 0, len(key_idxs) - 1)]
                video_frames = video_reader.get_batch(idxs)
            else:
                video_frames = video_reader.get_batch(idxs)
            if isinstance(video_frames, decord.ndarray.NDArray):
                video_frames_np = video_frames.asnumpy()
            else:
                video_frames_np = video_frames.detach().cpu().numpy()
            video = torch.from_numpy(video_frames_np).permute(3, 0, 1, 2).contiguous()
            video = video.float() / 255.0
            del video_frames_np
            del video_reader
            videos.append(video)
        return videos

    class VideoDecoderCache:

        def __init__(self):
            self._cache: dict[str, tuple[Any, Any]] = {}
            self._lock = Lock()

        def get_decoder(self, video_path: str):
            if importlib.util.find_spec('torchcodec'):
                from torchcodec.decoders import VideoDecoder
            else:
                raise ImportError('torchcodec is required but not available.')
            video_path = str(video_path)
            with self._lock:
                try:
                    if video_path not in self._cache:
                        decoder = VideoDecoder(video_path, seek_mode='approximate')
                        self._cache[video_path] = (decoder, None)
                except Exception as e:
                    raise RuntimeError(f'torchcodec_DECODE_ERROR|{video_path}')
                return self._cache[video_path][0]

        def clear(self):
            with self._lock:
                self._cache.clear()

        def size(self) -> int:
            with self._lock:
                return len(self._cache)
    _default_decoder_cache = VideoDecoderCache()

    def seek_mp4_torchcodec(self, video_path, cam_names, end_frames, slices, tolerance_s=0.1):
        videos = []
        is_v30 = isinstance(slices, dict)
        for cam_name in cam_names:
            if is_v30:
                cam_idxs = slices[cam_name]
                idxs = cam_idxs if end_frames == 0 else cam_idxs[:-end_frames]
                video_file = self.current_item['video_details'][cam_name]['path']
            else:
                idxs = slices if end_frames == 0 else slices[:-end_frames]
                video_file = video_path.format(cam_name)
                if not os.path.exists(video_file):
                    video_file = video_file + '.mp4'
            decoder_cache = getattr(self, 'decoder_cache', self._default_decoder_cache)
            decoder = decoder_cache.get_decoder(str(video_file))
            metadata = decoder.metadata
            fps = metadata.average_fps
            query_timestamps = [idx / fps for idx in idxs]
            frames_batch = decoder.get_frames_at(indices=idxs)
            loaded_frames = frames_batch.data
            loaded_ts = frames_batch.pts_seconds
            query_ts_tensor = torch.tensor(query_timestamps)
            diff = torch.abs(query_ts_tensor - loaded_ts)
            if not (diff < tolerance_s).all():
                max_diff = diff.max().item()
                raise Exception(f'video diff is too large ({max_diff:.4f}s > {tolerance_s}s)。video_path: {video_file}')
            video = loaded_frames.permute(1, 0, 2, 3).contiguous()
            video = video.float() / 255.0
            videos.append(video)
        return videos

    def _maybe_collect_worker_garbage(self):
        if self.worker_gc_interval <= 0 or get_worker_info() is None:
            return
        self._worker_getitem_count += 1
        if self._worker_getitem_count % self.worker_gc_interval == 0:
            gc.collect()

    def get_long_recaption(self, config, task_caption, idx, siblings=None):
        np_rng = np.random.default_rng(seed=idx) if self.fix_timestep else np.random.default_rng()
        py_rng = random.Random(idx) if self.fix_timestep else random
        newcap = []
        find = set()
        for entry in config:
            step_caption = entry['action_text']
            if step_caption in find:
                continue
            find.add(step_caption)
            step_caption = self._recap_or_drop(self.step_recap_map, step_caption, siblings, np_rng)
            newcap.append(str(step_caption))
        newcap = ', '.join(newcap)
        newcap = newcap.replace(' the ', ' ')
        task_newcap = str(self._recap_or_drop(self.task_recap_map, task_caption, siblings, np_rng))
        fullcap = task_newcap + ': ' + newcap
        cap_type = py_rng.randint(0, 2)
        allcap = [fullcap, task_newcap, newcap]
        recap = allcap[cap_type]
        return recap








    def transform_videos(self, videos, specific_transforms_resize, intrinsics, sample_size, idx=None):
        py_rng = random.Random(idx) if self.fix_timestep else random
        video_transforms = []
        if self.random_padding:
            video_transforms.append(RandomVideoPadding())
        if self.random_crop:
            video_transforms.append(RandomVideoCrop())
        video_transforms.append(ResizeVideo(specific_transforms_resize, sample_size, self.preprocess))
        transform = Compose(video_transforms)
        for (i, video) in enumerate(videos):
            if isinstance(video, list):
                if not video:
                    raise ValueError('Received an empty list of video frames')
                videos[i] = torch.stack([specific_transforms_resize(frame) for frame in video], dim=1)
                continue
            sample = {'video': video, 'resize_source_size': video.shape[-2:]}
            sample = transform(sample, rng=py_rng, skip_resize=getattr(self, 'use_frame_cache', False))
            transformed_video = sample['video']
            if transformed_video.dtype == torch.uint8:
                transformed_video = transformed_video.float() / 255.0
            elif not transformed_video.is_floating_point():
                raise TypeError(f'Video transforms expect uint8 or floating-point input, got {transformed_video.dtype}')
            videos[i] = transformed_video
        return (torch.stack(videos, dim=1), intrinsics)

    def normalize_videos(self, videos, specific_transforms_norm):
        return NormalizeVideo(specific_transforms_norm)(videos)

    def get_dir(self, data_source, video_root, clip, dataset_name):
        if data_source != 'processed':
            raise ValueError('Only processed Robotwin data is supported')
        return (os.path.join(video_root, clip, '{}'), os.path.join(video_root, clip, 'actions.pt'), None)

    def pad_action_and_state(self, action, state, target_dim=16):
        current_dim = action.shape[-1]
        T = action.shape[0]
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        if current_dim >= target_dim:
            current_dim = target_dim
            action = action[:, :target_dim]
            state = state[:, :target_dim]
            mask = torch.ones(current_dim)
            return (action, state, current_dim, mask)
        padded_action = torch.zeros((T, target_dim), dtype=action.dtype)
        padded_state = torch.zeros((state.shape[0], target_dim), dtype=state.dtype)
        padded_action[:, :current_dim] = action
        padded_state[:, :current_dim] = state[:, :current_dim]
        mask = torch.zeros(target_dim, dtype=torch.float32)
        mask[:current_dim] = 1.0
        return (padded_action, padded_state, current_dim, mask)

    def get_action(self, action_path, action_space, dataset_name, end_frames_action, action_slices, norm_action, filter_action, target_dim=None, info_path=None):
        if action_space not in {'abs_joint', 'rela_joint', 'delta_joint'}:
            raise ValueError('Robotwin requires joint actions')
        if self.data_source != 'processed' or not action_path.endswith('.pt'):
            raise ValueError('Robotwin requires processed actions.pt data')
        stat_info = self.stat_info[dataset_name] if norm_action or filter_action else None
        action, state, joint_dim = get_processed_actions(
            action_path, end_frames_action, self.n_previous_action, action_slices,
            stat_info, action_mode=action_space.split('_')[0],
            norm_action=norm_action, filter_action=filter_action)
        padded_action, padded_state, actual_dim, action_mask = self.pad_action_and_state(action, state, target_dim if target_dim is not None else 16)
        return padded_action, padded_state, actual_dim, action_mask, joint_dim

    def _select_step_prompt(self, entry, rng, *, frame_idx=None, use_dense_prompt=None):
        original_prompt = str(entry['action_text'])
        if not self.use_dense_prompt:
            return original_prompt
        dense_prompt = entry.get(self.dense_prompt_field)
        if isinstance(dense_prompt, list):
            candidates = [segment for segment in dense_prompt if isinstance(segment, dict) and isinstance(segment.get('caption'), str) and segment['caption'].strip()]
            if not candidates or frame_idx is None:
                return original_prompt
            containing = [segment for segment in candidates if int(segment['start_frame']) <= int(frame_idx) <= int(segment['end_frame'])]
            if not containing:
                return original_prompt
            dense_prompt = min(containing, key=lambda segment: (abs((int(segment['start_frame']) + int(segment['end_frame'])) / 2 - int(frame_idx)), int(segment['start_frame'])))['caption']
        if not isinstance(dense_prompt, str) or not dense_prompt.strip():
            return original_prompt
        if use_dense_prompt is None:
            use_dense_prompt = rng.random() >= self.dense_prompt_dropout_prob
        if not use_dense_prompt:
            return original_prompt
        return dense_prompt

    def get_step_caption(self, start_idx, config, *, rng=None):
        if rng is None:
            rng = random.Random(start_idx) if self.fix_timestep else random
        for entry in config:
            if entry['start_frame'] <= start_idx and entry['end_frame'] > start_idx:
                return self._select_step_prompt(entry, rng, frame_idx=start_idx)
    SPEED_CUE_TEMPLATES = {'very_slow': 'Very slow pace.', 'slow': 'Slow pace.', 'middle': 'Normal pace.', 'fast': 'Quick pace.', 'very_fast': 'Very quick pace.'}

    @staticmethod
    def _append_cue_clause(caption, cue):
        if not cue or not caption.strip():
            return caption
        base = caption.rstrip()
        sep = ' ' if base.endswith('.') else '. '
        return base + sep + cue

    @staticmethod
    def _prepend_cue_clause(caption, cue):
        if not cue or not caption.strip():
            return caption
        return cue + ' ' + caption.lstrip()

    def _embodiment_cue(self, idx):
        return compose_embodiment_cue(self.body_type_list[idx])

    def _speed_cue(self, moving_mean, idx):
        if self.speed_cue_prob <= 0.0 or moving_mean is None:
            return ''
        rng = random.Random(idx) if self.fix_timestep else random
        if rng.random() >= self.speed_cue_prob:
            return ''
        (very_slow_upper, slow_upper, middle_upper, fast_upper) = self.speed_cue_thresholds
        if moving_mean < very_slow_upper:
            tag = 'very_slow'
        elif moving_mean < slow_upper:
            tag = 'slow'
        elif moving_mean < middle_upper:
            tag = 'middle'
        elif moving_mean < fast_upper:
            tag = 'fast'
        else:
            tag = 'very_fast'
        self._recap_trace.append(f'speed_cue[{tag}] mm={moving_mean:.3f}')
        return self.SPEED_CUE_TEMPLATES[tag]

    @staticmethod
    def _entry_moving_mean(entry):
        eef = entry.get('eef_speed') if isinstance(entry, dict) else None
        if not isinstance(eef, dict) or eef.get('status') != 'ok':
            return None
        moving_mean = eef.get('moving_mean')
        return float(moving_mean) if moving_mean is not None else 0.0

    def _recap_pool(self, recap_map, caption, siblings=None):
        if siblings:
            recap_list = []
            for variant in siblings:
                recap_list.append(variant)
                recap_list.extend(recap_map.get(variant, []))
            if caption not in siblings:
                recap_list.append(caption)
                recap_list.extend(recap_map.get(caption, []))
            return recap_list
        recap_list = list(recap_map.get(caption, []))
        recap_list.append(caption)
        return recap_list

    def _recap_or_drop(self, recap_map, caption, siblings, rng):
        if not isinstance(caption, str):
            self._recap_trace.append(f"skip(junk-caption): {caption!r} -> ''")
            return ''
        if caption.strip().lower() in self._empty_caption_values:
            self._recap_trace.append(f"skip(junk-caption): '{caption}'")
            return caption
        roll_note = ''
        if self.cue_drop_enabled:
            drop_hand = self.p_hand_drop > 0.0 and rng.random() < self.p_hand_drop
            drop_color = self.p_color_drop > 0.0 and rng.random() < self.p_color_drop
            roll_note = f"roll[hand={('Y' if drop_hand else 'N')},color={('Y' if drop_color else 'N')}] "
            if drop_hand or drop_color:
                dropped = self._drop_cues_fn(caption, drop_hand, drop_color, rng)
                if dropped != caption:
                    kind = '+'.join((k for (k, on) in (('hand', drop_hand), ('color', drop_color)) if on))
                    self._recap_trace.append(f"{roll_note}drop[{kind}]: '{caption}' -> '{dropped}'")
                    return str(dropped)
                roll_note += 'drop-miss(no strippable cue) '
        recap_list = self._recap_pool(recap_map, caption, siblings)
        if siblings:
            n_llm = sum((len(recap_map.get(v, [])) for v in siblings))
            if caption not in siblings:
                n_llm += len(recap_map.get(caption, []))
            src = f'llm={n_llm},siblings={len(siblings)}'
        else:
            src = f'llm={len(recap_map.get(caption, []))}'
        out = str(rng.choice(recap_list))
        self._recap_trace.append(f'{roll_note}recap[pool={len(recap_list)},{src}]: kept-original' if out == caption else f"{roll_note}recap[pool={len(recap_list)},{src}]: '{caption}' -> '{out}'")
        return out

    def get_step_recap(self, step_caption, idx, siblings=None):
        rng = np.random.default_rng(seed=idx) if self.fix_timestep else np.random.default_rng()
        return self._recap_or_drop(self.step_recap_map, step_caption, siblings, rng)

    def get_task_recap(self, task_caption, idx, siblings=None):
        rng = np.random.default_rng(seed=idx) if self.fix_timestep else np.random.default_rng()
        return self._recap_or_drop(self.task_recap_map, task_caption, siblings, rng)

    def get_step_start(self, video_slices, action_slices):
        pass

    def sample_episode_frames(self, end: int, episode_index: int, episode: dict[str, Any], *, goal_mode: bool, datasource: str | None=None) -> EpisodeSampleSelection:
        raise NotImplementedError(f'{type(self).__name__} must implement sample_episode_frames()')

    def cal_continuous_index(self, cur_idx, action_chunk, end, *, step_caption):
        raise NotImplementedError

    def get_task_specific_results(self, idx, results):
        return {}

    def get_batch_new(self, idx, cam_names_override=None, view_ids_override=None, n_view_override=None, positions_override=None, goal_mode=None):
        results = {}
        results['sample_idx'] = torch.tensor(int(idx), dtype=torch.long)
        video_dict = self.dataset[idx]
        video_root = self.video_folder[idx]
        base_cam_names = self.cam_name[idx]
        dataset_name = self.dataset_name_list[idx]
        data_source = self.dataset_source_list[idx]
        clip = video_dict['clip']
        cam_name_source = list(cam_names_override) if cam_names_override is not None else base_cam_names
        cam_names = self.normalize_cam_names(cam_name_source, data_source=data_source, dataset_name=dataset_name)
        active_view_ids = list(view_ids_override) if view_ids_override is not None else list(self.view_ids[idx])
        active_n_view = int(n_view_override) if n_view_override is not None else self.n_view
        if positions_override is not None:
            (cam_names, active_view_ids, active_n_view) = self._apply_positions_override(cam_names, active_view_ids, positions_override)
        self.data_source = data_source
        self.dataset_name = dataset_name
        if self.return_action:
            action_space = self.action_space_list[idx]
        resolutions = video_dict.get('resolutions') if self.decode_type == 'accv' else None
        (video_path, action_path, info_path) = self.get_dir(data_source, video_root, clip, dataset_name)
        video_sync = True
        if data_source != 'lerobot':
            try:
                formatted_dir = os.path.dirname(video_path.format(cam_names[0]))
            except Exception:
                formatted_dir = os.path.dirname(video_path)
            if not os.path.exists(formatted_dir):
                video_sync = False
                video_path = os.path.join(os.path.dirname(os.path.dirname(video_path)), 'camera', '{}', video_path.split('/')[-1] + '.jpg')
        results['path'] = video_path
        results['view_ids'] = torch.tensor(active_view_ids, dtype=torch.long)
        results['body_type'] = self.body_type_list[idx]
        results['body_emb_id'] = torch.tensor(self.body_emb_ids[idx], dtype=torch.long)
        results['fps'] = torch.tensor(self.sample_fps[idx], dtype=torch.float32)
        cam_names = self.get_available_cams(video_path, cam_names, idx, video_sync=video_sync, n_view=active_n_view)
        end = self.get_total_timesteps(video_dict)
        end = end - 1
        sample_selection = self.sample_episode_frames(end, idx, video_dict, goal_mode=bool(goal_mode), datasource=dataset_name)
        sample_metadata = sample_selection.sample_metadata
        if not isinstance(sample_metadata, dict):
            raise TypeError('EpisodeSampleSelection.sample_metadata must be a dict')
        end_frames = sample_selection.end_frames
        end_frames_action = sample_selection.end_frames_action
        video_slices = np.clip(sample_selection.video_slices, 0, end)
        action_slices = np.clip(sample_selection.action_slices, 0, end)
        results['end_frames'] = torch.tensor(end_frames)
        results['end_frames_action'] = torch.tensor(end_frames_action)
        results['video_slices'] = torch.tensor(video_slices)
        results['action_slices'] = torch.tensor(action_slices)
        if goal_mode is not None:
            results['goal_cond'] = torch.tensor(bool(goal_mode))
        if self.return_caption:
            if (self.use_step_caption or self.use_recaption) and 'step_caption' in video_dict and ('task_caption' in video_dict):
                step_caption_config = video_dict['step_caption']
                task_caption = video_dict['task_caption']
            else:
                step_caption_config = None
                task_caption = video_dict['task_caption']
            raw_task_caption = task_caption
            self._recap_trace = []
            if self.mix_captions:
                if getattr(self, '_use_step_goal', False):
                    captions = getattr(self, '_step_goal_captions', [])
                    if self.use_recaption and (not self.fix_timestep):
                        captions = [self.get_step_recap(c, idx, siblings=video_dict.get('caption_variants')) for c in captions]
                    if len(captions) == 1:
                        caption = captions[0]
                        if caption.strip().lower() not in self._empty_caption_values:
                            caption = self._append_cue_clause(caption, self._speed_cue(getattr(self, '_step_goal_speed_value', None), idx))
                            caption = self._prepend_cue_clause(caption, self._embodiment_cue(idx))
                    else:
                        parts = []
                        for (i, c) in enumerate(captions):
                            c = c.rstrip()
                            if i > 0 and c and c[0].isupper():
                                c = c[0].lower() + c[1:]
                            if i < len(captions) - 1 and (not c.endswith('.')):
                                c += '.'
                            parts.append(c)
                        caption = ' And then, '.join(parts)
                    self._use_step_goal = False
                else:
                    if self.use_recaption and (not self.fix_timestep):
                        task_caption = self.get_task_recap(task_caption, idx, siblings=video_dict.get('caption_variants'))
                    caption = task_caption
            elif self.use_step_caption and step_caption_config is not None:
                step_start_idx = self.get_step_start(video_slices, action_slices)
                prompt_rng = random.Random(f'dense_prompt_{idx}_{step_start_idx}') if self.fix_timestep else random
                step_caption = self.get_step_caption(step_start_idx, step_caption_config, rng=prompt_rng)
                if self.use_recaption and (not self.fix_timestep):
                    step_caption = self.get_step_recap(step_caption, idx, siblings=video_dict.get('caption_variants'))
                    task_caption = self.get_task_recap(task_caption, idx, siblings=video_dict.get('caption_variants'))
                caption = 'Task: ' + (task_caption or '') + '. Current step: ' + (step_caption or '')
            elif self.use_step_caption and step_caption_config is None:
                caption = task_caption
                if not getattr(self, '_warned_missing_step_caption_config', False):
                    print(f'You choose to use step caption, but there is no step caption config in some samples. Falling back to task caption. First seen in {clip}')
                    self._warned_missing_step_caption_config = True
            elif self.use_recaption and step_caption_config is not None:
                caption = self.get_long_recaption(step_caption_config, task_caption, idx, siblings=video_dict.get('caption_variants'))
            elif self.use_recaption:
                caption = self.get_task_recap(task_caption, idx, siblings=video_dict.get('caption_variants'))
            else:
                caption = task_caption
            results['caption'] = caption
            results['task_caption'] = task_caption
            results['raw_task_caption'] = raw_task_caption
            results['caption_aug'] = ' | '.join(self._recap_trace) if self._recap_trace else 'none'
        if self.return_action:
            (action, state, actual_dim, action_mask, joint_dim) = self.get_action(action_path, action_space, dataset_name, end_frames_action, action_slices, norm_action=self.norm_action, filter_action=self.filter_action, target_dim=self.target_dim, info_path=info_path)
            results['actions'] = action
            results['state'] = state
            results['action_dim'] = actual_dim
            results['action_mask'] = action_mask
            results['joint_dim'] = joint_dim
            if self.norm_action:
                target_concat_dim = action.shape[-1] * 2
                need_dim = action.shape[-1]
                action_mean = torch.tensor(self.stat_info[self.dataset_name_list[idx]]['action']['mean'])[:need_dim]
                action_std = torch.tensor(self.stat_info[self.dataset_name_list[idx]]['action']['std'])[:need_dim]
                bias_std = torch.cat((action_mean, action_std), dim=-1)
                if bias_std.shape[-1] < target_concat_dim:
                    bias_std = torch.nn.functional.pad(bias_std, (0, target_concat_dim - bias_std.shape[-1]))
                results['bias_std'] = bias_std
                state_mean = torch.tensor(self.stat_info[self.dataset_name_list[idx]]['state']['mean'])[:need_dim]
                state_std = torch.tensor(self.stat_info[self.dataset_name_list[idx]]['state']['std'])[:need_dim]
                bias_std_state = torch.cat((state_mean, state_std), dim=-1)
                if bias_std_state.shape[-1] < target_concat_dim:
                    bias_std_state = torch.nn.functional.pad(bias_std_state, (0, target_concat_dim - bias_std_state.shape[-1]))
                results['bias_std_state'] = bias_std_state
        reference_request = None
        prepare_reference = getattr(self, '_prepare_bbox_reference_read', None)
        if prepare_reference is not None and video_sync and self.return_video:
            reference_request = prepare_reference(idx, results, sample_metadata, positions_override=positions_override)
        if self.return_video:
            decode_slices = video_slices
            if reference_request is not None:
                decode_slices = np.concatenate(([int(reference_request)], video_slices))
            if video_sync:
                if self.decode_type == 'cpu':
                    videos = self.seek_mp4(video_path, cam_names, end_frames, decode_slices)
                elif self.decode_type == 'torchcodec':
                    videos = self.seek_mp4_torchcodec(video_path, cam_names, end_frames, decode_slices)
                elif self.decode_type == 'accv':
                    (decode_cam_names, select_indices, decode_resolutions) = self._get_accv_decode_plan(video_path, cam_names, base_cam_names, resolutions=resolutions)
                    videos = self.seek_mp4_accv(video_path, decode_cam_names, end_frames, decode_slices, resolutions=decode_resolutions)
                    videos = [videos[i] for i in select_indices]
            else:
                videos = self.seek_img(video_path, cam_names, end_frames, video_slices)
            (intrinsics, extrinsics) = (None, None)
            (multi_view_videos, intrinsics) = self.transform_videos(videos, self.pixel_transforms_resize, intrinsics, self.sample_size, idx)
            multi_view_videos = self.normalize_videos(multi_view_videos, self.pixel_transforms_norm)
            if reference_request is not None:
                results['bbox_reference_head'] = multi_view_videos[:, 0, 0].clone()
                multi_view_videos = multi_view_videos[:, :, 1:]
            if end_frames != 0:
                multi_view_videos = torch.cat((multi_view_videos, -1 * torch.ones(multi_view_videos.shape[0], multi_view_videos.shape[1], end_frames, multi_view_videos.shape[-2], multi_view_videos.shape[-1])), dim=2)
            results['video'] = multi_view_videos
        conflicting_keys = results.keys() & sample_metadata.keys()
        if conflicting_keys:
            raise ValueError(f'sampling metadata must not overwrite MixBase result fields: {sorted(conflicting_keys)}')
        results.update(sample_metadata)
        extra_task_specific_results = self.get_task_specific_results(idx, results)
        results.update(extra_task_specific_results)
        if 'caption' in results:
            results['caption'] = condition_caption(results['caption'], self.body_type_list[idx])
        return results

    def __len__(self):
        return self.length

    def get_data_by_step(self, idx, cur_step):
        results = {}
        video_dict = self.dataset[idx]
        video_root = self.video_folder[idx]
        base_cam_names = self.cam_name[idx]
        dataset_name = self.dataset_name_list[idx]
        data_source = self.dataset_source_list[idx]
        clip = video_dict['clip']
        cam_names = self.normalize_cam_names(base_cam_names, data_source=data_source, dataset_name=dataset_name)
        active_view_ids = list(self.view_ids[idx])
        active_n_view = self.n_view
        self.data_source = data_source
        self.dataset_name = dataset_name
        if self.return_action:
            action_space = self.action_space_list[idx]
        resolutions = video_dict.get('resolutions') if self.decode_type == 'accv' else None
        (video_path, action_path, info_path) = self.get_dir(data_source, video_root, clip, dataset_name)
        video_sync = True
        if data_source != 'lerobot':
            try:
                formatted_dir = os.path.dirname(video_path.format(cam_names[0]))
            except Exception:
                formatted_dir = os.path.dirname(video_path)
            if not os.path.exists(formatted_dir):
                video_sync = False
                video_path = os.path.join(os.path.dirname(os.path.dirname(video_path)), 'camera', '{}', video_path.split('/')[-1] + '.jpg')
        results['path'] = video_path
        results['view_ids'] = torch.tensor(active_view_ids, dtype=torch.long)
        results['body_type'] = self.body_type_list[idx]
        results['body_emb_id'] = torch.tensor(self.body_emb_ids[idx], dtype=torch.long)
        results['fps'] = torch.tensor(self.sample_fps[idx], dtype=torch.float32)
        cam_names = self.get_available_cams(video_path, cam_names, idx, video_sync=video_sync, n_view=active_n_view)
        total_end = self.get_total_timesteps(video_dict)
        total_end = total_end - 1
        results['total_end'] = total_end
        action_chunk = getattr(self, 'action_chunk', 56)
        (end_frames, end_frames_action, video_slices, action_slices) = self.cal_continuous_index(cur_step, action_chunk, total_end, step_caption=video_dict['step_caption'])
        video_slices = np.clip(video_slices, 0, total_end)
        action_slices = np.clip(action_slices, 0, total_end)
        results['end_frames'] = torch.tensor(end_frames)
        results['end_frames_action'] = torch.tensor(end_frames_action)
        results['video_slices'] = torch.tensor(video_slices)
        results['action_slices'] = torch.tensor(action_slices)
        results['task_caption'] = video_dict.get('task_caption', video_dict.get('caption', ''))
        results['caption'] = results['task_caption']
        _step_cfg = video_dict.get('step_caption')
        if self.use_step_caption and _step_cfg:
            _start_idx = int(video_slices[0])
            if _start_idx is not None:
                _step_text = None
                for _entry in _step_cfg:
                    try:
                        if _entry['start_frame'] <= _start_idx < _entry['end_frame']:
                            _step_text = _entry.get('english_action_text') or _entry.get('action_text')
                            if self.use_dense_prompt:
                                _dense_text = resolve_dense_prompt(_entry, int(cur_step), self.dense_prompt_field)
                                if _dense_text is not None:
                                    _step_text = _dense_text
                            break
                    except (KeyError, TypeError):
                        continue
                if _step_text:
                    results['caption'] = _step_text
                    if _step_text.strip().lower() not in self._empty_caption_values:
                        if self.speed_cue_prob > 0.0:
                            results['caption'] = self._append_cue_clause(results['caption'], self._speed_cue(self._entry_moving_mean(_entry), idx))
                        results['caption'] = self._prepend_cue_clause(results['caption'], self._embodiment_cue(idx))
        if self.return_action:
            (action, state, actual_dim, action_mask, joint_dim) = self.get_action(action_path, action_space, dataset_name, end_frames_action, action_slices, norm_action=self.norm_action, filter_action=self.filter_action, target_dim=self.target_dim, info_path=info_path)
            results['actions'] = action
            results['state'] = state
            results['action_dim'] = actual_dim
            results['action_mask'] = action_mask
            results['joint_dim'] = joint_dim
            (raw_action, _, _, _, _) = self.get_action(action_path, action_space, dataset_name, end_frames_action, action_slices, norm_action=False, filter_action=False, target_dim=self.target_dim, info_path=info_path)
            results['raw_actions'] = raw_action
            if self.norm_action:
                target_concat_dim = action.shape[-1] * 2
                need_dim = action.shape[-1]
                action_mean = torch.tensor(self.stat_info[self.dataset_name_list[idx]]['action']['mean'])[:need_dim]
                action_std = torch.tensor(self.stat_info[self.dataset_name_list[idx]]['action']['std'])[:need_dim]
                bias_std = torch.cat((action_mean, action_std), dim=-1)
                if bias_std.shape[-1] < target_concat_dim:
                    bias_std = torch.nn.functional.pad(bias_std, (0, target_concat_dim - bias_std.shape[-1]))
                results['bias_std'] = bias_std
                state_mean = torch.tensor(self.stat_info[self.dataset_name_list[idx]]['state']['mean'])[:need_dim]
                state_std = torch.tensor(self.stat_info[self.dataset_name_list[idx]]['state']['std'])[:need_dim]
                bias_std_state = torch.cat((state_mean, state_std), dim=-1)
                if bias_std_state.shape[-1] < target_concat_dim:
                    bias_std_state = torch.nn.functional.pad(bias_std_state, (0, target_concat_dim - bias_std_state.shape[-1]))
                results['bias_std_state'] = bias_std_state
        if video_sync:
            if self.decode_type == 'cpu':
                videos = self.seek_mp4(video_path, cam_names, end_frames, video_slices)
            elif self.decode_type == 'torchcodec':
                videos = self.seek_mp4_torchcodec(video_path, cam_names, end_frames, video_slices)
            else:
                (decode_cam_names, select_indices, decode_resolutions) = self._get_accv_decode_plan(video_path, cam_names, base_cam_names, resolutions=resolutions)
                videos = self.seek_mp4_accv(video_path, decode_cam_names, end_frames, video_slices, resolutions=decode_resolutions)
                videos = [videos[i] for i in select_indices]
        else:
            videos = self.seek_img(video_path, cam_names, end_frames, video_slices)
        intrinsics = None
        (multi_view_videos, intrinsics) = self.transform_videos(videos, self.pixel_transforms_resize, intrinsics, self.sample_size, idx)
        multi_view_videos = self.normalize_videos(multi_view_videos, self.pixel_transforms_norm)
        if end_frames != 0:
            pad = -1 * torch.ones(multi_view_videos.shape[0], multi_view_videos.shape[1], end_frames, multi_view_videos.shape[-2], multi_view_videos.shape[-1])
            multi_view_videos = torch.cat((multi_view_videos, pad), dim=2)
        results['video'] = multi_view_videos
        if 'caption' in results:
            results['caption'] = condition_caption(results['caption'], self.body_type_list[idx])
        return results

    def _log_sample_drop(self, sample_idx, err):
        reason = getattr(err, 'reason', None) or err.__class__.__name__
        reason = str(reason).replace('\n', ' ')[:160]
        self._sample_drop_counts[reason] += 1
        count = self._sample_drop_counts[reason]
        every = int(getattr(self, 'drop_log_every_n', 100) or 0)
        if every <= 0 or (count != 1 and count % every != 0):
            return
        msg = f'[DROP] count={count} idx={int(sample_idx) % self.true_length} reason={reason}'
        if getattr(self, 'drop_log_paths', False):
            try:
                msg += f" path={self.dataset[int(sample_idx) % self.true_length]['clip']}"
            except Exception:
                pass
        print(msg, flush=True)

    def _handle_sample_error(self, sample_idx, err):
        if isinstance(err, AccvDecoderContractError):
            raise err
        if not getattr(self, 'drop_log_tracebacks', True):
            self._log_sample_drop(sample_idx, err)
        else:
            traceback.print_exc()
            print('drop:', sample_idx)
            try:
                print('path:', self.dataset[int(sample_idx) % self.true_length]['clip'])
            except Exception:
                pass
        return (int(sample_idx) + 503) % self.length

    def __getitem__(self, idx):
        if isinstance(idx, (tuple, list)):
            if len(idx) == 3 and idx[2] == 'train_mode':
                (sample_idx, mode_idx, _) = idx
                while True:
                    sample_idx = sample_idx % self.true_length
                    mode_overrides = self._get_train_mode_overrides(sample_idx, mode_idx)
                    try:
                        return self.get_batch_new(sample_idx, cam_names_override=mode_overrides.get('cam_names'), view_ids_override=mode_overrides.get('view_ids'), n_view_override=mode_overrides.get('n_view'), positions_override=mode_overrides.get('positions'))
                    except Exception as e:
                        sample_idx = self._handle_sample_error(sample_idx, e)
            if len(idx) == 4 and idx[2] == 'train_mode_goal':
                (sample_idx, mode_idx, _, goal_mode) = idx
                while True:
                    sample_idx = sample_idx % self.true_length
                    mode_overrides = self._get_train_mode_overrides(sample_idx, mode_idx) if mode_idx is not None else None
                    try:
                        return self.get_batch_new(sample_idx, cam_names_override=mode_overrides.get('cam_names') if mode_overrides else None, view_ids_override=mode_overrides.get('view_ids') if mode_overrides else None, n_view_override=mode_overrides.get('n_view') if mode_overrides else None, positions_override=mode_overrides.get('positions') if mode_overrides else None, goal_mode=bool(goal_mode))
                    except Exception as e:
                        sample_idx = self._handle_sample_error(sample_idx, e)
            if len(idx) == 2:
                (v_idx, cur_step) = idx
                for _ in range(10):
                    try:
                        return self.get_data_by_step(v_idx % self.true_length, cur_step)
                    except Exception as e:
                        v_idx = self._handle_sample_error(v_idx, e)
                raise RuntimeError(f'val fetch failed 10 times in a row (last tried v_idx={v_idx % self.true_length}, step={cur_step})')
            raise ValueError(f'Unsupported dataset tuple index format: {idx!r}')
        while True:
            try:
                results = self.get_batch_new(idx % self.true_length)
                break
            except Exception as e:
                idx = self._handle_sample_error(idx, e)
        self._maybe_collect_worker_garbage()
        return results
