import logging
import os
from pathlib import Path
from typing import Callable

import av
import datasets
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset as _LeRobotDataset
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata as _LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    check_timestamps_sync,
    embed_images,
    get_episode_data_index,
    validate_episode_buffer,
    validate_frame,
)
from PIL import Image
from typing_extensions import override

from .base_dataset import BaseDataset
from .dataset import register_dataset
from .. import utils

@register_dataset
class LeRobotDataset(BaseDataset):
    def __init__(
        self,
        data_path,
        data_size=None,
        delta_info=None,
        delta_frames=None,
        delta_timestamps=None,
        t5_embedding_dir=None,
        t5_embedding_pattern="episode_{episode_index:06d}.pt",
        t5_embedding_key="t5_embedding",
        t5_cache_size=256,
        meta_name=None,
        **kwargs,
    ):
        super(LeRobotDataset, self).__init__(data_path=data_path)
        self.data_size = data_size
        self.delta_info = delta_info
        self.delta_frames = delta_frames
        self.delta_timestamps = delta_timestamps
        self.t5_embedding_dir = t5_embedding_dir
        self.t5_embedding_pattern = str(t5_embedding_pattern)
        self.t5_embedding_key = str(t5_embedding_key)
        self.t5_cache_size = int(t5_cache_size) if t5_cache_size is not None else 0
        self._t5_cache = {}
        self._t5_cache_order = []
        self.meta_name = meta_name
        self.kwargs = kwargs
        self.dataset = None
        self.robotype = None

    @classmethod
    def load(cls, data_or_config):
        from .dataset import load_config

        config = load_config(data_or_config)
        keys = list(config.keys())
        for key in keys:
            if key.startswith('_'):
                config.pop(key)
        return cls(**config)

    def open(self):
        if self.dataset is None:
            repo_id = os.path.basename(self.data_path)
            dataset_meta = _LeRobotDatasetMetadata(repo_id, root=self.data_path)
            delta_timestamps = {}
            if self.delta_info is not None:
                for delta_name, delta_size in self.delta_info.items():
                    delta_timestamps[delta_name] = [i / dataset_meta.fps for i in range(int(delta_size))]
            if self.delta_frames is not None:
                for delta_name, frames in dict(self.delta_frames).items():
                    delta_timestamps[delta_name] = [float(i) / dataset_meta.fps for i in frames]
            if self.delta_timestamps is not None:
                for delta_name, stamps in dict(self.delta_timestamps).items():
                    delta_timestamps[delta_name] = [float(t) for t in stamps]
            if not delta_timestamps:
                delta_timestamps = {"action": [0.0]}
            if "video_backend" not in self.kwargs or self.kwargs.get("video_backend", None) is None:
                self.kwargs["video_backend"] = "pyav"

            info = None
            for name in ("info.json", "info.yaml", "info.yml"):
                p = os.path.join(self.data_path, "meta", name)
                if os.path.isfile(p):
                    info = utils.load_file(p)
                    break
            if info is None:
                raise FileNotFoundError(f"missing meta/info.(json|yaml|yml) under {self.data_path}")

            robotype = None
            for k in ("robotype", "robot_type", "robot", "robot_name", "robot_model"):
                if k in info:
                    robotype = info[k]
                    break
            if robotype is None:
                raise KeyError(f"missing robotype in meta/info.* under {self.data_path}, keys={list(info.keys())}")
            if isinstance(robotype, str):
                robotype = robotype.strip()
            self.robotype = robotype

            self.dataset = FastLeRobotDataset(
                repo_id,
                root=self.data_path,
                delta_timestamps=delta_timestamps,
                **self.kwargs,
            )
            if self.data_size is not None:
                assert self.data_size == len(self.dataset)
            else:
                self.data_size = len(self.dataset)
            if self.t5_embedding_dir is None:
                d = os.path.join(self.data_path, "t5_embedding")
                if os.path.isdir(d):
                    self.t5_embedding_dir = d

    def close(self):
        if self.dataset is not None:
            self.dataset = None
        self._t5_cache.clear()
        self._t5_cache_order.clear()
        super(LeRobotDataset, self).close()

    def __len__(self):
        if self.data_size is None:
            self.open()
        return self.data_size

    def _get_data(self, index):
        data_dict = self.dataset[index]
        data_dict["robotype"] = self.robotype
        if self.meta_name is not None:
            assert self.meta_name not in data_dict
            data_dict[self.meta_name] = self.dataset.meta
        if self.t5_embedding_dir is not None:
            episode_index = data_dict.get("episode_index", None)
            if hasattr(episode_index, "item"):
                try:
                    episode_index = episode_index.item()
                except Exception as e:
                    print(f"error in episode_index.item(): {e}")
            try:
                cache_key = int(episode_index)
            except Exception as e:
                print(f"error in int(episode_index): {e}")
                cache_key = None
            if cache_key is not None:
                if cache_key in self._t5_cache:
                    data_dict[self.t5_embedding_key] = self._t5_cache[cache_key]
                else:
                    p = os.path.join(self.t5_embedding_dir, self.t5_embedding_pattern.format(episode_index=cache_key))
                    if os.path.isfile(p):
                        obj = torch.load(p, map_location="cpu")
                        if isinstance(obj, dict):
                            if self.t5_embedding_key in obj:
                                emb = obj[self.t5_embedding_key]
                            elif "prompt_embeds" in obj:
                                emb = obj["prompt_embeds"]
                            elif "condition_dict" in obj and isinstance(obj["condition_dict"], dict) and "prompt_embeds" in obj["condition_dict"]:
                                emb = obj["condition_dict"]["prompt_embeds"]
                            else:
                                emb = next(iter(obj.values()))
                        else:
                            emb = obj
                        data_dict[self.t5_embedding_key] = emb
                        if self.t5_cache_size > 0:
                            if cache_key not in self._t5_cache:
                                self._t5_cache[cache_key] = emb
                                self._t5_cache_order.append(cache_key)
                                if len(self._t5_cache_order) > self.t5_cache_size:
                                    old = self._t5_cache_order.pop(0)
                                    self._t5_cache.pop(old, None)
        return data_dict

class FastLeRobotDataset(_LeRobotDataset):
    """This class overrides the `LeRobotDataset`(lerobot version 0.3.2) class
    to accelerate the data conversion process.

    What it does is:
    - Doesn't store temporary image files to disk, instead, it's kept in memory until the whole episode is saved.
    - Only consider observation.state and action features to compute episode statistics.

    Beside, it's recommended to use video mode rather than image mode when converting large datasets. It's easy for data transfer and storage.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = 'pyav',
        skip_video_decoding: bool = False,
        delta_frames = 0,
        data_size=None,
        delta_info=None,
    ):
        super().__init__(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            revision=revision,
            force_cache_sync=force_cache_sync,
            download_videos=download_videos,
            video_backend=video_backend,
        )
        self.skip_video_decoding = skip_video_decoding

    @override
    def add_frame(self, frame: dict, task: str, timestamp: float | None = None) -> None:
        """This function only adds the frame to the episode_buffer and nothing
        is written to disk.

        To save those frames, the 'save_episode()' method then needs to be called.
        """
        # Convert torch tensors to numpy arrays for serialization/storage
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()

        validate_frame(frame, self.features)

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        # Automatically add frame_index and timestamp to episode buffer
        frame_index = self.episode_buffer['size']
        if timestamp is None:
            timestamp = frame_index / self.fps
        self.episode_buffer['frame_index'].append(frame_index)
        self.episode_buffer['timestamp'].append(timestamp)
        self.episode_buffer['task'].append(task)

        # Add frame features to episode_buffer
        for key in frame:
            if key not in self.features:
                raise ValueError(f"An element of the frame is not in the features. '{key}' not in '{self.features.keys()}'.")

            self.episode_buffer[key].append(frame[key])

        self.episode_buffer['size'] += 1

    @override
    def save_episode(self, episode_data: dict | None = None) -> None:
        """This will save to disk the current episode in self.episode_buffer.

        Args:
            episode_data (dict | None, optional): Dict containing the episode data to save. If None, this will
                save the current episode in self.episode_buffer, which is filled with 'add_frame'. Defaults to
                None.
        """
        if not episode_data:
            episode_buffer = self.episode_buffer

        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        # 'size' and 'task' are bookkeeping fields, omitted from parquet payload
        episode_length = episode_buffer.pop('size')
        tasks = episode_buffer.pop('task')
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer['episode_index']

        episode_buffer['index'] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer['episode_index'] = np.full((episode_length,), episode_index)

        # Register any new tasks encountered during this episode
        for task in episode_tasks:
            task_index = self.meta.get_task_index(task)
            if task_index is None:
                self.meta.add_task(task)

        # Map natural-language task names to task indices
        episode_buffer['task_index'] = np.array([self.meta.get_task_index(task) for task in tasks])

        for key, ft in self.features.items():
            # index, episode_index, task_index are already processed above, and image and video
            # are processed separately by storing image path and frame info as meta data
            if key in ['index', 'episode_index', 'task_index'] or ft['dtype'] in ['image', 'video']:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        self._save_episode_table(episode_buffer, episode_index)
        ep_stats = _compute_episode_stats(episode_buffer)

        if len(self.meta.video_keys) > 0:
            video_paths = self.encode_episode_videos(episode_buffer, episode_index)
            for key in self.meta.video_keys:
                episode_buffer[key] = video_paths[key]

        # Persist episode meta after encoding videos to include video metadata
        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats)

        ep_data_index = get_episode_data_index(self.meta.episodes, [episode_index])
        ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}
        check_timestamps_sync(
            episode_buffer['timestamp'],
            episode_buffer['episode_index'],
            ep_data_index_np,
            self.fps,
            self.tolerance_s,
        )

        video_files = list(self.root.rglob('*.mp4'))
        assert len(video_files) == self.num_episodes * len(self.meta.video_keys)

        parquet_files = list(self.root.rglob('*.parquet'))
        assert len(parquet_files) == self.num_episodes

        if not episode_data:  # Reset the buffer
            self.episode_buffer = self.create_episode_buffer()

    @override
    def _save_episode_table(self, episode_buffer: dict, episode_index: int) -> None:
        episode_dict = {key: episode_buffer[key] for key in self.hf_features}
        ep_dataset = datasets.Dataset.from_dict(episode_dict, features=self.hf_features, split='train')
        ep_dataset = embed_images(ep_dataset)
        ep_data_path = self.root / self.meta.get_data_file_path(ep_index=episode_index)
        ep_data_path.parent.mkdir(parents=True, exist_ok=True)
        ep_dataset.to_parquet(ep_data_path)

    @override
    def encode_episode_videos(self, episode_buffer: dict, episode_index: int) -> dict:
        """Use ffmpeg to convert frames stored as png into mp4 videos.

        Note: `encode_video_frames` is a blocking call. Making it asynchronous shouldn't speedup encoding,
        since video encoding with ffmpeg is already using multithreading.
        """
        video_paths = {}
        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            video_paths[key] = str(video_path)
            if video_path.is_file():
                continue

            imgs = episode_buffer[key]

            _encode_video_frames(imgs, video_path, self.fps, overwrite=True)

        return video_paths

    @override
    def __getitem__(self, idx: int) -> dict:
        item = self.hf_dataset[idx]
        ep_idx = item['episode_index'].item()

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        # Optional: skip costly video decoding when only computing stats
        if not self.skip_video_decoding and len(self.meta.video_keys) > 0:
            current_ts = item['timestamp'].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            try:
                video_frames = self._query_videos(query_timestamps, ep_idx)
            except Exception as e:
                logging.warning(
                    f'Failed to decode video frames for episode {ep_idx} in timestamps: {query_timestamps}. Error: {e}. Falling back to zeros.'
                )
                video_frames = {}
                # Construct zero tensors matching expected shapes per key
                for vid_key, query_ts in query_timestamps.items():
                    num_queries = len(query_ts)
                    # Prefer shapes from metadata when available
                    ft_shape = self.meta.shapes.get(vid_key)

                    # Derive channel-first (C,H,W)
                    if isinstance(ft_shape, tuple) and len(ft_shape) == 3:
                        if ft_shape[0] in (1, 3, 4):  # likely CHW
                            c, h, w = ft_shape[0], ft_shape[1], ft_shape[2]
                        elif ft_shape[2] in (1, 3, 4):  # likely HWC
                            c, h, w = ft_shape[2], ft_shape[0], ft_shape[1]
                        else:
                            c, h, w = 3, ft_shape[0], ft_shape[1]
                    else:
                        # Conservative default
                        c, h, w = 3, 224, 224

                    if num_queries > 1:
                        zeros_shape = (num_queries, c, h, w)
                    else:
                        zeros_shape = (c, h, w)

                    video_frames[vid_key] = torch.zeros(zeros_shape, dtype=torch.float32)

            item = {**video_frames, **item}

        if self.image_transforms is not None:
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])

        # Add task as a string
        task_idx = item['task_index'].item()
        item['task'] = self.meta.tasks[task_idx]

        return item


def _encode_video_frames(
    imgs: list[np.ndarray],
    video_path: Path | str,
    fps: int,
    vcodec: str = 'libsvtav1',
    pix_fmt: str = 'yuv420p',
    g: int | None = 2,
    crf: int | None = 30,
    fast_decode: int = 0,
    log_level: int | None = av.logging.ERROR,
    overwrite: bool = False,
) -> None:
    """Encode a sequence of RGB frames into a video file using PyAV/ffmpeg.

    Args:
        imgs (list[np.ndarray]): List of frames as HxWx3 RGB numpy arrays. All frames
            are assumed to share the same spatial size.
        video_path (Path | str): Output path for the encoded video file.
        fps (int): Target frames per second for the output video.
        vcodec (str): Video codec passed to ffmpeg. Supported: {'h264', 'hevc', 'libsvtav1'}.
        pix_fmt (str): Pixel format for the encoder (e.g., 'yuv420p', 'yuv444p').
        g (int | None): GOP size (distance between keyframes). ``None`` keeps encoder default.
        crf (int | None): Constant Rate Factor controlling quality/bitrate (lower is higher quality).
            ``None`` keeps encoder default.
        fast_decode (int): Enable fast-decode tuning when non-zero (codec-dependent behavior).
        log_level (int | None): LibAV logging level (e.g., ``av.logging.ERROR``). ``None`` keeps default.
        overwrite (bool): If True, create parent directories as needed and allow overwriting.

    Raises:
        ValueError: If ``vcodec`` is unsupported.
        FileNotFoundError: If no frames are provided.
        OSError: If encoding appears to succeed but the output file is missing.

    Notes:
        More details about ffmpeg argument tuning can be found in `benchmark/video/README.md`.
    """
    # Check encoder availability
    if vcodec not in ['h264', 'hevc', 'libsvtav1']:
        raise ValueError(f'Unsupported video codec: {vcodec}. Supported codecs are: h264, hevc, libsvtav1.')

    video_path = Path(video_path)

    video_path.parent.mkdir(parents=True, exist_ok=overwrite)

    # Encoders/pixel formats incompatibility check
    if (vcodec == 'libsvtav1' or vcodec == 'hevc') and pix_fmt == 'yuv444p':
        logging.warning(f"Incompatible pixel format 'yuv444p' for codec {vcodec}, auto-selecting format 'yuv420p'")
        pix_fmt = 'yuv420p'

    # Define video output frame size (assuming all input frames are the same size)
    if len(imgs) == 0:
        raise FileNotFoundError('No images found.')
    dummy_image = Image.fromarray(imgs[0])
    width, height = dummy_image.size

    # Define video codec options
    video_options = {}

    if g is not None:
        video_options['g'] = str(g)

    if crf is not None:
        video_options['crf'] = str(crf)

    if fast_decode:
        key = 'svtav1-params' if vcodec == 'libsvtav1' else 'tune'
        value = f'fast-decode={fast_decode}' if vcodec == 'libsvtav1' else 'fastdecode'
        video_options[key] = value

    # Set logging level
    if log_level is not None:
        # "While less efficient, it is generally preferable to modify logging with Python’s logging"
        logging.getLogger('libav').setLevel(log_level)

    # Create and open output file (overwrite by default)
    with av.open(str(video_path), 'w') as output:
        output_stream = output.add_stream(vcodec, fps, options=video_options)
        output_stream.pix_fmt = pix_fmt
        output_stream.width = width
        output_stream.height = height

        # Loop through input frames and encode them
        for input_data in imgs:
            input_image = Image.fromarray(input_data).convert('RGB')
            input_frame = av.VideoFrame.from_image(input_image)
            packet = output_stream.encode(input_frame)
            if packet:
                output.mux(packet)

        # Flush the encoder
        packet = output_stream.encode()
        if packet:
            output.mux(packet)

    # Reset logging level
    if log_level is not None:
        av.logging.restore_default_callback()

    if not video_path.exists():
        raise OSError(f'Video encoding did not work. File not found: {video_path}.')


def _get_feature_stats(array: np.ndarray, axis: tuple, keepdims: bool) -> dict[str, np.ndarray]:
    return {
        'min': np.min(array, axis=axis, keepdims=keepdims),
        'max': np.max(array, axis=axis, keepdims=keepdims),
        'mean': np.mean(array, axis=axis, keepdims=keepdims),
        'std': np.std(array, axis=axis, keepdims=keepdims),
        'count': np.array([len(array)]),
    }


def _compute_episode_stats(episode_data: dict[str, list[str] | np.ndarray]) -> dict:
    ep_stats = {}
    for key, data in episode_data.items():
        if key not in ['observation.state', 'action']:
            continue

        ep_ft_array = data  # data is already a np.ndarray
        axes_to_reduce = 0  # compute stats over the first axis
        keepdims = data.ndim == 1  # keep as np.array

        ep_stats[key] = _get_feature_stats(ep_ft_array, axis=axes_to_reduce, keepdims=keepdims)

    return ep_stats
