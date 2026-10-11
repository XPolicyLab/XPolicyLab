"""Shared LeRobot v2/v3 providers, statistics and sample post-processing.

MapStyleRobotVideoDataset uses this passive group with sharding disabled.
The provider class names are retained so configuration targets and caches
remain compatible with the source training code.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
from omegaconf import DictConfig, OmegaConf

from leap.core.config import instantiate

from ._utils import get_logger
from .dataset_utils import CenterCrop, Normalize, ResizeSmallestSideAspectPreserving
from .hierarchical_index import full_temporal_future_offset
from .sharded_lerobot_v2_dataset import ShardedLerobotV2Dataset
from .sharded_lerobot_v3_dataset import ShardedLerobotV3Dataset
from .utils.normalizer import (
    load_dataset_stats_from_json,
    load_lerobot_v2_minmax,
    save_dataset_stats_to_json,
)

logger = get_logger(__name__)


# Copied verbatim from robot_video_dataset.py (keep in sync).
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"

STATS_CACHE_DIR = os.path.join(
    os.environ.get("LEAP_WORK_ROOT", "./work_dirs"), "cache", "stats"
)


def _resolve_repo_relative_path(path: str | os.PathLike[str]) -> str:
    """Resolve a configured relative asset path against the repository root.

    Hydra launchers need not retain the repository as their current directory,
    so ``leap/configs/...`` must not depend on ``os.getcwd()``.
    """
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    repository = next(
        (
            parent
            for parent in Path(__file__).resolve().parents
            if (parent / ".git").exists()
            or ((parent / "pyproject.toml").is_file() and (parent / "leap").is_dir())
        ),
        None,
    )
    if repository is None:
        raise RuntimeError("Cannot resolve repo-relative stats path: repository root not found")
    return str((repository / candidate).resolve())


def _rank_world():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))


def _worker_split():
    info = torch.utils.data.get_worker_info()
    if info is None:
        return 0, 1
    return info.id, info.num_workers


class ShardedRobotVideoDataset(torch.utils.data.IterableDataset):
    """Passive single-group sharded IterableDataset with processor + postprocessing.

    Scheduling and iteration live in :class:`ShardedLeRobotMixtureDataset`, which
    composes one or more instances of this class; this class only builds children
    + stats and exposes ``_postprocess``. It is not iterated directly.
    """

    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        metadata_enabled: bool = False,
        pretrained_norm_stats=None,
        stats_mode: str = "require",
        norm_stats_source: str = "cache",
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        action_video_freq_ratio: int = 1,
        video_read_stride: int = 1,
        # Shift every action query by this many sampled timesteps while leaving
        # state/video queries anchored at t. For an H-step chunk: 0 reads
        # [t, ..., t+H-1], while 1 reads [t+1, ..., t+H]. Delta EEF uses 1 so
        # its first target is T[t]^-1 T[t+1], not the identity T[t]^-1 T[t].
        action_start_offset: int = 0,
        minimum_future_offset: Optional[int] = None,
        concat_multi_camera: str = "horizontal",  # "horizontal", "vertical", "pyramid_3cam", or None
        override_instruction: Optional[str] = None,
        # sharded-mixture specific
        seed: int = 42,
        decode_workers: int = 1,
        skip_corrupt_data: bool = False,
        stats_shard: Optional[List[int]] = None,
        dataset_name: Optional[str] = None,
        stats_cache_name: Optional[str] = None,
        stats_cache_dir: Optional[str] = None,
        exclude_dirs: Optional[List[str]] = None,
    ):
        super().__init__()
        stats_mode = str(stats_mode)
        if stats_mode not in {"require", "build"}:
            raise ValueError("stats_mode must be 'require' or 'build'")
        self.stats_mode = stats_mode
        if norm_stats_source not in {"cache", "lerobot_v2_minmax"}:
            raise ValueError("norm_stats_source must be 'cache' or 'lerobot_v2_minmax'")
        if OmegaConf.is_config(dataset_dirs):
            dataset_dirs = OmegaConf.to_container(dataset_dirs, resolve=True)
        dataset_dirs = list(dataset_dirs)
        # Expand any glob patterns into concrete directories. Datasets stored as
        # many deeply-nested per-range LeRobot roots (e.g. AgiBotWorld2026) can
        # then be referenced by a single pattern instead of hundreds of paths.
        # Entries without wildcards pass through unchanged, so existing configs
        # are unaffected.
        expanded = []
        for d in dataset_dirs:
            if any(ch in str(d) for ch in "*?["):
                matches = sorted(m for m in glob.glob(str(d)) if os.path.isdir(m))
                if not matches:
                    raise FileNotFoundError(f"No directories match dataset_dir glob: {d}")
                expanded.extend(matches)
            else:
                expanded.append(d)
        dataset_dirs = expanded
        # Drop specific directories by basename (opt-in; default None = no-op).
        # Used to exclude dirs whose on-disk schema differs from this source's
        # shape_meta (e.g. galaxea has a few 7-DoF dirs among 6-DoF ones) — those
        # would otherwise fail per-sample dim checks and, when a whole shard is
        # dropped, unbalance per-rank sample counts and hang NCCL.
        if exclude_dirs:
            if OmegaConf.is_config(exclude_dirs):
                exclude_dirs = OmegaConf.to_container(exclude_dirs, resolve=True)
            exclude_set = set(exclude_dirs)
            kept = [d for d in dataset_dirs if os.path.basename(os.path.normpath(d)) not in exclude_set]
            dropped = len(dataset_dirs) - len(kept)
            if dropped:
                logger.info("exclude_dirs: dropped %d/%d dirs from %s",
                            dropped, len(dataset_dirs), dataset_dirs[0] if dataset_dirs else "?")
            dataset_dirs = kept
        shape_meta = (
            OmegaConf.to_container(shape_meta, resolve=True)
            if OmegaConf.is_config(shape_meta)
            else shape_meta
        )
        self.num_frames = int(num_frames)
        self.action_video_freq_ratio = int(action_video_freq_ratio)
        self.video_read_stride = int(video_read_stride)
        if self.video_read_stride < 1:
            raise ValueError("video_read_stride must be positive")
        if (self.num_frames - 1) % self.video_read_stride:
            raise ValueError(
                "num_frames-1 must be divisible by video_read_stride, got "
                f"{self.num_frames - 1} and {self.video_read_stride}"
            )
        if (
            self.video_read_stride > 1
            and self.action_video_freq_ratio != self.video_read_stride
        ):
            raise ValueError(
                "direct source-video sampling requires action_video_freq_ratio "
                "to equal video_read_stride"
            )
        self.action_start_offset = int(action_start_offset)
        if self.action_start_offset < 0:
            raise ValueError("action_start_offset must be non-negative")
        sample_stride = int(global_sample_stride)
        if sample_stride < 1:
            raise ValueError("global_sample_stride must be positive")

        # Raw annotation windows are always trimmed by this actual maximum
        # query offset. It is an internal consequence of the configured
        # temporal geometry, not a user-selectable sampling policy.
        self.full_temporal_future_offset = full_temporal_future_offset(
            num_frames=self.num_frames,
            sample_stride=sample_stride,
            action_start_offset=self.action_start_offset,
        )
        default_future_offset = self.action_video_freq_ratio * sample_stride
        self.minimum_future_offset = (
            default_future_offset
            if minimum_future_offset is None
            else int(minimum_future_offset)
        )
        if self.minimum_future_offset < 1:
            raise ValueError("minimum_future_offset must be positive")

        assert (self.num_frames - 1) % self.action_video_freq_ratio == 0, \
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {self.num_frames - 1} and {self.action_video_freq_ratio}"
        assert ((self.num_frames - 1) // self.action_video_freq_ratio) % 4 == 0, \
            f"video frames must be divisible by 4 for tokenization, got {(self.num_frames - 1) // self.action_video_freq_ratio}"
        provider_num_frames = 1 + (self.num_frames - 1) // self.video_read_stride
        provider_sample_stride = int(global_sample_stride) * self.video_read_stride
        if self.video_read_stride > 1:
            self.video_sample_indices = list(range(provider_num_frames))
        else:
            self.video_sample_indices = list(
                range(0, self.num_frames, self.action_video_freq_ratio)
            )

        self.camera_key = camera_key
        self.video_size = video_size
        self.text_metadata = None
        self.metadata_enabled = bool(metadata_enabled)
        if self.metadata_enabled and processor is None:
            raise ValueError("metadata_enabled requires a processor")
        # Keep the historical ME-U0 behavior by default. Model-specific
        # wrappers that tokenize the instruction themselves can disable this
        # output and avoid one unused torch.load per training sample.
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction
        # Propagated to each child provider (decode-time skip) and into
        # _build_stats (stats-time skip); default False = fail-fast.
        self.skip_corrupt_data = bool(skip_corrupt_data)
        # Map-style readers reuse the audited provider decode and normalization
        # path without constructing shard schedules or shard-prefetch state.
        # Offline distributed stats precompute only: [rank, world] tells each
        # child to keep only episodes[rank::world] so many cheap workers can
        # split the (NFS-IO-bound) stats pass. None in all training paths.
        self._stats_shard = list(stats_shard) if stats_shard is not None else None
        # Stats cache location + name. dataset_name defaults to the first
        # dataset dir's basename (old behavior); configs set it explicitly so
        # glob/multi-task sources are not named after one task dir.
        self._stats_cache_dir = stats_cache_dir if stats_cache_dir is not None else STATS_CACHE_DIR
        self._dataset_name = (
            dataset_name
            if dataset_name is not None
            else os.path.basename(os.path.normpath(dataset_dirs[0]))
        )
        # Keep source identity (dataset_name) independent from the statistics
        # representation.  The same raw source may need different stats after
        # an action/state transform (for example absolute joints vs OpenPI-style
        # chunk-relative joints); reusing the old filename would silently load
        # incompatible normalization ranges.
        self._stats_cache_name = (
            stats_cache_name if stats_cache_name is not None else self._dataset_name
        )
        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )

        # Build one shard provider per dataset directory.
        self.children = self._build_children(
            dataset_dirs,
            shape_meta,
            provider_num_frames,
            provider_sample_stride,
            val_set_proportion,
            is_training_set,
            seed,
            decode_workers,
            self.skip_corrupt_data,
            self._stats_shard,
        )
        if self.video_read_stride > 1:
            action_horizon = self.num_frames - 1
            action_offsets = np.arange(action_horizon, dtype=np.int64) * int(
                global_sample_stride
            )
            for child in self.children:
                child.action_size = action_horizon
                child.action_offsets = action_offsets.copy()
        if self.action_start_offset:
            # Providers historically expose action slots [t, ..., t+H-1].
            # Observation-derived achieved-motion targets need [t+1, ..., t+H]
            # so slot 0 is not an artificial identity pose.  Shift only the
            # action query offsets; state/video observations remain [t, ...,].
            shift = self.action_start_offset * int(global_sample_stride)
            for child in self.children:
                child.action_offsets = child.action_offsets + shift
        # skip_corrupt_data is passed INTO each child constructor above so that
        # construction-time per-episode reads (e.g. h5 meta/length reads) can
        # skip corrupt episodes instead of crashing __init__. This loop only
        # keeps the mixture path's single-source-of-truth override in sync (the
        # mixture re-sets child.skip_corrupt after this group is built).
        for child in self.children:
            child.skip_corrupt = self.skip_corrupt_data
        # Stats are an offline, versioned prerequisite.  Production training
        # uses stats_mode=require and never scans a dataset after model startup.
        # stats_mode=build is reserved for the explicit precompute CLI.
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if self.metadata_enabled:
                from leap.data.world_unified.metadata import Metadata

                self.text_metadata = (
                    Metadata.from_config(
                        shape_meta=shape_meta,
                        action_state_transforms=processor.action_state_transforms,
                    )
                )
            if not getattr(processor, "requires_normalizer_stats", True):
                if is_training_set:
                    processor.train()
                else:
                    processor.eval()
                for child in self.children:
                    child.set_processor(processor)
                return
            if norm_stats_source == "lerobot_v2_minmax":
                if (
                    pretrained_norm_stats is not None
                    or processor.action_state_transforms
                    or processor.use_stepwise_action_norm
                    or not all(isinstance(child, ShardedLerobotV2Dataset) for child in self.children)
                ):
                    raise ValueError("lerobot_v2_minmax requires native v2 fields, global normalization and no external stats")
                dataset_stats = load_lerobot_v2_minmax(dataset_dirs, shape_meta)
                processor.set_normalizer_from_stats(dataset_stats)
                processor.train() if is_training_set else processor.eval()
                for child in self.children:
                    child.set_processor(processor)
                logger.info("Using native LeRobot v2.1 episode min/max: %s", dataset_dirs)
                return
            stats_path = (
                _resolve_repo_relative_path(pretrained_norm_stats)
                if pretrained_norm_stats is not None
                else os.path.abspath(
                    os.path.join(
                        self._stats_cache_dir, f"{self._stats_cache_name}.stats.json"
                    )
                )
            )
            self._resolved_stats_path = stats_path
            if os.path.isfile(stats_path):
                dataset_stats = load_dataset_stats_from_json(stats_path)
                logger.info("Using offline native dataset stats: %s", stats_path)
            elif self.stats_mode == "require":
                raise FileNotFoundError(
                    f"Missing required normalization statistics for {self._dataset_name}: "
                    f"{stats_path}. Prepare the statistics before launching training."
                )
            else:
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    raise RuntimeError(
                        "stats_mode=build must run as a standalone offline process, "
                        "not inside distributed training"
                    )
                dataset_stats = self._build_stats(
                    processor,
                    dataset_dirs,
                    shape_meta,
                    num_frames,
                    global_sample_stride,
                    val_set_proportion,
                    is_training_set,
                    seed,
                    skip_corrupt_data=False,
                )
                parent = os.path.dirname(stats_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                tmp_path = f"{stats_path}.tmp.{os.getpid()}"
                save_dataset_stats_to_json(dataset_stats, tmp_path)
                with open(tmp_path, "rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(tmp_path, stats_path)
                directory_fd = os.open(parent or ".", os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
                logger.info("Published offline dataset stats: %s", stats_path)

            processor.set_normalizer_from_stats(dataset_stats)
            if is_training_set:
                processor.train()
            else:
                processor.eval()
            for child in self.children:
                child.set_processor(processor)

    def __iter__(self):
        # Passive container: scheduling/iteration live in
        # ShardedLeRobotMixtureDataset, which drives this group's children +
        # _postprocess directly. This concrete override also satisfies
        # torch.utils.data.IterableDataset.__iter__ (an @abstractmethod in some
        # torch builds, e.g. 2.7.0a) so the class stays instantiable.
        raise NotImplementedError(
            "ShardedRobotVideoDataset is a passive group container; iterate the "
            "owning ShardedLeRobotMixtureDataset instead."
        )

    # ------------------------------------------------------------------ #
    # Pluggable seams — the concrete subclass picks the shard provider and
    # its matching normalization-stats source.
    # ------------------------------------------------------------------ #
    def _build_children(
        self,
        dataset_dirs,
        shape_meta,
        num_frames,
        global_sample_stride,
        val_set_proportion,
        is_training_set,
        seed,
        decode_workers,
        skip_corrupt_data=False,
        stats_shard=None,
    ):
        raise NotImplementedError(
            "Subclasses must implement _build_children to return one shard "
            "provider per dataset_dir."
        )

    def _build_stats(
        self,
        processor,
        dataset_dirs,
        shape_meta,
        num_frames,
        global_sample_stride,
        val_set_proportion,
        is_training_set,
        seed,
        skip_corrupt_data: bool = False,
    ):
        raise NotImplementedError(
            "Subclasses must implement _build_stats to return the group's "
            "normalization stats."
        )

    # ------------------------------------------------------------------ #
    # Post-processing tail — ZERO-DIFF COPY of robot_video_dataset.py:143-240
    # (operates on the processor.preprocess output). Keep in sync.
    # ------------------------------------------------------------------ #
    def _postprocess(self, sample):
        image_is_pad = sample["image_is_pad"]

        video = sample["pixel_values"]  # [T, C, H, W] or [num_cameras, T, C, H, W]
        num_cameras = 1
        if video.ndim == 5:
            video = video[:, self.video_sample_indices, :, :, :] # [num_cameras, T_video, C, H, W]
            num_cameras, T_video, C, H, W = video.shape
        else:
            assert video.ndim == 4, f"Expected video to have shape [T, C, H, W], but got {video.shape}"
            video = video[self.video_sample_indices, :, :, :] # [T_video, C, H, W]
            T_video, C, H, W = video.shape
        image_is_pad = image_is_pad[self.video_sample_indices]

        video = video.view(num_cameras, T_video, C, H, W)  # [num_cameras, T_video, C, H, W]
        if self.concat_multi_camera in {"pyramid_3cam", "robotwin"}:
            if num_cameras not in {2, 3}:
                raise ValueError(
                    f"`concat_multi_camera={self.concat_multi_camera!r}` requires "
                    f"2 or 3 cameras, got {num_cameras}"
                )
            cam_top = transforms_F.resize(
                video[0],
                size=[256, 320],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 256, 320]
            cam_left = transforms_F.resize(
                video[1],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            if num_cameras == 3:
                cam_right = transforms_F.resize(
                    video[2],
                    size=[128, 160],
                    interpolation=transforms_F.InterpolationMode.BILINEAR,
                    antialias=True,
                )  # [T_video, C, 128, 160]
            else:
                cam_right = torch.zeros_like(cam_left)
            bottom = torch.cat([cam_left, cam_right], dim=-1)  # [T_video, C, 128, 320]
            video = torch.cat([cam_top, bottom], dim=-2)  # [T_video, C, 384, 320]
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)  # [T_video, C, H, num_cameras*W]
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)  # [T_video, C, num_cameras*H, W]
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                    "Expected one of: horizontal, vertical, pyramid_3cam "
                    "(robotwin is a legacy alias)."
                )
        else:
            video = video.squeeze(0)  # [T_video, C, H, W]

        # The per-source config declares the final mosaic geometry.
        expected_hw = tuple(int(v) for v in self.video_size)
        if tuple(video.shape[-2:]) != expected_hw:
            raise ValueError(
                "Lance large-data mosaic shape mismatch: expected "
                f"{expected_hw}, got {tuple(video.shape[-2:])}"
            )
        if video.shape[-2] % 16 or video.shape[-1] % 16:
            raise ValueError(
                "Lance large-data mosaic height and width must be divisible by 16, "
                f"got {tuple(video.shape[-2:])}"
            )
        video = self.normalize_transform(video)  # [T_video, C, H, W]

        video = video.permute(1, 0, 2, 3) # [C, T_video, H, W], range [-1, 1]

        # Match the established LanceVLA contract: 32 future actions are paired
        # with the *current* state only.  A state trajectory here is both unused
        # by the model and easy to misalign at episode padding boundaries.
        action = sample["action"]  # [32, action_dim]
        proprio = sample["proprio"][0, :]  # [state_dim]
        proprio_is_pad = torch.as_tensor(sample["proprio_is_pad"], dtype=torch.bool)
        if proprio_is_pad.ndim != 1 or proprio_is_pad.numel() == 0:
            raise ValueError("proprio_is_pad must describe the source state trajectory")
        if bool(proprio_is_pad[0].item()):
            raise ValueError("The current state (t=0) is padded; quarantine this anchor")
        if video.shape[1] <= 1:
            raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")
        if action.shape[0] % (video.shape[1] - 1) != 0:
            raise ValueError(
                f"`action` horizon must be divisible by `video` transitions, got {action.shape[0]} and {video.shape[1] - 1}"
            )

        task = str(sample["instruction"])
        if self.override_instruction is not None:
            task = self.override_instruction

        data = {
            "video": video,
            "action": action,
            "proprio": proprio,
            # Keep raw robot language here.  The model and serving path both call
            # build_lance_text_ids(), which owns the byte-identical Lance chat and
            # system template used by the frozen understanding expert.
            "instruction": task,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": proprio_is_pad[0],
        }
        if self.text_metadata is not None:
            data["text_metadata"] = self.text_metadata
        # Forward per-dim and per-camera masks if present (from ConcatLeftAlign + processor).
        if "action_dim_is_pad" in sample:
            data["action_dim_is_pad"] = sample["action_dim_is_pad"]
        if "proprio_dim_is_pad" in sample:
            data["proprio_dim_is_pad"] = sample["proprio_dim_is_pad"]
        if "camera_is_pad" in sample:
            data["camera_is_pad"] = sample["camera_is_pad"]
        if "metadata" in sample:
            metadata = sample["metadata"]
            if not isinstance(metadata, dict):
                raise TypeError("provider sample metadata must be a dictionary")
            # Non-model provenance: the canonical adapter retains this mapping,
            # while the model collator deliberately does not add it to forward().
            data["metadata"] = dict(metadata)
        if "main_image" in sample:
            main_image = torch.as_tensor(sample["main_image"]).float()
            if tuple(main_image.shape) != (3, 224, 224):
                raise ValueError(
                    f"main_image must be [3,224,224], got {tuple(main_image.shape)}"
                )
            data["main_image"] = main_image
        return data



class ShardedLerobotV3VideoDataset(ShardedRobotVideoDataset):
    """Sharded group backed by :class:`ShardedLerobotV3Dataset` children.

    Reads LeRobot **v3.0** datasets (via ``LeRobotDatasetMetadata`` /
    ``file-NNN.parquet`` packing). Stats come from a throwaway multi-dir
    map-style :class:`BaseLerobotDataset` so the sharded path uses the SAME
    normalization as the map-style path.
    """

    def __init__(
        self,
        *args,
        decode_resize: Optional[List[int]] = None,
        main_image_key: Optional[str] = None,
        main_image_size: Optional[List[int]] = None,
        **kwargs,
    ):
        # Stash before super().__init__ because it invokes _build_children.
        self._decode_resize = (
            list(decode_resize) if decode_resize is not None else None
        )
        self._main_image_key = main_image_key
        self._main_image_size = list(main_image_size or [224, 224])
        super().__init__(*args, **kwargs)

    def _build_children(
        self,
        dataset_dirs,
        shape_meta,
        num_frames,
        global_sample_stride,
        val_set_proportion,
        is_training_set,
        seed,
        decode_workers,
        skip_corrupt_data=False,
        stats_shard=None,
    ) -> List[ShardedLerobotV3Dataset]:
        if stats_shard is not None:
            raise NotImplementedError(
                "stats_shard (distributed stats precompute) is only implemented "
                "for the raw h5 sources (AgiBot / RoboMIND2)."
            )
        return [
            ShardedLerobotV3Dataset(
                dataset_dir=ds_dir,
                shape_meta=shape_meta,
                num_frames=num_frames,
                global_sample_stride=global_sample_stride,
                val_set_proportion=val_set_proportion,
                is_training_set=is_training_set,
                seed=seed,
                decode_workers=decode_workers,
                decode_resize=self._decode_resize,
                skip_corrupt_data=skip_corrupt_data,
                dataset_name=self._dataset_name,
                minimum_future_offset=self.minimum_future_offset,
                main_image_key=self._main_image_key,
                main_image_size=self._main_image_size,
            )
            for ds_dir in dataset_dirs
        ]

    def _build_stats(
        self,
        processor,
        dataset_dirs,
        shape_meta,
        num_frames,
        global_sample_stride,
        val_set_proportion,
        is_training_set,
        seed,
        skip_corrupt_data: bool = False,
    ):
        # Use the same direct parquet extractor and select_indices path as
        # runtime.  The former BaseLerobotDataset stats path read raw columns and
        # therefore produced incorrect widths for every source that selects a
        # canonical subset of a physical action/state column.
        return ShardedLerobotV3Dataset.pooled_dataset_stats(
            self.children,
            processor,
            skip_corrupt_data=skip_corrupt_data,
        )


class ShardedLerobotV2VideoDataset(ShardedRobotVideoDataset):
    """Sharded group backed by :class:`ShardedLerobotV2Dataset` children.

    Reads the raw LeRobot **v2.x** on-disk format directly (one episode per
    parquet / mp4, JSON meta) since the installed ``lerobot`` only understands
    v3.0. The v2-specific ``decode_resize`` arg is stashed on ``self`` **before**
    ``super().__init__()`` so the overridden ``_build_children`` (invoked from
    the base ``__init__``) can read it.
    """

    _provider_cls = ShardedLerobotV2Dataset

    def __init__(
        self,
        *args,
        decode_resize: Optional[List[int]] = None,
        main_image_key: Optional[str] = None,
        main_image_size: Optional[List[int]] = None,
        **kwargs,
    ):
        # Stash before super().__init__ so _build_children sees it.
        self._decode_resize = list(decode_resize) if decode_resize is not None else None
        self._main_image_key = main_image_key
        self._main_image_size = list(main_image_size or [224, 224])
        super().__init__(*args, **kwargs)

    def _build_children(
        self,
        dataset_dirs,
        shape_meta,
        num_frames,
        global_sample_stride,
        val_set_proportion,
        is_training_set,
        seed,
        decode_workers,
        skip_corrupt_data=False,
        stats_shard=None,
    ) -> List[ShardedLerobotV2Dataset]:
        if stats_shard is not None:
            raise NotImplementedError(
                "stats_shard (distributed stats precompute) is only implemented "
                "for the raw h5 sources (AgiBot / RoboMIND2)."
            )
        return [
            self._provider_cls(
                dataset_dir=ds_dir,
                shape_meta=shape_meta,
                num_frames=num_frames,
                decode_resize=self._decode_resize,
                global_sample_stride=global_sample_stride,
                val_set_proportion=val_set_proportion,
                is_training_set=is_training_set,
                seed=seed,
                decode_workers=decode_workers,
                skip_corrupt_data=skip_corrupt_data,
                dataset_name=self._dataset_name,
                minimum_future_offset=self.minimum_future_offset,
                main_image_key=self._main_image_key,
                main_image_size=self._main_image_size,
            )
            for ds_dir in dataset_dirs
        ]

    def _build_stats(
        self,
        processor,
        dataset_dirs,
        shape_meta,
        num_frames,
        global_sample_stride,
        val_set_proportion,
        is_training_set,
        seed,
        skip_corrupt_data: bool = False,
    ):
        # Children already enumerate/split their episodes and read parquet only;
        # the union of their per-episode accumulators is the mixture
        # normalization source (a single pooled reduction, matching one
        # multi-dir pass).
        return ShardedLerobotV2Dataset.pooled_dataset_stats(
            self.children, processor, skip_corrupt_data=skip_corrupt_data
        )
