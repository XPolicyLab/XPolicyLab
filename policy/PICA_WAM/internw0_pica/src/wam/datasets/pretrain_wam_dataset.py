from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from wam.datasets import pretrain_lerobot_loader as pretrain_loader
from wam.datasets.pretrain_sample_decoder import PretrainSampleDecoderMixin
from wam.datasets.pretrain_stats import load_pretrain_stats


def _rank0_log(message: str) -> None:
    if str(os.environ.get("RANK", "0")) == "0":
        print(message, flush=True)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    try:
        return cfg.get(key, default)
    except Exception:
        return getattr(cfg, key, default)


class _PretrainGroupedNormalizer:
    def __init__(
        self,
        stats: dict[str, Any],
        *,
        mode: str = "q01/q99",
        minimum_widths: dict[str, Any] | None = None,
        passthrough_dims: dict[str, Any] | None = None,
    ) -> None:
        self.mode = str(stats.get("norm_mode") or mode)
        self.minimum_widths = minimum_widths or {}
        self.passthrough_dims: dict[str, tuple[int, ...]] = {}
        for kind in ("action", "state"):
            raw_dims = (passthrough_dims or {}).get(kind, ())
            if isinstance(raw_dims, int):
                raw_dims = [raw_dims]
            if not isinstance(raw_dims, (list, tuple)):
                raise TypeError(
                    f"stats.passthrough_dims.{kind} must be an int/list"
                )
            dims = tuple(sorted({int(dim) for dim in raw_dims}))
            if any(dim < 0 for dim in dims):
                raise ValueError(
                    f"stats.passthrough_dims.{kind} contains a negative dim"
                )
            self.passthrough_dims[kind] = dims
        raw_groups = stats.get("groups") if isinstance(stats, dict) else None
        if not isinstance(raw_groups, dict):
            raw_groups = {"default": stats}
        self.groups: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
        for group, group_stats in raw_groups.items():
            if not isinstance(group_stats, dict):
                continue
            self.groups[str(group)] = {}
            for kind in ("action", "state"):
                params = self._params(group_stats, kind, str(group))
                self.groups[str(group)][kind] = self._apply_passthrough(
                    params, kind=kind
                )
        if not self.groups:
            raise ValueError("Grouped pretrain stats contains no usable groups.")

    def _apply_passthrough(
        self,
        params: tuple[torch.Tensor, torch.Tensor],
        *,
        kind: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dims = self.passthrough_dims.get(kind, ())
        if not dims:
            return params
        scale, offset = (tensor.clone() for tensor in params)
        if max(dims) >= int(scale.numel()):
            raise IndexError(
                f"stats.passthrough_dims.{kind}={list(dims)} exceeds "
                f"feature dim {scale.numel()}"
            )
        index = torch.as_tensor(dims, dtype=torch.long)
        scale[index] = 1.0
        offset[index] = 0.0
        return scale, offset

    def _feature_stats(self, group_stats: dict[str, Any], kind: str) -> dict[str, Any]:
        value = group_stats.get(kind)
        if isinstance(value, dict) and "default" in value:
            return value["default"]
        if isinstance(value, dict):
            return value
        raise KeyError(f"Stats group is missing {kind!r} stats.")

    def _params(self, group_stats: dict[str, Any], kind: str, group: str) -> tuple[torch.Tensor, torch.Tensor]:
        stats = self._feature_stats(group_stats, kind)
        prefix = "global_"
        if self.mode == "z-score":
            mean = torch.as_tensor(stats[prefix + "mean"], dtype=torch.float32)
            std = torch.as_tensor(stats[prefix + "std"], dtype=torch.float32).clamp_min(1e-8)
            return 1.0 / std, -mean / std
        if self.mode == "min/max":
            lo_key, hi_key = prefix + "min", prefix + "max"
        elif self.mode == "q01/q99":
            lo_key, hi_key = prefix + "q01", prefix + "q99"
        else:
            lo_text, hi_text = self.mode.split("/", 1)
            ref = torch.as_tensor(stats[prefix + "min"], dtype=torch.float32)
            lo = torch.full_like(ref, float(lo_text))
            hi = torch.full_like(ref, float(hi_text))
            return self._range_params(lo, hi, group=group, kind=kind)
        lo = torch.as_tensor(stats[lo_key], dtype=torch.float32)
        hi = torch.as_tensor(stats[hi_key], dtype=torch.float32)
        return self._range_params(lo, hi, group=group, kind=kind)

    def _range_params(
        self,
        lo: torch.Tensor,
        hi: torch.Tensor,
        *,
        group: str,
        kind: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        width = hi - lo
        ignore = width.abs() < 1e-4
        raw_rules = self.minimum_widths.get(kind, {}).get(group)
        rules = raw_rules if isinstance(raw_rules, list) else [raw_rules]
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            requested = torch.zeros_like(width)
            requested[torch.as_tensor(rule["dims"], dtype=torch.long)] = float(rule["width"])
            expanded_width = torch.maximum(width, requested)
            expand = (expanded_width > width) & ~ignore
            center = (lo + hi) * 0.5
            lo = torch.where(expand, center - expanded_width * 0.5, lo)
            width = expanded_width
        width = torch.where(ignore, torch.ones_like(width) * 2.0, width)
        scale = 2.0 / width
        offset = -1.0 - scale * lo
        offset = torch.where(ignore, -lo, offset)
        return scale, offset

    def _get_params(self, *, group: str, kind: str) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.groups.get(group) or self.groups.get("default")
        if params is None:
            raise KeyError(f"No pretrain stats group {group!r}; available={sorted(self.groups)}")
        return params[kind]

    def forward(self, value: torch.Tensor, *, group: str, kind: str) -> torch.Tensor:
        scale, offset = self._get_params(group=group, kind=kind)
        scale = scale.to(device=value.device, dtype=value.dtype)
        offset = offset.to(device=value.device, dtype=value.dtype)
        return torch.clamp(value * scale + offset, -5.0, 5.0)

    def inverse(self, value: torch.Tensor, *, group: str, kind: str) -> torch.Tensor:
        scale, offset = self._get_params(group=group, kind=kind)
        scale = scale.to(device=value.device, dtype=value.dtype).clamp_min(1e-8)
        offset = offset.to(device=value.device, dtype=value.dtype)
        return (value - offset) / scale


class PretrainWAMDataset(PretrainSampleDecoderMixin, Dataset):
    """Training wrapper around the local mixed LeRobot loader.

    The mixed LeRobot loader projects each source into the canonical robot
    canvas. This class adapts its samples to the final dictionary consumed by
    WAMTrainer/WAM.
    """

    CANONICAL_DIM = 80
    SUPPORTED_OUTPUT_DIMS = {80}

    @staticmethod
    def _weighted_action_loss_pool_families(
        specs: list[pretrain_loader.DatasetSpec],
    ) -> tuple[str, ...]:
        """Return stable family pools for weighted-valid-cell aggregation.

        A family must not mix legacy and weighted action-loss policies.  For
        older standalone Ego profiles without ``sampling_family``, retain one
        shared fallback pool instead of changing their historical behavior.
        """

        policies_by_family: dict[str, set[str]] = {}
        for spec in specs:
            family = str(spec.sampling_family or "__weighted_default__")
            policies_by_family.setdefault(family, set()).add(
                str(spec.action_loss_normalization).strip().lower()
            )
        mixed = {
            family: sorted(policies)
            for family, policies in policies_by_family.items()
            if "weighted_valid_cells" in policies and len(policies) != 1
        }
        if mixed:
            raise ValueError(
                "A sampling family cannot mix weighted_valid_cells and legacy "
                f"action-loss policies: {mixed}"
            )
        return tuple(
            sorted(
                family
                for family, policies in policies_by_family.items()
                if policies == {"weighted_valid_cells"}
            )
        )

    def __init__(
        self,
        dataset_config: str | Path,
        *,
        num_frames: int = 33,
        action_size: int | None = 32,
        global_sample_stride: int = 1,
        action_video_freq_ratio: int = 4,
        video_size: list[int] | tuple[int, int] = (224, 448),
        is_training_set: bool = True,
        epoch_length: int | None = None,
        eval_fixed_per_dataset: bool | None = None,
        eval_episode_pos: int | None = None,
        eval_frame_index: int | None = None,
        eval_scope: str | None = None,
        eval_segments_per_source: int | None = None,
        eval_total_samples: int | None = None,
        eval_episodes_per_dataset: int | None = None,
        eval_frames_per_episode: int | None = None,
        eval_frame_policy: str | None = None,
        text_context_required: bool = True,
        concat_multi_camera: str | None = None,
        shape_meta: Any | None = None,
        processor: Any | None = None,
        pretrained_norm_stats: str | None = None,
        norm_default_mode: str = "q01/q99",
        memory_video_anchor_size: int | None = None,
        memory_recent_frame_offset: int | None = None,
        skip_bad_videos: bool = True,
        max_decode_retries: int = 8,
        vlm_max_pixels: int = 65536,
        training_block_size: int | None = None,
        training_batch_size_per_rank: int | None = None,
        optimizer_gradient_accumulation_steps: int = 1,
    ) -> None:
        del shape_meta
        self.training_batch_size_per_rank = training_batch_size_per_rank
        self.optimizer_gradient_accumulation_steps = max(1, int(optimizer_gradient_accumulation_steps))
        self.dataset_config = Path(dataset_config)
        self.num_frames = int(num_frames)
        self.action_size = int(action_size if action_size is not None else max(self.num_frames - 1, 1))
        if self.action_size != self.num_frames - 1:
            raise ValueError(
                f"action_size must equal num_frames - 1 in frame mode, "
                f"got action_size={self.action_size}, num_frames={self.num_frames}. "
                f"Use num_frames={self.action_size + 1} (or action_size={self.num_frames - 1})."
            )
        self.global_sample_stride = int(global_sample_stride)
        self.action_video_freq_ratio = int(action_video_freq_ratio)
        self.video_size = (int(video_size[0]), int(video_size[1]))
        self.concat_multi_camera = None if concat_multi_camera in (None, "", "auto", "source") else str(concat_multi_camera).strip()
        self.is_training_set = bool(is_training_set)
        self.action_output_dim = int(_cfg_get(processor, "action_output_dim", self.CANONICAL_DIM))
        self.proprio_output_dim = int(_cfg_get(processor, "proprio_output_dim", self.action_output_dim))
        for name, dim in (("action_output_dim", self.action_output_dim), ("proprio_output_dim", self.proprio_output_dim)):
            if dim not in self.SUPPORTED_OUTPUT_DIMS:
                raise ValueError(f"PretrainWAMDataset supports 80D robot outputs, got {name}={dim}.")
        self.text_context_required = bool(text_context_required)
        self.benchmark_stage_times = False
        self.skip_bad_videos = bool(skip_bad_videos) and bool(is_training_set)
        self.max_decode_retries = max(0, int(max_decode_retries))
        self.vlm_max_pixels = max(1, int(vlm_max_pixels))
        self.training_block_size = (
            None if training_block_size is None else max(1, int(training_block_size))
        )
        self._bad_video_logs = 0
        self._bad_video_log_limit = 20

        init_started = time.perf_counter()
        split_name = "train" if bool(is_training_set) else "val"
        _rank0_log(f"[pretrain-dataset] init {split_name}: config={self.dataset_config}")
        config = pretrain_loader._read_dataset_config(self.dataset_config)
        self.config = config
        sampling_cfg = config.get("sampling", {}) if isinstance(config.get("sampling"), dict) else {}
        if memory_video_anchor_size is None:
            memory_video_anchor_size = sampling_cfg.get("memory_video_anchor_size", 0)
        if memory_recent_frame_offset is None:
            memory_recent_frame_offset = sampling_cfg.get("memory_recent_frame_offset", 16)
        self.memory_video_anchor_size = max(0, int(memory_video_anchor_size or 0))
        self.memory_recent_frame_offset = max(1, int(memory_recent_frame_offset or 16))
        stats_cfg = config.get("stats", {}) if isinstance(config.get("stats"), dict) else {}
        self.post_normalize_transforms = self._parse_post_normalize_transforms(config.get("post_normalize_transforms"))
        ns = argparse.Namespace(dataset_specs_json=None, remote_root=None, name="dataset")
        specs = pretrain_loader._load_specs(ns, config)
        expanded_specs = len(specs)
        specs = [spec for spec in specs if spec.dataset_weight > 0]
        _rank0_log(
            f"[pretrain-dataset] init {split_name}: active specs={len(specs)}/{expanded_specs}"
        )
        payload, stats_files = load_pretrain_stats(specs, config, pretrained_norm_stats)
        self.grouped_normalizer = None
        if payload is not None:
            self.grouped_normalizer = _PretrainGroupedNormalizer(
                payload,
                mode=str(stats_cfg.get("norm_mode", norm_default_mode)),
                minimum_widths=stats_cfg.get("minimum_widths"),
                passthrough_dims=stats_cfg.get("passthrough_dims"),
            )
            _rank0_log(
                f"[pretrain-dataset] normalization: files={len(stats_files)} "
                f"groups={len(payload['groups'])} mode={self.grouped_normalizer.mode}"
            )
        self.action_loss_weighted_valid_cells_families = self._weighted_action_loss_pool_families(specs)
        self.action_loss_weighted_valid_cells_family_to_pool = {
            name: i for i, name in enumerate(self.action_loss_weighted_valid_cells_families)
        }
        self.specs = specs
        self.stats_groups = [spec.stats_group or spec.name for spec in specs]
        self.datasets = []
        dataset_started = time.perf_counter()
        for spec_idx, spec in enumerate(specs, start=1):
            dataset = pretrain_loader.PretrainLeRobotDataset(
                spec,
                num_frames=self.num_frames,
                action_size=self.action_size,
                global_sample_stride=self.global_sample_stride,
                use_path_index_cache=True,
            )
            self.datasets.append(dataset)
            if spec_idx == len(specs) or spec_idx % 100 == 0:
                elapsed = time.perf_counter() - dataset_started
                _rank0_log(
                    f"[pretrain-dataset] init {split_name}: sharded metadata {spec_idx}/{len(specs)} elapsed={elapsed:.1f}s"
                )

        mixture_cfg = config.get("mixture", {}) if isinstance(config.get("mixture"), dict) else {}
        eval_cfg = mixture_cfg.get("eval_subset", mixture_cfg.get("eval", {}))
        if not isinstance(eval_cfg, dict):
            eval_cfg = {}
        training = bool(is_training_set)
        allow_padding_at_end = bool(mixture_cfg.get("allow_padding_at_end", False))
        self.allow_padding_at_end = allow_padding_at_end
        dataset_weights = pretrain_loader._mixture_dataset_weights(
            specs,
            self.datasets,
            mixture_cfg,
            allow_padding_at_end=allow_padding_at_end,
        )
        mixed_epoch_length = mixture_cfg.get("epoch_length") if epoch_length is None else epoch_length
        if not training:
            if eval_fixed_per_dataset is None:
                eval_fixed_per_dataset = bool(eval_cfg.get("fixed_per_dataset", True))
            if eval_episode_pos is None:
                eval_episode_pos = eval_cfg.get("episode_pos", 0)
            if eval_frame_index is None:
                eval_frame_index = eval_cfg.get("frame_index", 0)
            if eval_scope is None:
                eval_scope = eval_cfg.get("scope", eval_cfg.get("eval_scope", "dataset"))
            if eval_segments_per_source is None:
                eval_segments_per_source = eval_cfg.get("segments_per_source", eval_cfg.get("eval_segments_per_source"))
            if eval_total_samples is None:
                eval_total_samples = eval_cfg.get("total_samples", eval_cfg.get("eval_total_samples"))
            if eval_episodes_per_dataset is None:
                eval_episodes_per_dataset = eval_cfg.get("episodes_per_dataset", eval_cfg.get("eval_episodes_per_dataset"))
            if eval_frames_per_episode is None:
                eval_frames_per_episode = eval_cfg.get("frames_per_episode", eval_cfg.get("eval_frames_per_episode"))
            if eval_frame_policy is None:
                eval_frame_policy = eval_cfg.get("frame_policy", "first")

        self.mixed_dataset = self._make_mixed_dataset(
            training=training,
            dataset_weights=dataset_weights,
            mixture_cfg=mixture_cfg,
            eval_cfg=eval_cfg,
            mixed_epoch_length=mixed_epoch_length,
            eval_fixed_per_dataset=eval_fixed_per_dataset,
            eval_episode_pos=eval_episode_pos,
            eval_frame_index=eval_frame_index,
            eval_scope=eval_scope,
            eval_segments_per_source=eval_segments_per_source,
            eval_total_samples=eval_total_samples,
            eval_episodes_per_dataset=eval_episodes_per_dataset,
            eval_frames_per_episode=eval_frames_per_episode,
            eval_frame_policy=eval_frame_policy,
        )
        self.prefer_sequential_indices = bool(getattr(self.mixed_dataset, "prefer_sequential_indices", False))
        _rank0_log(f"[pretrain-dataset] init {split_name}: done elapsed={time.perf_counter() - init_started:.1f}s len={len(self)}")


    def _make_mixed_dataset(
        self,
        *,
        training: bool,
        dataset_weights: list[float],
        mixture_cfg: dict[str, Any],
        eval_cfg: dict[str, Any],
        mixed_epoch_length: Any = None,
        eval_fixed_per_dataset: Any = None,
        eval_episode_pos: Any = None,
        eval_frame_index: Any = None,
        eval_scope: Any = None,
        eval_segments_per_source: Any = None,
        eval_total_samples: Any = None,
        eval_episodes_per_dataset: Any = None,
        eval_frames_per_episode: Any = None,
        eval_frame_policy: Any = None,
    ) -> pretrain_loader.PretrainLeRobotMixture:
        allow_padding_at_end = bool(mixture_cfg.get("allow_padding_at_end", False))
        return pretrain_loader.PretrainLeRobotMixture(
            self.datasets,
            dataset_weights=dataset_weights,
            training=training,
            balance_dataset_weights=bool(mixture_cfg.get("balance_dataset_weights", False)),
            balance_trajectory_weights=bool(mixture_cfg.get("balance_trajectory_weights", True)),
            seed=int(mixture_cfg.get("seed", 42)),
            allow_padding_at_end=allow_padding_at_end,
            epoch_length=None if mixed_epoch_length in (None, "") else int(mixed_epoch_length),
            eval_episodes_per_dataset=None if eval_episodes_per_dataset in (None, "") else int(eval_episodes_per_dataset),
            eval_frames_per_episode=None if eval_frames_per_episode in (None, "") else int(eval_frames_per_episode),
            eval_frame_policy=str(eval_frame_policy or "first"),
            eval_scope=str(eval_scope or "dataset"),
            eval_segments_per_source=None if eval_segments_per_source in (None, "") else int(eval_segments_per_source),
            eval_total_samples=None if eval_total_samples in (None, "") else int(eval_total_samples),
            eval_fixed_per_dataset=bool(eval_fixed_per_dataset) if not training else False,
            eval_fixed_episode_pos=None if eval_episode_pos in (None, "") else int(eval_episode_pos),
            eval_fixed_frame_index=0 if eval_frame_index in (None, "") else int(eval_frame_index),
            eval_exclude_from_training=eval_cfg.get("exclude_from_training") if training else None,
            training_block_size=int(
                self.training_block_size
                or mixture_cfg.get("training_block_size", mixture_cfg.get("block_size", 1))
            ),
            training_batch_size_per_rank=int(
                self.training_batch_size_per_rank
                or mixture_cfg.get(
                    "training_batch_size_per_rank",
                    self.training_block_size
                    or mixture_cfg.get("training_block_size", mixture_cfg.get("block_size", 1)),
                )
            ),
            training_frame_stride=int(mixture_cfg.get("training_frame_stride", 1)),
            training_sampling_strategy=str(
                mixture_cfg.get("training_sampling_strategy", "optimizer_stratified_without_replacement")
            ),
            episode_block_tile_size=int(
                mixture_cfg.get("episode_block_tile_size", 64)
            ),
            bucket_wave_steps=int(mixture_cfg.get("bucket_wave_steps", 16)),
            optimizer_gradient_accumulation_steps=(
                self.optimizer_gradient_accumulation_steps
            ),
            bucket_wave_apply_dataset_weights=bool(
                mixture_cfg.get("bucket_wave_apply_dataset_weights", False)
            ),
            bucket_wave_epoch_anchor_family=mixture_cfg.get(
                "bucket_wave_epoch_anchor_family"
            ),
            bucket_wave_cyclic_families=mixture_cfg.get(
                "bucket_wave_cyclic_families", []
            ),
            coverage_min_samples_per_dataset=int(
                mixture_cfg.get("coverage_min_samples_per_dataset", 0)
            ),
        )

    def make_validation_dataset(self, val_cfg: Any = None, *, pretrained_norm_stats: str | None = None) -> "PretrainWAMDataset":
        del pretrained_norm_stats
        _rank0_log("[pretrain-dataset] init val: reusing train metadata")
        obj = object.__new__(type(self))
        obj.dataset_config = self.dataset_config
        obj.num_frames = int(_cfg_get(val_cfg, "num_frames", self.num_frames))
        obj.action_size = int(_cfg_get(val_cfg, "action_size", self.action_size))
        if obj.action_size != obj.num_frames - 1:
            raise ValueError(
                f"action_size must equal num_frames - 1 in frame mode, "
                f"got action_size={obj.action_size}, num_frames={obj.num_frames}. "
                f"Use num_frames={obj.action_size + 1} (or action_size={obj.num_frames - 1})."
            )
        obj.global_sample_stride = int(_cfg_get(val_cfg, "global_sample_stride", self.global_sample_stride))
        obj.action_video_freq_ratio = int(_cfg_get(val_cfg, "action_video_freq_ratio", self.action_video_freq_ratio))
        obj.video_size = tuple(_cfg_get(val_cfg, "video_size", self.video_size))
        obj.video_size = (int(obj.video_size[0]), int(obj.video_size[1]))
        obj.concat_multi_camera = self.concat_multi_camera
        obj.action_output_dim = self.action_output_dim
        obj.proprio_output_dim = self.proprio_output_dim
        obj.text_context_required = bool(_cfg_get(val_cfg, "text_context_required", self.text_context_required))
        obj.is_training_set = False
        obj.memory_video_anchor_size = int(_cfg_get(val_cfg, "memory_video_anchor_size", self.memory_video_anchor_size))
        obj.memory_recent_frame_offset = int(_cfg_get(val_cfg, "memory_recent_frame_offset", self.memory_recent_frame_offset))
        obj.skip_bad_videos = self.skip_bad_videos
        obj.max_decode_retries = self.max_decode_retries
        obj.vlm_max_pixels = int(_cfg_get(val_cfg, "vlm_max_pixels", self.vlm_max_pixels))
        obj.action_loss_weighted_valid_cells_families = self.action_loss_weighted_valid_cells_families
        obj.action_loss_weighted_valid_cells_family_to_pool = self.action_loss_weighted_valid_cells_family_to_pool
        obj.training_block_size = None
        obj.training_batch_size_per_rank = None
        obj.optimizer_gradient_accumulation_steps = self.optimizer_gradient_accumulation_steps
        obj._bad_video_logs = 0
        obj._bad_video_log_limit = self._bad_video_log_limit
        obj.benchmark_stage_times = False
        obj.config = self.config
        obj.grouped_normalizer = self.grouped_normalizer
        obj.post_normalize_transforms = self.post_normalize_transforms
        obj.specs = self.specs
        obj.stats_groups = self.stats_groups
        obj.datasets = self.datasets

        mixture_cfg = obj.config.get("mixture", {}) if isinstance(obj.config.get("mixture"), dict) else {}
        eval_cfg = mixture_cfg.get("eval_subset", mixture_cfg.get("eval", {}))
        if not isinstance(eval_cfg, dict):
            eval_cfg = {}
        dataset_weights = pretrain_loader._mixture_dataset_weights(
            obj.specs,
            obj.datasets,
            mixture_cfg,
            allow_padding_at_end=bool(mixture_cfg.get("allow_padding_at_end", False)),
        )
        obj.allow_padding_at_end = bool(mixture_cfg.get("allow_padding_at_end", False))
        eval_fixed_per_dataset = _cfg_get(val_cfg, "eval_fixed_per_dataset", eval_cfg.get("fixed_per_dataset", True))
        eval_episode_pos = _cfg_get(val_cfg, "eval_episode_pos", eval_cfg.get("episode_pos", 0))
        eval_frame_index = _cfg_get(val_cfg, "eval_frame_index", eval_cfg.get("frame_index", 0))
        eval_scope = _cfg_get(val_cfg, "eval_scope", eval_cfg.get("scope", eval_cfg.get("eval_scope", "dataset")))
        eval_segments_per_source = _cfg_get(
            val_cfg, "eval_segments_per_source", eval_cfg.get("segments_per_source", eval_cfg.get("eval_segments_per_source"))
        )
        eval_total_samples = _cfg_get(
            val_cfg, "eval_total_samples", eval_cfg.get("total_samples", eval_cfg.get("eval_total_samples"))
        )
        eval_episodes_per_dataset = _cfg_get(
            val_cfg, "eval_episodes_per_dataset", eval_cfg.get("episodes_per_dataset", eval_cfg.get("eval_episodes_per_dataset"))
        )
        eval_frames_per_episode = _cfg_get(
            val_cfg, "eval_frames_per_episode", eval_cfg.get("frames_per_episode", eval_cfg.get("eval_frames_per_episode"))
        )
        eval_frame_policy = _cfg_get(val_cfg, "eval_frame_policy", eval_cfg.get("frame_policy", "first"))
        obj.mixed_dataset = obj._make_mixed_dataset(
            training=False,
            dataset_weights=dataset_weights,
            mixture_cfg=mixture_cfg,
            eval_cfg=eval_cfg,
            mixed_epoch_length=None,
            eval_fixed_per_dataset=eval_fixed_per_dataset,
            eval_episode_pos=eval_episode_pos,
            eval_frame_index=eval_frame_index,
            eval_scope=eval_scope,
            eval_segments_per_source=eval_segments_per_source,
            eval_total_samples=eval_total_samples,
            eval_episodes_per_dataset=eval_episodes_per_dataset,
            eval_frames_per_episode=eval_frames_per_episode,
            eval_frame_policy=eval_frame_policy,
        )
        obj.prefer_sequential_indices = bool(getattr(obj.mixed_dataset, "prefer_sequential_indices", False))
        _rank0_log(f"[pretrain-dataset] init val: reused metadata len={len(obj)}")
        return obj


    @staticmethod
    def _parse_dims(raw: Any) -> list[int]:
        if raw is None:
            return []
        if isinstance(raw, int):
            raw = [raw]
        if not isinstance(raw, (list, tuple)):
            raise TypeError(f"post_normalize_transforms dims must be an int/list, got {type(raw)}")
        dims: list[int] = []
        for item in raw:
            dim = int(item)
            if dim < 0:
                raise ValueError(f"post_normalize_transforms dims must be non-negative, got {dim}")
            dims.append(dim)
        return sorted(set(dims))

    @classmethod
    def _parse_post_normalize_transforms(cls, config: Any) -> dict[str, dict[str, list[dict[str, Any]]]]:
        if not isinstance(config, dict):
            return {}
        parsed: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for kind in ("action", "state"):
            kind_cfg = config.get(kind)
            if not isinstance(kind_cfg, dict):
                continue
            by_group = kind_cfg.get("by_stats_group", kind_cfg.get("groups"))
            if not isinstance(by_group, dict):
                continue
            parsed[kind] = {}
            for group, entries in by_group.items():
                if isinstance(entries, dict):
                    entries = [entries]
                if not isinstance(entries, list):
                    raise TypeError(
                        f"post_normalize_transforms.{kind}.by_stats_group[{group!r}] must be a list"
                    )
                rules: list[dict[str, Any]] = []
                for entry in entries:
                    if not isinstance(entry, dict):
                        raise TypeError(
                            f"post_normalize_transforms.{kind}.by_stats_group[{group!r}] entries must be mappings"
                        )
                    dims = cls._parse_dims(entry.get("dims", entry.get("dim")))
                    if not dims:
                        raise ValueError(
                            f"post_normalize_transforms.{kind}.by_stats_group[{group!r}] is missing dims"
                        )
                    scale = float(entry.get("scale", 1.0))
                    if abs(scale) < 1e-12:
                        raise ValueError(
                            f"post_normalize_transforms.{kind}.by_stats_group[{group!r}] scale must be non-zero"
                        )
                    rule: dict[str, Any] = {
                        "dims": dims,
                        "scale": scale,
                        "offset": float(entry.get("offset", 0.0)),
                    }
                    clamp = entry.get("clamp")
                    if clamp is not None:
                        if not isinstance(clamp, (list, tuple)) or len(clamp) != 2:
                            raise ValueError(
                                f"post_normalize_transforms.{kind}.by_stats_group[{group!r}].clamp must be [min, max]"
                            )
                        rule["clamp"] = (float(clamp[0]), float(clamp[1]))
                    rules.append(rule)
                parsed[kind][str(group)] = rules
        return parsed

    def _post_normalize_rules(self, *, group: str, kind: str) -> list[dict[str, Any]]:
        by_group = self.post_normalize_transforms.get(kind, {})
        return by_group.get(str(group)) or by_group.get("*") or by_group.get("default") or []

    def _apply_post_normalize_transforms(
        self,
        value: torch.Tensor,
        *,
        group: str,
        kind: str,
        inverse: bool = False,
    ) -> torch.Tensor:
        rules = self._post_normalize_rules(group=group, kind=kind)
        if not rules:
            return value
        out = value.clone()
        feature_dim = int(out.shape[-1])
        for rule in rules:
            dims = [dim for dim in rule["dims"] if int(dim) < feature_dim]
            if not dims:
                continue
            idx = torch.as_tensor(dims, device=out.device, dtype=torch.long)
            selected = out.index_select(-1, idx)
            scale = torch.as_tensor(float(rule["scale"]), device=out.device, dtype=out.dtype)
            offset = torch.as_tensor(float(rule["offset"]), device=out.device, dtype=out.dtype)
            if inverse:
                selected = (selected - offset) / scale
            else:
                selected = selected * scale + offset
                if "clamp" in rule:
                    lo, hi = rule["clamp"]
                    selected = selected.clamp(min=float(lo), max=float(hi))
            out[..., idx] = selected
        return out

    @classmethod
    def _output_indices(cls, dim: int) -> tuple[int, ...] | None:
        dim = int(dim)
        if dim == cls.CANONICAL_DIM:
            return None
        raise ValueError(f"Unsupported pretrain output dim: {dim}")

    def _project_feature_dim(
        self,
        value: torch.Tensor,
        dim_is_pad: torch.Tensor,
        *,
        output_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices = self._output_indices(output_dim)
        if indices is None:
            return value, dim_is_pad
        idx = torch.as_tensor(indices, device=value.device, dtype=torch.long)
        value = value.index_select(-1, idx)
        dim_is_pad = dim_is_pad.to(device=value.device).index_select(0, idx).to(device=dim_is_pad.device)
        return value, dim_is_pad

    def _inverse_normalized_feature(self, value: torch.Tensor, *, group: str, kind: str) -> torch.Tensor:
        if self.grouped_normalizer is None:
            return value
        value = self._apply_post_normalize_transforms(value, group=group, kind=kind, inverse=True)
        feature_dim = int(value.shape[-1])
        if feature_dim == self.CANONICAL_DIM:
            return self.grouped_normalizer.inverse(value, group=group, kind=kind)
        indices = self._output_indices(feature_dim)
        if indices is None:
            return self.grouped_normalizer.inverse(value, group=group, kind=kind)
        scale, offset = self.grouped_normalizer._get_params(group=group, kind=kind)
        idx = torch.as_tensor(indices, device=value.device, dtype=torch.long)
        scale = scale.to(device=value.device, dtype=value.dtype).index_select(0, idx).clamp_min(1e-8)
        offset = offset.to(device=value.device, dtype=value.dtype).index_select(0, idx)
        return (value - offset) / scale

    def denormalize_action(
        self,
        action: torch.Tensor,
        *,
        stats_group: str | list[str] | tuple[str, ...] | None = None,
        dim_is_pad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.grouped_normalizer is None:
            return action.detach().to(device="cpu", dtype=torch.float32).clone()
        if not isinstance(action, torch.Tensor):
            raise TypeError(f"action must be a torch.Tensor, got {type(action)}")

        squeeze_batch = False
        value = action.detach().to(device="cpu", dtype=torch.float32)
        if value.ndim == 2:
            value = value.unsqueeze(0)
            squeeze_batch = True
        if value.ndim != 3:
            raise ValueError(f"action must have shape [T,D] or [B,T,D], got {tuple(value.shape)}")

        if stats_group is None:
            groups = [self.stats_groups[0] if self.stats_groups else "default"] * int(value.shape[0])
        elif isinstance(stats_group, str):
            groups = [stats_group] * int(value.shape[0])
        else:
            groups = [str(group) for group in stats_group]
            if len(groups) == 1 and value.shape[0] != 1:
                groups = groups * int(value.shape[0])
        if len(groups) != int(value.shape[0]):
            raise ValueError(f"stats_group batch mismatch: {len(groups)} vs action batch {value.shape[0]}")

        out = torch.empty_like(value)
        for batch_idx, group in enumerate(groups):
            out[batch_idx] = self._inverse_normalized_feature(value[batch_idx], group=group, kind="action")

        if dim_is_pad is not None:
            pad = dim_is_pad.detach().to(device=out.device, dtype=torch.bool)
            if pad.ndim == 1:
                out[:, :, pad] = 0.0
            elif pad.ndim == 2:
                if pad.shape[0] != out.shape[0]:
                    raise ValueError(f"dim_is_pad batch mismatch: {tuple(pad.shape)} vs action {tuple(out.shape)}")
                for batch_idx in range(out.shape[0]):
                    out[batch_idx, :, pad[batch_idx]] = 0.0
            else:
                raise ValueError(f"dim_is_pad must have shape [D] or [B,D], got {tuple(pad.shape)}")

        return out.squeeze(0) if squeeze_batch else out

    def denormalize_state(
        self,
        state: torch.Tensor,
        *,
        stats_group: str | list[str] | tuple[str, ...] | None = None,
        dim_is_pad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.grouped_normalizer is None:
            return state.detach().to(device="cpu", dtype=torch.float32).clone()
        if not isinstance(state, torch.Tensor):
            raise TypeError(f"state must be a torch.Tensor, got {type(state)}")

        squeeze_batch = False
        value = state.detach().to(device="cpu", dtype=torch.float32)
        if value.ndim == 2:
            value = value.unsqueeze(0)
            squeeze_batch = True
        if value.ndim != 3:
            raise ValueError(f"state must have shape [T,D] or [B,T,D], got {tuple(value.shape)}")

        if stats_group is None:
            groups = [self.stats_groups[0] if self.stats_groups else "default"] * int(value.shape[0])
        elif isinstance(stats_group, str):
            groups = [stats_group] * int(value.shape[0])
        else:
            groups = [str(group) for group in stats_group]
            if len(groups) == 1 and value.shape[0] != 1:
                groups = groups * int(value.shape[0])
        if len(groups) != int(value.shape[0]):
            raise ValueError(f"stats_group batch mismatch: {len(groups)} vs state batch {value.shape[0]}")

        out = torch.empty_like(value)
        for batch_idx, group in enumerate(groups):
            out[batch_idx] = self._inverse_normalized_feature(value[batch_idx], group=group, kind="state")

        if dim_is_pad is not None:
            pad = dim_is_pad.detach().to(device=out.device, dtype=torch.bool)
            if pad.ndim == 1:
                out[:, :, pad] = 0.0
            elif pad.ndim == 2:
                if pad.shape[0] != out.shape[0]:
                    raise ValueError(f"dim_is_pad batch mismatch: {tuple(pad.shape)} vs state {tuple(out.shape)}")
                for batch_idx in range(out.shape[0]):
                    out[batch_idx, :, pad[batch_idx]] = 0.0
            else:
                raise ValueError(f"dim_is_pad must have shape [D] or [B,D], got {tuple(pad.shape)}")

        return out.squeeze(0) if squeeze_batch else out

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        # Exact action-cell masks and per-dimension loss weights are emitted
        # only by datasets that need them (notably Ego).  Fill neutral values
        # for legacy robot samples so mixed batches retain a uniform schema.
        if any("action_mask" in item for item in batch):
            batch = [
                item
                if "action_mask" in item
                else {
                    **item,
                    "action_mask": torch.ones_like(
                        item["action"], dtype=torch.bool
                    ),
                }
                for item in batch
            ]
        if any("action_dim_loss_weight" in item for item in batch):
            batch = [
                item
                if "action_dim_loss_weight" in item
                else {
                    **item,
                    "action_dim_loss_weight": torch.ones_like(item["action"][0]),
                }
                for item in batch
            ]
        if any("action_loss_weight" in item for item in batch):
            batch = [
                item
                if "action_loss_weight" in item
                else {
                    **item,
                    "action_loss_weight": torch.tensor(1.0),
                }
                for item in batch
            ]
        if any("action_loss_weighted_valid_cells" in item for item in batch):
            batch = [
                item
                if "action_loss_weighted_valid_cells" in item
                else {
                    **item,
                    "action_loss_weighted_valid_cells": torch.tensor(False),
                }
                for item in batch
            ]
        if any("action_loss_weighted_valid_cells_pool" in item for item in batch):
            pool_template = next(
                item["action_loss_weighted_valid_cells_pool"]
                for item in batch
                if "action_loss_weighted_valid_cells_pool" in item
            )
            batch = [
                item
                if "action_loss_weighted_valid_cells_pool" in item
                else {
                    **item,
                    "action_loss_weighted_valid_cells_pool": torch.zeros_like(
                        pool_template, dtype=torch.bool
                    ),
                }
                for item in batch
            ]

        # Sources may have different camera counts. Pad only the VLM view axis
        # and retain an explicit mask; every other field keeps default_collate.
        image_key = "vlm_current_images"
        if not batch or image_key not in batch[0]:
            return default_collate(batch)

        images = [sample[image_key] for sample in batch]
        if any(
            not isinstance(value, torch.Tensor) or value.ndim != 4
            for value in images
        ):
            shapes = [
                tuple(value.shape)
                if isinstance(value, torch.Tensor)
                else type(value).__name__
                for value in images
            ]
            raise ValueError(
                "`vlm_current_images` must contain [V,C,H,W] tensors, "
                f"got {shapes}."
            )
        image_shape = tuple(images[0].shape[1:])
        if any(tuple(value.shape[1:]) != image_shape for value in images):
            raise ValueError(
                "`vlm_current_images` camera shapes must match apart from "
                f"the view axis, got {[tuple(value.shape) for value in images]}."
            )

        max_views = max(int(value.shape[0]) for value in images)
        padded_images = images[0].new_zeros(
            (len(images), max_views, *image_shape)
        )
        view_is_pad = torch.ones(
            (len(images), max_views), dtype=torch.bool
        )
        for batch_idx, value in enumerate(images):
            num_views = int(value.shape[0])
            padded_images[batch_idx, :num_views].copy_(value)
            view_is_pad[batch_idx, :num_views] = False

        collated = default_collate(
            [
                {
                    key: value
                    for key, value in sample.items()
                    if key != image_key
                }
                for sample in batch
            ]
        )
        collated[image_key] = padded_images
        collated["vlm_current_view_is_pad"] = view_is_pad
        return collated

    def __len__(self) -> int:
        return len(self.mixed_dataset)

    def set_epoch(self, epoch: int) -> None:
        self.mixed_dataset.set_epoch(int(epoch))
