"""Fixed t-binned validation loss (read-only eval side-car).

Measures the GT-anchored FM regression loss on a frozen evaluation set:
frozen clip subset (seeded random pick), frozen training-style windows
(fix_timestep), frozen noise (generator reset per eval), fixed timestep grid.
The result is a deterministic function of the weights, so curves are
comparable across checkpoints, runs, and configs — unlike the training loss,
whose (clip, eps, t) draws and t-distribution move with the config.

Design notes:
- Windows use the TRAINING sampler semantics (plain int indexing ->
  get_batch_new -> cal_sample_index/mix_captions), NOT the sequential-segment
  validation protocol (get_data_by_step). The render suite's segments are
  continuous-only (no end sparse frames) and distributionally unlike training
  windows; this eval deliberately measures the training objective on
  training-shaped windows instead.
- Eval dataset config = data.train wholesale, with only the dataset-identity
  keys (jsonl/video folders/source lists...) swapped in from data.val — so at
  pretrain scale, pointing data.val at held-out episodes makes this a true
  validation loss while the window semantics stay training-identical.
  `sources: [val, train]` adds a train-episode set; the gap between the two
  curves is the generalization gap.
- Main-rank only: the EMA validation pass swaps weights on rank 0 only (see
  runner/ema.py), so a multi-rank eval would mix EMA and live weights.
- The h=0/t=r grid logs the v head only. u-head endpoint diagnostics come
  from interval_grid, which queries u(t,r) and measures x_t -> x_r error.
- Optional interval metrics query u(t,r) for fixed [r,t] pairs and measure the
  one-step endpoint error x_t -> x_r without rendering videos.
- v1 limits (fail fast): dense non-mixed, non-frame-concat path only; int
  indices carry no goal tag -> text-conditioned rows only.
"""

from __future__ import annotations

import logging
import math
import os
import random
from typing import Dict, List, Optional, Tuple

import torch
from einops import rearrange
from torch.utils.data import default_collate

from go3_nano_preview.sft.vlm import compute_vlm_hidden_states
from policy_core.settings import settings
from policy_core.utils.WAM_policy.WAM_policy_config import (
    as_dataset_kwargs,
    get_action_model_kwargs,
)
from policy_core.utils.WAM_policy.data_utils import gen_noise_from_condition_frame_latent
from policy_core.utils.common_utils import get_device
from policy_core.utils.WAM_policy.model_utils import forward_pass

logger = logging.getLogger(__name__)

# Per-source list keys that identify WHICH data is read; swapped in from
# data.val so the eval set can be held-out while every sampling-semantics key
# (mix_captions, step_goal_prob, chunk layout, ...) stays training-identical.
_DATASET_IDENTITY_KEYS = (
    "dataset_type",
    "jsonl_path_list",
    "video_folder_list",
    "dataset_source_list",
    "dataset_name_list",
    "cam_use",
    "fps_list",
    "repeat",
    "action_space_list",
    "stat_file_list",
)

# Identity keys with safe dataset defaults (repeat -> 1, fps_list -> 30.0).
# When absent from data.val, train's per-source list cannot be reused (its
# length follows train's source count), so drop it and let the default apply
# — the same behavior the regular val dataloader gets.
_VAL_DEFAULTABLE_KEYS = ("repeat", "fps_list")

# Optional per-source settings and snapshot metadata belong to the selected
# dataset. Use data.val's values or dataset defaults, never data.train's.
_VAL_NONINHERITABLE_KEYS = (
    "action_space_list",
    "stat_file_list",
    "metadata_index_manifest",
    "source_weights",
)

_DEFAULT_T_GRID = (0.1, 0.25, 0.4, 0.55, 0.7, 0.8, 0.9, 0.97)


class FixedValLoss:
    """Coordinates the frozen eval sets and the per-t-bin loss evaluation."""

    def __init__(self, cfg: dict) -> None:
        self.num_samples = int(cfg.get("num_samples", 64))
        self.batch_size = int(cfg.get("batch_size", 8))
        self.seed = int(cfg.get("seed", 1234))
        self.t_grid = [float(t) for t in cfg.get("t_grid", _DEFAULT_T_GRID)]
        self.sources = list(cfg.get("sources", ("val",)))
        diagnostics_cfg = dict(cfg.get("diagnostics", {}) or {})
        raw_interval_grid = cfg.get("interval_grid", diagnostics_cfg.get("interval_grid", ()))
        if diagnostics_cfg.get("one_step_x0_loss", False) and not raw_interval_grid:
            raw_interval_grid = ((0.0, 1.0),)
        self.interval_grid = self._parse_interval_grid(raw_interval_grid)
        sampling_cfg = dict(cfg.get("sampling", {}) or {})
        self.sampling_mode = str(sampling_cfg.get("mode", "random")).lower()
        self.group_by = str(sampling_cfg.get("group_by", "dataset_name"))
        self.episodes_per_group = sampling_cfg.get("episodes_per_group")
        if self.episodes_per_group is not None:
            self.episodes_per_group = int(self.episodes_per_group)
        self.log_per_group = bool(sampling_cfg.get("log_per_group", False))
        self.log_per_view = bool(diagnostics_cfg.get("per_view_loss", False))
        self.log_per_frame = bool(diagnostics_cfg.get("per_frame_loss", False))
        self.log_per_view_frame = bool(
            diagnostics_cfg.get(
                "view_frame_loss",
                diagnostics_cfg.get("per_view_frame_loss", False),
            )
        )
        t_breakdown_axes = diagnostics_cfg.get("t_breakdown_axes", ("global",))
        if isinstance(t_breakdown_axes, str):
            t_breakdown_axes = [t_breakdown_axes]
        self.t_breakdown_axes = {str(axis) for axis in t_breakdown_axes}
        max_frame_metrics = diagnostics_cfg.get("max_frames")
        self.max_frame_metrics = None if max_frame_metrics is None else int(max_frame_metrics)
        self.log_interval_xr = bool(
            diagnostics_cfg.get(
                "interval_xr_loss",
                diagnostics_cfg.get("interval_xr_mse", False),
            )
        )
        self.log_interval_velocity = bool(
            diagnostics_cfg.get(
                "interval_velocity_loss",
                diagnostics_cfg.get("interval_velocity_mse", False),
            )
        )
        if self.num_samples <= 0 or self.batch_size <= 0:
            raise ValueError("fixed_val_loss: num_samples and batch_size must be > 0")
        if self.sampling_mode not in {"random", "stratified"}:
            raise ValueError(
                f"fixed_val_loss: sampling.mode must be 'random' or 'stratified', "
                f"got {self.sampling_mode!r}"
        )
        if self.episodes_per_group is not None and self.episodes_per_group <= 0:
            raise ValueError("fixed_val_loss: sampling.episodes_per_group must be > 0")
        if self.max_frame_metrics is not None and self.max_frame_metrics <= 0:
            raise ValueError("fixed_val_loss: diagnostics.max_frames must be > 0")
        valid_axes = {"global", "group", "view", "frame", "view_frame"}
        unknown_axes = self.t_breakdown_axes - valid_axes
        if unknown_axes:
            raise ValueError(
                "fixed_val_loss: diagnostics.t_breakdown_axes entries must be one of "
                f"{sorted(valid_axes)}, got {sorted(unknown_axes)}"
            )
        for t in self.t_grid:
            if not 0.0 < t <= 1.0:
                raise ValueError(f"fixed_val_loss: t_grid entries must be in (0, 1], got {t}")
        for source in self.sources:
            if source not in ("val", "train"):
                raise ValueError(
                    f"fixed_val_loss: sources entries must be 'val' or 'train', got {source!r}"
                )
        # source -> (dataset, frozen index list, group list, data_cfg). Built lazily on the
        # first evaluate() call so dataset construction cost is paid only when
        # (and where) the eval actually runs.
        self._eval_sets: Optional[Dict[str, Tuple[object, List[int], List[str], dict]]] = None

    def _parse_interval_grid(self, raw_grid) -> List[Tuple[float, float]]:
        intervals: List[Tuple[float, float]] = []
        if raw_grid is None:
            return intervals
        for entry in raw_grid:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise ValueError(
                    "fixed_val_loss.interval_grid entries must be [r, t] pairs, "
                    f"got {entry!r}"
                )
            r, t = float(entry[0]), float(entry[1])
            if not 0.0 <= r < t <= 1.0:
                raise ValueError(
                    "fixed_val_loss.interval_grid entries must satisfy "
                    f"0 <= r < t <= 1, got [r={r}, t={t}]"
                )
            intervals.append((r, t))
        return intervals

    @classmethod
    def from_config(cls, cfg) -> "Optional[FixedValLoss]":
        """Build from cfg.validation.fixed_val_loss, or None when disabled (no-op)."""
        fixed_val_cfg = cfg.validation.fixed_val_loss
        if not fixed_val_cfg.get("enabled", False):
            return None
        return cls(fixed_val_cfg)

    def _build_eval_dataset_cfg(self, cfg, source: str) -> dict:
        """data.train wholesale + data.val's dataset-identity keys (source='val')."""
        data_cfg = as_dataset_kwargs(cfg.data.train)
        if source == "val":
            val_cfg = as_dataset_kwargs(cfg.data.val)
            for key in _DATASET_IDENTITY_KEYS:
                if key in val_cfg:
                    data_cfg[key] = val_cfg[key]
                elif key in _VAL_DEFAULTABLE_KEYS:
                    data_cfg.pop(key, None)
            for key in _VAL_NONINHERITABLE_KEYS:
                if key in val_cfg:
                    data_cfg[key] = val_cfg[key]
                else:
                    data_cfg.pop(key, None)
            # return_action follows data.val only; train's flag must not apply
            # to a held-out video eval set.
            data_cfg["return_action"] = bool(val_cfg.get("return_action", False))
            # Any remaining per-source list inherited from train must align
            # with val's source count; fail with the fix instead of a deep
            # dataset traceback.
            num_sources = len(data_cfg["jsonl_path_list"])
            for key in _DATASET_IDENTITY_KEYS:
                value = data_cfg.get(key)
                if isinstance(value, (list, tuple)) and len(value) not in (1, num_sources):
                    raise ValueError(
                        f"fixed_val_loss: '{key}' has {len(value)} entries but data.val "
                        f"lists {num_sources} jsonl sources — add a matching '{key}' to "
                        f"data.val (it currently falls through from data.train)"
                    )
        data_cfg.pop("batch_size", None)
        data_cfg["gpuid"] = settings.local_rank
        data_cfg["target_dim"] = get_action_model_kwargs(cfg)["action_in_channels"]
        # Freeze windows/captions: both sampler branches seed their rng by
        # index under fix_timestep, and recaption is auto-disabled.
        data_cfg["fix_timestep"] = True
        return data_cfg

    def _sanitize_metric_component(self, value: object) -> str:
        text = str(value)
        cleaned = []
        for ch in text:
            if ch.isalnum() or ch in ("_", "-", "."):
                cleaned.append(ch)
            else:
                cleaned.append("_")
        name = "".join(cleaned).strip("_")
        return name or "unknown"

    def _time_label(self, value: float) -> str:
        return self._sanitize_metric_component(f"{value:g}")

    def _interval_label(self, r: float, t: float) -> str:
        return f"{self._time_label(r)}_{self._time_label(t)}"

    def _dataset_base_length(self, dataset) -> int:
        return int(getattr(dataset, "true_length", len(dataset)))

    def _sample_group(self, dataset, index: int) -> str:
        base_len = self._dataset_base_length(dataset)
        base_index = int(index) % max(base_len, 1)

        if self.group_by == "dataset_name":
            values = getattr(dataset, "dataset_name_list", None)
            if values is not None and base_index < len(values):
                return self._sanitize_metric_component(values[base_index])
        elif self.group_by == "embodiment":
            for attr in ("embodiment_list", "embodiments"):
                values = getattr(dataset, attr, None)
                if values is not None and base_index < len(values):
                    return self._sanitize_metric_component(values[base_index])
            records = getattr(dataset, "dataset", None)
            if records is not None and base_index < len(records):
                record = records[base_index]
                if isinstance(record, dict) and "embodiment" in record:
                    return self._sanitize_metric_component(record["embodiment"])
            values = getattr(dataset, "dataset_name_list", None)
            if values is not None and base_index < len(values):
                return self._sanitize_metric_component(values[base_index])
        elif self.group_by == "dataset_source":
            values = getattr(dataset, "dataset_source_list", None)
            if values is not None and base_index < len(values):
                return self._sanitize_metric_component(values[base_index])
        elif self.group_by == "source_index":
            values = getattr(dataset, "sample_source_indices", None)
            if values is not None and base_index < len(values):
                return self._sanitize_metric_component(values[base_index])
        elif self.group_by == "dataset_source_name":
            sources = getattr(dataset, "dataset_source_list", None)
            names = getattr(dataset, "dataset_name_list", None)
            if (
                sources is not None and names is not None
                and base_index < len(sources) and base_index < len(names)
            ):
                return self._sanitize_metric_component(f"{sources[base_index]}_{names[base_index]}")
        else:
            values = getattr(dataset, self.group_by, None)
            if values is not None and base_index < len(values):
                return self._sanitize_metric_component(values[base_index])
            records = getattr(dataset, "dataset", None)
            if records is not None and base_index < len(records):
                record = records[base_index]
                if isinstance(record, dict) and self.group_by in record:
                    return self._sanitize_metric_component(record[self.group_by])
            raise ValueError(
                f"fixed_val_loss: cannot group by {self.group_by!r}; add a dataset "
                "metadata list with that name, a jsonl field with that name, or use "
                "dataset_name/dataset_source/source_index/dataset_source_name"
            )

        return "unknown"

    def _build_random_indices(self, dataset) -> Tuple[List[int], List[str]]:
        n = min(self.num_samples, len(dataset))
        indices = random.Random(self.seed).sample(range(len(dataset)), n)
        groups = [self._sample_group(dataset, i) for i in indices]
        return indices, groups

    def _build_stratified_indices(self, dataset) -> Tuple[List[int], List[str]]:
        base_len = self._dataset_base_length(dataset)
        by_group: Dict[str, List[int]] = {}
        for i in range(base_len):
            by_group.setdefault(self._sample_group(dataset, i), []).append(i)

        if self.episodes_per_group is None:
            per_group = max(1, math.ceil(self.num_samples / max(len(by_group), 1)))
        else:
            per_group = int(self.episodes_per_group)

        rng = random.Random(self.seed)
        selected_indices: List[int] = []
        selected_groups: List[str] = []
        for group in sorted(by_group):
            indices = by_group[group]
            n = min(per_group, len(indices))
            selected = rng.sample(indices, n)
            for index in selected:
                selected_indices.append(index)
                selected_groups.append(group)

        order = list(range(len(selected_indices)))
        rng.shuffle(order)
        return [selected_indices[i] for i in order], [selected_groups[i] for i in order]

    def _build_indices(self, dataset) -> Tuple[List[int], List[str]]:
        if self.sampling_mode == "stratified":
            return self._build_stratified_indices(dataset)
        return self._build_random_indices(dataset)

    def _batch_view_labels(self, batch: dict, batch_size: int, n_view: int) -> List[List[str]]:
        view_ids = batch.get("view_ids")
        if view_ids is None:
            return [[f"view_pos_{v}" for v in range(n_view)] for _ in range(batch_size)]

        if torch.is_tensor(view_ids):
            ids = view_ids.detach().cpu().tolist()
        else:
            ids = view_ids

        if ids and not isinstance(ids[0], (list, tuple)):
            ids = [ids for _ in range(batch_size)]

        labels: List[List[str]] = []
        for item_i in range(batch_size):
            row = ids[item_i] if item_i < len(ids) else list(range(n_view))
            labels.append([
                f"view_id_{self._sanitize_metric_component(row[v])}"
                if v < len(row) else f"view_pos_{v}"
                for v in range(n_view)
            ])
        return labels

    def _log_axis_t_breakdown(self, axis: str) -> bool:
        return axis in self.t_breakdown_axes

    def _metric_name(self, prefix: str, axis: str, *parts: object) -> str:
        """Put diagnostic axes in the TB top-level group for readability."""
        if axis == "global":
            return "/".join([prefix, *(str(part) for part in parts)])
        return "/".join([f"{prefix}_{axis}", *(str(part) for part in parts)])

    def _ordered_metric_ts(self, keyed_sums: dict, head: str) -> List[float]:
        present = {t for key in keyed_sums for t in key[-1:] if key[-2] == head}
        ordered = [t for t in self.t_grid if t in present]
        ordered.extend(sorted(present - set(ordered)))
        return ordered

    def _build_eval_sets(self, inferrer) -> Dict[str, Tuple[object, List[int], List[str], dict]]:
        cfg = inferrer.cfg
        eval_sets: Dict[str, Tuple[object, List[int], List[str], dict]] = {}
        for source in self.sources:
            data_cfg = self._build_eval_dataset_cfg(cfg, source)
            # Match the regular train/validation builders so specialized
            # dataset contracts (for example BBoxConditionDataset) are kept
            # during fixed-loss evaluation. Legacy configs without an
            # explicit dataset_type continue to use MixMultiFrame.
            from go3_nano_preview.sft.dataset import (
                resolve_WAM_policy_dataset_class,
            )

            dataset_type = data_cfg.pop("dataset_type", "mix_multi_frame")
            dataset_class = resolve_WAM_policy_dataset_class(dataset_type)
            dataset = dataset_class(**data_cfg)
            if len(dataset) == 0:
                raise ValueError(f"fixed_val_loss: data.{source} dataset is empty")
            indices, groups = self._build_indices(dataset)
            if not indices:
                raise ValueError(f"fixed_val_loss: data.{source} produced no eval samples")
            eval_sets[source] = (dataset, indices, groups, data_cfg)
            group_counts: Dict[str, int] = {}
            for group in groups:
                group_counts[group] = group_counts.get(group, 0) + 1
            logger.info(
                "fixed_val_loss[%s]: %d samples (dataset len %d), mode=%s, "
                "group_by=%s, batch_size=%d, t_grid=%s, groups=%s",
                source, len(indices), len(dataset), self.sampling_mode,
                self.group_by, self.batch_size, self.t_grid, group_counts,
            )
        return eval_sets

    @torch.no_grad()
    def evaluate(
        self, inferrer, global_step: int, tag: str = "",
        video_frame_indices: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """Compute per-t-bin losses for every source; log to TB. Main rank only.

        `tag` distinguishes weight variants ("" = live, "_ema" = inside the
        EMA swap context). `video_frame_indices` is the Inferrer's selector,
        passed in by the call site. Returns {scalar_name: value}.
        """
        if video_frame_indices is None:
            raise ValueError(
                "fixed_val_loss.evaluate requires video_frame_indices "
                "(pass inferrer.video_frame_indices)"
            )
        if bool(getattr(inferrer, "use_video_mixed_path", False)):
            raise NotImplementedError("fixed_val_loss supports the dense non-mixed path only")
        if self._eval_sets is None:
            self._eval_sets = self._build_eval_sets(inferrer)
        VIDEO_FRAME_INDICES = list(video_frame_indices)

        device = getattr(inferrer, "device", None) or get_device()
        weight_dtype = getattr(inferrer, "dtype", None)
        if weight_dtype is None:
            mixed = getattr(getattr(inferrer.cfg, "train", None), "mixed_precision", "bf16")
            weight_dtype = torch.bfloat16 if mixed == "bf16" else torch.float32

        model = (inferrer.video_diffusion_model.module if hasattr(inferrer.video_diffusion_model, "module") else inferrer.video_diffusion_model)
        use_t_and_r = bool(getattr(model.config, "use_t_and_r_embedding", False))
        pixel_wise = bool(inferrer.cfg.cotrain.pixel_wise_timestep)
        timestep_scale = float(getattr(inferrer, "timestep_scale", 1.0))

        # NOTE: the Inferrer call site (_run_fixed_val_loss) wraps this call in
        # _validation_eager_model_context, swapping in the EAGER VAE encode and
        # the EAGER VGen forward — eval shapes and kwargs differ from training's
        # and must not add compiled static graphs. Done there rather than here
        # so the restore lives in an exception-safe finally. Consequently the
        # forward_pass calls below reach the module's eager forward even when
        # vgen_compile_forward=true.

        metrics: Dict[str, float] = {}
        for source, (dataset, indices, groups, data_cfg) in self._eval_sets.items():
            src_tag = "" if source == "val" else f"_{source}"
            prefix = f"val_fixed{src_tag}{tag}"
            mem_size = int(data_cfg["n_previous"])
            sums: Dict[Tuple[str, float], float] = {}
            denoms: Dict[Tuple[str, float], float] = {}
            group_sums: Dict[Tuple[str, str, float], float] = {}
            group_denoms: Dict[Tuple[str, str, float], float] = {}
            group_counts: Dict[str, int] = {}
            view_sums: Dict[Tuple[str, str, float], float] = {}
            view_denoms: Dict[Tuple[str, str, float], float] = {}
            frame_sums: Dict[Tuple[int, str, float], float] = {}
            frame_denoms: Dict[Tuple[int, str, float], float] = {}
            view_frame_sums: Dict[Tuple[str, int, str, float], float] = {}
            view_frame_denoms: Dict[Tuple[str, int, str, float], float] = {}
            interval_sums: Dict[str, float] = {}
            interval_denoms: Dict[str, float] = {}
            interval_group_sums: Dict[Tuple[str, str], float] = {}
            interval_group_denoms: Dict[Tuple[str, str], float] = {}
            interval_view_sums: Dict[Tuple[str, str], float] = {}
            interval_view_denoms: Dict[Tuple[str, str], float] = {}
            interval_frame_sums: Dict[Tuple[int, str], float] = {}
            interval_frame_denoms: Dict[Tuple[int, str], float] = {}
            interval_view_frame_sums: Dict[Tuple[str, int, str], float] = {}
            interval_view_frame_denoms: Dict[Tuple[str, int, str], float] = {}
            # Reset per eval: identical eps every evaluation. Keep the
            # generator on the eval device to avoid CPU->CUDA randn fallback.
            gen = torch.Generator(device=device)
            gen.manual_seed(self.seed)

            for start in range(0, len(indices), self.batch_size):
                chunk = indices[start:start + self.batch_size]
                batch_groups = groups[start:start + self.batch_size]
                batch = default_collate([dataset[i] for i in chunk])
                for group in batch_groups:
                    group_counts[group] = group_counts.get(group, 0) + 1
                video = batch["video"].to(device, dtype=weight_dtype)[:, :, :, VIDEO_FRAME_INDICES]
                b, _, n_view, _, h, w = video.shape
                view_labels = (
                    self._batch_view_labels(batch, b, n_view)
                    if self.log_per_view or self.log_per_view_frame else None
                )
                video_reshaped = rearrange(video, "b c v t h w -> (b v) c t h w")
                mem = video_reshaped[:, :, :mem_size]
                future_video = video_reshaped[:, :, mem_size:]
                raw_frames = future_video.shape[2]
                latent_height = h // inferrer.SPATIAL_DOWN_RATIO
                latent_width = w // inferrer.SPATIAL_DOWN_RATIO
                n_dense_frames = min(int(data_cfg["continious_vid_length"]), raw_frames)

                vlat = inferrer._vae_encode_mem_future(
                    mem, future_video, b, mem_size,
                    latent_height, latent_width, n_dense_frames,
                )
                mem_latents = vlat["mem_latents"]
                future_latents = vlat["future_video_latents"]
                if inferrer.vae.vae_encoding_mode == "3dvae":
                    latent_frames = raw_frames // inferrer.TEMPORAL_DOWN_RATIO + 1 + mem_size
                else:
                    latent_frames = raw_frames // inferrer.TEMPORAL_DOWN_RATIO + mem_size
                latents_flat = rearrange(
                    torch.cat((mem_latents, future_latents), dim=2),
                    "bv c f h w -> bv (f h w) c",
                )
                # noise_to_condition_frames=0.0: deterministic (skips the
                # module-RNG scale draw) AND inference-like (clean mem frames).
                noise, conditioning_mask, cond_indicator = gen_noise_from_condition_frame_latent(
                    mem_latents, latent_frames, latent_height, latent_width,
                    generator=gen, noise_to_condition_frames=0.0,
                )
                captions = list(batch["caption"])
                _, vlm_states, vlm_mask = compute_vlm_hidden_states(
                    inferrer.video_diffusion_model,
                    video,
                    captions,
                    use_vlm=bool(inferrer.cfg.model.use_vlm),
                    vlm_text_only=inferrer.vlm_text_only,
                    batch_size=b,
                    n_view=n_view,
                    mem_size=mem_size,
                    with_grad=False,
                    bboxes=batch.get("bbox"),
                    bbox_reference_images=batch.get("bbox_reference_head"),
                )

                keep_mask = conditioning_mask if pixel_wise else cond_indicator
                future_tok_start = mem_size * latent_height * latent_width
                cond_weight = (1.0 - conditioning_mask.float()).unsqueeze(-1)
                target = (noise - latents_flat).float()
                supervised_elems = float(
                    cond_weight[:, future_tok_start:, :].sum().item() * latents_flat.shape[-1]
                )
                bv = b * n_view

                def accumulate_interval_metric(metric_name: str, metric_err: torch.Tensor) -> None:
                    interval_sums[metric_name] = (
                        interval_sums.get(metric_name, 0.0)
                        + metric_err[:, future_tok_start:, :].sum().item()
                    )
                    interval_denoms[metric_name] = (
                        interval_denoms.get(metric_name, 0.0) + supervised_elems
                    )
                    if self.log_per_group:
                        per_item_err = metric_err[:, future_tok_start:, :].sum(dim=(1, 2))
                        per_item_denom = (
                            cond_weight[:, future_tok_start:, :].sum(dim=(1, 2))
                            * latents_flat.shape[-1]
                        )
                        per_item_err = per_item_err.view(b, n_view).sum(dim=1)
                        per_item_denom = per_item_denom.view(b, n_view).sum(dim=1)
                        for item_i, group in enumerate(batch_groups):
                            gkey = (group, metric_name)
                            interval_group_sums[gkey] = (
                                interval_group_sums.get(gkey, 0.0)
                                + float(per_item_err[item_i].item())
                            )
                            interval_group_denoms[gkey] = (
                                interval_group_denoms.get(gkey, 0.0)
                                + float(per_item_denom[item_i].item())
                            )
                    if self.log_per_view or self.log_per_frame or self.log_per_view_frame:
                        future_err = metric_err[:, future_tok_start:, :]
                        future_weight = cond_weight[:, future_tok_start:, :]
                        n_future_frames = future_err.shape[1] // (latent_height * latent_width)
                        usable_tokens = n_future_frames * latent_height * latent_width
                        if usable_tokens <= 0:
                            return
                        future_err = future_err[:, :usable_tokens, :]
                        future_weight = future_weight[:, :usable_tokens, :]
                        err_by_frame = rearrange(
                            future_err,
                            "(b v) (f hw) c -> b v f hw c",
                            b=b,
                            v=n_view,
                            f=n_future_frames,
                            hw=latent_height * latent_width,
                        )
                        weight_by_frame = rearrange(
                            future_weight,
                            "(b v) (f hw) c -> b v f hw c",
                            b=b,
                            v=n_view,
                            f=n_future_frames,
                            hw=latent_height * latent_width,
                        )
                        if self.log_per_view:
                            per_view_err = err_by_frame.sum(dim=(2, 3, 4))
                            per_view_denom = (
                                weight_by_frame.sum(dim=(2, 3, 4)) * latents_flat.shape[-1]
                            )
                            for item_i in range(b):
                                for view_i, view_label in enumerate(view_labels[item_i]):
                                    vkey = (view_label, metric_name)
                                    interval_view_sums[vkey] = (
                                        interval_view_sums.get(vkey, 0.0)
                                        + float(per_view_err[item_i, view_i].item())
                                    )
                                    interval_view_denoms[vkey] = (
                                        interval_view_denoms.get(vkey, 0.0)
                                        + float(per_view_denom[item_i, view_i].item())
                                    )
                        if self.log_per_frame:
                            frame_limit = n_future_frames
                            if self.max_frame_metrics is not None:
                                frame_limit = min(frame_limit, self.max_frame_metrics)
                            per_frame_err = err_by_frame[:, :, :frame_limit].sum(dim=(0, 1, 3, 4))
                            per_frame_denom = (
                                weight_by_frame[:, :, :frame_limit].sum(dim=(0, 1, 3, 4))
                                * latents_flat.shape[-1]
                            )
                            for frame_i in range(frame_limit):
                                fkey = (frame_i, metric_name)
                                interval_frame_sums[fkey] = (
                                    interval_frame_sums.get(fkey, 0.0)
                                    + float(per_frame_err[frame_i].item())
                                )
                                interval_frame_denoms[fkey] = (
                                    interval_frame_denoms.get(fkey, 0.0)
                                    + float(per_frame_denom[frame_i].item())
                                )
                        if self.log_per_view_frame:
                            frame_limit = n_future_frames
                            if self.max_frame_metrics is not None:
                                frame_limit = min(frame_limit, self.max_frame_metrics)
                            per_view_frame_err = err_by_frame[:, :, :frame_limit].sum(dim=(3, 4))
                            per_view_frame_denom = (
                                weight_by_frame[:, :, :frame_limit].sum(dim=(3, 4))
                                * latents_flat.shape[-1]
                            )
                            for item_i in range(b):
                                for view_i, view_label in enumerate(view_labels[item_i]):
                                    for frame_i in range(frame_limit):
                                        vfkey = (view_label, frame_i, metric_name)
                                        interval_view_frame_sums[vfkey] = (
                                            interval_view_frame_sums.get(vfkey, 0.0)
                                            + float(per_view_frame_err[item_i, view_i, frame_i].item())
                                        )
                                        interval_view_frame_denoms[vfkey] = (
                                            interval_view_frame_denoms.get(vfkey, 0.0)
                                            + float(per_view_frame_denom[item_i, view_i, frame_i].item())
                                        )

                for t in self.t_grid:
                    noisy = (1.0 - t) * latents_flat + t * noise
                    ts = (
                        torch.full((bv, 1), t, device=device, dtype=weight_dtype)
                        * (1.0 - keep_mask)
                        * timestep_scale
                    )
                    # h = 0 forward (r = t): mirrors the trainer's
                    # flow_forward_h0, where u(t, t) equals the instantaneous
                    # velocity, so both heads share the GT target below.
                    extra_ts = {"timestep_r": ts} if use_t_and_r else {}
                    out = forward_pass(
                        model=model,
                        timesteps=ts,
                        noisy_latents=noisy,
                        prompt_embeds=None,
                        prompt_attention_mask=None,
                        num_frames=latent_frames,
                        height=latent_height,
                        width=latent_width,
                        n_view=n_view,
                        view_ids=batch["view_ids"].to(device),
                        return_video=True,
                        return_action=False,
                        video_attention_mask=None,
                        history_action_state=None,
                        condition_mask=conditioning_mask,
                        vlm_inputs=None,
                        num_frames_wo_dropout=mem_size,
                        precomputed_vlm_hidden_states=vlm_states,
                        vlm_attention_mask=vlm_mask,
                        compute_v_head=True,
                        **extra_ts,
                    )["latents"]
                    preds = {}
                    if isinstance(out, dict) and out.get("v_video") is not None:
                        preds["v"] = out["v_video"]
                    for head, pred in preds.items():
                        err = (pred.float() - target).pow(2) * cond_weight
                        for metric_head, metric_err in [(head, err)]:
                            key = (metric_head, t)
                            sums[key] = (
                                sums.get(key, 0.0)
                                + metric_err[:, future_tok_start:, :].sum().item()
                            )
                            denoms[key] = denoms.get(key, 0.0) + supervised_elems
                            if self.log_per_group:
                                per_item_err = metric_err[:, future_tok_start:, :].sum(dim=(1, 2))
                                per_item_denom = (
                                    cond_weight[:, future_tok_start:, :].sum(dim=(1, 2))
                                    * latents_flat.shape[-1]
                                )
                                per_item_err = per_item_err.view(b, n_view).sum(dim=1)
                                per_item_denom = per_item_denom.view(b, n_view).sum(dim=1)
                                for item_i, group in enumerate(batch_groups):
                                    gkey = (group, metric_head, t)
                                    group_sums[gkey] = (
                                        group_sums.get(gkey, 0.0)
                                        + float(per_item_err[item_i].item())
                                    )
                                    group_denoms[gkey] = (
                                        group_denoms.get(gkey, 0.0)
                                        + float(per_item_denom[item_i].item())
                                    )
                            if self.log_per_view or self.log_per_frame or self.log_per_view_frame:
                                future_err = metric_err[:, future_tok_start:, :]
                                future_weight = cond_weight[:, future_tok_start:, :]
                                n_future_frames = future_err.shape[1] // (latent_height * latent_width)
                                usable_tokens = n_future_frames * latent_height * latent_width
                                future_err = future_err[:, :usable_tokens, :]
                                future_weight = future_weight[:, :usable_tokens, :]
                                err_by_frame = rearrange(
                                    future_err,
                                    "(b v) (f hw) c -> b v f hw c",
                                    b=b,
                                    v=n_view,
                                    f=n_future_frames,
                                    hw=latent_height * latent_width,
                                )
                                weight_by_frame = rearrange(
                                    future_weight,
                                    "(b v) (f hw) c -> b v f hw c",
                                    b=b,
                                    v=n_view,
                                    f=n_future_frames,
                                    hw=latent_height * latent_width,
                                )
                                if self.log_per_view:
                                    per_view_err = err_by_frame.sum(dim=(2, 3, 4))
                                    per_view_denom = weight_by_frame.sum(dim=(2, 3, 4)) * latents_flat.shape[-1]
                                    for item_i in range(b):
                                        for view_i, view_label in enumerate(view_labels[item_i]):
                                            vkey = (view_label, metric_head, t)
                                            view_sums[vkey] = (
                                                view_sums.get(vkey, 0.0)
                                                + float(per_view_err[item_i, view_i].item())
                                            )
                                            view_denoms[vkey] = (
                                                view_denoms.get(vkey, 0.0)
                                                + float(per_view_denom[item_i, view_i].item())
                                            )
                                if self.log_per_frame:
                                    frame_limit = n_future_frames
                                    if self.max_frame_metrics is not None:
                                        frame_limit = min(frame_limit, self.max_frame_metrics)
                                    per_frame_err = err_by_frame[:, :, :frame_limit].sum(dim=(0, 1, 3, 4))
                                    per_frame_denom = (
                                        weight_by_frame[:, :, :frame_limit].sum(dim=(0, 1, 3, 4))
                                        * latents_flat.shape[-1]
                                    )
                                    for frame_i in range(frame_limit):
                                        fkey = (frame_i, metric_head, t)
                                        frame_sums[fkey] = frame_sums.get(fkey, 0.0) + float(per_frame_err[frame_i].item())
                                        frame_denoms[fkey] = frame_denoms.get(fkey, 0.0) + float(per_frame_denom[frame_i].item())
                                if self.log_per_view_frame:
                                    frame_limit = n_future_frames
                                    if self.max_frame_metrics is not None:
                                        frame_limit = min(frame_limit, self.max_frame_metrics)
                                    per_view_frame_err = err_by_frame[:, :, :frame_limit].sum(dim=(3, 4))
                                    per_view_frame_denom = (
                                        weight_by_frame[:, :, :frame_limit].sum(dim=(3, 4))
                                        * latents_flat.shape[-1]
                                    )
                                    for item_i in range(b):
                                        for view_i, view_label in enumerate(view_labels[item_i]):
                                            for frame_i in range(frame_limit):
                                                vfkey = (view_label, frame_i, metric_head, t)
                                                view_frame_sums[vfkey] = (
                                                    view_frame_sums.get(vfkey, 0.0)
                                                    + float(per_view_frame_err[item_i, view_i, frame_i].item())
                                                )
                                                view_frame_denoms[vfkey] = (
                                                    view_frame_denoms.get(vfkey, 0.0)
                                                    + float(per_view_frame_denom[item_i, view_i, frame_i].item())
                                                )

                if self.interval_grid and (self.log_interval_xr or self.log_interval_velocity):
                    if not use_t_and_r and not getattr(self, "_interval_legacy_warned", False):
                        logger.warning(
                            "fixed_val_loss.interval_grid with use_t_and_r_embedding=False "
                            "uses only h=t-r; absolute r/t endpoints are not separately embedded."
                        )
                        self._interval_legacy_warned = True
                    for r, t in self.interval_grid:
                        delta = t - r
                        noisy = (1.0 - t) * latents_flat + t * noise
                        if use_t_and_r:
                            ts = (
                                torch.full((bv, 1), t, device=device, dtype=weight_dtype)
                                * (1.0 - keep_mask)
                                * timestep_scale
                            )
                            ts_r = (
                                torch.full((bv, 1), r, device=device, dtype=weight_dtype)
                                * (1.0 - keep_mask)
                                * timestep_scale
                            )
                            extra_ts = {"timestep_r": ts_r}
                        else:
                            ts = (
                                torch.full((bv, 1), delta, device=device, dtype=weight_dtype)
                                * (1.0 - keep_mask)
                                * timestep_scale
                            )
                            extra_ts = {}
                        out = forward_pass(
                            model=model,
                            timesteps=ts,
                            noisy_latents=noisy,
                            prompt_embeds=None,
                            prompt_attention_mask=None,
                            num_frames=latent_frames,
                            height=latent_height,
                            width=latent_width,
                            n_view=n_view,
                            view_ids=batch["view_ids"].to(device),
                            return_video=True,
                            return_action=False,
                            video_attention_mask=None,
                            history_action_state=None,
                            condition_mask=conditioning_mask,
                            vlm_inputs=None,
                            num_frames_wo_dropout=mem_size,
                            precomputed_vlm_hidden_states=vlm_states,
                            vlm_attention_mask=vlm_mask,
                            compute_v_head=False,
                            **extra_ts,
                        )["latents"]
                        x_r_gt = (1.0 - r) * latents_flat + r * noise
                        x_r_hat = noisy.float() - delta * out["video"].float()
                        xr_err = (x_r_hat - x_r_gt.float()).pow(2) * cond_weight
                        interval_label = self._interval_label(r, t)
                        if self.log_interval_xr:
                            accumulate_interval_metric(
                                f"u_{interval_label}_x",
                                xr_err,
                            )
                        if self.log_interval_velocity:
                            accumulate_interval_metric(
                                f"u_{interval_label}_v",
                                xr_err / (delta * delta),
                            )

            heads = sorted({head for head, _ in sums})
            for head in heads:
                bins = []
                for t in self._ordered_metric_ts(sums, head):
                    val = sums[(head, t)] / max(denoms[(head, t)], 1.0)
                    metrics[self._metric_name(prefix, "global", f"{head}_t{t:g}")] = val
                    bins.append(val)
                if bins:
                    metrics[self._metric_name(prefix, "global", f"{head}_mean")] = (
                        sum(bins) / len(bins)
                    )
                if self.log_per_group:
                    for group in sorted(group_counts):
                        group_bins = []
                        for t in self._ordered_metric_ts(group_sums, head):
                            gkey = (group, head, t)
                            if gkey not in group_sums:
                                continue
                            gval = group_sums[gkey] / max(group_denoms.get(gkey, 0.0), 1.0)
                            if self._log_axis_t_breakdown("group"):
                                metrics[
                                    self._metric_name(
                                        prefix, "by_group", f"{head}_t{t:g}", group
                                    )
                                ] = gval
                            group_bins.append(gval)
                        if group_bins:
                            metrics[
                                self._metric_name(prefix, "by_group", f"{head}_mean", group)
                            ] = sum(group_bins) / len(group_bins)
            if self.log_per_group:
                for group, count in sorted(group_counts.items()):
                    metrics[self._metric_name(prefix, "by_group", "count", group)] = float(count)
            if self.log_per_view:
                view_labels_seen = sorted({view for view, _, _ in view_sums})
                for view_label in view_labels_seen:
                    for head in heads:
                        view_bins = []
                        for t in self._ordered_metric_ts(view_sums, head):
                            vkey = (view_label, head, t)
                            if vkey not in view_sums:
                                continue
                            vval = view_sums[vkey] / max(view_denoms.get(vkey, 0.0), 1.0)
                            if self._log_axis_t_breakdown("view"):
                                metrics[
                                    self._metric_name(
                                        prefix, "by_view", f"{head}_t{t:g}", view_label
                                    )
                                ] = vval
                            view_bins.append(vval)
                        if view_bins:
                            metrics[
                                self._metric_name(prefix, "by_view", f"{head}_mean", view_label)
                            ] = sum(view_bins) / len(view_bins)
            if self.log_per_frame:
                frame_ids_seen = sorted({frame_i for frame_i, _, _ in frame_sums})
                for frame_i in frame_ids_seen:
                    for head in heads:
                        frame_bins = []
                        for t in self._ordered_metric_ts(frame_sums, head):
                            fkey = (frame_i, head, t)
                            if fkey not in frame_sums:
                                continue
                            fval = frame_sums[fkey] / max(frame_denoms.get(fkey, 0.0), 1.0)
                            if self._log_axis_t_breakdown("frame"):
                                metrics[
                                    self._metric_name(
                                        prefix,
                                        "by_frame",
                                        f"{head}_t{t:g}",
                                        f"future_{frame_i}",
                                    )
                                ] = fval
                            frame_bins.append(fval)
                        if frame_bins:
                            metrics[
                                self._metric_name(
                                    prefix, "by_frame", f"{head}_mean", f"future_{frame_i}"
                                )
                            ] = sum(frame_bins) / len(frame_bins)
            if self.log_per_view_frame:
                view_labels_seen = sorted({view for view, _, _, _ in view_frame_sums})
                frame_ids_seen = sorted({frame_i for _, frame_i, _, _ in view_frame_sums})
                for view_label in view_labels_seen:
                    for frame_i in frame_ids_seen:
                        for head in heads:
                            vf_bins = []
                            for t in self._ordered_metric_ts(view_frame_sums, head):
                                vfkey = (view_label, frame_i, head, t)
                                if vfkey not in view_frame_sums:
                                    continue
                                vfval = view_frame_sums[vfkey] / max(
                                    view_frame_denoms.get(vfkey, 0.0), 1.0
                                )
                                if self._log_axis_t_breakdown("view_frame"):
                                    metrics[
                                        self._metric_name(
                                            prefix,
                                            "by_view_frame",
                                            f"{head}_t{t:g}",
                                            f"future_{frame_i}",
                                            view_label,
                                        )
                                    ] = vfval
                                vf_bins.append(vfval)
                            if vf_bins:
                                metrics[
                                    self._metric_name(
                                        prefix,
                                        "by_view_frame",
                                        f"{head}_mean",
                                        f"future_{frame_i}",
                                        view_label,
                                    )
                                ] = sum(vf_bins) / len(vf_bins)
            for metric_name in sorted(interval_sums):
                metrics[self._metric_name(prefix, "global", metric_name)] = (
                    interval_sums[metric_name] / max(interval_denoms.get(metric_name, 0.0), 1.0)
                )
            if self.log_per_group:
                for group in sorted(group_counts):
                    for metric_name in sorted(interval_sums):
                        gkey = (group, metric_name)
                        if gkey not in interval_group_sums:
                            continue
                        metrics[self._metric_name(prefix, "by_group", metric_name, group)] = (
                            interval_group_sums[gkey]
                            / max(interval_group_denoms.get(gkey, 0.0), 1.0)
                        )
            if self.log_per_view:
                view_labels_seen = sorted({view for view, _ in interval_view_sums})
                for view_label in view_labels_seen:
                    for metric_name in sorted(interval_sums):
                        vkey = (view_label, metric_name)
                        if vkey not in interval_view_sums:
                            continue
                        metrics[self._metric_name(prefix, "by_view", metric_name, view_label)] = (
                            interval_view_sums[vkey]
                            / max(interval_view_denoms.get(vkey, 0.0), 1.0)
                        )
            if self.log_per_frame:
                frame_ids_seen = sorted({frame_i for frame_i, _ in interval_frame_sums})
                for frame_i in frame_ids_seen:
                    for metric_name in sorted(interval_sums):
                        fkey = (frame_i, metric_name)
                        if fkey not in interval_frame_sums:
                            continue
                        metrics[
                            self._metric_name(
                                prefix, "by_frame", metric_name, f"future_{frame_i}"
                            )
                        ] = interval_frame_sums[fkey] / max(
                            interval_frame_denoms.get(fkey, 0.0), 1.0
                        )
            if self.log_per_view_frame:
                view_labels_seen = sorted({view for view, _, _ in interval_view_frame_sums})
                frame_ids_seen = sorted({frame_i for _, frame_i, _ in interval_view_frame_sums})
                for view_label in view_labels_seen:
                    for frame_i in frame_ids_seen:
                        for metric_name in sorted(interval_sums):
                            vfkey = (view_label, frame_i, metric_name)
                            if vfkey not in interval_view_frame_sums:
                                continue
                            metrics[
                                self._metric_name(
                                    prefix,
                                    "by_view_frame",
                                    metric_name,
                                    f"future_{frame_i}",
                                    view_label,
                                )
                            ] = (
                                interval_view_frame_sums[vfkey]
                                / max(interval_view_frame_denoms.get(vfkey, 0.0), 1.0)
                            )

        logger.info(
            "fixed_val_loss step %d tag=%r: %s",
            global_step, tag, {k: round(v, 5) for k, v in sorted(metrics.items())},
        )
        return metrics
