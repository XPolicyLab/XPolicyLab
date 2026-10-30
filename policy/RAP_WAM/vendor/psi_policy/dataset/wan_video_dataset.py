from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch
import tqdm

from psi_policy.common.normalize_util import get_image_range_normalizer
from psi_policy.dataset.dataset_base import ClipIndexRecord
from psi_policy.dataset.dataset_base import DatasetBase
from psi_policy.dataset.fixed_normalizers import build_fixed_wan_normalizer
from psi_policy.dataset.normalizer_stats import StreamingArrayStats
from psi_policy.dataset.normalizer_stats import pop_progress_kwargs
from psi_policy.dataset.preprocessors import build_obs_preprocessors
from psi_policy.dataset.preprocessors.utils import normalize_image_size
from psi_policy.dataset.preprocessors.utils import shape_to_dim
from psi_policy.dataset.preprocessors.utils import to_float_2d_numpy
from psi_policy.model.common.normalizer import LinearNormalizer


class WanVideoDataset(DatasetBase):
    """基于 view 数据读取链路的 Wan 未来视频训练数据集。"""

    def __init__(
        self,
        *,
        data_root: str | Path,
        datahouse_id: str,
        view_id: str,
        condition_rgb_key: str = "rgb_head",
        target_rgb_key: str | None = None,
        output_condition_key: str = "rgb_head",
        output_future_key: str = "rgb_head_future",
        output_last_key: str = "rgb_head_last",
        state_keys: Sequence[str] | None = None,
        output_state_key: str | None = None,
        action_keys: Sequence[str] | None = None,
        include_action: bool = False,
        n_obs_steps: int = 1,
        n_action_steps: int = 0,
        video_future_horizon: int = 16,
        pad_before: int = 0,
        pad_after: int = 0,
        image_size: Tuple[int, int] | None = (224, 224),
        preprocessors: Any | None = None,
        split: str = "train",
    ) -> None:
        self.condition_rgb_key = str(condition_rgb_key).strip()
        self.target_rgb_key = (
            str(target_rgb_key).strip() if target_rgb_key is not None else self.condition_rgb_key
        )
        self.output_condition_key = str(output_condition_key).strip()
        self.output_future_key = str(output_future_key).strip()
        self.output_last_key = str(output_last_key).strip()
        self._state_keys = _clean_keys(state_keys or [], "state_keys", allow_empty=True)
        self.output_state_key = (
            str(output_state_key).strip()
            if output_state_key is not None
            else "robot_state"
        )
        self.include_action = bool(include_action)
        self.n_obs_steps = int(n_obs_steps)
        self.video_future_horizon = int(video_future_horizon)
        self.action_output_horizon = int(n_action_steps) if self.include_action else 0
        self.sequence_horizon = self.n_obs_steps + max(self.video_future_horizon, self.action_output_horizon)
        self.image_size = normalize_image_size(image_size)
        self._action_keys = _clean_keys(action_keys or [], "action_keys", allow_empty=True)

        if self.n_obs_steps <= 0:
            raise ValueError("n_obs_steps must be > 0")
        if self.video_future_horizon <= 0:
            raise ValueError("video_future_horizon must be > 0")
        if not self.condition_rgb_key:
            raise ValueError("condition_rgb_key cannot be empty")
        if not self.target_rgb_key:
            raise ValueError("target_rgb_key cannot be empty")
        if not self.output_condition_key or not self.output_future_key or not self.output_last_key:
            raise ValueError("output obs keys cannot be empty")
        if self._state_keys and not self.output_state_key:
            raise ValueError("output_state_key cannot be empty when state_keys is provided")

        super().__init__(
            data_root=data_root,
            datahouse_id=datahouse_id,
            view_id=view_id,
            n_action_steps=self.sequence_horizon,
            n_obs_steps=self.n_obs_steps,
            pad_before=pad_before,
            pad_after=pad_after,
            split=split,
        )

        output_image_shape = _resolve_output_image_shape(
            self.modality_shapes[self.condition_rgb_key],
            image_size=self.image_size,
        )
        self.obs_shapes: Dict[str, Tuple[int, ...]] = {
            self.output_condition_key: output_image_shape,
            self.output_future_key: output_image_shape,
            self.output_last_key: output_image_shape,
        }
        self.state_shapes: Dict[str, Tuple[int, ...]] = {
            key: self.modality_shapes[key] for key in self._state_keys
        }
        self.state_dims = {
            key: shape_to_dim(self.state_shapes[key]) for key in self._state_keys
        }
        self.state_dim = int(sum(self.state_dims.values()))
        if self._state_keys:
            self.obs_shapes[self.output_state_key] = (self.state_dim,)
        self._obs_keys = [
            self.output_condition_key,
            self.output_future_key,
            self.output_last_key,
        ]
        if self._state_keys:
            self._obs_keys.append(self.output_state_key)
        self._obs_preprocessors = build_obs_preprocessors(
            preprocessors,
            obs_keys=self._obs_keys,
            default_image_size=self.image_size,
        )

        self.action_shapes: Dict[str, Tuple[int, ...]] = {
            key: self.modality_shapes[key] for key in self._action_keys
        }
        self.action_dims = {
            key: shape_to_dim(self.action_shapes[key]) for key in self._action_keys
        }
        self.action_dim = int(sum(self.action_dims.values()))

    @property
    def modalities_need(self) -> Sequence[str]:
        raw_modalities = [self.condition_rgb_key, self.target_rgb_key]
        raw_modalities.extend(self._state_keys)
        if self.include_action:
            raw_modalities.extend(self._action_keys)
        return list(dict.fromkeys(raw_modalities))

    @property
    def obs_keys(self) -> Sequence[str]:
        return list(self._obs_keys)

    @property
    def action_keys(self) -> Sequence[str]:
        return list(self._action_keys)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        window = self._resolve_window(idx)

        condition_raw = self._load_obs_prefix_window(
            window=window,
            modality=self.condition_rgb_key,
            steps=self.n_obs_steps,
        )
        target_full = self._load_modality_window(
            window=window,
            modality=self.target_rgb_key,
            steps=self.sequence_horizon,
        )
        future_raw = target_full[self.n_obs_steps : self.n_obs_steps + self.video_future_horizon]
        if future_raw.shape[0] != self.video_future_horizon:
            raise ValueError(
                f"Future horizon mismatch: expected {self.video_future_horizon}, got {future_raw.shape[0]}"
            )
        last_raw = future_raw[-1:]

        obs = {
            self.output_condition_key: self.process_obs_array(self.output_condition_key, condition_raw),
            self.output_future_key: self.process_obs_array(self.output_future_key, future_raw),
            self.output_last_key: self.process_obs_array(self.output_last_key, last_raw),
        }
        if self._state_keys:
            state_parts = []
            for key in self._state_keys:
                state_full = self._load_obs_prefix_window(
                    window=window,
                    modality=key,
                    steps=self.n_obs_steps,
                )
                state_parts.append(
                    to_float_2d_numpy(state_full, key=key, expected_dim=self.state_dims[key])
                )
            state_raw = np.concatenate(state_parts, axis=1).astype(np.float32, copy=False)
            obs[self.output_state_key] = self.process_obs_array(self.output_state_key, state_raw)

        batch: Dict[str, torch.Tensor | Dict[str, torch.Tensor]] = {"obs": obs}
        if self.include_action and self._action_keys:
            action_parts = []
            for key in self._action_keys:
                action_full = self._load_modality_window(
                    window=window,
                    modality=key,
                    steps=self.sequence_horizon,
                )
                action_slice = action_full[self.n_obs_steps : self.n_obs_steps + self.action_output_horizon]
                action_parts.append(
                    torch.from_numpy(
                        to_float_2d_numpy(action_slice, key=key, expected_dim=self.action_dims[key])
                    )
                )
            batch["action"] = (
                torch.cat(action_parts, dim=-1)
                if action_parts
                else torch.zeros((self.action_output_horizon, 0), dtype=torch.float32)
            )
        return batch

    def get_all_actions(self) -> torch.Tensor:
        if not self.include_action or not self._action_keys or not self._clips:
            return torch.empty((0, self.action_dim), dtype=torch.float32)

        buffers = []
        for clip in self._clips:
            action_raw = self._build_action_array_for_clip(clip)
            if action_raw is not None:
                buffers.append(action_raw)

        if not buffers:
            return torch.empty((0, self.action_dim), dtype=torch.float32)
        return torch.from_numpy(np.concatenate(buffers, axis=0).astype(np.float32, copy=False))

    def get_normalizer(self, mode: str = "limits", **kwargs):
        normalizer = LinearNormalizer()
        fit_kwargs = dict(kwargs)
        fixed_preset = str(fit_kwargs.pop("fixed_preset", "")).strip()
        if fixed_preset:
            image_obs_keys = {
                self.output_condition_key,
                self.output_future_key,
                self.output_last_key,
            }
            return build_fixed_wan_normalizer(
                fixed_preset=fixed_preset,
                state_keys=self._state_keys,
                action_keys=self._action_keys,
                include_action=self.include_action,
                output_state_key=self.output_state_key,
                obs_keys=self._obs_keys,
                image_obs_keys=image_obs_keys,
            )

        last_n_dims = int(fit_kwargs.pop("last_n_dims", 1))
        show_progress, progress_desc = pop_progress_kwargs(
            fit_kwargs,
            default_desc=f"Loading Wan normalizer [{self.datahouse_id}:{self.view_id}]",
        )

        action_stats = (
            StreamingArrayStats(last_n_dims=last_n_dims)
            if self.include_action and self._action_keys
            else None
        )
        state_stats = StreamingArrayStats(last_n_dims=last_n_dims) if self._state_keys else None

        total_clips = len(self._clips)
        progress_bar = tqdm.tqdm(
            total=total_clips,
            desc=progress_desc,
            unit="clip",
            leave=show_progress,
            disable=(not show_progress) or total_clips <= 0,
        )
        try:
            for clip in self._clips:
                if action_stats is not None:
                    action_stats.update(self._build_action_array_for_clip(clip))
                if state_stats is not None:
                    state_stats.update(self._build_state_array_for_clip(clip))
                progress_bar.update(1)
        finally:
            progress_bar.close()

        if action_stats is not None and action_stats.count > 0:
            normalizer["action"] = action_stats.build_normalizer(mode=mode, **fit_kwargs)

        if state_stats is not None and state_stats.count > 0:
            normalizer[self.output_state_key] = state_stats.build_normalizer(
                mode=mode,
                **fit_kwargs,
            )

        image_obs_keys = {
            self.output_condition_key,
            self.output_future_key,
            self.output_last_key,
        }
        for key in self._obs_keys:
            if key not in image_obs_keys:
                continue
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def export_model_spec(self) -> Dict[str, object]:
        return {
            "obs_shapes": dict(self.obs_shapes),
            "action_dim": self.action_dim,
            "action_horizon": self.action_output_horizon,
            "obs_keys": list(self._obs_keys),
            "obs_horizon": self.n_obs_steps,
            "future_horizon": self.video_future_horizon,
            "state_dim": self.state_dim,
            "state_horizon": self.n_obs_steps if self._state_keys else 0,
            "state_key": self.output_state_key if self._state_keys else None,
        }

    def process_obs_array(self, key: str, value: np.ndarray) -> torch.Tensor:
        preprocessor = self._obs_preprocessors[key]
        return preprocessor.process(value, key=key, obs_shape=self.obs_shapes[key])

    def get_sample_meta(self, idx: int) -> Dict[str, object]:
        window = self._resolve_window(idx)
        clip = self._clips[window.clip_idx]
        condition_frame_idx = self._condition_frame_idx(window=window, clip=clip)
        return {
            "source_kind": "view",
            "data_root": str(self.data_root),
            "datahouse_id": str(self.datahouse_id),
            "view_id": str(self.view_id),
            "split": str(self.split),
            "sample_idx": int(idx),
            "clip_idx": int(window.clip_idx),
            "condition_frame_idx": int(condition_frame_idx),
            "task": str(clip.task),
            "volume_id": str(clip.volume_id),
            "clip_start_idx": int(clip.start_idx),
            "clip_end_idx": int(clip.end_idx),
            "output_condition_key": str(self.output_condition_key),
            "output_future_key": str(self.output_future_key),
            "output_last_key": str(self.output_last_key),
            "output_state_key": str(self.output_state_key),
            "action_dims": {
                str(key): int(value) for key, value in self.action_dims.items()
            },
        }

    def resolve_sample_idx(
        self,
        *,
        sample_idx: int | None = None,
        source_index: int | None = None,
        clip_idx: int | None = None,
        frame_idx: int = 0,
    ) -> int:
        total_samples = len(self)
        if total_samples <= 0:
            raise ValueError("dataset is empty")

        del source_index
        if sample_idx is not None:
            resolved = int(sample_idx)
            if resolved < 0:
                resolved += total_samples
            if resolved < 0 or resolved >= total_samples:
                raise IndexError(f"sample_idx out of range: {sample_idx}, total={total_samples}")
            return resolved

        total_clips = len(self._clips)
        resolved_clip_idx = 0 if clip_idx is None else int(clip_idx)
        if resolved_clip_idx < 0:
            resolved_clip_idx += total_clips
        if resolved_clip_idx < 0 or resolved_clip_idx >= total_clips:
            raise IndexError(f"clip_idx out of range: {clip_idx}, total={total_clips}")

        clip = self._clips[resolved_clip_idx]
        resolved_frame_idx = int(frame_idx)
        if resolved_frame_idx < 0:
            resolved_frame_idx += int(clip.clip_length)
        if resolved_frame_idx < 0 or resolved_frame_idx >= int(clip.clip_length):
            raise IndexError(
                f"frame_idx out of range: {frame_idx}, clip_length={int(clip.clip_length)}"
            )

        span = self._window_span_by_clip[resolved_clip_idx]
        if span is None:
            raise IndexError(
                f"Cannot find sample for clip_idx={resolved_clip_idx}, frame_idx={resolved_frame_idx}."
            )

        resolved_offset = int(resolved_frame_idx - self.n_obs_steps + 1)
        if resolved_offset < int(span.min_start) or resolved_offset > int(span.max_start):
            raise IndexError(
                f"Cannot find sample for clip_idx={resolved_clip_idx}, frame_idx={resolved_frame_idx}."
            )

        return int(span.global_start + (resolved_offset - int(span.min_start)))

    def get_rollout_arrays(self, idx: int) -> Dict[str, object]:
        window = self._resolve_window(idx)
        clip = self._clips[window.clip_idx]
        payload: Dict[str, object] = {
            "condition_rgb_raw": self.read_clip_modality(clip, self.condition_rgb_key),
            "target_rgb_raw": self.read_clip_modality(clip, self.target_rgb_key),
            "sample_meta": self.get_sample_meta(idx),
        }
        state_raw = self._build_state_array_for_clip(clip)
        if state_raw is not None:
            payload["state_raw"] = state_raw
        action_raw = self._build_action_array_for_clip(clip)
        if action_raw is not None:
            payload["action_raw"] = action_raw
        return payload

    def _build_state_array_for_clip(self, clip: ClipIndexRecord) -> np.ndarray | None:
        if not self._state_keys:
            return None

        parts = []
        for key in self._state_keys:
            arr = self.read_clip_modality(clip, key)
            parts.append(to_float_2d_numpy(arr, key=key, expected_dim=self.state_dims[key]))
        if not parts:
            return None
        return np.concatenate(parts, axis=1).astype(np.float32, copy=False)

    def _build_action_array_for_clip(self, clip: ClipIndexRecord) -> np.ndarray | None:
        if not self.include_action or not self._action_keys:
            return None

        parts = []
        for key in self._action_keys:
            arr = self.read_clip_modality(clip, key)
            parts.append(to_float_2d_numpy(arr, key=key, expected_dim=self.action_dims[key]))
        if not parts:
            return None
        return np.concatenate(parts, axis=1).astype(np.float32, copy=False)

    def _condition_frame_idx(self, *, window, clip: ClipIndexRecord) -> int:
        offset = int(window.buffer_start) - int(window.sample_start) - int(clip.start_idx)
        return int(offset + self.n_obs_steps - 1)


def _clean_keys(
    values: Sequence[str],
    name: str,
    *,
    allow_empty: bool = False,
) -> list[str]:
    cleaned: list[str] = []
    seen = set()
    for raw in values:
        key = str(raw).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        cleaned.append(key)
    if not cleaned and not allow_empty:
        raise ValueError(f"{name} cannot be empty")
    return cleaned


def _resolve_output_image_shape(
    raw_shape: Sequence[int],
    *,
    image_size: tuple[int, int],
) -> tuple[int, int, int]:
    if len(raw_shape) != 3:
        raise ValueError(f"Expected raw image shape with 3 dims, got {tuple(raw_shape)}")
    h, w = int(image_size[0]), int(image_size[1])
    if int(raw_shape[-1]) in (1, 3, 4):
        return 3, h, w
    if int(raw_shape[0]) in (1, 3, 4):
        return min(3, int(raw_shape[0])), h, w
    raise ValueError(f"Could not infer RGB channel dimension from raw shape {tuple(raw_shape)}")


__all__ = ["WanVideoDataset"]
