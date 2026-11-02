from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch

from psi_policy.common.obs_spec import resolve_obs_spec
from psi_policy.dataset.dataset_base import DatasetBase
from psi_policy.dataset.preprocessors import build_obs_preprocessors
from psi_policy.dataset.preprocessors.utils import normalize_image_size
from psi_policy.dataset.preprocessors.utils import shape_to_dim
from psi_policy.dataset.preprocessors.utils import to_float_2d_numpy
from psi_policy.model.common.normalizer import LinearNormalizer


class SingleViewDataset(DatasetBase):
    """训练数据集：输出 obs 字典和拼接后的 action。"""

    def __init__(
        self,
        *,
        data_root: str | Path,
        datahouse_id: str,
        view_id: str,
        obs_spec: Any,
        action_keys: Sequence[str],
        n_action_steps: int = 1,
        n_obs_steps: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        image_size: Tuple[int, int] | None = (224, 224),
        preprocessors: Any | None = None,
        split: str = "train",
    ) -> None:
        resolved_obs = resolve_obs_spec(obs_spec)
        resolved_groups = {
            str(name): list(values)
            for name, values in dict(resolved_obs.get("groups", {})).items()
        }
        self._obs_spec = {
            "obs_keys": list(resolved_obs.get("obs_keys", [])),
            "groups": resolved_groups,
        }
        for name, values in resolved_groups.items():
            self._obs_spec[str(name)] = list(values)

        self._obs_keys = list(self._obs_spec["obs_keys"])
        self._action_keys = _clean_keys(action_keys, "action_keys")
        self.image_size = normalize_image_size(image_size)

        super().__init__(
            data_root=data_root,
            datahouse_id=datahouse_id,
            view_id=view_id,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            pad_before=pad_before,
            pad_after=pad_after,
            split=split,
        )

        self.obs_shapes: Dict[str, Tuple[int, ...]] = {
            key: self.modality_shapes[key] for key in self._obs_keys
        }
        self.action_shapes: Dict[str, Tuple[int, ...]] = {
            key: self.modality_shapes[key] for key in self._action_keys
        }
        self._obs_preprocessors = build_obs_preprocessors(
            preprocessors,
            obs_keys=self._obs_keys,
            default_image_size=self.image_size,
        )

        self.action_dim = int(
            sum(np.prod(shape, dtype=np.int64) if len(shape) > 0 else 1 for shape in self.action_shapes.values())
        )
        self.action_dims = {
            key: shape_to_dim(self.action_shapes[key]) for key in self._action_keys
        }

    @property
    def modalities_need(self) -> Sequence[str]:
        return [*self._obs_keys, *self._action_keys]

    @property
    def obs_keys(self) -> Sequence[str]:
        return list(self._obs_keys)

    @property
    def action_keys(self) -> Sequence[str]:
        return list(self._action_keys)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        window = self._resolve_window(idx)

        obs: Dict[str, torch.Tensor] = {}
        for key in self._obs_keys:
            try:
                arr = self._load_obs_prefix_window(window=window, modality=key, steps=self.n_obs_steps)
            except KeyError:
                arr = self._make_zeros(modality=key, target_steps=self.n_obs_steps)
            obs[key] = self.process_obs_array(key=key, value=arr)

        action_parts = []
        for key in self._action_keys:
            arr = self._load_modality_window(window=window, modality=key, steps=self.action_horizon)
            action_parts.append(
                torch.from_numpy(to_float_2d_numpy(arr, key=key, expected_dim=self.action_dims[key]))
            )

        action = (
            torch.cat(action_parts, dim=-1)
            if action_parts
            else torch.zeros((self.action_horizon, 0), dtype=torch.float32)
        )
        return {"obs": obs, "action": action}

    def get_all_actions(self) -> torch.Tensor:
        if not self._clips:
            return torch.empty((0, self.action_dim), dtype=torch.float32)

        buffers = []
        for clip in self._clips:
            action_parts = []
            for key in self._action_keys:
                arr = self.read_clip_modality(clip, key)
                action_parts.append(to_float_2d_numpy(arr, key=key, expected_dim=self.action_dims[key]))
            if action_parts:
                buffers.append(np.concatenate(action_parts, axis=1))

        if not buffers:
            return torch.empty((0, self.action_dim), dtype=torch.float32)
        return torch.from_numpy(np.concatenate(buffers, axis=0).astype(np.float32, copy=False))

    def get_normalizer(self, mode: str = "limits", **kwargs):
        normalizer = LinearNormalizer()

        obs_fit_buffers: Dict[str, list[np.ndarray]] = {key: [] for key in self._obs_keys}
        action_buffers: list[np.ndarray] = []

        for clip in self._clips:
            for key in self._obs_keys:
                preprocessor = self._obs_preprocessors[key]
                if not preprocessor.needs_normalizer_fit_value(
                    key=key,
                    obs_shape=self.obs_shapes[key],
                ):
                    continue
                try:
                    arr = self.read_clip_modality(clip, key)
                except KeyError:
                    arr = self._make_zeros(modality=key, target_steps=clip.clip_length)
                fit_value = preprocessor.get_normalizer_fit_value(
                    arr,
                    key=key,
                    obs_shape=self.obs_shapes[key],
                )
                if fit_value is not None:
                    obs_fit_buffers[key].append(fit_value)

            action_parts = []
            for key in self._action_keys:
                arr = self.read_clip_modality(clip, key)
                action_parts.append(
                    to_float_2d_numpy(arr, key=key, expected_dim=self.action_dims[key])
                )
            if action_parts:
                action_buffers.append(np.concatenate(action_parts, axis=1))

        fit_dict = {}
        for key, values in obs_fit_buffers.items():
            if values:
                fit_dict[key] = np.concatenate(values, axis=0)

        if action_buffers:
            fit_dict["action"] = np.concatenate(action_buffers, axis=0)

        if fit_dict:
            normalizer.fit(data=fit_dict, last_n_dims=1, mode=mode, **kwargs)

        for key in self._obs_keys:
            normalizer_field = self._obs_preprocessors[key].get_normalizer(key=key)
            if normalizer_field is not None:
                normalizer[key] = normalizer_field

        return normalizer

    def export_model_spec(self) -> Dict[str, object]:
        return {
            "obs_shapes": dict(self.obs_shapes),
            "action_dim": self.action_dim,
            "action_horizon": self.action_horizon,
            "obs_spec": _copy_obs_spec(self._obs_spec),
            "obs_horizon": self.n_obs_steps,
        }

    def process_obs_array(self, *, key: str, value: np.ndarray) -> torch.Tensor:
        preprocessor = self._obs_preprocessors[key]
        return preprocessor.process(value, key=key, obs_shape=self.obs_shapes[key])

    def make_missing_obs_tensor(self, *, key: str, steps: int | None = None) -> torch.Tensor:
        target_steps = self.n_obs_steps if steps is None else int(steps)
        preprocessor = self._obs_preprocessors[key]
        return preprocessor.make_missing(
            key=key,
            steps=target_steps,
            obs_shape=self.obs_shapes[key],
        )


def _clean_keys(values: Sequence[str], name: str, allow_empty: bool = False) -> list[str]:
    cleaned: list[str] = []
    seen = set()
    for raw in values:
        key = str(raw).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        cleaned.append(key)
    if not cleaned and not allow_empty:
        raise ValueError(f"{name} 不能为空")
    return cleaned


def _copy_obs_spec(obs_spec: Dict[str, object]) -> Dict[str, object]:
    groups = {
        str(name): list(values)
        for name, values in dict(obs_spec.get("groups", {})).items()
    }
    copied: Dict[str, object] = {
        "obs_keys": list(obs_spec.get("obs_keys", [])),
        "groups": groups,
    }
    for name, values in groups.items():
        copied[name] = list(values)
    return copied


__all__ = ["SingleViewDataset"]
