from __future__ import annotations

import bisect
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch

from psi_policy.common.obs_spec import resolve_obs_spec
from psi_policy.dataset.preprocessors import build_obs_preprocessors
from psi_policy.dataset.preprocessors import remap_preprocessor_cfgs
from psi_policy.dataset.preprocessors.utils import normalize_image_size
from psi_policy.dataset.preprocessors.utils import shape_to_dim
from psi_policy.dataset.preprocessors.utils import to_float_2d_numpy
from psi_policy.dataset.single_view_dataset import SingleViewDataset
from psi_policy.model.common.normalizer import LinearNormalizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SourceConfig:
    datahouse_id: str
    view_id: str
    source_key: str
    alias_map: Dict[str, str]
    required_obs_keys: tuple[str, ...]
    optional_obs_keys: tuple[str, ...]


@dataclass(frozen=True)
class _SourceRuntime:
    config: _SourceConfig
    dataset: SingleViewDataset
    obs_key_map: Dict[str, str]
    action_key_map: Dict[str, str]
    obs_shapes: Dict[str, Tuple[int, ...]]
    action_shapes: Dict[str, Tuple[int, ...]]
    missing_optional_obs_keys: tuple[str, ...]


class MultiViewDataset(torch.utils.data.Dataset):
    """多视图聚合数据集：按样本量自然混合多个 SingleViewDataset。"""

    def __init__(
        self,
        *,
        data_root: str | Path,
        sources: Sequence[Mapping[str, Any]],
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
        super().__init__()

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
        self._preprocessor_cfg = preprocessors

        self.data_root = Path(data_root)
        self.action_horizon = int(n_action_steps)
        self.n_obs_steps = int(n_obs_steps)
        self.pad_before = int(pad_before)
        self.pad_after = int(pad_after)
        self.split = str(split).strip()
        if not self.split:
            raise ValueError("split 不能为空")
        self.image_size = normalize_image_size(image_size)

        source_cfgs = self._parse_sources(sources)
        self._runtimes = self._build_runtimes(source_cfgs)
        self._offsets = self._build_offsets(self._runtimes)

        self.obs_shapes = self._merge_obs_shapes(self._runtimes)
        self.action_shapes = self._merge_action_shapes(self._runtimes)
        self._obs_preprocessors = build_obs_preprocessors(
            preprocessors,
            obs_keys=self._obs_keys,
            default_image_size=self.image_size,
        )
        self.action_dims = {
            key: shape_to_dim(self.action_shapes[key]) for key in self._action_keys
        }
        self.action_dim = int(sum(self.action_dims.values()))

    @property
    def obs_keys(self) -> Sequence[str]:
        return list(self._obs_keys)

    @property
    def action_keys(self) -> Sequence[str]:
        return list(self._action_keys)

    def __len__(self) -> int:
        if not self._offsets:
            return 0
        return self._offsets[-1]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        runtime, local_idx = self._resolve_runtime_by_index(idx)
        sample = runtime.dataset[local_idx]

        obs_payload = sample.get("obs")
        if not isinstance(obs_payload, dict):
            raise TypeError("SingleViewDataset 返回的 obs 不是字典")

        obs: Dict[str, torch.Tensor] = {}
        for canonical_key in self._obs_keys:
            source_key = runtime.obs_key_map.get(canonical_key)
            if source_key is None:
                obs[canonical_key] = self._make_missing_obs_tensor(canonical_key)
                continue

            value = obs_payload.get(source_key)
            if value is None:
                if canonical_key in runtime.config.required_obs_keys:
                    raise KeyError(
                        f"required obs 缺失: source={runtime.config.source_key} key={canonical_key}"
                    )
                obs[canonical_key] = self._make_missing_obs_tensor(canonical_key)
                continue
            obs[canonical_key] = value

        action = sample.get("action")
        if not isinstance(action, torch.Tensor):
            raise TypeError("SingleViewDataset 返回的 action 不是 Tensor")
        return {"obs": obs, "action": action}

    def get_all_actions(self) -> torch.Tensor:
        if not self._runtimes:
            return torch.empty((0, self.action_dim), dtype=torch.float32)

        chunks = []
        for runtime in self._runtimes:
            values = runtime.dataset.get_all_actions()
            if values.numel() == 0:
                continue
            chunks.append(values)

        if not chunks:
            return torch.empty((0, self.action_dim), dtype=torch.float32)
        return torch.cat(chunks, dim=0)

    def get_normalizer(self, mode: str = "limits", **kwargs):
        normalizer = LinearNormalizer()

        obs_fit_buffers: Dict[str, list[np.ndarray]] = {key: [] for key in self._obs_keys}
        action_buffers: list[np.ndarray] = []

        for runtime in self._runtimes:
            ds = runtime.dataset
            for clip in ds._clips:
                for canonical_key in self._obs_keys:
                    preprocessor = self._obs_preprocessors[canonical_key]
                    if not preprocessor.needs_normalizer_fit_value(
                        key=canonical_key,
                        obs_shape=self.obs_shapes[canonical_key],
                    ):
                        continue
                    source_key = runtime.obs_key_map.get(canonical_key)
                    if source_key is None:
                        arr = _make_raw_zeros(
                            steps=clip.clip_length,
                            shape=self.obs_shapes[canonical_key],
                        )
                    else:
                        try:
                            arr = ds.read_clip_modality(clip, source_key)
                        except KeyError:
                            arr = _make_raw_zeros(
                                steps=clip.clip_length,
                                shape=self.obs_shapes[canonical_key],
                            )
                    fit_value = preprocessor.get_normalizer_fit_value(
                        arr,
                        key=canonical_key,
                        obs_shape=self.obs_shapes[canonical_key],
                    )
                    if fit_value is not None:
                        obs_fit_buffers[canonical_key].append(fit_value)

                action_parts = []
                for canonical_key in self._action_keys:
                    source_key = runtime.action_key_map[canonical_key]
                    arr = ds.read_clip_modality(clip, source_key)
                    action_parts.append(
                        to_float_2d_numpy(arr, key=canonical_key, expected_dim=self.action_dims[canonical_key])
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

    def _make_missing_obs_tensor(self, canonical_key: str) -> torch.Tensor:
        return self._obs_preprocessors[canonical_key].make_missing(
            key=canonical_key,
            steps=self.n_obs_steps,
            obs_shape=self.obs_shapes[canonical_key],
        )

    def _resolve_runtime_by_index(self, idx: int) -> tuple[_SourceRuntime, int]:
        total = len(self)
        if total == 0:
            raise IndexError("dataset is empty")

        if idx < 0:
            idx += total
        if idx < 0 or idx >= total:
            raise IndexError(f"index out of range: {idx}")

        source_idx = bisect.bisect_right(self._offsets, idx)
        start = 0 if source_idx == 0 else self._offsets[source_idx - 1]
        local_idx = idx - start
        return self._runtimes[source_idx], local_idx

    def _parse_sources(self, sources: Sequence[Mapping[str, Any]]) -> list[_SourceConfig]:
        if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
            raise TypeError("sources 必须是对象列表")
        if not sources:
            raise ValueError("sources 不能为空")

        known_modalities = set(self._obs_keys) | set(self._action_keys)
        configs: list[_SourceConfig] = []
        seen_source_keys: set[str] = set()

        for raw_source in sources:
            source_map = _to_mapping(raw_source)
            datahouse_id = _non_empty(str(source_map.get("datahouse_id", "")), "datahouse_id")
            view_id = _non_empty(str(source_map.get("view_id", "")), "view_id")
            source_key = f"{datahouse_id}:{view_id}"
            if source_key in seen_source_keys:
                raise ValueError(f"source 重复: {source_key}")
            seen_source_keys.add(source_key)

            alias_map_raw = source_map.get("alias_map") or {}
            alias_map = _normalize_alias_map(alias_map_raw)
            unknown_alias = sorted(key for key in alias_map if key not in known_modalities)
            if unknown_alias:
                raise ValueError(
                    f"source={source_key} alias_map 包含未知 key: {', '.join(unknown_alias)}"
                )

            optional_obs_keys = _clean_keys(
                source_map.get("optional_obs_keys", []),
                "optional_obs_keys",
                allow_empty=True,
            )
            _ensure_subset(optional_obs_keys, self._obs_keys, f"source={source_key} optional_obs_keys")

            raw_required = source_map.get("required_obs_keys")
            if raw_required is None:
                required_obs_keys = [key for key in self._obs_keys if key not in set(optional_obs_keys)]
            else:
                required_obs_keys = _clean_keys(
                    raw_required,
                    "required_obs_keys",
                    allow_empty=True,
                )
                _ensure_subset(required_obs_keys, self._obs_keys, f"source={source_key} required_obs_keys")
                overlap = sorted(set(required_obs_keys) & set(optional_obs_keys))
                if overlap:
                    raise ValueError(
                        f"source={source_key} required/optional obs 配置重复: {', '.join(overlap)}"
                    )
                covered = set(required_obs_keys) | set(optional_obs_keys)
                for key in self._obs_keys:
                    if key not in covered:
                        required_obs_keys.append(key)

            _ensure_unique_mapped_keys(
                required_obs_keys + optional_obs_keys,
                alias_map,
                f"source={source_key} obs alias",
            )
            _ensure_unique_mapped_keys(
                self._action_keys,
                alias_map,
                f"source={source_key} action alias",
            )

            configs.append(
                _SourceConfig(
                    datahouse_id=datahouse_id,
                    view_id=view_id,
                    source_key=source_key,
                    alias_map=alias_map,
                    required_obs_keys=tuple(required_obs_keys),
                    optional_obs_keys=tuple(optional_obs_keys),
                )
            )

        return configs

    def _build_runtimes(self, source_cfgs: Sequence[_SourceConfig]) -> list[_SourceRuntime]:
        runtimes: list[_SourceRuntime] = []

        for cfg in source_cfgs:
            manifest_modalities = _read_manifest_modalities(
                data_root=self.data_root,
                datahouse_id=cfg.datahouse_id,
                view_id=cfg.view_id,
            )

            selected_obs_keys: list[str] = []
            missing_optional_keys: list[str] = []
            optional_set = set(cfg.optional_obs_keys)
            required_set = set(cfg.required_obs_keys)
            for canonical_key in self._obs_keys:
                source_key = cfg.alias_map.get(canonical_key, canonical_key)
                if canonical_key in optional_set and source_key not in manifest_modalities:
                    missing_optional_keys.append(canonical_key)
                    continue
                selected_obs_keys.append(canonical_key)

            source_obs_spec = _build_source_obs_spec(
                selected_obs_keys=selected_obs_keys,
                alias_map=cfg.alias_map,
            )
            source_action_keys = [cfg.alias_map.get(key, key) for key in self._action_keys]
            source_preprocessors = remap_preprocessor_cfgs(
                self._preprocessor_cfg,
                selected_obs_keys=selected_obs_keys,
                alias_map=cfg.alias_map,
            )

            dataset = SingleViewDataset(
                data_root=self.data_root,
                datahouse_id=cfg.datahouse_id,
                view_id=cfg.view_id,
                obs_spec=source_obs_spec,
                action_keys=source_action_keys,
                n_action_steps=self.action_horizon,
                n_obs_steps=self.n_obs_steps,
                pad_before=self.pad_before,
                pad_after=self.pad_after,
                image_size=self.image_size,
                preprocessors=source_preprocessors,
                split=self.split,
            )

            obs_key_map: Dict[str, str] = {}
            obs_shapes: Dict[str, Tuple[int, ...]] = {}
            for canonical_key in selected_obs_keys:
                source_key = cfg.alias_map.get(canonical_key, canonical_key)
                if source_key in dataset.obs_shapes:
                    obs_key_map[canonical_key] = source_key
                    obs_shapes[canonical_key] = dataset.obs_shapes[source_key]
                elif canonical_key in required_set:
                    raise ValueError(
                        f"source={cfg.source_key} required obs 不存在: {canonical_key}"
                    )
                else:
                    missing_optional_keys.append(canonical_key)

            action_key_map: Dict[str, str] = {}
            action_shapes: Dict[str, Tuple[int, ...]] = {}
            for canonical_key in self._action_keys:
                source_key = cfg.alias_map.get(canonical_key, canonical_key)
                if source_key not in dataset.action_shapes:
                    raise ValueError(
                        f"source={cfg.source_key} action 不存在: {canonical_key}"
                    )
                action_key_map[canonical_key] = source_key
                action_shapes[canonical_key] = dataset.action_shapes[source_key]

            missing_optional_tuple = tuple(sorted(set(missing_optional_keys)))
            if missing_optional_tuple:
                logger.warning(
                    "source=%s optional obs 缺失，将在采样时补零: %s",
                    cfg.source_key,
                    ", ".join(missing_optional_tuple),
                )

            runtimes.append(
                _SourceRuntime(
                    config=cfg,
                    dataset=dataset,
                    obs_key_map=obs_key_map,
                    action_key_map=action_key_map,
                    obs_shapes=obs_shapes,
                    action_shapes=action_shapes,
                    missing_optional_obs_keys=missing_optional_tuple,
                )
            )

        return runtimes

    def _merge_obs_shapes(self, runtimes: Sequence[_SourceRuntime]) -> Dict[str, Tuple[int, ...]]:
        merged: Dict[str, Tuple[int, ...]] = {}

        for canonical_key in self._obs_keys:
            baseline: Tuple[int, ...] | None = None
            for runtime in runtimes:
                shape = runtime.obs_shapes.get(canonical_key)
                if shape is None:
                    continue
                if baseline is None:
                    baseline = tuple(shape)
                    continue
                if tuple(shape) != baseline:
                    raise ValueError(
                        "obs shape 冲突: "
                        f"key={canonical_key} source={runtime.config.source_key} "
                        f"expected={list(baseline)} got={list(shape)}"
                    )
            if baseline is None:
                raise ValueError(f"obs key 在所有 source 中都不可用: {canonical_key}")
            merged[canonical_key] = baseline

        return merged

    def _merge_action_shapes(self, runtimes: Sequence[_SourceRuntime]) -> Dict[str, Tuple[int, ...]]:
        merged: Dict[str, Tuple[int, ...]] = {}

        for canonical_key in self._action_keys:
            baseline: Tuple[int, ...] | None = None
            for runtime in runtimes:
                shape = runtime.action_shapes.get(canonical_key)
                if shape is None:
                    raise ValueError(
                        f"source={runtime.config.source_key} action shape 缺失: {canonical_key}"
                    )
                if baseline is None:
                    baseline = tuple(shape)
                    continue
                if tuple(shape) != baseline:
                    raise ValueError(
                        "action shape 冲突: "
                        f"key={canonical_key} source={runtime.config.source_key} "
                        f"expected={list(baseline)} got={list(shape)}"
                    )
            assert baseline is not None
            merged[canonical_key] = baseline

        return merged

    @staticmethod
    def _build_offsets(runtimes: Sequence[_SourceRuntime]) -> list[int]:
        offsets: list[int] = []
        total = 0
        for runtime in runtimes:
            total += len(runtime.dataset)
            offsets.append(total)
        return offsets


def _build_source_obs_spec(
    *,
    selected_obs_keys: Sequence[str],
    alias_map: Mapping[str, str],
) -> Dict[str, object]:
    mapped_obs_keys: list[str] = []
    seen: set[str] = set()
    for canonical_key in selected_obs_keys:
        source_key = str(alias_map.get(canonical_key, canonical_key)).strip()
        if not source_key or source_key in seen:
            continue
        seen.add(source_key)
        mapped_obs_keys.append(source_key)

    return resolve_obs_spec({"obs_keys": mapped_obs_keys})


def _read_manifest_modalities(*, data_root: Path, datahouse_id: str, view_id: str) -> set[str]:
    manifest_path = data_root / datahouse_id / "views" / view_id / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest 不存在: {manifest_path}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"manifest 内容必须为对象: {manifest_path}")

    spec = payload.get("spec")
    if not isinstance(spec, Mapping):
        raise ValueError(f"manifest.spec 缺失或格式错误: {manifest_path}")

    modalities = spec.get("modalities")
    if not isinstance(modalities, Mapping):
        raise ValueError(f"manifest.spec.modalities 缺失或格式错误: {manifest_path}")

    keys = {
        str(key).strip()
        for key in modalities.keys()
        if str(key).strip()
    }
    if not keys:
        raise ValueError(f"manifest.spec.modalities 为空: {manifest_path}")
    return keys


def _normalize_alias_map(raw_alias_map: object) -> Dict[str, str]:
    if raw_alias_map is None:
        return {}

    alias_mapping = _to_mapping(raw_alias_map)
    alias_map: Dict[str, str] = {}
    for raw_canonical_key, raw_source_key in alias_mapping.items():
        canonical_key = _non_empty(str(raw_canonical_key), "alias_map canonical key")
        source_key = _non_empty(str(raw_source_key), f"alias_map[{canonical_key}]")
        alias_map[canonical_key] = source_key
    return alias_map


def _ensure_unique_mapped_keys(keys: Sequence[str], alias_map: Mapping[str, str], label: str) -> None:
    reverse: Dict[str, str] = {}
    for canonical_key in keys:
        source_key = alias_map.get(canonical_key, canonical_key)
        previous = reverse.get(source_key)
        if previous is None:
            reverse[source_key] = canonical_key
            continue
        if previous != canonical_key:
            raise ValueError(
                f"{label} 存在冲突映射: {previous} 与 {canonical_key} -> {source_key}"
            )


def _to_mapping(raw_value: object) -> Mapping[str, Any]:
    if isinstance(raw_value, Mapping):
        return raw_value
    if hasattr(raw_value, "items") and callable(getattr(raw_value, "items")):
        return dict(raw_value.items())
    raise TypeError("配置项必须是对象")


def _clean_keys(values: Any, name: str, allow_empty: bool = False) -> list[str]:
    if values is None:
        values = []

    if isinstance(values, (str, bytes)):
        values = [values]

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


def _ensure_subset(values: Sequence[str], candidates: Sequence[str], name: str) -> None:
    candidate_set = set(candidates)
    extras = sorted(value for value in values if value not in candidate_set)
    if extras:
        raise ValueError(f"{name} 包含未知 key: {', '.join(extras)}")


def _non_empty(value: str, name: str) -> str:
    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError(f"{name} 不能为空")
    return cleaned


def _make_raw_zeros(*, steps: int, shape: Sequence[int]) -> np.ndarray:
    tail_shape = tuple(int(v) for v in shape)
    return np.zeros((int(steps),) + tail_shape, dtype=np.float32)


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


__all__ = ["MultiViewDataset"]
