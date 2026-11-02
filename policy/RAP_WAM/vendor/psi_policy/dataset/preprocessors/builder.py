from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

import hydra

from psi_policy.dataset.preprocessors.base import BaseObsPreprocessor
from psi_policy.dataset.preprocessors.passthrough import PassthroughPreprocessor


def build_obs_preprocessors(
    cfg: Any,
    *,
    obs_keys: Sequence[str],
    default_image_size: Sequence[int] | None = None,
) -> dict[str, BaseObsPreprocessor]:
    known_keys = _clean_keys(obs_keys)
    raw_cfg = _as_mapping_or_empty(cfg, field_name="dataset.preprocessors")

    unknown = sorted(key for key in raw_cfg.keys() if str(key).strip() not in set(known_keys))
    if unknown:
        raise ValueError("dataset.preprocessors 包含未知 obs key: " + ", ".join(unknown))

    preprocessors: dict[str, BaseObsPreprocessor] = {}
    for key in known_keys:
        raw_item = raw_cfg.get(key)
        if raw_item is None:
            preprocessors[key] = PassthroughPreprocessor()
            continue
        preprocessors[key] = _instantiate_preprocessor(
            raw_item,
            default_image_size=default_image_size,
            field_name=f"dataset.preprocessors[{key}]",
        )
    return preprocessors


def remap_preprocessor_cfgs(
    cfg: Any,
    *,
    selected_obs_keys: Sequence[str],
    alias_map: Mapping[str, str],
) -> dict[str, Any]:
    raw_cfg = _as_mapping_or_empty(cfg, field_name="dataset.preprocessors")
    remapped: dict[str, Any] = {}
    for canonical_key in _clean_keys(selected_obs_keys):
        if canonical_key not in raw_cfg:
            continue
        source_key = str(alias_map.get(canonical_key, canonical_key)).strip()
        if not source_key:
            raise ValueError(f"dataset.preprocessors[{canonical_key}] 映射后的 source key 为空")
        remapped[source_key] = copy.deepcopy(raw_cfg[canonical_key])
    return remapped


def _instantiate_preprocessor(
    raw_item: Any,
    *,
    default_image_size: Sequence[int] | None,
    field_name: str,
) -> BaseObsPreprocessor:
    if isinstance(raw_item, BaseObsPreprocessor):
        return raw_item

    errors: list[Exception] = []
    if default_image_size is not None:
        try:
            instance = hydra.utils.instantiate(raw_item, default_image_size=tuple(default_image_size))
            if not isinstance(instance, BaseObsPreprocessor):
                raise TypeError(f"{field_name} 实例类型非法: {type(instance)}")
            return instance
        except Exception as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            errors.append(exc)

    try:
        instance = hydra.utils.instantiate(raw_item)
    except Exception as exc:
        details = "\n".join(str(err) for err in errors)
        raise RuntimeError(f"实例化 {field_name} 失败：{exc}\n{details}") from exc

    if not isinstance(instance, BaseObsPreprocessor):
        raise TypeError(f"{field_name} 实例类型非法: {type(instance)}")
    return instance


def _as_mapping_or_empty(raw: Any, *, field_name: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return {str(key).strip(): value for key, value in raw.items()}
    if hasattr(raw, "items") and callable(getattr(raw, "items")):
        try:
            return {str(key).strip(): value for key, value in raw.items()}  # type: ignore[call-arg]
        except Exception as exc:
            raise TypeError(f"{field_name} 必须是映射") from exc
    raise TypeError(f"{field_name} 必须是映射")


def _clean_keys(values: Sequence[str]) -> list[str]:
    cleaned: list[str] = []
    seen = set()
    for raw in values:
        key = str(raw).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        cleaned.append(key)
    return cleaned


__all__ = ["build_obs_preprocessors", "remap_preprocessor_cfgs"]
