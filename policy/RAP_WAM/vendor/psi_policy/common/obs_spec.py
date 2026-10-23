from __future__ import annotations

from typing import Any, Iterable, Mapping


def resolve_obs_spec(obs_cfg: Any) -> dict[str, Any]:
    """将 obs_spec 解析为统一结构（不做语义校验）。

    - 支持显式 `obs_keys`。
    - 自动解析所有以 `_keys` 结尾的分组（如 `rgb_keys`/`low_dim_keys`/`language_keys`）。
    - 仅做清洗与去重，语义合法性由 dataset/encoder 自行校验。
    """

    explicit_obs_keys = _extract_obs_keys(obs_cfg)
    groups: dict[str, list[str]] = {}
    for group_name, raw_values in _iter_key_groups(obs_cfg):
        groups[group_name] = _clean_keys(raw_values)

    # 兼容常见分组，缺省时提供空列表，便于调用方直接读取。
    groups.setdefault("rgb_keys", [])
    groups.setdefault("low_dim_keys", [])

    obs_keys = _merge_keys(explicit_obs_keys, *(groups[name] for name in groups))

    resolved: dict[str, Any] = {
        "obs_keys": obs_keys,
        "groups": {name: list(values) for name, values in groups.items()},
    }
    for name, values in groups.items():
        resolved[name] = list(values)

    return resolved


def _extract_obs_keys(obs_cfg: Any) -> list[str]:
    if obs_cfg is None:
        return []

    raw_values: Any | None = None
    if isinstance(obs_cfg, Mapping):
        raw_values = obs_cfg.get("obs_keys")
    elif hasattr(obs_cfg, "items") and callable(getattr(obs_cfg, "items")):
        try:
            raw_values = obs_cfg.get("obs_keys")  # type: ignore[attr-defined]
        except Exception:
            raw_values = None
    elif hasattr(obs_cfg, "obs_keys"):
        try:
            raw_values = getattr(obs_cfg, "obs_keys")
        except Exception:
            raw_values = None

    return _clean_keys(raw_values)


def _iter_key_groups(obs_cfg: Any) -> Iterable[tuple[str, Any]]:
    if obs_cfg is None:
        return []

    items: Iterable[tuple[Any, Any]]
    if isinstance(obs_cfg, Mapping):
        items = obs_cfg.items()
    elif hasattr(obs_cfg, "items") and callable(getattr(obs_cfg, "items")):
        try:
            items = obs_cfg.items()  # type: ignore[assignment]
        except Exception:
            items = []
    else:
        dynamic_items: list[tuple[str, Any]] = []
        for name in dir(obs_cfg):
            if not str(name).endswith("_keys") or str(name).startswith("_") or str(name) == "obs_keys":
                continue
            try:
                value = getattr(obs_cfg, name)
            except Exception:
                continue
            if callable(value):
                continue
            dynamic_items.append((str(name), value))
        items = dynamic_items

    pairs: list[tuple[str, Any]] = []
    for raw_name, value in items:
        name = str(raw_name).strip()
        if not name.endswith("_keys") or name == "obs_keys":
            continue
        pairs.append((name, value))
    return pairs


def _clean_keys(raw_values: Any) -> list[str]:
    if raw_values is None:
        return []

    if isinstance(raw_values, (str, bytes)):
        candidates = [raw_values]
    else:
        try:
            candidates = list(iter(raw_values))
        except TypeError:
            candidates = [raw_values]

    cleaned: list[str] = []
    seen = set()
    for raw in candidates:
        key = str(raw).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        cleaned.append(key)
    return cleaned


def _merge_keys(*groups: list[str]) -> list[str]:
    merged: list[str] = []
    seen = set()
    for group in groups:
        for key in group:
            if key in seen:
                continue
            seen.add(key)
            merged.append(key)
    return merged


__all__ = ["resolve_obs_spec"]
