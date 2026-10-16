from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch

from lerobot.policies.names import WSA_LARGE, WSA_LARGE_LEGACY_ALIASES

from .core.utils.logging_config import get_logger


logger = get_logger(__name__)


def _is_field_stats(value: Any) -> bool:
    return isinstance(value, dict) and any(key in value for key in ("mean", "std", "min", "max", "q01", "q99"))


def _is_wsa_large_group(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    for child in value.values():
        if isinstance(child, dict) and any(
            key.startswith("global_") or key.startswith("stepwise_") for key in child
        ):
            return True
    return False


def _is_wsa_large_stats_payload(stats: Any) -> bool:
    return isinstance(stats, dict) and _is_wsa_large_group(stats.get("action"))


def _clone_stat_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone().float()
    if isinstance(value, np.ndarray):
        return torch.as_tensor(value).clone().float()
    if isinstance(value, (list, tuple)) and _is_numeric_sequence(value):
        return torch.as_tensor(value).clone().float()
    if isinstance(value, dict):
        return {key: _clone_stat_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_clone_stat_value(child) for child in value]
    return value


def _is_numeric_sequence(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        if not value:
            return True
        return all(_is_numeric_sequence(child) for child in value)
    return isinstance(value, (int, float, bool, np.number))


def _field_stats_to_wsa_large_stats(field_stats: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for stat_name in ("min", "max", "mean", "std", "q01", "q99"):
        if stat_name in field_stats:
            value = _clone_stat_value(field_stats[stat_name])
            out[f"global_{stat_name}"] = value
            # Framework stats are global. Providing the stepwise keys as the
            # same global values keeps the payload loadable, but official
            # RobotWin uses global normalization.
            out[f"stepwise_{stat_name}"] = _clone_stat_value(value)
    return out


def _candidate_framework_stat_keys(group: str, meta: dict[str, Any]) -> list[str]:
    key = str(meta.get("key", "default"))
    candidates = []
    lerobot_key = meta.get("lerobot_key")
    if lerobot_key is not None:
        candidates.append(str(lerobot_key))

    if group == "action":
        if key == "default":
            candidates.append("action")
        candidates.extend([f"action.{key}", key])
    elif group == "state":
        if key == "default":
            candidates.append("observation.state")
        candidates.extend([f"observation.state.{key}", key])

    deduped = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _convert_framework_group_stats(
    stats_payload: dict[str, Any],
    shape_meta: dict[str, Any] | None,
    group: Literal["action", "state"],
    *,
    required: bool,
) -> dict[str, Any]:
    grouped_stats = stats_payload.get(group)
    if isinstance(grouped_stats, dict) and not _is_field_stats(grouped_stats):
        keyed_group = {
            str(key): _field_stats_to_wsa_large_stats(value)
            for key, value in grouped_stats.items()
            if _is_field_stats(value)
        }
        if keyed_group and shape_meta is None:
            return keyed_group

    if shape_meta is None:
        if group == "action":
            for candidate in ("action", "action.default", "default"):
                if _is_field_stats(stats_payload.get(candidate)):
                    return {"default": _field_stats_to_wsa_large_stats(stats_payload[candidate])}
        if not required:
            return {}
        raise ValueError(
            f"Cannot adapt framework stats to WSA_Large format for `{group}` without shape_meta."
        )

    converted: dict[str, Any] = {}
    missing: list[str] = []
    for meta in shape_meta.get(group, []):
        key = str(meta["key"])
        selected_stats = None
        selected_source = None
        for candidate in _candidate_framework_stat_keys(group, meta):
            candidate_stats = stats_payload.get(candidate)
            if candidate_stats is None and isinstance(grouped_stats, dict):
                candidate_stats = grouped_stats.get(key)
            if _is_field_stats(candidate_stats):
                selected_stats = candidate_stats
                selected_source = candidate
                break
        if selected_stats is None:
            missing.append(key)
            continue
        converted[key] = _field_stats_to_wsa_large_stats(selected_stats)
        logger.info(
            "Adapted framework normalization stats key %s -> WSA_Large %s/%s.",
            selected_source,
            group,
            key,
        )

    if missing and required:
        raise ValueError(
            f"Missing framework stats for WSA_Large `{group}` keys {missing}. "
            f"Available stats keys: {list(stats_payload.keys())}"
        )
    return converted


def ensure_wsa_large_stats_format(
    stats_payload: dict[str, Any],
    shape_meta: dict[str, Any] | None = None,
    *,
    require_state: bool = True,
) -> dict[str, Any]:
    """Accept WSA_Large stats or framework flat stats and return WSA_Large stats."""
    for stats_key in (WSA_LARGE, "wsa_large", *WSA_LARGE_LEGACY_ALIASES):
        if stats_key in stats_payload:
            stats_payload = stats_payload[stats_key]
            break

    if _is_wsa_large_stats_payload(stats_payload):
        return stats_payload

    converted = {
        "action": _convert_framework_group_stats(
            stats_payload,
            shape_meta,
            "action",
            required=True,
        ),
    }
    state_stats = _convert_framework_group_stats(
        stats_payload,
        shape_meta,
        "state",
        required=require_state,
    )
    if state_stats:
        converted["state"] = state_stats
    for key in ("num_episodes", "num_transition"):
        if key in stats_payload:
            converted[key] = _clone_stat_value(stats_payload[key])
    return converted
