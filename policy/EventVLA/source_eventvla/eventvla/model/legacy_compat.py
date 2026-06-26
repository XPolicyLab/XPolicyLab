"""Compatibility helpers for legacy starVLA checkpoints.

This module is intentionally lightweight: it must be importable from the
RoboTwin eval environment, which doesn't include the model-server stack.
"""

from __future__ import annotations

from typing import Any

from eventvla.model.memory_ablation import (
    KEYFRAME_IMAGE_MEMORY_MODES,
    MEMORY_WRITE_POLICY_DISABLED,
    validate_temporal_image_profile,
)


PUBLIC_FRAMEWORK_NAME = "EventVLA"
LEGACY_QWENOFT_FRAMEWORK_NAME = "QwenOFT"
PURE_IMAGE_KEYFRAME_MEMORY_MODE = "pure_image_keyframe_memory"
RAW_ANCHORS_ONLY_MODE = "raw_anchors_only"
SUPPORTED_LEGACY_NON_TOKEN_MODES = {
    PURE_IMAGE_KEYFRAME_MEMORY_MODE,
    RAW_ANCHORS_ONLY_MODE,
}

_TOKEN_MEMORY_MODES = {
    "raw_anchors_token_memory",
    "memory_tokens_only",
    "replace_image_tokens",
}


def _cfg_get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    get_method = getattr(node, "get", None)
    if callable(get_method):
        sentinel = object()
        try:
            value = get_method(key, sentinel)
        except TypeError:
            value = sentinel
        if value is not sentinel:
            return value
    return getattr(node, key, default)


def _cfg_set(node: Any, key: str, value: Any) -> None:
    if isinstance(node, dict):
        node[key] = value
        return
    try:
        node[key] = value
    except Exception:
        setattr(node, key, value)


def _cfg_ensure_path(root: Any, path: tuple[str, ...]) -> Any:
    current = root
    for key in path:
        child = _cfg_get(current, key, None)
        if child is None:
            child = {}
            _cfg_set(current, key, child)
        current = child
    return current


def _cfg_set_path(root: Any, path: tuple[str, ...], value: Any) -> None:
    parent = _cfg_ensure_path(root, path[:-1])
    _cfg_set(parent, path[-1], value)


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_mode(value: Any) -> str:
    return _normalize_text(value).lower()


def _cfg_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off", "none", "null"}
    return bool(value)


def _legacy_error(message: str) -> ValueError:
    return ValueError(
        "Legacy QwenOFT checkpoint compatibility only supports "
        "non-token modes pure_image_keyframe_memory or raw_anchors_only "
        "with memory_buffer.enable=false. "
        f"{message}"
    )


def _legacy_qwenoft_info(enabled: bool, reason: str = "") -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "source_framework_name": LEGACY_QWENOFT_FRAMEWORK_NAME if enabled else None,
        "target_framework_name": PUBLIC_FRAMEWORK_NAME if enabled else None,
        "reason": reason,
    }


def normalize_legacy_checkpoint_config(model_config: Any) -> tuple[Any, dict[str, Any]]:
    """Map supported legacy starVLA checkpoint configs onto the public EventVLA path.

    This intentionally accepts only non-token legacy QwenOFT raw-image routes.
    Token-memory and other ablation modes stay unsupported in EventVLA.
    """

    framework_cfg = _cfg_get(model_config, "framework", None)
    framework_name = _normalize_text(_cfg_get(framework_cfg, "name", ""))
    if framework_name != LEGACY_QWENOFT_FRAMEWORK_NAME:
        return model_config, _legacy_qwenoft_info(False)

    memory_cfg = _cfg_get(framework_cfg, "memory_buffer", {}) or {}
    injection_cfg = _cfg_get(memory_cfg, "qwen_memory_injection", {}) or {}
    configured_mode = _normalize_mode(_cfg_get(framework_cfg, "memory_ablation_mode", ""))
    injection_mode = _normalize_mode(_cfg_get(injection_cfg, "mode", ""))
    mode = configured_mode or injection_mode

    if configured_mode in _TOKEN_MEMORY_MODES or injection_mode in _TOKEN_MEMORY_MODES:
        raise _legacy_error(
            f"Got token-memory mode framework.memory_ablation_mode={configured_mode!r}, "
            f"qwen_memory_injection.mode={injection_mode!r}."
        )
    if mode not in SUPPORTED_LEGACY_NON_TOKEN_MODES:
        raise _legacy_error(
            f"Got framework.memory_ablation_mode={configured_mode!r}, "
            f"qwen_memory_injection.mode={injection_mode!r}."
        )
    if injection_mode and injection_mode != mode:
        raise _legacy_error(f"Got qwen_memory_injection.mode={injection_mode!r}.")

    memory_enabled = _cfg_bool(_cfg_get(memory_cfg, "enable", False), False)
    if memory_enabled:
        raise _legacy_error("Got framework.memory_buffer.enable=true.")

    datasets_cfg = _cfg_get(model_config, "datasets", {}) or {}
    vla_data_cfg = _cfg_get(datasets_cfg, "vla_data", {}) or {}
    keyframe_image_cfg = _cfg_get(vla_data_cfg, "keyframe_image_memory", {}) or {}
    expects_keyframe_images = mode in KEYFRAME_IMAGE_MEMORY_MODES
    keyframe_images_enabled = _cfg_bool(
        _cfg_get(keyframe_image_cfg, "enabled", expects_keyframe_images),
        expects_keyframe_images,
    )
    if keyframe_images_enabled != expects_keyframe_images:
        raise _legacy_error(
            "Got datasets.vla_data.keyframe_image_memory.enabled="
            f"{keyframe_images_enabled!r} for mode={mode!r}."
        )

    provide_teacher_commit_images = _cfg_bool(
        _cfg_get(vla_data_cfg, "provide_teacher_commit_images", False),
        False,
    )
    if provide_teacher_commit_images:
        raise _legacy_error("Got datasets.vla_data.provide_teacher_commit_images=true.")

    temporal_cfg = _cfg_get(vla_data_cfg, "temporal", {}) or {}
    temporal_image_cfg = _cfg_get(temporal_cfg, "image", {}) or {}
    validate_temporal_image_profile(
        mode=mode,
        absolute_indices=_cfg_get(temporal_image_cfg, "absolute_indices", []),
        delta_indices=_cfg_get(temporal_image_cfg, "delta_indices", [0]),
        source="legacy QwenOFT checkpoint temporal image config",
    )

    _cfg_set(framework_cfg, "legacy_source_name", framework_name)
    _cfg_set(framework_cfg, "compat_loaded_as", PUBLIC_FRAMEWORK_NAME)
    _cfg_set(framework_cfg, "name", PUBLIC_FRAMEWORK_NAME)
    _cfg_set_path(model_config, ("framework", "memory_ablation_mode"), mode)
    _cfg_set_path(model_config, ("framework", "memory_buffer", "enable"), False)
    _cfg_set_path(model_config, ("framework", "memory_buffer", "memory_write_policy"), MEMORY_WRITE_POLICY_DISABLED)
    _cfg_set_path(model_config, ("framework", "memory_buffer", "qwen_memory_injection", "enabled"), True)
    _cfg_set_path(
        model_config,
        ("framework", "memory_buffer", "qwen_memory_injection", "mode"),
        mode,
    )
    _cfg_set_path(model_config, ("framework", "memory_buffer", "force_memory_write_current_for_event_commit"), False)
    _cfg_set_path(model_config, ("framework", "memory_buffer", "disable_current_frame_keyframe_write_in_eval"), True)
    _cfg_set_path(model_config, ("framework", "memory_buffer", "use_teacher_future_frame_write_in_train"), False)
    _cfg_set_path(model_config, ("datasets", "vla_data", "provide_teacher_commit_images"), False)
    _cfg_set_path(
        model_config,
        ("datasets", "vla_data", "keyframe_image_memory", "enabled"),
        expects_keyframe_images,
    )

    return model_config, _legacy_qwenoft_info(
        True,
        f"mapped legacy QwenOFT {mode} checkpoint to EventVLA",
    )


def normalize_legacy_framework_config(cfg: Any) -> dict[str, Any]:
    """Normalize a framework config object in-place for direct build_framework calls."""

    _, compat_info = normalize_legacy_checkpoint_config(cfg)
    return compat_info
