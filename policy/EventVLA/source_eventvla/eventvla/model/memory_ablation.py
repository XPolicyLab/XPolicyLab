from __future__ import annotations

from dataclasses import dataclass


CANONICAL_MEMORY_ABLATION_MODES = (
    "pure_image_keyframe_memory",
    "raw_anchors_only",
)

KEYFRAME_IMAGE_MEMORY_MODES = frozenset({"pure_image_keyframe_memory"})
TOKEN_MEMORY_MODES = frozenset()
MEMORYLESS_MODES = frozenset({"raw_anchors_only"})
CURRENT_FRAME_ONLY_MEMORYLESS_MODES = frozenset()

MEMORY_WRITE_POLICY_DISABLED = "disabled"
MEMORY_WRITE_POLICY_TEACHER_FUTURE_PLUS_CURRENT = "teacher_future_plus_current"

DEFAULT_TEMPORAL_ABSOLUTE_INDICES = (0,)
DEFAULT_TEMPORAL_ANCHOR_DELTAS = (-30, -15, 0)


@dataclass(frozen=True)
class MemoryAblationProfile:
    name: str
    qwen_memory_injection_mode: str
    enable_memory_buffer: bool
    enable_keyframe_image_memory: bool
    provide_teacher_commit_images: bool
    force_memory_write_current_for_event_commit: bool
    disable_current_frame_keyframe_write_in_eval: bool
    use_teacher_future_frame_write_in_train: bool
    memory_write_policy: str
    temporal_absolute_indices: tuple[int, ...]
    temporal_anchor_deltas: tuple[int, ...]


_PROFILE_BY_MODE = {
    "pure_image_keyframe_memory": MemoryAblationProfile(
        name="pure_image_keyframe_memory",
        qwen_memory_injection_mode="pure_image_keyframe_memory",
        enable_memory_buffer=False,
        enable_keyframe_image_memory=True,
        provide_teacher_commit_images=False,
        force_memory_write_current_for_event_commit=False,
        disable_current_frame_keyframe_write_in_eval=True,
        use_teacher_future_frame_write_in_train=False,
        memory_write_policy=MEMORY_WRITE_POLICY_DISABLED,
        temporal_absolute_indices=DEFAULT_TEMPORAL_ABSOLUTE_INDICES,
        temporal_anchor_deltas=DEFAULT_TEMPORAL_ANCHOR_DELTAS,
    ),
    "raw_anchors_only": MemoryAblationProfile(
        name="raw_anchors_only",
        qwen_memory_injection_mode="raw_anchors_only",
        enable_memory_buffer=False,
        enable_keyframe_image_memory=False,
        provide_teacher_commit_images=False,
        force_memory_write_current_for_event_commit=False,
        disable_current_frame_keyframe_write_in_eval=True,
        use_teacher_future_frame_write_in_train=False,
        memory_write_policy=MEMORY_WRITE_POLICY_DISABLED,
        temporal_absolute_indices=DEFAULT_TEMPORAL_ABSOLUTE_INDICES,
        temporal_anchor_deltas=DEFAULT_TEMPORAL_ANCHOR_DELTAS,
    ),
}


def _normalize_mode(mode: object) -> str:
    return str(mode or "").strip().lower()


def _cfg_get(node, key: str, default=None):
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    get_method = getattr(node, "get", None)
    if callable(get_method):
        sentinel = object()
        value = get_method(key, sentinel)
        if value is not sentinel:
            return value
    return getattr(node, key, default)


def _cfg_set(node, key: str, value) -> None:
    if isinstance(node, dict):
        node[key] = value
        return
    setattr(node, key, value)


def _cfg_ensure_path(root, path: tuple[str, ...]):
    current = root
    for key in path:
        child = _cfg_get(current, key, None)
        if child is None:
            _cfg_set(current, key, {})
            child = _cfg_get(current, key, None)
        current = child
    return current


def _cfg_set_path(root, path: tuple[str, ...], value) -> None:
    parent = _cfg_ensure_path(root, path[:-1])
    _cfg_set(parent, path[-1], value)


def get_memory_ablation_profile(mode: object) -> MemoryAblationProfile:
    resolved_mode = _normalize_mode(mode)
    if resolved_mode not in _PROFILE_BY_MODE:
        supported_modes = ", ".join(CANONICAL_MEMORY_ABLATION_MODES)
        raise ValueError(
            f"Unsupported framework.memory_ablation_mode `{resolved_mode}`. "
            f"EventVLA only supports: {supported_modes}."
        )
    return _PROFILE_BY_MODE[resolved_mode]


def _normalize_temporal_index_sequence(values) -> tuple[int, ...]:
    if values is None:
        return ()
    return tuple(int(value) for value in values)


def validate_temporal_image_profile(
    mode: object,
    absolute_indices,
    delta_indices,
    source: str = "datasets.vla_data.temporal.image",
) -> None:
    profile = get_memory_ablation_profile(mode)
    resolved_absolute_indices = _normalize_temporal_index_sequence(absolute_indices)
    resolved_delta_indices = _normalize_temporal_index_sequence(delta_indices)
    if (
        resolved_absolute_indices != profile.temporal_absolute_indices
        or resolved_delta_indices != profile.temporal_anchor_deltas
    ):
        raise ValueError(
            "Resolved temporal image config mismatch: "
            f"{source}.absolute_indices={list(resolved_absolute_indices)} and "
            f"{source}.delta_indices={list(resolved_delta_indices)} but "
            f"mode={profile.name} expects absolute_indices={list(profile.temporal_absolute_indices)} "
            f"and delta_indices={list(profile.temporal_anchor_deltas)}."
        )


def resolve_memory_ablation_profile(cfg) -> MemoryAblationProfile:
    framework_cfg = _cfg_get(cfg, "framework", None)
    configured_mode = _cfg_get(framework_cfg, "memory_ablation_mode", None)
    if configured_mode is None:
        raise ValueError(
            "Missing `framework.memory_ablation_mode`. "
            "EventVLA requires one of: "
            f"{', '.join(CANONICAL_MEMORY_ABLATION_MODES)}."
        )
    return get_memory_ablation_profile(configured_mode)


def apply_memory_ablation_profile(cfg, profile: MemoryAblationProfile):
    _cfg_set_path(cfg, ("framework", "memory_ablation_mode"), profile.name)
    _cfg_set_path(cfg, ("framework", "memory_buffer", "enable"), profile.enable_memory_buffer)
    _cfg_set_path(cfg, ("framework", "memory_buffer", "memory_write_policy"), profile.memory_write_policy)
    _cfg_set_path(cfg, ("framework", "memory_buffer", "qwen_memory_injection", "enabled"), True)
    _cfg_set_path(
        cfg,
        ("framework", "memory_buffer", "qwen_memory_injection", "mode"),
        profile.qwen_memory_injection_mode,
    )
    _cfg_set_path(
        cfg,
        ("framework", "memory_buffer", "force_memory_write_current_for_event_commit"),
        profile.force_memory_write_current_for_event_commit,
    )
    _cfg_set_path(
        cfg,
        ("framework", "memory_buffer", "disable_current_frame_keyframe_write_in_eval"),
        profile.disable_current_frame_keyframe_write_in_eval,
    )
    _cfg_set_path(
        cfg,
        ("framework", "memory_buffer", "use_teacher_future_frame_write_in_train"),
        profile.use_teacher_future_frame_write_in_train,
    )
    _cfg_set_path(
        cfg,
        ("datasets", "vla_data", "provide_teacher_commit_images"),
        profile.provide_teacher_commit_images,
    )
    _cfg_set_path(
        cfg,
        ("datasets", "vla_data", "keyframe_image_memory", "enabled"),
        profile.enable_keyframe_image_memory,
    )
    _cfg_set_path(cfg, ("datasets", "vla_data", "temporal", "enabled"), True)
    _cfg_set_path(cfg, ("datasets", "vla_data", "temporal", "image", "include_current"), True)
    _cfg_set_path(
        cfg,
        ("datasets", "vla_data", "temporal", "image", "absolute_indices"),
        list(profile.temporal_absolute_indices),
    )
    _cfg_set_path(
        cfg,
        ("datasets", "vla_data", "temporal", "image", "delta_indices"),
        list(profile.temporal_anchor_deltas),
    )
    return cfg


def resolve_and_apply_memory_ablation_profile(cfg) -> MemoryAblationProfile:
    profile = resolve_memory_ablation_profile(cfg)
    apply_memory_ablation_profile(cfg, profile)
    return profile
