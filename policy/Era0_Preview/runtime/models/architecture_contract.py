"""Resolved architecture-path contract for X-VLA checkpoints.

Ported from the x-vla training repository so Hy-Embodied can load and
fine-tune checkpoints written by the ``vlm_fast_action_bridge`` (G0.5
prefix-KV) pretraining path. The legacy fused-MHA Florence/Qwen families
remain the default: a checkpoint that omits every field below resolves to
``legacy`` and is byte-for-byte unchanged.

This module is deliberately dependency-free so ``XVLAConfig`` can import it
without pulling in torch.
"""

from __future__ import annotations

from typing import Any, Mapping


# VLA action-path language token cap shared by every wide contract. Florence
# keeps the narrower 50 (``processing_xvla.DEFAULT_ACTION_LANGUAGE_MAX_LENGTH``)
# because its checkpoints are baked at that value.
WIDE_ACTION_LANGUAGE_MAX_LENGTH = 128

ARCHITECTURE_PATHS = (
    "legacy",
    "vlm_kv_reader",
    "vlm_fast_action_bridge",
)
ARCHITECTURE_PATH_ALIASES = {
    "conditioned_reader": "vlm_kv_reader",
    "matched_dual_stream": "vlm_fast_action_bridge",
    "figure18_joint_mask": "legacy",
}

# action Q reads frozen VLM K/V (was conditioned_reader)
VLM_KV_READER_VERSION = "xvla_vlm_kv_reader_v1"
VLM_KV_READER_VERSIONS = frozenset(
    {
        VLM_KV_READER_VERSION,
        "xvla_conditioned_reader_v1",
    }
)
# prefix / FAST / action bridge; Qwen uses native prefix KV
VLM_FAST_ACTION_BRIDGE_VERSION = "vlm_fast_action_bridge"
VLM_FAST_ACTION_BRIDGE_VERSIONS = frozenset(
    {
        VLM_FAST_ACTION_BRIDGE_VERSION,
        "xvla_vlm_fast_action_bridge_v1",
        "xvla_matched_dual_stream_v1",
    }
)
ARCHITECTURE_VERSION_ALIASES = {
    "xvla_conditioned_reader_v1": VLM_KV_READER_VERSION,
    "xvla_vlm_fast_action_bridge_v1": VLM_FAST_ACTION_BRIDGE_VERSION,
    "xvla_matched_dual_stream_v1": VLM_FAST_ACTION_BRIDGE_VERSION,
}
LEGACY_ARCHITECTURE_VERSION = "legacy_cross_emb_v5"
ALLOWED_ARCHITECTURE_VERSIONS = frozenset(
    {
        LEGACY_ARCHITECTURE_VERSION,
        *VLM_KV_READER_VERSIONS,
        *VLM_FAST_ACTION_BRIDGE_VERSIONS,
    }
)
PREFIX_KV_ACTION_CONTEXT_MODE = "prefix_kv"
PREFIX_KV_LAYOUT_VERSION = "prefix_kv"
PREFIX_KV_FAST_VISIBILITY = "prefix_kv_isolated"
PREFIX_KV_CACHE_TOPOLOGY = "prefix_kv_cache"
PREFIX_KV_CACHE_TOPOLOGY_ALIASES = frozenset(
    {
        PREFIX_KV_CACHE_TOPOLOGY,
        "g05_prefix_kv_cache_v1",
    }
)
ACTION_CONTEXT_MODE_ALIASES = {
    "g05_prefix_kv_v1": PREFIX_KV_ACTION_CONTEXT_MODE,
}
PREFIX_LAYOUT_ALIASES = {
    "g05_prefix_kv_v1": PREFIX_KV_LAYOUT_VERSION,
}
FAST_VISIBILITY_ALIASES = {
    "g05_prefix_kv_isolated_v1": PREFIX_KV_FAST_VISIBILITY,
}
CACHE_TOPOLOGY_ALIASES = {
    "g05_prefix_kv_cache_v1": PREFIX_KV_CACHE_TOPOLOGY,
}
REMOVED_SUFFIX_ADAPTER_MESSAGE = (
    "pi05_joint_attention_mask / pi05_suffix_adapter_v0 were removed to "
    "prevent combinable architecture flags. Figure 18 visibility is built "
    "into vlm_fast_action_bridge."
)

# The G0.5 expert geometry is fixed by the Qwen full-attention layers it
# consumes K/V from, so it is a constant of the architecture rather than a
# tunable. Recorded here so both the config validator and the transformer
# constructor read one source.
PREFIX_KV_ROPE_PARTIAL_FACTOR = 0.25
PREFIX_KV_ROPE_THETA = 10_000_000.0

CONDITION_SOURCES = (
    "ground_truth",
    "generated",
    "scheduled_mixture",
)

EVAL_CONDITION_SOURCE = "generated"


def normalize_architecture_version(config_or_mapping: Any) -> str:
    return str(
        _config_value(
            config_or_mapping,
            "architecture_version",
            LEGACY_ARCHITECTURE_VERSION,
        )
        or LEGACY_ARCHITECTURE_VERSION
    ).strip().lower()


def canonicalize_architecture_path(path: str) -> str:
    normalized = str(path or "").strip().lower()
    return ARCHITECTURE_PATH_ALIASES.get(normalized, normalized)


def canonicalize_architecture_version(version: str) -> str:
    """Map legacy family aliases to the current public names.

    Load/resume keeps the serialized string; new initializers write the
    canonical name.
    """
    normalized = str(version or LEGACY_ARCHITECTURE_VERSION).strip().lower()
    return ARCHITECTURE_VERSION_ALIASES.get(normalized, normalized)


SEMANTIC_ANCHOR_POLICIES = (
    "quarter_uniform",
    "full_only",
)
SEMANTIC_ANCHOR_ALIASES = {
    "quarter_uniform_v5": "quarter_uniform",
    "full_only_v1": "full_only",
}


def canonicalize_semantic_anchor_policy(
    raw: Any,
    *,
    default: str = "quarter_uniform",
) -> str:
    """Strip version suffixes; old checkpoint strings remain load aliases."""
    policy = str(raw or default).strip().lower() or default
    policy = SEMANTIC_ANCHOR_ALIASES.get(policy, policy)
    if policy not in SEMANTIC_ANCHOR_POLICIES:
        raise ValueError(
            "semantic_anchor_policy must be quarter_uniform or full_only, "
            f"got {raw!r}."
        )
    return policy


def is_full_only_semantic_anchor(raw: Any) -> bool:
    return canonicalize_semantic_anchor_policy(raw) == "full_only"


def is_vlm_kv_reader_version(version: str) -> bool:
    return canonicalize_architecture_version(version) == VLM_KV_READER_VERSION


def is_vlm_fast_action_bridge_version(version: str) -> bool:
    return (
        canonicalize_architecture_version(version)
        == VLM_FAST_ACTION_BRIDGE_VERSION
    )


def canonicalize_action_context_mode(
    raw: Any,
    *,
    default: str = "auto",
) -> str:
    mode = str(raw or default).strip().lower() or default
    return ACTION_CONTEXT_MODE_ALIASES.get(mode, mode)


def is_prefix_kv_action_context_mode(raw: Any) -> bool:
    return canonicalize_action_context_mode(raw) == PREFIX_KV_ACTION_CONTEXT_MODE


def canonicalize_prefix_layout_version(
    raw: Any,
    *,
    default: str = "legacy_v5",
) -> str:
    version = str(raw or default).strip().lower() or default
    return PREFIX_LAYOUT_ALIASES.get(version, version)


def canonicalize_fast_visibility_policy(
    raw: Any,
    *,
    default: str = "legacy_semantic_capture_v1",
) -> str:
    policy = str(raw or default).strip().lower() or default
    return FAST_VISIBILITY_ALIASES.get(policy, policy)


def canonicalize_cache_topology(
    raw: Any,
    *,
    default: str = "legacy_joint_v1",
) -> str:
    topology = str(raw or default).strip().lower() or default
    return CACHE_TOPOLOGY_ALIASES.get(topology, topology)


def normalize_condition_source(raw: Any, *, default: str = "ground_truth") -> str:
    source = str(raw or default).strip().lower()
    if source not in CONDITION_SOURCES:
        raise ValueError(
            "pi05_condition_source must be one of "
            f"{CONDITION_SOURCES}, got {source!r}."
        )
    return source


def reject_removed_suffix_adapter(config_or_mapping: Any = None) -> None:
    """Refuse the deleted combinable suffix-adapter flag and version."""
    version = str(
        _config_value(config_or_mapping, "architecture_version", "") or ""
    ).strip().lower()
    if version == "pi05_suffix_adapter_v0":
        raise ValueError(REMOVED_SUFFIX_ADAPTER_MESSAGE)
    if bool(_config_value(config_or_mapping, "pi05_joint_attention_mask", False)):
        raise ValueError(REMOVED_SUFFIX_ADAPTER_MESSAGE)


def resolve_architecture_path(config_or_mapping: Any) -> dict[str, Any]:
    """Return the exclusive runtime path and its cache/visibility contract."""
    reject_removed_suffix_adapter(config_or_mapping)
    version = normalize_architecture_version(config_or_mapping)
    share_kv = bool(
        _config_value(
            config_or_mapping,
            "action_context_share_kv_by_semantic_depth",
            False,
        )
    )
    prefix_kv = False
    if is_vlm_fast_action_bridge_version(version):
        if share_kv:
            raise ValueError(
                "vlm_fast_action_bridge does not use read-only shared K/V."
            )
        path = "vlm_fast_action_bridge"
        prefix_kv = (
            is_prefix_kv_action_context_mode(
                _config_value(config_or_mapping, "action_context_mode", "")
            )
            or _config_value(config_or_mapping, "num_kv_heads", None) is not None
        )
        cache_topology = (
            PREFIX_KV_CACHE_TOPOLOGY
            if prefix_kv
            else "matched_bridge_kv_cache_v1"
        )
        fast_visibility = canonicalize_fast_visibility_policy(
            _config_value(
                config_or_mapping,
                "fast_visibility_policy",
                PREFIX_KV_FAST_VISIBILITY
                if prefix_kv
                else "figure18_matched_dual_stream_v1",
            )
        )
    elif is_vlm_kv_reader_version(version):
        path = "vlm_kv_reader"
        cache_topology = "readonly_reader_kv_v1"
        fast_visibility = canonicalize_fast_visibility_policy(
            _config_value(
                config_or_mapping,
                "fast_visibility_policy",
                "legacy_semantic_capture_v1",
            )
        )
    else:
        path = "legacy"
        cache_topology = (
            "readonly_reader_kv_v1" if share_kv else "legacy_joint_v1"
        )
        fast_visibility = canonicalize_fast_visibility_policy(
            _config_value(
                config_or_mapping,
                "fast_visibility_policy",
                "legacy_semantic_capture_v1",
            )
        )
    return {
        "architecture_path": path,
        "architecture_version": canonicalize_architecture_version(version),
        "cache_topology": canonicalize_cache_topology(cache_topology),
        "fast_visibility_policy": fast_visibility,
        "condition_state_version": str(
            _config_value(config_or_mapping, "condition_state_version", "none")
        ),
        "prefix_layout_version": canonicalize_prefix_layout_version(
            _config_value(config_or_mapping, "prefix_layout_version", "legacy_v5")
        ),
        "semantic_anchor_policy": canonicalize_semantic_anchor_policy(
            _config_value(
                config_or_mapping,
                "semantic_anchor_policy",
                "quarter_uniform",
            )
        ),
        "condition_source_eval": EVAL_CONDITION_SOURCE,
    }


def uses_prefix_kv(config_or_mapping: Any) -> bool:
    """True when the expert consumes native VLM prefix K/V per layer."""
    resolved = resolve_architecture_path(config_or_mapping)
    return (
        resolved["architecture_path"] == "vlm_fast_action_bridge"
        and resolved["cache_topology"] == PREFIX_KV_CACHE_TOPOLOGY
    )


def _config_value(config_or_mapping: Any, key: str, default: Any) -> Any:
    if config_or_mapping is None:
        return default
    if isinstance(config_or_mapping, Mapping):
        return config_or_mapping.get(key, default)
    return getattr(config_or_mapping, key, default)
