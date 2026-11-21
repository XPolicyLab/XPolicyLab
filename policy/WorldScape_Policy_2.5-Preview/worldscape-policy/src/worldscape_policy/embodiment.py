"""Platform-oriented embodiment identifiers and legacy checkpoint aliases."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

AGILEX = "agilex"
LIBERO = "libero"
ROBOTWIN2 = "robotwin2"

_LEGACY_TO_CANONICAL: dict[str, str] = {
    "lerobot_eef_lctx": AGILEX,
    "lerobot_eef": AGILEX,
    "agilex_eef": AGILEX,
    "eef": AGILEX,
    "robotwin": ROBOTWIN2,
}

_BUNDLE_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    AGILEX: ("lerobot_eef_lctx", "lerobot_eef", "agilex_eef", "eef"),
    ROBOTWIN2: ("robotwin",),
}


def canonical_embodiment(value: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError("embodiment must be non-empty")
    return _LEGACY_TO_CANONICAL.get(normalized, normalized)


def canonical_embodiment_tag(tag: str) -> str:
    """Deprecated alias for :func:`canonical_embodiment`."""

    return canonical_embodiment(tag)


def coalesce_embodiment(
    mapping: Mapping[str, Any] | None,
    *,
    default: str | None = None,
) -> str:
    """Read ``embodiment`` from a config mapping, falling back to legacy keys."""

    if mapping is None:
        if default is None:
            raise ValueError("embodiment is required")
        return canonical_embodiment(default)
    for key in ("embodiment", "embodiment_tag", "robot_type"):
        raw = mapping.get(key)
        if raw is not None and str(raw).strip():
            return canonical_embodiment(str(raw))
    if default is None:
        raise ValueError("embodiment is required")
    return canonical_embodiment(default)


def bundle_key_aliases(canonical: str) -> tuple[str, ...]:
    return _BUNDLE_KEY_ALIASES.get(canonical_embodiment(canonical), ())


def resolve_bundle_embodiment_key(
    embodiments: Mapping[str, Any],
    requested: str,
) -> str:
    canonical = canonical_embodiment(requested)
    if canonical in embodiments:
        return canonical
    for alias in bundle_key_aliases(canonical):
        if alias in embodiments:
            return alias
    if requested in embodiments:
        return requested
    available = ", ".join(sorted(embodiments))
    raise KeyError(
        f"Checkpoint transform bundle has no embodiment entry for {requested!r} "
        f"(canonical {canonical!r}); available: {available or '<none>'}"
    )


def expand_embodiment_ids(mapping: Mapping[str, int]) -> dict[str, int]:
    expanded: dict[str, int] = {}
    for tag, embodiment_id in mapping.items():
        if isinstance(embodiment_id, bool) or not isinstance(embodiment_id, int):
            raise TypeError(f"Embodiment ID for {tag!r} must be an integer")
        if embodiment_id < 0:
            raise ValueError("Embodiment ID must be non-negative")
        canonical = canonical_embodiment(tag)
        keys = {tag, canonical, *bundle_key_aliases(canonical)}
        for key in keys:
            if key in expanded and expanded[key] != embodiment_id:
                raise ValueError(
                    f"Conflicting embodiment IDs for {key!r}: "
                    f"{expanded[key]} != {embodiment_id}"
                )
            expanded[key] = embodiment_id
    return expanded


def is_agilex_embodiment(value: str) -> bool:
    return canonical_embodiment(value) == AGILEX


__all__ = [
    "AGILEX",
    "LIBERO",
    "ROBOTWIN2",
    "bundle_key_aliases",
    "canonical_embodiment",
    "canonical_embodiment_tag",
    "coalesce_embodiment",
    "expand_embodiment_ids",
    "is_agilex_embodiment",
    "resolve_bundle_embodiment_key",
]
