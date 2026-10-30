from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

_REQUIRES_KEY = "_requires_"


def resolve_config_profiles(
    config: DictConfig,
    *,
    overrides: DictConfig | None = None,
) -> DictConfig:
    """Resolve selector-driven overlays and return an executable config."""

    if not isinstance(config, DictConfig):
        raise TypeError("profiled config must contain a mapping")
    if "profiles" not in config:
        return OmegaConf.merge(config, overrides) if overrides else config

    order = _profile_order(config)
    candidate = OmegaConf.merge(config, overrides) if overrides else config
    selectors = _resolved_selectors(candidate, order)
    profiles = config.profiles
    if not isinstance(profiles, DictConfig):
        raise TypeError("profiles must contain a mapping")

    result = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    del result["profiles"]
    if "profile_order" in result:
        del result["profile_order"]

    for axis in order:
        group = profiles.get(axis)
        if not isinstance(group, DictConfig):
            raise ValueError(f"profiles is missing selector group {axis!r}")
        selected = selectors[axis]
        if selected not in group:
            choices = ", ".join(sorted(str(key) for key in group))
            raise ValueError(
                f"Unsupported {axis} selector {selected!r}; expected one of: {choices}"
            )
        profile = OmegaConf.create(
            OmegaConf.to_container(group[selected], resolve=False)
        )
        requirements = profile.pop(_REQUIRES_KEY, None)
        _validate_requirements(
            axis=axis,
            selected=selected,
            requirements=requirements,
            selectors=selectors,
        )
        result = OmegaConf.merge(result, profile)

    result.selectors = OmegaConf.create(selectors)
    if overrides:
        result = OmegaConf.merge(result, overrides)
    return result


def _profile_order(config: DictConfig) -> tuple[str, ...]:
    raw = config.get("profile_order")
    if not isinstance(raw, (list, tuple, ListConfig)) or not raw:
        raise ValueError("profile_order must be a non-empty selector list")
    order = tuple(str(value) for value in raw)
    if len(set(order)) != len(order):
        raise ValueError("profile_order must not contain duplicate selectors")
    return order


def _resolved_selectors(
    config: DictConfig,
    order: tuple[str, ...],
) -> dict[str, str]:
    raw = config.get("selectors")
    if not isinstance(raw, DictConfig):
        raise TypeError("selectors must contain a mapping")
    resolved = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(resolved, Mapping):
        raise TypeError("selectors must resolve to a mapping")
    result = {str(key): str(value) for key, value in resolved.items()}
    missing = [axis for axis in order if not result.get(axis)]
    if missing:
        raise ValueError(f"Missing config selector(s): {', '.join(missing)}")
    return result


def _validate_requirements(
    *,
    axis: str,
    selected: str,
    requirements: Any,
    selectors: Mapping[str, str],
) -> None:
    if requirements is None:
        return
    resolved = OmegaConf.to_container(requirements, resolve=True)
    if not isinstance(resolved, Mapping):
        raise TypeError(f"profiles.{axis}.{selected}.{_REQUIRES_KEY} must be a mapping")
    for required_axis, allowed_values in resolved.items():
        if isinstance(allowed_values, str):
            allowed = {allowed_values}
        elif isinstance(allowed_values, (list, tuple, ListConfig)):
            allowed = {str(value) for value in allowed_values}
        else:
            raise TypeError(
                f"profiles.{axis}.{selected}.{_REQUIRES_KEY}.{required_axis} "
                "must be a string or list"
            )
        actual = selectors.get(str(required_axis))
        if actual not in allowed:
            choices = ", ".join(sorted(allowed))
            raise ValueError(
                f"{axis}={selected!r} requires {required_axis} in "
                f"[{choices}], got {actual!r}"
            )
