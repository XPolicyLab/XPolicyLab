from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from torch import nn

from worldscape_policy.wam.protocol import WAMPlugin

ConfigT = TypeVar("ConfigT")
WAMFactory = Callable[..., nn.Module]


@dataclass(frozen=True)
class WAMPluginMetadata:
    """Stable identity and feature declarations for a WAM implementation."""

    name: str
    version: str
    capabilities: frozenset[str]


@dataclass(frozen=True)
class WAMRegistration(Generic[ConfigT]):
    metadata: WAMPluginMetadata
    config_type: type[ConfigT]
    factory: WAMFactory


class WAMRegistry:
    """Explicit, closed registry of trusted WAM factories and config types."""

    def __init__(self) -> None:
        self._by_name: dict[str, WAMRegistration[Any]] = {}
        self._by_config: dict[type[Any], WAMRegistration[Any]] = {}
        self._sealed = False

    def register(
        self,
        *,
        name: str,
        version: str,
        capabilities: Iterable[str],
        config_type: type[ConfigT],
        factory: WAMFactory,
    ) -> None:
        if self._sealed:
            raise RuntimeError("WAM registry is sealed")
        if not name or not version:
            raise ValueError("WAM name and version must be non-empty")
        if not isinstance(config_type, type):
            raise TypeError("config_type must be a type")
        if not callable(factory):
            raise TypeError("factory must be callable")
        if name in self._by_name:
            raise ValueError(f"WAM plugin {name!r} is already registered")
        if config_type in self._by_config:
            raise ValueError(
                f"WAM config type {config_type.__name__!r} is already registered"
            )
        normalized_capabilities = frozenset(capabilities)
        if not all(isinstance(item, str) and item for item in normalized_capabilities):
            raise ValueError("WAM capabilities must be non-empty strings")
        registration = WAMRegistration(
            metadata=WAMPluginMetadata(
                name=name,
                version=version,
                capabilities=normalized_capabilities,
            ),
            config_type=config_type,
            factory=factory,
        )
        self._by_name[name] = registration
        self._by_config[config_type] = registration

    def seal(self) -> None:
        """Prevent registration changes after application startup."""

        self._sealed = True

    def get(self, name: str) -> WAMRegistration[Any]:
        try:
            return self._by_name[name]
        except KeyError as error:
            available = ", ".join(sorted(self._by_name)) or "<none>"
            raise KeyError(
                f"Unknown WAM plugin {name!r}; registered plugins: {available}"
            ) from error

    def for_config(self, config: object) -> WAMRegistration[Any]:
        config_type = type(config)
        try:
            return self._by_config[config_type]
        except KeyError as error:
            raise TypeError(
                f"Unregistered WAM config type: {config_type.__name__}"
            ) from error

    def construct(
        self,
        config: object,
        *,
        required_capabilities: Iterable[str] = (),
        **dependencies: Any,
    ) -> nn.Module:
        registration = self.for_config(config)
        required = frozenset(required_capabilities)
        missing = required - registration.metadata.capabilities
        if missing:
            raise NotImplementedError(
                f"WAM plugin {registration.metadata.name!r} version "
                f"{registration.metadata.version} does not support capabilities: "
                f"{', '.join(sorted(missing))}"
            )
        plugin = registration.factory(config=config, **dependencies)
        if not isinstance(plugin, nn.Module) or not isinstance(plugin, WAMPlugin):
            raise TypeError(
                f"WAM factory {registration.metadata.name!r} must return an "
                "nn.Module implementing WAMPlugin"
            )
        return plugin

    def metadata(self) -> tuple[WAMPluginMetadata, ...]:
        return tuple(
            registration.metadata
            for _, registration in sorted(self._by_name.items())
        )


def create_default_wam_registry() -> WAMRegistry:
    """Build the built-in registry without entry points or dynamic imports."""

    from worldscape_policy.wam.wan21 import register_wan21
    from worldscape_policy.wam.wan22 import register_wan22

    registry = WAMRegistry()
    register_wan21(registry)
    register_wan22(registry)
    registry.seal()
    return registry


DEFAULT_WAM_REGISTRY = create_default_wam_registry()


__all__ = [
    "DEFAULT_WAM_REGISTRY",
    "WAMPluginMetadata",
    "WAMRegistration",
    "WAMRegistry",
    "create_default_wam_registry",
]
