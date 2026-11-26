"""Explicit VLM backend registry."""

from __future__ import annotations

import importlib
from typing import Type

from .base import VLMBackend

_BACKENDS: dict[str, Type[VLMBackend]] = {}
_ALIASES = {
    "florence": "florence2",
    "qwen3.5": "qwen3_5",
    # Every Qwen3.5 size shares one backend implementation; the decoder width
    # comes from the checkpoint config, so the size suffix is only an alias.
    "qwen3.5-2b": "qwen3_5",
    "qwen3.5-4b": "qwen3_5",
    "qwen35": "qwen3_5",
    "dm0.5": "dm05",
    "dm05": "dm05",
    "dm_05": "dm05",
}
_BUILTIN_MODULES = {
    "qwen3_5": ".qwen3_5",
}


def register_vlm_backend(name: str):
    key = str(name).strip().lower()

    def _wrap(cls: Type[VLMBackend]) -> Type[VLMBackend]:
        if key in _BACKENDS:
            raise KeyError(f"VLM backend already registered: {key}")
        cls.name = key
        _BACKENDS[key] = cls
        return cls

    return _wrap


def _canonical_name(name: str) -> str:
    key = str(name or "florence2").strip().lower()
    return _ALIASES.get(key, key)


def _load_builtin_backend(name: str) -> None:
    module_name = _BUILTIN_MODULES.get(name)
    if module_name is not None and name not in _BACKENDS:
        importlib.import_module(module_name, package=__package__)


def get_vlm_backend_class(name: str) -> Type[VLMBackend]:
    key = _canonical_name(name)
    _load_builtin_backend(key)
    if key not in _BACKENDS:
        available = sorted(set(_BACKENDS) | set(_BUILTIN_MODULES))
        raise KeyError(
            f"Unknown VLM backend {name!r}; available={available}"
        )
    return _BACKENDS[key]


def build_vlm_backend(name: str) -> VLMBackend:
    return get_vlm_backend_class(name)()
