from __future__ import annotations

import importlib
import pkgutil
from typing import Iterator

_REGISTRY: list["Program"] = []
_LOADED = False


class Program:
    name = "base"

    def match(self, instruction: str, first_obs) -> float:
        return 0.0

    def run(self, ctx) -> Iterator[list[dict]]:
        raise NotImplementedError


def register(cls: type[Program]) -> type[Program]:
    if not any(type(p) is cls for p in _REGISTRY):
        _REGISTRY.append(cls())
    return cls


def _load() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    for mod in sorted(pkgutil.iter_modules(__path__), key=lambda m: m.name):
        if not mod.name.startswith("_"):
            importlib.import_module(f"{__name__}.{mod.name}")


def select_program(instruction: str, first_obs) -> Program:
    _load()
    best, best_score = None, 0.0
    for p in _REGISTRY:
        try:
            s = float(p.match(instruction, first_obs))
        except Exception:
            s = 0.0
        if s > best_score:
            best, best_score = p, s
    if best is None:
        from .idle import Idle

        return Idle()
    return best
