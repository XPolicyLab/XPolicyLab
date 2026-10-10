"""Translate environment observations/actions to the model canonical layout."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from mmabc.eval.policy import Observation, MMABCInferencePolicy

_REGISTRY: dict[str, type["BenchmarkAdapter"]] = {}


def register(name: str):
    def deco(cls: type[BenchmarkAdapter]) -> type[BenchmarkAdapter]:
        _REGISTRY[name] = cls
        return cls

    return deco


def build_adapter(name: str, **kwargs) -> "BenchmarkAdapter":
    if name not in _REGISTRY:
        raise KeyError(f"unknown adapter {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available() -> list[str]:
    return sorted(_REGISTRY)


class BenchmarkAdapter(ABC):
    """Translates one benchmark's observation/action conventions to canonical."""

    #: Which canonical view slot each of the benchmark's cameras maps to.
    view_map: dict[str, str] = {}

    @abstractmethod
    def to_canonical_state(self, raw_obs: dict) -> np.ndarray:
        """Benchmark observation -> (80,) absolute canonical state."""

    @abstractmethod
    def to_canonical_images(self, raw_obs: dict) -> dict[str, np.ndarray]:
        """Benchmark observation -> {canonical view slot: HxWx3 uint8}."""

    @abstractmethod
    def from_canonical_action(self, canonical: np.ndarray) -> np.ndarray:
        """(T, 80) absolute canonical actions -> whatever the benchmark steps with."""

    def observe(self, raw_obs: dict, instruction: str) -> Observation:
        return Observation(
            state=self.to_canonical_state(raw_obs),
            images=self.to_canonical_images(raw_obs),
            instruction=instruction,
        )

    def rollout_step(
        self, policy: MMABCInferencePolicy, raw_obs: dict, instruction: str
    ) -> np.ndarray:
        result = policy.act(self.observe(raw_obs, instruction))
        return self.from_canonical_action(result["action_canonical"])
