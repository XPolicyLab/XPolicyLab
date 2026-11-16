"""The four program primitives and utilities for public observations."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

PROGRAM_PRIMITIVES = frozenset({"get_instruction", "get_observation", "step", "reset"})
SURFACES = {
    "robotwin": "RoboTwinLowLevelApi",
    "robocasa": "RoboCasaLowLevelApi",
    "robodojo": "RoboDojoLowLevelApi",
}


class PrimitiveApi:
    """The four primitives bound into submitted code."""

    def __init__(self, env):
        self._env = env

    def reset(self) -> tuple:
        """Restore this scene's initial state. PlayGround/exploration only.

        Returns (observation, info): the native observation and reset info for
        the active benchmark. The host reuses the current scene's initialization;
        this function accepts no seed or options and cannot select another scene.
        Submission counts and wall-clock budgets are not reset. Program variables,
        cumulative action telemetry and recorded frames are retained. RoboDojo
        renews its official per-attempt native action allowance. Other benchmarks
        retain consumed native actions and stay blocked if that budget is exhausted.
        In Test this raises PermissionError before touching simulator state,
        even when called through an alias or a user-defined function.
        """
        return self._env.reset_from_program()



def rgb_images(obs: dict) -> dict[str, np.ndarray]:
    """Extract only RGB already present in the public policy observation."""
    frames = {}
    for name, value in obs.items():
        if name.startswith("video."):
            frames[name.removeprefix("video.")] = np.asarray(value)
        elif name.endswith("_image") and isinstance(value, np.ndarray):
            frames[name.removesuffix("_image")] = value
        elif isinstance(value, dict):
            rgb = value.get("rgb", value.get("color"))
            if rgb is None:
                rgb = value.get("images", {}).get("rgb")
            if rgb is not None:
                frames[name] = np.asarray(rgb)
    if isinstance(obs.get("observation"), dict):
        frames.update(rgb_images(obs["observation"]))
    if isinstance(obs.get("vision"), dict):
        frames.update(rgb_images(obs["vision"]))
    return frames


class ObservationSpace(spaces.Space):
    """Native dictionaries may contain empty lists and camera metadata."""

    def __init__(self, example):
        super().__init__()
        self.example = example

    def sample(self, mask=None):
        import copy

        return copy.deepcopy(self.example)

    def contains(self, value):
        def matches(reference, candidate):
            if isinstance(reference, dict):
                return (
                    isinstance(candidate, dict)
                    and reference.keys() == candidate.keys()
                    and all(matches(v, candidate[k]) for k, v in reference.items())
                )
            if isinstance(reference, np.ndarray):
                return (
                    isinstance(candidate, np.ndarray)
                    and reference.shape == candidate.shape
                    and reference.dtype == candidate.dtype
                )
            if isinstance(reference, list | tuple):
                return (
                    isinstance(candidate, type(reference))
                    and len(candidate) == len(reference)
                    and all(matches(a, b) for a, b in zip(reference, candidate, strict=True))
                )
            return isinstance(candidate, type(reference))

        return matches(self.example, value)
