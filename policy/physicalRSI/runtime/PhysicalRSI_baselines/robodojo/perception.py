"""Observation-bound obstacle estimation for trusted runtime perception.

This adapter checks local receipt age, not camera capture latency. The estimator
must use only the provided RGB/state plus its pinned public calibration and model,
return world-frame cuboids, and honor the absolute deadline.
"""

import math
from copy import deepcopy


class ObservedObstacles:
    def __init__(self, *, channel, estimate, max_age_s):
        if not callable(estimate) or not channel.cameras:
            raise ValueError("RGB channel and callable estimator required")
        if not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError("Positive observation age limit required")
        self.channel, self.estimate, self.max_age_s = channel, estimate, max_age_s

    def __call__(self, *, state, deadline):
        return self.observe(state=state, deadline=deadline)["objects"]

    def observe(self, *, state, deadline):
        snapshot = self.channel.snapshot(deadline=deadline)
        self.channel.require_current(
            snapshot, max_age_s=self.max_age_s, deadline=deadline
        )
        if state != snapshot["state"]:
            raise ValueError("Planning state differs from perception observation")
        result = self.estimate(snapshot=deepcopy(snapshot), deadline=deadline)
        self.channel.require_current(
            snapshot, max_age_s=self.max_age_s, deadline=deadline
        )
        if not isinstance(result, dict):
            raise ValueError("Estimator must return world-frame obstacle mapping")
        return project_scene(
            dict(
                frame="world",
                unit="m",
                objects=result,
                observation=dict(
                    channel_id=snapshot["channel_id"], sequence=snapshot["sequence"]
                ),
            )
        )


def project_scene(result, *, max_objects=256):
    """Bounded policy-visible geometry; no RGB, scores or simulator metadata."""
    import re
    from .collision_world import validate_scene

    if result["frame"] != "world" or result["unit"] != "m":
        raise ValueError("Perception must produce world-frame metres")
    observation = result["observation"]
    channel = observation["channel_id"]
    sequence = observation["sequence"]
    if (
        not isinstance(channel, str)
        or not re.fullmatch(r"[0-9a-f]{32}", channel)
        or type(sequence) is not int
        or sequence < 1
    ):
        raise ValueError("Scene observation identity required")
    objects = result["objects"]
    if (
        not isinstance(objects, dict)
        or len(objects) > max_objects
        or any(not isinstance(name, str) or len(name) > 128 for name in objects)
    ):
        raise ValueError("Bounded named scene objects required")
    objects = (
        validate_scene({"cuboid": objects}, max_objects)["cuboid"] if objects else {}
    )
    return dict(
        frame="world",
        unit="m",
        objects=objects,
        observation=dict(channel_id=channel, sequence=sequence),
    )
