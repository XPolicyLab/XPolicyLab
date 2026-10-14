"""Construct the one Gymnasium environment selected by a task definition.

The simulator-specific imports belong here, at construction time.  Callers
receive an :class:`environments.Environment` and interact only through
``reset``, ``step(code)``, ``render`` and ``close``.

The import is lazy on purpose: each backend can only be imported inside its own
virtualenv, and importing the wrong one raises before it can be useful.
"""

from __future__ import annotations

from environments import Environment
from orchestrator.config import TaskConfig
from simulators.primitives import SURFACES


def build(cfg: TaskConfig, *, seed: int | None = None) -> Environment:
    """Build the common Gymnasium environment for ``cfg``."""
    backend = cfg.simulator.backend
    if backend not in SURFACES:
        raise ValueError(f"unknown simulator backend {backend!r}")
    expected = SURFACES[backend]
    if cfg.simulator.api_surface != expected:
        raise ValueError(f"This task requires the primitive surface {expected}")
    if backend == "robodojo":
        from simulators.robodojo import RoboDojoTask

        environment = RoboDojoTask(
            task_name=cfg.simulator.env_task,
            seed=seed,
            solve_articulation_contact_last=cfg.simulator.solve_articulation_contact_last,
            min_position_iterations=cfg.simulator.min_position_iterations,
            min_velocity_iterations=cfg.simulator.min_velocity_iterations,
        )
    else:
        raise ValueError(f"unknown simulator backend {backend!r}")
    if not isinstance(environment, Environment):
        raise TypeError("task adapter must implement environments.Environment")
    return environment
