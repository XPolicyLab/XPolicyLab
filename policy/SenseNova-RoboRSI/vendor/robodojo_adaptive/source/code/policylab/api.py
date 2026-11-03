"""The complete benchmark adapter contract. Standard library only."""
from pathlib import Path
from typing import Any, Protocol


class InvalidAction(ValueError):
    """Reject before advancing physics. The message must be safe for the policy."""


class Adapter(Protocol):
    def __init__(self, environment_dir: Path, options: dict): ...

    def reset(self, episode: dict, record_dir: Path, observation_dir: Path) -> dict:
        """Return {observation: dict, done: bool, metrics: dict}.

        Only observation crosses to the agent. Files referenced there use
        obs://relative/path and live below observation_dir. Write full raw
        trajectories/videos/diagnostics below record_dir. Episode fields and
        options are private. metrics.success is a native bool on termination.
        """
        ...

    def step(self, action: Any) -> dict:
        """Validate, execute one native action, record it and return the same envelope.

        Include applied_action when the executor transforms targets. Preserve
        native intermediate samples in record_dir. Native action horizons and
        timing are adapter responsibilities.
        """
        ...

    def finish(self) -> dict:
        """End early under native scoring; return done=True and native metrics."""
        ...

    def close(self) -> None: ...


def validate_transition(value):
    import json
    if not isinstance(value, dict) or not isinstance(value.get("observation"), dict):
        raise ValueError("Adapter transition needs an observation object")
    if type(value.get("done")) is not bool or not isinstance(value.get("metrics"), dict):
        raise ValueError("Adapter transition needs boolean done and metrics object")
    if value["done"] and type(value["metrics"].get("success")) is not bool:
        raise ValueError("Terminal metrics.success must be a native boolean")
    json.dumps(value, allow_nan=False)
    return value
