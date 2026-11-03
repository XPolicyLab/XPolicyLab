from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evals.agilex.robot import (
    AgileXReadRequest,
    HDF5ReplayRobot,
)
from evals.common.hdf5_replay import _to_observation_batch
from evals.common.suite import EvaluationTask
from worldscape_policy.types import (
    InteractionMode,
    PromptBatch,
    WorldActionOutput,
)


class HDF5EvaluationEnvironment:
    """Read-only HDF5 replay exposed through the common environment contract."""

    def __init__(
        self,
        path: str | Path,
        *,
        use_history: bool = True,
        num_history_frames: int = 4,
        action_mode: str = "eef",
    ) -> None:
        self.path = Path(path)
        self.use_history = use_history
        self.num_history_frames = num_history_frames
        self.action_mode = action_mode
        self._robot: Any = None

    def reset(self, task: EvaluationTask, *, seed: int | None = None) -> Any:
        del task, seed
        self._robot = HDF5ReplayRobot(self.path)
        return self._read()

    def step(self, action: Any) -> Any:
        del action
        return self._read(), 0.0, False, False, {}

    def success(self) -> bool:
        # Offline replay files do not encode intervention-based task success.
        return False

    def close(self) -> None:
        close = getattr(self._robot, "close", None)
        if callable(close):
            close()
        self._robot = None

    def _read(self) -> Any:
        if self._robot is None:
            raise RuntimeError("Call reset before reading HDF5 replay")
        return self._robot.observe(AgileXReadRequest(
            use_history=self.use_history,
            num_history_frames=self.num_history_frames,
            action_mode=self.action_mode,
        )).as_tuple()


class HDF5EvaluationAdapter:
    def __init__(self, *, embodiment_id: int) -> None:
        self.embodiment_id = embodiment_id

    def observation(
        self,
        value: Any,
        *,
        device: torch.device | str = "cpu",
    ):
        high, left, right, state, _ = value
        return _to_observation_batch(
            high=high,
            left=left,
            right=right,
            state=state,
            embodiment_id=self.embodiment_id,
            device=device,
        )

    def prompt(
        self,
        instruction: str,
        *,
        mode: InteractionMode | str,
    ) -> PromptBatch:
        if InteractionMode.parse(mode) is InteractionMode.AUTO:
            return PromptBatch(
                vlm_planning_text=[instruction],
                negative_vlm_text=[""],
            )
        return PromptBatch(
            language_instruction=[instruction],
            negative_language_instruction=[""],
        )

    def action(self, output: WorldActionOutput) -> np.ndarray:
        return output.require_action().detach().cpu().numpy()


def backend_components(
    backend: str,
    config: Mapping[str, Any],
) -> tuple[Any, Any]:
    """Build an evaluation environment and adapter with lazy dependencies."""

    backend_config = config.get("backend_config", {})
    if not isinstance(backend_config, Mapping):
        raise TypeError("backend_config must be a mapping")
    adapter_config = backend_config.get("adapter", {})
    if not isinstance(adapter_config, Mapping):
        raise TypeError("backend_config.adapter must be a mapping")

    if backend == "hdf5":
        path = backend_config.get("path")
        if not path:
            raise ValueError("The hdf5 backend requires backend_config.path")
        environment = HDF5EvaluationEnvironment(
            path,
            use_history=bool(backend_config.get("use_history", True)),
            num_history_frames=int(
                backend_config.get("num_history_frames", 4)
            ),
            action_mode=str(backend_config.get("action_mode", "eef")),
        )
        adapter = HDF5EvaluationAdapter(
            embodiment_id=int(adapter_config.get("embodiment_id", 0))
        )
        return environment, adapter

    args = backend_config.get("environment_args", [])
    kwargs = backend_config.get("environment_kwargs", {})
    if not isinstance(args, list) or not isinstance(kwargs, Mapping):
        raise TypeError("environment_args/list and environment_kwargs/mapping required")
    capture_frames = bool(backend_config.get("capture_frames", False))
    if backend == "libero":
        from evals.libero.adapter import LiberoAdapter
        from evals.libero.task_suite import LiberoTaskSuiteEnvironment

        environment = LiberoTaskSuiteEnvironment(
            module_name=str(
                backend_config.get("module_name", "libero.libero.envs")
            ),
            factory_name=str(
                backend_config.get("factory_name", "OffScreenRenderEnv")
            ),
            environment_args=tuple(args),
            environment_kwargs=dict(kwargs),
            metadata_to_factory=dict(
                backend_config.get(
                    "metadata_to_factory",
                    {"bddl_file_name": "bddl_file_name"},
                )
            ),
            capture_frames=capture_frames,
        )
        adapter = LiberoAdapter(**dict(adapter_config))
    elif backend == "agilex":
        raise RuntimeError(
            "The common CLI does not start AgileX hardware. Use the existing "
            "real-robot entrypoint until its transport is migrated explicitly."
        )
    else:
        raise ValueError(f"Unknown evaluation backend: {backend!r}")
    return environment, adapter
