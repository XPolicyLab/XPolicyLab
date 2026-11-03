from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from evals.agilex.robot import (
    AgileXReadRequest,
    HDF5ReplayRobot,
)
from evals.common.protocols import RolloutInput
from worldscape_policy.rollout.runner import (
    RolloutConfig,
    RolloutRunner,
)
from worldscape_policy.rollout.session import PolicyRuntime
from worldscape_policy.types import (
    InteractionMode,
    ObservationBatch,
    PromptBatch,
    WorldActionOutput,
)


@dataclass(frozen=True)
class HDF5ReplayConfig:
    path: str | Path
    mode: InteractionMode | str
    instruction: str
    embodiment_id: int
    max_steps: int | None = None
    use_history: bool = True
    num_history_frames: int = 4
    action_mode: str = "eef"


def run_hdf5_replay(
    runtime: PolicyRuntime,
    config: HDF5ReplayConfig,
    *,
    generator: torch.Generator,
) -> list[WorldActionOutput]:
    """Run the native transactional policy against the validated HDF5 robot."""

    robot = HDF5ReplayRobot(config.path)
    policy_parameter = next(runtime.policy.parameters(), None)
    policy_device = (
        policy_parameter.device if policy_parameter is not None else torch.device("cpu")
    )
    max_steps = config.max_steps or robot.default_max_steps(rollout_steps=1)
    result = RolloutRunner(
        runtime,
        _HDF5ObservationSource(robot, config, policy_device),
        _HDF5ActionExecutor(),
    ).run(
        RolloutConfig(mode=config.mode, max_steps=max_steps),
        generator=generator,
    )
    result.raise_for_error()
    return list(result.outputs)


class _HDF5ObservationSource:
    def __init__(
        self,
        robot: object,
        config: HDF5ReplayConfig,
        device: torch.device,
    ) -> None:
        self._robot = robot
        self._config = config
        self._device = device

    def read(self, step_index: int) -> RolloutInput:
        del step_index
        config = self._config
        value = self._robot.observe(AgileXReadRequest(
            use_history=config.use_history,
            num_history_frames=config.num_history_frames,
            action_mode=config.action_mode,
        ))
        observation = _to_observation_batch(
            high=value.high,
            left=value.left,
            right=value.right,
            state=value.state,
            embodiment_id=config.embodiment_id,
            device=self._device,
        )
        prompts = _replay_prompts(config, batch_size=observation.images.shape[0])
        return RolloutInput(observation=observation, prompts=prompts)


class _HDF5ActionExecutor:
    def execute(
        self,
        output: WorldActionOutput,
        *,
        timeout_s: float | None,
    ) -> None:
        del timeout_s
        output.require_action()


def _replay_prompts(
    config: HDF5ReplayConfig,
    *,
    batch_size: int,
) -> PromptBatch:
    mode = InteractionMode.parse(config.mode)
    instructions = [config.instruction] * batch_size
    negatives = [""] * batch_size
    if mode is InteractionMode.AUTO:
        return PromptBatch(
            vlm_planning_text=instructions,
            negative_vlm_text=negatives,
        )
    return PromptBatch(
        language_instruction=instructions,
        negative_language_instruction=negatives,
    )


def _to_observation_batch(
    *,
    high: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    state: dict[str, np.ndarray],
    embodiment_id: int,
    device: torch.device | str = "cpu",
) -> ObservationBatch:
    videos = [_ensure_video(value) for value in (high, left, right)]
    if len({video.shape[0] for video in videos}) != 1:
        raise ValueError("Replay camera histories must have equal length")
    stacked = np.stack(videos, axis=1)
    images = (
        torch.from_numpy(stacked)
        .permute(0, 1, 4, 2, 3)
        .unsqueeze(0)
        .float()
        .div(255.0)
    )
    head_view = (
        torch.from_numpy(videos[0][-1:])
        .permute(0, 3, 1, 2)
        .unsqueeze(0)
        .float()
        .div(255.0)
    )
    state_tensor = torch.from_numpy(_flatten_state(state)).view(1, 1, -1)
    return ObservationBatch(
        images=images.to(device),
        head_view=head_view.to(device),
        proprioception=state_tensor.to(device),
        embodiment_id=torch.tensor(
            [embodiment_id], dtype=torch.long, device=device
        ),
    )


def _ensure_video(value: np.ndarray) -> np.ndarray:
    video = np.asarray(value)
    if video.ndim == 3:
        video = video[None]
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError(f"Expected [T,H,W,C] replay video, got {video.shape}")
    return np.ascontiguousarray(video)


def _flatten_state(state: dict[str, np.ndarray]) -> np.ndarray:
    preferred_order = (
        "state.left_pos",
        "state.left_rot6d",
        "state.left_gripper",
        "state.right_pos",
        "state.right_rot6d",
        "state.right_gripper",
        "state.left_joint",
        "state.right_joint",
    )
    values = [
        np.asarray(state[key], dtype=np.float32).reshape(-1)
        for key in preferred_order
        if key in state
    ]
    if not values:
        raise ValueError("Replay state did not contain a supported state field")
    return np.concatenate(values)
