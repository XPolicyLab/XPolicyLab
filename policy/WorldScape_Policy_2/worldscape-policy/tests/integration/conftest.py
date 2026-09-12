from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import nn

from worldscape_policy.conditioning import ConditionRouter
from evals.common import RolloutConfig, RolloutInput, RolloutRunner
from worldscape_policy.memory.visual import VisualPrefillManager
from worldscape_policy.policy import WorldScapePolicy
from worldscape_policy.rollout.session import PolicyRuntime
from worldscape_policy.types import (
    Conditioning,
    InteractionMode,
    ObservationBatch,
    PromptBatch,
    WorldActionOutput,
)


class _NumericalConditioner(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, *, observation, prompts, event_memory, training, **kwargs):
        del prompts, event_memory, training, kwargs
        return Conditioning(
            cross_attention_tokens=torch.full(
                (observation.images.shape[0], 2, 3), self.value
            )
        )


class _NumericalCodec(nn.Module):
    def encode_visual(self, video: torch.Tensor) -> torch.Tensor:
        return video.float().mean(dim=(-2, -1), keepdim=True)


class _NumericalWAM(nn.Module):
    def training_forward(self, **kwargs) -> WorldActionOutput:
        return WorldActionOutput(
            video_velocity=kwargs["noisy_video"],
            action_velocity=kwargs["noisy_action"],
        )

    def sample(self, **kwargs) -> WorldActionOutput:
        state = kwargs["state"]
        tokens = kwargs["cross_attention_tokens"]
        value = tokens.mean().to(dtype=state.dtype)
        return WorldActionOutput(
            action=value.expand(state.shape[0], 2, state.shape[-1]).clone()
        )


class _ObservationSource:
    def __init__(self, mode: InteractionMode) -> None:
        self.mode = mode

    def read(self, step_index: int) -> RolloutInput:
        images = torch.full((1, 1, 1, 3, 4, 4), float(step_index + 1))
        observation = ObservationBatch(
            images=images,
            head_view=images[:, :, 0],
            proprioception=torch.zeros(1, 1, 3),
            embodiment_id=torch.zeros(1, dtype=torch.long),
        )
        prompts = (
            PromptBatch(vlm_planning_text=["plan"])
            if self.mode is InteractionMode.AUTO
            else PromptBatch(language_instruction=["move"])
        )
        return RolloutInput(observation=observation, prompts=prompts)


class _ActionExecutor:
    def __init__(self) -> None:
        self.actions: list[torch.Tensor] = []

    def execute(
        self,
        output: WorldActionOutput,
        *,
        timeout_s: float | None,
    ) -> None:
        del timeout_s
        self.actions.append(output.require_action().clone())


@pytest.fixture
def composed_rollout() -> Callable[[InteractionMode], tuple[object, _ActionExecutor]]:
    def run(mode: InteractionMode) -> tuple[object, _ActionExecutor]:
        policy = WorldScapePolicy(
            condition_router=ConditionRouter(
                auto_conditioner=_NumericalConditioner(2.0),
                interactive_conditioner=_NumericalConditioner(3.0),
            ),
            visual_memory=VisualPrefillManager(_NumericalCodec()),
            wam=_NumericalWAM(),
        )
        executor = _ActionExecutor()
        result = RolloutRunner(
            PolicyRuntime(policy),
            _ObservationSource(mode),
            executor,
        ).run(
            RolloutConfig(mode=mode, max_steps=2, episode_id=f"{mode.value}-cpu"),
            generator=torch.Generator(device="cpu").manual_seed(7),
        )
        return result, executor

    return run
