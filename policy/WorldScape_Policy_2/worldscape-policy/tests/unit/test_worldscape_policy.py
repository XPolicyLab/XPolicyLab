from __future__ import annotations

import pytest
import torch
from torch import nn

from worldscape_policy.conditioning import (
    AutoConditioner,
    AutoPlanningFeatures,
    ConditionRouter,
    InteractiveConditioner,
)
from worldscape_policy.memory import EventMemoryQueue
from worldscape_policy.memory.visual import VisualPrefillManager
from worldscape_policy.policy import WorldScapePolicy
from worldscape_policy.rollout.session import PolicyRuntime
from worldscape_policy.types import (
    Conditioning,
    EventMemoryState,
    InteractionMode,
    ObservationBatch,
    PromptBatch,
    WorldActionOutput,
)


class FakeConditioner(nn.Module):
    def __init__(self, value: float, *, uses_memory: bool) -> None:
        super().__init__()
        self.value = value
        self.uses_memory = uses_memory

    def forward(self, *, observation, prompts, event_memory, training):
        del prompts
        tokens = torch.full(
            (observation.images.shape[0], 2, 4),
            self.value,
            device=observation.images.device,
        )
        return Conditioning(
            cross_attention_tokens=tokens,
            event_memory=event_memory if self.uses_memory else None,
            semantic_prediction=tokens if training else None,
            semantic_target=tokens if training else None,
        )


class FakeAutoWithoutSemanticTarget(nn.Module):
    def forward(
        self,
        *,
        observation,
        prompts,
        event_memory,
        training,
        planning_supervision=False,
    ):
        del prompts, planning_supervision
        tokens = torch.ones(
            observation.images.shape[0],
            2,
            4,
            device=observation.images.device,
        )
        return Conditioning(
            cross_attention_tokens=tokens,
            event_memory=event_memory,
            semantic_prediction=tokens if training else None,
            semantic_target=None,
        )


class FakePlanningVLM(nn.Module):
    def encode_planning(
        self,
        *,
        images,
        planning_text,
        negative_text,
        training,
        visual_input_range,
        planning_supervision,
    ):
        del (
            planning_text,
            negative_text,
            training,
            visual_input_range,
            planning_supervision,
        )
        batch_size = images.shape[0]
        return AutoPlanningFeatures(
            perception_features=torch.ones(batch_size, 2, 4),
            planning_features=torch.full((batch_size, 1, 4), 2.0),
            task_embedding=torch.full((batch_size, 1, 4), 4.0),
        )


class FakeEventMemory(nn.Module):
    def forward(self, current_tokens, **kwargs):
        del kwargs
        return current_tokens, {}


class FakeT5(nn.Module):
    def encode_text(self, instructions):
        return torch.full((len(instructions), 3, 4), 3.0)


class FakeVisualCodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.prepared_inputs = []

    def encode_visual(self, video):
        reduce_dims = tuple(range(1, video.ndim))
        return video.float().mean(dim=reduce_dims).reshape(video.shape[0], 1, 1)

    def prepare_diffusion_video(self, views):
        self.prepared_inputs.append(views.detach().clone())
        return views[:, :, 0].float() + 100

    def encode_normalized(self, video):
        reduce_dims = tuple(range(1, video.ndim))
        return video.float().mean(dim=reduce_dims).reshape(video.shape[0], 1, 1)


class FakeWAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.reference_frames = []
        self.reference_frame_normalized = []

    def training_forward(self, **kwargs):
        return WorldActionOutput(
            action_velocity=kwargs["noisy_action"] - kwargs["clean_action"],
            video_velocity=kwargs["noisy_video"] - kwargs["clean_video"],
        )

    def sample(
        self,
        *,
        reference_frame,
        reference_frame_normalized,
        chunk_latents,
        observation_num_frames,
        prompt_signature,
        state,
        embodiment_id,
        cross_attention_tokens,
        negative_cross_attention_tokens,
        visual_memory,
        generator,
    ):
        self.reference_frames.append(reference_frame.detach().clone())
        self.reference_frame_normalized.append(reference_frame_normalized)
        del (
            chunk_latents,
            observation_num_frames,
            prompt_signature,
            embodiment_id,
            negative_cross_attention_tokens,
            visual_memory,
            generator,
        )
        action = cross_attention_tokens.mean(dim=(1, 2), keepdim=True)
        action = action.expand(state.shape[0], 2, state.shape[-1])
        return WorldActionOutput(action=action)


@pytest.fixture
def observation():
    return ObservationBatch(
        images=torch.ones(2, 3, 1, 3, 4, 4),
        head_view=torch.ones(2, 1, 3, 4, 4),
        proprioception=torch.zeros(2, 1, 3),
        embodiment_id=torch.zeros(2, dtype=torch.long),
    )


@pytest.fixture
def policy():
    router = ConditionRouter(
        auto_conditioner=FakeConditioner(1.0, uses_memory=True),
        interactive_conditioner=FakeConditioner(2.0, uses_memory=False),
    )
    return WorldScapePolicy(
        condition_router=router,
        visual_memory=VisualPrefillManager(FakeVisualCodec()),
        wam=FakeWAM(),
    )


def test_router_enforces_mode_specific_prompt(observation, policy):
    with pytest.raises(ValueError, match="vlm_planning_text"):
        policy.condition(
            mode=InteractionMode.AUTO,
            observation=observation,
            prompts=PromptBatch(language_instruction=["event"] * 2),
        )

    with pytest.raises(ValueError, match="language_instruction"):
        policy.condition(
            mode=InteractionMode.INTERACTIVE,
            observation=observation,
            prompts=PromptBatch(vlm_planning_text=["task"] * 2),
        )


def test_interactive_mode_does_not_forward_event_memory(observation, policy):
    memory = EventMemoryState(perception_tokens=torch.ones(2, 1, 4))
    condition = policy.condition(
        mode="interactive",
        observation=observation,
        prompts=PromptBatch(
            vlm_planning_text=["fold the shirt"] * 2,
            language_instruction=["move"] * 2,
        ),
        event_memory=memory,
        training=False,
    )
    assert condition.event_memory is None
    assert torch.all(condition.cross_attention_tokens == 2)


def test_auto_semantic_target_uses_t5_subtask_text(observation):
    router = ConditionRouter(
        auto_conditioner=FakeAutoWithoutSemanticTarget(),
        interactive_conditioner=InteractiveConditioner(t5=FakeT5()),
    )
    policy = WorldScapePolicy(
        condition_router=router,
        visual_memory=VisualPrefillManager(FakeVisualCodec()),
        wam=FakeWAM(),
    )

    condition = policy.condition(
        mode="auto",
        observation=observation,
        prompts=PromptBatch(
            vlm_planning_text=["fold shirt"] * 2,
            language_instruction=["grasp sleeve"] * 2,
        ),
        training=True,
    )

    assert condition.semantic_target is not None
    assert torch.all(condition.semantic_target == 3)


def test_sample_keeps_persistent_prompt_and_refreshes_recent_memory(observation, policy):
    prompts = PromptBatch(
        vlm_planning_text=["task"] * 2,
        goal_images=torch.full((2, 1, 3, 4, 4), 3.0),
    )
    first = policy.sample(
        mode="auto",
        observation=observation,
        prompts=prompts,
        generator=torch.Generator().manual_seed(0),
    )
    assert first.action is not None
    assert first.next_visual_memory is not None
    assert torch.all(first.next_visual_memory.persistent_prompt_latents == 3)

    second_observation = ObservationBatch(
        images=torch.full_like(observation.images, 5.0),
        head_view=observation.head_view,
        proprioception=observation.proprioception,
        embodiment_id=observation.embodiment_id,
    )
    second = policy.sample(
        mode="auto",
        observation=second_observation,
        prompts=PromptBatch(vlm_planning_text=["task"] * 2),
        visual_memory=first.next_visual_memory,
        generator=torch.Generator().manual_seed(1),
    )
    assert torch.all(second.next_visual_memory.persistent_prompt_latents == 3)
    assert torch.all(second.next_visual_memory.recent_observation_latents == 5)


def test_sample_uses_diffusion_reference_for_three_view_observations(policy):
    images = torch.stack(
        (
            torch.full((1, 2, 3, 4, 4), 1.0),
            torch.full((1, 2, 3, 4, 4), 2.0),
            torch.full((1, 2, 3, 4, 4), 3.0),
        ),
        dim=2,
    )
    observation = ObservationBatch(
        images=images,
        head_view=torch.zeros(1, 1, 3, 4, 4),
        proprioception=torch.zeros(1, 1, 3),
        embodiment_id=torch.zeros(1, dtype=torch.long),
    )

    policy.sample(
        mode="auto",
        observation=observation,
        prompts=PromptBatch(vlm_planning_text=["task"]),
        generator=torch.Generator().manual_seed(0),
    )

    assert policy.wam.reference_frame_normalized == [True]
    torch.testing.assert_close(
        policy.wam.reference_frames[0],
        torch.full((1, 1, 3, 4, 4), 101.0),
    )


def test_training_forward_delegates_velocity_without_auto_semantics_on_t5_batch(
    observation, policy
):
    shape = (2, 2, 3)
    output = policy.training_forward(
        mode="interactive",
        observation=observation,
        prompts=PromptBatch(
            vlm_planning_text=["fold the shirt"] * 2,
            language_instruction=["move"] * 2,
        ),
        clean_video=torch.zeros(2, 3, 2, 4, 4),
        clean_action=torch.zeros(shape),
        noisy_video=torch.ones(2, 3, 2, 4, 4),
        noisy_action=torch.ones(shape),
        video_timestep=torch.ones(2),
        action_timestep=torch.ones(2),
    )
    assert torch.all(output.action_velocity == 1)
    assert torch.all(output.video_velocity == 1)
    assert "semantic_prediction" not in output.metrics
    assert "semantic_target" not in output.metrics


def test_event_memory_queue_is_bounded_and_resets():
    queue = EventMemoryQueue(max_steps=2)
    for value in (1.0, 2.0, 3.0):
        queue.append(
            perception_tokens=torch.full((2, 3, 4), value),
            planning_tokens=torch.full((2, 1, 4), value),
        )

    assert queue.length == 2
    assert torch.all(queue.state.perception_tokens[:, 0] == 2)
    assert torch.all(queue.state.perception_tokens[:, 1] == 3)
    assert queue.state.valid_mask.shape == (2, 2)

    state = queue.reset_episode()
    assert queue.length == 0
    assert state.perception_tokens is None


def test_event_memory_queue_rejects_modality_changes_mid_episode():
    queue = EventMemoryQueue(max_steps=2)
    queue.append(torch.ones(1, 2, 4))
    with pytest.raises(ValueError, match="presence"):
        queue.append(
            torch.ones(1, 2, 4),
            planning_tokens=torch.ones(1, 1, 4),
        )


def test_native_conditioners_own_stable_checkpoint_names(observation):
    auto = AutoConditioner(
        vlm=FakePlanningVLM(),
        token_pooler=nn.Identity(),
        projector=nn.Linear(4, 4, bias=False),
        event_memory=FakeEventMemory(),
    )
    interactive = InteractiveConditioner(t5=FakeT5())
    router = ConditionRouter(auto, interactive)

    auto_output = router(
        mode="auto",
        observation=observation,
        prompts=PromptBatch(vlm_planning_text=["plan"] * 2),
        training=False,
    )
    assert auto_output.cross_attention_tokens.shape == (2, 3, 4)
    assert auto_output.event_memory.perception_tokens is None
    assert auto_output.event_memory.pending_perception_tokens.shape == (2, 1, 2, 4)
    assert auto_output.event_memory.pending_planning_tokens.shape == (2, 1, 1, 4)
    assert "auto.projector.weight" in router.state_dict()

    auto_training = router(
        mode="auto",
        observation=observation,
        prompts=PromptBatch(vlm_planning_text=["plan"] * 2),
        training=True,
    )
    assert auto_training.semantic_target is None

    interactive_output = router(
        mode="interactive",
        observation=observation,
        prompts=PromptBatch(language_instruction=["move"] * 2),
        training=False,
    )
    assert interactive_output.cross_attention_tokens.shape == (2, 3, 4)
    assert interactive_output.event_memory is None


def test_runtime_advances_memory_only_after_commit(observation, policy):
    runtime = PolicyRuntime(policy)
    runtime.reset("auto")
    output = runtime.predict(
        observation=observation,
        prompts=PromptBatch(vlm_planning_text=["plan"] * 2),
        generator=torch.Generator().manual_seed(0),
    )
    assert runtime.visual_memory is None

    with pytest.raises(RuntimeError, match="Commit or discard"):
        runtime.predict(
            observation=observation,
            prompts=PromptBatch(vlm_planning_text=["plan"] * 2),
            generator=torch.Generator().manual_seed(1),
        )

    runtime.commit(output)
    assert runtime.visual_memory is output.next_visual_memory

    second = runtime.predict(
        observation=observation,
        prompts=PromptBatch(vlm_planning_text=["plan"] * 2),
        generator=torch.Generator().manual_seed(2),
    )
    runtime.discard()
    assert runtime.visual_memory is output.next_visual_memory
    assert second is not output
