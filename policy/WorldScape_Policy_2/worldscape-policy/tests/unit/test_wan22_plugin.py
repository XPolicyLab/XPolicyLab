from __future__ import annotations

import torch
import pytest
from torch import nn

from worldscape_policy.conditioning import AutoPlanningFeatures
from worldscape_policy.registry import Wan22PolicyBuildConfig, build_wan22_policy
from worldscape_policy.types import (
    ObservationBatch,
    PromptBatch,
    VisualMemoryState,
    WAMInferenceState,
    WanI2VCondition,
    WorldActionOutput,
)
from worldscape_policy.visual_mosaic import prepare_diffusion_mosaic


class FakeVLM(nn.Module):
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
        del planning_text, training, visual_input_range, planning_supervision
        perception = (
            images.float()
            .mean(dim=tuple(range(1, images.ndim)))
            .view(images.shape[0], 1, 1)
            .expand(images.shape[0], 2, 4)
        )
        return AutoPlanningFeatures(
            perception_features=perception,
            negative_perception_features=(
                torch.full((images.shape[0], 2, 4), 3.0)
                if negative_text is not None
                else None
            ),
        )


class FakeT5(nn.Module):
    def encode_text(self, instructions):
        return torch.ones(len(instructions), 2, 4)


class FakeEventMemory(nn.Module):
    def forward(self, current_tokens, **kwargs):
        del kwargs
        return current_tokens, {}


class FakeVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.encode_calls = 0
        self.inputs = []

    def encode(self, video, **kwargs):
        del kwargs
        self.encode_calls += 1
        self.inputs.append(video.detach().clone())
        return video * self.weight


class FakeImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.calls = 0
        self.inputs = []

    def encode_image(self, image):
        self.calls += 1
        self.inputs.append(image.detach().clone())
        return image.mean(dim=(-1, -2)) * self.weight


class FakeKernel:
    def __init__(self):
        self.anchor_latents = []
        self.positive_conditions = []
        self.negative_conditions = []

    def prepare_inference_state(
        self, *, core, inference_state, observation_num_frames
    ):
        del core
        if inference_state.current_start_frame and (
            observation_num_frames == 1
            or inference_state.current_start_frame >= 2
        ):
            return WAMInferenceState()
        return inference_state

    def training_forward(self, **kwargs):
        return WorldActionOutput(
            video_velocity=kwargs["noisy_video"],
            action_velocity=kwargs["noisy_action"],
        )

    def sample(self, **kwargs):
        self.anchor_latents.append(kwargs["anchor_latent"])
        self.positive_conditions.append(kwargs["cross_attention_tokens"])
        self.negative_conditions.append(kwargs["negative_cross_attention_tokens"])
        previous = kwargs["inference_state"]
        if torch.is_tensor(previous.positive_kv_cache):
            previous.positive_kv_cache.add_(1)
        next_state = WAMInferenceState(
            i2v_condition=previous.i2v_condition,
            current_start_frame=previous.current_start_frame + 1,
            positive_kv_cache=(
                previous.positive_kv_cache
                if previous.positive_kv_cache is not None
                else "positive-kv"
            ),
            negative_kv_cache="negative-kv",
            positive_cross_attention_cache="positive-cross",
            negative_cross_attention_cache="negative-cross",
            condition_tokens=previous.condition_tokens,
            negative_condition_tokens=previous.negative_condition_tokens,
            prompt_signature=previous.prompt_signature,
        )
        return WorldActionOutput(action=torch.zeros(1, 2, 3)), next_state


def _build_policy(*, visual_input_range="zero_one"):
    vae = FakeVAE()
    image_encoder = FakeImageEncoder()
    kernel = FakeKernel()
    policy = build_wan22_policy(
        config=Wan22PolicyBuildConfig(
            num_frames=3,
            visual_input_range=visual_input_range,
        ),
        vlm=FakeVLM(),
        token_pooler=nn.Identity(),
        projector=nn.Linear(4, 4, bias=False),
        event_memory=FakeEventMemory(),
        t5=FakeT5(),
        vae=vae,
        core=nn.Linear(4, 4, bias=False),
        image_encoder=image_encoder,
        numerical_kernel=kernel,
    )
    return policy, vae, image_encoder, kernel


def _observation():
    return ObservationBatch(
        images=torch.ones(1, 2, 1, 3, 4, 4),
        head_view=torch.ones(1, 1, 3, 4, 4),
        proprioception=torch.zeros(1, 1, 3),
        embodiment_id=torch.zeros(1, dtype=torch.long),
    )


def test_builder_has_one_canonical_checkpoint_owner_per_wan_module():
    policy, _, _, _ = _build_policy()
    keys = set(policy.state_dict())

    assert "condition_router.auto.projector.weight" in keys
    assert "visual_memory.codec.vae.weight" in keys
    assert "wam.core.weight" in keys
    assert "wam.image_encoder.weight" in keys
    assert not any(key.startswith("wam.") and ".vae." in key for key in keys)


def test_builder_rejects_shared_registered_module_ownership():
    shared = nn.Linear(4, 4)
    with pytest.raises(ValueError, match="checkpoint ownership must be unique"):
        build_wan22_policy(
            config=Wan22PolicyBuildConfig(num_frames=3),
            vlm=FakeVLM(),
            token_pooler=nn.Identity(),
            projector=shared,
            event_memory=FakeEventMemory(),
            t5=FakeT5(),
            vae=FakeVAE(),
            core=shared,
            image_encoder=FakeImageEncoder(),
            numerical_kernel=FakeKernel(),
        )


def test_plugin_builds_i2v_anchor_once_and_commits_causal_state():
    policy, _, image_encoder, kernel = _build_policy()
    observation = _observation()
    prompts = PromptBatch(vlm_planning_text=["plan"])

    first = policy.sample(
        mode="auto",
        observation=observation,
        prompts=prompts,
        generator=torch.Generator().manual_seed(0),
    )
    assert image_encoder.calls == 1
    assert kernel.anchor_latents[0] is not None
    assert first.next_visual_memory.wam_state.current_start_frame == 1

    second = policy.sample(
        mode="auto",
        observation=observation,
        prompts=prompts,
        visual_memory=first.next_visual_memory,
        event_memory=first.next_memory,
        generator=torch.Generator().manual_seed(1),
    )
    assert image_encoder.calls == 1
    assert kernel.anchor_latents[1] is None
    assert second.next_visual_memory.wam_state.current_start_frame == 2


def test_single_frame_observation_starts_a_fresh_causal_window():
    policy, _, image_encoder, _ = _build_policy()
    first = policy.sample(
        mode="auto",
        observation=_observation(),
        prompts=PromptBatch(vlm_planning_text=["plan"]),
        generator=torch.Generator().manual_seed(0),
    )
    single = _observation()
    single.images = single.images[:, :1]
    second = policy.sample(
        mode="auto",
        observation=single,
        prompts=PromptBatch(vlm_planning_text=["plan"]),
        visual_memory=first.next_visual_memory,
        event_memory=first.next_memory,
        generator=torch.Generator().manual_seed(1),
    )

    assert image_encoder.calls == 2
    assert second.next_visual_memory.wam_state.current_start_frame == 1
    assert second.next_memory.perception_tokens is None
    assert second.next_memory.pending_perception_tokens is not None


def test_auto_condition_is_cached_inside_window_and_refreshed_at_boundary():
    policy, _, _, kernel = _build_policy()
    prompts = PromptBatch(vlm_planning_text=["plan"])
    first = policy.sample(
        mode="auto",
        observation=_observation(),
        prompts=prompts,
        generator=torch.Generator().manual_seed(0),
    )
    changed = _observation()
    changed.images = torch.zeros_like(changed.images)
    changed.head_view = torch.zeros_like(changed.head_view)
    second = policy.sample(
        mode="auto",
        observation=changed,
        prompts=prompts,
        visual_memory=first.next_visual_memory,
        event_memory=first.next_memory,
        generator=torch.Generator().manual_seed(1),
    )
    boundary = _observation()
    boundary.images = torch.zeros_like(boundary.images)
    boundary.head_view = torch.zeros_like(boundary.head_view)
    third = policy.sample(
        mode="auto",
        observation=boundary,
        prompts=prompts,
        visual_memory=second.next_visual_memory,
        event_memory=second.next_memory,
        generator=torch.Generator().manual_seed(2),
    )

    torch.testing.assert_close(
        kernel.positive_conditions[1],
        kernel.positive_conditions[0],
    )
    assert not torch.allclose(
        kernel.positive_conditions[2],
        kernel.positive_conditions[0],
    )
    assert third.next_memory.perception_tokens.shape[1] == 2


def test_prompt_change_resets_causal_and_event_windows():
    policy, _, image_encoder, _ = _build_policy()
    first = policy.sample(
        mode="auto",
        observation=_observation(),
        prompts=PromptBatch(vlm_planning_text=["first"]),
        generator=torch.Generator().manual_seed(0),
    )
    second = policy.sample(
        mode="auto",
        observation=_observation(),
        prompts=PromptBatch(vlm_planning_text=["second"]),
        visual_memory=first.next_visual_memory,
        event_memory=first.next_memory,
        generator=torch.Generator().manual_seed(1),
    )

    assert image_encoder.calls == 2
    assert second.next_visual_memory.wam_state.current_start_frame == 1
    assert second.next_memory.perception_tokens is None
    assert second.next_memory.pending_perception_tokens.shape[1] == 1


def test_replacing_persistent_visual_prompt_resets_causal_state():
    policy, _, image_encoder, _ = _build_policy()
    first = policy.sample(
        mode="auto",
        observation=_observation(),
        prompts=PromptBatch(
            vlm_planning_text=["plan"],
            goal_images=torch.ones(1, 1, 3, 4, 4),
        ),
        generator=torch.Generator().manual_seed(0),
    )
    second = policy.sample(
        mode="auto",
        observation=_observation(),
        prompts=PromptBatch(
            vlm_planning_text=["plan"],
            goal_images=torch.zeros(1, 1, 3, 4, 4),
        ),
        visual_memory=first.next_visual_memory,
        event_memory=first.next_memory,
        generator=torch.Generator().manual_seed(1),
    )

    assert image_encoder.calls == 2
    assert second.next_visual_memory.persistent_prompt_version == 2
    assert second.next_visual_memory.wam_state.current_start_frame == 1
    assert second.next_memory.perception_tokens is None


def test_auto_guidance_and_candidate_cache_are_explicit_and_transactional():
    policy, _, _, kernel = _build_policy()
    committed_cache = torch.zeros(1)
    visual_state = VisualMemoryState(
        wam_state=WAMInferenceState(
            i2v_condition=WanI2VCondition(
                clip_features=torch.ones(1, 1, 1),
                masked_latent_y=torch.ones(1, 5, 3, 4, 4),
            ),
            current_start_frame=1,
            positive_kv_cache=committed_cache,
        )
    )

    output = policy.sample(
        mode="auto",
        observation=_observation(),
        prompts=PromptBatch(
            vlm_planning_text=["plan"],
            negative_vlm_text=[""],
        ),
        visual_memory=visual_state,
        generator=torch.Generator().manual_seed(0),
    )

    assert kernel.negative_conditions[0] is not None
    torch.testing.assert_close(
        kernel.negative_conditions[0],
        kernel.positive_conditions[0] * 3,
    )
    assert torch.all(committed_cache == 0)
    assert torch.all(output.next_visual_memory.wam_state.positive_kv_cache == 1)


def test_without_runtime_cache_preserves_i2v_condition_and_progress():
    policy, _, _, _ = _build_policy()
    output = policy.sample(
        mode="auto",
        observation=_observation(),
        prompts=PromptBatch(vlm_planning_text=["plan"]),
        generator=torch.Generator().manual_seed(0),
    )

    state = output.next_visual_memory.without_runtime_cache()
    assert state.wam_state.i2v_condition is not None
    assert state.wam_state.current_start_frame == 1
    assert state.wam_state.positive_kv_cache is None
    assert state.wam_state.negative_kv_cache is None
    assert state.wam_state.positive_cross_attention_cache is None
    assert state.wam_state.negative_cross_attention_cache is None


def test_already_normalized_visual_inputs_are_not_normalized_twice():
    policy, vae, image_encoder, _ = _build_policy(
        visual_input_range="minus_one_one"
    )
    observation = _observation()
    observation.images.fill_(-1)
    observation.head_view.fill_(-1)

    policy.sample(
        mode="auto",
        observation=observation,
        prompts=PromptBatch(vlm_planning_text=["plan"]),
        generator=torch.Generator().manual_seed(0),
    )

    assert torch.all(image_encoder.inputs[0] == -1)
    assert torch.all(vae.inputs[0] == -1)
    assert torch.all(vae.inputs[1][:, :, :1] == -1)


def test_three_view_sample_uses_normalized_diffusion_reference():
    policy, _, image_encoder, _ = _build_policy(visual_input_range="zero_one")
    images = torch.stack(
        (
            torch.full((1, 2, 3, 4, 4), 0.2),
            torch.full((1, 2, 3, 4, 4), 0.4),
            torch.full((1, 2, 3, 4, 4), 0.6),
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
        prompts=PromptBatch(vlm_planning_text=["plan"]),
        generator=torch.Generator().manual_seed(0),
    )

    expected = prepare_diffusion_mosaic(images, input_range="zero_one")[:, :1]
    torch.testing.assert_close(image_encoder.inputs[0], expected)


@pytest.mark.parametrize(
    ("max_chunk_size", "expected_frame"),
    [(1, -1), (2, 0), (4, 0)],
)
def test_sampling_reference_frame_follows_checkpoint_chunk_geometry(
    max_chunk_size,
    expected_frame,
):
    policy, _, _, _ = _build_policy(visual_input_range="zero_one")
    policy.wam.core.max_chunk_size = max_chunk_size
    first = torch.full((1, 1, 3, 4, 4), 0.2)
    current = torch.full((1, 1, 3, 4, 4), 0.8)
    images = torch.stack(
        (
            torch.cat((first, current), dim=1),
            torch.cat((first + 0.1, current + 0.1), dim=1),
            torch.cat((first + 0.2, current + 0.2), dim=1),
        ),
        dim=2,
    )
    observation = ObservationBatch(
        images=images,
        head_view=images[:, -1:, 0],
        proprioception=torch.zeros(1, 1, 3),
        embodiment_id=torch.zeros(1, dtype=torch.long),
    )

    reference, normalized = policy._sample_reference_frame(observation)

    prepared = prepare_diffusion_mosaic(images, input_range="zero_one")
    expected = (
        prepared[:, -1:]
        if expected_frame == -1
        else prepared[:, expected_frame : expected_frame + 1]
    )
    assert normalized is True
    torch.testing.assert_close(reference, expected)


def test_nonnegative_minus_one_one_inputs_still_preserve_their_range():
    policy, vae, image_encoder, _ = _build_policy(
        visual_input_range="minus_one_one"
    )
    observation = _observation()
    observation.images.fill_(0.5)
    observation.head_view.fill_(0.5)

    policy.sample(
        mode="auto",
        observation=observation,
        prompts=PromptBatch(vlm_planning_text=["plan"]),
        generator=torch.Generator().manual_seed(0),
    )

    assert torch.all(image_encoder.inputs[0] == 0.5)
    assert torch.all(vae.inputs[0] == 0.5)
    assert torch.all(vae.inputs[1][:, :, :1] == 0.5)
