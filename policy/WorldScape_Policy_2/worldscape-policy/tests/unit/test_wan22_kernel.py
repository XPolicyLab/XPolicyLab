from __future__ import annotations

import pytest
import torch
from torch import nn

from worldscape_policy.types import WAMInferenceState, WanI2VCondition
from worldscape_policy.wam.wan22 import (
    Wan22KernelConfig,
    Wan22LegacyExactKernel,
)


class FakeCore(nn.Module):
    dim = 4
    num_heads = 1
    num_layers = 1
    action_dim = 3
    patch_size = (1, 2, 2)
    local_attn_size = 2

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.calls = []

    def forward(self, video, timestep=None, **kwargs):
        self.calls.append((video.clone(), timestep, kwargs))
        video_velocity = torch.ones_like(video) * self.weight
        action = kwargs.get("action")
        action_velocity = (
            torch.ones_like(action) * self.weight if action is not None else None
        )
        if "kv_cache" in kwargs:
            return video_velocity, action_velocity, kwargs["kv_cache"]
        return video_velocity, action_velocity


class FakeScheduler:
    def __init__(self):
        self.timesteps = torch.tensor([1, 0])
        self.sigmas = torch.tensor([1.0, 0.5, 0.0])

    def step(self, *, model_output, timestep, sample, step_index, return_dict):
        del timestep, step_index, return_dict
        return (sample - model_output * 0.1,)


def _condition():
    return WanI2VCondition(
        clip_features=torch.ones(1, 1, 4),
        masked_latent_y=torch.ones(1, 6, 4, 4, 4),
    )


def _kernel(action_horizon: int = 24):
    return Wan22LegacyExactKernel(
        Wan22KernelConfig(
            num_train_timesteps=1000,
            num_inference_steps=2,
            num_frame_per_block=2,
            num_action_per_block=action_horizon,
            action_horizon=action_horizon,
            cfg_scale=5.0,
        )
    )


def test_training_kernel_returns_core_velocities_without_owning_modules():
    kernel = _kernel()
    core = FakeCore()
    clean = torch.zeros(1, 2, 3, 4, 4)
    noisy = torch.ones_like(clean)
    action = torch.ones(1, 24, 3)

    output = kernel.training_forward(
        core=core,
        i2v_condition=_condition(),
        clean_video_latents=clean,
        clean_action=torch.zeros_like(action),
        noisy_video=noisy,
        noisy_action=action,
        video_timestep=torch.ones(1, 3),
        action_timestep=torch.ones(1, 24),
        state=torch.zeros(1, 1, 3),
        embodiment_id=torch.zeros(1, dtype=torch.long),
        cross_attention_tokens=torch.ones(1, 2, 4),
        negative_cross_attention_tokens=torch.zeros(1, 2, 4),
        persistent_prefill=None,
        recent_visual_prefill=None,
    )

    assert torch.all(output.video_velocity == 1)
    assert torch.all(output.action_velocity == 1)
    assert list(kernel.__dict__) == ["config"]


def test_training_noise_obeys_flow_matching_interpolation():
    kernel = _kernel()
    clean_video = torch.ones(2, 2, 3, 4, 4)
    clean_action = torch.ones(2, 24, 3)
    prepared = kernel.prepare_training_inputs(
        clean_video_latents=clean_video,
        clean_action=clean_action,
        generator=torch.Generator().manual_seed(11),
    )
    video_scale = (
        prepared.video_timestep.float()
        .div(kernel.config.num_train_timesteps)
        .view(2, 1, 3, 1, 1)
    )
    action_scale = (
        prepared.action_timestep.float()
        .div(kernel.config.num_train_timesteps)
        .view(2, 24, 1)
    )

    torch.testing.assert_close(
        prepared.noisy_video,
        clean_video + video_scale * prepared.video_velocity_target,
        atol=5e-3,
        rtol=0,
    )
    torch.testing.assert_close(
        prepared.noisy_action,
        clean_action + action_scale * prepared.action_velocity_target,
        atol=5e-3,
        rtol=0,
    )


def test_training_generator_advances_and_restores_deterministically():
    kernel = _kernel()
    clean_video = torch.ones(1, 2, 3, 4, 4)
    clean_action = torch.ones(1, 24, 3)
    generator = torch.Generator().manual_seed(23)

    first = kernel.prepare_training_inputs(
        clean_video_latents=clean_video,
        clean_action=clean_action,
        generator=generator,
    )
    checkpoint_state = generator.get_state()
    second = kernel.prepare_training_inputs(
        clean_video_latents=clean_video,
        clean_action=clean_action,
        generator=generator,
    )
    restored_generator = torch.Generator().set_state(checkpoint_state)
    restored = kernel.prepare_training_inputs(
        clean_video_latents=clean_video,
        clean_action=clean_action,
        generator=restored_generator,
    )

    assert not torch.equal(first.video_velocity_target, second.video_velocity_target)
    for field in (
        "noisy_video",
        "noisy_action",
        "video_timestep",
        "action_timestep",
        "video_velocity_target",
        "action_velocity_target",
    ):
        torch.testing.assert_close(getattr(second, field), getattr(restored, field))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_generator_device_conversion_advances_and_restores_source_state():
    kernel = _kernel()
    source = torch.Generator(device="cpu").manual_seed(29)
    cuda_generator = torch.Generator(device="cuda").manual_seed(31)

    assert (
        kernel._clone_generator(cuda_generator, device=torch.device("cuda"))
        is cuda_generator
    )

    first_generator = kernel._clone_generator(source, device=torch.device("cuda"))
    first = torch.randn(8, device="cuda", generator=first_generator)
    checkpoint_state = source.get_state()
    second_generator = kernel._clone_generator(source, device=torch.device("cuda"))
    second = torch.randn(8, device="cuda", generator=second_generator)

    restored_source = torch.Generator(device="cpu").set_state(checkpoint_state)
    restored_generator = kernel._clone_generator(
        restored_source, device=torch.device("cuda")
    )
    restored = torch.randn(8, device="cuda", generator=restored_generator)

    assert not torch.equal(first, second)
    torch.testing.assert_close(second, restored)


def test_training_timesteps_are_constant_per_causal_block():
    kernel = Wan22LegacyExactKernel(
        Wan22KernelConfig(
            num_train_timesteps=1000,
            num_inference_steps=2,
            num_frame_per_block=2,
            action_horizon=24,
        )
    )
    prepared = kernel.prepare_training_inputs(
        clean_video_latents=torch.ones(1, 2, 5, 4, 4),
        clean_action=torch.ones(1, 48, 3),
        generator=torch.Generator().manual_seed(7),
    )

    torch.testing.assert_close(
        prepared.video_timestep[:, 1],
        prepared.video_timestep[:, 2],
    )
    torch.testing.assert_close(
        prepared.video_timestep[:, 3],
        prepared.video_timestep[:, 4],
    )
    torch.testing.assert_close(
        prepared.action_timestep.reshape(1, 4, 12),
        prepared.video_timestep[:, 1:].unsqueeze(-1).expand(-1, -1, 12),
    )


def test_sampling_kernel_runs_cfg_and_returns_candidate_caches():
    kernel = _kernel()
    kernel._make_scheduler = lambda device: FakeScheduler()
    core = FakeCore()

    output, next_state = kernel.sample(
        core=core,
        i2v_condition=_condition(),
        anchor_latent=torch.ones(1, 2, 1, 4, 4),
        chunk_latents=torch.ones(1, 2, 1, 4, 4),
        state=torch.zeros(1, 1, 3),
        embodiment_id=torch.zeros(1, dtype=torch.long),
        cross_attention_tokens=torch.ones(1, 2, 4),
        negative_cross_attention_tokens=torch.zeros(1, 2, 4),
        persistent_prefill=None,
        inference_state=WAMInferenceState(),
        generator=torch.Generator().manual_seed(0),
    )

    assert output.action.shape == (1, 24, 3)
    assert output.video.shape == (1, 2, 3, 4, 4)
    assert next_state.current_start_frame == 3
    assert next_state.positive_kv_cache is not None
    assert next_state.negative_kv_cache is not None
    assert len(core.calls) == 6


def test_sampling_kernel_supports_single_48_action_block():
    kernel = _kernel(action_horizon=48)
    kernel._make_scheduler = lambda device: FakeScheduler()
    core = FakeCore()

    output, next_state = kernel.sample(
        core=core,
        i2v_condition=_condition(),
        anchor_latent=torch.ones(1, 2, 1, 4, 4),
        chunk_latents=torch.ones(1, 2, 1, 4, 4),
        state=torch.zeros(1, 1, 3),
        embodiment_id=torch.zeros(1, dtype=torch.long),
        cross_attention_tokens=torch.ones(1, 2, 4),
        negative_cross_attention_tokens=torch.zeros(1, 2, 4),
        persistent_prefill=None,
        inference_state=WAMInferenceState(),
        generator=torch.Generator().manual_seed(0),
    )

    assert output.action.shape == (1, 48, 3)
    assert output.video.shape == (1, 2, 3, 4, 4)
    assert next_state.current_start_frame == 3


def test_kernel_rebases_local_window_and_resets_single_frame_state():
    kernel = _kernel()
    core = FakeCore()
    state = WAMInferenceState(
        i2v_condition=_condition(),
        current_start_frame=2,
        positive_kv_cache=[torch.ones(1)],
    )

    local_boundary = kernel.prepare_inference_state(
        core=core,
        inference_state=state,
        observation_num_frames=4,
    )
    single_frame = kernel.prepare_inference_state(
        core=core,
        inference_state=state,
        observation_num_frames=1,
    )

    assert local_boundary.current_start_frame == 3
    assert local_boundary.rebase_observation_window is True
    assert local_boundary.positive_kv_cache is None
    assert single_frame == WAMInferenceState()


def test_sampling_kernel_rebuilds_observed_block_at_training_positions():
    kernel = _kernel()
    kernel._make_scheduler = lambda device: FakeScheduler()
    core = FakeCore()
    rebased = kernel.prepare_inference_state(
        core=core,
        inference_state=WAMInferenceState(current_start_frame=2),
        observation_num_frames=9,
    )

    output, next_state = kernel.sample(
        core=core,
        i2v_condition=_condition(),
        anchor_latent=torch.ones(1, 2, 1, 4, 4),
        chunk_latents=torch.ones(1, 2, 3, 4, 4),
        state=torch.zeros(1, 1, 3),
        embodiment_id=torch.zeros(1, dtype=torch.long),
        cross_attention_tokens=torch.ones(1, 2, 4),
        negative_cross_attention_tokens=torch.zeros(1, 2, 4),
        persistent_prefill=None,
        inference_state=rebased,
        generator=torch.Generator().manual_seed(0),
    )

    assert output.action.shape == (1, 24, 3)
    assert next_state.current_start_frame == 5
    assert next_state.rebase_observation_window is False
    assert [call[2]["current_start_frame"] for call in core.calls] == [
        0,
        0,
        1,
        1,
        3,
        3,
        3,
        3,
    ]


def test_single_chunk_model_restarts_at_training_positions_each_replan():
    kernel = _kernel()
    kernel._make_scheduler = lambda device: FakeScheduler()
    core = FakeCore()
    core.max_chunk_size = 1
    core.local_attn_size = 3

    prepared = kernel.prepare_inference_state(
        core=core,
        inference_state=WAMInferenceState(
            i2v_condition=_condition(),
            current_start_frame=3,
            positive_kv_cache=[torch.ones(1)],
        ),
        observation_num_frames=9,
    )

    assert prepared == WAMInferenceState()

    output, next_state = kernel.sample(
        core=core,
        i2v_condition=_condition(),
        anchor_latent=torch.ones(1, 2, 1, 4, 4),
        chunk_latents=torch.ones(1, 2, 3, 4, 4),
        state=torch.zeros(1, 1, 3),
        embodiment_id=torch.zeros(1, dtype=torch.long),
        cross_attention_tokens=torch.ones(1, 2, 4),
        negative_cross_attention_tokens=torch.zeros(1, 2, 4),
        persistent_prefill=None,
        inference_state=prepared,
        generator=torch.Generator().manual_seed(0),
    )

    assert output.action.shape == (1, 24, 3)
    assert next_state.current_start_frame == 3
    # Anchor stays at visual position 0. Every denoising call starts at
    # visual positions 1-2, which also makes action/state index equal to 0.
    assert [call[2]["current_start_frame"] for call in core.calls] == [
        0,
        0,
        1,
        1,
        1,
        1,
    ]


@pytest.mark.parametrize("max_chunk_size", [2, 4])
def test_multi_chunk_model_preserves_existing_causal_progress(max_chunk_size):
    kernel = _kernel()
    core = FakeCore()
    core.max_chunk_size = max_chunk_size
    core.local_attn_size = max_chunk_size * 2 + 1
    previous = WAMInferenceState(
        i2v_condition=_condition(),
        current_start_frame=3,
        positive_kv_cache=[torch.ones(1)],
    )

    prepared = kernel.prepare_inference_state(
        core=core,
        inference_state=previous,
        observation_num_frames=9,
    )

    assert prepared.current_start_frame == 3
    assert prepared.rebase_observation_window is False
    assert prepared.positive_kv_cache is not None
