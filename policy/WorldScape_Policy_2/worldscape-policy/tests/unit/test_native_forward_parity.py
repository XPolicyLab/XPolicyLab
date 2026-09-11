from __future__ import annotations

import pytest
import torch
from torch import nn

from worldscape_policy.conditioning.auto_conditioner import AutoConditioner
from worldscape_policy.conditioning.vlm.protocol import AutoPlanningFeatures
from worldscape_policy.data.schema import TrainingBatch
from worldscape_policy.memory.visual import VisualPrefillManager
from worldscape_policy.model_config import ModelConfig
from worldscape_policy.training.objective import ActionFlowLoss
from worldscape_policy.training.trainer import (
    NativeTrainer,
    NativeWan22BatchAdapter,
)
from worldscape_policy.types import (
    InteractionMode,
    ObservationBatch,
    PromptBatch,
)
from worldscape_policy.wam.wan22 import Wan22KernelConfig


class _CountingCodec(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def encode_visual(self, video):
        self.calls += 1
        return video.float().mean(dim=(-1, -2), keepdim=True)


def test_official_none_never_encodes_or_reuses_persistent_prefill():
    codec = _CountingCodec()
    manager = VisualPrefillManager(codec, persistent_prompt="none")
    state = manager.prepare(
        images=torch.ones(1, 1, 1, 3, 2, 2),
        prompts=PromptBatch(visual_prompt="none"),
    )

    assert state.persistent_prompt_latents is None
    assert codec.calls == 1  # recent observations only; no fake zero prompt


class _HistoryVLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.raw = nn.Parameter(torch.ones(1, 2, 4))

    def encode_planning(self, *, images, history_images, history_mask, **_):
        batch = images.shape[0]
        return AutoPlanningFeatures(
            perception_features=self.raw.expand(batch, -1, -1),
            history_perception_features=torch.ones(batch, 2, 1, 4),
            history_mask=history_mask,
        )


class _GateMemory(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Linear(4, 4, bias=False)
        self.seen_history = None

    def forward(self, current_tokens, *, history_tokens, history_mask, **_):
        self.seen_history = (history_tokens, history_mask)
        if history_tokens is None:
            return current_tokens, {}
        readout = history_tokens.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        return current_tokens + self.gate(readout), {}


def test_native_history_reaches_event_memory_and_gate_only_detaches_vlm():
    vlm = _HistoryVLM()
    memory = _GateMemory()
    conditioner = AutoConditioner(
        vlm=vlm,
        token_pooler=nn.Identity(),
        projector=nn.Linear(4, 4, bias=False),
        event_memory=memory,
        semantic_gate_only=True,
    )
    images = torch.ones(1, 3, 1, 3, 2, 2)
    history_mask = torch.tensor([[True, False]])
    output = conditioner(
        observation=ObservationBatch(
            images=images,
            head_view=images[:, :1, 0],
            proprioception=torch.zeros(1, 1, 3),
            embodiment_id=torch.zeros(1, dtype=torch.long),
            vlm_history_images=torch.ones(1, 2, 3, 2, 2),
            vlm_history_mask=history_mask,
        ),
        prompts=PromptBatch(vlm_planning_text=["plan"]),
        event_memory=None,
        training=True,
    )
    output.semantic_prediction.sum().backward()

    assert memory.seen_history is not None
    torch.testing.assert_close(memory.seen_history[1], history_mask)
    assert vlm.raw.grad is None
    assert conditioner.projector.weight.grad is not None
    assert memory.gate.weight.grad is not None


def test_auto_conditioner_casts_float_vlm_features_to_bfloat16_projector():
    conditioner = AutoConditioner(
        vlm=_HistoryVLM(),
        token_pooler=nn.Identity(),
        projector=nn.Linear(4, 4, bias=False).to(dtype=torch.bfloat16),
        event_memory=_GateMemory().to(dtype=torch.bfloat16),
    )
    images = torch.ones(1, 1, 1, 3, 2, 2)
    output = conditioner(
        observation=ObservationBatch(
            images=images,
            head_view=images[:, :1, 0],
            proprioception=torch.zeros(1, 1, 3),
            embodiment_id=torch.zeros(1, dtype=torch.long),
            vlm_history_images=torch.ones(1, 2, 3, 2, 2),
            vlm_history_mask=torch.ones(1, 2, dtype=torch.bool),
        ),
        prompts=PromptBatch(vlm_planning_text=["plan"]),
        event_memory=None,
        training=True,
    )

    assert output.cross_attention_tokens.dtype == torch.bfloat16


def test_semantic_projector_and_gate_gradients_clip_to_half_norm():
    auto = nn.Module()
    auto.semantic_gate_only = True
    auto.semantic_grad_clip_norm = 0.5
    auto.projector = nn.Linear(4, 4, bias=False)
    auto.event_memory = _GateMemory()
    policy = nn.Module()
    policy.condition_router = nn.Module()
    policy.condition_router.auto = auto
    for parameter in policy.parameters():
        parameter.grad = torch.full_like(parameter, 10.0)
    trainer = object.__new__(NativeTrainer)
    trainer.policy = policy

    trainer._apply_semantic_gate_clip()

    for parameter in policy.parameters():
        assert parameter.grad.norm().item() <= 0.500001


def test_native_adapter_and_objective_preserve_action_validity_fields():
    batch = TrainingBatch(
        episode_ids=("episode",),
        event_ids=("event",),
        observations={"video": torch.zeros(1, 3, 2, 2, 3)},
        observation_masks={"video": torch.ones(1, 3, dtype=torch.bool)},
        actions=torch.zeros(1, 24, 3),
        action_mask=torch.ones(1, 24, dtype=torch.bool),
        robot_state=torch.zeros(1, 1, 3),
        robot_state_mask=torch.ones(1, 1, dtype=torch.bool),
        high_level_instructions=("move",),
        event_instructions=(None,),
        embodiments=("toy",),
        modes=(InteractionMode.INTERACTIVE,),
        mode_mask=torch.zeros(1, dtype=torch.bool),
        action_dim_mask=torch.tensor([[True, False, True]]),
        has_real_action=torch.tensor([False]),
    )
    ready = NativeWan22BatchAdapter(
        video_latent_encoder=lambda video: video.movedim(2, 1),
        diffusion_video_preprocessor=lambda views: views[:, :, 0].float(),
        embodiment_ids={"toy": 0},
    )(batch)
    result = ActionFlowLoss()(
        torch.ones_like(ready.clean_action),
        torch.zeros_like(ready.clean_action),
        mask=ready.action_mask,
        dim_mask=ready.action_dim_mask,
        has_real_action=ready.has_real_action,
    )

    torch.testing.assert_close(ready.action_dim_mask, batch.action_dim_mask)
    torch.testing.assert_close(ready.has_real_action, batch.has_real_action)
    assert result.loss.item() == 0


def test_block_geometry_fails_closed_everywhere():
    with pytest.raises(ValueError, match="num_frame_per_block=2"):
        Wan22KernelConfig(
            num_train_timesteps=1000,
            num_inference_steps=16,
            num_frame_per_block=3,
            action_horizon=24,
        )

    config = {
        "schema_version": "1",
        "model": {
            "mode": "auto",
            "shape": {
                "num_frames": 3,
                "frame_block_size": 2,
                "actions_per_block": 24,
                "states_per_block": 1,
                "action_horizon": 24,
                "action_dim": 3,
                "max_state_dim": 3,
                "vlm_token_dim": 4,
                "condition_dim": 4,
            },
            "condition_router": {
                "auto": {
                    "vlm": {"target": "x.VLM", "parameters": {}},
                    "projector": {
                        "kind": "linear",
                        "input_dim": 4,
                        "output_dim": 4,
                    },
                    "output_norm": True,
                    "semantic_gate_only": True,
                    "semantic_grad_clip_norm": 0.5,
                },
                "interactive": {
                    "t5": {"target": "x.T5", "parameters": {}}
                },
            },
            "event_memory": {
                "enabled": False,
                "history_steps": 8,
                "global_slots": 1,
                "local_steps": 4,
                "boundary_steps": 8,
                "boundary_min_gap": 1,
                "perception_gist_tokens": 8,
                "residual_scale": 0.1,
                "dropout": 0.0,
            },
            "visual_memory": {
                "vae": {"target": "x.VAE", "parameters": {}},
                "image_encoder": {"target": "x.Image", "parameters": {}},
                "persistent_prompt": "none",
                "view_index": 0,
                "tiled": False,
                "tile_size": [34, 34],
                "tile_stride": [18, 16],
            },
            "wam": {
                "plugin": "wan22",
                "variant": "ti2v-5b",
                "core": {"target": "x.Core", "parameters": {}},
                "num_timestep_buckets": 1000,
                "train_architecture": "full",
                "decouple_inference_noise": False,
                "video_inference_final_noise": 0.8,
                "decouple_video_action_noise": False,
                "video_noise_beta_alpha": 3.0,
                "video_noise_beta_beta": 1.0,
                "use_high_noise_emphasis": False,
                "high_noise_beta_alpha": 3.0,
                "high_noise_beta_beta": 1.0,
            },
        },
    }
    assert ModelConfig.from_dict(config).model.visual_memory.persistent_prompt == "none"
    config["model"]["shape"]["actions_per_block"] = 23
    with pytest.raises(ValueError, match="actions_per_block"):
        ModelConfig.from_dict(config)
