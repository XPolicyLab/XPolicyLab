"""Contracts for full-head-history post-training."""

import dataclasses
import json
from types import SimpleNamespace

import pytest
import torch

from focus_vlwa.configs.model import FocusVLWAConfig, load_model_config
from focus_vlwa.configs.training import PostTrainingConfig, stage_defaults
from focus_vlwa.model.observation import FocusVLWAObservation
from focus_vlwa.model.prefix_encoder import VisualPrefixEncoder
from focus_vlwa.model.preprocessing import preprocess_observation_pytorch
from focus_vlwa.post_training.sampler import _ResumableDistributedSampler, batch_change_resume_cursor
from focus_vlwa.post_training.trainer import _combined_loss, _learning_rate


def test_stage_schedule_preserves_separate_decay_horizon(tmp_path):
    values = dict(dataset="data", init_checkpoint=tmp_path, output_dir=tmp_path, norm_stats_dir=tmp_path)
    joint = PostTrainingConfig(**values, **stage_defaults("joint"))
    frozen = PostTrainingConfig(**values, **stage_defaults("frozen"))
    assert (joint.num_steps, joint.warmup_steps, joint.decay_steps) == (5000, 200, 30000)
    assert (frozen.num_steps, frozen.warmup_steps, frozen.decay_steps) == (10000, 200, 10000)
    assert _learning_rate(0, joint) == pytest.approx(2.5e-5 / 201)
    assert _learning_rate(200, joint) == pytest.approx(2.5e-5)
    assert _learning_rate(10000, frozen) == pytest.approx(2.5e-6)
    assert _learning_rate(5000, joint) > 2.5e-6
    assert frozen.freeze_world_model_expert and frozen.world_model_loss_weight == 0


def test_action_only_loss_accepts_pruned_world_outputs():
    action = torch.tensor([[[2.0], [0.0]]], requires_grad=True)
    obs = SimpleNamespace(action_mask=torch.tensor([[1.0, 0.0]]))
    model = SimpleNamespace(
        world_model_loss_weight=0, world_model_event_loss_weight=1, world_model_state_loss_weight=0.3
    )
    loss, metrics = _combined_loss({"action": action}, obs, model, balance_world_model=True)
    assert loss.item() == 2 and metrics["world_model_state_loss"] == 0
    loss.backward()
    assert action.grad is not None


def test_sampler_continuation_skips_consumed_rows():
    sampler = _ResumableDistributedSampler(list(range(100)), 2, 0, shuffle=True, seed=42, drop_last=True)
    sampler.set_epoch(3)
    sequence = list(sampler)
    sampler.start_index = 18
    assert list(sampler) == sequence[18:]
    assert batch_change_resume_cursor(3, 3, 4, 2, 40) == (0, 12)
    assert batch_change_resume_cursor(17, 3, 4, 2, 40) == (1, 0)


def test_head_history_shares_augmentation_per_sample():
    torch.set_num_threads(1)
    image = torch.linspace(-1, 1, 224 * 224 * 3).reshape(1, 224, 224, 3).repeat(2, 1, 1, 1)
    obs = FocusVLWAObservation(
        images={
            key: image.permute(0, 3, 1, 2).clone() for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        },
        image_masks={},
        state=torch.zeros(2, 32),
        tokenized_prompt=torch.zeros(2, 4, dtype=torch.long),
        tokenized_prompt_mask=torch.ones(2, 4, dtype=torch.bool),
        history_images=image[:, None].repeat(1, 20, 1, 1, 1),
        history_mask=torch.ones(2, 20),
        world_state=torch.randn(2, 270, 32),
        event_action=torch.randn(2, 32),
    )
    torch.manual_seed(17)
    processed = preprocess_observation_pytorch(obs, train=True)
    for index in range(2):
        head = processed.images["base_0_rgb"][index].permute(1, 2, 0)
        torch.testing.assert_close(processed.history_images[index], head[None].expand(20, -1, -1, -1))
    assert not torch.equal(processed.history_images[0], processed.history_images[1])
    assert processed.world_state is obs.world_state
    assert processed.event_action is obs.event_action
    assert preprocess_observation_pytorch(obs, train=False).history_images is obs.history_images


class PrefixEncoder(VisualPrefixEncoder):
    config = FocusVLWAConfig(history_mode="head_history")

    def __init__(self):
        self.scale = torch.tensor(1.0, requires_grad=True)
        self.joint_experts = SimpleNamespace(embed_language_tokens=lambda tokens: torch.zeros(*tokens.shape, 2))

    def _embed_siglip(self, image):
        value = image.mean(dim=(1, 2, 3))[:, None, None]
        return value.expand(-1, 256, 2) * self.scale

    def _apply_checkpoint(self, function, *args):
        return function(*args)


def test_prefix_keeps_current_views_and_masks_invalid_history():
    encoder = PrefixEncoder()
    hist = torch.zeros(1, 20, 224, 224, 3)
    hist[:, -1] = 2
    valid = torch.zeros(1, 20)
    valid[:, -1] = 1
    obs = SimpleNamespace(history_images=hist, history_mask=valid)
    images = [torch.ones(1, 3, 224, 224) * value for value in (3, 4, 5)]
    tokens, mask, attention = encoder.embed_prefix(
        images,
        [torch.ones(1, dtype=torch.bool)] * 3,
        torch.zeros(1, 4, dtype=torch.long),
        torch.ones(1, 4, dtype=torch.bool),
        obs,
    )
    assert tokens.shape == (1, 1092, 2)
    assert mask[:, :304].sum() == 0 and mask.sum() == 16 + 768 + 4
    assert not attention.any()
    assert tokens[0, 304:320].mean() == 2 and tokens[0, 320:576].mean() == 3
    tokens.sum().backward()
    assert encoder.scale.grad.item() > 0
    with pytest.raises(ValueError, match="explicit history"):
        encoder.embed_prefix(images, [], None, None)


def test_head_profile_roundtrip(tmp_path):
    config = FocusVLWAConfig(history_mode="head_history")
    (tmp_path / "model_config.json").write_text(json.dumps(dataclasses.asdict(config)))
    assert load_model_config(tmp_path) == config
