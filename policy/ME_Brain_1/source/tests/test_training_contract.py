import json
from types import SimpleNamespace

import pytest
import torch

from focus_vlwa.configs.model import FocusVLWAConfig, load_model_config
from focus_vlwa.configs.training import PostTrainingConfig
from focus_vlwa.model.world_model import WorldModelExpert
from focus_vlwa.post_training.trainer import _combined_loss, _save_checkpoint


def test_freezing_world_expert_keeps_action_trainable():
    class Model(WorldModelExpert, torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.use_world_model = True
            self.joint_experts = torch.nn.Module()
            self.joint_experts.action_expert = torch.nn.Linear(2, 2)
            self.joint_experts.world_model_expert = torch.nn.Linear(2, 2)
            self.world_model_input_projection = torch.nn.Linear(2, 2)
            self.world_model_output_projection = torch.nn.Linear(2, 2)

    model = Model()
    assert model.freeze_world_model_expert() == 6
    assert all(p.requires_grad for p in model.joint_experts.action_expert.parameters())
    assert not any(p.requires_grad for p in model.joint_experts.world_model_expert.parameters())
    assert model.use_world_model


def test_zero_world_weight_has_no_world_loss_gradient():
    action = torch.tensor([[[2.0], [0.0]]], requires_grad=True)
    event = torch.tensor([[[5.0]]], requires_grad=True)
    world = torch.tensor([[[10.0], [0.0]]], requires_grad=True)
    model = SimpleNamespace(world_model_loss_weight=0, world_model_event_loss_weight=0, world_model_state_loss_weight=0)
    obs = SimpleNamespace(
        action_mask=torch.tensor([[1.0, 0.0]]),
        event_action_mask=torch.ones(1),
        world_state_mask=torch.tensor([[1.0, 0.0]]),
    )
    loss, _ = _combined_loss(
        {"action": action, "world_model_event": event, "world_model_state": world}, obs, model, balance_world_model=True
    )
    loss.backward()
    assert loss.item() == 2
    assert event.grad.count_nonzero() == world.grad.count_nonzero() == 0


def test_checkpoint_preserves_profiles_and_refuses_overwrite(tmp_path):
    stats = tmp_path / "stats"
    stats.mkdir()
    (stats / "norm_stats.json").write_text("{}")
    config = PostTrainingConfig(
        dataset="test",
        init_checkpoint=tmp_path,
        output_dir=tmp_path / "output",
        norm_stats_dir=stats,
        freeze_world_model_expert=True,
    )
    model = torch.nn.Linear(2, 2)
    model.config = FocusVLWAConfig(max_token_len=400)
    optimizer = torch.optim.AdamW(model.parameters())
    _save_checkpoint(model, optimizer, 1, config)
    assert load_model_config(config.output_dir / "1") == model.config
    metadata = json.loads((config.output_dir / "1/metadata.json").read_text())
    assert metadata["training_config"]["freeze_world_model_expert"]
    with pytest.raises(FileExistsError):
        _save_checkpoint(model, optimizer, 1, config)
