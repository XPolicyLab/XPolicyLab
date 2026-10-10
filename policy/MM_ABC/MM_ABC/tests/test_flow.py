"""Regression checks for prediction defaults and velocity-space flow loss."""

from pathlib import Path

import pytest
import torch

from mmabc.models.flow import FlowConfig, RectifiedFlow
from mmabc.models.mmabc import load_model_config

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("path", sorted((REPO / "configs/model").glob("*.yaml")))
def test_model_configs_default_to_clean_prediction_and_velocity_loss(path):
    cfg = load_model_config(str(path), repo_root=str(REPO))
    assert cfg.flow.pred_type == "x"
    assert cfg.flow.loss_type == "v"
    assert FlowConfig().pred_type == "x"
    assert FlowConfig().loss_type == "v"


def test_clean_prediction_loss_matches_velocity_error():
    flow = RectifiedFlow(FlowConfig(normalize_weight=False))
    actions = torch.tensor([[[1.0, 2.0]], [[-1.0, 3.0]]])
    noise = torch.tensor([[[0.3, -0.2]], [[1.4, 0.7]]])
    t = torch.tensor([0.2, 0.8])
    x_t = flow.interpolate(actions, noise, t)
    prediction = actions + torch.tensor([[[0.1, -0.4]], [[0.2, 0.3]]])
    target, weight = flow.training_target(actions, noise, t)
    sample_loss = (prediction - target).square() * weight[:, None, None]
    velocity_loss = (flow.to_velocity(prediction, x_t, t) - (actions - noise)).square()
    torch.testing.assert_close(sample_loss, velocity_loss)


def test_velocity_prediction_remains_an_explicit_option():
    flow = RectifiedFlow(FlowConfig(pred_type="v"))
    actions = torch.randn(2, 3, 4)
    noise = torch.randn_like(actions)
    t = torch.tensor([0.1, 0.9])
    target, weight = flow.training_target(actions, noise, t)
    torch.testing.assert_close(target, actions - noise)
    torch.testing.assert_close(weight, torch.ones_like(t))
    torch.testing.assert_close(flow.to_velocity(target, actions, t), target)


def test_normalized_velocity_weights_are_finite_at_endpoint():
    flow = RectifiedFlow(FlowConfig())
    weight = flow.loss_weight(torch.tensor([0.0, 0.8, 1.0]))
    assert torch.isfinite(weight).all()
    torch.testing.assert_close(weight.mean(), torch.tensor(1.0))
    assert weight[0] < weight[1] < weight[2]
