"""Run with pytest in the LeRobot policy environment."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from XPolicyLab.policy.LeRobot import model as adapter


@pytest.fixture
def model(monkeypatch, tmp_path):
    dims = {"arm_dim": [6, 6], "ee_dim": [1, 1]}
    config = SimpleNamespace(
        type="dm05",
        image_features={"observation.images.cam_high": None},
        output_features={"action": SimpleNamespace(shape=(14,))},
    )
    policy = Mock()
    policy.eval.return_value = policy
    policy.select_action.side_effect = lambda batch: batch["observation.state"].unsqueeze(0)
    processor = Mock(side_effect=lambda batch: batch)
    monkeypatch.setattr(adapter, "get_robot_action_dim_info", lambda _: dims)
    monkeypatch.setattr(adapter.PreTrainedConfig, "from_pretrained", lambda _: config)
    monkeypatch.setattr(adapter, "get_policy_class", lambda _: SimpleNamespace(from_pretrained=lambda *a, **kw: policy))
    monkeypatch.setattr(
        adapter, "make_pre_post_processors", lambda *a, **kw: (processor, Mock(side_effect=lambda x: x))
    )
    monkeypatch.setattr(adapter, "bind_relative_anchor", Mock())
    return adapter.Model(
        {
            "action_type": "joint",
            "env_cfg_type": "arx_x5",
            "task_name": "cover_blocks",
            "model_path": str(tmp_path),
            "device": "cpu",
            "camera_mapping": {"observation.images.cam_high": "cam_head"},
        }
    )


def observation():
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    image[..., 0] = 255
    image.setflags(write=False)
    return {
        "instruction": "cover the block",
        "vision": {"cam_head": {"color": image}},
        "state": {
            "left_arm_joint_state": np.arange(6, dtype=np.float32),
            "left_ee_joint_state": [6.0],
            "right_arm_joint_state": np.arange(7, 13, dtype=np.float32),
            "right_ee_joint_state": [13.0],
        },
    }


def test_per_frame_public_api_rgb_and_action_mapping(model):
    for _ in range(51):
        model.update_obs(observation())
        (action,) = model.get_action()
        assert action["right_ee_joint_state"].tolist() == [13.0]
    assert model.policy.select_action.call_count == 51
    batch = model.preprocessor.call_args.args[0]
    assert batch["task"] == "cover the block"
    torch.testing.assert_close(batch["observation.state"], torch.arange(14, dtype=torch.float32))
    image = batch["observation.images.cam_high"]
    assert image.shape == (3, 4, 5)
    assert image[0].eq(1).all() and image[1:].eq(0).all()
    with pytest.raises(RuntimeError, match="once per control frame"):
        model.get_action()


def test_reset_and_single_environment_contract(model):
    model.update_obs_batch([observation()])
    assert len(model.get_action_batch([7])) == 1
    model.update_obs(observation())
    model.reset()
    assert model.observation is None
    assert model.policy.reset.call_count == 2
    assert model.preprocessor.reset.call_count == 2
    assert model.postprocessor.reset.call_count == 2
    with pytest.raises(ValueError, match="one environment"):
        model.update_obs_batch([observation(), observation()])
    with pytest.raises(ValueError, match="one environment"):
        model.get_action_batch([0, 1])


def test_invalid_rgb_is_rejected(model):
    obs = observation()
    obs["vision"]["cam_head"]["color"] = np.zeros((3, 4, 5), dtype=np.float32)
    model.update_obs(obs)
    with pytest.raises(ValueError, match="HWC uint8 RGB"):
        model.get_action()
