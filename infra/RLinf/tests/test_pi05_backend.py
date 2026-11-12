"""Exercise v0.4's actual sampler/recompute with a small differentiable velocity field."""

import json
import random

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from XPolicyLab.policy.Pi_05.rl import Pi05RLConfig, Pi05RLPolicy, load_weights
from rlinf.models.embodiment.openpi.modules.utils import set_torch_compile


class SmallFlow(Pi05RLPolicy):
    """Small CPU network; inherits production sampling and logprob code unchanged."""

    def __init__(
        self, method="flow_sde", joint=False, value=False, value_after_vlm=True
    ):
        nn.Module.__init__(self)
        self.action_out_proj = nn.Linear(4, 4)
        self.action_horizon = 3
        self.action_chunk = 2
        self.action_env_dim = 3
        self.model_action_dim = 4
        self.num_steps = 3
        self.rl_cfg = Pi05RLConfig.from_dict(
            {
                "num_steps": 3,
                "action_chunk": 2,
                "action_env_dim": 3,
                "noise_method": method,
                "joint_logprob": joint,
                "add_value_head": value,
                "value_after_vlm": value_after_vlm,
            }
        ).native()
        if value:
            self.value_head = nn.Linear(4, 1)
        if method == "flow_noise":
            self.noise_head = nn.Sequential(nn.Linear(4, 4), nn.Sigmoid())

    def build_prefix_cache(self, observation):
        prefix = observation.state[:, None, :]
        return prefix, torch.ones(prefix.shape[:2], dtype=torch.bool), None

    def run_suffix(self, observation, x_t, timestep, cache, prefix_mask):
        return x_t + observation.state[:, None, :] * 0.1

    def decode_actions(self, actions, state):
        return actions[:, : self.action_chunk, : self.action_env_dim]


def inputs():
    return {
        "state": torch.ones(2, 4),
        "image/base_0_rgb": torch.zeros(2, 224, 224, 3),
        "image/left_wrist_0_rgb": torch.zeros(2, 224, 224, 3),
        "image/right_wrist_0_rgb": torch.zeros(2, 224, 224, 3),
        "image_mask/base_0_rgb": torch.ones(2, dtype=torch.bool),
        "image_mask/left_wrist_0_rgb": torch.ones(2, dtype=torch.bool),
        "image_mask/right_wrist_0_rgb": torch.ones(2, dtype=torch.bool),
        "tokenized_prompt": torch.ones(2, 3, dtype=torch.long),
        "tokenized_prompt_mask": torch.ones(2, 3, dtype=torch.bool),
    }


@pytest.mark.parametrize("method", ["flow_sde", "flow_cps", "flow_noise"])
@pytest.mark.parametrize(
    "joint,value_after_vlm", [(False, True), (True, True), (True, False)]
)
def test_sample_recompute_and_real_update(method, joint, value_after_vlm):
    random.seed(7)
    torch.manual_seed(7)
    set_torch_compile(False)
    policy = SmallFlow(method, joint, value=True, value_after_vlm=value_after_vlm)
    obs = inputs()
    actions, extras = policy.sample_preprocessed(obs)
    assert actions.shape == (2, 2, 3)
    assert extras["prev_values"].shape == (2, 1)
    replay = extras["forward_inputs"]
    obs["state"].fill_(100)  # replay must own its observation
    result = policy.default_forward(replay, compute_values=True)
    torch.testing.assert_close(
        result["logprobs"], extras["prev_logprobs"], atol=1e-6, rtol=1e-5
    )
    torch.testing.assert_close(result["values"], extras["prev_values"].squeeze(-1))
    assert result["entropy"].shape == (2, 1)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.01)
    loss = -result["logprobs"].mean() + result["values"].square().mean()
    loss.backward()
    assert policy.action_out_proj.weight.grad.abs().sum() > 0
    optimizer.step()
    updated = policy.default_forward(replay)["logprobs"]
    assert not torch.allclose(updated, extras["prev_logprobs"])
    assert torch.isfinite(updated).all()


def test_grpo_actor_update_without_critic():
    from rlinf.algorithms.registry import policy_loss

    torch.manual_seed(17)
    policy = SmallFlow(value=False)
    _, rollout = policy.sample_preprocessed(inputs(), compute_values=False)
    old = rollout["prev_logprobs"]
    replay = rollout["forward_inputs"]
    current = policy.default_forward(replay, compute_values=False)["logprobs"]
    torch.testing.assert_close(current, old)
    loss, _ = policy_loss(
        loss_type="actor", task_type="embodied", reward_type="chunk_level",
        logprob_type="chunk_level", single_action_dim=3,
        logprobs=current, old_logprobs=old, advantages=torch.tensor([1.0, -1.0]),
        loss_mask=torch.ones(2, dtype=torch.bool), clip_ratio_low=0.2, clip_ratio_high=0.2,
    )
    loss.backward()
    assert torch.isfinite(policy.action_out_proj.weight.grad).all()
    torch.optim.SGD(policy.parameters(), lr=1e-3).step()
    updated = policy.default_forward(replay, compute_values=False)["logprobs"]
    assert not torch.allclose(updated, old)


def test_sampler_rejects_fake_deterministic_training():
    with pytest.raises(ValueError, match="stochastic"):
        SmallFlow("flow_ode").sample_preprocessed(inputs())


@pytest.mark.parametrize(
    "cfg",
    [
        {"safe_get_logprob": True},
        {"value_hidden_sizes": [4]},
        {"num_steps": 0},
        {"noise_method": "bad"},
        {"unknown": 1},
    ],
)
def test_reject_unsupported_settings(cfg):
    with pytest.raises(ValueError):
        Pi05RLConfig.from_dict(cfg)


def tiny_backbone():
    model = nn.Module()
    model.action_out_proj = nn.Linear(2, 2)
    return model


def test_wrapped_full_checkpoint_and_missing_weights(tmp_path):
    source, target = tiny_backbone(), tiny_backbone()
    path = tmp_path / "full_weights.pt"
    torch.save({"backbone." + k: v for k, v in source.state_dict().items()}, path)
    load_weights(target, tmp_path)
    for a, b in zip(source.parameters(), target.parameters()):
        torch.testing.assert_close(a, b)
    torch.save({"backbone.action_out_proj.bias": source.action_out_proj.bias}, path)
    with pytest.raises(RuntimeError, match="missing"):
        load_weights(target, path)


def test_sharded_checkpoint_and_missing_shard(tmp_path):
    source, target = tiny_backbone(), tiny_backbone()
    mapping = {}
    for i, (key, value) in enumerate(source.state_dict().items()):
        name = f"shard-{i}.safetensors"
        save_file({key: value}, tmp_path / name)
        mapping[key] = name
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": mapping})
    )
    load_weights(target, tmp_path)
    for a, b in zip(source.parameters(), target.parameters()):
        torch.testing.assert_close(a, b)
    (tmp_path / "shard-0.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="shard"):
        load_weights(target, tmp_path)


def test_legacy_vision_stem_conversion(tmp_path):
    target = nn.Module()
    target.img = nn.Module()
    target.img.stem = nn.Linear(2, 3)
    prefix = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding."
    old = {
        prefix + k: torch.randn_like(v) for k, v in target.img.stem.state_dict().items()
    }
    save_file(old, tmp_path / "model.safetensors")
    load_weights(target, tmp_path)
    torch.testing.assert_close(target.img.stem.weight, old[prefix + "weight"])


def test_reject_suffix_critic_with_single_step_logprob():
    with pytest.raises(ValueError, match="suffix values"):
        Pi05RLConfig.from_dict(
            {"add_value_head": True, "value_after_vlm": False, "joint_logprob": False}
        )


def test_uint8_rgb_is_normalized_once_for_sampling_and_replay():
    from XPolicyLab.policy.Pi_05.rl import observation_from_inputs, replay_observation

    obs = inputs()
    obs["image/base_0_rgb"] = (
        torch.tensor([255, 0, 128], dtype=torch.uint8).expand(2, 224, 224, 3).clone()
    )
    normalized = observation_from_inputs(obs)
    replay = replay_observation(normalized)
    expected = torch.tensor([1.0, -1.0, 128 / 255 * 2 - 1])
    torch.testing.assert_close(normalized.images["base_0_rgb"][0, 0, 0], expected)
    rebuilt = SmallFlow()._observation_from_forward_inputs(replay)
    torch.testing.assert_close(
        rebuilt.images["base_0_rgb"], normalized.images["base_0_rgb"]
    )
    obs["image/base_0_rgb"].zero_()
    torch.testing.assert_close(replay["obs_image__base_0_rgb"][0, 0, 0], expected)
