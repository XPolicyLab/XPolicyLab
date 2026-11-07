"""CPU checks of scientifically important packaging contracts."""
from pathlib import Path
import copy
import importlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

POLICY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POLICY.parents[2]))
from XPolicyLab.policy.bagel_robotics_1 import assets, deploy
from XPolicyLab.policy.bagel_robotics_1.temporal_ensemble import TemporalEnsemble
from bagel_cp1000.models.wan21_14b.helpers.compact_channels import build


def observation(step=0):
    return {"env_idx": 0, "instruction": "stack blocks",
            "vision": {name: {"color": np.full((24, 32, 3), step, dtype=np.uint8)}
                       for name in ["cam_head", "cam_left_wrist", "cam_right_wrist"]},
            "state": {f"{side}_{key}": np.array(value, np.float32)
                      for side in ["left", "right"]
                      for key, value in [("ee_pose", [step, 0, 0, 1, 0, 0, 0]),
                                         ("ee_joint_state", [0.2])]}}


def test_frozen_settings_and_seed_override():
    cfg = assets.validate_settings({"seed": 42})
    assert cfg["seed"] == 42 and cfg["replan_steps"] == 16
    for key, value in [("replan_steps", 24), ("num_inference_steps", 8),
                       ("mixed_precision", "fp16"), ("eval_batch", True),
                       ("env_cfg_type", "piper"), ("seed", -1)]:
        with pytest.raises(ValueError):
            assets.validate_settings({key: value})


def test_blending_uses_aligned_raw_predictions_and_quaternion_hemisphere():
    def chunk(value):
        x = np.full((32, 16), value, np.float32)
        x[:, 3:7] = [1, 0, 0, 0]
        x[:, 11:15] = [1, 0, 0, 0]
        return x
    ensemble = TemporalEnsemble()
    first = chunk(1)
    first[16:, 0] = 4
    assert np.array_equal(ensemble.apply(first), first[:16])
    second = chunk(2)
    second[:, 3:7] *= -1
    result = ensemble.apply(second)
    assert np.allclose(result[:, 0], 2.2)
    assert np.allclose(np.linalg.norm(result[:, 3:7], axis=-1), 1)
    third = ensemble.apply(chunk(3))
    assert np.allclose(third[:, 0], 2.9)  # previous raw chunk, not previous blend


def test_actual_architecture_on_meta_device():
    # Optional real manifest from metadata download; requires no model tensors/GPU.
    path = POLICY / "checkpoints/cp1000/compact_architecture.json"
    if not path.exists():
        pytest.skip("Run download_checkpoint.py --metadata-only to check the real architecture")
    spec = json.loads(path.read_text())
    model = build(spec, device="meta")
    assert len(model.blocks) == 40
    assert sum(p.numel() for p in model.parameters()) == 12468409664
    assert [b.ffn[0].out_features for b in model.blocks] == [len(x) for x in spec["selection"]["ffn_indices"]]
    assert [b.cross_attn.num_heads for b in model.blocks] == [len(x) for x in spec["selection"]["cross_head_indices"]]
    assert not any("mask" in name for name, _ in model.named_parameters())
    malformed = copy.deepcopy(spec)
    malformed["selection"]["ffn_indices"][0][1] = malformed["selection"]["ffn_indices"][0][0]
    with pytest.raises(ValueError, match="Invalid per-layer"):
        build(malformed, device="meta")


def test_tampered_checkpoint_rejected_before_tensor_loading(tmp_path, monkeypatch):
    content = b"approved tensor bytes"
    (tmp_path / "video.pt").write_bytes(content)
    manifest = {"files": {"video.pt": {"size": len(content), "sha256": "0" * 64}}}
    monkeypatch.setattr(assets, "MANIFEST", manifest)
    with pytest.raises(ValueError, match="SHA-256"):
        assets.verify_bundle(tmp_path)
    (tmp_path / "video.pt").write_bytes(content[:3])
    with pytest.raises(ValueError, match="size mismatch"):
        assets.verify_bundle(tmp_path)


class FakePolicy:
    def __init__(self):
        self.action_horizon, self.replan_steps = 32, 16
        self.history_condition_frames = 8
        self._history_step_by_env = {}
        self.model = SimpleNamespace(video_expert=SimpleNamespace(
            parameters=lambda: [SimpleNamespace(numel=lambda: 12468409664)],
            named_parameters=lambda: []))

    def observe(self, obs):
        key = obs["env_idx"]
        self._history_step_by_env[key] = self._history_step_by_env.get(key, -1) + 1

    def _infer_action_chunk(self, obs, instruction):
        return np.repeat(obs["joint_action"]["vector"][None], 32, axis=0)

    def reset(self):
        self._history_step_by_env.clear()

    def reset_env(self, key):
        self._history_step_by_env.pop(key, None)


def make_adapter(monkeypatch):
    module = importlib.import_module("XPolicyLab.policy.bagel_robotics_1.model")
    runtime = importlib.import_module("bagel_cp1000.policy")
    cfg = assets.validate_settings({"checkpoint_path": "/unused/action.pt", "dataset_stats_path": "/unused/stats.json"})
    monkeypatch.setattr(module, "prepare_model_config", lambda c, p: cfg)
    monkeypatch.setattr(module, "configure_backend", lambda seed: {})
    monkeypatch.setattr(module, "get_robot_action_dim_info", lambda _: {"arm_dim": [6, 6], "ee_dim": [1, 1]})
    monkeypatch.setattr(runtime, "get_model", lambda _: FakePolicy())
    return module.Model(cfg)


def test_reset_restores_first_episode_behavior_with_client_ids(monkeypatch):
    model = make_adapter(monkeypatch)
    obs = observation()
    obs[model.CLIENT_ID_FIELD] = "client-a"
    model.update_obs(obs)
    initial = model.get_action({model.CLIENT_ID_FIELD: "client-a"})
    assert len(initial) == 16
    assert set(initial[0]) == {"left_ee_pose", "right_ee_pose", "left_ee_joint_state", "right_ee_joint_state"}
    model.reset()
    assert not model._ensemble_by_virtual and not model.model._history_step_by_env
    model.update_obs(obs)
    after = model.get_action({model.CLIENT_ID_FIELD: "client-a"})
    for a, b in zip(initial, after):
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
    assert model.history_guard_mismatches == 0


def test_deployment_records_each_step_and_stops_at_episode_end(monkeypatch):
    model = make_adapter(monkeypatch)
    class Environment:
        step = 0
        observations = []
        def get_obs(self):
            self.observations.append(self.step)
            return observation(self.step)
        def is_episode_end(self):
            return self.step >= 20
        def take_action(self, action):
            assert self.step < 20
            self.step += 1
    class Client:
        def call(self, func_name, **kwargs):
            return getattr(model, func_name)(**kwargs)
    env = Environment()
    deploy.eval_one_episode(env, Client())
    assert env.step == 20
    assert env.observations == list(range(20))
    assert model.history_guard_mismatches == 0
    assert model._last_ensemble["overlap"] is True
