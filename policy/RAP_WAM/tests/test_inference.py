"""Inference-contract regressions; no training or simulation is started."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from XPolicyLab.policy.RAP_WAM.common import build_tshape, instruction_text
from XPolicyLab.policy.RAP_WAM.model import Model
from XPolicyLab.policy.RAP_WAM.language import LanguageContext
from XPolicyLab.policy.RAP_WAM.runtime import VENDOR, historical_module
from XPolicyLab.policy.RAP_WAM.assets import manifest_fingerprint, tensor_manifest
from XPolicyLab.utils.process_data import pack_robot_state, unpack_robot_state

DIMS = {"arm_dim": [6, 6], "ee_dim": [1, 1]}


def obs(idx=0, instruction="Stack the bowls."):
    colors = ((100, 10, 1), (2, 110, 20), (30, 3, 120))
    frames = [np.full((48, 64, 3), c, dtype=np.uint8) for c in colors]
    for a in frames:
        a.flags.writeable = False
    return {"env_idx": idx, "instruction": instruction,
            "vision": {n: {"color": a} for n, a in zip(("cam_head", "cam_left_wrist", "cam_right_wrist"), frames)},
            "state": unpack_robot_state(np.arange(14, dtype=np.float32), "joint", DIMS)}


def adapter():
    obj = Model.__new__(Model)
    obj.action_type, obj.dim_info, obj.action_dim, obj.action_horizon = "joint", DIMS, 14, 16
    obj.replan_steps, obj.savgol_window, obj.savgol_polyorder, obj.boundary_blend_steps = 8, 9, 2, 4
    obj.rgb_processor = historical_module("psi_policy.dataset.preprocessors.rgb").RGBPreprocessor(image_size=(384, 320))
    obj.state_processor = historical_module("psi_policy.dataset.preprocessors.low_dim").LowDimPreprocessor()
    obj.lang = LanguageContext(["Stack the bowls."], Path("/unused/wan"), None)
    obj.policy = SimpleNamespace(device="cpu", _cur_task_index=None)
    obj.reset()
    return obj


def packed(chunk):
    return np.stack([pack_robot_state({"state": a}, "joint", DIMS) for a in chunk])


def test_camera_layout_rgb_and_readonly_input():
    o = obs()
    images = [o["vision"][n]["color"] for n in ("cam_head", "cam_left_wrist", "cam_right_wrist")]
    result = build_tshape(*images)
    assert result.shape == (384, 320, 3)
    for pixel, color in [((30, 30), [100, 10, 1]), ((300, 30), [2, 110, 20]), ((300, 200), [30, 3, 120])]:
        np.testing.assert_array_equal(result[pixel], color)
    assert all(not a.flags.writeable for a in images)


def test_encoded_images_are_rejected_in_adapter():
    with pytest.raises(ValueError, match="decoded"):
        build_tshape(np.zeros(20, np.uint8), np.zeros((2, 2, 3), np.uint8), np.zeros((2, 2, 3), np.uint8))


def test_historical_preprocessor_layout_and_joint_order():
    result = adapter()._process_obs(obs())
    assert result["rgb_head"].shape == (1, 3, 384, 320)
    torch.testing.assert_close(result["robot_state"], torch.arange(14).float()[None], rtol=0, atol=0)
    np.testing.assert_allclose(result["rgb_head"][0, :, 30, 30], np.array([100, 10, 1]) / 255, atol=1e-8)


def test_savgol_preserves_quadratic_trajectory():
    obj = adapter()
    actions = np.repeat((np.arange(16, dtype=np.float32) ** 2)[:, None], 14, axis=1)
    result = packed(obj._postprocess(actions.copy(), 7))
    np.testing.assert_allclose(result, actions[:8], rtol=0, atol=1e-5)
    assert result.dtype == np.float32


def test_blending_uses_old_continuation_and_retains_preblend():
    obj = adapter()
    old = np.tile(np.arange(16, dtype=np.float32)[:, None], (1, 14)); new = old + 100
    obj._postprocess(old.copy(), 7)
    result = packed(obj._postprocess(new.copy(), 7))
    alpha = np.arange(1, 5, dtype=np.float32) / 5
    expected = alpha[:, None] * new[:4] + (1 - alpha[:, None]) * old[8:12]
    np.testing.assert_allclose(result[:4], expected, rtol=0, atol=1e-5)
    np.testing.assert_allclose(result[4:], new[4:8], rtol=0, atol=1e-5)
    np.testing.assert_allclose(obj._prev_chunk[7], new, rtol=0, atol=1e-5)


def test_blending_states_do_not_cross_environments_and_reset():
    obj = adapter()
    obj._postprocess(np.zeros((16, 14), np.float32), 4)
    np.testing.assert_array_equal(packed(obj._postprocess(np.ones((16, 14), np.float32), 9)), np.ones((8, 14), np.float32))
    obj.reset()
    assert not obj._prev_chunk and not obj._obs_processed_batch
    with pytest.raises(ValueError, match="update_obs"):
        obj.get_action()


def test_batch_retains_requested_order_and_rejects_duplicates():
    obj = adapter(); obj.update_obs_batch([obs(9), obs(4)])
    obj._infer_one = lambda x, instruction: np.full((16, 14), float(x["robot_state"][0, 0]), np.float32)
    obj._obs_processed_batch[9]["robot_state"][0, 0] = 9
    obj._obs_processed_batch[4]["robot_state"][0, 0] = 4
    results = obj.get_action_batch([4, 9])
    assert packed(results[0])[0, 0] == 4 and packed(results[1])[0, 0] == 9
    with pytest.raises(ValueError, match="Duplicate"):
        obj.update_obs_batch([obs(1), obs(1)])
    with pytest.raises(ValueError, match="Missing"):
        obj.get_action_batch([99])


def test_context_matching_and_instruction_precedence():
    obj = adapter()
    assert obj.lang.resolve_index(" STACK THE BOWLS. ") == 0
    assert obj.lang.resolve_index("unknown task") is None
    assert instruction_text({"instruction": ["task", "alternative"]}) == "task"
    assert instruction_text({"task_instruction": None, "instruction": "task"}) == "task"
    assert instruction_text({"task_instruction": "historical", "instruction": "new"}) == "historical"
    with pytest.raises(ValueError):
        instruction_text({"instruction": " "})


def test_trained_forward_uses_saved_row_and_cleans_index():
    obj = adapter(); seen = []
    def predict(x):
        seen.append(int(obj.policy._cur_task_index.item()))
        assert x["rgb_head"].shape == (1, 1, 3, 384, 320)
        return {"action_pred": torch.ones(1, 16, 14)}
    obj.policy.predict_action = predict
    assert obj._infer_one(obj._process_obs(obs()), "Stack the bowls.").shape == (16, 14)
    assert seen == [0] and obj.policy._cur_task_index is None


def test_zero_shot_context_restored_on_forward_failure():
    obj = adapter()
    old = lambda size: [torch.zeros(0, 4096)] * size
    obj.policy._build_context_list = old
    obj.policy.model = SimpleNamespace(text_embedding=[SimpleNamespace(weight=torch.ones(1))])
    obj.lang.encode_zero_shot = lambda instruction, device, dtype: torch.ones(2, 4096)
    def fail(x):
        assert torch.equal(obj.policy._build_context_list(1)[0], torch.ones(2, 4096))
        raise RuntimeError("intentional forward failure")
    obj.policy.predict_action = fail
    with pytest.raises(RuntimeError, match="intentional"):
        obj._infer_one(obj._process_obs(obs()), "unseen instruction")
    assert obj.policy._build_context_list is old and obj.policy._cur_task_index is None


def test_vendor_sources_match_historical_capture():
    record = json.loads((VENDOR / "SOURCE_MANIFEST.json").read_text())
    for name, digest in record["sha256"].items():
        expected = record.get("packaged_sha256", {}).get(name, digest)
        assert hashlib.sha256((VENDOR / name).read_bytes()).hexdigest() == expected, name


def test_tensor_fingerprint_detects_dtype_shape_and_value_changes():
    a = {"weight": torch.arange(8).reshape(2, 4).float()}
    expected = manifest_fingerprint(tensor_manifest(a))
    for changed in [a["weight"].double(), a["weight"].reshape(4, 2), a["weight"] + 1]:
        assert manifest_fingerprint(tensor_manifest({"weight": changed})) != expected


def test_benchmark_seed_does_not_reseed_policy_rng(monkeypatch):
    import XPolicyLab.policy.RAP_WAM.model as module
    fake = SimpleNamespace(action_horizon=16)
    monkeypatch.setattr(module, "get_robot_action_dim_info", lambda robot: DIMS)
    monkeypatch.setattr(module, "load_policy", lambda *a: (fake, ["Stack the bowls."], Path("/unused"), Path("/unused/wan")))
    torch.manual_seed(12345)
    before = torch.get_rng_state().clone()
    module.Model(dict(bench_name="RoboDojo", env_cfg_type="arx_x5", action_type="joint", seed=999))
    assert torch.equal(torch.get_rng_state(), before)
