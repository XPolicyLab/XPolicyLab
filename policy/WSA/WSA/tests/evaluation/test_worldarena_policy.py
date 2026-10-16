from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from evaluation.worldarena.wsa_base_adapter import (
    RuntimeSettings,
    WSABaseWorldArenaAdapter,
    _resolve_checkpoint_dir,
    build_transform_stats,
    checkpoint_model_type,
    extract_agilex_state,
    extract_franka_state,
    normalize_pose_quaternion,
    select_feature_stats,
    stats_key_candidates,
    validate_current_image_offsets,
)


def _clean_worldarena_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.startswith("WSA_") or name.startswith("WORLDARENA_"):
            monkeypatch.delenv(name, raising=False)


def _rgb() -> np.ndarray:
    return np.zeros((4, 5, 3), dtype=np.uint8)


def _agilex_images() -> dict[str, np.ndarray]:
    return {
        "cam_high": _rgb(),
        "cam_left_wrist": _rgb(),
        "cam_right_wrist": _rgb(),
        "cam_wrist": _rgb(),
    }


def _franka_images() -> dict[str, np.ndarray]:
    return {"cam_high": _rgb(), "cam_left_wrist": _rgb()}


def test_quaternion_normalize_preserves_xyzw_order() -> None:
    xyzw = np.asarray([1, 2, 3, 4, 1, 2, 3, 0.5], dtype=np.float32)
    normalized = normalize_pose_quaternion(xyzw)
    np.testing.assert_array_equal(normalized[:3], xyzw[:3])
    np.testing.assert_array_equal(normalized[7:], xyzw[7:])
    np.testing.assert_allclose(
        normalized[3:7], xyzw[3:7] / np.linalg.norm(xyzw[3:7])
    )
    assert np.linalg.norm(normalized[3:7]) == pytest.approx(1.0)
    with pytest.raises(FloatingPointError, match="near-zero quaternion"):
        normalize_pose_quaternion(np.zeros(8, dtype=np.float32))


def test_agilex_state_strictly_concatenates_official_per_arm_qpos() -> None:
    left = np.arange(7, dtype=np.float32)
    right = np.arange(7, 14, dtype=np.float32)
    obs = {
        "joint_qpos_left": left,
        "joint_qpos_right": right,
        # The official bridge exposes this single-arm alias as well. It must not
        # replace either of the two required per-arm vectors.
        "joint_qpos": right,
    }
    np.testing.assert_array_equal(
        extract_agilex_state(obs), np.concatenate([left, right])
    )

    with pytest.raises(KeyError, match="joint_qpos_left"):
        extract_agilex_state({"joint_qpos": np.concatenate([left, right])})
    with pytest.raises(KeyError, match="joint_qpos_right"):
        extract_agilex_state({"joint_qpos_left": left, "joint_qpos": right})
    with pytest.raises(ValueError, match="exact shape"):
        extract_agilex_state(
            {"joint_qpos_left": left[None, :], "joint_qpos_right": right}
        )


def test_franka_state_only_uses_documented_xyzw_pose_and_joint_gripper() -> None:
    obs = {
        "left_end_pose": np.asarray([1, 2, 3, 0.9, 0.1, 0.2, 0.3], dtype=np.float32),
        "joint_qpos": np.asarray([0, 1, 2, 3, 4, 5, 6, 0.75], dtype=np.float32),
        "state": np.zeros(32, dtype=np.float32),
    }
    state = extract_franka_state(obs)
    np.testing.assert_allclose(state, [1, 2, 3, 0.9, 0.1, 0.2, 0.3, 0.75])

    with pytest.raises(KeyError, match="left_end_pose"):
        extract_franka_state({"end_pose": obs["left_end_pose"], "joint_qpos": obs["joint_qpos"]})
    with pytest.raises(ValueError, match="exact shape"):
        extract_franka_state({"left_end_pose": obs["left_end_pose"], "joint_qpos": np.zeros(7)})


def test_worldarena_requires_two_current_conditioning_frames() -> None:
    assert validate_current_image_offsets([0, 0, 15]) == (0, 0, 15)
    with pytest.raises(ValueError, match=r"\[0, 0, future\]"):
        validate_current_image_offsets([-15, 0, 15])


def test_nested_stats_selection_builds_native_transform_stats() -> None:
    payload = {
        "cobot_magic_max": {
            "observation.state": {
                "mean": [1.0] * 14,
                "std": [2.0] * 14,
            },
            "action": {"mean": [3.0] * 14, "std": [4.0] * 14},
        },
        "other": {
            "observation.state": {"mean": [0.0], "std": [1.0]},
        },
    }
    selected_key, feature_stats = select_feature_stats(payload, "cobot_magic_max")
    assert selected_key == "cobot_magic_max"
    keys, norm_stats = build_transform_stats(
        feature_stats,
        explicit_keys=(),
        candidate_groups=(("observation.state",),),
        expected_dim=14,
        mode="mean_std",
        label="state",
        canonical_key="observation.state",
    )
    assert keys == ("observation.state",)
    np.testing.assert_array_equal(norm_stats["observation.state"]["mean"], [1.0] * 14)
    np.testing.assert_array_equal(norm_stats["observation.state"]["std"], [2.0] * 14)


def test_franka_does_not_auto_accept_ambiguous_flat_joint_stats() -> None:
    feature_stats = {
        "observation.state": {"mean": [0.0] * 8, "std": [1.0] * 8},
        "action": {"mean": [0.0] * 8, "std": [1.0] * 8},
    }
    state_groups, action_groups = stats_key_candidates("franka")
    with pytest.raises(KeyError, match="WSA_STATE_STATS_KEYS"):
        build_transform_stats(
            feature_stats,
            explicit_keys=(),
            candidate_groups=state_groups,
            expected_dim=8,
            mode="mean_std",
            label="state",
            canonical_key="observation.state",
        )
    with pytest.raises(KeyError, match="WSA_ACTION_STATS_KEYS"):
        build_transform_stats(
            feature_stats,
            explicit_keys=(),
            candidate_groups=action_groups,
            expected_dim=8,
            mode="mean_std",
            label="action",
            canonical_key="action",
        )


def test_franka_auto_selects_wr_franka_endpose_stats() -> None:
    feature_stats = {
        "observation.state.endpose": {
            "mean": [0.0] * 8,
            "std": [1.0] * 8,
        },
        "action.endpose": {"mean": [0.0] * 8, "std": [1.0] * 8},
    }
    state_groups, action_groups = stats_key_candidates("franka")
    state_keys, _ = build_transform_stats(
        feature_stats,
        explicit_keys=(),
        candidate_groups=state_groups,
        expected_dim=8,
        mode="mean_std",
        label="state",
        canonical_key="observation.state",
    )
    action_keys, _ = build_transform_stats(
        feature_stats,
        explicit_keys=(),
        candidate_groups=action_groups,
        expected_dim=8,
        mode="mean_std",
        label="action",
        canonical_key="action",
    )
    assert state_keys == ("observation.state.endpose",)
    assert action_keys == ("action.endpose",)


def test_agilex_accepts_official_qpos_stats_name() -> None:
    feature_stats = {
        "observations.qpos": {"mean": [0.0] * 14, "std": [1.0] * 14},
    }
    state_groups, _ = stats_key_candidates("agilex")
    keys, _ = build_transform_stats(
        feature_stats,
        explicit_keys=(),
        candidate_groups=state_groups,
        expected_dim=14,
        mode="mean_std",
        label="state",
        canonical_key="observation.state",
    )
    assert keys == ("observations.qpos",)


def test_index_only_sharded_checkpoint_is_rejected(tmp_path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({}), encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="model safetensors"):
        _resolve_checkpoint_dir(str(tmp_path))


def test_checkpoint_model_type_accepts_only_wsa_base() -> None:
    assert checkpoint_model_type(SimpleNamespace(type="WSA_Base")) == "wsa_base"
    assert checkpoint_model_type(SimpleNamespace(type="wsa_base")) == "wsa_base"
    with pytest.raises(TypeError, match="only original WSA-Base"):
        checkpoint_model_type(SimpleNamespace(type="other"))


class _RecordingProcessor:
    def __init__(self) -> None:
        self.samples: list[dict] = []

    def __call__(self, sample: dict) -> dict:
        self.samples.append(dict(sample))
        return sample


class _AffineActionTransform:
    """Small stand-in for native action unnormalization in the adapter unit test."""

    def __call__(self, sample: dict) -> dict:
        sample["action"] = sample["action"] * 2.0 + 10.0
        return sample


def _batch_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[WSABaseWorldArenaAdapter, _RecordingProcessor]:
    import torch

    _clean_worldarena_env(monkeypatch)
    monkeypatch.setenv("WORLDARENA_PLATFORM", "agilex")
    monkeypatch.setenv("WSA_DRY_RUN", "1")
    monkeypatch.setenv("WSA_DEVICE", "cpu")
    monkeypatch.setenv("WSA_DTYPE", "float32")
    adapter = WSABaseWorldArenaAdapter(RuntimeSettings.from_sources())
    adapter.device = torch.device("cpu")
    adapter.dtype = torch.float32
    processor = _RecordingProcessor()
    adapter.input_transforms = processor
    return adapter, processor


def _tensor_images(value: float):
    import torch

    return tuple(torch.full((3, 4, 5), value) for _ in range(3))


def test_base_batch_duplicates_current_images_and_uses_current_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    adapter, processor = _batch_adapter(monkeypatch)
    state0 = np.arange(14, dtype=np.float32)
    batch0 = adapter._prepare_model_batch(
        "complete task", state0, _tensor_images(0.0)
    )
    assert processor.samples[-1]["task"] == "complete task"
    assert batch0["observation.state"].shape == (1, 14)
    assert batch0["observation.images.image0"].shape == (1, 2, 3, 4, 5)
    torch.testing.assert_close(
        batch0["observation.images.image0"][0, 0],
        batch0["observation.images.image0"][0, 1],
    )

    state1 = state0 + 1
    batch1 = adapter._prepare_model_batch(
        "complete task", state1, _tensor_images(1.0)
    )
    torch.testing.assert_close(batch1["observation.state"][0], torch.from_numpy(state1))
    assert torch.all(batch1["observation.images.image0"][0, 0] == 1)
    assert torch.all(batch1["observation.images.image0"][0, 1] == 1)


def test_delta_restore_adds_state_only_on_delta_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _ = _batch_adapter(monkeypatch)
    adapter.output_transforms = _AffineActionTransform()
    adapter.action_mode = "delta"
    adapter.delta_mask = np.asarray(
        [True] * 6 + [False] + [True] * 6 + [False], dtype=bool
    )
    state = np.arange(14, dtype=np.float32)
    actions = adapter._restore_actions(np.ones((2, 14), dtype=np.float32), state)
    # Delta is applied after unnormalization: normalized 1 -> physical delta 12.
    expected = np.full((2, 14), 12.0, dtype=np.float32)
    expected[:, adapter.delta_mask] += state[adapter.delta_mask]
    np.testing.assert_array_equal(actions, expected)
    np.testing.assert_array_equal(actions[:, [6, 13]], 12.0)


def test_agilex_dry_run_returns_hold_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_worldarena_env(monkeypatch)
    monkeypatch.setenv("WORLDARENA_PLATFORM", "agilex")
    monkeypatch.setenv("WSA_DRY_RUN", "1")
    monkeypatch.setenv("WSA_EXECUTE_CHUNK_SIZE", "2")
    adapter = WSABaseWorldArenaAdapter(RuntimeSettings.from_sources())
    qpos = np.arange(14, dtype=np.float32)
    result = adapter.infer(
        {
            "prompt": "clean the table",
            "joint_qpos_left": qpos[:7],
            "joint_qpos_right": qpos[7:],
            "joint_qpos": qpos[7:],
            "images": _agilex_images(),
        }
    )
    np.testing.assert_array_equal(result["actions"], np.stack([qpos, qpos]))
    assert result["actions"].dtype == np.float32
    assert result["policy_metadata"]["action_format"] == "joint_absolute"


def test_franka_dry_run_returns_documented_xyzw_hold_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clean_worldarena_env(monkeypatch)
    monkeypatch.setenv("WORLDARENA_PLATFORM", "franka")
    monkeypatch.setenv("WSA_DRY_RUN", "1")
    monkeypatch.setenv("WSA_EXECUTE_CHUNK_SIZE", "2")
    settings = RuntimeSettings.from_sources()
    adapter = WSABaseWorldArenaAdapter(settings)
    pose_xyzw = np.asarray([0.4, 0.1, 0.3, 0.9, 0.1, 0.2, 0.3], dtype=np.float32)
    result = adapter.infer(
        {
            "prompt": "wipe the table",
            "left_end_pose": pose_xyzw,
            "joint_qpos": np.asarray([0, 1, 2, 3, 4, 5, 6, 0.25], dtype=np.float32),
            "images": _franka_images(),
        }
    )
    expected = np.concatenate([pose_xyzw, [0.25]]).astype(np.float32)
    expected[3:7] /= np.linalg.norm(expected[3:7])
    np.testing.assert_allclose(result["actions"], np.stack([expected, expected]))
    assert result["policy_metadata"]["action_format"] == "end_pose_base"
    assert result["policy_metadata"]["control_arm"] == "left"


def test_dry_run_rejects_prompt_and_camera_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clean_worldarena_env(monkeypatch)
    monkeypatch.setenv("WORLDARENA_PLATFORM", "agilex")
    monkeypatch.setenv("WSA_DRY_RUN", "1")
    adapter = WSABaseWorldArenaAdapter(RuntimeSettings.from_sources())
    qpos = np.zeros(14, dtype=np.float32)
    state = {
        "joint_qpos_left": qpos[:7],
        "joint_qpos_right": qpos[7:],
        "joint_qpos": qpos[7:],
    }

    with pytest.raises(KeyError, match="prompt"):
        adapter.infer({"task": "clean", **state, "images": _agilex_images()})

    wrong_images = {
        "cam_high": _rgb(),
        "cam_wrist_left": _rgb(),
        "cam_wrist_right": _rgb(),
    }
    with pytest.raises(KeyError, match="cam_left_wrist"):
        adapter.infer({"prompt": "clean", **state, "images": wrong_images})


def test_dry_run_rejects_non_uint8_or_chw_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clean_worldarena_env(monkeypatch)
    monkeypatch.setenv("WORLDARENA_PLATFORM", "franka")
    monkeypatch.setenv("WSA_DRY_RUN", "1")
    adapter = WSABaseWorldArenaAdapter(RuntimeSettings.from_sources())
    obs = {
        "prompt": "wipe",
        "left_end_pose": np.asarray([0, 0, 0, 1, 0, 0, 0], dtype=np.float32),
        "joint_qpos": np.zeros(8, dtype=np.float32),
        "images": _franka_images(),
    }
    obs["images"]["cam_high"] = np.zeros((3, 4, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="HWC RGB"):
        adapter.infer(obs)

    obs["images"] = _franka_images()
    obs["images"]["cam_high"] = _rgb().astype(np.float32)
    with pytest.raises(TypeError, match="dtype=uint8"):
        adapter.infer(obs)


def test_franka_does_not_fall_back_to_duplicate_cam_wrist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clean_worldarena_env(monkeypatch)
    monkeypatch.setenv("WORLDARENA_PLATFORM", "franka")
    monkeypatch.setenv("WSA_DRY_RUN", "1")
    adapter = WSABaseWorldArenaAdapter(RuntimeSettings.from_sources())
    obs = {
        "prompt": "wipe",
        "left_end_pose": np.asarray([0, 0, 0, 1, 0, 0, 0], dtype=np.float32),
        "joint_qpos": np.zeros(8, dtype=np.float32),
        "images": {"cam_high": _rgb(), "cam_wrist": _rgb()},
    }
    with pytest.raises(KeyError, match="cam_left_wrist"):
        adapter.infer(obs)
