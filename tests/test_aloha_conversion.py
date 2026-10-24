"""Exercise the ALOHA converter with actual HDF5 trajectories."""

import h5py
import numpy as np
import pytest

from XPolicyLab.scripts.transform_aloha_hdf5_format import _extract_action, convert_one


KEYS = (
    "left_arm_joint_states", "left_ee_joint_states",
    "right_arm_joint_states", "right_ee_joint_states",
)


def write_episode(path, action_layout, widths=(6, 1, 6, 1)):
    horizon = 3
    parts = [
        np.arange(horizon * width, dtype=np.float32).reshape(horizon, width) + i * 100
        for i, width in enumerate(widths)
    ]
    targets = [part + 0.5 for part in parts]
    images = np.full((horizon, 8, 10, 3), [20, 90, 180], dtype=np.uint8)
    with h5py.File(path, "w") as episode:
        for key, part in zip(KEYS, parts):
            episode.create_dataset(f"state/{key}", data=part)
        if action_layout == "group":
            # HDF5 key iteration is alphabetical; output must follow arm/EE order.
            for key, target in reversed(list(zip(KEYS, targets))):
                episode.create_dataset(f"action/{key}", data=target)
        elif action_layout == "array":
            episode.create_dataset("action", data=np.concatenate(targets, axis=1))
        elif action_layout == "legacy":
            for path, target in zip(
                ("left_arm/joint_action", "left_arm/gripper_action",
                 "right_arm/joint_action", "right_arm/gripper_action"), targets
            ):
                episode.create_dataset(path, data=target)
        episode.create_dataset("vision/cam_head/colors", data=images)
        episode.create_dataset("instructions", data='["pick up the cup"]')
    return np.concatenate(parts, axis=1), np.concatenate(targets, axis=1), images


@pytest.mark.parametrize("widths", [(6, 1, 6, 1), (6, 2, 7, 1)])
@pytest.mark.parametrize("action_layout", ["group", "array", "legacy", "absent"])
def test_convert_hdf5_episode(tmp_path, action_layout, widths):
    source, destination = tmp_path / "input.hdf5", tmp_path / "output.hdf5"
    qpos, targets, images = write_episode(source, action_layout, widths)
    convert_one(source, destination, "xspark", "v1.0")
    expected = targets if action_layout != "absent" else np.concatenate(
        [qpos[1:], np.zeros_like(qpos[:1])], axis=0
    )
    with h5py.File(destination, "r") as converted:
        np.testing.assert_array_equal(converted["action"][()], expected)
        assert converted["action"].dtype == np.float32
        np.testing.assert_array_equal(converted["observations/qpos"][()], qpos)
        np.testing.assert_array_equal(converted["observations/images/cam_high"][()], images)
        assert converted["instructions"].asstr()[()].tolist() == ["pick up the cup"]


def test_group_grippers_can_be_one_dimensional():
    action = {key: np.full((3, 2), i, dtype=np.float64) for i, key in enumerate(KEYS)}
    for key in (KEYS[1], KEYS[3]):
        action[key] = action[key][:, 0]
    actual = _extract_action({"action": action}, np.zeros((3, 6)))
    np.testing.assert_array_equal(actual, np.tile([0, 0, 1, 2, 2, 3], (3, 1)))
    assert actual.dtype == np.float32


def test_group_action_horizon_must_match_state():
    with pytest.raises(ValueError, match="action horizon mismatch"):
        _extract_action({"action": {KEYS[0]: np.zeros((2, 6))}}, np.zeros((3, 6)))


def test_group_fields_must_share_horizon():
    with pytest.raises(ValueError, match="horizon mismatch"):
        _extract_action({"action": {
            KEYS[0]: np.zeros((3, 6)), KEYS[1]: np.zeros((2, 1))
        }}, np.zeros((3, 7)))


def test_pose_only_group_is_not_silently_replaced_with_shifted_states():
    with pytest.raises(ValueError, match="joint action"):
        _extract_action({"action": {"left_ee_poses": np.zeros((3, 7))}}, np.zeros((3, 7)))


def test_flat_action_horizon_validation_remains():
    with pytest.raises(ValueError, match="action horizon mismatch"):
        _extract_action({"action": np.zeros((2, 7))}, np.zeros((3, 7)))
