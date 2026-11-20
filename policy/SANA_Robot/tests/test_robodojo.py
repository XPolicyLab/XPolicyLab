"""The RoboDojo ARX-X5 conversions: an observation's state into a robot row and robot rows into actions."""

import numpy as np
import pytest

pytest.importorskip("sana_robot")

from sana_robot.data.format.robot80 import ROBOT_FIELDS  # noqa: E402
from sana_robot.data.utils.convert_robodojo import ARX_X5_JOINTS  # noqa: E402
from sana_robot.data.utils.kinematics import flange_poses  # noqa: E402

from XPolicyLab.policy.SANA_Robot import robodojo  # noqa: E402

LEFT = np.array([0.1, 1.2, 1.1, -0.3, 0.2, -0.4])
RIGHT = np.array([-0.2, 0.9, 1.3, 0.25, -0.1, 0.5])


def observed_state(left_opening=1.0, right_opening=0.25) -> dict:
    return {
        "left_arm_joint_state": LEFT,
        "left_ee_joint_state": np.array([left_opening]),
        "right_arm_joint_state": RIGHT,
        "right_ee_joint_state": np.array([right_opening]),
    }


def test_state_row_holds_joints_flange_pose_and_closedness():
    row = robodojo.state_row(observed_state())
    assert row.shape == (80,)
    np.testing.assert_allclose(row[ROBOT_FIELDS["left_joints"]][:6], LEFT, atol=1e-6)
    np.testing.assert_allclose(row[ROBOT_FIELDS["right_joints"]][:6], RIGHT, atol=1e-6)
    np.testing.assert_allclose(row[ROBOT_FIELDS["left_eef_position"]], flange_poses(ARX_X5_JOINTS, LEFT[None])[0, :3, 3], atol=1e-6)
    assert row[ROBOT_FIELDS["left_gripper"]][0] == pytest.approx(0.0)
    assert row[ROBOT_FIELDS["right_gripper"]][0] == pytest.approx(0.75)


def test_state_row_refuses_another_arm():
    state = observed_state()
    state["left_arm_joint_state"] = np.zeros(7)
    with pytest.raises(ValueError, match="6 joint angles"):
        robodojo.state_row(state)


def test_joint_actions_return_the_rows_joints_and_openings():
    row = robodojo.state_row(observed_state())
    (action,) = robodojo.actions_from_rows(row[None], "joint", robodojo.root_transforms())
    assert sorted(action) == [
        "left_arm_joint_state",
        "left_ee_joint_state",
        "right_arm_joint_state",
        "right_ee_joint_state",
    ]
    np.testing.assert_allclose(action["left_arm_joint_state"], LEFT, atol=1e-6)
    np.testing.assert_allclose(action["right_arm_joint_state"], RIGHT, atol=1e-6)
    np.testing.assert_allclose(action["left_ee_joint_state"], [1.0], atol=1e-6)
    np.testing.assert_allclose(action["right_ee_joint_state"], [0.25], atol=1e-6)
    assert all(value.dtype == np.float32 for value in action.values())


def test_gripper_opening_is_clipped_to_the_range_of_the_gripper():
    row = np.zeros(80)
    row[ROBOT_FIELDS["left_gripper"]] = 1.4
    row[ROBOT_FIELDS["right_gripper"]] = -0.3
    assert robodojo.gripper_opening(row, "left")[0] == pytest.approx(0.0)
    assert robodojo.gripper_opening(row, "right")[0] == pytest.approx(1.0)


def test_ee_actions_are_the_link6_pose_in_the_environment_frame():
    """The pose target of a row is the URDF flange (before the dataset's axis remap) seen through the arm's base pose."""
    world_from_base = robodojo.root_transforms()
    row = robodojo.state_row(observed_state())
    (action,) = robodojo.actions_from_rows(row[None], "ee", world_from_base)
    assert sorted(action) == ["left_ee_joint_state", "left_ee_pose", "right_ee_joint_state", "right_ee_pose"]
    for side, joints in (("left", LEFT), ("right", RIGHT)):
        flange = flange_poses(ARX_X5_JOINTS, joints[None])[0]
        flange[:3, :3] = flange[:3, :3] @ robodojo.FLANGE_AXES.T
        expected = world_from_base[side] @ flange
        pose = action[f"{side}_ee_pose"]
        assert pose.shape == (7,) and pose.dtype == np.float32
        np.testing.assert_allclose(robodojo.pose_to_transform(pose), expected, atol=1e-5)


def test_reported_pose_check_measures_a_wrong_root_pose():
    world_from_base = robodojo.root_transforms()
    state = observed_state()
    row = robodojo.state_row(state)
    for side in robodojo.SIDES:
        state[f"{side}_ee_pose"] = robodojo.transform_to_pose(world_from_base[side] @ robodojo.flange_transform(row, side))
    errors = robodojo.flange_pose_errors(state, row, world_from_base)
    assert all(metres < 1e-6 and degrees < 1e-3 for metres, degrees in errors.values())
    shifted = robodojo.root_transforms(
        {"left": (-0.3, -0.45, 0.775, 0.707, 0.0, 0.0, 0.707), "right": (0.3, -0.45, 0.765, 1.0, 0.0, 0.0, 0.0)}
    )
    errors = robodojo.flange_pose_errors(state, row, shifted)
    assert errors["left"][0] == pytest.approx(0.01, abs=1e-6) and errors["left"][1] < 1e-3
    assert errors["right"][1] == pytest.approx(90.0, abs=1e-3)
    del state["left_ee_pose"], state["right_ee_pose"]
    assert robodojo.flange_pose_errors(state, row, world_from_base) == {}


@pytest.mark.parametrize("seed", range(5))
def test_quaternion_and_rotation_are_inverse(seed):
    quaternion = np.random.default_rng(seed).normal(size=4)
    quaternion = quaternion / np.linalg.norm(quaternion) * np.sign(quaternion[0])
    rotation = robodojo.quaternion_to_rotation(quaternion)
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    np.testing.assert_allclose(robodojo.rotation_to_quaternion(rotation), quaternion, atol=1e-12)


def test_half_turns_keep_a_unit_quaternion():
    for axis in range(3):
        rotation = -np.eye(3)
        rotation[axis, axis] = 1.0
        quaternion = robodojo.rotation_to_quaternion(rotation)
        assert np.linalg.norm(quaternion) == pytest.approx(1.0)
        np.testing.assert_allclose(robodojo.quaternion_to_rotation(quaternion), rotation, atol=1e-12)


def test_camera_images_are_writable_copies_in_the_given_order():
    head = np.full((4, 6, 3), 7, dtype=np.uint8)
    head.setflags(write=False)
    vision = {"cam_head": {"color": head}, "cam_left_wrist": {"color": np.zeros((4, 6, 3), dtype=np.uint8)}}
    images = robodojo.camera_images(vision, ["cam_left_wrist", "cam_head"])
    assert images[1].flags.writeable and images[1] is not head and int(images[1][0, 0, 0]) == 7
    assert int(images[0].sum()) == 0
    vision["cam_head"]["color"] = np.zeros((4, 6, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="uint8 RGB"):
        robodojo.camera_images(vision, ["cam_head"])


def test_check_robot_accepts_the_arx_x5_only():
    robodojo.check_robot({"arm_dim": [6, 6], "ee_dim": [1, 1]})
    with pytest.raises(ValueError, match="ARX-X5"):
        robodojo.check_robot({"arm_dim": [7, 7], "ee_dim": [1, 1]})
    with pytest.raises(ValueError, match="ARX-X5"):
        robodojo.check_robot({"arm_dim": [6], "ee_dim": [1]})
