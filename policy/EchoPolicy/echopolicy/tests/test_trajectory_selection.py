import random

import numpy as np
import pytest

from vlm_orchestrator.trajectory_selection import (
    _orientation_error,
    current_ee_pose,
    evaluate_wrist_target,
    fk_trajectory,
    normalize_points,
    project_points,
    select_candidate,
    average_most_similar_chunks,
    average_consistent_chunks,
    group_trajectory_endpoints,
    ik_approach_chunk,
    target_gripper_rotation,
    target_pixel_to_world,
    wrist_centered_ee_target,
    x5_gripper_center,
    x5_gripper_pose,
)


def _chunk(value: float, steps: int = 3) -> np.ndarray:
    row = np.zeros(14, dtype=np.float64)
    row[:6] = value
    row[7:13] = value
    return np.repeat(row[None, :], steps, axis=0)


def test_fk_and_projection_are_finite_for_flipped_camera():
    trajectory = fk_trajectory(_chunk(0.0))
    assert trajectory["left"].shape == (3, 3)
    camera_to_world = np.diag([1.0, 1.0, -1.0, 1.0])
    pixels = project_points(trajectory["left"], np.diag([100.0, 100.0, 1.0]), camera_to_world)
    assert np.isfinite(pixels).all()


def test_current_ee_pose_selects_requested_arm():
    joints = np.arange(12, dtype=np.float64) * 0.01
    left_position, left_rotation = current_ee_pose(joints, "left")
    right_position, right_rotation = current_ee_pose(joints, "right")
    np.testing.assert_allclose(
        left_position,
        x5_gripper_pose(joints[:6], [-0.3, -0.45, 0.765, 0.707, 0, 0, 0.707])[0],
    )
    np.testing.assert_allclose(
        right_position,
        x5_gripper_pose(joints[6:], [0.3, -0.45, 0.765, 0.707, 0, 0, 0.707])[0],
    )
    np.testing.assert_allclose(left_rotation.T @ left_rotation, np.eye(3), atol=1e-8)
    np.testing.assert_allclose(right_rotation.T @ right_rotation, np.eye(3), atol=1e-8)


def test_normalize_points_filters_invalid_values():
    points = normalize_points([[0, 10], [255, 255], [-1, 2], [np.nan, 3]])
    np.testing.assert_array_equal(points, [[0, 10], [255, 255]])


def test_ordered_geometry_uses_best_plus_ten_percent_pool():
    chunks = [_chunk(0.0), _chunk(0.4), _chunk(1.2)]
    camera_to_world = np.diag([1.0, 1.0, -1.0, 1.0])
    intrinsic = np.diag([100.0, 100.0, 1.0])
    index, diagnostics = select_candidate(
        chunks,
        ordered=True,
        points=[[0, 0]],
        intrinsic=intrinsic,
        camera_to_world=camera_to_world,
        image_shape=(256, 256),
        rng=random.Random(1),
    )
    assert index in diagnostics["candidate_pool"]
    assert diagnostics["selection_reason"] == "ordered_geometry_random"
    assert len(diagnostics["geometry_scores"]) == 3


def test_target_distance_selects_near_or_far_trajectory_window():
    chunks = [_chunk(0.0, steps=40), _chunk(0.2, steps=40)]
    intrinsic = np.asarray([[100.0, 0.0, 112.0], [0.0, 100.0, 112.0], [0.0, 0.0, 1.0]])
    camera_to_world = np.diag([1.0, 1.0, -1.0, 1.0])
    common = {
        "ordered": True,
        "arm": "left",
        "intrinsic": intrinsic,
        "camera_to_world": camera_to_world,
        "image_shape": (256, 256),
        "current_joints": np.zeros(12),
        "depth_image": np.ones((256, 256), dtype=np.float32),
        "far_distance_m": 0.15,
        "near_trajectory_steps": 10,
        "far_trajectory_steps": 30,
        "rng": random.Random(1),
    }

    _, near = select_candidate(chunks, points=[[80, 135]], **common)
    _, far = select_candidate(chunks, points=[[200, 200]], **common)

    assert near["distance_class"] == "near"
    assert near["target_distance_m"] < 0.15
    assert near["trajectory_steps_requested"] == 10
    assert near["trajectory_steps_compared"] == [10, 10]
    assert far["distance_class"] == "far"
    assert far["target_distance_m"] > 0.15
    assert far["trajectory_steps_requested"] == 30
    assert far["trajectory_steps_compared"] == [30, 30]


def test_target_distance_uses_default_table_plane_without_depth():
    chunks = [_chunk(0.0, steps=40)]
    intrinsic = np.asarray([[100.0, 0.0, 112.0], [0.0, 100.0, 112.0], [0.0, 0.0, 1.0]])
    _, diagnostics = select_candidate(
        chunks,
        ordered=True,
        arm="left",
        points=[[127, 127]],
        intrinsic=intrinsic,
        camera_to_world=np.diag([1.0, 1.0, -1.0, 1.0]),
        image_shape=(256, 256),
        current_joints=np.zeros(12),
        depth_image=None,
        default_plane_height_m=0.765,
        far_distance_m=0.15,
        near_trajectory_steps=10,
        far_trajectory_steps=30,
    )
    assert diagnostics["distance_class"] in {"near", "far"}
    assert diagnostics["target_distance_m"] is not None


def test_unordered_and_missing_geometry_choose_from_all_candidates():
    chunks = [_chunk(0.0), _chunk(1.0), _chunk(2.0)]
    for kwargs in ({"ordered": False}, {"ordered": True, "points": None}):
        index, diagnostics = select_candidate(chunks, rng=random.Random(3), **kwargs)
        assert index in range(3)
        assert diagnostics["selection_reason"] == "random_all"


def test_average_most_similar_chunks_averages_closest_pair():
    chunks = [
        np.full((2, 14), 0.0),
        np.full((2, 14), 1.0),
        np.full((2, 14), 1.1),
    ]
    averaged, indices, distance = average_most_similar_chunks(chunks, [0, 1, 2])
    assert indices == [1, 2]
    assert distance == pytest.approx(0.1)
    np.testing.assert_allclose(averaged, 1.05)


def test_observe_scores_entire_chunk_and_ignores_gripper():
    first = _chunk(0.0)
    second = _chunk(0.2)
    first[:, 6] = 1_000  # gripper must not affect the score
    index, diagnostics = select_candidate([first, second], ordered=False, observe=True)
    assert index == 0
    assert diagnostics["selection_reason"] == "observe_zero_pose"


def test_consistent_subset_averages_three_and_excludes_outlier():
    chunks = [np.full((50, 14), value) for value in [0.0, 0.01, 0.02, 0.9]]
    averaged, indices, distance = average_consistent_chunks(chunks, [0, 1, 2, 3])
    assert indices == [0, 1, 2]
    np.testing.assert_allclose(averaged, 0.01)
    assert distance == pytest.approx((0.01 + 0.02 + 0.01) / 3)


def test_consistent_subset_does_not_chain_and_prefers_larger_population():
    chunks = [np.full((30, 14), value) for value in [0, 0.03, 0.06]]
    _, indices, _ = average_consistent_chunks(chunks, [0, 1, 2])
    assert len(indices) == 2
    chunks = [np.full((30, 14), value) for value in [0, 0.01, 0.02, 1, 1.001]]
    _, indices, _ = average_consistent_chunks(chunks, list(range(5)))
    assert indices == [0, 1, 2]  # largest coherent set, not the closest pair


def test_consistent_subset_checks_full_tail_and_gripper_difference():
    chunks = [np.zeros((50, 14)) for _ in range(3)]
    chunks[1][35:, :] = 1.0
    _, indices, _ = average_consistent_chunks(chunks, [0, 1, 2])
    assert indices == [0, 2]
    chunks[1][:] = 0
    chunks[1][:, 6] = 0.3
    _, indices, _ = average_consistent_chunks(chunks, [0, 1, 2])
    assert indices == [0, 2]


def test_consistent_subset_unequal_lengths_and_dictionary_actions():
    chunks = [[{"joints": np.array([value, value]), "gripper": value, "mode": "joint"}
               for _ in range(length)] for value, length in [(0, 30), (0.01, 20), (0.02, 25)]]
    averaged, indices, _ = average_consistent_chunks(chunks, [0, 1, 2])
    assert indices == [0, 1, 2]
    assert len(averaged) == 20
    np.testing.assert_allclose(averaged[0]["joints"], 0.01)
    assert averaged[0]["gripper"] == pytest.approx(0.01)
    assert averaged[0]["mode"] == "joint"
    assert chunks[0][0]["gripper"] == 0


def test_ee_grouping_separates_height_without_camera_projection():
    trajectories = [{"left": np.array([[0.1, 0.2, height]]), "right": np.array([[0, 0, 0]])}
                    for height in [0.0, 0.01, 0.1]]
    groups = group_trajectory_endpoints(trajectories, "left", radius=0.05)
    assert [group["candidate_indices"] for group in groups] == [[0, 1], [2]]


@pytest.mark.parametrize("depth_shape", [(5, 5), (1, 5, 5), (5, 5, 1)])
def test_target_pixel_backprojects_metric_depth_to_world(depth_shape):
    depth = np.full(depth_shape, 2.0)
    intrinsic = np.array([[2.0, 0, 2.0], [0, 2.0, 2.0], [0, 0, 1.0]])
    camera_to_world = np.eye(4)
    point = target_pixel_to_world([127.5, 127.5], depth, intrinsic, camera_to_world, (5, 5))
    np.testing.assert_allclose(point, [0, 0, -2])
    depth[:] = 0
    with pytest.raises(ValueError, match="valid local depth"):
        target_pixel_to_world([127.5, 127.5], depth, intrinsic, camera_to_world, (5, 5))


def test_target_pixel_rejects_multichannel_depth():
    with pytest.raises(ValueError, match=r"single-channel.*\(5, 5, 3\)"):
        target_pixel_to_world(
            [127.5, 127.5], np.ones((5, 5, 3)), np.eye(3), np.eye(4), (5, 5)
        )


def test_target_pixel_rejects_missing_depth():
    with pytest.raises(ValueError, match="no valid local depth"):
        target_pixel_to_world([127.5, 127.5], None, np.eye(3), np.eye(4), (5, 5))


def test_target_pixel_uses_default_table_plane_without_depth():
    intrinsic = np.asarray(
        [[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]]
    )
    camera_to_world = np.diag([1.0, 1.0, -1.0, 1.0])
    point = target_pixel_to_world(
        [127.5, 127.5], None, intrinsic, camera_to_world, (5, 5),
        default_plane_height_m=0.765,
    )
    assert point[2] == pytest.approx(0.765)
    assert np.isfinite(point).all()


def test_target_gripper_rotation_uses_approach_and_image_grasp_axis():
    depth = np.full((5, 5), 2.0)
    intrinsic = np.array([[2.0, 0, 2.0], [0, 2.0, 2.0], [0, 0, 1.0]])
    rotation = target_gripper_rotation(
        "top_down", [[0, 127.5], [255, 127.5]], depth, intrinsic, np.eye(4), (5, 5)
    )
    np.testing.assert_allclose(rotation[:, 0], [0, 0, -1], atol=1e-8)
    np.testing.assert_allclose(rotation[:, 1], [1, 0, 0], atol=1e-8)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


@pytest.mark.parametrize("arm", ["left", "right"])
@pytest.mark.parametrize("axis_sign", [1.0, -1.0])
def test_wrist_centered_target_preserves_mount_and_centers_target(arm, axis_sign):
    joints = np.zeros(12)
    ee_position, ee_rotation = current_ee_pose(joints, arm)
    current_ee = np.eye(4)
    current_ee[:3, :3] = ee_rotation
    current_ee[:3, 3] = ee_position
    ee_to_camera = np.eye(4)
    ee_to_camera[:3, :3] = np.column_stack((
        [0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0],
    ))
    ee_to_camera[:3, 3] = [0.02, 0.0, 0.0]
    current_camera = current_ee @ ee_to_camera
    original_camera = current_camera.copy()
    preferred = ee_rotation @ np.diag([1.0, axis_sign, axis_sign])
    target = current_camera[:3, 3] + 0.15 * ee_rotation[:, 0]

    target_position, target_rotation, diagnostics = wrist_centered_ee_target(
        joints, arm, target, current_camera, preferred,
    )

    np.testing.assert_array_equal(current_camera, original_camera)
    target_ee = np.eye(4)
    target_ee[:3, :3] = target_rotation
    target_ee[:3, 3] = target_position
    np.testing.assert_allclose(
        target_ee @ ee_to_camera,
        diagnostics["target_wrist_camera_to_world"],
        atol=1e-10,
    )
    np.testing.assert_allclose(target_position, ee_position, atol=1e-10)
    assert np.linalg.norm(_orientation_error(ee_rotation, target_rotation)) < 1e-10

    intrinsic = np.asarray([[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]])
    actions, ik_diagnostics = ik_approach_chunk(
        joints, np.asarray([0.25, 0.75]), arm, target_position,
        target_rotation=target_rotation, grasp_axis_symmetric=False,
        tolerance_m=0.002, orientation_tolerance_rad=0.01,
    )
    assert ik_diagnostics["target_reached"] is True
    result = evaluate_wrist_target(
        actions[-1], arm, target, diagnostics["ee_to_wrist_camera"], intrinsic, (48, 64),
    )
    assert result["wrist_target_reached"] is True
    assert result["wrist_target_visible"] is True
    assert result["wrist_center_error_px"] == pytest.approx(0.0, abs=1e-10)
    assert result["wrist_target_distance_m"] == pytest.approx(0.15)


@pytest.mark.parametrize("view_direction", [
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
])
def test_wrist_centered_target_uses_requested_approach_direction(view_direction):
    joints = np.zeros(12)
    ee_position, ee_rotation = current_ee_pose(joints, "left")
    current_camera = np.eye(4)
    current_camera[:3, :3] = ee_rotation
    current_camera[:3, 3] = ee_position
    direction = np.asarray(view_direction, dtype=np.float64)
    reference = np.asarray([0.0, 1.0, 0.0])
    if abs(float(np.dot(direction, reference))) > 0.9:
        reference = np.asarray([0.0, 0.0, 1.0])
    closing_axis = reference - np.dot(reference, direction) * direction
    closing_axis /= np.linalg.norm(closing_axis)
    preferred = np.column_stack((
        direction, closing_axis, np.cross(direction, closing_axis),
    ))
    target = np.asarray([0.1, -0.2, 0.8])

    _, _, diagnostics = wrist_centered_ee_target(
        joints, "left", target, current_camera, preferred,
    )

    target_camera = np.asarray(diagnostics["target_wrist_camera_to_world"])
    np.testing.assert_allclose(target_camera[:3, 3], target - 0.15 * direction)
    np.testing.assert_allclose(-target_camera[:3, 2], direction, atol=1e-10)


def test_wrist_target_validation_rejects_target_behind_camera():
    joints = np.zeros(12)
    ee_position, ee_rotation = current_ee_pose(joints, "left")
    ee_to_camera = np.eye(4)
    camera_to_world = np.eye(4)
    camera_to_world[:3, :3] = ee_rotation
    camera_to_world[:3, 3] = ee_position
    ee_to_camera = np.linalg.inv(camera_to_world) @ camera_to_world
    target = camera_to_world[:3, 3] + camera_to_world[:3, 2] * 0.15

    result = evaluate_wrist_target(
        joints, "left", target, ee_to_camera,
        np.asarray([[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]]),
        (48, 64),
    )

    assert result["wrist_target_depth_m"] < 0
    assert result["wrist_target_visible"] is False
    assert result["wrist_target_reached"] is False


def test_orientation_error_does_not_collapse_at_180_degrees():
    half_turn = np.diag([1.0, -1.0, -1.0])
    assert np.linalg.norm(_orientation_error(np.eye(3), half_turn)) == pytest.approx(np.pi)


@pytest.mark.parametrize("arm", ["left", "right"])
def test_ik_approach_moves_only_requested_arm_and_preserves_grippers(arm):
    joints = np.zeros(12)
    grippers = np.array([0.25, 0.75])
    base = np.array([-0.3, -0.45, 0.765, 0.707, 0, 0, 0.707]) if arm == "left" else np.array([0.3, -0.45, 0.765, 0.707, 0, 0, 0.707])
    desired = x5_gripper_center(np.zeros(6), base) + np.array([0.02, 0.0, 0.0])
    actions, diagnostics = ik_approach_chunk(joints, grippers, arm, desired)
    assert actions.shape == (10, 14)
    np.testing.assert_allclose(actions[:, 6], 0.25)
    np.testing.assert_allclose(actions[:, 13], 0.75)
    inactive = actions[:, 7:13] if arm == "left" else actions[:, :6]
    np.testing.assert_allclose(inactive, 0)
    final_joints = actions[-1, :6] if arm == "left" else actions[-1, 7:13]
    np.testing.assert_allclose(x5_gripper_center(final_joints, base), desired, atol=0.01)
    np.testing.assert_allclose(diagnostics["target_world"], desired)
    assert diagnostics["approach_height_m"] == 0.0
    assert diagnostics["position_error_m"] <= 0.01


def test_ik_approach_solves_distant_target_but_bounds_one_chunk():
    joints = np.zeros(12)
    grippers = np.array([0.25, 0.75])
    base = np.array([0.3, -0.45, 0.765, 0.707, 0, 0, 0.707])
    target = x5_gripper_center([0.625, 1.986, 1.378, -1.374, -0.999, 1.868], base)
    start_error = np.linalg.norm(target - x5_gripper_center(np.zeros(6), base))

    actions, diagnostics = ik_approach_chunk(joints, grippers, "right", target)

    assert np.max(np.abs(actions[-1, 7:13])) <= 0.8 + 1e-12
    assert diagnostics["solver_position_error_m"] <= 0.01
    assert diagnostics["position_error_m"] < start_error
    assert diagnostics["target_reached"] is False


def test_ik_approach_reaches_distant_target_in_ten_steps_without_chunk_limit():
    joints = np.zeros(12)
    grippers = np.array([0.25, 0.75])
    base = np.array([0.3, -0.45, 0.765, 0.707, 0, 0, 0.707])
    target = x5_gripper_center([0.625, 1.986, 1.378, -1.374, -0.999, 1.868], base)

    actions, diagnostics = ik_approach_chunk(
        joints, grippers, "right", target, steps=10, max_joint_delta=np.pi
    )

    assert actions.shape == (10, 14)
    np.testing.assert_allclose(
        x5_gripper_center(actions[-1, 7:13], base), target, atol=0.01
    )
    assert diagnostics["target_reached"] is True
    assert diagnostics["converged_solution_count"] >= 1
    assert diagnostics["selected_max_joint_delta"] == pytest.approx(
        np.max(np.abs(diagnostics["joint_delta"]))
    )


def test_ik_approach_tracks_target_orientation():
    joints = np.zeros(12)
    grippers = np.array([0.25, 0.75])
    base = np.array([0.3, -0.45, 0.765, 0.707, 0, 0, 0.707])
    desired_joints = np.array([0.2, -0.3, 0.4, -0.2, 0.1, 0.2])
    target, target_rotation = x5_gripper_pose(desired_joints, base)

    actions, diagnostics = ik_approach_chunk(
        joints, grippers, "right", target, target_rotation=target_rotation
    )

    final_joints = actions[-1, 7:13]
    final_position, final_rotation = x5_gripper_pose(final_joints, base)
    np.testing.assert_allclose(final_position, target, atol=0.01)
    assert diagnostics["solver_orientation_error_rad"] <= 0.2
    assert diagnostics["orientation_error_rad"] <= 0.2
    assert diagnostics["target_reached"] is True


def test_ik_approach_rejects_wrist_flip_and_keeps_cartesian_path_local():
    base = np.array([0.3, -0.45, 0.765, 0.707, 0, 0, 0.707])
    start = np.array([
        0.6502586758475504, 2.458593327781948, 2.432124249966243,
        -1.5452580614703868, 0.004295858946542275, 0.6503201154647966,
    ])
    joints = np.concatenate((np.zeros(6), start))
    start_position, start_rotation = x5_gripper_pose(start, base)

    actions, diagnostics = ik_approach_chunk(
        joints, np.ones(2), "right", start_position + [0.0, 0.0, 0.1],
        target_rotation=start_rotation, steps=20, max_joint_delta=np.pi,
    )

    arm_path = np.vstack((start, actions[:, 7:13]))
    ee_path = np.asarray([x5_gripper_pose(step, base)[0] for step in arm_path[1:]])
    assert np.max(np.abs(np.diff(arm_path, axis=0))) <= 0.35
    assert np.max(np.linalg.norm(ee_path[:, :2] - start_position[:2], axis=1)) < 0.01
    assert np.all(np.diff(ee_path[:, 2]) >= -1e-9)
    assert diagnostics["selected_max_joint_delta"] < 0.5
    assert diagnostics["waypoint_failure"]["waypoint"] == 17


def test_ik_approach_chooses_nearest_equivalent_grasp_axis_direction():
    joints = np.zeros(12)
    grippers = np.array([0.25, 0.75])
    base = np.array([0.3, -0.45, 0.765, 0.707, 0, 0, 0.707])
    target, current_rotation = x5_gripper_pose(np.zeros(6), base)
    reversed_axis_rotation = current_rotation @ np.diag([1.0, -1.0, -1.0])

    actions, diagnostics = ik_approach_chunk(
        joints, grippers, "right", target,
        target_rotation=reversed_axis_rotation, grasp_axis_symmetric=True,
    )

    np.testing.assert_allclose(actions[:, 7:13], 0, atol=1e-10)
    assert diagnostics["grasp_axis_flipped"] is True
    np.testing.assert_allclose(diagnostics["target_rotation"], current_rotation)
    np.testing.assert_allclose(diagnostics["input_target_rotation"], reversed_axis_rotation)
    assert diagnostics["target_reached"] is True
