import asyncio

import numpy as np
import pytest

from vlm_orchestrator.protocols.robodojo import RoboDojoFrame, RoboDojoObservationAdapter
from vlm_orchestrator.protocols.robodojo_ws import RoboDojoWsSession
from vlm_orchestrator.protocols.xpolicylab_ws import (
    XPolicyLabWsBackendConnection,
    _canonical_to_xpolicylab,
)


def _obs():
    return {
        "observation/exterior_image_1_left": np.zeros((2, 2, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.zeros((2, 2, 3), dtype=np.uint8),
        "observation/wrist_image_right": np.zeros((2, 2, 3), dtype=np.uint8),
        "observation/joint_position": np.zeros(12),
        "observation/gripper_position": np.zeros(2),
        "prompt": "move",
    }


def test_xpolicylab_batch_length_is_explicit_error():
    connection = XPolicyLabWsBackendConnection(None, evaluation_id="e", trial_id="t")

    async def wrong_response(*args, **kwargs):
        return RoboDojoFrame(
            message_type="call_result",
            message_id="x",
            evaluation_id="e",
            payload={"result": {"actions": []}},
        )

    connection._request = wrong_response
    with pytest.raises(ValueError, match="length mismatch"):
        asyncio.run(connection.infer_batch([_obs()]))


def test_xpolicylab_batch_uses_atomic_official_call():
    connection = XPolicyLabWsBackendConnection(None, evaluation_id="e", trial_id="t")
    requests = []

    async def batch_response(message_type, payload, **kwargs):
        requests.append((message_type, payload, kwargs))
        return RoboDojoFrame(
            message_type="call_result",
            message_id="x",
            evaluation_id="e",
            payload={"result": {"actions": [[{"joint_state": np.zeros(14)}]]}},
        )

    connection._request = batch_response
    result = asyncio.run(connection.infer_batch([_obs()]))

    assert len(requests) == 1
    assert requests[0][0] == "call"
    assert requests[0][1]["func_name"] == "infer_batch"
    assert len(requests[0][1]["obs"]) == 1
    assert "arm_joint_state" in result[0]["actions"][0]


def test_nested_robodojo_state_is_flattened_for_pi05():
    observation = _obs()
    observation["state"] = {
        "left_arm_joint_state": np.arange(6),
        "left_ee_joint_state": np.asarray([6]),
        "right_arm_joint_state": np.arange(7, 13),
        "right_ee_joint_state": np.asarray([13]),
        "left_ee_pose": np.zeros(7),
        "right_ee_pose": np.zeros(7),
    }

    encoded = _canonical_to_xpolicylab(observation)

    assert encoded["state"].dtype == np.float32
    np.testing.assert_array_equal(encoded["state"], np.arange(14, dtype=np.float32))


def test_robodojo_adapter_preserves_both_wrist_camera_calibrations():
    left_intrinsic = np.diag([100.0, 101.0, 1.0])
    right_intrinsic = np.diag([200.0, 201.0, 1.0])
    left_extrinsic = np.eye(4)
    right_extrinsic = np.eye(4)
    left_extrinsic[0, 3] = 0.1
    right_extrinsic[0, 3] = -0.1
    observation = {
        "vision": {
            "cam_left_wrist": {
                "color": np.zeros((8, 8, 3), dtype=np.uint8),
                "intrinsic_matrix": left_intrinsic,
                "extrinsic_matrix": left_extrinsic,
            },
            "cam_right_wrist": {
                "color": np.zeros((8, 8, 3), dtype=np.uint8),
                "intrinsics_matrix": right_intrinsic,
                "extrinsics_matrix": right_extrinsic,
            },
        },
    }

    canonical, _ = RoboDojoObservationAdapter().to_canonical(observation)

    np.testing.assert_array_equal(canonical["observation/wrist_camera_K_left"], left_intrinsic)
    np.testing.assert_array_equal(canonical["observation/wrist_camera_K_right"], right_intrinsic)
    np.testing.assert_array_equal(
        canonical["observation/wrist_camera_extrinsic_left"], left_extrinsic,
    )
    np.testing.assert_array_equal(
        canonical["observation/wrist_camera_extrinsic_right"], right_extrinsic,
    )


def test_robodojo_session_extracts_task_name_from_action_case_id():
    observation = {}
    frame = RoboDojoFrame(
        message_type="infer",
        message_id="message",
        evaluation_id="evaluation",
        action_case_id="press_by_number_case",
    )

    RoboDojoWsSession._add_task_name(observation, frame)

    assert observation["__task_name"] == "press_by_number"


def test_robodojo_batch_propagates_task_name_to_every_environment():
    session = RoboDojoWsSession(None)
    frame = RoboDojoFrame(
        message_type="batch_infer",
        message_id="message",
        evaluation_id="evaluation",
        action_case_id="push_T_random_case",
    )

    session._start_batch(frame, [{"env_idx": 0}, {"env_idx": 1}])

    assert [item["__task_name"] for item in session._batch_observations] == [
        "push_T_random",
        "push_T_random",
    ]


@pytest.mark.parametrize("key", ["color", "colors", "rgb", "image"])
def test_official_encoded_images_are_rgb_at_frontend_boundary(key):
    process_data = pytest.importorskip("XPolicyLab.utils.process_data")
    from vlm_orchestrator.protocols.robodojo_ws import _decode_images

    rgb = np.full((12, 12, 3), [230, 30, 10], dtype=np.uint8)
    observation = {"vision": {"cam_head": {key: process_data.encode_image_bit(rgb)}}}
    _decode_images(observation)
    decoded = observation["vision"]["cam_head"][key]
    assert decoded.shape == rgb.shape
    assert np.max(np.abs(decoded.astype(int) - rgb.astype(int))) < 5


def test_debug_replicated_depth_preserves_metric_values():
    from vlm_orchestrator.protocols.robodojo import _camera_depth

    depth = np.arange(6, dtype=np.float32).reshape(2, 3)
    repeated = np.repeat(depth[..., None], 3, axis=-1)
    result = _camera_depth({"cam_head": {"depth": repeated}}, ("cam_head",))
    np.testing.assert_array_equal(result, depth)
    repeated[0, 0, 1] += 1
    with pytest.raises(ValueError, match="2D map"):
        _camera_depth({"cam_head": {"depth": repeated}}, ("cam_head",))
