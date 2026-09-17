from __future__ import annotations

import numpy as np
import pytest

from worldscape_policy.action_space import (
    compose_rotation6d,
    convert_eef_actions_to_relative,
    matrix_to_rotation6d,
    parse_action_mode,
    relative_rotation6d,
    rotation6d_to_matrix,
)

IDENTITY = np.array([1, 0, 0, 1, 0, 0], dtype=np.float32)
ROTATE_Z_90 = np.array([0, -1, 1, 0, 0, 0], dtype=np.float32)
ROTATE_Z_180 = np.array([-1, 0, 0, -1, 0, 0], dtype=np.float32)


def _eef(rows: int) -> np.ndarray:
    value = np.zeros((rows, 20), dtype=np.float32)
    value[:, 3:9] = IDENTITY
    value[:, 13:19] = IDENTITY
    value[:, 9] = 0.25
    value[:, 19] = 0.75
    return value


def test_action_mode_accepts_only_eef() -> None:
    assert parse_action_mode("eef") == "eef"
    with pytest.raises(ValueError, match="supports only 'eef'"):
        parse_action_mode("eef6d")
    with pytest.raises(ValueError, match="supports only 'eef'"):
        parse_action_mode("joint")


def test_relative_rotation_is_expressed_in_anchor_frame() -> None:
    anchor = np.array(
        [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
        dtype=np.float32,
    )
    local_delta = np.array(
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        dtype=np.float32,
    )
    target = anchor @ local_delta
    anchor6d = matrix_to_rotation6d(anchor)
    target6d = matrix_to_rotation6d(target)

    relative6d = relative_rotation6d(target6d, anchor6d)

    np.testing.assert_allclose(
        rotation6d_to_matrix(relative6d), local_delta, atol=1e-6
    )
    np.testing.assert_allclose(
        rotation6d_to_matrix(compose_rotation6d(anchor6d, relative6d)),
        target,
        atol=1e-6,
    )


def test_relative_eef_uses_each_horizon_anchor_and_keeps_grippers_absolute() -> None:
    state = _eef(48)
    state[:24, :3] = 2.0
    state[:24, 10:13] = 3.0
    state[24:, :3] = 5.0
    state[24:, 10:13] = 7.0
    state[24:, 3:9] = ROTATE_Z_90
    state[24:, 13:19] = ROTATE_Z_90
    action = _eef(48)
    action[:, :3] = 11.0
    action[:, 10:13] = 13.0
    action[:24, 3:9] = ROTATE_Z_90
    action[:24, 13:19] = ROTATE_Z_90
    action[24:, 3:9] = ROTATE_Z_180
    action[24:, 13:19] = ROTATE_Z_180

    relative = convert_eef_actions_to_relative(action, state)

    np.testing.assert_allclose(relative[:24, :3], 9.0)
    np.testing.assert_allclose(relative[:24, 10:13], 10.0)
    np.testing.assert_allclose(relative[24:, :3], 6.0)
    np.testing.assert_allclose(relative[24:, 10:13], 6.0)
    expected_relative_rotation = np.broadcast_to(ROTATE_Z_90, (48, 6))
    np.testing.assert_allclose(
        relative[:, 3:9], expected_relative_rotation, atol=1e-6
    )
    np.testing.assert_allclose(
        relative[:, 13:19], expected_relative_rotation, atol=1e-6
    )
    np.testing.assert_allclose(
        compose_rotation6d(state[24:, 3:9], relative[24:, 3:9]),
        action[24:, 3:9],
        atol=1e-6,
    )
    np.testing.assert_array_equal(relative[:, 9], action[:, 9])
    np.testing.assert_array_equal(relative[:, 19], action[:, 19])


def test_relative_eef_supports_temporal_packing_anchor_states() -> None:
    state = _eef(2)
    state[0, :3] = 1.0
    state[1, :3] = 4.0
    action = _eef(48)
    action[:, :3] = 10.0

    relative = convert_eef_actions_to_relative(action, state)

    np.testing.assert_allclose(relative[:24, :3], 9.0)
    np.testing.assert_allclose(relative[24:, :3], 6.0)
