"""CPU contracts for the shared 58-D WUJI hybrid codec (`XPolicyLab.utils.wuji58`)."""

import json
import unittest
from pathlib import Path

import numpy as np

from XPolicyLab.utils import wuji58 as W
from XPolicyLab.utils.process_data import pack_robot_state, unpack_robot_state

REPO = Path(__file__).resolve().parents[1]


def _rand_quat(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def _rand_pose(rng):
    return np.concatenate([rng.uniform(-1, 1, 3), _rand_quat(rng)])


def _obs_state(rng):
    return {
        "left_ee_pose": _rand_pose(rng),
        "left_ee_joint_state": rng.uniform(-0.4, 1.5, 20),
        "right_ee_pose": _rand_pose(rng),
        "right_ee_joint_state": rng.uniform(-0.4, 1.5, 20),
    }


class LayoutTest(unittest.TestCase):
    def test_slices_match_wuji_simbench(self):
        self.assertEqual((W.LEFT_HAND.start, W.LEFT_HAND.stop), (0, 20))
        self.assertEqual((W.LEFT_WRIST.start, W.LEFT_WRIST.stop), (20, 29))
        self.assertEqual((W.RIGHT_HAND.start, W.RIGHT_HAND.stop), (29, 49))
        self.assertEqual((W.RIGHT_WRIST.start, W.RIGHT_WRIST.stop), (49, 58))
        self.assertEqual(W.WUJI58_DIM, 58)
        self.assertEqual(len(W.HAND_JOINT_NAMES), 20)
        self.assertEqual(W.HAND_JOINT_NAMES[0], "finger1_joint1")
        self.assertEqual(W.HAND_JOINT_NAMES[-1], "finger5_joint4")

    def test_te2_frame_matches_wrist_codec(self):
        # Same construction as WUJI SimBench wrist_codec.camera_world_matrix.
        R = W.camera_world_matrix()
        eye, look = np.array(W.TE2_EYE), np.array(W.TE2_LOOK)
        forward = (look - eye) / np.linalg.norm(look - eye)
        np.testing.assert_allclose(R[:, 2], -forward, atol=1e-12)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        self.assertGreater(np.linalg.det(R), 0)

    def test_robot_registry_entry(self):
        registry = json.loads((REPO / "utils/robot/_robot_info.json").read_text())
        self.assertEqual(registry["tianji_wuji"], W.TIANJI_WUJI_ROBOT_ACTION_DIM_INFO)
        self.assertTrue(W.is_wuji58(registry["tianji_wuji"], "ee"))
        self.assertFalse(W.is_wuji58(registry["tianji_wuji"], "joint"))
        self.assertFalse(W.is_wuji58(registry["arx_x5"], "ee"))


class RotationTest(unittest.TestCase):
    def test_quat_matrix_round_trip(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            q = _rand_quat(rng)
            np.testing.assert_allclose(W.matrix_to_quat_wxyz(W.quat_wxyz_to_matrix(q)), q, atol=1e-9)

    def test_rot6d_round_trip_and_convention(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            R = W.quat_wxyz_to_matrix(_rand_quat(rng))
            r6 = W.matrix_to_rot6d(R)
            np.testing.assert_allclose(r6, R[:2].reshape(6))  # first two ROWS
            np.testing.assert_allclose(W.rot6d_to_matrix(r6), R, atol=1e-9)

    def test_rot6d_rejects_degenerate(self):
        with self.assertRaises(ValueError):
            W.rot6d_to_matrix(np.zeros(6))
        with self.assertRaises(ValueError):
            W.rot6d_to_matrix(np.array([1, 0, 0, 2, 0, 0]))


class WristCodecTest(unittest.TestCase):
    def test_state_round_trip_every_frame(self):
        rng = np.random.default_rng(2)
        for frame in (W.WristFrame.te2(), W.WristFrame.world(), W.WristFrame.from_look_at((1, 2, 3), (0, 0, 0.5))):
            for _ in range(20):
                pose = _rand_pose(rng)
                np.testing.assert_allclose(W.decode_wrist_state(W.encode_wrist_state(pose, frame), frame), pose, atol=1e-9)

    def test_world_frame_is_identity(self):
        pose = np.array([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0])
        s = W.encode_wrist_state(pose, W.WristFrame.world())
        np.testing.assert_allclose(s, [0.1, 0.2, 0.3, 1, 0, 0, 0, 1, 0])

    def test_relative_is_spatial_delta_and_inverts(self):
        rng = np.random.default_rng(3)
        frame = W.WristFrame.te2()
        for _ in range(30):
            cur, tgt = _rand_pose(rng), _rand_pose(rng)
            rel = W.encode_wrist_relative(tgt, cur, frame)
            # dp = p_t_cam - p_c_cam ; dR = R_t_cam @ R_c_cam.T
            p_t, R_t = frame.world_to_frame(tgt[:3], tgt[3:])
            p_c, R_c = frame.world_to_frame(cur[:3], cur[3:])
            np.testing.assert_allclose(rel[:3], p_t - p_c, atol=1e-12)
            np.testing.assert_allclose(W.rot6d_to_matrix(rel[3:]), R_t @ R_c.T, atol=1e-9)
            np.testing.assert_allclose(W.decode_wrist_relative(rel, cur, frame), tgt, atol=1e-9)

    def test_zero_delta_holds_pose(self):
        rng = np.random.default_rng(4)
        cur = _rand_pose(rng)
        hold = np.concatenate([np.zeros(3), np.eye(3)[:2].reshape(6)])
        np.testing.assert_allclose(W.decode_wrist_relative(hold, cur, W.WristFrame.te2()), cur, atol=1e-9)

    def test_frame_from_name(self):
        self.assertIsInstance(W.WristFrame.from_name(None), W.WristFrame)
        np.testing.assert_allclose(W.WristFrame.from_name("WORLD").R_wc, np.eye(3))
        with self.assertRaises(ValueError):
            W.WristFrame.from_name("base")


class Pack58Test(unittest.TestCase):
    def test_pack_unpack(self):
        rng = np.random.default_rng(5)
        parts = (rng.normal(size=20), rng.normal(size=9), rng.normal(size=20), rng.normal(size=9))
        vec = W.pack58(*parts)
        self.assertEqual(vec.shape, (58,))
        for got, want in zip(W.unpack58(vec), parts):
            np.testing.assert_array_equal(got, want)

    def test_state58_from_obs_layout(self):
        rng = np.random.default_rng(6)
        state = _obs_state(rng)
        frame = W.WristFrame.te2()
        vec = W.state58_from_obs(state, frame)
        np.testing.assert_array_equal(vec[W.LEFT_HAND], state["left_ee_joint_state"])
        np.testing.assert_array_equal(vec[W.RIGHT_HAND], state["right_ee_joint_state"])
        np.testing.assert_allclose(vec[W.LEFT_WRIST], W.encode_wrist_state(state["left_ee_pose"], frame))
        np.testing.assert_allclose(vec[W.RIGHT_WRIST], W.encode_wrist_state(state["right_ee_pose"], frame))

    def test_action_round_trip_and_chunk(self):
        rng = np.random.default_rng(7)
        frame = W.WristFrame.te2()
        obs = _obs_state(rng)
        targets = [_obs_state(rng) for _ in range(5)]
        chunk = np.stack([W.action58_from_target(t, obs, frame) for t in targets])
        decoded = W.decode_action_chunk58(chunk, obs, frame, reference="chunk_start")
        self.assertEqual(len(decoded), 5)
        for got, want in zip(decoded, targets):
            self.assertEqual(set(got), set(W.STATE_KEYS))
            for key in W.STATE_KEYS:
                self.assertEqual(got[key].dtype, np.float32)
                np.testing.assert_allclose(got[key], want[key], atol=1e-5)

    def test_decoded_dicts_satisfy_the_standard_ee_contract(self):
        """Decoded dicts pack through the shared helper with the registry entry (54-D wire format)."""
        rng = np.random.default_rng(8)
        frame = W.WristFrame.te2()
        obs = _obs_state(rng)
        action = W.decode_action58(W.action58_from_target(_obs_state(rng), obs, frame), obs, frame)
        packed = pack_robot_state({"state": action}, "ee", W.TIANJI_WUJI_ROBOT_ACTION_DIM_INFO, source_type="obs")
        self.assertEqual(packed.shape, (54,))
        back = unpack_robot_state(packed, "ee", W.TIANJI_WUJI_ROBOT_ACTION_DIM_INFO, source_type="obs")
        for key in W.STATE_KEYS:
            np.testing.assert_allclose(back[key], action[key], atol=1e-6)
        # and the 54-D packed vector re-expands to the same 58-D state
        np.testing.assert_allclose(
            W.state58_from_obs(W.packed54_to_state_dict(packed), frame), W.state58_from_obs(action, frame), atol=1e-5
        )

    def test_missing_or_wrong_width_keys_fail_loudly(self):
        rng = np.random.default_rng(9)
        obs = _obs_state(rng)
        frame = W.WristFrame.te2()
        bad = dict(obs)
        bad["left_ee_joint_state"] = np.zeros(1)  # a gripper scalar is not a Wuji hand
        with self.assertRaises(ValueError):
            W.state58_from_obs(bad, frame)
        del bad["left_ee_joint_state"]
        with self.assertRaises(KeyError):
            W.state58_from_obs(bad, frame)
        with self.assertRaises(ValueError):
            W.decode_action_chunk58(np.zeros((3, 54)), obs, frame)


class TrajectoryTest(unittest.TestCase):
    def test_trajectory58_matches_per_frame_encoding(self):
        rng = np.random.default_rng(10)
        frame = W.WristFrame.te2()
        T = 6
        states = [_obs_state(rng) for _ in range(T)]
        targets = [_obs_state(rng) for _ in range(T)]
        plural = lambda rows: {k + "s": np.stack([r[k] for r in rows]) for k in W.STATE_KEYS}
        state58, action58 = W.trajectory58(plural(states), plural(targets), frame)
        self.assertEqual(state58.shape, (T, 58))
        self.assertEqual(action58.shape, (T, 58))
        for t in range(T):
            np.testing.assert_allclose(state58[t], W.state58_from_obs(states[t], frame))
            np.testing.assert_allclose(action58[t], W.action58_from_target(targets[t], states[t], frame))
            # decoding the stored action against the stored state recovers the recorded target
            dec = W.decode_action58(action58[t], states[t], frame)
            for key in W.STATE_KEYS:
                np.testing.assert_allclose(dec[key], targets[t][key], atol=1e-5)

    def test_trajectory58_length_mismatch(self):
        rng = np.random.default_rng(11)
        plural = lambda rows: {k + "s": np.stack([r[k] for r in rows]) for k in W.STATE_KEYS}
        with self.assertRaises(ValueError):
            W.trajectory58(plural([_obs_state(rng)] * 3), plural([_obs_state(rng)] * 2), W.WristFrame.world())


class TrajectoryFromDataTest(unittest.TestCase):
    """`trajectory58_from_data` is what the official LeRobot converters call."""

    def _episode(self, rng, T=5, with_action_poses=True):
        plural = lambda rows: {k + "s": np.stack([r[k] for r in rows]) for k in W.STATE_KEYS}
        states = [_obs_state(rng) for _ in range(T)]
        targets = [_obs_state(rng) for _ in range(T)]
        data = {"state": plural(states), "action": plural(targets)}
        if not with_action_poses:
            del data["action"]["left_ee_poses"]
            del data["action"]["right_ee_poses"]
        return data, states, targets

    def test_recorded_targets_are_used(self):
        rng = np.random.default_rng(12)
        frame = W.WristFrame.te2()
        data, states, targets = self._episode(rng)
        state58, action58 = W.trajectory58_from_data(data, frame)
        for t in range(5):
            dec = W.decode_action58(action58[t], states[t], frame)
            for key in W.STATE_KEYS:
                np.testing.assert_allclose(dec[key], targets[t][key], atol=1e-5)

    def test_missing_action_poses_fall_back_to_next_state(self):
        rng = np.random.default_rng(13)
        frame = W.WristFrame.te2()
        data, states, targets = self._episode(rng, with_action_poses=False)
        state58, action58 = W.trajectory58_from_data(data, frame)
        for t in range(5):
            dec = W.decode_action58(action58[t], states[t], frame)
            nxt = states[min(t + 1, 4)]
            np.testing.assert_allclose(dec["left_ee_pose"], nxt["left_ee_pose"], atol=1e-5)
            np.testing.assert_allclose(dec["right_ee_pose"], nxt["right_ee_pose"], atol=1e-5)
            # hands still come from the recorded action group
            np.testing.assert_allclose(dec["left_ee_joint_state"], targets[t]["left_ee_joint_state"], atol=1e-5)

    def test_no_action_group_at_all(self):
        rng = np.random.default_rng(14)
        data, states, _ = self._episode(rng)
        del data["action"]
        state58, action58 = W.trajectory58_from_data(data, W.WristFrame.world())
        self.assertEqual(action58.shape, (5, 58))
        # last row holds: zero delta against itself
        np.testing.assert_allclose(action58[-1][W.LEFT_WRIST][:3], 0, atol=1e-9)

    def test_motor_names(self):
        self.assertEqual(len(W.MOTOR_NAMES), 58)
        self.assertEqual(W.MOTOR_NAMES[0], "left_hand_finger1_joint1")
        self.assertEqual(W.MOTOR_NAMES[20], "left_wrist_x")
        self.assertEqual(W.MOTOR_NAMES[29], "right_hand_finger1_joint1")
        self.assertEqual(W.MOTOR_NAMES[57], "right_wrist_r12")
        self.assertEqual(W.REPR_MODE, "wrist_rel_te2")


class ChunkReferenceTest(unittest.TestCase):
    """Per-frame deltas (the data convention) must decode correctly chunk-wise and per-step."""

    def _rollout(self, rng, T=6):
        """A rollout where the wrist tracks the previous target exactly; hands arbitrary."""
        frame = W.WristFrame.te2()
        obs0 = _obs_state(rng)
        targets = [_obs_state(rng) for _ in range(T)]
        measured = [obs0]
        for t in range(1, T):
            m = dict(targets[t - 1])  # wrist at t == target commanded at t-1
            m["left_ee_joint_state"] = rng.uniform(-0.4, 1.5, 20)
            m["right_ee_joint_state"] = rng.uniform(-0.4, 1.5, 20)
            measured.append(m)
        per_frame = np.stack([W.action58_from_target(targets[t], measured[t], frame) for t in range(T)])
        return frame, obs0, measured, targets, per_frame

    def test_iterative_default_recovers_per_frame_targets(self):
        frame, obs0, measured, targets, per_frame = self._rollout(np.random.default_rng(20))
        decoded = W.decode_action_chunk58(per_frame, obs0, frame)  # default reference
        for got, want in zip(decoded, targets):
            for key in W.STATE_KEYS:
                np.testing.assert_allclose(got[key], want[key], atol=1e-5)

    def test_chunk_start_is_wrong_for_per_frame_data_but_right_for_its_own(self):
        frame, obs0, measured, targets, per_frame = self._rollout(np.random.default_rng(21))
        wrong = W.decode_action_chunk58(per_frame, obs0, frame, reference="chunk_start")
        self.assertGreater(np.abs(wrong[-1]["left_ee_pose"][:3] - targets[-1]["left_ee_pose"][:3]).max(), 1e-3)
        chunk_start = np.stack([W.action58_from_target(t, obs0, frame) for t in targets])
        right = W.decode_action_chunk58(chunk_start, obs0, frame, reference="chunk_start")
        for got, want in zip(right, targets):
            np.testing.assert_allclose(got["left_ee_pose"], want["left_ee_pose"], atol=1e-5)
        with self.assertRaises(ValueError):
            W.decode_action_chunk58(chunk_start, obs0, frame, reference="live")

    def test_per_step_decoder_against_live_wrist(self):
        frame, obs0, measured, targets, per_frame = self._rollout(np.random.default_rng(22))
        dec = W.Wuji58ChunkDecoder(frame)
        self.assertEqual(dec.pending, 0)
        with self.assertRaises(RuntimeError):
            dec.next(obs0)
        dec.start(per_frame)
        self.assertEqual(dec.pending, len(targets))
        for t, want in enumerate(targets):
            got = dec.next(measured[t])
            for key in W.STATE_KEYS:
                np.testing.assert_allclose(got[key], want[key], atol=1e-5)
        self.assertEqual(dec.pending, 0)
        dec.reset()
        self.assertEqual(dec.pending, 0)


if __name__ == "__main__":
    unittest.main()
