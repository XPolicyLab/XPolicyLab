"""58-D hybrid wrist/hand codec for the WUJI dual-arm dexterous-hand platform.

This module is the single XPolicyLab implementation of the ``wrist58`` layout
defined by WUJI SimBench (``RoboVerse/roboverse_learn/il/utils/wrist_codec.py``,
``vla/gr00t/wuji_layout.py``), so an adapter never spells the slices, the
rotation conventions, or the camera frame out on its own.

Robot
-----
``tianji_wuji``: two 7-DOF arms, each carrying a 20-DOF Wuji hand. In the
XPolicyLab robot registry it is ``{"arm_dim": [7, 7], "ee_dim": [20, 20]}``;
with ``action_type="ee"`` the *standard* packed vector is therefore 54-D::

    [ 0: 7]  left_ee_pose          x y z qw qx qy qz   (world frame)
    [ 7:27]  left_ee_joint_state   20 hand joints
    [27:34]  right_ee_pose
    [34:54]  right_ee_joint_state

The 58-D hybrid vector a policy trains on and predicts is a *re-encoding* of the
same blocks (rotation as rot6d, hand first) that matches WUJI SimBench byte for
byte::

    [ 0:20]  left hand qpos          absolute joints, finger-major
    [20:29]  left wrist xyz+rot6d    state: absolute in the camera frame
                                     action: camera-frame delta vs. the live wrist
    [29:49]  right hand qpos
    [49:58]  right wrist xyz+rot6d

Conventions (identical to WUJI SimBench, do not "fix" one side only):

* quaternions are ``wxyz`` everywhere (the XPolicyLab pose convention too);
* rot6d is the first two ROWS of the rotation matrix, flattened; decoding
  Gram-Schmidts the rows and completes the third with a cross product;
* the wrist frame defaults to the fixed TE2 head camera (``TE2_EYE`` /
  ``TE2_LOOK``, OpenGL axes: the camera looks along -Z); ``WristFrame.world()``
  is the identity frame for data that was recorded in world coordinates;
* the relative wrist action is a *spatial* (camera-frame) delta::

      dp = p_target_cam - p_current_cam
      dR = R_target_cam @ R_current_cam.T

  and :func:`decode_action58` inverts it exactly against the live wrist.

Hand joints are absolute targets on both sides and pass through untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

# ── layout ───────────────────────────────────────────────────────────────────

WUJI58_DIM = 58
HAND_DIM = 20
WRIST_DIM = 9
POSE_DIM = 7  # x y z qw qx qy qz

LEFT_HAND = slice(0, 20)
LEFT_WRIST = slice(20, 29)
RIGHT_HAND = slice(29, 49)
RIGHT_WRIST = slice(49, 58)

#: Registry entry for the robot (``utils/robot/_robot_info.json`` and the parent
#: ``env_cfg/robot/_robot_info.json`` carry the same numbers).
TIANJI_WUJI_ROBOT_ACTION_DIM_INFO = {"arm_dim": [7, 7], "ee_dim": [20, 20]}

#: Canonical finger-major Wuji hand joint order (``roboverse_pack.robots.tianji_wuji_cfg``).
HAND_JOINT_NAMES = tuple(
    f"finger{finger}_joint{joint}" for finger in range(1, 6) for joint in range(1, 5)
)

#: ``wrist_frame`` values an adapter may accept from ``deploy.yml``.
WRIST_FRAMES = ("te2", "world")

#: Per-dimension names of the 58-D vector, used as LeRobot ``observation.state`` /
#: ``action`` motor names by the official converters (``--representation wuji58``)
#: and asserted by adapters that read such an export.
_WRIST_COMPONENTS = ("x", "y", "z", "r00", "r01", "r02", "r10", "r11", "r12")
MOTOR_NAMES = tuple(
    [f"left_hand_{name}" for name in HAND_JOINT_NAMES]
    + [f"left_wrist_{name}" for name in _WRIST_COMPONENTS]
    + [f"right_hand_{name}" for name in HAND_JOINT_NAMES]
    + [f"right_wrist_{name}" for name in _WRIST_COMPONENTS]
)
assert len(MOTOR_NAMES) == WUJI58_DIM

#: Value stamped into a converted dataset so a loader can refuse a mismatched layout
#: instead of guessing from the 58-dim width alone (WUJI SimBench ``repr_mode``).
REPR_MODE = "wrist_rel_te2"

# TE2 head-camera preset (WUJI SimBench ``wrist_codec.TE2_*``).
TE2_EYE = (0.30, 0.0, 1.32)
TE2_LOOK = (0.62, 0.0, 0.75)

_WORLD_UP = np.array([0.0, 0.0, 1.0])

STATE_KEYS = ("left_ee_pose", "left_ee_joint_state", "right_ee_pose", "right_ee_joint_state")


# ── rotation helpers (numpy, float64) ────────────────────────────────────────


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix from a wxyz quaternion (normalizes defensively)."""
    w, x, y, z = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        raise ValueError(f"degenerate quaternion {q!r}")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """wxyz quaternion from a rotation matrix (Shepperd's method)."""
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    q = q / np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def matrix_to_rot6d(R: np.ndarray) -> np.ndarray:
    """First two rows of ``R``, flattened (sova / pytorch3d convention)."""
    return np.asarray(R, dtype=np.float64)[:2, :].reshape(6).copy()


def rot6d_to_matrix(r6: np.ndarray) -> np.ndarray:
    """Rotation matrix from 6-D (Gram-Schmidt rows, third row = cross)."""
    r6 = np.asarray(r6, dtype=np.float64).reshape(2, 3)
    n1 = np.linalg.norm(r6[0])
    if n1 < 1e-8:
        raise ValueError(f"degenerate rot6d {r6!r} (first row is zero)")
    r1 = r6[0] / n1
    r2 = r6[1] - np.dot(r1, r6[1]) * r1
    n2 = np.linalg.norm(r2)
    if n2 < 1e-8:
        raise ValueError(f"degenerate rot6d {r6!r} (rows colinear)")
    r2 = r2 / n2
    return np.stack([r1, r2, np.cross(r1, r2)])


# ── wrist frame ──────────────────────────────────────────────────────────────


def camera_world_matrix(eye=TE2_EYE, look=TE2_LOOK) -> np.ndarray:
    """``R_world_from_cam`` for a MuJoCo-style pos/look-at camera (OpenGL axes)."""
    eye = np.asarray(eye, dtype=np.float64)
    forward = np.asarray(look, dtype=np.float64) - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, _WORLD_UP)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    return np.stack([right, up, -forward], axis=1)


@dataclass(frozen=True)
class WristFrame:
    """The fixed frame wrist poses are expressed in: ``p_cam = R_cw @ (p_w - origin)``."""

    R_wc: np.ndarray  # world <- cam, 3x3
    origin: np.ndarray  # camera position in world, 3

    @classmethod
    def te2(cls) -> "WristFrame":
        """The WUJI SimBench TE2 head camera (the default, matches the trained layout)."""
        return cls(camera_world_matrix(), np.asarray(TE2_EYE, dtype=np.float64))

    @classmethod
    def world(cls) -> "WristFrame":
        """Identity: wrist poses stay in world coordinates."""
        return cls(np.eye(3), np.zeros(3))

    @classmethod
    def from_look_at(cls, eye, look) -> "WristFrame":
        return cls(camera_world_matrix(eye, look), np.asarray(eye, dtype=np.float64))

    @classmethod
    def from_name(cls, name: Any) -> "WristFrame":
        """Resolve a ``deploy.yml`` ``wrist_frame`` value (``te2`` by default)."""
        key = str(name or "te2").strip().lower()
        if key == "te2":
            return cls.te2()
        if key == "world":
            return cls.world()
        raise ValueError(f"unknown wrist_frame {name!r}; choose from {WRIST_FRAMES}")

    def world_to_frame(self, pos_w, quat_wxyz_w) -> tuple[np.ndarray, np.ndarray]:
        R_cw = self.R_wc.T
        p = R_cw @ (np.asarray(pos_w, dtype=np.float64).reshape(3) - self.origin)
        R = R_cw @ quat_wxyz_to_matrix(quat_wxyz_w)
        return p, R

    def frame_to_world(self, p_cam, R_cam) -> tuple[np.ndarray, np.ndarray]:
        pos_w = self.origin + self.R_wc @ np.asarray(p_cam, dtype=np.float64).reshape(3)
        return pos_w, matrix_to_quat_wxyz(self.R_wc @ np.asarray(R_cam, dtype=np.float64))


# ── the 9-D wrist encode / decode pair ───────────────────────────────────────


def _split_pose(pose7) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(pose7, dtype=np.float64).reshape(-1)
    if pose.shape != (POSE_DIM,):
        raise ValueError(f"expected a 7-D [x y z qw qx qy qz] pose, got shape {pose.shape}")
    return pose[:3], pose[3:]


def encode_wrist_state(pose7, frame: WristFrame) -> np.ndarray:
    """Absolute world pose (7-D wxyz) → 9-D ``xyz+rot6d`` in ``frame``."""
    p, R = frame.world_to_frame(*_split_pose(pose7))
    return np.concatenate([p, matrix_to_rot6d(R)])


def decode_wrist_state(state9, frame: WristFrame) -> np.ndarray:
    """Invert :func:`encode_wrist_state` → absolute world pose (7-D wxyz)."""
    s = np.asarray(state9, dtype=np.float64).reshape(WRIST_DIM)
    pos, quat = frame.frame_to_world(s[:3], rot6d_to_matrix(s[3:]))
    return np.concatenate([pos, quat])


def encode_wrist_relative(target_pose7, current_pose7, frame: WristFrame) -> np.ndarray:
    """Relative 9-D wrist action: frame-space delta from the current to the target pose."""
    p_t, R_t = frame.world_to_frame(*_split_pose(target_pose7))
    p_c, R_c = frame.world_to_frame(*_split_pose(current_pose7))
    return np.concatenate([p_t - p_c, matrix_to_rot6d(R_t @ R_c.T)])


def decode_wrist_relative(rel9, current_pose7, frame: WristFrame) -> np.ndarray:
    """Invert :func:`encode_wrist_relative` → absolute world target pose (7-D wxyz)."""
    rel = np.asarray(rel9, dtype=np.float64).reshape(WRIST_DIM)
    p_c, R_c = frame.world_to_frame(*_split_pose(current_pose7))
    pos, quat = frame.frame_to_world(p_c + rel[:3], rot6d_to_matrix(rel[3:]) @ R_c)
    return np.concatenate([pos, quat])


# ── 58-D packing ─────────────────────────────────────────────────────────────


def pack58(left_hand20, left_wrist9, right_hand20, right_wrist9) -> np.ndarray:
    """Assemble one 58-D vector in the WUJI slice order."""
    out = np.concatenate(
        [
            np.asarray(left_hand20, dtype=np.float64).reshape(HAND_DIM),
            np.asarray(left_wrist9, dtype=np.float64).reshape(WRIST_DIM),
            np.asarray(right_hand20, dtype=np.float64).reshape(HAND_DIM),
            np.asarray(right_wrist9, dtype=np.float64).reshape(WRIST_DIM),
        ]
    )
    assert out.shape == (WUJI58_DIM,)
    return out


def unpack58(vec) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split a 58-D vector into ``(left_hand, left_wrist, right_hand, right_wrist)``."""
    v = np.asarray(vec, dtype=np.float64).reshape(WUJI58_DIM)
    return v[LEFT_HAND], v[LEFT_WRIST], v[RIGHT_HAND], v[RIGHT_WRIST]


def is_wuji58(robot_action_dim_info: Mapping[str, Any], action_type: str) -> bool:
    """True when the run is the hybrid 58-D WUJI layout: 20-joint hands driven in ``ee`` space."""
    return (
        action_type == "ee"
        and list(robot_action_dim_info.get("ee_dim", [])) == [HAND_DIM, HAND_DIM]
        and len(robot_action_dim_info.get("arm_dim", [])) == 2
    )


def _state_block(state: Mapping[str, Any], key: str, dim: int) -> np.ndarray:
    if key not in state:
        raise KeyError(f"WUJI 58-D encoding needs '{key}' in the state dict")
    arr = np.asarray(state[key], dtype=np.float64).reshape(-1)
    if arr.shape != (dim,):
        raise ValueError(f"'{key}' must have {dim} values, got shape {arr.shape}")
    return arr


def state58_from_obs(obs_state: Mapping[str, Any], frame: WristFrame) -> np.ndarray:
    """Observation ``obs["state"]`` (standard keys) → 58-D hybrid state.

    Wrists are the absolute measured poses re-expressed in ``frame``; hands are absolute
    joint positions.
    """
    return pack58(
        _state_block(obs_state, "left_ee_joint_state", HAND_DIM),
        encode_wrist_state(_state_block(obs_state, "left_ee_pose", POSE_DIM), frame),
        _state_block(obs_state, "right_ee_joint_state", HAND_DIM),
        encode_wrist_state(_state_block(obs_state, "right_ee_pose", POSE_DIM), frame),
    )


def action58_from_target(
    target: Mapping[str, Any], obs_state: Mapping[str, Any], frame: WristFrame
) -> np.ndarray:
    """Absolute action target (standard keys) + the current observation → 58-D relative action."""
    return pack58(
        _state_block(target, "left_ee_joint_state", HAND_DIM),
        encode_wrist_relative(
            _state_block(target, "left_ee_pose", POSE_DIM),
            _state_block(obs_state, "left_ee_pose", POSE_DIM),
            frame,
        ),
        _state_block(target, "right_ee_joint_state", HAND_DIM),
        encode_wrist_relative(
            _state_block(target, "right_ee_pose", POSE_DIM),
            _state_block(obs_state, "right_ee_pose", POSE_DIM),
            frame,
        ),
    )


def decode_action58(
    action58, obs_state: Mapping[str, Any], frame: WristFrame, dtype=np.float32
) -> dict[str, np.ndarray]:
    """One 58-D relative action → the standard XPolicyLab action dict (absolute targets).

    ``obs_state`` is the observation the action is decoded *against*: the wrist the
    delta was formed relative to. For a whole chunk see :func:`decode_action_chunk58`
    and :class:`Wuji58ChunkDecoder`, which spell out which reference wrist each
    element owes.
    """
    lh, lw, rh, rw = unpack58(action58)
    return {
        "left_ee_pose": decode_wrist_relative(
            lw, _state_block(obs_state, "left_ee_pose", POSE_DIM), frame
        ).astype(dtype),
        "left_ee_joint_state": lh.astype(dtype),
        "right_ee_pose": decode_wrist_relative(
            rw, _state_block(obs_state, "right_ee_pose", POSE_DIM), frame
        ).astype(dtype),
        "right_ee_joint_state": rh.astype(dtype),
    }


#: How the wrist deltas inside a predicted chunk relate to a reference wrist.
#:
#: ``iterative``   (default) the training data carries PER-FRAME deltas — each
#:                 ``action[t]`` is ``target[t]`` vs. the wrist measured at ``t`` — which
#:                 is what WUJI SimBench records and what :func:`trajectory58` and the
#:                 official converters write. XPolicyLab hands the whole chunk to the
#:                 environment at once, so the live wrist of later elements is unknown;
#:                 element 0 decodes against the observation and element ``k`` against
#:                 the absolute pose decoded for ``k-1`` (the wrist is assumed to track the
#:                 previous target). Exact per-step decoding against the live wrist is
#:                 :class:`Wuji58ChunkDecoder`.
#: ``chunk_start`` every element is a delta vs. the wrist at the chunk's first frame
#:                 (the XR-1 convention). Only correct when the dataloader formed the
#:                 deltas that way — a per-frame dataset decoded like this is off by the
#:                 wrist's motion inside the chunk.
CHUNK_REFERENCES = ("iterative", "chunk_start")


def _wrist_reference(obs_state: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        "left_ee_pose": _state_block(obs_state, "left_ee_pose", POSE_DIM),
        "right_ee_pose": _state_block(obs_state, "right_ee_pose", POSE_DIM),
    }


def decode_action_chunk58(
    chunk,
    obs_state: Mapping[str, Any],
    frame: WristFrame,
    dtype=np.float32,
    reference: str = "iterative",
) -> list[dict[str, np.ndarray]]:
    """``(T, 58)`` relative chunk → ``list[dict]`` of absolute action dicts.

    ``obs_state`` is the observation the chunk was predicted from. ``reference`` names
    the convention the deltas were trained with (see :data:`CHUNK_REFERENCES`).
    """
    arr = np.asarray(chunk, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != WUJI58_DIM:
        raise ValueError(f"expected a (T, {WUJI58_DIM}) action chunk, got {arr.shape}")
    if reference not in CHUNK_REFERENCES:
        raise ValueError(f"unknown chunk reference {reference!r}; choose from {CHUNK_REFERENCES}")
    if reference == "chunk_start":
        return [decode_action58(row, obs_state, frame, dtype=dtype) for row in arr]
    out = []
    ref = _wrist_reference(obs_state)
    for row in arr:
        action = decode_action58(row, ref, frame, dtype=dtype)
        out.append(action)
        ref = {
            "left_ee_pose": action["left_ee_pose"].astype(np.float64),
            "right_ee_pose": action["right_ee_pose"].astype(np.float64),
        }
    return out


class Wuji58ChunkDecoder:
    """Per-step decoder for per-frame deltas: each element against the *live* wrist.

    This is exactly WUJI SimBench's runner: hold the predicted ``(T, 58)`` chunk and,
    on every environment step, decode the next element against the observation just
    received. An adapter that uses it returns one action per ``get_action`` call
    (``[decoder.next(obs["state"])]``) and re-runs the model when ``pending == 0``.
    """

    def __init__(self, frame: WristFrame, dtype=np.float32):
        self.frame = frame
        self.dtype = dtype
        self._chunk: np.ndarray | None = None
        self._cursor = 0

    @property
    def pending(self) -> int:
        return 0 if self._chunk is None else int(self._chunk.shape[0] - self._cursor)

    def start(self, chunk) -> None:
        arr = np.asarray(chunk, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != WUJI58_DIM:
            raise ValueError(f"expected a (T, {WUJI58_DIM}) action chunk, got {arr.shape}")
        self._chunk = arr
        self._cursor = 0

    def next(self, obs_state: Mapping[str, Any]) -> dict[str, np.ndarray]:
        if self.pending == 0:
            raise RuntimeError("no pending chunk element; call start(chunk) first")
        row = self._chunk[self._cursor]
        self._cursor += 1
        return decode_action58(row, obs_state, self.frame, dtype=self.dtype)

    def reset(self) -> None:
        self._chunk = None
        self._cursor = 0


# ── standard 54-D packed vector ↔ 58-D ───────────────────────────────────────


def packed54_to_state_dict(packed54) -> dict[str, np.ndarray]:
    """``pack_robot_state(..., action_type="ee")`` output for ``tianji_wuji`` → standard keys."""
    v = np.asarray(packed54, dtype=np.float64).reshape(-1)
    if v.shape != (54,):
        raise ValueError(f"expected the 54-D [pose7, hand20, pose7, hand20] vector, got {v.shape}")
    return {
        "left_ee_pose": v[0:7],
        "left_ee_joint_state": v[7:27],
        "right_ee_pose": v[27:34],
        "right_ee_joint_state": v[34:54],
    }


# ── trajectories (dataset conversion) ────────────────────────────────────────


def _dataset_block(group: Mapping[str, Any], key: str, dim: int) -> np.ndarray:
    if key not in group:
        raise KeyError(f"WUJI 58-D trajectory encoding needs '{key}'")
    arr = np.asarray(group[key], dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != dim:
        raise ValueError(f"'{key}' must be (T, {dim}), got {arr.shape}")
    return arr


def trajectory58(
    state: Mapping[str, Any], action: Mapping[str, Any], frame: WristFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Trajectory ``state/`` and ``action/`` groups (plural standard keys) → ``(T, 58)`` pair.

    State wrists are absolute in ``frame``; action wrists are the per-frame delta of the
    recorded target vs. that frame's measured wrist, so a policy trained on the output can
    be decoded with :func:`decode_action58` against the live observation.
    """
    s_lp = _dataset_block(state, "left_ee_poses", POSE_DIM)
    s_lh = _dataset_block(state, "left_ee_joint_states", HAND_DIM)
    s_rp = _dataset_block(state, "right_ee_poses", POSE_DIM)
    s_rh = _dataset_block(state, "right_ee_joint_states", HAND_DIM)
    a_lp = _dataset_block(action, "left_ee_poses", POSE_DIM)
    a_lh = _dataset_block(action, "left_ee_joint_states", HAND_DIM)
    a_rp = _dataset_block(action, "right_ee_poses", POSE_DIM)
    a_rh = _dataset_block(action, "right_ee_joint_states", HAND_DIM)
    lengths = {arr.shape[0] for arr in (s_lp, s_lh, s_rp, s_rh, a_lp, a_lh, a_rp, a_rh)}
    if len(lengths) != 1:
        raise ValueError(f"state/action lengths disagree: {sorted(lengths)}")
    T = lengths.pop()
    state58 = np.empty((T, WUJI58_DIM), dtype=np.float64)
    action58 = np.empty((T, WUJI58_DIM), dtype=np.float64)
    for t in range(T):
        state58[t] = pack58(
            s_lh[t], encode_wrist_state(s_lp[t], frame), s_rh[t], encode_wrist_state(s_rp[t], frame)
        )
        action58[t] = pack58(
            a_lh[t],
            encode_wrist_relative(a_lp[t], s_lp[t], frame),
            a_rh[t],
            encode_wrist_relative(a_rp[t], s_rp[t], frame),
        )
    return state58, action58


def _next_step_targets(states: np.ndarray) -> np.ndarray:
    """Emulate action targets as the next state (last row repeated), as sim exports do."""
    arr = np.asarray(states, dtype=np.float64)
    return np.concatenate([arr[1:], arr[-1:]], axis=0)


def trajectory58_from_data(data: Mapping[str, Any], frame: WristFrame) -> tuple[np.ndarray, np.ndarray]:
    """A loaded trajectory dict (``data["state"]`` / ``data["action"]`` groups) → ``(T, 58)`` pair.

    ``action/`` keys that are missing are emulated as next-step state targets, which is
    what the simulator exports carry (they record no leader command); recorded targets
    are used whenever present, per key.
    """
    if "state" not in data:
        raise KeyError("trajectory has no 'state' group")
    state = {k: np.asarray(v) for k, v in dict(data["state"]).items() if k in _TRAJ_KEYS}
    action_src = dict(data["action"]) if "action" in data and data["action"] is not None else {}
    action = {}
    for key in _TRAJ_KEYS:
        if key not in state:
            raise KeyError(f"WUJI 58-D trajectory encoding needs 'state/{key}'")
        if key in action_src and action_src[key] is not None:
            action[key] = np.asarray(action_src[key])
        else:
            action[key] = _next_step_targets(state[key])
    return trajectory58(state, action, frame)


_TRAJ_KEYS = ("left_ee_poses", "left_ee_joint_states", "right_ee_poses", "right_ee_joint_states")


__all__ = [
    "trajectory58_from_data",
    "WUJI58_DIM",
    "HAND_DIM",
    "WRIST_DIM",
    "POSE_DIM",
    "LEFT_HAND",
    "LEFT_WRIST",
    "RIGHT_HAND",
    "RIGHT_WRIST",
    "TIANJI_WUJI_ROBOT_ACTION_DIM_INFO",
    "HAND_JOINT_NAMES",
    "WRIST_FRAMES",
    "MOTOR_NAMES",
    "REPR_MODE",
    "TE2_EYE",
    "TE2_LOOK",
    "STATE_KEYS",
    "WristFrame",
    "quat_wxyz_to_matrix",
    "matrix_to_quat_wxyz",
    "matrix_to_rot6d",
    "rot6d_to_matrix",
    "camera_world_matrix",
    "encode_wrist_state",
    "decode_wrist_state",
    "encode_wrist_relative",
    "decode_wrist_relative",
    "pack58",
    "unpack58",
    "is_wuji58",
    "state58_from_obs",
    "action58_from_target",
    "decode_action58",
    "decode_action_chunk58",
    "CHUNK_REFERENCES",
    "Wuji58ChunkDecoder",
    "packed54_to_state_dict",
    "trajectory58",
]
