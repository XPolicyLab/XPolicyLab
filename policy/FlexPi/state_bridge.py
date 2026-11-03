"""Bridge between a 32D YAM-layout checkpoint and RoboDojo's 14D joint contract.

A checkpoint trained on flex-pi's YAM 32D layout (yam_eef.STATE_LAYOUT) expects
a raw 32D state and emits a raw 32D action; RoboDojo's arx_x5 packs 14D.
This module converts at the XPolicyLab boundary only — the checkpoint's own
processor (normalizer, merger, Yam32DRelativeAction) runs unchanged upstream.

32D layout (first two ROWS of R for rot6d):

    [ L_pos(3) L_rot6d(6) R_pos(3) R_rot6d(6) L_grip R_grip L_joint(6) R_joint(6) ]
      0:3      3:9        9:12     12:18      18     19     20:26      26:32

RoboDojo 14D (pack_robot_state, dual arm): [L_arm(6) L_grip R_arm(6) R_grip].
The 14 dims are a permutation of 32D slots, not a slice.

EEF block frame contract: RoboDojo's ``{left,right}_ee_pose`` is
``[x, y, z, qw, qx, qy, qz]`` in the frame the robot reports it in: in sim the
X5 ``link6`` pose, with position relative to the env origin and a world-frame
orientation; on a RoboDojo_real robot (arx_x5, piper, piper_x) that robot's own
EE pose as it records it. **Which frame the EEF block must be in depends on what
the checkpoint was trained on**, so ``eef_frame`` selects it:

``world_link6`` (the default)
    Pass that pose straight through. This is what the RoboDojo 34D dataset
    stores: ``get_real_endpose`` -> ``get_link_pose(is_relative=True)`` returns
    ``body_link_pose_w`` minus the env origin (translation only, orientation
    untouched), and the converter writes ``state/{left,right}_ee_poses``
    verbatim. Correct for any checkpoint trained on RoboDojo sim data, and for
    the raw RoboDojo_real corpus (``robodojo_realworld_34d``), which stores
    each robot's EE pose exactly as the robot reports it.

``arm_base_grasp``
    Re-express in each arm's own robot-base frame (``env_cfg/robot/<robot>.yml``
    default_root_pos/rot), then move from ``link6`` to the YAM ``grasp_site``
    tool frame the real-robot YAM checkpoints were trained on (the YAM rig's FK:
    i2rt YAM arm + linear_4310 gripper):

        R_grasp = R_link6 @ _LINK6_TO_GRASP_ROT
        p_grasp = p_link6 + R_link6 @ _LINK6_TO_GRASP_POS

    The constants were fit over 964 RoboDojo sim expert frames (stack_bowls,
    build_tower) against YAM FK evaluated at the same joint angles: the rotation
    is constant to 0.03 deg and the residual position error after the offset is
    under 1.4 cm, i.e. X5 and YAM share joint conventions and differ only in
    tool frame.

``relabeled``
    Keep the observed position and turn the rotation onto the YAM
    ``grasp_site`` axes (x away from the wrist camera, y the finger axis, z the
    approach axis) with one constant per robot, picked by ``env_cfg_type``:

        R_relabeled = R_obs @ _RELABEL_ROT[env_cfg_type]

    This is what the relabeled RoboDojo_real corpus
    ``robodojo_realworld_34d_relabeled`` stores (its ``meta/info.json``
    ``eef_frame.P`` is this table, applied to the raw corpus).
    Each of its three robots records the EE rotation in its own tool convention:
    arx_x5 in X5 link6 (approach +x, hence ``_LINK6_TO_GRASP_ROT``), piper
    already in the grasp_site axes, piper_x in those axes turned 90 deg about
    the approach axis. Positions keep each robot's own origin. An
    ``env_cfg_type`` without a constant raises.

Using the wrong one is silent and catastrophic, which is why
``_warn_if_arms_coincide`` prints the tell: applied to a RoboDojo-trained
checkpoint, ``arm_base_grasp`` put the EEF 0.64 m below anything in training
(the table height) with a 90 deg rotation error, and collapsed both arms onto
the same coordinates, scoring exactly 0.0 on all 40 tasks while every other
check passed.

Action direction. ``yam32_to_rd14`` executes the joint half of the 32D action
(``action_type: joint``). ``yam32_to_rd_ee16`` executes the EEF half instead
(``action_type: ee``): RoboDojo reads ``{left,right}_ee_pose`` as a target in
exactly the frame of the observed ``*_ee_pose`` (in sim the X5 link6 pose:
env-origin-relative position, world orientation, wxyz) and solves cuRobo IK
for it per arm and step. ``eef_block_to_pose7`` is the exact inverse of the
state path for every ``eef_frame``, so a checkpoint's EEF block goes back
through the frame it came in by. A target the IK cannot reach is not reported:
the simulator silently holds the arm at its previous command.

Pure numpy, no flexpi import, so it also runs under allow_dummy_policy.
"""

import os
import sys
from typing import Dict, Tuple

import numpy as np

YAM32_DIM = 32

# Entry i is the 32D slot that RoboDojo 14D dim i maps to.
_YAM32_IDS_FOR_RD14 = np.array(
    [20, 21, 22, 23, 24, 25, 18, 26, 27, 28, 29, 30, 31, 19], dtype=np.int64
)
# Gripper columns in RoboDojo 14D order.
_RD14_GRIP_COLS = (6, 13)
_EEF_SLICES = {"left": slice(0, 9), "right": slice(9, 18)}
# X5 link6 -> YAM grasp_site, expressed in the link6 frame (see module docstring).
# The tool approach axis is link6 +x and grasp_site +z.
_LINK6_TO_GRASP_ROT = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
_LINK6_TO_GRASP_POS = np.array([0.1439, 0.0033, 0.0123])
# Each robot's recorded EE rotation -> YAM grasp_site axes, right-multiplied, keyed
# by env_cfg_type (eef_frame relabeled; see the module docstring). The relabel table
# of robodojo_realworld_34d_relabeled, meta/info.json eef_frame.P.
_RELABEL_ROT = {
    "arx_x5": _LINK6_TO_GRASP_ROT,
    "piper": np.eye(3),
    "piper_x": np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
}

BRIDGE_MODES = ("auto", "none", "yam32")

# EEF-block frame conventions; see the module docstring.
EEF_FRAMES = ("world_link6", "arm_base_grasp", "relabeled")
DEFAULT_EEF_FRAME = "world_link6"

# What the adapter sends RoboDojo. `ee` needs the EEF block, i.e. the yam32 bridge.
ACTION_TYPES = ("joint", "ee")
# RoboDojo's ee target is [x, y, z, qw, qx, qy, qz] per arm (validate_action_dict
# in src/eval_client/eval_env.py). A simulator contract, not a robot dim, which is
# why _robot_info.json has no entry for it.
EE_POSE_DIM = 7
# Below this a rot6d row cannot be normalized, so the pose has no rotation.
_ROT6D_MIN_NORM = 1e-6


def _env_cfg_root() -> str:
    # Resolve env_cfg the way get_robot_action_dim_info does: relative to the
    # imported XPolicyLab package, which stays correct when this policy dir is
    # symlinked into the benchmark (Path(__file__).resolve() would follow the
    # link out of the benchmark tree).
    import XPolicyLab.utils.process_data as process_data

    return os.path.normpath(os.path.join(os.path.dirname(process_data.__file__), "../../env_cfg"))


def resolve_bridge(mode: str, ckpt_state_dim, robot_dim: int) -> str:
    """``ckpt_state_dim`` is the processor's raw state width: an int, or a tuple of
    admissible widths for mixed-embodiment checkpoints (``raw_shape: [32, 34]``,
    unified to 34D by AlignTo34DByWidth, whose YAM path takes 32-wide input)."""
    mode = str(mode or "auto").strip().lower()
    if mode not in BRIDGE_MODES:
        raise ValueError(f"state_bridge must be one of {BRIDGE_MODES}, got {mode!r}")
    if mode != "auto":
        return mode
    if ckpt_state_dim is None:
        return "none"
    widths = {int(ckpt_state_dim)} if isinstance(ckpt_state_dim, int) else {int(w) for w in ckpt_state_dim}
    if int(robot_dim) in widths:
        return "none"
    if YAM32_DIM in widths and int(robot_dim) == 14:
        return "yam32"
    raise ValueError(
        f"No state bridge for checkpoint state dim {ckpt_state_dim} and robot dim {robot_dim}. "
        "Set state_bridge explicitly in deploy.yml."
    )


def load_base_transforms(env_cfg_type: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Return ``{"left": (pos3, quat_wxyz), "right": (...)}`` in the env-origin frame."""
    from XPolicyLab.utils.load_file import load_yaml

    root = _env_cfg_root()
    env_cfg = load_yaml(os.path.join(root, f"{env_cfg_type}.yml"))
    robot_cfg = load_yaml(os.path.join(root, "robot", f"{env_cfg['config']['robot']}.yml"))
    robots = robot_cfg["robots"]
    if len(robots) != 2:
        raise ValueError(f"yam32 bridge needs a dual-arm robot, got {len(robots)} arms in {env_cfg_type}")
    return {
        side: (
            np.asarray(robots[i]["default_root_pos"], dtype=np.float64),
            _normalize_quat(robots[i]["default_root_rot"]),
        )
        for i, side in enumerate(("left", "right"))
    }


def relabel_rot(env_cfg_type: str) -> np.ndarray:
    """The ``relabeled`` tool-frame constant for ``env_cfg_type``; raises for a robot without one."""
    if env_cfg_type not in _RELABEL_ROT:
        raise ValueError(
            f"eef_frame relabeled has no tool-frame constant for env_cfg_type {env_cfg_type!r}; "
            f"it knows {sorted(_RELABEL_ROT)}"
        )
    return _RELABEL_ROT[env_cfg_type]


def _normalize_quat(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-8:
        raise ValueError(f"Degenerate quaternion {q}")
    return q / n


def _quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def env_pose_to_base(pose7, base_pos: np.ndarray, base_quat_wxyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Env-relative/world-oriented pose -> ``(pos3, R3x3)`` in the robot base frame."""
    pose7 = np.asarray(pose7, dtype=np.float64).reshape(-1)
    if pose7.shape[0] != 7:
        raise ValueError(f"Expected a 7D [x,y,z,qw,qx,qy,qz] pose, got shape {pose7.shape}")
    R_base = _quat_wxyz_to_mat(base_quat_wxyz)
    R_world = _quat_wxyz_to_mat(_normalize_quat(pose7[3:7]))
    pos = R_base.T @ (pose7[:3] - base_pos)
    return pos, R_base.T @ R_world


_warned_arms_coincide = False


def _warn_if_arms_coincide(state32: np.ndarray, eef_frame: str) -> None:
    """Say out loud when the EEF block puts both arms at one point.

    A dual-arm robot whose bases sit 0.6 m apart cannot have both end effectors
    at the same place in any frame the two arms SHARE — but in each arm's own
    base frame they coincide exactly whenever the arms mirror each other, which
    includes the home pose every episode starts from. So this is not an error:
    under ``arm_base_grasp`` it is the expected, correct behaviour, and the
    real-robot YAM checkpoints want it.

    It is worth printing because it is the visible signature of the failure mode
    that costs the most to diagnose — the same collapse is catastrophic for a
    checkpoint trained in a common frame, and nothing else catches it. A norm
    stats range check does not: a checkpoint fine-tuned from a mixture has q01/
    q99 wide enough to admit both the right and the wrong value, which is how an
    EEF block 0.64 m out of distribution scored 0.0 on all 40 RoboDojo tasks
    with every other check passing.
    """
    global _warned_arms_coincide
    if _warned_arms_coincide or eef_frame != "arm_base_grasp":
        return
    if not np.allclose(state32[0:3], state32[9:12], atol=1e-4):
        return
    _warned_arms_coincide = True
    print(
        f"[FlexPi bridge] NOTE: eef_frame={eef_frame!r} puts both arms at the same "
        f"EEF position {np.round(state32[0:3], 4).tolist()} — expected for a per-arm "
        "frame. If this checkpoint was trained on RoboDojo sim data, its EEF block "
        "is in a COMMON frame and this is wrong: set eef_frame: world_link6.",
        file=sys.stderr,
    )


def build_yam32_state(
    obs_state: dict,
    packed14: np.ndarray,
    base_tf: dict,
    eef_frame: str = DEFAULT_EEF_FRAME,
    env_cfg_type: str = None,
) -> np.ndarray:
    packed14 = np.asarray(packed14, dtype=np.float64).reshape(-1)
    if packed14.shape[0] != len(_YAM32_IDS_FOR_RD14):
        raise ValueError(f"Expected a 14D packed state, got {packed14.shape}")
    if eef_frame not in EEF_FRAMES:
        raise ValueError(f"eef_frame must be one of {EEF_FRAMES}, got {eef_frame!r}")

    state32 = np.zeros(YAM32_DIM, dtype=np.float64)
    state32[_YAM32_IDS_FOR_RD14] = packed14
    for side, sl in _EEF_SLICES.items():
        key = f"{side}_ee_pose"
        if key not in obs_state:
            raise KeyError(
                f"obs['state'] has no '{key}', which the yam32 bridge needs for the EEF block. "
                "Enable observation.robot.world_ee_state in the RoboDojo env_cfg."
            )
        if eef_frame in ("world_link6", "relabeled"):
            pose7 = np.asarray(obs_state[key], dtype=np.float64).reshape(-1)
            if pose7.shape[0] != 7:
                raise ValueError(
                    f"Expected a 7D [x,y,z,qw,qx,qy,qz] pose for '{key}', got {pose7.shape}"
                )
            pos = pose7[:3]
            R = _quat_wxyz_to_mat(_normalize_quat(pose7[3:7]))
            if eef_frame == "relabeled":
                R = R @ relabel_rot(env_cfg_type)
        else:
            pos, R = env_pose_to_base(obs_state[key], *base_tf[side])
            pos = pos + R @ _LINK6_TO_GRASP_POS
            R = R @ _LINK6_TO_GRASP_ROT
        state32[sl] = np.concatenate([pos, R[:2, :].reshape(6)])
    _warn_if_arms_coincide(state32, eef_frame)
    return state32.astype(np.float32)


def yam32_to_rd14(chunk32: np.ndarray) -> np.ndarray:
    chunk32 = np.asarray(chunk32, dtype=np.float32)
    if chunk32.shape[-1] != YAM32_DIM:
        raise ValueError(f"Expected a 32D action chunk, got {chunk32.shape}")
    chunk14 = chunk32[..., _YAM32_IDS_FOR_RD14].copy()
    for col in _RD14_GRIP_COLS:
        chunk14[..., col] = np.clip(chunk14[..., col], 0.0, 1.0)
    return chunk14


def ee_action_dim_info(robot_action_dim_info: dict) -> dict:
    """``get_robot_action_dim_info`` with each arm's width set to the ee pose width.

    Before XPolicyLab bb87d19, ``unpack_robot_state(..., "ee", ...)`` sized
    ``*_ee_pose`` by ``arm_dim``, which for arx_x5 is the 6 JOINTS, so the stock info
    cut the 7D pose to 6 and the simulator rejected it. Current XPolicyLab sizes it as
    7 itself; passing the width explicitly keeps the adapter right on either. Arm
    count and gripper widths still come from the robot info.
    """
    return {
        "arm_dim": [EE_POSE_DIM] * len(robot_action_dim_info["arm_dim"]),
        "ee_dim": list(robot_action_dim_info["ee_dim"]),
    }


def rot6d_rows_to_mat(r6) -> np.ndarray:
    """rot6d (the first two ROWS of R, row-major) -> R, Gram-Schmidt orthonormalized."""
    r6 = np.asarray(r6, dtype=np.float64).reshape(6)
    r1 = r6[0:3] / np.linalg.norm(r6[0:3])
    r2 = r6[3:6] - r1 * np.dot(r1, r6[3:6])
    r2 = r2 / np.linalg.norm(r2)
    return np.stack([r1, r2, np.cross(r1, r2)])


def _rot6d_is_valid(r6) -> bool:
    r6 = np.asarray(r6, dtype=np.float64).reshape(6)
    if not np.isfinite(r6).all() or np.linalg.norm(r6[0:3]) < _ROT6D_MIN_NORM:
        return False
    r1 = r6[0:3] / np.linalg.norm(r6[0:3])
    return np.linalg.norm(r6[3:6] - r1 * np.dot(r1, r6[3:6])) >= _ROT6D_MIN_NORM


def mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion ``[qw, qx, qy, qz]``, canonical ``qw >= 0``."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.asarray(q) / np.linalg.norm(q)
    return -q if q[0] < 0 else q


def eef_block_to_pose7(
    block9,
    side: str,
    eef_frame: str = DEFAULT_EEF_FRAME,
    base_tf: dict = None,
    env_cfg_type: str = None,
) -> np.ndarray:
    """One arm's EEF block ``[pos3, rot6d rows]`` -> RoboDojo ``[x, y, z, qw, qx, qy, qz]``.

    The exact inverse of what ``build_yam32_state`` does to ``obs['state']['<side>_ee_pose']``:
    ``world_link6`` only converts the rotation; ``relabeled`` also takes the
    robot's tool-frame constant back off it; ``arm_base_grasp`` moves the YAM
    grasp_site back to link6 and the arm's base frame back to the env frame.
    """
    if eef_frame not in EEF_FRAMES:
        raise ValueError(f"eef_frame must be one of {EEF_FRAMES}, got {eef_frame!r}")
    block9 = np.asarray(block9, dtype=np.float64).reshape(9)
    pos, R = block9[:3], rot6d_rows_to_mat(block9[3:9])
    if eef_frame == "arm_base_grasp":
        base_pos, base_quat = base_tf[side]
        R_base = _quat_wxyz_to_mat(base_quat)
        R = R @ _LINK6_TO_GRASP_ROT.T
        pos = pos - R @ _LINK6_TO_GRASP_POS
        R, pos = R_base @ R, R_base @ pos + base_pos
    elif eef_frame == "relabeled":
        R = R @ relabel_rot(env_cfg_type).T
    return np.concatenate([pos, mat_to_quat_wxyz(R)])


def yam32_to_rd_ee16(
    chunk32: np.ndarray,
    eef_frame: str = DEFAULT_EEF_FRAME,
    base_tf: dict = None,
    hold_pose7: dict = None,
    env_cfg_type: str = None,
):
    """32D action chunk -> RoboDojo ee layout ``[T, 16]``, plus the count of held poses.

    Layout ``[L_pose7 L_grip R_pose7 R_grip]``, the XPolicyLab pack order with each
    arm as its 7D pose, so ``unpack_robot_state(.., "ee", ee_action_dim_info(..))``
    turns it into the simulator's dicts. Grippers are clipped to [0, 1] as in
    ``yam32_to_rd14``.

    A pose that has no rotation (non-finite, or a rot6d row that cannot be
    normalized) is replaced by ``hold_pose7[side]`` -- the arm's current pose -- and
    counted rather than raised: an exception here reaches the eval client, whose
    handler silently drops the rest of the batch. Without ``hold_pose7`` it raises.
    """
    chunk32 = np.asarray(chunk32, dtype=np.float32)
    if chunk32.shape[-1] != YAM32_DIM or chunk32.ndim != 2:
        raise ValueError(f"Expected a [T, 32] action chunk, got {chunk32.shape}")
    width = EE_POSE_DIM + 1
    chunk16 = np.zeros((chunk32.shape[0], 2 * width), dtype=np.float32)
    held = 0
    for i, (side, sl) in enumerate(_EEF_SLICES.items()):
        out = slice(i * width, i * width + EE_POSE_DIM)
        for t, block9 in enumerate(chunk32[:, sl]):
            if np.isfinite(block9[:3]).all() and _rot6d_is_valid(block9[3:9]):
                chunk16[t, out] = eef_block_to_pose7(block9, side, eef_frame, base_tf, env_cfg_type)
            elif hold_pose7 is not None:
                chunk16[t, out] = hold_pose7[side]
                held += 1
            else:
                raise ValueError(f"{side} EEF target at step {t} has no valid rotation: {block9}")
        chunk16[:, i * width + EE_POSE_DIM] = np.clip(chunk32[:, 18 + i], 0.0, 1.0)
    return chunk16, held


def pose7_error(a, b):
    """``(position error mm, rotation error deg)`` between two ``[x, y, z, qw, qx, qy, qz]``."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    dot = abs(float(np.dot(_normalize_quat(a[3:7]), _normalize_quat(b[3:7]))))
    return 1000.0 * float(np.linalg.norm(a[:3] - b[:3])), float(np.degrees(2.0 * np.arccos(min(dot, 1.0))))
