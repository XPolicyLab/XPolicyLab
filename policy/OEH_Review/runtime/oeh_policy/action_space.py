"""The unified action space: OpenWAM-α's 80 slots with one meaning per slot, and the review layout, its compact view.

Each benchmark's official loop takes its own native action (``oeh_eval.benchmarks.Benchmark.action_space`` names it):
RoboDojo's dict rows of absolute EE poses, RoboTwin's flat EE16 vector, LIBERO's 7D OSC delta. The fast and hybrid
conditions keep those native rows untouched; the official conversions own them.

The unified space is OpenWAM-α's action and state layout (XPolicyLab fa431ec, assets/openwam_usage_docs/
openwam-alpha-finetuning.md, "The α action and preprocessing contract"; ``SLOTS``): one 80-slot vector for every robot.
Each arm has an end-effector block, position (3), rot6d (6, the rotation matrix's first two columns) and gripper
(1, -1 closed, +1 open), then 24 hand joints; slots 68-79 are reserved, where OpenWAM puts a mobile base, a torso or a
waist. A robot fills the slots it has and masks the rest, and a single arm fills the left block. Each ``ActionSpace``
declares the slot map OpenWAM's own dataloader declares for its benchmark (``unify_action_map``, as in
configs/dataloader/<benchmark>.yaml), and its arms are the blocks that map fills (``unified_arms``): ``left`` and
``right`` on RoboDojo and RoboTwin, ``left`` for LIBERO's single Panda. OEH gives the reserved slots no unified meaning
yet, because OpenWAM fills them with a different command on each benchmark (EBench a per-step base displacement,
RoboCasa365 base motion and a control mode): a space whose map fills some names what they carry (``carried``), keeps
their native values in OpenWAM's rows, and leaves them 0 and masked out of the unified vector (``reserved_slots``).
An arm without a gripper slot is declared by its space (``ActionSpace.gripper`` False; OpenWAM's GR1 map fills each
arm's pose, six joints of its Fourier hand and the waist, never a gripper slot): its block is the 9 pose slots, its
review layout has no opening, and the space carries the hand slots it fills the same way (``carried_slots``).

OpenWAM's contract asks every reader for the same physical meaning per slot, but its readers differ by dataset: they
fill RoboDojo's blocks in each arm's robot-base frame and RoboTwin's in the world frame, build RoboTwin's rot6d from
its wxyz quaternion read as xyzw (the identity lands as a half turn about x), put a normalized OSC delta rather than a
pose in LIBERO's action slots, and min-max normalize every slot but rot6d with each checkpoint's statistics. OEH's
unified space is re-adjusted to one meaning on every benchmark. Positions are metres in the benchmark's world frame,
rot6d is of the end effector's actual orientation, and the gripper slot is 2 * opening - 1: in a row the commanded
opening, never a measured finger gap; in a state what its benchmark observes, the last command on RoboDojo and
RoboTwin and the finger gap on LIBERO (``LiberoEEF10``), whose review method keeps the command in its ledger. Rows are
absolute targets, never deltas, and nothing is normalized. ``ActionSpace.unified_state`` and ``unified_rows`` give any
benchmark's state and rows in it. OEH never sends this vector to OpenWAM: the official clients own OpenWAM's inputs
and outputs, so the re-adjustment changes no run.

Review mode reads the same end-effector blocks in a compact view that the teacher, the review method
(``oeh_policy.review.ReviewMethod``) and the review guard share on every benchmark, the *review layout*: one entry per
arm with the same meaning, the orientation as a unit wxyz quaternion (four numbers, not rot6d's six) and the gripper as
the opening, 1 = open and 0 = closed. One row, or one measured state, is::

    {"<arm>_ee_pose": [x, y, z, qw, qx, qy, qz], "<arm>_ee_joint_state": [opening]}

(without ``<arm>_ee_joint_state`` for arms without a gripper slot). ``to_unified`` and ``from_unified`` convert
between the two.

RoboDojo's native rows and state already have this layout. The formal rounds froze its review packets, and they are
the reference every other benchmark's packets follow. Every other benchmark converts at its boundary through an
``ActionSpace``, and the review method computes every correction (``edit``, ``eef``, ``home``, ``hold``) in the review
layout, so the same decision means the same motion on every benchmark. ``student`` rows are OpenWAM's native rows,
unchanged, on every benchmark.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from .contracts import project_state
from .gateway import FollowingRows
from .libero_boundary import validate_native_action
from .single_arm import achieved, matrix_rotvec, rotvec_matrix

REVIEW_KEYS = ("_ee_pose", "_ee_joint_state")

# OpenWAM-α's slot table (half-open ranges; the module docstring cites it), and where each arm's 10-slot end-effector
# block [xyz, rot6d, gripper] starts.
UNIFY_DIM = 80
SLOTS = {"left_eef_xyz": (0, 3), "left_eef_rot6d": (3, 9), "left_gripper": (9, 10), "left_hand": (10, 34),
         "right_eef_xyz": (34, 37), "right_eef_rot6d": (37, 43), "right_gripper": (43, 44), "right_hand": (44, 68),
         "reserved": (68, 80)}
EEF_SLOTS = {"left": 0, "right": 34}


def slot_map(spec) -> tuple[int, ...]:
    """A ``unify_action_map`` in OpenWAM's single-list form (``openwam.dataloader.utils.unify_action.
    parse_unify_spec``) -> the unified slot of each native dimension, in order. A token is a slot or a closed range
    ``"a-b"``; the slots are distinct and below ``UNIFY_DIM``."""
    slots = []
    for token in spec:
        if isinstance(token, (int, np.integer)) and not isinstance(token, bool):
            slots.append(int(token))
            continue
        low, dash, high = str(token).partition("-")
        try:
            first, last = int(low), int(high if dash else low)
        except ValueError:
            raise ValueError(f"unified slot map token {token!r} is a slot or a closed range 'a-b'") from None
        if last < first:
            raise ValueError(f"unified slot map range {token!r} ends before it starts")
        slots.extend(range(first, last + 1))
    if not slots or len(set(slots)) != len(slots) or min(slots) < 0 or max(slots) >= UNIFY_DIM:
        raise ValueError(f"unified slot map {list(spec)!r} must name distinct slots in [0, {UNIFY_DIM})")
    return tuple(slots)


def unified_arms(spec, gripper: bool = True) -> tuple[str, ...]:
    """The arms whose end-effector blocks a slot map fills, in slot order. The review layout has fields for whole
    end-effector blocks only, so a map that fills part of one or a hand is refused. Reserved slots are allowed: the
    space carries them natively (``reserved_slots``; OpenWAM's EBench and RoboCasa365 maps put a base there).

    ``gripper=False`` is for a space whose arms have no gripper slot (``ActionSpace.gripper``; OpenWAM's GR1 map): an
    arm's block is then its 9 pose slots [xyz, rot6d], its gripper slot stays empty, and the hand slots of the arms it
    fills are allowed besides the reserved ones, carried natively the same way (``carried_slots``)."""
    filled = set(slot_map(spec)) - set(range(*SLOTS["reserved"]))
    if not gripper:
        arms = tuple(arm for arm, start in EEF_SLOTS.items() if filled >= set(range(start, start + 9)))
        allowed = {slot for arm in arms
                   for slot in (*range(EEF_SLOTS[arm], EEF_SLOTS[arm] + 9), *range(*SLOTS[arm + "_hand"]))}
        if not arms or not filled <= allowed:
            raise ValueError(f"unified slot map {list(spec)!r} of arms without a gripper slot must fill whole "
                             "end-effector pose blocks (left 0-8, right 34-42) and otherwise only those arms' hand "
                             "slots (10-33, 44-67) and reserved slots (68-79)")
        return arms
    arms = tuple(arm for arm, start in EEF_SLOTS.items() if filled >= set(range(start, start + 10)))
    if not arms or filled != {EEF_SLOTS[arm] + offset for arm in arms for offset in range(10)}:
        raise ValueError(f"unified slot map {list(spec)!r} must fill whole end-effector blocks (left 0-9, "
                         "right 34-43) and otherwise only reserved slots (68-79)")
    return arms


def reserved_slots(spec) -> tuple[int, ...]:
    """The reserved slots (68-79) a slot map fills, in native order. They have no unified meaning yet: ``to_unified``
    leaves them 0 and masked out, and the benchmark's space carries their native values (``ActionSpace.carried``)."""
    low, high = SLOTS["reserved"]
    return tuple(slot for slot in slot_map(spec) if low <= slot < high)


def carried_slots(spec, gripper: bool = True) -> tuple[int, ...]:
    """Every slot a slot map fills outside its arms' end-effector blocks, in native order: the reserved slots and, for
    arms without a gripper slot, the hand slots (``unified_arms``). Like the reserved slots they have no unified
    meaning: ``to_unified`` leaves them 0 and masked out and the space carries their native values
    (``ActionSpace.carried``). With grippers this is ``reserved_slots``."""
    width = 10 if gripper else 9
    blocks = {EEF_SLOTS[arm] + offset for arm in unified_arms(spec, gripper) for offset in range(width)}
    return tuple(slot for slot in slot_map(spec) if slot not in blocks)


def rot6d_matrix(six, name: str = "rot6d") -> np.ndarray:
    """A rot6d (a rotation matrix's first two columns) -> the rotation matrix, orthonormalized first (Gram-Schmidt,
    as OpenWAM decodes it)."""
    first, second = np.asarray(six[:3], dtype=float), np.asarray(six[3:6], dtype=float)
    if np.linalg.norm(first) < 1e-9:
        raise ValueError(f"degenerate {name}")
    first = first / np.linalg.norm(first)
    second = second - float(first @ second) * first
    if np.linalg.norm(second) < 1e-9:
        raise ValueError(f"degenerate {name}")
    second = second / np.linalg.norm(second)
    return np.stack([first, second, np.cross(first, second)], axis=1)


def to_unified(row: Mapping, arms: tuple[str, ...], gripper: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """A review-layout row or state -> the unified 80-slot vector and its mask. Each arm's block holds its position
    (metres, the benchmark's world frame), the rot6d of its quaternion and its opening on OpenWAM's gripper scale,
    2 * opening - 1; every other slot is 0 and masked out. Nothing is normalized: the slots are OpenWAM's, the meaning
    is the one the module docstring fixes. Arms without a gripper slot (``gripper=False``) fill their 9 pose slots
    only."""
    checked = check_review_row(row, arms, gripper)
    vector, mask = np.zeros(UNIFY_DIM, dtype=np.float32), np.zeros(UNIFY_DIM, dtype=bool)
    width = 10 if gripper else 9
    for arm in arms:
        start, pose = EEF_SLOTS[arm], checked[arm + "_ee_pose"].astype(float)
        rotation = quat_matrix(pose[3:])
        block = [pose[:3], rotation[:, 0], rotation[:, 1]]
        if gripper:
            block.append([2.0 * float(checked[arm + "_ee_joint_state"][0]) - 1.0])
        vector[start:start + width] = np.concatenate(block)
        mask[start:start + width] = True
    return vector, mask


def from_unified(vector, arms: tuple[str, ...], gripper: bool = True) -> dict:
    """The arms' end-effector blocks of an 80-slot vector -> a review-layout row: the inverse of ``to_unified``. The
    rot6d is orthonormalized first (Gram-Schmidt, as OpenWAM decodes it); the quaternion is ``matrix_quat``'s, which
    may have the opposite sign of the one encoded."""
    value = np.asarray(vector, dtype=float).reshape(-1)
    if value.shape != (UNIFY_DIM,) or not np.isfinite(value).all():
        raise ValueError(f"a unified vector has {UNIFY_DIM} finite slots")
    row = {}
    for arm in arms:
        block = value[EEF_SLOTS[arm]:EEF_SLOTS[arm] + 10]
        rotation = rot6d_matrix(block[3:9], f"{arm} rot6d")
        row[arm + "_ee_pose"] = np.concatenate([block[:3], matrix_quat(rotation)]).astype(np.float32)
        if gripper:
            row[arm + "_ee_joint_state"] = np.array([(block[9] + 1.0) / 2.0], dtype=np.float32)
    return row


class ActionSpace:
    """A benchmark's native EE action and state <-> the review layout.

    Conversions take the measured state they depend on as an argument and keep no episode state of their own. A
    delta controller such as LIBERO's needs the measured pose to place a native row in the world, and the post-step
    pose to compute the next native row.
    """

    name = ""
    # OpenWAM's slot map for this benchmark's native end-effector vector (its dataloader yaml's unify_action_map); the
    # arms are the end-effector blocks it fills.
    unify_action_map: tuple = ("0-9", "34-43")
    arms: tuple[str, ...] = unified_arms(unify_action_map)
    native = ""  # one line: the native action the official loop takes
    # One line: what the native vector carries in the reserved slots its map fills, which OpenWAM's rows keep and a
    # correction sets to the space's hold value (``native_rows``); empty when the map fills none.
    carried = ""
    # Whether each arm's block has a gripper slot (the review layout's ``<arm>_ee_joint_state``). A space whose map
    # fills none sets it False and carries its hand slots like the reserved ones (``carried_slots``).
    gripper = True

    def review_state(self, state) -> dict:
        """The projected native state (``Runtime.observation["state"]``) in the review layout."""
        raise NotImplementedError

    def review_rows(self, rows, state) -> list[dict]:
        """Native rows proposed on ``state`` in the review layout: the targets the teacher reads and edits ramp from."""
        raise NotImplementedError

    def native_rows(self, rows: list[dict], state):
        """Validated review-layout rows of a correction -> the native rows offered to the gate. A delta controller
        returns an ``oeh_policy.gateway.FollowingRows``, whose later rows are computed from the post-step state."""
        raise NotImplementedError

    def openings(self, action) -> dict[str, float]:
        """The opening each arm's gripper is commanded to by one native action (the command ledger)."""
        raise NotImplementedError

    def unified_state(self, state) -> tuple[np.ndarray, np.ndarray]:
        """The projected native state in the unified space: its 80-slot vector and mask."""
        return to_unified(self.review_state(state), self.arms, self.gripper)

    def unified_rows(self, rows, state) -> tuple[np.ndarray, np.ndarray]:
        """Native rows proposed on ``state`` in the unified space, absolute targets on every benchmark: an array of
        one 80-slot vector per row, and the mask they share."""
        unified = [to_unified(row, self.arms, self.gripper) for row in self.review_rows(rows, state)]
        if not unified:
            raise ValueError("unified rows need at least one native row")
        return np.stack([vector for vector, _ in unified]), unified[0][1]


class RoboDojoEE(ActionSpace):
    """RoboDojo: the native rows and state are the review layout (env-relative world frame). OpenWAM's reader fills
    the same two blocks after moving each pose into its arm's robot-base frame (openwam/dataloader/robodojo.py); the
    unified space keeps both arms in the env-relative world frame."""

    name = "robodojo_ee"
    unify_action_map = ("0-9", "34-43")  # OpenWAM configs/dataloader/robodojo.yaml: EEF20, left then right
    arms = unified_arms(unify_action_map)
    native = "per-arm dict rows <arm>_ee_pose (xyz, wxyz; env-relative world) and <arm>_ee_joint_state (1 open)"

    def review_state(self, state) -> dict:
        return project_state(state)

    def review_rows(self, rows, state) -> list[dict]:
        return rows

    def native_rows(self, rows, state):
        return rows

    def openings(self, action) -> dict[str, float]:
        return {arm: float(action[arm + "_ee_joint_state"][0]) for arm in self.arms}


def endpose_to_dual(state) -> dict:
    """Projected RoboTwin endpose -> the review layout (wxyz kept)."""
    result = {}
    for arm in ("left", "right"):
        pose = np.asarray(state[arm + "_endpose"], dtype=np.float32).reshape(-1)
        grip = np.asarray(state[arm + "_gripper"], dtype=np.float32).reshape(-1)
        if pose.shape != (7,) or grip.shape != (1,) or not np.isfinite(np.concatenate([pose, grip])).all():
            raise ValueError(f"invalid RoboTwin {arm} endpose")
        result[arm + "_ee_pose"], result[arm + "_ee_joint_state"] = pose.copy(), grip.copy()
    return result


def ee16_to_dual(action) -> dict:
    row = np.asarray(action, dtype=np.float32).reshape(-1)
    if row.shape != (16,) or not np.isfinite(row).all():
        raise ValueError("RoboTwin native EE action must contain 16 finite scalars")
    return {"left_ee_pose": row[0:7].copy(), "left_ee_joint_state": row[7:8].copy(),
            "right_ee_pose": row[8:15].copy(), "right_ee_joint_state": row[15:16].copy()}


def dual_to_ee16(row) -> np.ndarray:
    return np.concatenate([np.asarray(row[key], dtype=np.float32).reshape(-1) for key in
                           ("left_ee_pose", "left_ee_joint_state", "right_ee_pose", "right_ee_joint_state")])


class RoboTwinEE16(ActionSpace):
    """RoboTwin 0aeea2d: ``take_action(action, action_type="ee")`` reads ``[l_xyz, l_quat, l_grip, r_xyz, r_quat,
    r_grip]`` in its world frame, quaternions wxyz (transforms3d), the gripper 1 open / 0 closed, and the observed
    gripper is the last command (oeh_policy.robotwin_hybrid). The layouts differ only in shape. OpenWAM's reader
    fills the same two blocks in the world frame but reads the quaternion as xyzw, so its rot6d is of the reordered
    quaternion, consistently in training and deployment (oeh_policy.robotwin); the review layout keeps it wxyz, and
    the unified space holds the rot6d of that actual orientation."""

    name = "robotwin_ee16"
    unify_action_map = ("0-9", "34-43")  # OpenWAM configs/dataloader/robotwin.yaml (action_mode eef)
    arms = unified_arms(unify_action_map)
    native = "EE16 [l_xyz, l_quat_wxyz, l_grip, r_xyz, r_quat_wxyz, r_grip] in RoboTwin's world frame (1 open)"

    def review_state(self, state) -> dict:
        return endpose_to_dual(state)

    def review_rows(self, rows, state) -> list[dict]:
        return [ee16_to_dual(row) for row in rows]

    def native_rows(self, rows, state):
        return [dual_to_ee16(row) for row in rows]

    def openings(self, action) -> dict[str, float]:
        row = ee16_to_dual(action)
        return {arm: float(row[arm + "_ee_joint_state"][0]) for arm in self.arms}


def matrix_quat(matrix) -> np.ndarray:
    """Rotation matrix -> unit quaternion wxyz (Shepperd: stable at every angle), in the hemisphere of the
    gripper-down orientation (0, 1, 0, 0) (180 deg about world x, LIBERO's working orientation, where qw is about 0):
    a w >= 0 convention would flip the sign between packets there. Every check and correction is sign-invariant
    (oeh_policy.review.rotvec_between); the hemisphere only keeps the numbers the teacher reads steady."""
    r = np.asarray(matrix, dtype=float)
    trace = float(np.trace(r))
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        q = [(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        q = [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        q = [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s]
    q = np.asarray(q, dtype=float)
    q /= np.linalg.norm(q)
    return -q if q[1] < 0 or (q[1] == 0 and q[0] < 0) else q


def quat_matrix(quaternion) -> np.ndarray:
    """Quaternion wxyz (normalized first) -> rotation matrix."""
    q = np.asarray(quaternion, dtype=float)
    norm = float(np.linalg.norm(q))
    if q.shape != (4,) or not np.isfinite(q).all() or norm < 1e-9:
        raise ValueError("quaternion must be four finite numbers, not all zero")
    w, x, y, z = q / norm
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def bounded_osc(scale, physical) -> np.ndarray:
    """The normalized OSC command for a physical delta (m, rad): translation and rotation each scaled down along
    their own direction until they fit robosuite's input range. ``OscScale.to_native`` alone clips every axis, which
    bends a move the arm lags far behind toward the box corner; a correction keeps its direction."""
    command = (np.asarray(physical, dtype=float) - scale.output_offset) / scale.scale + scale.input_offset
    for part in (slice(0, 3), slice(3, 6)):
        reach = np.abs(command[part] - scale.input_offset[part]) / ((scale.input_max[part] - scale.input_min[part]) / 2)
        factor = float(np.max(reach))
        if factor > 1:
            command[part] = scale.input_offset[part] + (command[part] - scale.input_offset[part]) / factor
    return np.clip(command, scale.input_min, scale.input_max)


class LiberoEEF10(ActionSpace):
    """The LIBERO family (LIBERO, LIBERO-Plus, LIBERO-Pro): robosuite OSC_POSE with ``control_delta``.

    Native action: OpenWAM's raw EEF10 converted by the pinned released client (``native_eef10_to_libero7d``) into
    one 7D OSC row ``[dxyz, drotvec, gripper]`` in [-1, 1] per official step, scaled by the frozen OSC_POSE ranges
    (``osc_controller``; 0.05 m and 0.5 rad at robosuite 1.4.0). robosuite sets the step's goal to the pose measured
    at the start of the step plus that scaled delta (``ee_pos + dx``, ``R(dr) @ ee_ori``; an exactly zero rotation
    keeps the previous orientation goal) and the arm lags behind each goal (kp 150). The gripper element is a
    command: -1 opens, +1 closes; opening = (1 - g) / 2.

    State: the achieved EEF10 the released client builds (``libero_obs_to_eef10``): ``robot0_eef_pos`` (the grip
    site, world frame), the OpenWAM rot6d of ``robot0_eef_quat`` (the hand body; its constant offset from the grip
    site cancels in every world-frame delta) and the *measured* finger gap on OpenWAM's open scale,
    ``clip(2 * width / 0.08 - 1, -1, 1)``. LIBERO's state carries no command, so ``review_state`` reports the measured
    opening ``width / 0.08`` (1 open, 0 closed) and the review method keeps the commanded one in its ledger; the packet
    shows both (docs/benchmark-adapters/README.md, principle 3).

    ``review_rows`` turns native rows into absolute review targets by chaining their scaled deltas from the measured
    pose: the goals the rows command if every step reached its goal. They are commanded references, not poses the arm
    will reach (it lags). ``native_rows`` goes the other way closed loop: each row of a correction is the delta from
    the newest achieved pose to its review target (``Toward``, an ``oeh_policy.gateway.FollowingRows`` follower), so
    tracking lag never accumulates, as ``oeh_policy.single_arm.Plan`` does for the hybrid primitives.

    The one arm is ``left``: OpenWAM's LIBERO slot map fills the left end-effector block, as every single-arm map
    of OpenWAM-α does. In that block OpenWAM's action is the normalized delta ``native_eef10_to_libero7d`` decodes
    and its state the achieved EEF10; the review layout and the unified space read both as absolute world-frame poses.

    The conversions need the frozen OSC scale and the official conversion (``codec.encode``,
    ``oeh_policy.libero_runtime.LiberoBackend``); the registry's bare instance (``SPACES``) names the space only.
    """

    name = "libero_eef10"
    unify_action_map = ("0-9",)  # OpenWAM configs/dataloader/libero.yaml: the one arm fills the left block
    arms = unified_arms(unify_action_map)
    arm = arms[0]
    native = "7D OSC delta [dxyz, drotvec, gripper] in [-1, 1] (robosuite OSC_POSE control_delta; gripper -1 open)"
    frame = "LIBERO world frame (robosuite), position m, quaternion wxyz"

    def __init__(self, scale=None, codec=None):
        self.scale, self.codec = scale, codec

    def _scale(self):
        if self.scale is None:
            raise ValueError("LIBERO review conversions need the frozen OSC_POSE scale")
        return self.scale

    @classmethod
    def _pose_row(cls, position, rotation, opening) -> dict:
        return {cls.arm + "_ee_pose": np.concatenate([position, matrix_quat(rotation)]).astype(np.float32),
                cls.arm + "_ee_joint_state": np.array([float(np.clip(opening, 0.0, 1.0))], dtype=np.float32)}

    def review_state(self, state) -> dict:
        position, rotation = achieved(state)
        return self._pose_row(position, rotation, (float(np.asarray(state, dtype=float)[9]) + 1.0) / 2.0)

    def review_rows(self, rows, state) -> list[dict]:
        scale = self._scale()
        position, rotation = achieved(state)
        result = []
        for row in rows:
            native = np.asarray(validate_native_action(row), dtype=float)
            physical = scale.to_physical(native[:6])
            position = position + physical[:3]
            if np.any(physical[3:] != 0):  # robosuite keeps the orientation goal on an exactly zero rotation
                rotation = rotvec_matrix(physical[3:]) @ rotation
            result.append(self._pose_row(position, rotation, (1.0 - native[6]) / 2.0))
        return result

    def toward(self, row: Mapping, state) -> list[float]:
        """The native row from the achieved pose in ``state`` toward one review-layout target: the world-frame delta,
        bounded to the OSC input range along its own direction, and the target's opening as the gripper command;
        admitted through the official conversion exactly as the released client will apply it."""
        if self.codec is None or not callable(getattr(self.codec, "encode", None)):
            raise ValueError("LIBERO review rows need the official OSC conversion")
        position, rotation = achieved(state)
        pose = np.asarray(row[self.arm + "_ee_pose"], dtype=float)
        physical = np.concatenate([pose[:3] - position, matrix_rotvec(quat_matrix(pose[3:]) @ rotation.T)])
        grip = 1.0 - 2.0 * float(np.clip(np.asarray(row[self.arm + "_ee_joint_state"], dtype=float)[0], 0.0, 1.0))
        return self.codec.encode(np.concatenate([bounded_osc(self._scale(), physical), [grip]]))

    def native_rows(self, rows, state):
        targets = [check_review_row(row, self.arms) for row in rows]
        if not targets:
            raise ValueError("a LIBERO correction has at least one row")
        follow = Toward(self, targets)
        return FollowingRows([follow(state)], len(targets), follow)

    def openings(self, action) -> dict[str, float]:
        return {self.arm: float(np.clip((1.0 - validate_native_action(action)[6]) / 2.0, 0.0, 1.0))}


class Toward:
    """The follower of one LIBERO correction (``FollowingRows.follow``): call ``k`` returns the native row toward
    review target ``k`` from the state it is given, the post-step observation for every row after the first
    (``Runtime.next_action``). ``targets`` are the correction's review rows, which the record reports."""

    def __init__(self, space: LiberoEEF10, targets: list[dict]):
        self.space, self.targets, self.index = space, targets, 0

    def __call__(self, state) -> list[float]:
        if self.index >= len(self.targets):
            raise RuntimeError("the correction's rows are exhausted")
        target = self.targets[self.index]
        self.index += 1
        return self.space.toward(target, state)


class RoboCasaGR1EEF33(ActionSpace):
    """RoboCasa GR1 tabletop (robocasa-gr1-tabletop-tasks 4840e67 on robosuite 1.5.1): the GR1 humanoid's two arms,
    Fourier hands and waist under OpenWAM's EEF33.

    Native action and state: OpenWAM's EEF33 (openwam/dataloader/utils/gr1_kinematics.py at fa431ec,
    ``EEF33_SLICES``): per arm the pose of the hand's grip site, xyz (m) and rot6d (the rotation matrix's first two
    columns), in the robot-base frame (the pelvis body ``robot0_base``, welded in place, so the frame does not move
    during an episode: +x forward, +y the robot's left, +z up), then the hand's six joint targets (rad); after both
    arms the waist's yaw, pitch and roll (rad). The released client solves the poses into absolute arm joint targets
    with its damped least-squares IK (``GR1Kinematics.eef33_to_action_dict``: the waist is set to its target first,
    and a solve that misses 2 mm / 0.02 rad keeps both arms' joints where they are) and passes the hands and the
    waist through unchanged; the state is the same layout of the measured joints and grip sites
    (``observation_to_eef33``).

    OpenWAM's map fills each arm's 9 pose slots, six of its hand slots and three reserved slots, never a gripper
    slot: the arms end in hands, not grippers (``gripper`` False). The review layout therefore has one field per arm,
    the pose, in this frame (the benchmark's world frame for review: the one OpenWAM's rows and the IK use); the
    teacher corrects the arms only. The hands and the waist are carried: OpenWAM's rows keep theirs, and a
    correction's rows hold the values ``hold()`` returns, the last commanded ones (the review method's ledger,
    ``oeh_policy.robocasa_gr1_review``). The registry's bare instance (``SPACES``) names the space only.
    """

    name = "robocasa_gr1_eef33"
    # OpenWAM configs/dataloader/robocasa_gr1.yaml (EEF33): left pose, left hand, right pose, right hand, waist.
    unify_action_map = ("0-8", "10-15", "34-42", "44-49", "68-70")
    gripper = False
    arms = unified_arms(unify_action_map, gripper)
    native = ("EEF33 [l_xyz, l_rot6d, l_hand6, r_xyz, r_rot6d, r_hand6, waist3] in the GR1 robot-base frame "
              "(m; joints rad), solved to absolute arm joint targets by the released client's IK")
    carried = "each Fourier hand's six joint targets (rad, 0 open) and the waist's yaw, pitch and roll (rad)"
    frame = "GR1 robot-base frame (pelvis, fixed in the scene: +x forward, +y left, +z up), position m, quaternion wxyz"
    DIM = 33
    POSE = {"left": slice(0, 9), "right": slice(15, 24)}
    HAND = {"left": slice(9, 15), "right": slice(24, 30)}
    WAIST = slice(30, 33)
    CARRIED = ("left_hand", "right_hand", "waist")

    def __init__(self, hold=None):
        self.hold = hold  # () -> {"left_hand": (6,), "right_hand": (6,), "waist": (3,)}: what a correction holds

    @classmethod
    def eef33(cls, value) -> np.ndarray:
        row = np.asarray(value, dtype=np.float32).reshape(-1)
        if row.shape != (cls.DIM,) or not np.isfinite(row).all():
            raise ValueError("GR1 EEF33 must contain 33 finite scalars")
        return row

    @classmethod
    def carried_values(cls, value) -> dict:
        """One EEF33 row's or state's hands and waist."""
        row = cls.eef33(value)
        return {"left_hand": row[cls.HAND["left"]].copy(), "right_hand": row[cls.HAND["right"]].copy(),
                "waist": row[cls.WAIST].copy()}

    @classmethod
    def poses(cls, value) -> dict:
        """One EEF33 row or state in the review layout: each arm's pose, the rot6d orthonormalized as the IK reads it."""
        row = cls.eef33(value)
        result = {}
        for arm in cls.arms:
            pose = row[cls.POSE[arm]].astype(float)
            rotation = rot6d_matrix(pose[3:9], f"{arm} rot6d")
            result[arm + "_ee_pose"] = np.concatenate([pose[:3], matrix_quat(rotation)]).astype(np.float32)
        return result

    def review_state(self, state) -> dict:
        return self.poses(state["eef33"])

    def review_rows(self, rows, state) -> list[dict]:
        return [self.poses(row) for row in rows]

    def native_rows(self, rows, state):
        """Review-layout rows of a correction -> EEF33 rows: the arms' poses, the hands and the waist held."""
        if self.hold is None:
            raise ValueError("GR1 correction rows need the hands' and the waist's last commands")
        held = self.hold()
        result = []
        for row in rows:
            checked = check_review_row(row, self.arms, self.gripper)
            native = np.zeros(self.DIM, dtype=np.float32)
            for arm in self.arms:
                pose = checked[arm + "_ee_pose"].astype(float)
                rotation = quat_matrix(pose[3:])
                native[self.POSE[arm]] = np.concatenate([pose[:3], rotation[:, 0], rotation[:, 1]])
                native[self.HAND[arm]] = np.asarray(held[arm + "_hand"], dtype=np.float32).reshape(6)
            native[self.WAIST] = np.asarray(held["waist"], dtype=np.float32).reshape(3)
            result.append(self.eef33(native))
        return result

    def openings(self, action) -> dict[str, float]:
        return {}  # no gripper slot: the hands are carried, never a review-layout opening


# Registry action-space id -> its ActionSpace class (oeh_eval.benchmarks.Benchmark.action_space; a test holds them
# together). A benchmark whose review port is still being written has no entry yet.
SPACES: dict[str, type[ActionSpace]] = {space.name: space for space in (RoboDojoEE, RoboTwinEE16, LiberoEEF10,
                                                                         RoboCasaGR1EEF33)}


def review_row_keys(arms: tuple[str, ...], gripper: bool = True) -> tuple[str, ...]:
    """The keys of one review-layout row, in order (the pose only for arms without a gripper slot)."""
    return tuple(arm + key for arm in arms for key in (REVIEW_KEYS if gripper else REVIEW_KEYS[:1]))


def check_review_row(row: Mapping, arms: tuple[str, ...], gripper: bool = True) -> dict:
    """A review-layout row with finite float32 poses and openings of the right shapes (bounds are the method's)."""
    if not isinstance(row, Mapping) or set(row) != set(review_row_keys(arms, gripper)):
        raise ValueError(f"a review row has exactly {list(review_row_keys(arms, gripper))}")
    result = {}
    for key in review_row_keys(arms, gripper):
        value = np.asarray(row[key])
        size = 7 if key.endswith("_pose") else 1
        if value.shape != (size,) or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError(f"invalid review row {key}")
        result[key] = value.astype(np.float32)
    return result


def euler_xyz_matrix(angles) -> np.ndarray:
    """Extrinsic xyz Euler angles [roll, pitch, yaw] (rad) -> rotation matrix Rz(yaw) Ry(pitch) Rx(roll): the
    orientation VLABench's ``euler_to_quaternion`` builds from an agent's euler (VLABench cf588fe utils/utils.py:36-49)
    and OpenWAM's dataloader builds before taking rot6d."""
    roll, pitch, yaw = (float(v) for v in np.asarray(angles, dtype=float).reshape(3))
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def matrix_euler_xyz(matrix) -> np.ndarray:
    """Rotation matrix -> extrinsic xyz Euler angles [roll, pitch, yaw] (rad), with the decomposition and gimbal-lock
    branch of OpenWAM's ``rot6d_to_euler_xyz`` (XPolicyLab fa431ec benchmarks/utils/action_conversion.py:676-704), so a
    harness-made row reaches VLABench as OpenWAM's own rows do."""
    r = np.asarray(matrix, dtype=float)
    sy = -r[2, 0]
    if r[0, 0] ** 2 + r[1, 0] ** 2 < 1e-12:  # gimbal lock: roll pinned to 0, folded into yaw
        return np.array([0.0, np.arcsin(np.clip(sy, -1.0, 1.0)), np.arctan2(-r[0, 1], r[1, 1])])
    return np.array([np.arctan2(r[2, 1], r[2, 2]), np.arcsin(np.clip(sy, -1.0, 1.0)), np.arctan2(r[1, 0], r[0, 0])])


def vlabench_row(action) -> np.ndarray:
    """One VLABench native row: 8 finite float64 values [x, y, z, roll, pitch, yaw, finger1, finger2]."""
    row = np.asarray(action, dtype=np.float64).reshape(-1)
    if row.shape != (8,) or not np.isfinite(row).all():
        raise ValueError("a VLABench native EE row has 8 finite values [xyz, roll, pitch, yaw, finger1, finger2]")
    return row.copy()


class VLABenchEE(ActionSpace):
    """VLABench cf588fe: one Franka Panda, one absolute end-effector target per native step.

    Native action: the agent's ``(pos, euler, gripper_state)`` that VLABench's evaluator turns into joint targets
    (evaluation/evaluator/base.py:147-151), as one 8-vector ``[x, y, z, roll, pitch, yaw, finger1, finger2]``: the
    ``end_effector`` site's target position in the world frame (metres), its orientation as extrinsic xyz Euler angles
    (``euler_to_quaternion`` builds Rz Ry Rx, utils/utils.py:36-49) solved by IK (robots/single_arm/base.py:79-91), and
    each finger's slide target, 0.04 m open and 0 closed (assets/robots/franka_emika_panda/panda.xml:11-12). OpenWAM's
    released client makes it from its raw EEF10 (``eef10_to_vlabench_ee``, XPolicyLab fa431ec benchmarks/utils/
    action_conversion.py:734-768): base-frame position plus the robot base, ``rot6d_to_euler_xyz`` and both fingers 0.04
    when the gripper slot is 0.5 or more, else 0. Rows are absolute targets, never deltas.

    State (``oeh_policy.vlabench.VLABenchContract.project_state``): the evaluator's ``ee_state`` (the site's world
    position and wxyz quaternion, then ``get_ee_open_state``'s flag, which upstream inverts: 1 when both fingers are
    nearly closed, robots/single_arm/franka.py:52-60), the two finger joints' measured positions (the joints that flag
    reads) and the robot base position. The review layout reads the site pose and the measured finger gap, opening
    ``(finger1 + finger2) / 0.08`` (1 = open), never the flag; the review method keeps the command in its ledger.
    OpenWAM's reader (``vlabench_obs_to_eef10``, action_conversion.py:707-731) fills its state block with the
    base-frame position and the flag; the unified space holds the world-frame position and the measured opening.

    The one arm is ``left``: OpenWAM's VLABench slot map fills the left end-effector block.
    """

    name = "vlabench_ee"
    unify_action_map = ("0-9",)  # OpenWAM configs/dataloader/vlabench.yaml: the one arm fills the left block
    arms = unified_arms(unify_action_map)
    arm = arms[0]
    native = ("8D absolute EE target [xyz, roll/pitch/yaw extrinsic xyz, finger1, finger2] in VLABench's world frame "
              "(fingers 0.04 m open, 0 closed)")
    frame = "VLABench world frame, position m, quaternion wxyz"
    OPEN_WIDTH = 0.04  # one finger's slide when fully open (panda.xml finger joint range 0-0.04)
    OPEN_THRESHOLD = 0.5  # eef10_to_vlabench_ee: an opening of 0.5 or more opens both fingers

    @classmethod
    def _pose_row(cls, position, rotation, opening) -> dict:
        return {cls.arm + "_ee_pose": np.concatenate([position, matrix_quat(rotation)]).astype(np.float32),
                cls.arm + "_ee_joint_state": np.array([float(np.clip(opening, 0.0, 1.0))], dtype=np.float32)}

    def review_state(self, state) -> dict:
        ee = np.asarray(state["ee_state"], dtype=float).reshape(-1)
        fingers = np.asarray(state["finger_qpos"], dtype=float).reshape(-1)
        if ee.shape != (8,) or fingers.shape != (2,) or not np.isfinite(np.concatenate([ee, fingers])).all():
            raise ValueError("a VLABench state has a finite 8-value ee_state and two finger positions")
        return self._pose_row(ee[:3], quat_matrix(ee[3:7]), float(fingers.sum()) / (2 * self.OPEN_WIDTH))

    def review_rows(self, rows, state) -> list[dict]:
        result = []
        for action in rows:
            row = vlabench_row(action)
            result.append(self._pose_row(row[:3], euler_xyz_matrix(row[3:6]), float(row[6:8].mean()) / self.OPEN_WIDTH))
        return result

    def native_rows(self, rows, state):
        result = []
        for row in rows:
            checked = check_review_row(row, self.arms)
            pose = checked[self.arm + "_ee_pose"].astype(np.float64)
            opening = float(checked[self.arm + "_ee_joint_state"][0])
            fingers = [self.OPEN_WIDTH] * 2 if opening >= self.OPEN_THRESHOLD else [0.0, 0.0]
            result.append(np.concatenate([pose[:3], matrix_euler_xyz(quat_matrix(pose[3:])), fingers]))
        return result

    def openings(self, action) -> dict[str, float]:
        row = vlabench_row(action)
        return {self.arm: float(np.clip(row[6:8].mean() / self.OPEN_WIDTH, 0.0, 1.0))}


SPACES[VLABenchEE.name] = VLABenchEE


# RoboCasa365 (robocasa a07e365 on robosuite 5ce6643; oeh_policy.robocasa365): one Franka Panda on the Omron mobile
# base (PandaOmron). Each gripper and base constant below is the pinned robot's, which the RoboCasa365 contract
# checks against the pinned files at every validation.
RC365_STATE_KEYS = ("base_position", "base_rotation", "end_effector_position_relative",
                    "end_effector_rotation_relative", "gripper_qpos")
# robosuite 5ce6643 controllers/config/robots/default_pandaomron.json, arms.right OSC_POSE: input [-1, 1] maps to
# output_max/min 0.05 m and 0.5 rad, the step's delta.
RC365_OSC_SCALE = (0.05, 0.5)
# models/assets/grippers/panda_gripper.xml finger_joint1 range 0..0.04, finger_joint2 -0.04..0: the gap q0 - q1 of a
# fully open gripper.
RC365_FINGER_GAP_M = 0.08
# What a correction row carries in the native base_motion and control_mode dims: base and torso still, arm mode.
RC365_HOLD = (0.0, 0.0, 0.0, 0.0, -1.0)


def xyzw_matrix(quaternion) -> np.ndarray:
    """Quaternion xyzw (robosuite's order) -> rotation matrix."""
    q = np.asarray(quaternion, dtype=float).reshape(-1)
    if q.shape != (4,):
        raise ValueError("a quaternion has four numbers")
    return quat_matrix([q[3], q[0], q[1], q[2]])


def robocasa365_state(state) -> dict:
    """The projected RoboCasa365 state (``RoboCasa365Contract.project_state``), checked: the base centre's world
    position and xyzw orientation, the grip site's position and the hand's xyzw orientation in the base frame, and
    the two finger joints."""
    if not isinstance(state, Mapping) or set(state) != set(RC365_STATE_KEYS):
        raise ValueError(f"a RoboCasa365 state has exactly {list(RC365_STATE_KEYS)}")
    sizes = {"base_position": 3, "base_rotation": 4, "end_effector_position_relative": 3,
             "end_effector_rotation_relative": 4, "gripper_qpos": 2}
    result = {}
    for key, size in sizes.items():
        value = np.asarray(state[key], dtype=float).reshape(-1)
        if value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError(f"RoboCasa365 state {key} must be {size} finite numbers")
        if size == 4 and np.linalg.norm(value) < 1e-9:
            raise ValueError(f"RoboCasa365 state {key} is not a rotation")
        result[key] = value
    return result


def validate_robocasa365_action(action) -> np.ndarray:
    """One native action12: 12 finite float32, gripper_close (6) and control_mode (11) each exactly -1 or +1, the two
    values the released client's binarization gives and RoboCasa's gym wrapper reads (below 0.5 is -1). Nothing is
    clipped: the controllers clip their own inputs."""
    row = np.asarray(action, dtype=np.float32).reshape(-1)
    if row.shape != (12,) or not np.isfinite(row).all():
        raise ValueError("a RoboCasa365 native action has 12 finite scalars")
    for index, name in ((6, "gripper_close"), (11, "control_mode")):
        if float(row[index]) not in (-1.0, 1.0):
            raise ValueError(f"RoboCasa365 {name} must be -1 or +1 (the released client binarizes it)")
    return row.copy()


class RoboCasa365Action12(ActionSpace):
    """RoboCasa365 a07e365 on robosuite 5ce6643: PandaOmron, one Panda arm on a mobile base, at 20 Hz.

    Native action (RoboCasa's gym wrapper, ``PandaOmronKeyConverter.unmap_action``): ``[dxyz, drotvec,
    gripper_close, forward, side, yaw, torso, control_mode]``. The arm part is robosuite's OSC_POSE delta in the
    mobile base's frame (``input_ref_frame`` base; the controller's origin is the arm's ``right_center`` site, which
    sits on the base's ``center`` site), scaled 0.05 m and 0.5 rad per unit; in arm mode (control_mode -1) the step's
    goal is the measured base-frame grip-site pose plus that delta, in base mode (+1) the previous goal plus it. Either
    way the goal is held in the base frame, so the base carries the arm. gripper_close +1 closes and -1 opens (the
    Panda gripper ramps its fingers over about 10 steps). The base dims are the base's forward and side velocity (m/s)
    and yaw rate (1.5 rad/s per unit) and the torso's height delta (0.05 m per unit).

    State: the base centre's world pose (``robot0_base_pos``, ``robot0_base_quat`` xyzw) and the grip site's position
    and hand body's orientation in that base frame (``robot0_base_to_eef_pos``, ``robot0_base_to_eef_quat``, the
    hand's constant offset from the grip site cancels in every left-multiplied delta), plus the two finger joints.
    ``review_state`` places the gripper in the world through the base pose, ``p_base + R_base p_rel`` and ``R_base
    R_rel``, and reports the measured opening, the finger gap over 0.08 m; the review method keeps the command in its
    ledger.

    ``review_rows`` chains each row's scaled arm delta from the measured base-frame pose (exact in base mode, the goals
    the rows command if every step reached its goal in arm mode) and places the goals in the world at the current base
    pose: the arm's motion relative to the base. The base's own commanded motion is not integrated into them; the
    review method reports it beside the table. ``native_rows`` goes the other way closed loop, as LIBERO's space does:
    each row of a correction is the base-frame delta from the newest achieved pose to its world target, through the
    newest base pose, bounded along its direction to the OSC input range, and it holds the base and torso still in arm
    mode (``RC365_HOLD``, ``carried``): a correction moves the arm only.

    The one arm is ``left``, OpenWAM-α's single-arm block; OpenWAM's RoboCasa365 map puts base_motion and control_mode
    in reserved slots 68-72, which the unified space leaves masked.
    """

    name = "robocasa365_action12"
    unify_action_map = ("0-9", "68-72")  # OpenWAM configs/dataloader/robocasa365.yaml unify_action_map
    arms = unified_arms(unify_action_map)
    arm = arms[0]
    native = ("12D [dxyz, drotvec (robosuite OSC_POSE delta in the mobile base frame), gripper_close (+1 closes), "
              "base forward, side, yaw, torso, control_mode (+1 base mode)]")
    carried = ("slots 68-72: base_motion (forward and side velocity, yaw rate, torso height delta) and control_mode, "
               "native values; a correction row sets 0, 0, 0, 0 and -1 (base and torso still, arm mode)")
    frame = "RoboCasa365 world frame (robosuite), position m, quaternion wxyz"

    @staticmethod
    def base(state) -> tuple[np.ndarray, np.ndarray]:
        """The base centre's world position and rotation matrix."""
        checked = robocasa365_state(state)
        return checked["base_position"], xyzw_matrix(checked["base_rotation"])

    @staticmethod
    def arm_in_base(state) -> tuple[np.ndarray, np.ndarray]:
        """The grip site's position and the hand's rotation matrix in the base frame."""
        checked = robocasa365_state(state)
        return (checked["end_effector_position_relative"],
                xyzw_matrix(checked["end_effector_rotation_relative"]))

    @staticmethod
    def measured_opening(state) -> float:
        qpos = robocasa365_state(state)["gripper_qpos"]
        return float(np.clip((qpos[0] - qpos[1]) / RC365_FINGER_GAP_M, 0.0, 1.0))

    @classmethod
    def _pose_row(cls, position, rotation, opening) -> dict:
        return {cls.arm + "_ee_pose": np.concatenate([position, matrix_quat(rotation)]).astype(np.float32),
                cls.arm + "_ee_joint_state": np.array([float(np.clip(opening, 0.0, 1.0))], dtype=np.float32)}

    def world_pose(self, state, position, rotation) -> tuple[np.ndarray, np.ndarray]:
        """A base-frame pose placed in the world at ``state``'s base pose."""
        base_position, base_rotation = self.base(state)
        return base_position + base_rotation @ np.asarray(position, dtype=float), base_rotation @ rotation

    def review_state(self, state) -> dict:
        position, rotation = self.world_pose(state, *self.arm_in_base(state))
        return self._pose_row(position, rotation, self.measured_opening(state))

    def review_rows(self, rows, state) -> list[dict]:
        position, rotation = self.arm_in_base(state)
        translation, turn = RC365_OSC_SCALE
        result = []
        for row in rows:
            native = validate_robocasa365_action(row).astype(float)
            position = position + translation * np.clip(native[0:3], -1.0, 1.0)
            rotation = rotvec_matrix(turn * np.clip(native[3:6], -1.0, 1.0)) @ rotation
            world_position, world_rotation = self.world_pose(state, position, rotation)
            result.append(self._pose_row(world_position, world_rotation, 0.0 if native[6] > 0 else 1.0))
        return result

    def toward(self, row: Mapping, state) -> np.ndarray:
        """The native row from the achieved pose in ``state`` toward one review-layout world target: the base-frame
        delta through the newest base pose, each part bounded to the OSC input range along its own direction, the
        target's opening as the gripper command, and the base held (``RC365_HOLD``)."""
        base_position, base_rotation = self.base(state)
        position, rotation = self.arm_in_base(state)
        pose = np.asarray(row[self.arm + "_ee_pose"], dtype=float)
        goal_position = base_rotation.T @ (pose[:3] - base_position)
        goal_rotation = base_rotation.T @ quat_matrix(pose[3:])
        translation, turn = RC365_OSC_SCALE
        command = np.concatenate([(goal_position - position) / translation,
                                  matrix_rotvec(goal_rotation @ rotation.T) / turn])
        for part in (slice(0, 3), slice(3, 6)):
            reach = float(np.max(np.abs(command[part])))
            if reach > 1:
                command[part] = command[part] / reach
        opening = float(np.clip(np.asarray(row[self.arm + "_ee_joint_state"], dtype=float)[0], 0.0, 1.0))
        grip = 1.0 if opening < 0.5 else -1.0
        return validate_robocasa365_action(np.concatenate([np.clip(command, -1.0, 1.0), [grip], RC365_HOLD]))

    def native_rows(self, rows, state):
        targets = [check_review_row(row, self.arms) for row in rows]
        if not targets:
            raise ValueError("a RoboCasa365 correction has at least one row")
        follow = Toward(self, targets)
        return FollowingRows([follow(state)], len(targets), follow)

    def openings(self, action) -> dict[str, float]:
        return {self.arm: 0.0 if validate_robocasa365_action(action)[6] > 0 else 1.0}


SPACES[RoboCasa365Action12.name] = RoboCasa365Action12


# EBench (appended after the registry above, which it joins at the end of this block).
from .ebench_boundary import HOLD_BASE, fingers as ebench_fingers, native_action as ebench_action  # noqa: E402
from .ebench_boundary import opening as ebench_opening  # noqa: E402


class EBenchEE(ActionSpace):
    """EBench (GenManip 1958e37): the lift2 robot's two R5a arms, each driven by an absolute EE pose target that
    GenManip solves with cuRobo IK on the server (``oeh_policy.ebench_boundary``).

    Native action: one dict per step, per arm ``(xyz, quaternion wxyz, [finger, finger])`` with the fingers in metres
    (0 closed, 0.044 open), plus ``base_motion`` ``[dx_m, dy_m, dyaw_deg]``, a relative base displacement. OpenWAM's
    released bridge produces it from the raw 23-D action (``raw23_to_ebench_action``); OpenWAM's slot map
    (configs/dataloader/ebench.yaml) puts the left block in 0-9, the right in 34-43 and the base displacement in the
    reserved slots 68-70, which this space carries natively (``carried``).

    State: the projected observation (``oeh_policy.ebench``) holds each arm's measured pose (forward kinematics,
    ``state.ee_pose``) and the four measured finger positions (``state.gripper``). It has no gripper command, so
    ``review_state`` reports the measured opening, mean finger / 0.044, and the review method keeps the commanded one
    in its ledger, as on LIBERO.

    The frame: GenManip's poses, targets and states alike, are in each arm's own base frame (the cuRobo ``base_link``
    of that arm's planner, mounted on the lift; ``dualarm_manip.py`` ``_fk_single``), which moves with the mobile
    base. GenManip publishes no pose of the arm bases in its world, so OEH cannot place these poses in a world frame:
    on EBench the review layout and the unified space hold positions in metres in each arm's own base frame, the
    frame OpenWAM's EBench reader also uses (openwam/dataloader/ebench.py). The rest of the unified meaning holds: the
    rot6d is of the true wxyz orientation (OpenWAM's EBench reader reorders the quaternion before building it, unlike
    its RoboTwin reader), the gripper slot is 2 * opening - 1 and rows are absolute targets.

    A correction's rows (``native_rows``) hold the base: ``base_motion`` ``[0, 0, 0]`` keeps the base's target pose.
    While they run, both arm frames therefore stay where they are, and an absolute target means one place.
    """

    name = "ebench_ee"
    unify_action_map = ("0-9", "34-43", "68-70")  # OpenWAM configs/dataloader/ebench.yaml (action_mode eef)
    arms = unified_arms(unify_action_map)
    native = ("GenManip ee_pose dict: per arm (xyz m, quaternion wxyz, [finger, finger] m, 0.044 open) in that arm's "
              "base frame, and base_motion [dx m, dy m, dyaw deg] relative")
    carried = ("base_motion [dx m, dy m, dyaw deg], the per-step base displacement GenManip clips to 0.015 m / 1 deg "
               "(slots 68-70); a correction holds the base, [0, 0, 0]")
    frame = "each arm's own base frame (cuRobo base_link of its R5a planner); position m, quaternion wxyz"

    def review_state(self, state) -> dict:
        grip = np.asarray(state["gripper"], dtype=float).reshape(-1)
        if grip.shape != (4,) or not np.isfinite(grip).all():
            raise ValueError("EBench state.gripper holds four finite finger positions [L, L, R, R]")
        result = {}
        for index, arm in enumerate(self.arms):
            pose = np.asarray(state[arm + "_ee_pose"], dtype=float).reshape(-1)
            if pose.shape != (7,) or not np.isfinite(pose).all():
                raise ValueError(f"invalid EBench {arm} EE pose")
            result[arm + "_ee_pose"] = pose.astype(np.float32)
            result[arm + "_ee_joint_state"] = np.array([ebench_opening(grip[2 * index:2 * index + 2])], np.float32)
        return result

    def review_rows(self, rows, state) -> list[dict]:
        result = []
        for row in rows:
            native = ebench_action(row)
            item = {}
            for arm, (position, quaternion, pair) in zip(self.arms, native["action"]):
                item[arm + "_ee_pose"] = np.asarray(position + quaternion, dtype=np.float32)
                item[arm + "_ee_joint_state"] = np.array([ebench_opening(pair)], np.float32)
            result.append(item)
        return result

    def native_rows(self, rows, state):
        result = []
        for row in rows:
            checked = check_review_row(row, self.arms)
            arms = []
            for arm in self.arms:
                pose = [float(v) for v in checked[arm + "_ee_pose"]]
                arms.append((pose[:3], pose[3:], ebench_fingers(checked[arm + "_ee_joint_state"][0])))
            result.append(ebench_action({"action": arms, "control_type": "ee_pose", "is_rel": False,
                                         "base_motion": list(HOLD_BASE), "base_is_rel": True}))
        return result

    def openings(self, action) -> dict[str, float]:
        native = ebench_action(action)
        return {arm: ebench_opening(pair) for arm, (_, _, pair) in zip(self.arms, native["action"])}


SPACES[EBenchEE.name] = EBenchEE
