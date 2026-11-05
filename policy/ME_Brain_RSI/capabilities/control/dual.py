"""Joint-space motion layer for the ARX X5 dual arm (public URDF kinematics only).

Everything here works from measured proprioception plus the public kinematic
chain; no simulator or scene state is accessed. Paths are lists of
``(joints6, grip)`` per control step for one arm and can be interleaved by the
scheduler so both arms act in the same control steps.
"""
import math
import numpy as np
from scipy.spatial.transform import Rotation

from .motion import TABLE_Z, TCP_OFFSET, ARMS, quat_matrix
from .kinematics import JOINT_LOWER, JOINT_UPPER, local_link6_matrix

# Tool frame: X points along the fingers (outward), Y is the closing axis,
# Z points to the wrist camera side.
R_TOOL_DOWN = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
GRIP_OPEN = 1.0
# Pad gap in metres for a normalized gripper reading (public URDF: 2 x 44 mm).
GAP_PER_UNIT = 0.088
# Finger tip extends below the TCP point; pads contact between these offsets.
TIP_BELOW_TCP = 0.0126
PAD_BAND = (-0.008, 0.0126)
MAX_JOINT_STEP = 0.07
# Soft joint ranges keeping the arm in its natural forward, elbow-up posture family.
# (Public URDF limits are +-10 rad on joints 1-5, which admits mirrored/wrapped branches.)
SOFT_LOWER = np.array([-2.2, -0.3, -0.3, -3.1, -2.0, -3.14])
SOFT_UPPER = np.array([2.2, 3.3, 3.3, 1.0, 2.0, 3.14])


def rz(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def ry(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def rx(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def down_rotation(yaw, pitch=0.0, pitch_axis=None):
    """Tool pointing down with closing axis along Rz(yaw)·Y.

    ``pitch`` tilts the tool about ``pitch_axis`` (world unit vector, default
    the closing axis) so the finger tips lean while the closing axis stays put.
    """
    rotation = rz(yaw) @ R_TOOL_DOWN
    if pitch:
        axis = rotation[:, 1] if pitch_axis is None else np.asarray(pitch_axis, float)
        rotation = Rotation.from_rotvec(axis / np.linalg.norm(axis) * pitch).as_matrix() @ rotation
    return rotation


def matrix_to_quat(rotation):
    q = Rotation.from_matrix(rotation).as_quat()
    return np.r_[q[3], q[:3]]


def pose_matrix(position, rotation):
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = position
    return matrix


class ArmModel:
    """Fast FK and damped-least-squares IK for one arm, with public-geometry validity checks."""

    def __init__(self, kinematics, arm, table_z=TABLE_Z):
        from .robot_collision import PublicSphereCheck
        self.kinematics = kinematics
        self.arm = arm
        self.base = kinematics.base_transforms[arm]
        self.table_z = table_z
        self.checker = PublicSphereCheck()
        config = self.checker.config
        # Flatten all collision spheres once; pairwise self-collision mask excludes adjacent links.
        self._sphere_links, centres, radii = [], [], []
        for link, items in config["collision_spheres"].items():
            for item in items:
                self._sphere_links.append(link)
                centres.append(item["center"])
                radii.append(item["radius"])
        self._centres = np.asarray(centres, float)
        self._radii = np.asarray(radii, float)
        links = sorted(set(self._sphere_links))
        self._link_index = {link: i for i, link in enumerate(links)}
        self._sphere_link_ids = np.asarray([self._link_index[l] for l in self._sphere_links])
        ignore = config["self_collision_ignore"]
        buffer = config["self_collision_buffer"]
        n = len(self._sphere_links)
        mask = np.ones((n, n), bool)
        buffers = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                a, b = self._sphere_links[i], self._sphere_links[j]
                if a == b or b in ignore.get(a, []) or a in ignore.get(b, []) or j <= i:
                    mask[i, j] = False
                buffers[i, j] = buffer.get(a, 0.0) + buffer.get(b, 0.0)
        self._pair_mask = mask
        self._pair_buffers = buffers
        self._moving_mask = np.asarray([l not in ("base_link", "link1") for l in self._sphere_links])

    def sphere_positions(self, joints):
        frames = self.checker.frames(self.base, joints)
        world = np.empty_like(self._centres)
        for link, index in self._link_index.items():
            sel = self._sphere_link_ids == index
            matrix = frames[link]
            world[sel] = self._centres[sel] @ matrix[:3, :3].T + matrix[:3, 3]
        return world

    def valid(self, joints, self_gap_min=0.002, table_margin=0.0):
        """Reject self-colliding, below-table or out-of-posture arm configurations."""
        joints = np.asarray(joints, float)
        if np.any(joints < SOFT_LOWER - 1e-6) or np.any(joints > SOFT_UPPER + 1e-6):
            return False
        world = self.sphere_positions(joints)
        lowest = world[self._moving_mask, 2] - self._radii[self._moving_mask]
        if lowest.min() < self.table_z + table_margin:
            return False
        delta = world[:, None, :] - world[None, :, :]
        distance = np.sqrt(np.sum(delta * delta, axis=-1))
        clearance = distance - self._radii[:, None] - self._radii[None, :] - self._pair_buffers
        return float(clearance[self._pair_mask].min()) > self_gap_min

    def tcp_matrix(self, joints):
        matrix = self.base @ local_link6_matrix(joints)
        matrix = matrix.copy()
        matrix[:3, 3] += matrix[:3, :3] @ TCP_OFFSET
        return matrix

    def link6_matrix(self, joints):
        return self.base @ local_link6_matrix(joints)

    def jacobian(self, joints):
        """Geometric Jacobian of the TCP (world frame) from the public joint chain."""
        from .kinematics import JOINT_ORIGINS, JOINT_ORIGIN_ROLLS, JOINT_AXES, axis_matrix
        rotation = self.base[:3, :3].copy()
        position = self.base[:3, 3].copy()
        axes, origins = [], []
        for index, value in enumerate(np.asarray(joints, float)):
            position = position + rotation @ JOINT_ORIGINS[index]
            rotation = rotation @ axis_matrix(0, float(JOINT_ORIGIN_ROLLS[index]))
            unit = np.zeros(3)
            unit[JOINT_AXES[index]] = 1.0
            axes.append(rotation @ unit)
            origins.append(position.copy())
            rotation = rotation @ axis_matrix(JOINT_AXES[index], float(value))
        tcp = position + rotation @ TCP_OFFSET
        jac = np.zeros((6, 6))
        for index in range(6):
            jac[:3, index] = np.cross(axes[index], tcp - origins[index])
            jac[3:, index] = axes[index]
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = tcp
        return jac, matrix

    @staticmethod
    def _rotvec(matrix):
        """Rotation vector of a rotation matrix (small-angle safe)."""
        cos = np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0)
        angle = math.acos(cos)
        axis = np.array([matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]])
        norm = np.linalg.norm(axis)
        if norm < 1e-9:
            if angle < 1e-6:
                return np.zeros(3)
            return Rotation.from_matrix(matrix).as_rotvec()
        return axis / norm * angle

    def ik(self, position, rotation, seed, iterations=60, pos_tol=5e-4, rot_tol=4e-3, damping=0.02, max_step=0.6):
        q = np.clip(np.asarray(seed, float).copy(), SOFT_LOWER + 1e-6, SOFT_UPPER - 1e-6)
        target = np.asarray(position, float)
        rotation = np.asarray(rotation, float)
        for _ in range(iterations):
            jac, current = self.jacobian(q)
            dp = target - current[:3, 3]
            dr = self._rotvec(rotation @ current[:3, :3].T)
            if np.linalg.norm(dp) < pos_tol and np.linalg.norm(dr) < rot_tol:
                return self._nearest_periodic(q, seed)
            error = np.r_[dp, dr]
            step = np.linalg.solve(jac.T @ jac + damping ** 2 * np.eye(6), jac.T @ error)
            norm = np.linalg.norm(step)
            if norm > max_step:
                step *= max_step / norm
            q = np.clip(q + step, SOFT_LOWER + 1e-6, SOFT_UPPER - 1e-6)
        jac, current = self.jacobian(q)
        if (np.linalg.norm(target - current[:3, 3]) < pos_tol and
                np.linalg.norm(self._rotvec(rotation @ current[:3, :3].T)) < rot_tol):
            return self._nearest_periodic(q, seed)
        return None

    @staticmethod
    def _nearest_periodic(q, seed):
        """Shift revolute joints by multiples of 2*pi towards the seed within limits."""
        q = np.asarray(q, float).copy()
        seed = np.asarray(seed, float)
        for i in range(6):
            best = q[i]
            for k in (-2, -1, 1, 2):
                candidate = q[i] + 2 * math.pi * k
                if SOFT_LOWER[i] <= candidate <= SOFT_UPPER[i] and abs(candidate - seed[i]) < abs(best - seed[i]):
                    best = candidate
            q[i] = best
        return q

    def solve(self, position, rotation, seed, extra_seeds=True, check=True, prefer_small_roll=False):
        """Multi-start IK preferring the valid branch nearest the seed (memoised).

        With ``prefer_small_roll`` all seeds are tried and the valid solution with the
        smallest wrist roll magnitude is returned (used before large reorientations).
        """
        seed = np.asarray(seed, float)
        key = (tuple(np.round(np.asarray(position, float), 4)), tuple(np.round(np.asarray(rotation, float).ravel(), 3)),
               tuple(np.round(seed, 2)), bool(extra_seeds), bool(check), bool(prefer_small_roll))
        cache = self.__dict__.setdefault("_solve_cache", {})
        if key in cache:
            hit = cache[key]
            return None if hit is None else hit.copy()
        if len(cache) > 4000:
            cache.clear()
        seeds = [seed]
        if extra_seeds:
            local = self.base[:3, :3].T @ (np.asarray(position, float) - self.base[:3, 3])
            heading = math.atan2(local[1], local[0])
            for shoulder, elbow in ((1.5, 0.9), (2.4, 2.5), (1.7, 1.6), (2.0, 1.0), (1.0, 0.3)):
                for roll in (seed[5], seed[5] + 1.6, seed[5] - 1.6, seed[5] + 3.0, seed[5] - 3.0):
                    candidate = seed.copy()
                    candidate[:4] = [heading, shoulder, elbow, -shoulder + elbow]
                    candidate[4] = 0.0
                    candidate[5] = roll
                    seeds.append(candidate)
        result = None
        found = []
        for start in seeds:
            q = self.ik(position, rotation, start)
            if q is None:
                continue
            if check and not self.valid(q):
                continue
            if not prefer_small_roll:
                result = q
                break
            found.append(q)
        if prefer_small_roll and found:
            result = min(found, key=lambda q: abs(q[5]) + 0.05 * np.sum(np.abs(q - seed)))
        cache[key] = None if result is None else result.copy()
        return result


def joint_path(start, goal, max_step=MAX_JOINT_STEP, min_steps=1):
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    count = max(min_steps, int(math.ceil(np.max(np.abs(goal - start)) / max_step)))
    return [start + (goal - start) * (i + 1) / count for i in range(count)]


def smooth_joint_path(start, goal, max_step=MAX_JOINT_STEP, min_steps=2, roll_step=None):
    """Cosine-eased joint interpolation (keeps peak velocity below max_step).

    ``roll_step`` lets the light wrist roll joint move faster than the arm joints.
    """
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    delta = np.abs(goal - start)
    span = np.max(delta[:5]) / max_step if len(delta) > 5 else np.max(delta) / max_step
    if len(delta) > 5:
        span = max(span, delta[5] / (max_step if roll_step is None else roll_step))
    count = max(min_steps, int(math.ceil(span * (math.pi / 2))))
    out = []
    for i in range(count):
        s = (i + 1) / count
        eased = 0.5 - 0.5 * math.cos(math.pi * s)
        out.append(start + (goal - start) * eased)
    return out


class CartesianPathError(RuntimeError):
    pass


def cartesian_path(model, start_joints, goal_position, goal_rotation, step_m=0.02, step_rad=0.12,
                   max_joint_step=MAX_JOINT_STEP, start_pose=None, flip_threshold=0.5, rescue=True):
    """Straight-line TCP motion solved waypoint by waypoint with continuity checks.

    Consecutive waypoint solutions differing by more than ``flip_threshold`` are treated
    as IK branch changes (the expanded path is interpolated at ``max_joint_step``).
    """
    start_joints = np.asarray(start_joints, float)
    current = model.tcp_matrix(start_joints) if start_pose is None else start_pose
    p0, r0 = current[:3, 3], current[:3, :3]
    p1, r1 = np.asarray(goal_position, float), np.asarray(goal_rotation, float)
    delta_rot = Rotation.from_matrix(r1 @ r0.T)
    angle = np.linalg.norm(delta_rot.as_rotvec())
    count = max(1, int(math.ceil(np.linalg.norm(p1 - p0) / step_m)), int(math.ceil(angle / step_rad)))
    # Cosine ease-in/out keeps the peak speed near step_m per step while removing the
    # acceleration jumps that make a pinched object swing.
    if count >= 4:
        count = int(math.ceil(count * math.pi / 2.0))
    threshold = max(max_joint_step * 3, flip_threshold)
    joints = [start_joints]
    for i in range(count):
        linear = (i + 1) / count
        s = 0.5 - 0.5 * math.cos(math.pi * linear) if count >= 4 else linear
        position = p0 + (p1 - p0) * s
        rotation = Rotation.from_rotvec(delta_rot.as_rotvec() * s).as_matrix() @ r0
        q = model.ik(position, rotation, joints[-1])
        if q is not None and np.max(np.abs(q - joints[-1])) > threshold:
            # Stay on the current branch: conservative damped steps from the previous joints.
            cautious = model.ik(position, rotation, joints[-1], iterations=150, damping=0.08, max_step=0.08)
            if cautious is not None and np.max(np.abs(cautious - joints[-1])) < np.max(np.abs(q - joints[-1])):
                q = cautious
        if q is not None and not model.valid(q):
            q = None
        if q is None:
            # Near-singular starts: a heavily damped, small-step solve from the same branch.
            careful = model.ik(position, rotation, joints[-1], iterations=240, damping=0.12, max_step=0.05)
            if careful is not None and model.valid(careful) and np.max(np.abs(careful - joints[-1])) <= threshold:
                q = careful
        if q is None and rescue:
            q = model.solve(position, rotation, joints[-1], extra_seeds=True)
        failure = None
        if q is None:
            failure = f"IK failed at fraction {s:.2f} for {model.arm} target {position.round(3).tolist()}"
        elif np.max(np.abs(q - joints[-1])) > threshold:
            failure = f"Joint discontinuity {np.max(np.abs(q - joints[-1])):.3f} rad for {model.arm} at fraction {s:.2f}"
        if failure is not None:
            error = CartesianPathError(failure)
            error.partial = _expand(joints, max_joint_step)
            error.fraction = i / count
            raise error
        joints.append(q)
    return _expand(joints, max_joint_step)


def _expand(joints, max_joint_step):
    out = []
    for previous, q in zip(joints[:-1], joints[1:]):
        out.extend(joint_path(previous, q, max_joint_step))
    return out


class Scheduler:
    """Interleave per-arm step generators into whole-robot actions.

    Each arm generator yields ``(joints6, grip)`` per control step or the
    string ``'hold'``. Generators may read ``ctx.state`` whenever resumed.
    """

    def __init__(self, ctx):
        self.ctx = ctx

    def run(self, left=None, right=None):
        ctx = self.ctx
        generators = {"left": left, "right": right}
        active = {arm: gen is not None for arm, gen in generators.items()}
        while any(active.values()):
            action = {}
            for arm in ARMS:
                item = "hold"
                if active[arm]:
                    try:
                        item = next(generators[arm])
                    except StopIteration:
                        active[arm] = False
                        item = "hold"
                if item != "hold":
                    joints, grip = item
                    ctx.joint_targets[arm] = np.asarray(joints, float).copy()
                    if grip is not None:
                        ctx.gripper_targets[arm] = float(np.clip(grip, 0, 1))
                action[f"{arm}_arm_joint_state"] = ctx.joint_targets[arm].copy()
                action[f"{arm}_ee_joint_state"] = np.array([ctx.gripper_targets[arm]])
            yield action


def hold_steps(count):
    for _ in range(max(0, int(count))):
        yield "hold"


def grip_steps(ctx, arm, value, count=8):
    """Ramp the gripper command while keeping the current joint targets."""
    start = ctx.gripper_targets[arm]
    for i in range(max(1, int(count))):
        s = (i + 1) / max(1, int(count))
        yield ctx.joint_targets[arm].copy(), (1 - s) * start + s * value


def measured_gap(ctx, arm):
    return GAP_PER_UNIT * float(ctx.state[f"{arm}_ee_joint_state"][0])


def steps_from_joint_list(joint_list, grip=None):
    for q in joint_list:
        yield q, grip
