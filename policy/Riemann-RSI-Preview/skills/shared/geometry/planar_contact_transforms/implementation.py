"""Canonical stateless helpers; caller-owned state and clocks remain local."""
from __future__ import annotations
import math as math
import numpy as np


def U27_as_numpy(value):
    """Convert torch/numpy/list values to a detached float numpy array."""
    if value is None:
        return np.asarray([], dtype=np.float32)
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    elif hasattr(value, 'cpu') and hasattr(value, 'numpy'):
        value = value.cpu().numpy()
    return np.asarray(value, dtype=np.float32)

def U27_mat_to_quat_wxyz(matrix):
    """Stable 3x3 rotation matrix to (w,x,y,z)."""
    m = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = np.trace(m)
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = np.array([s / 4, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = np.array([(m[2, 1] - m[1, 2]) / s, s / 4, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, s / 4, (m[1, 2] + m[2, 1]) / s])
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, s / 4])
    q /= max(np.linalg.norm(q), 1e-12)
    if q[0] < 0:
        q = -q
    return q.astype(np.float32)

def U27_quat_mul_wxyz(q1, q2):
    (w1, x1, y1, z1) = U27_as_numpy(list(q1)).reshape(4)
    (w2, x2, y2, z2) = U27_as_numpy(list(q2)).reshape(4)
    return np.asarray([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], dtype=np.float32)

def U27_quaternion_angle_deg(q_a, q_b):
    """Shortest scalar-first quaternion angle in degrees."""
    a = U27_as_numpy(list(q_a)).reshape(4).astype(np.float64)
    b = U27_as_numpy(list(q_b)).reshape(4).astype(np.float64)
    a /= max(np.linalg.norm(a), 1e-12)
    b /= max(np.linalg.norm(b), 1e-12)
    return float(np.degrees(2.0 * np.arccos(np.clip(abs(float(np.dot(a, b))), -1.0, 1.0))))

def U74_array(value, shape, name):
    try:
        result = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be finite with shape {shape}') from exc
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f'{name} must be finite with shape {shape}')
    return result

def U74_scalar(value, name, *, positive=False, nonnegative=False):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be finite') from exc
    if not math.isfinite(result) or (positive and result <= 0) or (nonnegative and result < 0):
        raise ValueError(f'{name} must be finite' + (' and positive' if positive else ' and nonnegative' if nonnegative else ''))
    return result

def U74_normalize_quat_wxyz(quaternion):
    """Normalize a scalar-first quaternion; reject zero/NaN/Inf, never repair.

    The first numerically significant component is positive, so q and -q
    share a deterministic representative, including exact half turns.
    Scaling before the norm avoids overflow for finite non-unit inputs.
    """
    quaternion = U74_array(quaternion, (4,), 'quaternion')
    scale = float(np.max(np.abs(quaternion)))
    if scale == 0:
        raise ValueError('zero quaternion has no rotation')
    quaternion /= scale
    quaternion /= np.linalg.norm(quaternion)
    first = np.flatnonzero(np.abs(quaternion) > 1e-15)[0]
    if quaternion[first] < 0:
        quaternion *= -1
    return quaternion

def U74_validate_pose(pose):
    """Copy/normalize a finite 7D XYZ/wxyz pose; reject malformed rotations."""
    result = U74_array(pose, (7,), 'pose')
    result[3:] = U74_normalize_quat_wxyz(result[3:])
    return result

def U74_quat_mul_wxyz(left, right):
    """Compose two rotation quaternions, returning a unit quaternion."""
    (w, x, y, z) = U74_normalize_quat_wxyz(left)
    (a, b, c, d) = U74_normalize_quat_wxyz(right)
    return U74_normalize_quat_wxyz([w * a - x * b - y * c - z * d, w * b + x * a + y * d - z * c, w * c - x * d + y * a + z * b, w * d + x * c - y * b + z * a])

def U74_quat_inverse_wxyz(quaternion):
    result = U74_normalize_quat_wxyz(quaternion)
    result[1:] *= -1
    return U74_normalize_quat_wxyz(result)

def U74_quat_to_mat_wxyz(quaternion):
    (w, x, y, z) = U74_normalize_quat_wxyz(quaternion)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], dtype=np.float64)

def U74_quat_axis_angle_wxyz(axis, angle):
    """Active right-handed rotation by ``angle`` radians around ``axis``."""
    axis = U74_array(axis, (3,), 'rotation axis')
    scale = float(np.max(np.abs(axis)))
    if scale == 0:
        raise ValueError('rotation axis must be nonzero')
    axis /= scale
    axis /= np.linalg.norm(axis)
    half = 0.5 * U74_scalar(angle, 'angle')
    return U74_normalize_quat_wxyz(np.r_[math.cos(half), axis * math.sin(half)])

def U74_yaw_quat(angle):
    return U74_quat_axis_angle_wxyz([0, 0, 1], angle)

def U74_wrap_yaw(angle):
    """Canonical radians in [-pi, pi); an exact pi tie turns clockwise."""
    angle = U74_scalar(angle, 'yaw')
    return (angle + math.pi) % (2 * math.pi) - math.pi

def U74_yaw_of_quaternion(quaternion):
    """Heading of body-local X projected on world XY (ZYX yaw).

    A vertical local X has no planar heading. Fail explicitly instead of
    inventing yaw zero; full quaternion errors remain valid in that case.
    """
    rotation = U74_quat_to_mat_wxyz(quaternion)
    if np.linalg.norm(rotation[:2, 0]) < 1e-10:
        raise ValueError('yaw is undefined when body-local X is vertical')
    return U74_wrap_yaw(math.atan2(float(rotation[1, 0]), float(rotation[0, 0])))

def U74_rotate_vec_by_quat(vector, quaternion):
    return U74_quat_to_mat_wxyz(quaternion) @ U74_array(vector, (3,), 'vector')

def U74_compose_pose(parent, child):
    """SE(3) product: a parent/world pose times a child/parent pose."""
    (parent, child) = (U74_validate_pose(parent), U74_validate_pose(child))
    return np.r_[parent[:3] + U74_rotate_vec_by_quat(child[:3], parent[3:]), U74_quat_mul_wxyz(parent[3:], child[3:])]

def U74_inverse_pose(pose):
    pose = U74_validate_pose(pose)
    inverse = U74_quat_inverse_wxyz(pose[3:])
    return np.r_[U74_rotate_vec_by_quat(-pose[:3], inverse), inverse]

def U74_capture_contact_transform(object_pose, tool_pose):
    """Record object-in-tool from the *measured* live world poses.

    ``relative = inv(T_world_tool) @ T_world_object``. Capture after contact
    settles, rather than using the commanded tool pose or a guessed offset.
    """
    return U74_compose_pose(U74_inverse_pose(tool_pose), object_pose)

def U74_tool_pose_for_object(desired_object_pose, object_in_tool):
    """Solve the rigid-contact target: T_world_object_goal @ inv(relative)."""
    return U74_compose_pose(desired_object_pose, U74_inverse_pose(object_in_tool))

def U74_planar_step(current_pose, goal_pose, initial_pose, *, max_translation=0.02, max_yaw=math.radians(8)):
    """One bounded XY/yaw command on the initial object's planar manifold.

    XY displacement is bounded in Euclidean norm; yaw uses the shortest
    wrapped world-Z direction. Z is always initial Z, even if the observed
    object or requested goal has lifted. The initial full roll/pitch is
    retained by ``q_next = q_world_z(yaw_next-yaw_initial) * q_initial``.
    This deliberately cannot repair a tilted goal: use full-pose errors to
    detect a goal outside the planar manifold and live lift/slip guards to
    decide whether another physical push is appropriate.
    """
    (current, goal, initial) = map(U74_validate_pose, (current_pose, goal_pose, initial_pose))
    distance = U74_scalar(max_translation, 'max_translation', positive=True)
    angle = U74_scalar(max_yaw, 'max_yaw', positive=True)
    if angle > math.pi:
        raise ValueError('max_yaw must not exceed pi radians')
    delta = goal[:2] - current[:2]
    norm = float(np.linalg.norm(delta))
    if not math.isfinite(norm):
        raise ValueError('XY displacement overflow')
    next_xy = current[:2] + delta * min(1.0, distance / norm) if norm else current[:2].copy()
    current_yaw = U74_yaw_of_quaternion(current[3:])
    goal_yaw = U74_yaw_of_quaternion(goal[3:])
    initial_yaw = U74_yaw_of_quaternion(initial[3:])
    step = float(np.clip(U74_wrap_yaw(goal_yaw - current_yaw), -angle, angle))
    quaternion = U74_quat_mul_wxyz(U74_yaw_quat(U74_wrap_yaw(current_yaw + step - initial_yaw)), initial[3:])
    return np.r_[next_xy, initial[2], quaternion]

def U74_closest_joint_solution(solutions, current, limits, success=None):
    """Select a nearby successful revolute-joint IK branch within limits.

    ``solutions`` is [N,D], ``current`` is [D], ``limits`` is [D,2], and
    optional ``success`` is [N] (bool or 0/1). For every candidate and joint,
    select the 2*pi-equivalent angle nearest current among those inside the
    joint's limits. Rank candidates by maximum absolute joint displacement,
    then Euclidean displacement; preserve input order for exact ties.
    Failed/nonfinite/unrepresentable rows are skipped; no legal row -> None.

    A 1e-10-radian tolerance only handles floating-point residue at limits;
    the returned values are clamped back to the exact bounds. This does not
    infer collision safety, validate contact, or impose a motion step cap.
    All columns must be revolute joints; do not include a prismatic gripper.
    """
    try:
        candidates = np.array(solutions, dtype=np.float64, copy=True)
        position = np.array(current, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('solutions/current must be numeric [N,D]/[D] arrays') from exc
    if candidates.ndim != 2 or position.ndim != 1 or position.size == 0 or (candidates.shape[1] != position.size):
        raise ValueError('solutions/current must have shapes [N,D]/[D] with D > 0')
    if not np.all(np.isfinite(position)):
        raise ValueError('current joints must be finite')
    bounds = U74_array(limits, (position.size, 2), 'joint limits')
    if np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError('joint lower limits must not exceed upper limits')
    if success is None:
        successful = np.ones(candidates.shape[0], dtype=bool)
    else:
        try:
            flags = np.asarray(success, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('success must be a finite bool/0-1 mask with shape [N]') from exc
        if flags.shape != (candidates.shape[0],) or not np.all(np.isfinite(flags) & ((flags == 0) | (flags == 1))):
            raise ValueError('success must be a finite bool/0-1 mask with shape [N]')
        successful = flags.astype(bool)
    (best, best_score) = (None, None)
    (period, tolerance) = (2 * math.pi, 1e-10)
    for (candidate, accepted) in zip(candidates, successful):
        if not accepted or not np.all(np.isfinite(candidate)):
            continue
        adjusted = np.empty_like(position)
        for (column, (raw, actual, (lower, upper))) in enumerate(zip(candidate, position, bounds)):
            base = math.remainder(float(raw), period)
            low = math.ceil((float(lower) - base - tolerance) / period)
            high = math.floor((float(upper) - base + tolerance) / period)
            if low > high:
                break
            winding = min(high, max(low, round((float(actual) - base) / period)))
            value = base + period * winding
            if not math.isfinite(value) or value < lower - tolerance or value > upper + tolerance:
                break
            adjusted[column] = min(float(upper), max(float(lower), value))
        else:
            with np.errstate(over='ignore', invalid='ignore'):
                delta = adjusted - position
            if not np.all(np.isfinite(delta)):
                continue
            score = (float(np.max(np.abs(delta))), math.hypot(*delta))
            if not all((math.isfinite(value) for value in score)):
                continue
            if best_score is None or score < best_score:
                (best, best_score) = (adjusted, score)
    return best

def U74_quaternion_angle(quaternion, target_quaternion):
    """Full SO(3) geodesic error in [0, pi], invariant under q -> -q."""
    relative = U74_quat_mul_wxyz(U74_quat_inverse_wxyz(quaternion), target_quaternion)
    return 2 * math.atan2(float(np.linalg.norm(relative[1:])), abs(float(relative[0])))

def U74_pose_errors(actual_pose, target_pose):
    """Position/full orientation errors; yaw is diagnostic, never success.

    yaw_rad is None for an undefined planar heading (e.g. a tipped T with
    vertical local X). orientation_rad still measures that full error.
    """
    (actual, target) = (U74_validate_pose(actual_pose), U74_validate_pose(target_pose))
    delta = actual[:3] - target[:3]
    angle = U74_quaternion_angle(actual[3:], target[3:])
    try:
        yaw = abs(U74_wrap_yaw(U74_yaw_of_quaternion(actual[3:]) - U74_yaw_of_quaternion(target[3:])))
    except ValueError:
        yaw = None
    return {'position_m': float(np.linalg.norm(delta)), 'xy_m': float(np.linalg.norm(delta[:2])), 'z_m': abs(float(delta[2])), 'orientation_rad': angle, 'orientation_deg': math.degrees(angle), 'yaw_rad': yaw, 'yaw_deg': math.degrees(yaw) if yaw is not None else None}

def U53_quat_inverse_wxyz(quat):
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    norm_sq = float(quat @ quat)
    if norm_sq < 1e-12:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (np.asarray([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64) / norm_sq).astype(np.float32)

def U53_rotate_vec_by_quat(vector, quat):
    """Rotate a three vector with a scalar first quaternion."""
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    quat /= max(float(np.linalg.norm(quat)), 1e-12)
    w = float(quat[0])
    xyz = quat[1:]
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    cross = 2.0 * np.cross(xyz, vector)
    return (vector + w * cross + np.cross(xyz, cross)).astype(np.float32)

def U53_tool_pose_for_tip(tip_xyz, tool_quat, finger_offset):
    """Return the link6 pose whose closed finger tip reaches ``tip_xyz``."""
    local_x_world = U53_rotate_vec_by_quat(np.asarray([1.0, 0.0, 0.0], dtype=np.float32), tool_quat)
    wrist = np.asarray(tip_xyz, dtype=np.float32).reshape(3) - local_x_world * float(finger_offset)
    return np.asarray([*wrist, *np.asarray(tool_quat, dtype=np.float32).reshape(4)], dtype=np.float32)

def U51_yaw_delta(source, target):
    rel = U27_quat_mul_wxyz(target, U53_quat_inverse_wxyz(source))
    return (2 * math.atan2(float(rel[3]), float(rel[0])) + math.pi) % (2 * math.pi) - math.pi
