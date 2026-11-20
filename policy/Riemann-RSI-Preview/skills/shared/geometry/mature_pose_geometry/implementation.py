from __future__ import annotations
import math
import numpy as np

def _array(value, shape, name):
    try:
        result = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be finite with shape {shape}') from exc
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f'{name} must be finite with shape {shape}')
    return result

def _scalar(value, name, *, positive=False, nonnegative=False):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be finite') from exc
    if not math.isfinite(result) or (positive and result <= 0) or (nonnegative and result < 0):
        raise ValueError(f'{name} must be finite' + (' and positive' if positive else ' and nonnegative' if nonnegative else ''))
    return result

def normalize_quat_wxyz(quaternion):
    """Normalize a scalar-first quaternion; reject zero/NaN/Inf, never repair.

    The first numerically significant component is positive, so q and -q
    share a deterministic representative, including exact half turns.
    Scaling before the norm avoids overflow for finite non-unit inputs.
    """
    quaternion = _array(quaternion, (4,), 'quaternion')
    scale = float(np.max(np.abs(quaternion)))
    if scale == 0:
        raise ValueError('zero quaternion has no rotation')
    quaternion /= scale
    quaternion /= np.linalg.norm(quaternion)
    first = np.flatnonzero(np.abs(quaternion) > 1e-15)[0]
    if quaternion[first] < 0:
        quaternion *= -1
    return quaternion

def validate_pose(pose):
    """Copy/normalize a finite 7D XYZ/wxyz pose; reject malformed rotations."""
    result = _array(pose, (7,), 'pose')
    result[3:] = normalize_quat_wxyz(result[3:])
    return result

def quat_mul_wxyz(left, right):
    """Compose two rotation quaternions, returning a unit quaternion."""
    (w, x, y, z) = normalize_quat_wxyz(left)
    (a, b, c, d) = normalize_quat_wxyz(right)
    return normalize_quat_wxyz([w * a - x * b - y * c - z * d, w * b + x * a + y * d - z * c, w * c - x * d + y * a + z * b, w * d + x * c - y * b + z * a])

def quat_inverse_wxyz(quaternion):
    result = normalize_quat_wxyz(quaternion)
    result[1:] *= -1
    return normalize_quat_wxyz(result)

def quat_to_mat_wxyz(quaternion):
    (w, x, y, z) = normalize_quat_wxyz(quaternion)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], dtype=np.float64)

def mat_to_quat_wxyz(matrix):
    """Convert a proper orthonormal 3x3 rotation, rejecting shear/reflection."""
    matrix = _array(matrix, (3, 3), 'rotation matrix')
    if not np.allclose(matrix.T @ matrix, np.eye(3), rtol=0, atol=1e-06) or not math.isclose(float(np.linalg.det(matrix)), 1.0, rel_tol=0, abs_tol=2e-06):
        raise ValueError('rotation matrix must be orthonormal with determinant +1')
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2
        result = [scale / 4, (matrix[2, 1] - matrix[1, 2]) / scale, (matrix[0, 2] - matrix[2, 0]) / scale, (matrix[1, 0] - matrix[0, 1]) / scale]
    else:
        axis = int(np.argmax(np.diag(matrix)))
        (j, k) = ((axis + 1) % 3, (axis + 2) % 3)
        scale = math.sqrt(1 + matrix[axis, axis] - matrix[j, j] - matrix[k, k]) * 2
        result = np.zeros(4)
        result[0] = (matrix[k, j] - matrix[j, k]) / scale
        result[axis + 1] = scale / 4
        result[j + 1] = (matrix[j, axis] + matrix[axis, j]) / scale
        result[k + 1] = (matrix[k, axis] + matrix[axis, k]) / scale
    return normalize_quat_wxyz(result)

def wrap_yaw(angle):
    """Canonical radians in [-pi, pi); an exact pi tie turns clockwise."""
    angle = _scalar(angle, 'yaw')
    return (angle + math.pi) % (2 * math.pi) - math.pi

def yaw_of_quaternion(quaternion):
    """Heading of body-local X projected on world XY (ZYX yaw).

    A vertical local X has no planar heading. Fail explicitly instead of
    inventing yaw zero; full quaternion errors remain valid in that case.
    """
    rotation = quat_to_mat_wxyz(quaternion)
    if np.linalg.norm(rotation[:2, 0]) < 1e-10:
        raise ValueError('yaw is undefined when body-local X is vertical')
    return wrap_yaw(math.atan2(float(rotation[1, 0]), float(rotation[0, 0])))

def rotate_vec_by_quat(vector, quaternion):
    return quat_to_mat_wxyz(quaternion) @ _array(vector, (3,), 'vector')

def compose_pose(parent, child):
    """SE(3) product: a parent/world pose times a child/parent pose."""
    (parent, child) = (validate_pose(parent), validate_pose(child))
    return np.r_[parent[:3] + rotate_vec_by_quat(child[:3], parent[3:]), quat_mul_wxyz(parent[3:], child[3:])]

def inverse_pose(pose):
    pose = validate_pose(pose)
    inverse = quat_inverse_wxyz(pose[3:])
    return np.r_[rotate_vec_by_quat(-pose[:3], inverse), inverse]

def tool_pose_for_tip(tip_xyz, tool_quaternion, finger_offset):
    """Tool-link target for a tip extending ``finger_offset`` along local +X."""
    tip = _array(tip_xyz, (3,), 'tip position')
    quaternion = normalize_quat_wxyz(tool_quaternion)
    offset = _scalar(finger_offset, 'finger_offset', nonnegative=True)
    return np.r_[tip - rotate_vec_by_quat([offset, 0, 0], quaternion), quaternion]

def closest_joint_solution(solutions, current, limits, success=None):
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
    bounds = _array(limits, (position.size, 2), 'joint limits')
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

def quaternion_angle(quaternion, target_quaternion):
    """Full SO(3) geodesic error in [0, pi], invariant under q -> -q."""
    relative = quat_mul_wxyz(quat_inverse_wxyz(quaternion), target_quaternion)
    return 2 * math.atan2(float(np.linalg.norm(relative[1:])), abs(float(relative[0])))

def pose_errors(actual_pose, target_pose):
    """Position/full orientation errors; yaw is diagnostic, never success.

    yaw_rad is None for an undefined planar heading (e.g. a tipped T with
    vertical local X). orientation_rad still measures that full error.
    """
    (actual, target) = (validate_pose(actual_pose), validate_pose(target_pose))
    delta = actual[:3] - target[:3]
    angle = quaternion_angle(actual[3:], target[3:])
    try:
        yaw = abs(wrap_yaw(yaw_of_quaternion(actual[3:]) - yaw_of_quaternion(target[3:])))
    except ValueError:
        yaw = None
    return {'position_m': float(np.linalg.norm(delta)), 'xy_m': float(np.linalg.norm(delta[:2])), 'z_m': abs(float(delta[2])), 'orientation_rad': angle, 'orientation_deg': math.degrees(angle), 'yaw_rad': yaw, 'yaw_deg': math.degrees(yaw) if yaw is not None else None}
