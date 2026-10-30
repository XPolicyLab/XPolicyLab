"""Smooth loaded TCP travel while retaining the selected endpoint IK branch."""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from coap.kinematics import JOINT_LOWER, JOINT_UPPER
from coap.motion import ARMS, TCP_OFFSET, link_to_tcp


def transport_path(ctx, targets, starts, endpoints, tcp=True):
    """Soft-fit intermediate Cartesian points around the original joint chord."""
    fractions = np.linspace(0., 1., 9)
    path = [np.concatenate([starts[arm] for arm in ARMS])]
    source = {arm: np.asarray(ctx.state[arm + '_ee_pose'], float).copy()
              for arm in targets}
    if tcp:
        source = {arm: np.r_[link_to_tcp(pose), pose[3:]] for arm, pose in source.items()}
    diagnostics = []
    for fraction in fractions[1:-1]:
        joints = {arm: (1. - fraction) * starts[arm] + fraction * endpoints[arm]
                  for arm in ARMS}
        for arm, goal in targets.items():
            reference = joints[arm].copy()
            frame = ctx.kinematics.forward_matrix(arm, reference)
            orientation = frame[:3, :3].copy()
            position = (1. - fraction) * source[arm][:3] + fraction * np.asarray(goal[:3])

            def residual(values):
                measured = ctx.kinematics.forward_matrix(arm, values)
                point = measured[:3, 3].copy()
                if tcp:
                    point += measured[:3, :3] @ TCP_OFFSET
                angular = Rotation.from_matrix(orientation.T @ measured[:3, :3]).as_rotvec()
                return np.r_[8. * (point - position), .35 * angular,
                             .025 * (values - reference)]

            solution = least_squares(residual, reference,
                bounds=(JOINT_LOWER + 1e-7, JOINT_UPPER - 1e-7),
                max_nfev=90, ftol=1e-8, xtol=1e-8, gtol=1e-8)
            joints[arm] = solution.x
            measured = ctx.kinematics.forward_matrix(arm, solution.x)
            point = measured[:3, 3].copy()
            if tcp:
                point += measured[:3, :3] @ TCP_OFFSET
            diagnostics.append(dict(arm=arm, fraction=float(fraction),
                position_error_m=float(np.linalg.norm(point - position)),
                rotation_error_rad=float(np.linalg.norm(Rotation.from_matrix(
                    orientation.T @ measured[:3, :3]).as_rotvec())),
                reference_joint_correction_rad=float(np.linalg.norm(solution.x - reference))))
        path.append(np.concatenate([joints[arm] for arm in ARMS]))
    path.append(np.concatenate([endpoints[arm] for arm in ARMS]))
    # Preserve both endpoint solutions and smooth tangent changes before retiming.
    spline = CubicSpline(fractions, np.asarray(path), axis=0, bc_type='natural')
    return spline(np.linspace(0., 1., 65)), diagnostics
