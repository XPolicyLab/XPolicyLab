"""Fixed-duration Cartesian IK shared only by independent hanging candidates."""
import math

import numpy as np

from coap.motion import ARMS, down_quat, interpolate_pose, link_to_tcp, tcp_to_link
from coap.tasks.memory_unchecked import SortingContext


def cartesian_joint_path(kinematics, joints, targets, tcp=True):
    """Solve each short Cartesian waypoint from the preceding joint solution."""
    current = {arm: np.asarray(value, float).copy() for arm, value in joints.items()}
    starts = {arm: kinematics.forward(arm, current[arm]) for arm in targets}
    if tcp:
        starts = {arm: np.r_[link_to_tcp(pose), pose[3:]] for arm, pose in starts.items()}
    count = 1
    for arm, goal in targets.items():
        distance = float(np.linalg.norm(np.asarray(goal[:3])-starts[arm][:3]))
        angle = 2.*math.acos(float(np.clip(abs(np.dot(goal[3:], starts[arm][3:])), 0., 1.)))
        count = max(count, int(math.ceil(distance/.020)), int(math.ceil(angle/.10)))
    path = [np.concatenate([current[arm] for arm in ARMS])]
    diagnostics = []
    for index in range(1, count+1):
        fraction = index/count
        for arm, goal in targets.items():
            waypoint = interpolate_pose(starts[arm], goal, fraction)
            pose = tcp_to_link(waypoint[:3], waypoint[3:]) if tcp else waypoint
            result = kinematics.solve(arm, pose, current_joints=current[arm],
                position_tol=.002, rotation_tol=.020, max_starts=12, max_nfev=160)
            current[arm] = np.asarray(result['joint_value'], float)
            diagnostics.append(dict(arm=arm, fraction=fraction, status=result['status'],
                position_error_m=result['position_error'], rotation_error_rad=result['rotation_error']))
        path.append(np.concatenate([current[arm] for arm in ARMS]))
    return np.asarray(path), diagnostics


def fixed_duration_samples(path, duration):
    """Traverse every joint waypoint using the caller's fixed frame budget."""
    path = np.asarray(path, float)
    duration = max(1, int(duration))
    fraction = np.arange(1, duration+1)/duration
    smooth = fraction*fraction*(3.-2.*fraction)
    parameter = smooth*(len(path)-1)
    lower = np.minimum(np.floor(parameter).astype(int), len(path)-2)
    local = parameter-lower
    return path[lower]+local[:, None]*(path[lower+1]-path[lower])


class HangCurveContext(SortingContext):
    """Execute a numerical Cartesian route without arrival or tracking gates."""

    def _emit_path(self, path, arms, duration, velocity, goals=None):
        from coap.motion import _joint_action
        samples = fixed_duration_samples(path, duration)
        self.report_motion(dict(mode='hang_fixed_duration_cartesian_waypoints',
            frames=len(samples), planned_waypoints=len(path), safety_checks_enabled=False,
            nominal_frame_goals=goals or {}))
        for point in samples:
            commands = {arm: point[offset*6:(offset+1)*6].copy()
                        for offset, arm in enumerate(ARMS)}
            self.joint_targets = {arm: value.copy() for arm, value in commands.items()}
            yield _joint_action(commands, self.gripper_targets)
        self.joint_command_velocity = {arm: np.zeros(6) for arm in ARMS}

    def _move_targets(self, targets, duration, tcp=True, motion_profile=None):
        starts = {arm: np.asarray(self.state[arm+'_arm_joint_state'], float).copy() for arm in ARMS}
        self.joint_targets = {arm: value.copy() for arm, value in starts.items()}
        path, diagnostics = cartesian_joint_path(self.kinematics, starts, targets, tcp)
        yield from self._emit_path(path, tuple(targets), duration, 2.,
            {arm: np.asarray(goal).tolist() for arm, goal in targets.items()})
        self.motion_plan['cartesian_ik_diagnostics'] = diagnostics
