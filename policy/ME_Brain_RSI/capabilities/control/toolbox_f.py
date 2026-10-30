"""Preserve held-tool attitude with short Cartesian numerical-IK waypoints."""
import math

import numpy as np
from scipy.spatial.transform import Rotation

from coap.motion import ARMS, down_quat, interpolate_pose, link_to_tcp, tcp_to_link
from coap.tasks.memory_unchecked import SortingContext
from coap.tasks.toolbox_support import GRASP_SHIFT, measure, _lift


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


class ToolboxCurveContext(SortingContext):
    """Execute a numerical Cartesian route without arrival or tracking gates."""

    def _emit_path(self, path, arms, duration, velocity, goals=None):
        from coap.motion import _joint_action
        samples = fixed_duration_samples(path, duration)
        self.report_motion(dict(mode='toolbox_fixed_duration_cartesian_waypoints',
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


def tape_frame(ctx, source, destination, arm):
    """Retain candidate D's tape contact and release geometry unchanged."""
    direction = np.asarray(destination[:2])-[-.3 if arm == 'left' else .3, -.45]
    direction /= np.linalg.norm(direction)
    angle = math.radians(35.)
    forward = np.r_[direction*math.cos(angle), -math.sin(angle)]
    across = np.r_[-direction[1], direction[0], 0.]
    rotation = np.column_stack([forward, across, np.cross(forward, across)])
    quat = np.roll(Rotation.from_matrix(rotation).as_quat(), 1)
    source_tcp = np.asarray(source).copy()
    source_tcp[2] = ctx.table_z+.020
    target_tcp = np.asarray(destination)+(source_tcp-source)
    target_tcp[2] += .047
    return source_tcp, target_tcp, quat


def run(ctx):
    """Compare route sampling with D while retaining its contacts and durations."""
    ctx.stage = 'toolbox_measure'
    moves = measure(ctx)
    from coap.tasks.open_motion import enable
    enable(ctx)
    ctx.__class__ = ToolboxCurveContext
    ctx.memory['toolbox_strategy'] = dict(candidate='F', planned_tools=len(moves),
        safety_checks_enabled=False, execution='fixed_duration_cartesian_numeric_ik',
        tape_tilt_degrees=35., cartesian_waypoint_spacing_m=.020)
    priority = {'tape_measure': 0, 'wrench': 1, 'pliers': 2, 'hammer': 3}
    moves.sort(key=lambda item: priority[item['label']])
    for item in moves:
        name = item['label']
        source = np.asarray(item['source'])
        arm = 'left' if source[0] < 0 else 'right'
        if name == 'tape_measure':
            grasp, target, quat = tape_frame(ctx, source, item['destination'], arm)
        else:
            grasp = source+item['source_rotation'] @ [GRASP_SHIFT[name], 0., 0.]
            grasp[2] = ctx.table_z+.015
            quat = down_quat(math.pi/2+float(item['source_yaw']))
            target = np.asarray(item['destination'])+(grasp-source)
            target[2] += .040
        ctx.stage = 'toolbox_approach_'+name
        yield from ctx.move(arm, _lift(grasp, .085), quat, 1., 20)
        yield from ctx.move(arm, grasp, quat, 1., 18)
        ctx.stage = 'toolbox_close_'+name
        yield from ctx.grip(arm, 0., duration=14)
        yield from ctx.move(arm, _lift(grasp, .100), quat, 0., 20)
        ctx.stage = 'toolbox_carry_'+name
        yield from ctx.move(arm, _lift(target, .020), quat, 0., 26)
        yield from ctx.move(arm, target, quat, 0., 20)
        ctx.stage = 'toolbox_release_'+name
        yield from ctx.grip(arm, 1., duration=12)
        yield from ctx.move(arm, _lift(target, .055), quat, 1., 18)
        yield from ctx.home(arm, duration=24)
    ctx.stage = 'done'
    yield from ctx.home(duration=30)
    yield from ctx.hold(8)
