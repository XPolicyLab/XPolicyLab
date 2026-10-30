"""RGB garment folding inspired by the public WorldScape R2SI Skill sequence."""
import math

import numpy as np

from coap.motion import (ARMS, _joint_action, down_quat, interpolate_pose,
                         link_to_tcp, tcp_to_link)
from coap.tasks.finite_ik import FiniteSeedKinematics
from coap.tasks.garment import _garment_points, _carry_quat
from coap.tasks.memory_unchecked import SortingContext


CONTACT_HEIGHT = .0138


def lift(point, height):
    """Translate one current RGB target vertically."""
    result = np.asarray(point, float).copy()
    result[2] += height
    return result


def fixed_samples(path, duration):
    """Retain the declared native-frame budget for each manipulation segment."""
    path = np.asarray(path, float)
    fraction = np.arange(1, int(duration)+1)/int(duration)
    fraction = fraction*fraction*(3.-2.*fraction)
    coordinate = fraction*(len(path)-1)
    lower = np.minimum(coordinate.astype(int), len(path)-2)
    local = coordinate-lower
    return path[lower]+local[:, None]*(path[lower+1]-path[lower])


class FoldContext(SortingContext):
    """Issue finite numerical paths without arrival or grasp interruption gates."""

    def _move_targets(self, targets, duration, tcp=True, motion_profile=None):
        joints = {arm: np.asarray(self.state[arm+'_arm_joint_state'], float).copy()
                  for arm in ARMS}
        starts = {arm: self.kinematics.forward(arm, joints[arm]) for arm in targets}
        if tcp:
            starts = {arm: np.r_[link_to_tcp(pose), pose[3:]]
                      for arm, pose in starts.items()}
        count = 1
        for arm, goal in targets.items():
            distance = np.linalg.norm(goal[:3]-starts[arm][:3])
            angle = 2.*math.acos(float(np.clip(abs(np.dot(goal[3:], starts[arm][3:])), 0., 1.)))
            count = max(count, int(math.ceil(distance/.018)), int(math.ceil(angle/.10)))
        path = [np.concatenate([joints[arm] for arm in ARMS])]
        diagnostics = []
        for index in range(1, count+1):
            for arm, goal in targets.items():
                point = interpolate_pose(starts[arm], goal, index/count)
                pose = tcp_to_link(point[:3], point[3:]) if tcp else point
                solution = self.kinematics.solve(arm, pose, current_joints=joints[arm],
                    position_tol=.002, rotation_tol=.020, max_starts=8, max_nfev=120)
                joints[arm] = np.asarray(solution['joint_value'], float)
                diagnostics.append({'arm': arm, 'fraction': index/count,
                    'position_error_m': solution['position_error'],
                    'rotation_error_rad': solution['rotation_error']})
            path.append(np.concatenate([joints[arm] for arm in ARMS]))
        yield from self._emit_path(np.asarray(path), tuple(targets), duration, 2.)
        self.motion_plan['ik_diagnostics'] = diagnostics

    def _emit_path(self, path, arms, duration, velocity, goals=None):
        samples = fixed_samples(path, duration)
        self.report_motion({'mode': 'fold_fixed_duration_cartesian_numeric_ik',
            'frames': len(samples), 'planned_waypoints': len(path),
            'safety_checks_enabled': False})
        for sample in samples:
            commands = {arm: sample[index*6:(index+1)*6].copy()
                        for index, arm in enumerate(ARMS)}
            self.joint_targets = commands
            yield _joint_action(commands, self.gripper_targets)
        self.joint_command_velocity = {arm: np.zeros(6) for arm in ARMS}

    def home(self, arm=None, duration=32, grip=1., motion_profile=None):
        from coap.kinematics import nearest_periodic
        arms = ARMS if arm is None else (arm,)
        start = np.concatenate([np.asarray(self.state[side+'_arm_joint_state'], float)
                                for side in ARMS])
        end = start.copy()
        for index, side in enumerate(ARMS):
            if side in arms:
                end[index*6:(index+1)*6] = nearest_periodic(
                    self.initial_state[side+'_arm_joint_state'], start[index*6:(index+1)*6])
                self.gripper_targets[side] = float(grip)
        yield from self._emit_path(np.stack([start, end]), arms, duration, 2.)


def enable(ctx):
    """Keep this candidate's executor local to the current episode context."""
    if ctx.kinematics is None:
        from coap.kinematics import X5Kinematics
        ctx.kinematics = X5Kinematics(ctx.initial_state)
    if not isinstance(ctx.kinematics, FiniteSeedKinematics):
        ctx.kinematics = FiniteSeedKinematics(ctx.kinematics)
    ctx.numeric_ik = True
    ctx.__class__ = FoldContext


def measure(ctx):
    """Reuse only the established public RGB garment contour detector."""
    found = ctx.detect('clothes')
    return max(found, key=lambda item: item.get('confidence', 0.)) if found else None


def contour_world(ctx, item):
    """Project the current visible material silhouette onto its support plane."""
    if item is None:
        return np.empty((0, 3))
    pixels = np.asarray(item.get('contour', []), float).reshape(-1, 2)
    return np.asarray([ctx.project(pixel, ctx.table_z+.006,
                                  camera=item.get('camera', 'cam_head'))
                       for pixel in pixels]).reshape(-1, 3)


def snap_edge(point, contour, inward, search_radius=.10):
    """Associate a semantic search hint with the nearest present cloth edge."""
    point = np.asarray(point, float).copy()
    contour = np.asarray(contour, float).reshape(-1, 3)
    if not len(contour):
        return point
    distance = np.linalg.norm(contour[:, :2]-point[:2], axis=1)
    selected = contour[np.argmin(distance)].copy()
    blend = min(1., search_radius/max(float(np.min(distance)), search_radius))
    point[:2] = blend*selected[:2]+(1.-blend)*point[:2]
    direction = np.asarray(inward, float)[:2]
    point[:2] += .005*direction/max(np.linalg.norm(direction), 1e-9)
    return point


def body_frame(points):
    """Infer width and hem-to-neck directions from the observed garment."""
    left, right = (np.asarray(points[side+'_hem'], float) for side in ('left', 'right'))
    width = right[:2]-left[:2]
    width /= max(np.linalg.norm(width), 1e-9)
    upper = .5*(points['left_shoulder']+points['right_shoulder'])
    height = upper[:2]-.5*(left[:2]+right[:2])
    height -= width*np.dot(height, width)
    height /= max(np.linalg.norm(height), 1e-9)
    return width, height


def paired_targets(points, sources):
    """Preserve the captured hem vector while matching the upper body direction."""
    width, height = body_frame(points)
    upper = .5*(points['left_shoulder']+points['right_shoulder'])
    center = np.asarray(upper, float).copy()
    center[:2] -= .035*height
    span = np.linalg.norm(np.asarray(sources[1])[:2]-np.asarray(sources[0])[:2])
    return [center+np.r_[sign*.5*span*width, 0.] for sign in (-1., 1.)]


def paired_arc(sources, targets, fraction, height):
    """Move both captured corners through the same fold parameter."""
    return [(1.-fraction)*np.asarray(source)+fraction*np.asarray(target)
            +[0., 0., height*math.sin(math.pi*fraction)]
            for source, target in zip(sources, targets)]


def _grip_both(ctx, value, duration):
    """Close or open both hands on the same native action frames."""
    starts = dict(ctx.gripper_targets)
    for arm in ARMS:
        ctx.gripper_targets[arm] = float(value)
    for index in range(duration):
        action = ctx.hold_action()
        for arm in ARMS:
            action[arm+'_ee_joint_state'] = np.array([
                starts[arm]+(value-starts[arm])*(index+1)/duration])
        yield action


def _sleeve(ctx, side, points):
    """Fold one current cuff, release on support, then separate vertically."""
    other = 'right' if side == 'left' else 'left'
    center = .5*(points['left_chest']+points['right_chest'])
    target = center+.5*(points[other+'_chest']-center)
    contour = contour_world(ctx, measure(ctx))
    source = snap_edge(points[side+'_sleeve'], contour, target-points[side+'_sleeve'])
    source[2] = ctx.table_z+CONTACT_HEIGHT
    target[2] = ctx.table_z+CONTACT_HEIGHT+.006
    arm = 'left' if source[0] < 0 else 'right'
    yaw = math.atan2(target[1]-source[1], target[0]-source[0])
    quat = down_quat(yaw)
    ctx.memory.setdefault('fold_sleeve_targets', {})[side] = {
        'source': source.tolist(), 'destination': target.tolist(), 'arm': arm}
    ctx.stage = 'fold_'+side+'_sleeve_approach'
    yield from ctx.move(arm, lift(source, .065), quat, .6, 16)
    yield from ctx.move(arm, source, quat, .6, 12)
    ctx.stage = 'fold_'+side+'_sleeve_close'
    yield from ctx.grip(arm, 0., 12)
    yield from ctx.move(arm, lift(source, .035), quat, 0., 8)
    for index, fraction in enumerate((1./3., 2./3., 1.)):
        point = (1.-fraction)*source+fraction*target
        point[2] += .060*math.sin(math.pi*fraction)
        carry = _carry_quat(ctx.kinematics, arm, target, quat, 20., fraction)
        ctx.stage = 'fold_'+side+'_sleeve_arc_'+str(index)
        yield from ctx.move(arm, point, carry, 0., 10)
    ctx.stage = 'fold_'+side+'_sleeve_release'
    yield from ctx.grip(arm, 1., 12)
    carry = _carry_quat(ctx.kinematics, arm, target, quat, 20.)
    yield from ctx.move(arm, lift(target, .035), carry, 1., 8)
    park = lift(source, .080)
    ctx.stage = 'fold_'+side+'_sleeve_park'
    yield from ctx.move(arm, park, quat, 1., 10)


def run(ctx):
    """Execute the public sleeves-first, paired-hem strategy in 418 frames."""
    enable(ctx)
    ctx.stage = 'fold_measure'
    item = measure(ctx)
    points = _garment_points(ctx, item) if item else None
    if points is None:
        ctx.memory['fold_missing_rgb_landmarks'] = True
        yield from ctx.hold(1)
        return
    ctx.memory['fold_strategy'] = {'candidate': 'A', 'reference': 'WorldScape-R2SI',
        'method': 'sleeves_then_paired_hem', 'safety_checks_enabled': False,
        'native_frame_budget': 418, 'input': 'current_RGB_and_proprioception'}
    ctx.memory['fold_initial_landmarks'] = {key: value.tolist() for key, value in points.items()}
    for side in ('left', 'right'):
        yield from _sleeve(ctx, side, points)
    width, inward = body_frame(points)
    contour = contour_world(ctx, measure(ctx))
    sources = [snap_edge(points[side+'_hem'], contour, inward)
               for side in ('left', 'right')]
    sources.sort(key=lambda point: point[0])
    for source in sources:
        source[2] = ctx.table_z+CONTACT_HEIGHT
    targets = paired_targets(points, sources)
    for target in targets:
        target[2] = ctx.table_z+CONTACT_HEIGHT+.008
    yaw = math.atan2(width[1], width[0])
    quats = [down_quat(yaw), down_quat(yaw+math.pi)]
    entry = [source-np.r_[.018*inward, 0.] for source in sources]
    ctx.memory['fold_current_sources'] = [source.tolist() for source in sources]
    ctx.memory['fold_destinations'] = [target.tolist() for target in targets]
    ctx.stage = 'fold_paired_hem_approach'
    yield from ctx.move_both(*[lift(point, .065) for point in entry], *quats, .45, .45, 18)
    yield from ctx.move_both(*entry, *quats, .45, .45, 14)
    ctx.stage = 'fold_paired_hem_open_rake'
    gathered = [source+np.r_[.006*inward, 0.] for source in sources]
    yield from ctx.move_both(*gathered, *quats, .45, .45, 8)
    ctx.stage = 'fold_paired_hem_close'
    yield from _grip_both(ctx, 0., 14)
    yield from ctx.move_both(*[lift(point, .035) for point in gathered], *quats, 0., 0., 10)
    travel = np.linalg.norm((.5*(targets[0]+targets[1])-.5*(gathered[0]+gathered[1]))[:2])
    arc_height = float(np.clip(.5*travel, .095, .140))
    for index, fraction in enumerate(np.linspace(1./6., 1., 6)):
        goals = paired_arc(gathered, targets, float(fraction), arc_height)
        carry = [_carry_quat(ctx.kinematics, arm, target, quat, 20., float(fraction), True)
                 for arm, target, quat in zip(ARMS, targets, quats)]
        ctx.stage = 'fold_paired_hem_arc_'+str(index)
        yield from ctx.move_both(*goals, *carry, 0., 0., 12)
    ctx.stage = 'fold_paired_hem_release'
    yield from _grip_both(ctx, 1., 14)
    carry = [_carry_quat(ctx.kinematics, arm, target, quat, 20., 1., True)
             for arm, target, quat in zip(ARMS, targets, quats)]
    ctx.stage = 'fold_paired_hem_vertical_separation'
    yield from ctx.move_both(*[lift(target, .055) for target in targets], *carry, 1., 1., 12)
    ctx.stage = 'done'
    yield from ctx.home(duration=32)
    yield from ctx.hold(8)


fold_clothes = run
fold_clothes_random = run
