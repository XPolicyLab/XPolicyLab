"""Rim/pivot transfer (pouring) and release-retreat. Preloaded into every program; nothing moves at load.

The geometry assumes a rigidly held upright cylindrical source and a measured two-cylinder receiver envelope.
Every motion is an ordinary move_to in the calibrated 'tcp' frame, so each native action is counted, planned and
budget-checked as usual. No object poses, simulator internals or scores are read.
"""
import math

import numpy as np

GRAVITY_M_S2 = 9.81


def jet_landing(mouth_xyz, outflow, receiver_xyz, receiver_radius, exit_speed=0.3, margin=0.01):
    """Where a stream leaving the mouth lands at the receiver's rim height: check it before any liquid flows.

    mouth_xyz: the lip point the liquid leaves from; outflow: the direction it leaves in (e.g. the source's axis
    toward the mouth at the planned tilt; any length); receiver_xyz: centre of the receiver's opening at rim
    height; exit_speed in m/s (calibrate it with jet_speed from the first observed landing; the default is only
    a starting guess). Ballistic: the stream keeps its horizontal velocity while it falls to the rim height, so
    it lands `throw_m` beyond the mouth, farther the higher the mouth and the faster the flow; directly below the
    mouth only when the flow is slow. Statuses: inside (miss_m <= receiver_radius - margin), outside,
    mouth_below_rim. aim_mouth_xyz: the mouth position (same height and outflow) whose stream lands on the centre.
    Spilled liquid does not come back: check before the flow, and again on the first visible stream.
    """
    mouth, centre = np.asarray(mouth_xyz, float), np.asarray(receiver_xyz, float)
    direction = np.asarray(outflow, float)
    if mouth.shape != (3,) or centre.shape != (3,) or direction.shape != (3,) or np.linalg.norm(direction) < 1e-9 \
            or not np.isfinite(np.concatenate([mouth, centre, direction])).all() or not exit_speed >= 0 \
            or not receiver_radius > 0:
        raise ValueError('finite mouth, receiver and outflow, nonnegative speed and positive radius required')
    direction = direction / np.linalg.norm(direction)
    fall = float(mouth[2] - centre[2])
    if fall <= 0:
        return {'status': 'mouth_below_rim', 'fall_m': round(fall, 4)}
    vz = float(exit_speed) * float(direction[2])
    t = (vz + math.sqrt(vz * vz + 2.0 * GRAVITY_M_S2 * fall)) / GRAVITY_M_S2
    throw = float(exit_speed) * t * direction[:2]
    landing = np.array([mouth[0] + throw[0], mouth[1] + throw[1], centre[2]])
    miss = float(np.linalg.norm(landing[:2] - centre[:2]))
    aim = [float(centre[0] - throw[0]), float(centre[1] - throw[1]), float(mouth[2])]
    return {'status': 'inside' if miss <= float(receiver_radius) - float(margin) else 'outside',
            'landing_xyz': [round(float(c), 4) for c in landing], 'miss_m': round(miss, 4),
            'throw_m': round(float(np.linalg.norm(throw)), 4), 'fall_m': round(fall, 4), 'time_s': round(t, 3),
            'aim_mouth_xyz': [round(c, 4) for c in aim]}


def jet_speed(mouth_xyz, outflow, landing_xyz):
    """Exit speed that explains an observed landing point, for jet_landing's next prediction.

    landing_xyz: where the stream meets the rim plane (e.g. ray_plane_point of the stream's pixel at rim height,
    or the visible splash on the table at table height). Only the part of the landing along the outflow's
    horizontal direction is explained by speed; lateral_m (across it) is not: the mouth estimate or the outflow
    direction is off by about that much, so correct the mouth before the next flow. Statuses: measured,
    vertical_outflow (no horizontal direction to fit), inconsistent (landing behind or above the mouth).
    """
    mouth, landing = np.asarray(mouth_xyz, float), np.asarray(landing_xyz, float)
    direction = np.asarray(outflow, float) / np.linalg.norm(outflow)
    horizontal = float(np.linalg.norm(direction[:2]))
    if horizontal < 1e-6:
        return {'status': 'vertical_outflow'}
    heading = direction[:2] / horizontal
    offset = landing[:2] - mouth[:2]
    along = float(offset @ heading)
    lateral = offset - along * heading
    fall = float(mouth[2] - landing[2])
    out = {'along_m': round(along, 4), 'lateral_m': round(float(np.linalg.norm(lateral)), 4),
           'lateral_xy': [round(float(c), 4) for c in lateral], 'fall_m': round(fall, 4)}
    drop = fall + float(direction[2]) * along / horizontal
    if along <= 0 or fall <= 0 or drop <= 0:
        out['status'] = 'inconsistent'
        return out
    out.update(status='measured', exit_speed=round(along / horizontal * math.sqrt(GRAVITY_M_S2 / (2.0 * drop)), 4))
    return out


def pour_axis_rotation(axis, degrees):
    x, y, z = np.eye(3)['xyz'.index(axis)]
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    a = math.radians(degrees)
    return np.eye(3) + math.sin(a) * skew + (1 - math.cos(a)) * (skew @ skew)


def pour_pivot_pose(initial, point, destination, axis, degrees):
    rot = pour_axis_rotation(axis, degrees)
    target = initial.copy()
    target[:3, :3] = rot @ initial[:3, :3]
    target[:3, 3] = destination + rot @ (initial[:3, 3] - point)
    return target


def pour_rim_edge(center, radius, axis, angle):
    direction = np.cross(np.eye(3)['xyz'.index(axis)], [0., 0., 1.])
    return np.asarray(center, float) + math.copysign(radius, angle) * direction


def pour_overlap_depth(height, receiver_radius, degrees):
    a = math.radians(abs(degrees))
    if a >= math.pi / 2:
        return 0.
    return min(height, receiver_radius / max(math.sin(a), 1e-12)) * math.cos(a)


def pour_profiled_depth(height, envelope, neck_radius, neck_depth, degrees, margin=0.):
    return max(0., pour_overlap_depth(height, neck_radius + margin, degrees),
               pour_overlap_depth(height, envelope + margin, degrees) - neck_depth)


def pour_session(arm, tcp_offset_local, calibration_ref, max_total_steps, max_stage_steps):
    """Bookkeeping for one bounded pouring call: max_total_steps native actions over all its stages."""
    if arm not in ARMS or not calibration_ref:
        raise ValueError('arm and TCP calibration provenance required')
    for name, value, high in (('max_total_steps', max_total_steps, 1000), ('max_stage_steps', max_stage_steps,
                                                                            max_total_steps)):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= high:
            raise ValueError(name + ' must be an integer in 1..' + str(high))
    offset = np.asarray(tcp_offset_local, float)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError('tcp_offset_local must be three finite metres')
    return {'arm': arm, 'offset': offset, 'calibration_ref': calibration_ref, 'max_total_steps': max_total_steps,
            'max_stage_steps': max_stage_steps, 'calls': 0, 'stages': []}


def pour_tcp(session):
    pose = np.asarray(get_observation()['state'][session['arm'] + '_ee_pose'], float)
    w, x, y, z = pose[3:] / np.linalg.norm(pose[3:])
    t = np.eye(4)
    t[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                 [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                 [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]
    t[:3, 3] = pose[:3] + t[:3, :3] @ session['offset']
    return t


def pour_move(session, name, target):
    remaining = session['max_total_steps'] - session['calls']
    if remaining <= 0:
        raise RuntimeError('pour native action budget exhausted')
    result = move_to(session['arm'], target[:3, 3].tolist(), rotation_quaternion(target[:3, :3]), frame='tcp',
                     tcp_offset_local=session['offset'].tolist(), calibration_ref=session['calibration_ref'],
                     max_steps=min(session['max_stage_steps'], remaining), label='pour_' + name)
    session['calls'] += int(result.get('calls') or 0)
    session['stages'].append(dict(result, stage=name))
    if result['status'] != 'tracking_reached':
        raise RuntimeError('pour stage ' + name + ' stopped: ' + str(result['status']))
    return result


def rim_transfer_preflight(session, stages):
    """check_reach of each (name, TCP pose) before anything moves: the first stage the robot model finds out of
    reach (no IK solution counts; a missing model does not), or None. Recorded pours stalled 3-22 cm short of
    the high upright approach and spilled on the way. Named apart from the agents' own `pour_preflight`
    helpers: a skill defining that name replaced this one and broke rim_transfer (pour_balls pg-s23, 10-31)."""
    for name, target in stages:
        flange = target[:3, 3] - target[:3, :3] @ session['offset']
        verdict = check_reach(flange.tolist(), rotation_quaternion(target[:3, :3]), frame='ee',
                              arms=[session['arm']])[session['arm']]
        no_model = 'no kinematics model' in str(verdict.get('reason') or '')
        if verdict['reachable'] is False or (verdict['reachable'] != True and not no_model):
            return {'stage': name, 'reachable': verdict['reachable'], 'reason': verdict.get('reason'),
                    'tcp_xyz': [round(float(v), 4) for v in target[:3, 3]]}
    return None


def rim_transfer(*, arm, center, destination, radius, height, envelope, neck_radius, neck_depth, angle, axis,
                 geometry_ref, attachment_ref, tcp_offset_local, calibration_ref, max_total_steps,
                 max_stage_steps=40, increment=10., gap=.02, hold_steps=0, restore=True, grip_command=None):
    """Move the measured source's low rim over a receiver, tilt about a horizontal axis and optionally undo.

    center: current source rim centre XYZ; destination: current receiver opening XYZ; radius/height: source rim
    radius and height; envelope/neck_radius/neck_depth: receiver body and neck radii and neck depth; angle: signed
    tilt degrees (<= 160); axis 'x' or 'y' (perpendicular to the finger closing line). Inspect the grasp and the
    clearance first: references are provenance, not proof of attachment. Returns tracking only; pouring,
    holding and placement stay 'unknown'. grip_command is accepted for compatibility (the host keeps the last
    gripper command).
    """
    point, destination = np.asarray(center, float), np.asarray(destination, float)
    values = [radius, height, envelope, neck_radius, neck_depth, angle, increment, gap]
    if point.shape != (3,) or destination.shape != (3,) or not np.isfinite(values).all() or axis not in ('x', 'y') \
            or not isinstance(restore, bool):
        raise ValueError('finite geometry, horizontal axis and boolean restore required')
    if not (radius > 0 and height > 0 and envelope >= neck_radius > 0 and neck_depth >= 0
            and 0 < abs(angle) <= 160 and 0 < increment <= 20 and gap > 0):
        raise ValueError('invalid rim/envelope/sweep bounds')
    if not geometry_ref or not attachment_ref:
        raise ValueError('current geometry and inspected attachment evidence references required')
    if isinstance(hold_steps, bool) or not isinstance(hold_steps, int) or hold_steps < 0:
        raise ValueError('hold_steps must be a nonnegative integer')
    session = pour_session(arm, tcp_offset_local, calibration_ref, max_total_steps, max_stage_steps)
    initial = pour_tcp(session)
    axis_vector = np.eye(3)['xy'.index(axis)]
    if abs(float(axis_vector @ (initial[:3, :3] @ np.asarray(CLOSING_AXIS, float)))) > math.sin(math.radians(15)):
        raise ValueError('rotation_axis_not_perpendicular_to_fingers')
    edge = pour_rim_edge(point, radius, axis, angle)
    margin = (2 * radius + height) * (1 - math.cos(math.radians(10)))
    final = destination + [0., 0., gap]
    aligned = final + [0., 0., height + margin]
    transit_z = max(float(edge[2]), float(aligned[2]))
    count = math.ceil(abs(angle) / increment)
    if 3 + count * (2 if restore else 1) + hold_steps > max_total_steps:
        raise ValueError('insufficient minimum native action budget before motion')
    result = {'status': 'incomplete', 'stages': session['stages'], 'attachment': 'unknown', 'pouring': 'unknown',
              'placement': 'unknown', 'geometry_ref': geometry_ref, 'attachment_ref': attachment_ref,
              'evidence_scope': 'caller references; rigid attachment assumed', 'source_edge_world': edge.tolist(),
              'interpolation_margin_m': margin, 'completed_angle_deg': 0., 'restored_upright': False}
    elevated = edge.copy()
    elevated[2] = transit_z
    overhead = aligned.copy()
    overhead[2] = transit_z
    clearance = [(name, pour_pivot_pose(initial, edge, target_point, axis, 0)) for name, target_point in
                 [('clearance_lift', elevated), ('clearance_transit', overhead), ('clearance_align', aligned)]]
    tilts = [(f'pivot_{degrees:.0f}', pour_pivot_pose(initial, edge, final + [0., 0., margin + pour_profiled_depth(
        height, envelope, neck_radius, neck_depth, abs(degrees), margin)], axis, degrees))
        for degrees in (np.linspace(0, angle, count + 1)[1:][[0, -1]])]
    blocked = rim_transfer_preflight(session, clearance + tilts)
    if blocked is not None:
        result.update(status='unreachable', unreachable=blocked, calls=0,
                      hint='nothing moved; try the other arm, the other tilt axis or sign, a lower lift (grasp the '
                           'source lower), or bring the receiver closer to the pouring arm')
        return result
    for name, target in clearance:
        pour_move(session, name, target)
    restore_path = [pour_tcp(session)]
    for index, degrees in enumerate(np.linspace(0, angle, count + 1)[1:]):
        previous = abs(angle) * index / count
        depth = pour_profiled_depth(height, envelope, neck_radius, neck_depth, previous, margin)
        target = pour_pivot_pose(initial, edge, final + [0., 0., margin + depth], axis, degrees)
        pour_move(session, 'pivot', target)
        restore_path.append(target.copy())
        result['completed_angle_deg'] = float(degrees)
    for _ in range(hold_steps):
        pour_move(session, 'settle', pour_tcp(session))
    if restore:
        for target in reversed(restore_path[:-1]):
            pour_move(session, 'restore', target)
        result['restored_upright'] = True
    result.update(status='tracking_reached', calls=session['calls'], reached_tcp=pour_tcp(session).tolist())
    return result


def release_retreat(*, arm, tcp_offset_local, calibration_ref, support_ref, max_total_steps, distance=.10,
                    lift=.10, max_stage_steps=40, grip_command=None):
    """Open, withdraw `distance` along the negative tool axis, then lift `lift`, keeping the orientation.

    Lower onto observed support and inspect it first. Opening does not prove release; it stays 'unknown'.
    """
    if not support_ref or not np.isfinite([distance, lift]).all() or min(distance, lift) <= 0:
        raise ValueError('support evidence and positive finite retreat distances required')
    session = pour_session(arm, tcp_offset_local, calibration_ref, max_total_steps, max_stage_steps)
    initial = pour_tcp(session)
    tool = initial[:3, :3] @ np.asarray(TOOL_AXIS, float)
    if tool[2] > .01:
        raise ValueError('upward_approach would retreat downward into support')
    opened = set_gripper(arm, 1.)
    session['calls'] += int(opened.get('calls') or 0)
    if opened['status'] == 'episode_over':
        return {'status': 'episode_over', 'released': 'unknown', 'gripper': opened, 'calls': session['calls']}
    target = initial.copy()
    target[:3, 3] -= distance * tool
    pour_move(session, 'withdraw', target)
    target = pour_tcp(session)
    target[:3, 3] += [0., 0., lift]
    pour_move(session, 'lift', target)
    return {'status': 'tracking_reached', 'released': 'unknown', 'support_ref': support_ref, 'gripper': opened,
            'stages': session['stages'], 'calls': session['calls']}
