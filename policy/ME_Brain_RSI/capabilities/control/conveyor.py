"""Remember a whole RGB object, match its duplicate, and servo a CAD grasp."""
import math
import numpy as np
from ..motion import down_quat, quat_multiply, quat_matrix, tcp_to_link
from ..robot_collision import PublicSphereCheck
from .conveyor_vision import (BELT_Z, ConveyorTracker, appearance_distance,
                              public_models, _FULL_MODELS)

# 18+14 preparation/arrival, 22+12 descent/arrival, 16 closing, 32+15
# camera lift/arrival, 8 settling, and 20 duplicate re-observation frames.
_RECOVERY_ATTEMPT_RESERVE = 157


def _record(ctx, name, **data):
    record = dict(stage=name, step=int(ctx.steps))
    for key, value in data.items():
        record[key] = value.tolist() if isinstance(value, np.ndarray) else value
    ctx.memory.setdefault('task_events', []).append(record)
    ctx.stage = name


def _tilt(degrees, yaw=math.pi/2):
    angle = math.radians(degrees)
    return quat_multiply([math.cos(yaw/2), 0, 0, math.sin(yaw/2)],
                         [math.cos(angle/2), 0, math.sin(angle/2), 0])


def _grasp_geometry(geometry):
    """Grasp points come from public shape sections, not sampled object poses."""
    info, _ = public_models()
    key, family = geometry['model_key'], geometry['family']
    cloud = _FULL_MODELS[key]
    bounds = np.asarray(geometry['bounds'], float)
    local = (bounds[0]+bounds[1])*.5
    yaw = geometry['yaw']
    height = geometry['height']
    if family in ('phone', 'game_machine'):
        local[2] = .022
        headings = [yaw+math.pi/2, yaw-math.pi/2] if family == 'phone' else [0., math.pi/2]
        quats = [down_quat(value) for value in headings]
        side_approach = False
    elif family == 'plate':
        local[1] = bounds[0, 1]+.016
        local[2] = .023
        quats = [down_quat(yaw), down_quat(yaw+math.pi)]
        side_approach = False
    elif family == 'garage':
        # A central head section avoids the wide base and outstretched arms.
        band = cloud[abs(cloud[:, 2]-height*.88) < .007]
        central = band[(abs(band[:, 0]) < .045) & (abs(band[:, 1]) < .03)]
        if len(central):
            local[:2] = (central[:, :2].min(0)+central[:, :2].max(0))*.5
        local[2] = height*.88+.006
        quats = [_tilt(60)]
        side_approach = True
    elif family == 'ice_cream':
        # Keep the upper finger corners below the widening cup rim. The public
        # finger/CAD sweep at 60 degrees and 45 mm intersects that rim while
        # still fully open; this lower 45-degree pinch clears the approach.
        local[:2] = 0.
        local[2] = .025
        quats = [_tilt(45)]
        side_approach = True
    elif family == 'camera':
        lens = cloud[(cloud[:, 1] < -.080) & (cloud[:, 2] < .065)]
        if len(lens):
            local[:2] = (lens[:, :2].min(0)+lens[:, :2].max(0))*.5
        local[2] = .040
        quats = [down_quat(yaw+math.pi/2), down_quat(yaw-math.pi/2)]
        side_approach = False
    elif family == 'bottle' and info[key]['placement'] == 'up':
        local[:2] = 0.
        local[2] = .080
        quats = [_tilt(60)]
        side_approach = True
    elif family == 'puppet':
        # The long bounding box includes the thin tail. The upper public mesh
        # section isolates the torso, which is displaced from that box centre.
        torso = cloud[cloud[:, 2] > height*.66]
        local[:2] = (torso[:, :2].min(0)+torso[:, :2].max(0))*.5
        local[2] = height*.60
        quats = [down_quat(yaw+math.pi/2), down_quat(yaw-math.pi/2)]
        side_approach = False
    else:
        local[2] = max(.024, height*.5+.006)
        quats = [down_quat(yaw+math.pi/2), down_quat(yaw-math.pi/2)]
        side_approach = False
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    clearance = .035 if side_approach else max(.065, height-local[2]+.025)
    return geometry['position'] + rotation @ local, quats, side_approach, clearance


def _lift_distance(geometry):
    if geometry['family'] != 'camera':
        return .145
    # A front-lens grasp lets the actor frame swing below the TCP. The public
    # cloud is normalized to its support plane; the native rigid actor frame
    # remains at -raw_floor. Bound that frame using the public CAD transform
    # and contact uncertainty, without reading any runtime object pose.
    info, _ = public_models()
    anchor, _, _, _ = _grasp_geometry(geometry)
    angle = geometry['yaw']
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    floor = float(info[geometry['model_key']]['raw_floor'])
    actor = geometry['position'] + rotation @ np.array([0., 0., -floor])
    radius = float(np.linalg.norm(actor-anchor))
    initial_actor_height = BELT_Z-floor
    return max(.145, initial_actor_height+.120+radius+.0444-anchor[2])


def _side_approach_offsets(geometry):
    if geometry['family'] == 'ice_cream':
        # The former 120 mm frontal offset sends the tilted wrist into the
        # fixed public conveyor frame for near-side cups. Use the last half
        # of that same local sweep: preserve its 35/120 vertical slope, tool
        # rotation, cup stem contact and measured-arrival tolerances.
        return .060, .0175
    return .120, .035


def _preflight(ctx, arm, contact, quats, side_approach, velocity, clearance=.12,
               future_side_approach=False, lift_height=.145, camera_lift=False,
               return_cost=False, side_offset=.12, side_rise=.035):
    choices = []
    collision = PublicSphereCheck()
    initial = np.asarray(ctx.state[arm+'_arm_joint_state'], float)
    for quat in quats:
        approach = contact.copy()
        if not future_side_approach:
            approach[:2] -= velocity*(22/25.)
        if side_approach:
            approach[1] -= side_offset
            approach[2] += side_rise
        else:
            approach[2] += clearance
        closed = contact.copy()
        closed[:2] += velocity*.64
        lift_quats = ([quat, _tilt(60), _tilt(45), _tilt(30)] if camera_lift else
                      [quat, _tilt(30)] if side_approach else [quat])
        for lift_quat in lift_quats:
            lift = closed.copy()
            lift[1] = min(lift[1]-.10, -.22)
            lift[2] += lift_height
            reference = initial.copy()
            cost = 0.
            feasible = True
            for position, rotation in [(approach, quat), (contact, quat), (closed, quat), (lift, lift_quat)]:
                result = ctx.kinematics.solve(arm, tcp_to_link(position, rotation), reference,
                                             position_tol=.002, rotation_tol=.02, max_starts=12, diagnostic_candidates=True)
                if result['status'] != 'Success':
                    feasible = False
                    break
                branches = []
                for branch in result['feasible_branches']:
                    candidate = np.asarray(branch['joint_value'], float)
                    gaps = [collision.check(ctx.kinematics, arm, reference*(1-t)+candidate*t)['self_gap']
                            for t in np.linspace(0, 1, 9)]
                    if min(gaps) > -.001:
                        branches.append(candidate)
                if not branches:
                    feasible = False
                    break
                joints = min(branches, key=lambda candidate: np.linalg.norm(candidate-reference))
                cost += np.linalg.norm(joints-reference)
                reference = joints
            if feasible:
                choices.append((cost, quat, lift_quat))
    if not choices:
        return None
    cost, quat, lift_quat = min(choices, key=lambda value: value[0])
    return (float(cost), quat, lift_quat) if return_cost else (quat, lift_quat)


def _intercept_arm(ctx, geometry, contact, quats, side, velocity, clearance):
    """Choose the shorter feasible arm route for the low frontal cup grasp.

    A moving object can currently be left of the centre while its future lift
    point is right of it. Selecting the arm from that last point alone sends
    the right hand across its base during the first approach. Public IK checks
    already compute the whole route's joint travel; compare those two routes
    instead of discarding the second arm after the first feasible endpoint.
    This is a planning hint, and the existing measured-arrival guards remain.
    """
    preference = ctx.arm_for(contact[:2]+velocity*.64)
    arms = (preference, 'right' if preference == 'left' else 'left')
    side_offset, side_rise = _side_approach_offsets(geometry)
    choices = []
    for arm in arms:
        result = _preflight(ctx, arm, contact, quats, side, velocity, clearance,
                            lift_height=_lift_distance(geometry),
                            camera_lift=geometry['family'] == 'camera',
                            return_cost=True, side_offset=side_offset,
                            side_rise=side_rise)
        if result is not None:
            cost, quat, lift_quat = result
            choices.append((cost, arm, (quat, lift_quat)))
            if geometry['family'] != 'ice_cream':
                # Retain the proven arm choice for other grasp families.
                break
    if not choices:
        return None
    cost, arm, rotations = min(choices, key=lambda value: value[0])
    _record(ctx, 'conveyor_intercept_arm_plan', arm=arm, joint_travel=cost,
            candidates=[dict(arm=row[1], joint_travel=row[0]) for row in choices])
    return arm, rotations


def _pose_error(ctx, arm, position, quat):
    measured = np.asarray(ctx.state[arm+'_ee_pose'], float)
    angular = 2*math.acos(np.clip(abs(np.dot(measured[3:], quat)), 0., 1.))
    return float(np.linalg.norm(ctx.tcp(arm)-position)), angular


def _geometry_consistent(tracker, measured):
    """Reject merged silhouettes before interpreting them as the remembered CAD.

    The reference provides a same-camera quality baseline. A loose forced
    single-template fit can otherwise turn adjacent unrelated objects into the
    reference class even when their material descriptor is only a weak match.
    """
    if measured is None:
        return False
    if measured.get('identity_confuser_rejected', False):
        return False
    if measured['model_key'] == 'camera_00002_up':
        # This asymmetric asset has a mirrored convex-silhouette minimum.
        # Its measured public lens-ring landmark resolves that ambiguity;
        # the shadow-contaminated whole-body error is still bounded separately.
        return (measured.get('landmark_kind') == 'public_camera_red_ring' and
                measured['landmark_error'] <= 1.5 and measured['fit_error'] <= 5. and
                measured['silhouette_iou'] >= .50)
    reference = tracker.reference_model
    error_limit = max(3., float(reference['fit_error'])+.5)
    iou_limit = max(.30, float(reference['silhouette_iou'])-.18)
    return measured['fit_error'] <= error_limit and measured['silhouette_iou'] >= iou_limit


def _cad_velocity(first, second, elapsed):
    """Measure object motion from two RGB CAD roots at the same placement.

    Reusing the reference's centroid depth scales apparent motion incorrectly
    when its duplicate is upright. Whole-object pose measurements instead
    compare the same public root frame, with lateral and orientation checks.
    """
    if elapsed < .6 or first['model_key'] != second['model_key']:
        return None
    displacement = second['position'][:2]-first['position'][:2]
    velocity = displacement/elapsed
    angle = abs(math.atan2(math.sin(second['yaw']-first['yaw']),
                          math.cos(second['yaw']-first['yaw'])))
    if first['family'] in ('camera', 'puppet') and angle > .5:
        return None
    if abs(displacement[1]) > .020 or not .025 < velocity[0] < .25:
        return None
    return np.array([float(velocity[0]), 0.])


def _gripper_contact_present(ctx, arm):
    """A necessary contact signal, never proof of the correct object's identity."""
    measured = float(ctx.state[arm+'_ee_joint_state'][0])
    command = float(ctx.gripper_targets[arm])
    return command <= .001 and measured > .008


def _identity_midpoint_veto(ctx, tracker, item, previous, checked, negative):
    """Check the unchanged cup/plate identity guard once between CAD roots.

    A raster phase between the two motion samples can give much clearer
    alternative-CAD evidence. Keep that strong negative identity evidence for
    the same public RGB track instead of accepting its next ambiguous fit.
    This adds no robot actions and at most one fit per 20-frame sample pair.
    """
    if (tracker.reference_model['family'] != 'ice_cream' or previous is None or
            not 10 <= ctx.steps-previous['step'] < 20):
        return False
    token = (item['id'], previous['step'])
    if token in checked:
        return False
    checked.add(token)
    item['geometry_yaw_hint'] = previous['geometry']['yaw']
    measured = tracker.geometry(ctx, item)
    if measured is None or not measured.get('identity_confuser_rejected', False):
        return False
    negative.add(item['id'])
    _record(ctx, 'conveyor_identity_midpoint_rejected', target_id=item['id'],
            sample_step=previous['step'],
            identity_confuser=measured.get('identity_confuser_diagnostic'))
    return True


def _recover_visible_geometry(ctx, tracker, anchor, anchor_step, velocity):
    """Re-observe failed grasps, requiring two fresh accepted CAD measurements.

    Keep the original geometry and measured-motion checks. Reserve the full
    worst-case grasp/lift/re-observation budget (157 frames) inside the public
    700-step cap;
    avoid spending that remaining time attempting a stale or merged contour.
    """
    deadline = min(int(ctx.steps)+32, 700-_RECOVERY_ATTEMPT_RESERVE)
    samples = {}
    while ctx.steps <= deadline:
        visible = tracker.observe(ctx)
        coast = anchor[:2]+velocity*((ctx.steps-anchor_step)/25.)
        choices = [item for item in tracker.matching(visible)
                   if abs(item['position'][0]) < .6 and
                   np.linalg.norm(item['position'][:2]-coast) < .24]
        for item in sorted(choices, key=lambda value: appearance_distance(
                tracker.reference, value['descriptor'])):
            previous = samples.get(item['id'])
            if previous is not None and ctx.steps-previous['step'] < 20:
                continue
            if previous is not None:
                item['geometry_yaw_hint'] = previous['geometry']['yaw']
            measured = tracker.geometry(ctx, item)
            if not _geometry_consistent(tracker, measured):
                _record(ctx, 'conveyor_recovery_reject_geometry', target_id=item['id'],
                        identity_confuser=None if measured is None else measured.get(
                            'identity_confuser_diagnostic'))
                continue
            if previous is None:
                samples[item['id']] = dict(step=int(ctx.steps), geometry=measured)
                continue
            fresh_velocity = _cad_velocity(previous['geometry'], measured,
                                           (ctx.steps-previous['step'])/25.)
            if fresh_velocity is not None:
                item['cad_velocity'] = fresh_velocity
                _record(ctx, 'conveyor_recovery_measured_duplicate', target_id=item['id'],
                        measured_velocity=fresh_velocity, elapsed=(ctx.steps-previous['step'])/25.)
                return item, measured, fresh_velocity
            samples[item['id']] = dict(step=int(ctx.steps), geometry=measured)
        if ctx.steps+4 > deadline:
            break
        yield from ctx.hold(4)
    _record(ctx, 'conveyor_recovery_no_measured_duplicate', remaining_frames=700-int(ctx.steps))
    return None


def match_and_pick_from_conveyor(ctx):
    ctx.table_z = BELT_Z
    frequency = 25.
    tracker = ConveyorTracker(frequency)
    ctx.memory['conveyor'] = dict(reference_observed=False, plane_z=BELT_Z)
    _record(ctx, 'conveyor_observe_reference')
    visible = tracker.observe(ctx)
    selected = geometry = chosen = None
    rejected = {}
    samples = {}
    identity_checked = set()
    identity_negative = set()
    while ctx.steps < 560:
        # A thin coloured ring can disappear for one raster phase. Sample it
        # more densely without changing the physical time budget for waiting.
        camera_reference = (tracker.reference_model is not None and
                            tracker.reference_model['family'] == 'camera')
        yield from ctx.hold(2 if camera_reference and tracker.reference_gone else 4)
        visible = tracker.observe(ctx)
        if tracker.reference is None:
            continue
        ctx.memory['conveyor'].update(reference_observed=True, reference_id=tracker.reference_id,
                                      reference_rgb=tracker.reference['rgb'].tolist(),
                                      measured_velocity=tracker.velocity.tolist())
        if not tracker.reference_gone:
            continue
        width = next(iter(ctx.vision.values()))['color'].shape[1]
        candidates = [item for item in tracker.matching(visible) if
                      item['id'] not in identity_negative and
                      12 < item['bbox'][0] and item['bbox'][0]+item['bbox'][2] < width-12 and
                      -.66 < item['position'][0] < .25 and
                      ctx.steps-rejected.get(item['id'], -1000) > (1 if camera_reference else 20)]
        for item in candidates:
            previous = samples.get(item['id'])
            if previous is not None and ctx.steps-previous['step'] < 20:
                if _identity_midpoint_veto(ctx, tracker, item, previous,
                                           identity_checked, identity_negative):
                    samples.pop(item['id'], None)
                continue
            if previous is not None:
                item['geometry_yaw_hint'] = previous['geometry']['yaw']
            measured = tracker.geometry(ctx, item)
            if not _geometry_consistent(tracker, measured):
                if measured is not None and measured.get('identity_confuser_rejected', False):
                    identity_negative.add(item['id'])
                _record(ctx, 'conveyor_reject_geometry', target_id=item['id'],
                        fit_error=None if measured is None else measured['fit_error'],
                        silhouette_iou=None if measured is None else measured['silhouette_iou'],
                        landmark_diagnostic=None if measured is None else measured.get('landmark_diagnostic'),
                        identity_confuser=None if measured is None else measured.get('identity_confuser_diagnostic'))
                rejected[item['id']] = ctx.steps
                # Retain a good landmark sample through temporary glare or a
                # foreground-mask dropout, then measure a longer time baseline.
                if not camera_reference or previous is None or ctx.steps-previous['step'] > 60:
                    samples.pop(item['id'], None)
                continue
            if previous is None:
                samples[item['id']] = dict(step=int(ctx.steps), geometry=measured)
                _record(ctx, 'conveyor_sample_motion', target_id=item['id'],
                        model=measured['model_key'], position=measured['position'])
                continue
            elapsed = (ctx.steps-previous['step'])/frequency
            velocity = _cad_velocity(previous['geometry'], measured, elapsed)
            if velocity is None:
                _record(ctx, 'conveyor_reject_motion', target_id=item['id'], elapsed=elapsed,
                        old_position=previous['geometry']['position'], position=measured['position'])
                samples[item['id']] = dict(step=int(ctx.steps), geometry=measured)
                continue
            item['cad_velocity'] = velocity
            tracker.belt_velocity = velocity.copy()
            _record(ctx, 'conveyor_measured_cad_velocity', target_id=item['id'],
                    elapsed=elapsed, belt_velocity=velocity,
                    landmark_error=measured.get('landmark_error'))
            anchor, quats, side, clearance = _grasp_geometry(measured)
            contact = anchor.copy()
            contact[:2] += velocity*(40/frequency)
            chosen = _intercept_arm(ctx, measured, contact, quats, side, velocity, clearance)
            if chosen is None:
                _record(ctx, 'conveyor_wait_reachable_intercept', target_id=item['id'], contact=contact)
                rejected[item['id']] = ctx.steps
                continue
            selected, geometry = item, measured
            break
        if selected is not None:
            break
    if selected is None:
        raise RuntimeError('No whole-object duplicate reached a measured interception region')
    locked_velocity = tracker.intercept_velocity(selected)
    for attempt in range(3):
        if attempt and ctx.steps > 700-_RECOVERY_ATTEMPT_RESERVE:
            _record(ctx, 'conveyor_retry_budget_exhausted', remaining_frames=700-int(ctx.steps))
            break
        arm = chosen[0] if attempt == 0 else ctx.arm_for(selected['position'])
        velocity = locked_velocity.copy()
        anchor, quats, side, clearance = _grasp_geometry(geometry)
        side_offset, side_rise = _side_approach_offsets(geometry)
        lift_height = _lift_distance(geometry)
        anchor_step = int(ctx.steps)
        prepare_frames, descend_frames = 18, 22
        contact = anchor.copy()
        contact[:2] += velocity*((prepare_frames+descend_frames)/frequency)
        _record(ctx, 'conveyor_grasp_preflight', target_id=selected['id'], model=geometry['model_key'],
                geometry_position=geometry['position'], geometry_yaw=geometry['yaw'],
                fit_error=geometry['fit_error'], silhouette_iou=geometry['silhouette_iou'],
                landmark_error=geometry.get('landmark_error'),
                anchor=anchor, contact=contact, belt_velocity=velocity, lift_height=lift_height)
        # With a directly measured CAD speed, keep the head centred in the
        # open jaws during the frontal approach as well as a downward grasp.
        # An early fixed future-X finger can strike a wide head before closing.
        future_side = False
        rotations = chosen[1] if attempt == 0 else _preflight(
            ctx, arm, contact, quats, side, velocity, clearance, future_side_approach=future_side,
            lift_height=lift_height, camera_lift=geometry['family'] == 'camera',
            side_offset=side_offset, side_rise=side_rise)
        if rotations is None:
            raise RuntimeError('Public IK rejected the measured conveyor grasp and lift endpoints')
        quat, lift_quat = rotations
        approach = contact.copy()
        if not future_side:
            approach[:2] = anchor[:2]+velocity*(prepare_frames/frequency)
        if side:
            approach[1] -= side_offset
            approach[2] += side_rise
        else:
            approach[2] += clearance
        _record(ctx, 'conveyor_locked_intercept', target_id=selected['id'], model=geometry['model_key'],
                anchor=anchor, anchor_step=anchor_step, belt_velocity=velocity, target=contact)
        yield from ctx.move(arm, approach, quat, 1., duration=prepare_frames)
        for _ in range(14):
            error, angle = _pose_error(ctx, arm, approach, quat)
            if error < .009 and angle < .10:
                break
            remaining_descent = descend_frames if future_side else 1
            approach[:2] = anchor[:2]+velocity*((ctx.steps+remaining_descent-anchor_step)/frequency)
            if side:
                approach[1] -= side_offset
            yield from ctx.move(arm, approach, quat, 1., duration=1)
        error, angle = _pose_error(ctx, arm, approach, quat)
        _record(ctx, 'conveyor_approach_feedback', position_error=error, angular_error=angle)
        if error > .016 or angle > .16:
            raise RuntimeError('Measured conveyor approach did not reach its target')
        if future_side:
            # This frontal interception path has real successful garage
            # grasps. Keep its future-X approach while improving RGB speed.
            contact[:2] = anchor[:2]+velocity*((ctx.steps+descend_frames-anchor_step)/frequency)
            yield from ctx.move(arm, contact, quat, 1., duration=descend_frames)
        else:
            # Downward and can approaches follow the object throughout descent
            # to keep a leading finger from striking it while still open.
            current_anchor = anchor.copy()
            current_anchor[:2] += velocity*((ctx.steps-anchor_step)/frequency)
            relative = ctx.tcp(arm)-current_anchor
            _record(ctx, 'conveyor_track_descent', relative_offset=relative)
            for index in range(descend_frames):
                contact = anchor.copy()
                contact[:2] += velocity*((ctx.steps+1-anchor_step)/frequency)
                tracked = contact+relative*(1-(index+1)/descend_frames)
                yield from ctx.move(arm, tracked, quat, 1., duration=1)
        for _ in range(12):
            contact[:2] = anchor[:2]+velocity*((ctx.steps+1-anchor_step)/frequency)
            error, angle = _pose_error(ctx, arm, contact, quat)
            if error < .008 and angle < .10:
                break
            yield from ctx.move(arm, contact, quat, 1., duration=1)
        error, angle = _pose_error(ctx, arm, contact, quat)
        _record(ctx, 'conveyor_close', attempt=attempt, target=contact, position_error=error, angular_error=angle)
        if error > .020 or angle > .16:
            raise RuntimeError('Measured conveyor contact did not reach its target')
        for index in range(16):
            contact[:2] = anchor[:2]+velocity*((ctx.steps+1-anchor_step)/frequency)
            yield from ctx.move(arm, contact, quat, max(0., 1-(index+1)/14), duration=1)
        lift = ctx.tcp(arm).copy()
        lift[1] = min(lift[1]-.10, -.22)
        lift[2] += lift_height
        _record(ctx, 'conveyor_lift', target=lift)
        yield from ctx.move(arm, lift, lift_quat, .0, duration=32 if geometry['family'] == 'camera' else 22)
        for _ in range(15):
            error, angle = _pose_error(ctx, arm, lift, lift_quat)
            if error < .012 and angle < .12:
                break
            yield from ctx.move(arm, lift, lift_quat, .0, duration=1)
        _record(ctx, 'conveyor_lift_feedback', position_error=error, angular_error=angle)
        yield from ctx.hold(8)
        contact_present = _gripper_contact_present(ctx, arm)
        _record(ctx, 'conveyor_held_contact_check', arm=arm,
                measured_gripper=float(ctx.state[arm+'_ee_joint_state'][0]),
                contact_present=bool(contact_present))
        visible = tracker.observe(ctx)
        coast = anchor[:2]+velocity*((ctx.steps-anchor_step)/frequency)
        remaining = [item for item in tracker.matching(visible)
                     if np.linalg.norm(item['position'][:2]-coast) < .18 and abs(item['position'][0]) < .5]
        if not remaining:
            for _ in range(5):
                if ctx.steps+4 > 700:
                    break
                yield from ctx.hold(4)
                visible = tracker.observe(ctx)
                remaining = [item for item in tracker.matching(visible) if abs(item['position'][0]) < .5]
                if remaining:
                    break
        if not remaining:
            if contact_present:
                _record(ctx, 'conveyor_contact_present_no_belt_duplicate')
                if ctx.steps < 700:
                    yield from ctx.hold(min(35, 700-int(ctx.steps)))
                return
            # Disappearance can mean leaving the image or passing behind the
            # hand. An observed fully closed jaw is empty, so keep looking for
            # accepted RGB geometry rather than terminating this attempt.
            _record(ctx, 'conveyor_empty_after_lift')
            recovery = yield from _recover_visible_geometry(
                ctx, tracker, anchor, anchor_step, velocity)
            if recovery is None:
                break
            selected, geometry, locked_velocity = recovery
            continue
        selected = min(remaining, key=lambda item: appearance_distance(tracker.reference, item['descriptor']))
        geometry = tracker.geometry(ctx, selected)
        if not _geometry_consistent(tracker, geometry):
            recovery = yield from _recover_visible_geometry(
                ctx, tracker, anchor, anchor_step, velocity)
            if recovery is None:
                break
            selected, geometry, locked_velocity = recovery
        _record(ctx, 'conveyor_retry_visible_duplicate', target_id=selected['id'])
    if ctx.steps < 700:
        yield from ctx.hold(min(35, 700-int(ctx.steps)))
