# Copyright © 2026 Martingale. All Rights Reserved.
import math
import json
from martingale.inference.extensions.key.key_insertion import _rotation_error_degrees
PROGRESS = ('clear', 'progress', 'stalled', 'slipped', 'uncertain')
LANDMARKS = ('tip_pixel', 'shaft_pixel', 'rim_pixel')

def _vector(value, length, name):
    try:
        result = tuple((float(v) for v in value))
    except (TypeError, ValueError):
        raise ValueError(name + ' must contain finite numbers') from None
    if len(result) != length or not all((math.isfinite(v) for v in result)):
        raise ValueError(name + ' must contain ' + str(length) + ' finite numbers')
    return result

def _pose(hand):
    return (tuple(hand.tcp_pos), tuple(hand.eef_quat))

def _changed(a, b):
    return math.dist(a[0], b[0]) > 0.0005 or _rotation_error_degrees(a[1], b[1]) > 0.5

def _evidence(value, observed_step):
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError('key_evidence must be a mapping or None')
    if 'step' in value and (type(value['step']) is not int or value['step'] != observed_step):
        raise ValueError('key_evidence.step must match observed_step')
    camera = value.get('camera')
    if camera not in (None, 'cam_left_wrist', 'cam_right_wrist', 'cam_head'):
        raise ValueError('key_evidence.camera must name a current camera')
    tracking = value.get('tracking', 'tip')
    if tracking not in ('tip', 'shaft'):
        raise ValueError('tracking must be tip or shaft')
    entry = value.get('entry_confirmed', False)
    if type(entry) is not bool:
        raise ValueError('entry_confirmed must be boolean')
    landmark = value.get('landmark_id', 'terminal_tip' if tracking == 'tip' else 'proximal_shaft')
    if not isinstance(landmark, str) or not landmark.strip():
        raise ValueError('landmark_id must identify the same physical feature across images')
    result = dict(step=observed_step, camera=camera, tracking=tracking, landmark_id=landmark, entry_confirmed=entry)
    rim_id = value.get('rim_landmark_id')
    if rim_id is not None and (not isinstance(rim_id, str) or not rim_id.strip()):
        raise ValueError('rim_landmark_id must identify the same aperture feature')
    result['rim_landmark_id'] = rim_id
    crossing = value.get('entry_observation')
    if crossing is not None:
        if not isinstance(crossing, dict) or type(crossing.get('step')) is not int or crossing.get('camera') not in ('cam_left_wrist', 'cam_right_wrist', 'cam_head') or (not isinstance(crossing.get('description'), str)) or (not crossing['description'].strip()):
            raise ValueError('entry_observation needs image step, camera and a description of observed crossing')
        result['entry_observation'] = dict(crossing)
    for key in LANDMARKS:
        if value.get(key) is not None:
            result[key] = list(_vector(value[key], 2, key))
    return result

def _evidence_status(robot, evidence):
    problems = []
    camera = evidence['camera']
    feature = 'tip_pixel' if evidence['tracking'] == 'tip' else 'shaft_pixel'
    if camera is None:
        problems.append('camera_not_supplied')
    if feature not in evidence or 'rim_pixel' not in evidence:
        problems.append('landmarks_unavailable')
    if evidence['tracking'] == 'shaft' and (not evidence['entry_confirmed']):
        problems.append('shaft_tracking_entry_unconfirmed')
    shape = None
    getter = getattr(robot, 'camera_meta', None)
    if getter is not None and camera is not None and any((k in evidence for k in LANDMARKS)):
        try:
            shape = getter(camera).get('shape')
        except (ValueError, KeyError, TypeError, AttributeError):
            problems.append('image_bounds_unavailable')
    for key in LANDMARKS:
        if key not in evidence:
            continue
        x, y = evidence[key]
        if x < 0 or y < 0 or (shape is not None and (x >= shape[1] or y >= shape[0])):
            problems.append(key + '_outside_image')
    return (not problems, problems)

def _image_change(previous, current):
    if any((previous.get(key) != current.get(key) for key in ('camera', 'tracking', 'landmark_id', 'rim_landmark_id'))):
        return None
    feature = 'tip_pixel' if current['tracking'] == 'tip' else 'shaft_pixel'
    if not all((k in previous and k in current for k in (feature, 'rim_pixel'))):
        return None
    before = [a - b for a, b in zip(previous[feature], previous['rim_pixel'])]
    after = [a - b for a, b in zip(current[feature], current['rim_pixel'])]
    return dict(displacement_px=math.dist(before, after), delta_px=[a - b for a, b in zip(after, before)], before_error_xy_px=before, after_error_xy_px=after, before_error_px=math.hypot(*before), after_error_px=math.hypot(*after), tracking=current['tracking'], landmark_id=current['landmark_id'])

def _correction_feedback(old, current, change):
    feedback = dict(status='uncomparable', metric='projected_tip_to_entry_distance', certifies_3d_alignment=False)
    if not old or old['mode'] != 'correction':
        return dict(feedback, reason='No preceding lateral correction to evaluate.')
    motion = old.get('motion', {})
    if motion.get('achieved_fraction', 0.0) < 0.5 or motion.get('along_command_m', 0.0) < 0.0001 or old.get('motion_reason') not in ('inspect_key_progress', 'key_progress_concern'):
        return dict(feedback, status='insufficient_motion', reason='The commanded correction did not track reliably; do not count it as recentering.')
    if change is None or current['tracking'] != 'tip' or (not current.get('rim_landmark_id')) or (not old.get('tip_clear')) or (motion.get('rotation_error_deg', math.inf) > 0.5):
        return dict(feedback, reason='Need comparable visible tip/entry identities after a clear lateral correction.')
    improvement = change['before_error_px'] - change['after_error_px']
    status = 'improved' if improvement > 1.0 else 'worsened' if improvement < -1.0 else 'unchanged'
    return dict(feedback, status=status, previous_error_px=change['before_error_px'], current_error_px=change['after_error_px'], improvement_px=improvement, reason='A projection change only; perspective, annotation error and depth remain uncertain.')

def step(robot, delta_xyz, *, arm, observed_step, observation, normal_xyz, key_progress, key_evidence, tip_clear=False):
    """Command one <=2 mm move; diagnostics advise the next bounded action."""
    if arm not in ('left', 'right'):
        raise ValueError('arm must be left or right')
    delta = _vector(delta_xyz, 3, 'delta_xyz')
    distance = math.sqrt(sum((v * v for v in delta)))
    if not 0 < distance <= 0.002 + 1e-12:
        raise ValueError('Each visual step must command more than zero and at most 2 mm')
    normal = _vector(normal_xyz, 3, 'normal_xyz')
    if abs(math.sqrt(sum((v * v for v in normal))) - 1.0) > 1e-06:
        raise ValueError('normal_xyz must be the unit outward opening normal')
    axial = sum((d * n for d, n in zip(delta, normal)))
    mode = 'correction' if abs(axial) <= 0.05 * distance else 'withdraw' if axial > 0 else 'feed'
    if not isinstance(observation, str) or not observation.strip():
        raise ValueError('Describe current key/rim progress, retention and the intended move')
    if key_progress not in PROGRESS or type(tip_clear) is not bool:
        raise ValueError('key_progress must be clear/progress/stalled/slipped/uncertain; tip_clear is boolean')
    evidence = _evidence(key_evidence, observed_step)
    state = robot.state()
    common = dict(steps_used=0, inspection_required=True, entry_complete=False, observation=observation, observed_step=observed_step, mode=mode, key_progress=key_progress, key_evidence=evidence, advisory_only=True, recovery_required=False)
    if type(observed_step) is not int or observed_step != state.step:
        return dict(common, started=False, reason='stale_observation')
    if state.truncated:
        return dict(common, started=False, reason='episode_ended')
    hand = state.arm(arm)
    if not math.isfinite(hand.gripper) or hand.gripper >= 0.185:
        return dict(common, started=False, reason='gripper_not_closed')
    start = _vector(hand.tcp_pos, 3, 'holder TCP')
    holder_pose = _pose(hand)
    poses = {name: _pose(state.arm(name)) for name in ('left', 'right')}
    if not all((math.isfinite(v) for v in (hand.pitch, hand.yaw, hand.roll))) or not math.isfinite(_rotation_error_degrees(hand.eef_quat, hand.eef_quat)):
        return dict(common, started=False, reason='invalid_pose')
    evidence_available, evidence_problems = _evidence_status(robot, evidence)
    history = getattr(robot, '_key_visual_feedback_history', None)
    if history is None:
        history = {}
        robot._key_visual_feedback_history = history
    old = history.get(arm)
    changed = bool(old and (state.step != old['end_step'] or any((_changed(old['end_poses'][name], poses[name]) for name in poses)) or math.dist(old['normal'], normal) > 1e-06))
    image_change = None
    if old and (not changed) and old['evidence_available'] and evidence_available:
        image_change = _image_change(old['evidence'], evidence)
    correction = _correction_feedback(old, evidence, image_change)
    flags = list(evidence_problems)
    if correction['status'] in ('worsened', 'unchanged', 'insufficient_motion'):
        flags.append('correction_' + correction['status'])
    if old and old['mode'] == 'correction' and (mode == 'feed') and (correction['status'] != 'improved'):
        flags.append('centering_improvement_not_established')
    if changed:
        flags.append('external_motion_invalidates_image_comparison')
    if evidence['camera'] == 'cam_' + arm + '_wrist':
        flags.append('holder_camera_moves_use_relative_landmarks_and_complementary_view')
    if key_progress in ('stalled', 'slipped', 'uncertain'):
        flags.append('visible_key_' + key_progress)
    if old and old['mode'] == 'feed' and (key_progress == 'progress') and (image_change is not None) and (image_change['displacement_px'] < 0.25):
        flags.append('visible_key_progress_unresolved')
    if mode == 'correction' and (not tip_clear):
        flags.append('clear_contact_before_lateral_correction')
    old_entry = (old or {}).get('entry_assessment', {})
    doubt = bool(old_entry.get('recheck_required'))
    doubt_step = old_entry.get('last_contradiction_step', -1)
    reported = bool(old_entry.get('status') == 'reported_unverified')
    crossing = evidence.get('entry_observation')
    fresh_crossing = bool(crossing and crossing['step'] == observed_step and (crossing['step'] > doubt_step) and evidence['entry_confirmed'] and (key_progress not in ('stalled', 'slipped', 'uncertain')))
    if fresh_crossing:
        doubt, reported = (False, True)
    elif evidence['entry_confirmed'] and (not reported):
        flags.append('entry_report_without_fresh_crossing_evidence')
    if key_progress in ('stalled', 'slipped') or 'visible_key_progress_unresolved' in flags:
        doubt, doubt_step = (True, observed_step)
    if tip_clear and mode in ('withdraw', 'correction'):
        reported = False
    if doubt:
        flags.append('reconsider_entry_before_seating')
    common.update(evidence_available=evidence_available, image_comparison=image_change, observation_history_changed=changed, image_landmark_rim_change_px=image_change['displacement_px'] if image_change else None, image_tip_rim_change_px=image_change['displacement_px'] if image_change and image_change['tracking'] == 'tip' else None, image_comparison_from_step=old['evidence']['step'] if old else None)
    target = [p + d for p, d in zip(start, delta)]
    result = robot.move_to(target, arm=arm, gripper=robot.KEEP, path='line', pitch=hand.pitch, yaw=hand.yaw, roll=hand.roll)
    after = robot.state()
    current = after.arm(arm)
    final = tuple(current.tcp_pos)
    valid = len(final) == 3 and all((math.isfinite(v) for v in final))
    rotation = _rotation_error_degrees(holder_pose[1], current.eef_quat)
    actual = math.dist(start, final) if valid else math.inf
    error = math.dist(target, final) if valid else math.inf
    along = sum(((b - a) * d / distance for a, b, d in zip(start, final, delta))) if valid else 0.0
    ratio = along / distance
    cross = math.sqrt(max(0.0, actual * actual - along * along)) if valid else math.inf
    error_limit = max(0.00015, min(0.00075, 0.5 * distance))
    if result.truncated or after.truncated:
        reason = 'episode_ended'
    elif not valid or not math.isfinite(rotation):
        reason = 'invalid_pose'
    elif result.blocked or result.ik_error or (not result.reached):
        reason = 'blocked'
    elif rotation > 1.0:
        reason = 'orientation_drift'
    elif along < 0.0001 or ratio < 0.5:
        reason = 'contact_suspected' if mode == 'feed' else 'low_progress'
    elif error > error_limit or actual > distance + 0.0005 or cross > max(0.00025, 0.25 * distance):
        reason = 'tracking_error'
    else:
        reason = 'inspect_key_progress'
    if reason not in ('inspect_key_progress', 'episode_ended', 'blocked'):
        flags.append(reason)
    if reason == 'inspect_key_progress' and any((flag in flags for flag in ('visible_key_stalled', 'visible_key_slipped', 'visible_key_progress_unresolved'))):
        reason = 'key_progress_concern'
    contradicted = mode == 'feed' and reason in ('contact_suspected', 'tracking_error', 'orientation_drift', 'blocked')
    if contradicted:
        doubt, doubt_step = (True, after.step)
        flags.extend(['entry_assumption_contradicted', 'reconsider_entry_before_seating'])
    entry_assessment = dict(status='doubtful' if doubt else 'reported_unverified' if reported else 'unconfirmed', recheck_required=doubt, last_contradiction_step=doubt_step, independently_verified=False)
    if evidence['tracking'] == 'shaft':
        flags.append('shaft_rim_shortening_does_not_prove_entry')
    motion = dict(target_xyz=target, final_xyz=list(final), reached=result.reached, blocked=result.blocked, ik_error=result.ik_error, truncated=result.truncated, actual_travel_m=actual, along_command_m=along, achieved_fraction=ratio, cross_track_m=cross, target_error_m=error, target_error_limit_m=error_limit, rotation_error_deg=rotation)
    history[arm] = dict(end_step=after.step, end_poses={name: _pose(after.arm(name)) for name in poses}, tip_clear=tip_clear, mode=mode, normal=normal, evidence=evidence, evidence_available=evidence_available, motion=motion, motion_reason=reason, entry_assessment=entry_assessment)
    robot.show('cam_left_wrist')
    robot.show('cam_right_wrist')
    if evidence['camera'] == 'cam_head':
        robot.show('cam_head')
    concern = doubt or correction['status'] in ('worsened', 'unchanged', 'insufficient_motion') or reason in ('contact_suspected', 'tracking_error', 'key_progress_concern', 'orientation_drift')
    next_action = 'End this turn and inspect the images. If contact/slip is supported, choose one small outward withdrawal or changed alignment correction after clearance; if still progressing, continue one small feed. Do not repeat an unchanged failed hypothesis.' if concern else 'End this turn and inspect key/rim progress; choose the next small feed, correction, or withdrawal. Use a currently useful view if landmarks are unavailable.'
    if doubt:
        next_action = 'Reconsider the entry assumption. Do not label another axial push as seating from shaft-rim shortening alone. Inspect fresh images; if contact persists, unload locally and correct the visible tip/entry error at clearance. Check actual correction travel and whether the same projected error shrinks before the next approach. Inspect the existing holding-wrist and head images/crops; keep the empty arm parked and continue local recovery; do not waste actions. Do not plan or relocate cameras.'
    elif correction['status'] in ('worsened', 'unchanged', 'insufficient_motion'):
        next_action = 'The last correction did not establish improved centering. While clear, reverse a worsening direction or try a different small correction, then compare the same tip and aperture feature. Do not treat the command alone as completed recentering.'
    elif evidence['entry_confirmed'] and (not reported):
        next_action = 'Entry is unconfirmed. Tip occlusion and wrist-image gap shrinking are not a crossing observation. Continue local centering/recovery; describe actual aperture crossing in fresh images before switching from visible-tip alignment to shaft seating.'
    flags = list(dict.fromkeys(flags))
    print(json.dumps(dict(key_visual_feedback=reason, step=after.step, feedback_flags=flags, correction_feedback=correction, entry_assessment=entry_assessment, next_action=next_action)))
    return dict(common, started=True, reason=reason, steps_used=result.steps_used, feedback_flags=flags, next_observed_step=after.step, correction_feedback=correction, entry_assessment=entry_assessment, commanded_delta_xyz=list(delta), progress_concern=concern, contact_suspected=reason == 'contact_suspected' or (mode == 'feed' and 'visible_key_stalled' in flags), motion=motion, next_action=next_action, note='Diagnostics are advisory; they do not disable the next valid bounded command. Continue the task; do not waste actions. Never invent buried-tip pixels: after confirmed shallow entry track the same visible shaft feature against the rim. No automatic retry, and no certificate of entry, seating or task success.')
