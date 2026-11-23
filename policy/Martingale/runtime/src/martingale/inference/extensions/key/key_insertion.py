# Copyright © 2026 Martingale. All Rights Reserved.
import math

def _rotation_error_degrees(a, b):
    if len(a) != 4 or len(b) != 4:
        return math.inf
    scale = math.sqrt(sum((v * v for v in a)) * sum((v * v for v in b)))
    if not math.isfinite(scale) or scale <= 0:
        return math.inf
    return math.degrees(2 * math.acos(min(1.0, abs(sum((x * y for x, y in zip(a, b)))) / scale)))

def begin(robot, opening, tip_pixels, shaft_pixels, tooth_pixels, *, arm, entry_depth_m=0.001, tip_model='end_face_center'):
    if arm not in ('left', 'right'):
        raise ValueError('arm must be left or right')
    depth = float(entry_depth_m)
    if not math.isfinite(depth) or not 0.001 <= depth <= 0.002:
        raise ValueError('Entry depth must be between 1 and 2 mm; this only begins insertion.')
    normal = tuple((float(v) for v in opening['normal_xyz']))
    if len(normal) != 3 or not all((math.isfinite(v) for v in normal)) or abs(math.sqrt(sum((v * v for v in normal))) - 1.0) > 1e-06:
        raise ValueError('The measured opening normal must be a finite unit vector.')
    options = dict(arm=arm)
    if tip_model != 'end_face_center':
        options['tip_model'] = tip_model
    alignment = robot.key_insertion_alignment(opening, tip_pixels, shaft_pixels, tooth_pixels, **options)
    if not alignment.get('ready_to_insert'):
        return dict(started=False, reason='alignment_not_ready', steps_used=0, alignment=alignment, inspection_required=True, entry_complete=False)
    state = robot.state()
    if state.truncated:
        return dict(started=False, reason='episode_ended', steps_used=0, alignment=alignment, inspection_required=True, entry_complete=False)
    hand = state.arm(arm)
    if not math.isfinite(hand.gripper) or hand.gripper >= 0.185:
        return dict(started=False, reason='gripper_not_closed', steps_used=0, alignment=alignment, inspection_required=True, entry_complete=False)
    remaining = float(alignment['tip_above_rim_m']) + depth
    if not math.isfinite(remaining):
        raise ValueError('The measured tip height must be finite.')
    if remaining <= 0:
        upper_height = float((alignment.get('uncertainty') or {}).get('max_above_m', math.inf))
        if not math.isfinite(upper_height) or upper_height > -depth:
            return dict(started=False, reason='entry_depth_uncertain', steps_used=0, alignment=alignment, inspection_required=True, entry_complete=False, stage='verify_shallow_entry', requires_visual_entry_confirmation=True, next_action='Inspect fresh close views for the actual shaft entering the fixed rim and the key following the hand. Do not repeat this helper waiting for a numerical depth pass. Only after visually confirming shallow entry, continue seating in at most 2-mm feeds with fresh key-progress checks. If entry is not visible or the key stalls, improve the view or retreat straight clear before correction.', note='The estimated shallow depth is reached but its uncertainty overlaps the target. This is an inspection stage, not numerical proof of entry or seating; no further motion is commanded.')
        return dict(started=False, reason='entry_depth_reached', steps_used=0, alignment=alignment, inspection_required=True, entry_complete=True, note='The measured tip reached the shallow entry depth; this does not prove seating or authorize turning.')
    travel = min(remaining, 0.002)
    start = tuple(hand.tcp_pos)
    if len(start) != 3 or not all((math.isfinite(v) for v in start)) or (not math.isfinite(_rotation_error_degrees(hand.eef_quat, hand.eef_quat))) or (not all((math.isfinite(v) for v in (hand.pitch, hand.yaw, hand.roll)))):
        return dict(started=False, reason='invalid_pose', steps_used=0, alignment=alignment, inspection_required=True, entry_complete=False)
    target = [p - travel * n for p, n in zip(start, normal)]
    result = robot.move_to(target, arm=arm, gripper=robot.KEEP, path='line', pitch=hand.pitch, yaw=hand.yaw, roll=hand.roll)
    after = robot.state()
    current = after.arm(arm)
    final = tuple(current.tcp_pos)
    rotation_error = _rotation_error_degrees(hand.eef_quat, current.eef_quat)
    progress = sum(((a - b) * n for a, b, n in zip(start, final, normal)))
    lateral = math.sqrt(sum(((b - a + progress * n) ** 2 for a, b, n in zip(start, final, normal))))
    motion = dict(reached=result.reached, blocked=result.blocked, ik_error=result.ik_error, truncated=result.truncated, steps_used=result.steps_used, target_xyz=target, final_xyz=final, progress_m=progress, lateral_drift_m=lateral, rotation_error_deg=rotation_error)
    if result.truncated or after.truncated:
        reason = 'episode_ended'
    elif len(final) != 3 or not all((math.isfinite(v) for v in final)) or (not math.isfinite(rotation_error)):
        reason = 'invalid_pose'
    elif result.blocked or result.ik_error or (not result.reached):
        reason = 'blocked'
    elif rotation_error > 1.0:
        reason = 'orientation_drift'
    elif lateral > 0.0005:
        reason = 'lateral_drift'
    elif progress < 0.0001:
        reason = 'no_progress'
    elif progress > travel + 0.0005 or math.dist(final, target) > 0.0025:
        reason = 'tracking_error'
    else:
        reason = 'reobserve_before_next_segment'
    return dict(started=True, reason=reason, steps_used=motion['steps_used'], alignment=alignment, motion=motion, commanded_travel_m=travel, inspection_required=True, entry_complete=False, note='This call commands at most 2 mm of straight travel. Reobserve the actual key and call again with fresh landmarks before another segment. Reached describes the gripper motion, not key progress or seating. Stop on obstruction or slipping; clear the opening before sideways or rotational correction.')
