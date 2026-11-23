# Copyright © 2026 Martingale. All Rights Reserved.
"""General skills for a small held object: pick from above, lift, hand over, stand up, insert and turn.

These helpers serve any task in which a small rigid object (a key, peg, tube, pen, plug, tool...) is picked from a
surface, possibly handed to the other gripper, turned so one end leads, lowered into an opening and possibly turned
in it. They use only public observations (RGB, camera calibration, proprioception, SAM masks) and ``robo`` motions,
and contain no object identity, scene coordinates or task tolerance: every target is computed from what the caller
measured in the current episode.

Conventions (the same as ``robo``): world metres; gripper orientation R = Rz(90 + yaw) @ Ry(pitch) @ Rx(roll), in
degrees; column 0 of R is the pointing direction, column 1 the jaw (finger-closing) axis. ``camera_meta`` extrinsics
are camera-to-world, the camera looking down its own -z axis with +y up. Pixels are (row, col).

Every function that moves returns a dict with ``status``, the receipts it executed and ``actions`` (native actions it
spent), and stops at the first unreached, blocked or suspicious step; nothing is retried. Functions that end at a
decision point attach the relevant current images to the turn (``robo.show``): look at them before the next call.
A receipt never proves that the object followed the hand.

Import: ``from library.held_object import held_object_skills as H``.
"""
import math

import numpy as np

TABLE_Z = 0.765
FINGER_BELOW_TCP_M = 0.01265


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _wrap(deg):
    return (deg + 180.0) % 360.0 - 180.0


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError('zero-length direction')
    return v / n


def gripper_rotation(pitch, yaw, roll):
    """3x3 gripper rotation for robo (pitch, yaw, roll) in degrees.

    Column 0 is the pointing direction and column 1 the jaw (finger-closing) axis, in world coordinates.
    """
    return _rz(math.radians(90.0 + yaw)) @ _ry(math.radians(pitch)) @ _rx(math.radians(roll))


def gripper_angles(rotation):
    """Inverse of gripper_rotation: (pitch, yaw, roll) in degrees for move_to.

    Within one degree of pointing straight down or up only yaw is reported (roll 0), as robo itself does.
    """
    R = np.asarray(rotation, float)
    point = R[:, 0]
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -point[2]))))
    if math.hypot(point[0], point[1]) < math.sin(math.radians(1.0)):
        jaw = R[:, 1]
        return pitch, _wrap(math.degrees(math.atan2(-jaw[1], -jaw[0]))), 0.0
    yaw = math.degrees(math.atan2(point[1], point[0])) - 90.0
    rest = gripper_rotation(pitch, yaw, 0.0).T @ R
    return pitch, _wrap(yaw), _wrap(math.degrees(math.atan2(rest[2, 1], rest[1, 1])))


def axis_angle(axis, degrees):
    """3x3 rotation of ``degrees`` about world ``axis`` (right-hand rule)."""
    a = np.asarray(axis, float)
    n = np.linalg.norm(a)
    if n < 1e-9:
        return np.eye(3)
    a = a / n
    t = math.radians(degrees)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(t) * K + (1.0 - math.cos(t)) * K @ K


def top_grasp_yaw(axis_xy):
    """Yaw for a gripper pointing straight down whose jaws close ACROSS a horizontal object axis.

    axis_xy: the object's long direction on the table (either sign). yaw + 180 is the same finger placement.
    """
    return _wrap(math.degrees(math.atan2(axis_xy[1], axis_xy[0])) - 90.0)


def pointing_yaw(direction_xy):
    """Yaw that makes a LEVEL gripper (pitch 0) point along the horizontal ``direction_xy``.

    Typical use: a receiving hand that approaches a held object horizontally along the object's current axis, with
    roll +90 or -90 so the jaws close vertically on the object's flat faces.
    """
    return _wrap(math.degrees(math.atan2(direction_xy[1], direction_xy[0])) - 90.0)


def jaw_alignment(robo, arm, face_normal):
    """How well ``arm``'s CURRENT jaw axis matches a measured face normal.

    face_normal: world normal of the flat faces the pads should close on (either sign). Returns
    {'jaw_axis', 'angle_deg'}: angle between the jaw axis and the normal, 0 to 90 degrees. Pads close squarely on
    the faces only when this angle is small; otherwise they catch an edge. No motion.
    """
    _, R, _ = arm_pose(robo, arm)
    jaw = R[:, 1]
    c = abs(float(jaw @ _unit(face_normal)))
    return {'jaw_axis': jaw.tolist(), 'angle_deg': math.degrees(math.acos(min(1.0, c)))}


def arm_pose(robo, arm):
    """Current (tcp_xyz ndarray, rotation 3x3, ArmState) of ``arm`` from public proprioception."""
    s = robo.state().arm(arm)
    return np.asarray(s.tcp_pos, float), gripper_rotation(s.pitch, s.yaw, s.roll), s


def _hand_yaw(robo, arm):
    """Heading (degrees) of the hand's jaw axis projected on the table, from eef_quat; used to measure turns."""
    w, x, y, z = robo.state().arm(arm).eef_quat
    jaw = np.array([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)])
    point = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)])
    v = jaw if math.hypot(jaw[0], jaw[1]) > math.hypot(point[0], point[1]) else point
    return math.degrees(math.atan2(v[1], v[0]))


def _actions(receipt):
    if isinstance(receipt, dict):
        return int(receipt.get('steps_used') or 0)
    return int(getattr(receipt, 'steps_used', 0) or 0)


def _ok(receipt):
    if isinstance(receipt, dict):
        return bool(receipt.get('reached'))
    return bool(getattr(receipt, 'reached', False)) and not getattr(receipt, 'blocked', False)


def _show(robo, cameras):
    for cam in cameras:
        try:
            robo.show(cam)
        except Exception:
            pass


def _wrist(arm):
    return 'cam_%s_wrist' % arm


def _other(arm):
    return 'right' if arm == 'left' else 'left'


def pixel_ray(robo, camera, row, col):
    """World ray (origin, unit direction) through pixel (row, col) of ``camera``'s current calibration."""
    meta = robo.camera_meta(camera)
    K = np.asarray(meta['intrinsic_matrix'], float)
    T = np.asarray(meta['extrinsic_matrix'], float)
    d = np.array([(col - K[0, 2]) / K[0, 0], -(row - K[1, 2]) / K[1, 1], -1.0])
    d = T[:3, :3] @ d
    return T[:3, 3].copy(), d / np.linalg.norm(d)


def project(robo, camera, xyz):
    """Pixel (row, col) of a world point in ``camera`` (None when the point is behind the camera)."""
    meta = robo.camera_meta(camera)
    K = np.asarray(meta['intrinsic_matrix'], float)
    T = np.asarray(meta['extrinsic_matrix'], float)
    p = T[:3, :3].T @ (np.asarray(xyz, float) - T[:3, 3])
    if p[2] >= -1e-6:
        return None
    return (float(K[1, 2] - K[1, 1] * p[1] / -p[2]), float(K[0, 2] + K[0, 0] * p[0] / -p[2]))


def triangulate(robo, pixels):
    """Least-squares 3D point of the SAME physical feature marked in two or more current camera images.

    pixels: {camera: (row, col)} in each camera's CURRENT image (do not mix views from before and after a motion).
    Returns {'xyz', 'ray_distance_m': {camera: d}, 'reprojection_px': {camera: e}}. Large residuals mean the pixels
    are not the same feature or a view is stale: mark the feature again instead of using the point.
    """
    rays = [(cam,) + pixel_ray(robo, cam, r, c) for cam, (r, c) in pixels.items()]
    if len(rays) < 2:
        raise ValueError('Need the same feature in at least two cameras')
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for _, o, d in rays:
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ o
    x = np.linalg.solve(A, b)
    dist, rep = {}, {}
    for cam, o, d in rays:
        v = x - o
        dist[cam] = float(np.linalg.norm(v - (v @ d) * d))
        pr = project(robo, cam, x)
        rr, cc = pixels[cam]
        rep[cam] = None if pr is None else float(math.hypot(pr[0] - rr, pr[1] - cc))
    return {'xyz': x, 'ray_distance_m': dist, 'reprojection_px': rep}


def held_axis(robo, lower, upper):
    """Axis and lean of a held elongated object from two features marked in two current views.

    lower / upper: {camera: (row, col)} of the same lower feature (for example the leading tip) and the same higher
    feature on the object's centre line (for example a point on the shaft well above the tip), each in at least two
    current images taken at the same step. Returns {'lower_xyz', 'upper_xyz', 'axis' (unit, upper -> lower),
    'lean_deg' (angle between the axis and straight down), 'lean_direction_xy' (where the lower end is displaced
    to, unit or None), 'residual_px'}. Short feature spacing or large residuals give an unreliable lean: mark
    features farther apart or on clearer views. No motion.
    """
    lo = triangulate(robo, lower)
    up = triangulate(robo, upper)
    axis = _unit(lo['xyz'] - up['xyz'])
    lean = math.degrees(math.acos(max(-1.0, min(1.0, -axis[2]))))
    h = np.array([axis[0], axis[1]])
    return {'lower_xyz': lo['xyz'].tolist(), 'upper_xyz': up['xyz'].tolist(), 'axis': axis.tolist(),
            'lean_deg': lean, 'lean_direction_xy': None if np.linalg.norm(h) < 1e-6 else (h / np.linalg.norm(h)).tolist(),
            'spacing_m': float(np.linalg.norm(lo['xyz'] - up['xyz'])),
            'residual_px': {'lower': lo['reprojection_px'], 'upper': up['reprojection_px']}}


def aimed_view_pose(robo, arm, aim_xyz, *, pitch=30.0, yaw=0.0, distance=0.25):
    """TCP position that puts ``aim_xyz`` near the centre of ``arm``'s wrist camera, for an EMPTY hand.

    Uses the arm's current pose and the wrist camera's current calibration to find the camera mount, then places
    the camera ``distance`` metres from the aim point looking at it with the given gripper pitch and yaw (degrees;
    pitch 30 views from roughly 45 degrees above the table and shows an object's side and thickness). Returns
    {'xyz', 'pitch', 'yaw', 'roll'} for move_to (use the same angles, roll 0, path='line'). It does not move and does
    not check reach or clearance: preview it, and keep the other arm and the object out of the path.
    """
    from scipy.spatial.transform import Rotation
    hand = robo.state().arm(arm)
    C = np.asarray(robo.camera_meta(_wrist(arm))['extrinsic_matrix'], float)
    q = hand.eef_quat
    R0 = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    R = Rotation.from_euler('ZYX', [90.0 + yaw, pitch, 0.0], degrees=True).as_matrix()
    mount_R = R0.T @ C[:3, :3]
    mount_t = R0.T @ (C[:3, 3] - np.asarray(hand.tcp_pos))
    ray = R @ mount_R @ np.array([0.0, 0.25, -1.0])
    ray /= np.linalg.norm(ray)
    xyz = np.asarray(aim_xyz, float) - distance * ray - R @ mount_t
    return {'xyz': xyz.tolist(), 'pitch': pitch, 'yaw': yaw, 'roll': 0.0}


def preview_poses(robo, candidates):
    """Zero-action reachability check of several candidate poses (dicts of move_to keyword arguments with 'xyz').

    Returns [{'candidate', 'reachable', 'planned_native_actions', 'ik_error'}] in the same order. Use it to pick an
    arm, a roll sign (+90 or -90) or an approach direction before spending actions, and to check that the poses a
    plan needs LATER (for example the upright pose above an opening and the turned pose) are reachable for the arm
    and grasp you are about to choose. A reachable preview is not a collision check against objects or the other arm.
    """
    out = []
    for c in candidates:
        kw = dict(c)
        xyz = kw.pop('xyz')
        try:
            p = robo.preview_move(xyz, **kw)
            out.append({'candidate': c, 'reachable': bool(p.get('reachable')),
                        'planned_native_actions': p.get('planned_native_actions'), 'ik_error': p.get('ik_error')})
        except Exception as error:
            out.append({'candidate': c, 'reachable': False, 'planned_native_actions': None,
                        'ik_error': '%s: %s' % (type(error).__name__, error)})
    return out


def in_hand_view(robo, arm, prompt, *, min_score=0.2):
    """Where the held object appears in the HOLDING hand's own wrist camera (no motion).

    The wrist camera moves with the hand, so an object that stays put in the grip keeps its place, heading and size
    in this image; a change between two calls made before and after a motion means the object moved in the hand.
    Returns {'found', 'centroid_rc', 'axis_deg' (image heading of the mask's long side), 'area_px', 'step'}. The
    mask may include a finger or miss a thin part: check it against the image when it matters.
    """
    seg = robo.segment(prompt, camera=_wrist(arm), min_score=min_score, include_contours=True)
    step = robo.state().step
    if not seg.found or not seg.contours_rc:
        return {'found': False, 'step': step}
    pts = np.array([p for outline in seg.contours_rc for p in outline], float)
    axis = None
    if len(pts) >= 5:
        vals, vecs = np.linalg.eigh(np.cov((pts - pts.mean(0)).T))
        v = vecs[:, -1]
        axis = math.degrees(math.atan2(v[0], v[1])) % 180.0
    return {'found': True, 'centroid_rc': [float(seg.centroid[0]), float(seg.centroid[1])], 'axis_deg': axis,
            'area_px': int(seg.area_px), 'step': step}


def moved_in_hand(before, after, *, shift_px=14.0, turn_deg=7.0, area_ratio=(0.7, 1.4)):
    """Compare two in_hand_view results. Returns {'comparable', 'moved', 'shift_px', 'turn_deg', 'area_ratio'}.

    moved is True when the mask shifted, turned or changed size by more than the given amounts. Not comparable when
    either view did not find the object. A False 'moved' is not proof of a firm grip: it covers only what this
    camera sees.
    """
    if not (before or {}).get('found') or not (after or {}).get('found'):
        return {'comparable': False, 'moved': None}
    shift = math.dist(before['centroid_rc'], after['centroid_rc'])
    turn = None
    if before.get('axis_deg') is not None and after.get('axis_deg') is not None:
        d = abs(before['axis_deg'] - after['axis_deg']) % 180.0
        turn = min(d, 180.0 - d)
    ratio = after['area_px'] / max(1, before['area_px'])
    moved = shift > shift_px or (turn is not None and turn > turn_deg) or not (area_ratio[0] <= ratio <= area_ratio[1])
    return {'comparable': True, 'moved': bool(moved), 'shift_px': shift, 'turn_deg': turn, 'area_ratio': ratio}


def pick_from_above(robo, arm, grasp_xyz, yaw, gap_m, *, pregrasp_height_m=0.026, clearance_m=None):
    """Open over a resting object and descend straight down to the grasp point; stop BEFORE closing.

    arm: the picking hand (prefer the one nearer the object with a clear corridor). grasp_xyz: where the middle of
    the finger pads should be (world metres; choose a straight, flat-sided part and a height inside its thickness;
    a z lower than the fingertips allow is raised so they stop at the table surface).
    yaw: gripper yaw so the jaws close across that part (``top_grasp_yaw``). gap_m: opening, the part's width across
    the jaws plus about 8 mm.

    Moves the empty hand through clear space to ``pregrasp_height_m`` above the grasp point, pointing straight down,
    opening to gap_m on the way (move_to with gap_m), then lowers ONLY z with ``robo.approach_to(..., empty_hand=True,
    vertical=True)``, which slows down near contact. Raise pregrasp_height_m when something taller is in the corridor.
    Returns {'status': 'ready_to_close' | 'pregrasp_unreached' | 'descent_stopped', 'receipts', 'actions'} and
    attaches the picking wrist image and the head image. On 'ready_to_close', check in the wrist image that both
    fingers straddle the intended part and clear everything else, then call ``close_and_test_lift``. On any other
    status nothing has closed: look, re-plan (another height, yaw, point or arm) and never close on a stopped descent.
    """
    g = np.asarray(grasp_xyz, float).copy()
    g[2] = max(g[2], TABLE_Z + FINGER_BELOW_TCP_M)
    above = g + np.array([0.0, 0.0, pregrasp_height_m])
    receipts = [robo.move_to([float(v) for v in above], arm=arm, pitch=90.0, yaw=float(yaw), roll=0.0,
                             gripper=robo.OPEN, gap_m=float(gap_m))]
    if not _ok(receipts[-1]):
        _show(robo, (_wrist(arm), 'cam_head'))
        return {'status': 'pregrasp_unreached', 'receipts': receipts, 'actions': sum(map(_actions, receipts))}
    kw = {'empty_hand': True, 'vertical': True}
    if clearance_m is not None:
        kw['clearance_m'] = float(clearance_m)
    receipts.append(robo.approach_to([float(v) for v in g], arm=arm, **kw))
    _show(robo, (_wrist(arm), 'cam_head'))
    status = 'ready_to_close' if _ok(receipts[-1]) else 'descent_stopped'
    return {'status': status, 'receipts': receipts, 'actions': sum(map(_actions, receipts))}


def close_and_test_lift(robo, arm, *, lift_m=0.009, prompt=None):
    """Close the stationary hand on the object, then make one short straight test lift.

    Closes with set_gripper (both arms still), then lifts lift_m metres straight up with the gripper kept closed and
    the orientation fixed. With ``prompt`` (a SAM description of the object) the object's place in the holding
    wrist camera is compared before and after the lift (``in_hand_view``). Returns {'status': 'lifted' |
    'lift_unreached' | 'moved_in_hand', 'gripper' (the close receipt; its holding flag is not a reliable contact
    measurement), 'in_hand', 'receipts', 'actions'} and attaches both wrist images and the head image. 'lifted' only
    means the hand rose: confirm in the images that the object rose with the fingers without tipping before carrying it farther. A part
    that hangs or pivots between the fingertips is a weak grasp: put it back down gently and grasp closer to its
    heavy end or on broader faces instead of carrying it.
    """
    receipts = [robo.set_gripper(arm, robo.CLOSE)]
    before = in_hand_view(robo, arm, prompt) if prompt else None
    p, _, s = arm_pose(robo, arm)
    receipts.append(robo.move_to([float(p[0]), float(p[1]), float(p[2] + lift_m)], arm=arm, pitch=s.pitch, yaw=s.yaw,
                                 roll=s.roll, gripper=robo.CLOSE, path='line'))
    check = moved_in_hand(before, in_hand_view(robo, arm, prompt)) if prompt else None
    _show(robo, (_wrist(arm), _wrist(_other(arm)), 'cam_head'))
    status = 'lifted'
    if not _ok(receipts[-1]):
        status = 'lift_unreached'
    elif check and check.get('moved'):
        status = 'moved_in_hand'
    return {'status': status, 'gripper': receipts[0], 'in_hand': check, 'receipts': receipts,
            'actions': sum(map(_actions, receipts))}


def lift_held(robo, arm, dz, *, segment_m=0.02, prompt=None):
    """Raise a held object straight up by ``dz`` metres in short segments, orientation and grip unchanged.

    Each segment is one move_to on a straight line with the gripper kept closed. With ``prompt`` the object's place
    in the holding wrist camera is compared after every segment and the lift stops when it moved in the hand.
    Returns {'status': 'done' | 'unreached' | 'moved_in_hand', 'raised_m', 'in_hand', 'receipts', 'actions'} and
    attaches the holding wrist image and the head image. Use a small first dz (about 1 cm), look, then the rest.
    Stop and support the object when it sags: lifting faster does not fix a weak grasp.
    """
    p0, _, s = arm_pose(robo, arm)
    n = max(1, int(math.ceil(abs(dz) / max(1e-3, segment_m))))
    receipts, check = [], None
    ref = in_hand_view(robo, arm, prompt) if prompt else None
    status = 'done'
    for k in range(1, n + 1):
        target = [float(p0[0]), float(p0[1]), float(p0[2] + dz * k / n)]
        receipts.append(robo.move_to(target, arm=arm, pitch=s.pitch, yaw=s.yaw, roll=s.roll, gripper=robo.CLOSE,
                                     path='line'))
        if not _ok(receipts[-1]):
            status = 'unreached'
            break
        if prompt:
            check = moved_in_hand(ref, in_hand_view(robo, arm, prompt))
            if check.get('moved'):
                status = 'moved_in_hand'
                break
    p1, _, _ = arm_pose(robo, arm)
    _show(robo, (_wrist(arm), 'cam_head'))
    return {'status': status, 'raised_m': float(p1[2] - p0[2]), 'in_hand': check, 'receipts': receipts,
            'actions': sum(map(_actions, receipts))}


def rotate_held(robo, arm, axis, degrees, *, pivot=None, max_step_deg=4.0, retention_checked=False,
                check_deg=3.0, allow_jaw_axis=False, prompt=None, path='line'):
    """Turn the holding hand (and what it holds) about a world axis through ``pivot``, in small checked steps.

    The hand turns rigidly: the TCP orbits ``pivot`` so that the object point at ``pivot`` stays still (pivot None
    turns about the TCP). Positive degrees turn counter-clockwise seen from the axis tip; a clockwise turn seen from
    above is negative about [0, 0, 1]. The rotation is split into steps of at most ``max_step_deg``; each step is
    one straight-line move_to with the gripper closed; with path='line' nothing moves when the straight path leaves
    the arm's reach.

    Retention check first: unless ``retention_checked`` is True, a rotation larger than ``check_deg`` executes only
    its first ``check_deg`` degrees and returns status 'check_retention' with the holding wrist image, the other wrist
    image and the head image. Compare them with the images from before the rotation: the object must sit between the
    pads exactly as before and must have turned by the same amount as the hand (it must not swing, slide along the
    pads or keep its old direction). If so, call again for the remaining angle (``degrees - turned_deg``) with
    retention_checked=True; if not, stop rotating and improve the grip while the object is supported. Pass
    retention_checked=True only after such a check for the CURRENT grip (a new grasp or a regrasp needs a new check).

    Rotation about the jaw axis: a pinch holds an object like a hinge about the line between the pads, so turning
    the hand about that line mostly does not turn the object (it keeps hanging the way gravity pulls it) and works
    the grip loose. A rotation whose axis is within about 25 degrees of the hand's current jaw axis is therefore
    refused with zero actions (status 'about_jaw_axis') unless ``allow_jaw_axis`` is True. Lean of a hanging object
    across the pads' line settles by itself (let it settle, then measure); lean along that line is corrected by
    turning about a horizontal axis perpendicular to it.

    With ``prompt`` the object's place in the holding wrist camera is also compared after each step and the turn
    stops when it moved in the hand. Returns {'status': 'done' | 'check_retention' | 'about_jaw_axis' | 'unreached' |
    'moved_in_hand', 'turned_deg' (commanded part completed), 'remaining_deg', 'in_hand', 'receipts', 'actions'}
    and attaches images. Kinematic only: re-measure the object afterwards; a commanded angle is not the object's.
    Turn a held object in clear space well away from fixtures and the other hand.
    """
    p0, R0, _ = arm_pose(robo, arm)
    if not allow_jaw_axis and abs(degrees) > 1e-9 and abs(float(_unit(axis) @ R0[:, 1])) > math.cos(math.radians(25.0)):
        return {'status': 'about_jaw_axis', 'turned_deg': 0.0, 'remaining_deg': degrees, 'in_hand': None,
                'receipts': [], 'actions': 0, 'jaw_axis': R0[:, 1].tolist()}
    gate = (not retention_checked) and abs(degrees) > check_deg + 1e-9
    total = math.copysign(check_deg, degrees) if gate else degrees
    c = p0 if pivot is None else np.asarray(pivot, float)
    n = max(1, int(math.ceil(abs(total) / max(0.5, float(max_step_deg)))))
    ref = in_hand_view(robo, arm, prompt) if prompt else None
    receipts, check, done, status = [], None, 0.0, 'done'
    for k in range(1, n + 1):
        D = axis_angle(axis, total * k / n)
        pk = c + D @ (p0 - c)
        pitch, yaw, roll = gripper_angles(D @ R0)
        receipts.append(robo.move_to([float(v) for v in pk], arm=arm, pitch=pitch, yaw=yaw, roll=roll,
                                     gripper=robo.CLOSE, path=path))
        if not _ok(receipts[-1]):
            status = 'unreached'
            break
        done = total * k / n
        if prompt:
            check = moved_in_hand(ref, in_hand_view(robo, arm, prompt))
            if check.get('moved'):
                status = 'moved_in_hand'
                break
    if gate and status == 'done':
        status = 'check_retention'
    _show(robo, (_wrist(arm), _wrist(_other(arm)), 'cam_head') if gate else (_wrist(arm), 'cam_head'))
    return {'status': status, 'turned_deg': done, 'remaining_deg': degrees - done, 'in_hand': check,
            'receipts': receipts, 'actions': sum(map(_actions, receipts))}


def stand_upright(robo, arm, long_axis, *, target=(0.0, 0.0, -1.0), pivot=None, max_step_deg=4.0,
                  retention_checked=False, prompt=None):
    """Turn the hand so a held object's MEASURED long axis points along ``target`` (default straight down).

    long_axis: the object's CURRENT world direction from the end that should be on top toward the end that must
    lead (``held_axis(...)['axis']``); a resting-pose estimate is stale once the object is lifted. The rotation is
    about long_axis x target through ``pivot`` (default the TCP; use the leading tip for small final corrections).
    Returns rotate_held's dict plus 'angle_deg'; the same retention check applies (without retention_checked=True
    only the first few degrees are turned and status is 'check_retention'; then call rotate_held for the rest about
    the same axis, or stand_upright again with the re-measured axis). Re-measure with held_axis afterwards.
    A large turn about the TCP can run into the wrist's joint limits part-way ('unreached'): preview the final
    pose first. When the grip is firm and the space is clear, one move_to straight to the final pose (for example
    pitch 90 above the opening) is a cheaper alternative to many small steps.
    """
    u, v = _unit(long_axis), _unit(target)
    angle = math.degrees(math.acos(max(-1.0, min(1.0, float(u @ v)))))
    axis = np.cross(u, v)
    if np.linalg.norm(axis) < 1e-6:
        return {'status': 'done', 'turned_deg': 0.0, 'remaining_deg': 0.0, 'in_hand': None, 'receipts': [],
                'actions': 0, 'angle_deg': angle, 'axis': [0.0, 0.0, 0.0]}
    out = rotate_held(robo, arm, axis, angle, pivot=pivot, max_step_deg=max_step_deg,
                      retention_checked=retention_checked, prompt=prompt)
    out['axis'] = (axis / np.linalg.norm(axis)).tolist()
    out['angle_deg'] = angle
    return out


def move_held(robo, arm, xyz, *, pitch, yaw, roll, retention_checked=False, check_deg=3.0, check_m=0.005,
              path='line'):
    """Carry a held object to a new hand pose (position and orientation), starting with a retention check.

    xyz / pitch / yaw / roll: the target as for move_to (for example the hand pose that holds the object upright
    above an opening; check it first with ``preview_poses``). Unless ``retention_checked`` is True, only the first
    part of the motion is made (at most ``check_m`` metres of the translation and ``check_deg`` degrees of the
    rotation, on a straight line) and the call returns 'check_retention' with both wrist images and the head image:
    the object must sit between the pads exactly as before and follow the hand. Then call again with
    retention_checked=True for the full motion. path='line' (default) moves the grasp point on a straight line or
    not at all ('unreached', with the reason); a joint swing (path='auto' or 'joint') whips a held object around
    and can throw it out of a fingertip grip, so with a held object prefer short line moves: carry it level to
    above the target first, then turn it there. Returns {'status': 'done' | 'check_retention' | 'unreached',
    'reason', 'receipts', 'actions'}.
    """
    p0, R0, _ = arm_pose(robo, arm)
    target = np.asarray(xyz, float)
    R1 = gripper_rotation(pitch, yaw, roll)
    if retention_checked:
        r = robo.move_to([float(v) for v in target], arm=arm, pitch=pitch, yaw=yaw, roll=roll, gripper=robo.CLOSE,
                         path=path)
        _show(robo, (_wrist(arm), 'cam_head'))
        return {'status': 'done' if _ok(r) else 'unreached', 'reason': getattr(r, 'ik_error', None), 'receipts': [r],
                'actions': _actions(r)}
    rel = R1 @ R0.T
    angle = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(rel) - 1.0) / 2.0))))
    axis = np.array([rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]])
    dist = float(np.linalg.norm(target - p0))
    f_rot = 1.0 if angle < 1e-6 else min(1.0, check_deg / angle)
    f_pos = 1.0 if dist < 1e-6 else min(1.0, check_m / dist)
    Rk = (axis_angle(axis, angle * f_rot) if np.linalg.norm(axis) > 1e-9 else np.eye(3)) @ R0
    pk = p0 + f_pos * (target - p0)
    kp, ky, kr = gripper_angles(Rk)
    r = robo.move_to([float(v) for v in pk], arm=arm, pitch=kp, yaw=ky, roll=kr, gripper=robo.CLOSE, path='line')
    _show(robo, (_wrist(arm), _wrist(_other(arm)), 'cam_head'))
    return {'status': 'check_retention' if _ok(r) else 'unreached', 'reason': getattr(r, 'ik_error', None),
            'receipts': [r], 'actions': _actions(r)}


def move_tip_over(robo, arm, tip_xyz, target_xyz, *, clearance_m=0.01, settle_tol_m=0.0015):
    """Translate the holding hand so a measured tip ends ``clearance_m`` above a target point; orientation kept.

    tip_xyz: CURRENT measured leading tip (for example ``held_axis(...)['lower_xyz']``); target_xyz: measured centre
    of the opening (its z is the opening's surface). One straight move; when the TCP ends more than settle_tol_m
    from the goal, one more short straight move toward the SAME goal. Returns {'status': 'done' | 'unreached',
    'commanded_offset_m', 'tcp_residual_m', 'receipts', 'actions'}. Re-measure the tip afterwards.
    """
    p0, _, s = arm_pose(robo, arm)
    goal = np.asarray(target_xyz, float) + np.array([0.0, 0.0, clearance_m])
    delta = goal - np.asarray(tip_xyz, float)
    want = p0 + delta
    receipts = [robo.move_to([float(v) for v in want], arm=arm, pitch=s.pitch, yaw=s.yaw, roll=s.roll,
                             gripper=robo.CLOSE, path='line')]
    p1, _, _ = arm_pose(robo, arm)
    if np.linalg.norm(p1 - want) > settle_tol_m and not getattr(receipts[-1], 'blocked', False):
        receipts.append(robo.move_to([float(v) for v in want], arm=arm, pitch=s.pitch, yaw=s.yaw, roll=s.roll,
                                     gripper=robo.CLOSE, path='line'))
        p1, _, _ = arm_pose(robo, arm)
    residual = float(np.linalg.norm(p1 - want))
    return {'status': 'done' if residual < 0.003 else 'unreached', 'commanded_offset_m': delta.tolist(),
            'tcp_residual_m': residual, 'receipts': receipts, 'actions': sum(map(_actions, receipts))}


def handoff_contact(robo, receiver, part_center_xyz, toward_holder_xyz, *, part_radius_m, part_thickness_m,
                    face_normal=(0.0, 0.0, 1.0), center_on='midplane', approach='along', rim_margin_m=0.001,
                    max_jaw_deg=10.0, upright_tcp_xyz=None, turn_deg=0.0):
    """Choose the receiving hand's contact pose on a flat part that the other hand holds (no motion).

    The rule: clamp the part's two broad flat faces squarely with the full pads, centred in the part's thickness
    and covering as much solid face as the holding fingers allow.

    part_center_xyz: measured centre of the flat part (for a ring or disc, the centre of its hole or outline) in
    current images, for example ``triangulate`` of the same feature in two wrist views. center_on: 'midplane' when
    that point is in the middle of the thickness, 'top_face' when it is on the upper face (it is then moved down by
    half the thickness along face_normal). toward_holder_xyz: the object's axis from the part toward where the
    other hand holds it (for a key held by its shaft: from the bow toward the tip). part_radius_m /
    part_thickness_m: the part's measured radius and thickness. face_normal: measured normal of the broad faces
    (default level faces); a tilted part needs its own normal.

    approach='along' (default): the receiver comes level along the object's axis from the free end; the contact is
    at the part's far edge (``part_radius_m - rim_margin_m`` toward the holder) and ``handoff_approach`` stops when
    the holder is first pushed, which is as deep as the holding fingers allow. When the object later hangs down,
    the receiver points down and its jaw axis is horizontal across the object's flat faces.
    approach='across': the receiver comes level from the side, perpendicular to the object's axis, pad centre on the
    part's centre, so the pads cover the part's full width; the object is then stood up by a pure wrist roll, but
    when it hangs down the receiver points level along the line of the object's flat faces, which is often out of
    reach above an opening: check later_reachable.
    In both cases the jaw axis lies along face_normal (roll +90 or -90), and every candidate gets a zero-action
    reach preview of its standoff and contact. With ``upright_tcp_xyz`` (where the receiver's grasp point must be
    when the object hangs straight down, for example above an opening) each candidate also previews its pose there
    after the object is turned to point down, and that pose turned ``turn_deg`` about the vertical (negative =
    clockwise seen from above). Best first: reachable now, reachable later, roll nearer the receiver's current roll.

    Returns {'contact_xyz', 'candidates': [{'contact_xyz', 'approach_xyz', 'pitch', 'yaw', 'roll',
    'jaw_angle_deg', 'reachable', 'ik_error', 'later_reachable', 'upright_angles', 'turned_angles'}], 'gap_m'
    (thickness + 12 mm)}; the top-level contact_xyz is the best candidate's. Pass a candidate's contact, angles and
    gap to ``handoff_approach`` together with face_normal. Reach is the robot's own kinematics only.
    """
    n = _unit(face_normal)
    c = np.asarray(part_center_xyz, float)
    if center_on == 'top_face':
        c = c - 0.5 * part_thickness_m * n
    elif center_on != 'midplane':
        raise ValueError("center_on must be 'midplane' or 'top_face'")
    a = np.asarray(toward_holder_xyz, float)
    a = _unit(a - (a @ n) * n)
    if approach == 'along':
        options = [(a, c + max(0.0, part_radius_m - rim_margin_m) * a)]
    elif approach == 'across':
        side = _unit(np.cross(n, a))
        options = [(side, c.copy()), (-side, c.copy())]
    else:
        raise ValueError("approach must be 'across' or 'along'")
    current = robo.state().arm(receiver).roll
    down = np.array([0.0, 0.0, -1.0])
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(a @ down)))))
    out = []
    for d, contact in options:
        for sign in (1.0, -1.0):
            jaw = sign * n
            pitch, yaw, rr = gripper_angles(np.column_stack([d, jaw, np.cross(d, jaw)]))
            Rg = gripper_rotation(pitch, yaw, rr)
            jaw_angle = math.degrees(math.acos(min(1.0, abs(float(Rg[:, 1] @ n)))))
            prev = preview_poses(robo, [dict(xyz=(contact - 0.025 * d).tolist(), arm=receiver, pitch=pitch, yaw=yaw,
                                             roll=rr),
                                        dict(xyz=contact.tolist(), arm=receiver, pitch=pitch, yaw=yaw, roll=rr)])
            item = {'contact_xyz': contact.tolist(), 'approach_xyz': d.tolist(), 'pitch': pitch, 'yaw': yaw,
                    'roll': rr, 'jaw_angle_deg': jaw_angle, 'reachable': all(x['reachable'] for x in prev),
                    'ik_error': next((x['ik_error'] for x in prev if x['ik_error']), None), 'later_reachable': None}
            if upright_tcp_xyz is not None:
                Rup = (axis_angle(np.cross(a, down), tilt) if tilt > 1e-6 else np.eye(3)) @ Rg
                Rturn = axis_angle((0.0, 0.0, 1.0), turn_deg) @ Rup
                up, tu = gripper_angles(Rup), gripper_angles(Rturn)
                later = preview_poses(robo, [dict(xyz=list(upright_tcp_xyz), arm=receiver, pitch=up[0], yaw=up[1],
                                                  roll=up[2]),
                                             dict(xyz=list(upright_tcp_xyz), arm=receiver, pitch=tu[0], yaw=tu[1],
                                                  roll=tu[2])])
                item.update(later_reachable=all(x['reachable'] for x in later), upright_angles=list(up),
                            turned_angles=list(tu))
            out.append(item)
    out.sort(key=lambda o: (not o['reachable'], o['later_reachable'] is False, abs(_wrap(o['roll'] - current))))
    return {'contact_xyz': out[0]['contact_xyz'], 'candidates': out, 'gap_m': part_thickness_m + 0.012,
            'max_jaw_deg': max_jaw_deg}


def handoff_approach(robo, receiver, contact_xyz, *, pitch, yaw, roll, gap_m, standoff_m=0.025, clearance_m=None,
                     face_normal=None, max_jaw_deg=10.0):
    """Bring the EMPTY receiving hand, open, to a contact pose on an object the other hand holds; never closes.

    contact_xyz and the angles: normally from ``handoff_contact`` (deepest broad-face contact; the approach stops
    when the holding hand is first pushed). gap_m: opening, the part's thickness plus about 12 mm. face_normal: the
    measured normal of the faces to clamp; when given, nothing moves unless the planned jaw axis is within
    ``max_jaw_deg`` of it (status 'jaw_misaligned'), so the pads close flat on the faces instead of on an edge.

    The receiver first moves to a standoff ``standoff_m`` short of the contact along its own pointing direction,
    opening to gap_m on the way (move_to with gap_m), then advances with ``robo.approach_to(..., empty_hand=True)``,
    which stops when the holding hand is pushed, on tracking error or obstruction. The holding hand never moves here.
    Returns {'status': 'at_contact' | 'stopped' | 'standoff_unreached' | 'jaw_misaligned', 'reason',
    'holder_shift_m', 'jaw_angle_deg', 'receiver_xyz', 'holder_xyz', 'receipts', 'actions'} and attaches both wrist
    images and the head image. 'stopped' with reason 'other_arm_moved' is the expected end of a deepest approach.

    Before closing, judge in BOTH wrist views: each pad lies flat on one broad face, the part's thickness is centred
    between the pads, the pads cover solid material (not a hole, a rim or an edge) over most of their length, and the
    receiver's fingers clear the holder's fingers. Then ``handoff_close``. If the contact is shallow or on an edge,
    back off a few millimetres along the approach with the fingers open, correct and approach again; never push on.
    """
    holder = _other(receiver)
    h0, _, _ = arm_pose(robo, holder)
    R = gripper_rotation(pitch, yaw, roll)
    jaw_angle = None
    if face_normal is not None:
        jaw_angle = math.degrees(math.acos(min(1.0, abs(float(R[:, 1] @ _unit(face_normal))))))
        if jaw_angle > max_jaw_deg:
            _show(robo, (_wrist(receiver), _wrist(holder)))
            return {'status': 'jaw_misaligned', 'reason': 'jaw axis %.1f deg from the face normal' % jaw_angle,
                    'holder_shift_m': 0.0, 'jaw_angle_deg': jaw_angle, 'receipts': [], 'actions': 0}
    c = np.asarray(contact_xyz, float)
    standoff = c - standoff_m * R[:, 0]
    receipts = [robo.move_to([float(v) for v in standoff], arm=receiver, pitch=pitch, yaw=yaw, roll=roll,
                             gripper=robo.OPEN, gap_m=float(gap_m))]
    if not _ok(receipts[-1]):
        _show(robo, (_wrist(receiver), _wrist(holder), 'cam_head'))
        return {'status': 'standoff_unreached', 'reason': getattr(receipts[-1], 'ik_error', None),
                'holder_shift_m': float(np.linalg.norm(arm_pose(robo, holder)[0] - h0)), 'jaw_angle_deg': jaw_angle,
                'receipts': receipts, 'actions': sum(map(_actions, receipts))}
    kw = {'empty_hand': True}
    if clearance_m is not None:
        kw['clearance_m'] = float(clearance_m)
    receipts.append(robo.approach_to([float(v) for v in c], arm=receiver, **kw))
    hp, _, _ = arm_pose(robo, holder)
    rp, _, _ = arm_pose(robo, receiver)
    _show(robo, (_wrist(receiver), _wrist(holder), 'cam_head'))
    reached = _ok(receipts[-1])
    return {'status': 'at_contact' if reached else 'stopped', 'reason': receipts[-1].get('reason'),
            'holder_shift_m': float(np.linalg.norm(hp - h0)), 'jaw_angle_deg': jaw_angle,
            'receiver_xyz': rp.tolist(), 'holder_xyz': hp.tolist(), 'receipts': receipts,
            'actions': sum(map(_actions, receipts))}


def handoff_close(robo, receiver, approach, *, still_tol_m=0.0015):
    """Close the receiving hand on the part after ``handoff_approach``, only if neither hand has moved since.

    approach: the dict returned by handoff_approach (its images must have been judged first). Refuses with zero
    actions ('moved_since_approach') when either hand is more than still_tol_m from where the approach ended, since
    the judged contact no longer holds. Otherwise closes with both arms still and attaches both wrist images and the
    head image. Returns {'status': 'closed' | 'moved_since_approach' | 'no_approach', 'gripper', 'actions'}.
    Next: look again in both views that the pads clamp the part flat and deep, then ``handoff_release``. A part
    that tipped or slid while closing is not a firm grip: open the receiver and correct while the holder still holds.
    """
    if not approach or approach.get('status') not in ('at_contact', 'stopped') or 'receiver_xyz' not in approach:
        return {'status': 'no_approach', 'gripper': None, 'actions': 0}
    holder = _other(receiver)
    moved = max(float(np.linalg.norm(arm_pose(robo, receiver)[0] - np.asarray(approach['receiver_xyz']))),
                float(np.linalg.norm(arm_pose(robo, holder)[0] - np.asarray(approach['holder_xyz']))))
    if moved > still_tol_m:
        return {'status': 'moved_since_approach', 'moved_m': moved, 'gripper': None, 'actions': 0}
    g = robo.set_gripper(receiver, robo.CLOSE)
    _show(robo, (_wrist(receiver), _wrist(holder), 'cam_head'))
    return {'status': 'closed', 'gripper': g, 'actions': _actions(g)}


def handoff_release(robo, holder, receiver, *, check_m=0.004, prompt=None):
    """Finish a hand-over after the receiver has CLOSED on the part: open the holder, then test the new grip.

    Call it only after ``handoff_close`` and a look at both views showing the pads clamping the part. Opens the
    holder with both arms still, then moves the receiver ``check_m`` metres straight back along its own pointing
    direction (away from the holder) with its gripper closed, as a first retention check. The holder stays open
    where it is: it is not withdrawn here. With ``prompt`` the object's place in the receiver's wrist camera is
    compared before and after the check move. Returns {'status': 'transferred' | 'check_unreached' |
    'moved_in_hand', 'in_hand', 'receipts', 'actions'} and attaches both wrist images and the head image. Confirm
    in the images that the part stayed between the receiver's pads and did not turn; then move the empty holder
    straight away (up) from the part. The first rotation of the new grip goes through rotate_held's retention check.
    """
    _, Rr, s = arm_pose(robo, receiver)
    receipts = [robo.set_gripper(holder, robo.OPEN)]
    before = in_hand_view(robo, receiver, prompt) if prompt else None
    p, _, _ = arm_pose(robo, receiver)
    target = p - check_m * Rr[:, 0]
    receipts.append(robo.move_to([float(v) for v in target], arm=receiver, pitch=s.pitch, yaw=s.yaw, roll=s.roll,
                                 gripper=robo.CLOSE, path='line'))
    check = moved_in_hand(before, in_hand_view(robo, receiver, prompt)) if prompt else None
    _show(robo, (_wrist(receiver), _wrist(holder), 'cam_head'))
    status = 'transferred'
    if not _ok(receipts[-1]):
        status = 'check_unreached'
    elif check and check.get('moved'):
        status = 'moved_in_hand'
    return {'status': status, 'in_hand': check, 'receipts': receipts, 'actions': sum(map(_actions, receipts))}


def insert_until_stall(robo, arm, depth_m, *, step_m=0.002, direction=(0.0, 0.0, -1.0), stall_fraction=0.3):
    """Feed a held object along ``direction`` in short straight steps until it stops following or depth_m is fed.

    Start with the leading end already aligned just above (or just inside) the opening. Step k commands the hand to
    start + k * step_m along ``direction`` (one straight move_to, orientation fixed, gripper closed), so tracking lag
    does not accumulate; the hand's achieved travel in each step is compared with step_m. When a step achieves less
    than ``stall_fraction`` of it, or is blocked, the feed stops: the object (or a finger) rests on something.
    Returns {'status': 'fed' | 'stalled' | 'unreached', 'fed_m' (total hand travel along direction), 'lag_m'
    (commanded minus achieved travel), 'last_fraction', 'receipts', 'actions'} and attaches the holding wrist image
    and the head image. A free feed does not prove entry: the object may have passed beside the opening, so look.
    A stall right at the opening means misalignment (lean, offset or a rotated profile): lift a few millimetres,
    re-measure and correct, then feed again; do not push harder. Fully in means, in a side view, nothing of the
    inserted part is left above the opening face and the object's shoulder sits on that face; a further short feed
    then gains almost nothing. A blocked step can cost several native actions.
    """
    d = _unit(direction)
    p0, _, s = arm_pose(robo, arm)
    n = max(1, int(math.ceil(depth_m / max(1e-4, step_m))))
    receipts, frac, status = [], 1.0, 'fed'
    for k in range(1, n + 1):
        before, _, _ = arm_pose(robo, arm)
        target = p0 + min(depth_m, k * step_m) * d
        receipts.append(robo.move_to([float(v) for v in target], arm=arm, pitch=s.pitch, yaw=s.yaw, roll=s.roll,
                                     gripper=robo.CLOSE, path='line'))
        after, _, _ = arm_pose(robo, arm)
        frac = float((after - before) @ d) / step_m
        if getattr(receipts[-1], 'ik_error', None) and not getattr(receipts[-1], 'path', None):
            status = 'unreached'
            break
        if getattr(receipts[-1], 'blocked', False) or frac < stall_fraction:
            status = 'stalled'
            break
    p1, _, _ = arm_pose(robo, arm)
    _show(robo, (_wrist(arm), 'cam_head'))
    fed = float((p1 - p0) @ d)
    return {'status': status, 'fed_m': fed, 'lag_m': min(depth_m, len(receipts) * step_m) - fed, 'last_fraction': frac, 'receipts': receipts,
            'actions': sum(map(_actions, receipts))}


def turn_in_place(robo, arm, degrees, pivot_xy, *, step_deg=10.0, press_m=0.0, resist_fraction=0.5):
    """Turn a held, inserted object about a vertical line through ``pivot_xy`` in steps; stop on resistance.

    degrees: total turn, positive counter-clockwise seen from above, NEGATIVE CLOCKWISE. pivot_xy: the measured
    centre of the opening the object sits in. Each step is one straight-line move_to that keeps the pivot fixed,
    with the gripper closed; press_m lowers the hand that much at each step so a seated object does not ride up.
    After each step the hand's actual heading change (from proprioception) is compared with the command; when it
    achieves less than ``resist_fraction`` the turn stops ('resisted'). Returns {'status': 'done' | 'resisted' |
    'unreached', 'turned_deg' (measured hand heading change, signed), 'receipts', 'actions'} and attaches the
    holding wrist image and the head image. Check in the images that the OBJECT turned with the hand and stayed
    fully in and upright; if it rides up or tilts, stop, re-seat it and continue in the same direction.
    """
    p0, _, _ = arm_pose(robo, arm)
    y0 = _hand_yaw(robo, arm)
    n = max(1, int(math.ceil(abs(degrees) / max(0.5, float(step_deg)))))
    receipts, status, prev = [], 'done', y0
    for k in range(1, n + 1):
        p, R, _ = arm_pose(robo, arm)
        piv = np.array([pivot_xy[0], pivot_xy[1], p[2]], float)
        D = axis_angle((0.0, 0.0, 1.0), degrees / n)
        pk = piv + D @ (p - piv) - np.array([0.0, 0.0, press_m])
        pitch, yaw, roll = gripper_angles(D @ R)
        receipts.append(robo.move_to([float(v) for v in pk], arm=arm, pitch=pitch, yaw=yaw, roll=roll,
                                     gripper=robo.CLOSE, path='line'))
        now = _hand_yaw(robo, arm)
        got = _wrap(now - prev)
        prev = now
        if getattr(receipts[-1], 'ik_error', None) and not getattr(receipts[-1], 'path', None):
            status = 'unreached'
            break
        if got * math.copysign(1.0, degrees) < resist_fraction * abs(degrees / n):
            status = 'resisted'
            break
    _show(robo, (_wrist(arm), 'cam_head'))
    return {'status': status, 'turned_deg': _wrap(_hand_yaw(robo, arm) - y0), 'receipts': receipts,
            'actions': sum(map(_actions, receipts))}
