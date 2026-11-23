"""Physical opening candidates for accepted, upright mug geometry; no motion.

Only horizontal closing axes are supported. Width is local to the material
between the pads. The clearance is not a collision or reachability certificate.
"""
import math
try:
    from martingale.robot_dimensions import GRIP_SPAN_M
except ModuleNotFoundError:
    # The isolated worker mounts only the shared dimensions module.
    from robot_dimensions import GRIP_SPAN_M

TABLE_Z = 0.765

# Total extra jaw gap from directed self-play; reports manifest, MUG_GRASP_CLEARANCE_M.
def mug_grasp_plan(mug, part='body', contact_z=None, clearance_m=0.01):
    if not mug.get('valid'):
        raise ValueError('Need an accepted mug body and handle fit before planning a grasp')
    if part not in ('body', 'handle'):
        raise ValueError('part must be body or handle')
    clearance = float(clearance_m)
    if not math.isfinite(clearance) or clearance < 0:
        raise ValueError('clearance_m must be finite and nonnegative')
    body = mug['body']
    handle = mug['handle']
    x, y, rb, rt, h = map(float, body['parameters'])
    yaw, a, b, t, offset, zfraction = map(float, handle['parameters'])
    if not all((math.isfinite(v) for v in [x, y, rb, rt, h, yaw, a, b, t, offset, zfraction])) or min(rb, rt, h, a, b, t) <= 0:
        raise ValueError('Invalid fitted mug geometry')
    closing = [-math.sin(yaw), math.cos(yaw), 0.0]
    if part == 'body':
        z = TABLE_Z + 0.5 * h if contact_z is None else float(contact_z)
        if not math.isfinite(z) or not TABLE_Z < z < TABLE_Z + h:
            raise ValueError('Body contact_z must lie strictly inside the upright body height')
        local_radius = rb + (rt - rb) * (z - TABLE_Z) / h
        width = 2 * local_radius
        approach_width = 2 * max(local_radius, rt)
        target = [x, y, z]
        note = 'Candidate exterior body pinch; inspect pad and palm clearance at the rim and table. Keep the handle free for hanging.'
    else:
        if contact_z is not None:
            raise ValueError('Handle planning uses the fitted outermost vertical bar, not an arbitrary contact_z')
        radius = rb + (rt - rb) * zfraction
        distance = radius + a * (offset + 1)
        target = [x + distance * math.cos(yaw), y + distance * math.sin(yaw), TABLE_Z + h * zfraction]
        width = approach_width = 2 * t
        note = 'Candidate outer handle-bar pinch, not a pinch across the loop hole. Inspect the entire finger/palm corridor and plan how to free the handle for hanging.'
    gap = approach_width + clearance
    return dict(valid=gap <= GRIP_SPAN_M, part=part, contact_xyz=target, closing_axis_xyz=closing, pitch_deg=90.0, yaw_deg=math.degrees(math.atan2(closing[1], closing[0])), roll_deg=0.0, local_width_m=width, approach_width_m=approach_width, clearance_m=clearance, gap_m=gap, max_gap_m=GRIP_SPAN_M, reason=None if gap <= GRIP_SPAN_M else 'Requested width plus clearance exceeds the X5 opening; choose another contact, do not clamp the gap.', note=note + ' Geometry is from the initial upright pose; re-estimate if the mug moved.')
