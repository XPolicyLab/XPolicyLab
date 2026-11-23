# Copyright © 2026 Martingale. All Rights Reserved.
import numpy as np
def target_from_mask(sight, mask):
    rows, cols = np.nonzero(mask)
    if len(rows) < sight.MIN_PIXELS:
        return None
    view = sight.head_view()
    z = sight.kin.TABLE_Z
    ray = view.rays(np.array([rows.mean()]), np.array([cols.mean()]))[0]
    return view.pos + ray * ((z - view.pos[2]) / ray[2])

def lift(tools, state, start_joints, arm, remaining):
    kin = tools.kin
    q = tools._joints(state, arm)
    current = tools._pose_of(arm, q)
    target = current.copy()
    target[2, 3] = max(current[2, 3], kin.TABLE_Z)
    ways, path, why, note = tools._plan(arm, q, target, 'line', start_joints)
    if why is None and len(ways) + tools.SETTLE_ACTIONS + 1 <= remaining:
        pitch, yaw, roll = kin.angles(target[:3, :3])
        return ({'xyz': target[:3, 3].tolist(), 'arm': arm, 'pitch': pitch, 'yaw': yaw, 'roll': roll, 'gripper': 0, 'path': 'line'}, ways)
    return (None, None)

def choose_view(tools, sight, mask, state, start_joints, arm, remaining, used=()):
    kin = tools.kin
    target = target_from_mask(sight, mask)
    if target is None:
        return None
    return None
