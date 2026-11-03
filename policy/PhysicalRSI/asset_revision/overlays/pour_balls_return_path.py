"""Continuous return from official proprioception and RGB-estimated rigid geometry."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def bounded_joint_step(kin, q, goal, base):
    delta = np.asarray(goal) - q
    delta *= min(1., .04 / max(float(np.max(np.abs(delta))), 1e-9))
    current = base @ kin.fk(q)
    for _ in range(12):
        proposed = q + delta
        target = base @ kin.fk(proposed)
        distance = np.linalg.norm(target[:3, 3] - current[:3, 3])
        angle = Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).magnitude()
        if distance <= .012 and angle <= .085:
            return proposed
        delta *= min(.95, .012 / max(distance, 1e-9), .085 / max(angle, 1e-9))
    return q + delta


def clearance(kin, q, base, grasp_local, cup_points, tree, aperture, support_plane):
    ee = base @ kin.fk(q)
    cup = ee @ np.linalg.inv(grasp_local)
    world = cup_points @ cup[:3, :3].T + cup[:3, 3]
    return min(
        float(tree.query(world)[0].min()) - .005,
        float((world @ np.asarray(support_plane['normal']) + support_plane['offset']).min()) - .005,
        kin.finger_separation(ee, tree, aperture) - .006,
        kin.tool_separation(ee, tree) - .005,
        .002 - kin.collision(q, base, tree),
        kin.table_separation(q, base, support_plane) - .005,
        kin.self_separation(q),
    )


def plan_return(kin, q, current_ee, grasp_local, cup_points, vase_points, targets, aperture, support_plane):
    """Two geometric skill_choices; every interpolation sample respects original action bounds."""
    q = np.asarray(q, float)
    base = current_ee @ np.linalg.inv(kin.fk(q))
    tree = cKDTree(vase_points)
    cup = cup_points[::max(1, len(cup_points) // 1200)]
    diagnostics = []
    for skill_choice in [[targets['return_high']], [targets['withdraw_tilted'], targets['return_high']]]:
        cursor = q.copy()
        goals = []
        steps = 0
        minimum = float('inf')
        valid = True
        for target in skill_choice:
            goal, errors = kin.solve(np.linalg.inv(base) @ target, cursor, position_tolerance=.0005, rotation_tolerance=.004)
            if errors[0] >= .008 or errors[1] >= .08:
                valid = False
                break
            for _ in range(240):
                # Includes initial point and denser midpoint samples, never endpoint-only.
                next_q = bounded_joint_step(kin, cursor, goal, base)
                for sample in [cursor, (cursor + next_q) / 2, next_q]:
                    gap = clearance(kin, sample, base, grasp_local, cup, tree, aperture, support_plane)
                    minimum = min(minimum, gap)
                    if gap < 0:
                        valid = False
                        break
                if not valid:
                    break
                cursor = next_q
                steps += 1
                if np.max(np.abs(cursor - goal)) < .001:
                    break
            else:
                valid = False
            if not valid:
                break
            goals.append({'joints': goal.tolist(), 'matrix': target})
        diagnostics.append({'waypoints': len(skill_choice), 'valid': valid, 'nominal_actions': steps, 'minimum_margin_m': minimum if np.isfinite(minimum) else None})
        if valid:
            return goals, diagnostics
    return None, diagnostics
