"""implementation capabilities."""
from __future__ import annotations
import numpy as np

def _trajectory(result, dof):
    position = np.asarray(result.get('position'), dtype=float)
    if position.ndim != 2 or position.shape[1] != dof or (not len(position)) or (not np.isfinite(position).all()):
        raise RuntimeError('Invalid cuRobo joint trajectory')
    return position

def plan_pose_result(planner, current, target, base, limits):
    """与v8 _plan的路径、IK回退和最近限位绕圈保持一致，不运行物理。"""
    current = np.asarray(current, dtype=np.float32).reshape(-1).copy()
    target = np.asarray(target, dtype=np.float32).reshape(7).copy()
    limits = np.asarray(limits, dtype=float)
    if limits.shape != (len(current), 2) or not np.isfinite(limits).all():
        raise RuntimeError('Invalid observed robot joint limits')
    diag = {'mode': 'pose', 'start_joint': current.tolist(), 'target': target.tolist()}
    result = planner.plan_path(current, target.tolist(), np.asarray(base).tolist())
    if result.get('status') != 'Success':
        diag['mode'] = 'ik_joint'
        ik = planner.solve_ik_to_joint(current, target.tolist(), real_robot_pose=np.asarray(base).tolist())
        if ik.get('status') == 'Success':
            result = planner.plan_joint(current, np.asarray(ik['joint_value']).reshape(-1))
    if result.get('status') == 'Success':
        path = _trajectory(result, len(current))
        if not np.any(np.abs(path[-1] - current) > np.pi):
            diag.update(accepted=True, controls=len(path), goal_joint=path[-1].tolist(), max_joint_delta_rad=float(np.max(np.abs(path[-1] - current))))
            return (result, diag)
        diag['pose_result_distant_winding'] = True
    else:
        diag['pose_and_raw_ik_joint_failed'] = True
    ik = planner.solve_ik_to_joint(current, target.tolist(), real_robot_pose=np.asarray(base).tolist())
    if ik.get('status') != 'Success':
        diag.update(accepted=False, reason='pose_and_ik_failed')
        return (None, diag)
    goal = np.asarray(ik['joint_value'], dtype=float).reshape(-1).copy()
    if goal.shape != current.shape or not np.isfinite(goal).all():
        raise RuntimeError('Invalid cuRobo IK joint solution')
    for j in range(len(goal)):
        choices = [goal[j] + k * 2 * np.pi for k in range(-3, 4) if limits[j, 0] <= goal[j] + k * 2 * np.pi <= limits[j, 1]]
        if not choices:
            diag.update(accepted=False, reason='no_in_limit_ik_solution', joint=j)
            return (None, diag)
        goal[j] = min(choices, key=lambda value: abs(value - current[j]))
    result = planner.plan_joint(current, goal)
    if result.get('status') != 'Success':
        diag.update(accepted=False, reason='canonical_ik_path_failed')
        return (None, diag)
    path = _trajectory(result, len(current))
    diag.update(accepted=True, mode='canonical_ik_joint', controls=len(path), goal_joint=path[-1].tolist(), max_joint_delta_rad=float(np.max(np.abs(path[-1] - current))))
    return (result, diag)

def plan_pose_sequence(planner, current, route, base, limits):
    """端点串成真正的cuRobo路径；后一步起点是前一步轨迹终点。"""
    current = np.asarray(current, dtype=np.float32).reshape(-1).copy()
    results = []
    diagnostics = {'method': 'sequential_curobo_paths', 'steps': []}
    for phase, target in route:
        result, row = plan_pose_result(planner, current, target, base, limits)
        row['phase'] = phase
        diagnostics['steps'].append(row)
        if result is None:
            diagnostics.update(accepted=False, failed_phase=phase, reason=row['reason'])
            return (None, diagnostics)
        results.append(result)
        current = np.asarray(result['position'][-1], dtype=np.float32).copy()
    diagnostics.update(accepted=True, total_controls=sum((len(r['position']) for r in results)))
    return (results, diagnostics)
