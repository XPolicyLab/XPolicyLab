"""Nominal sampled IK/budget audit for xylophone tool-point trajectories.

This does not check world collisions or guarantee actual control duration.
Policy uses continuous local solutions; failed checks never imply task success.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import matrix_quat, pose_matrix, bounded_anchor_pose
from .xylophone_geometry import rigid_transform



def tool_translation_step(current, goal, anchor, translation, slow_distance=0.,
                          slow_translation=.003):
    """Fast free approach, slow final tool-point distance; no boundary jump."""
    c,g=pose_matrix(current),pose_matrix(goal)
    anchor=np.asarray(anchor,float)
    distance=np.linalg.norm(c[:3,3]+c[:3,:3]@anchor-g[:3,3]-g[:3,:3]@anchor)
    if slow_distance<=0:return translation
    if distance<=slow_distance+1e-8:return min(translation,slow_translation)
    return min(translation,max(1e-8,distance-slow_distance))

def audit_tool_path(arm, start_ee, targets, ee_anchor, initial_joints,
                    remaining_steps, reserve_steps=60, translation=.01,
                    rotation=.08, dwell=2, position_tolerance=.002,
                    orientation_tolerance=.035, target_translations=None,
                    target_dwells=None, ik_restarts_on_failure=0,
                    target_slow_distances=None, max_joint_step=None, joint_limit_margin=0.,
                    local_seed_retries=False):
    if (type(remaining_steps) is not int or type(reserve_steps) is not int
            or type(dwell) is not int or reserve_steps<0 or dwell<0
            or remaining_steps<=0 or not targets):
        raise ValueError("Nonempty path and explicit integer budget required")
    if not all(np.isfinite(v) and v>0 for v in
               (translation,rotation,position_tolerance,orientation_tolerance)):
        raise ValueError("Positive motion/error bounds required")
    if (max_joint_step is not None and (not np.isfinite(max_joint_step) or max_joint_step<=0)) or not np.isfinite(joint_limit_margin) or joint_limit_margin<0:
        raise ValueError('Finite positive continuity bound and nonnegative joint margin required')
    anchor=np.asarray(ee_anchor,float)
    if anchor.shape!=(3,) or not np.isfinite(anchor).all():
        raise ValueError("Finite bound tool point required")
    start=rigid_transform(start_ee)
    frames=[rigid_transform(t) for t in targets]
    translations=([translation]*len(targets) if target_translations is None else target_translations)
    dwells=([dwell]*len(targets) if target_dwells is None else target_dwells)
    if (len(translations)!=len(targets) or len(dwells)!=len(targets)
            or any(not np.isfinite(v) or v<=0 for v in translations)
            or any(type(v) is not int or v<0 for v in dwells)
            or type(ik_restarts_on_failure) is not int or not 0<=ik_restarts_on_failure<=4):
        raise ValueError('Per-target timing and bounded IK restart count required')
    slow_distances=([0.]*len(targets) if target_slow_distances is None else target_slow_distances)
    if len(slow_distances)!=len(targets) or any(not np.isfinite(x) or x<0 for x in slow_distances):
        raise ValueError("Finite nonnegative slow-zone distances required")
    current=np.r_[start[:3,3],matrix_quat(start[:3,:3])]
    samples=[]; seed=np.asarray(initial_joints,float)
    maximum_joint_step=0.; minimum_joint_margin=float('inf')
    overhead=sum(dwells)+reserve_steps
    if overhead>=remaining_steps:
        return dict(accepted=False,reason="budget",nominal_steps=overhead,
                    collision_checked=False)
    for index,target in enumerate(frames):
        goal=np.r_[target[:3,3],matrix_quat(target[:3,:3])]
        while True:
            actual=pose_matrix(current)
            angle=Rotation.from_matrix(actual[:3,:3].T@target[:3,:3]).magnitude()
            if np.linalg.norm(actual[:3,3]-target[:3,3])<1e-7 and angle<1e-7:
                break
            if len(samples)+overhead+1>remaining_steps:
                return dict(accepted=False,reason="budget",target_index=index,
                            nominal_steps=len(samples)+overhead+1,collision_checked=False)
            step_size=tool_translation_step(current,goal,anchor,translations[index],slow_distances[index])
            current=bounded_anchor_pose(current,goal,anchor,step_size,rotation)
            sample=pose_matrix(current)
            fit=arm.inverse(sample,initial=seed,restarts=0)
            errors=np.array([fit["position_error_m"],fit["orientation_error_rad"]])
            if (local_seed_retries and np.isfinite(errors).all()
                    and (errors[0]>position_tolerance or errors[1]>orientation_tolerance)):
                # Escape the exactly-zero numerical stationary point using two
                # nearby deterministic starts, never a global IK branch search.
                for offset in (-.05,.05):
                    nearby=seed.copy();nearby[min(1,len(nearby)-1)]+=offset
                    trial=arm.inverse(sample,initial=nearby,restarts=0)
                    e=np.array([trial['position_error_m'],trial['orientation_error_rad']])
                    joints=np.asarray(trial['joints'],float)
                    connected=(max_joint_step is None or np.max(np.abs(joints-seed))<=max_joint_step)
                    within=(not hasattr(arm,'lower') or
                        np.min(np.minimum(joints-arm.lower,arm.upper-joints))>=joint_limit_margin)
                    if (np.isfinite(e).all() and np.isfinite(joints).all()
                            and e[0]<=position_tolerance and e[1]<=orientation_tolerance
                            and connected and within):
                        fit=trial;errors=e;break
            if (ik_restarts_on_failure and np.isfinite(errors).all()
                    and (errors[0]>position_tolerance or errors[1]>orientation_tolerance)):
                # Local-search failure near a singularity is not proof of
                # unreachability. Keep the sample and its geometric bounds.
                fit=arm.inverse(sample,initial=seed,restarts=ik_restarts_on_failure)
                errors=np.array([fit['position_error_m'],fit['orientation_error_rad']])
            if not np.isfinite(errors).all() or errors[0]>position_tolerance or errors[1]>orientation_tolerance:
                return dict(accepted=False,reason="sample_ik",target_index=index,
                            sample_index=len(samples),position_error_m=float(errors[0]),
                            orientation_error_rad=float(errors[1]),collision_checked=False)
            solved=np.asarray(fit["joints"],float)
            if not np.isfinite(solved).all():
                return dict(accepted=False,reason="invalid_joint_solution",collision_checked=False)
            # A new IK root is not a connected executable path. Use physical
            # joint coordinates, not wrapped angles across a bounded stop.
            delta=float(np.max(np.abs(solved-seed)))
            maximum_joint_step=max(maximum_joint_step,delta)
            if max_joint_step is not None and delta>max_joint_step:
                return dict(accepted=False,reason="joint_discontinuity",target_index=index,
                            sample_index=len(samples),joint_step_rad=delta,collision_checked=False)
            if hasattr(arm,'lower') and hasattr(arm,'upper'):
                margin=float(np.min(np.minimum(solved-arm.lower,arm.upper-solved)))
                minimum_joint_margin=min(minimum_joint_margin,margin)
                if margin<joint_limit_margin:
                    return dict(accepted=False,reason="joint_limit_margin",target_index=index,
                                sample_index=len(samples),margin_rad=margin,collision_checked=False)
            seed=solved
            samples.append(dict(ee=sample,joints=seed.copy(),target_index=index))
    return dict(accepted=True,reason="nominal_sampled_ik_only",samples=samples,
                nominal_steps=len(samples)+overhead,dwell_steps=sum(dwells),
                reserve_steps=reserve_steps,collision_checked=False,
                duration_verified=False,max_joint_step_rad=maximum_joint_step,
                min_joint_margin_rad=None if not np.isfinite(minimum_joint_margin) else minimum_joint_margin)
