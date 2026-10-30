"""B21 choose a qualified direct-left route before requiring right-hand transfer.
RGB source geometry and own TCP state only; no world collision claim.
"""
import numpy as np
from .policy import Waypoint
from .geometry import quat_matrix
from .bottle_grasp import upright_descent_clearance

def choose_source_side(policy, source, variant=0):
    if source.xy[0] < .10:
        return "left"
    # First scope is upright bodies; lying/handoff geometry stays unchanged.
    if source.grasp_height is None or source.grasp_height <= .075:
        return "right"
    right=np.asarray(policy.robot_state.get("right_ee_pose",[]),float)
    opening=np.asarray(policy.robot_state.get("right_ee_joint_state",[]),float).reshape(-1)
    if (right.shape!=(7,) or opening.size!=1 or
            not np.isfinite(np.r_[right,opening]).all() or opening[0]<.95):
        return "right"
    h=float(source.grasp_height);radius=float(max(source.size)/2)
    clearance=upright_descent_clearance(h,radius)
    if clearance is None:return "right"
    from .bottle_policy import exposed_drop_xy
    drop=exposed_drop_xy(radius)
    if drop is None:return "right"
    xy=source.xy
    yaw=float(np.arctan2(xy[1]+.45,xy[0]+.3))+(0.,np.pi/6,-np.pi/6)[min(variant,2)]
    stages=[Waypoint("left",policy.pose("left",xy,z,yaw,0.),g,n,d)
        for z,g,n,d in [(clearance["hover_height"],1.,"bottle_approach",3),
            (h,1.,"bottle_insert_fingers",3),(h,0.,"bottle_close",10),
            (h+.08,0.,"bottle_lift",5)]]
    try:stages+=policy._transport(xy,drop,h+.08,"left")
    except ValueError:return "right"
    # Nominal gripper-to-gripper spacing only, not full arm/world collision.
    offset=np.asarray(policy.calibration["tool_offset_in_ee"])
    right_tcp=right[:3]+quat_matrix(right[3:])@offset
    left=np.asarray(policy.robot_state.get("left_ee_pose",[]),float)
    if left.shape!=(7,) or not np.isfinite(left).all():return "right"
    previous=left[:3]+quat_matrix(left[3:])@offset
    margin=.10+radius
    for stage in stages:
        target=stage.pose[:3]+quat_matrix(stage.pose[3:])@offset
        segment=target-previous
        fraction=np.clip(np.dot(right_tcp-previous,segment)/max(np.dot(segment,segment),1e-12),0.,1.)
        if np.linalg.norm(previous+fraction*segment-right_tcp)<margin:return "right"
        previous=target
    if not policy._qualify(stages)[0]:return "right"
    if policy.estimate_cycle(source,"left")>=policy.estimate_cycle(source,"right"):return "right"
    return "left"
