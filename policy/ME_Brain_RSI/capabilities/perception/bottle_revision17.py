"""B17 bounded, geometry-qualified alternatives on top of B15.

Independent implementation using RGB fits, proprioception and public X5/table
geometry. Medium NCC is NOT sufficient alone: require two observations at
distinct measured poses, blocked fingers and a moving/stationary contrast.
No new pregrasp visual servo, transport vision, oracle or success signal.
"""
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import quat_matrix
from .policy import Waypoint


def tcp(policy, side):
    pose=np.asarray(policy.robot_state[f"{side}_ee_pose"],float)
    return pose[:3]+quat_matrix(pose[3:])@np.asarray(policy.calibration["tool_offset_in_ee"])


def stat(policy, key):
    policy.b17_stats[key]=policy.b17_stats.get(key,0)+1


def attachment_probe(policy,pending,strong,robot_ok,moved,stationary):
    """Return True/False, or None when the single extra proof motion is queued."""
    side=pending.get("side","left")
    probe=pending.get("attachment_probe")
    if strong:
        return True
    medium=bool(robot_ok and np.isfinite([moved,stationary]).all()
                and moved>=.60 and moved-stationary>=.30)
    if not medium:
        return False
    current=tcp(policy,side)
    opening=float(np.asarray(policy.robot_state[f"{side}_ee_joint_state"]).reshape(-1)[0])
    if probe is not None:
        delta=current-np.asarray(probe["tcp"])
        accepted=bool(.008<=delta[2]<=.020 and np.linalg.norm(delta[:2])<=.006
                      and abs(opening-probe["opening"])<=.10)
        policy.program_notes["attachment_two_pose"]=dict(accepted=accepted,
            delta=delta.tolist(),first_moved=probe["moved"],second_moved=moved,
            first_stationary=probe["stationary"],second_stationary=stationary)
        stat(policy,"two_pose_accepted" if accepted else "two_pose_rejected")
        return accepted
    if policy.remaining_steps()<120:
        return False
    target=np.asarray(policy.robot_state[f"{side}_ee_pose"],float).copy()
    target[2]+=.012
    stage=Waypoint(side,target,0.,"bottle_attachment_probe",5)
    if not policy._qualify([stage])[0]:
        stat(policy,"probe_ik_rejected")
        return False
    pending["attachment_probe"]=dict(tcp=current.tolist(),opening=opening,
                                     moved=float(moved),stationary=float(stationary))
    pending["proof_extra"]=.012
    policy.plan=[stage];policy.index=policy.age=policy.dwell=0
    policy.reobserve_after_plan=True
    policy.reason="bottle_bounded_second_attachment_observation"
    policy.program_notes["attachment_probe"]=dict(rise=.012,moved=moved,
                                                  stationary=stationary)
    stat(policy,"probes")
    return None


def lying_clearance(params):
    # Bounding circle around the RGB-fitted bottle centre encloses every
    # intermediate orientation. 30 mm support-plane margin, not a fixed .26 m.
    return float(np.hypot(params["length"]/2,params["radius"])+.030)


def remember_held(policy,pending):
    policy.held_bottle=copy.deepcopy(pending)
    policy.held_bottle["replans"]=0
    policy.held_bottle["remaining_plan"]=copy.deepcopy(policy.plan)


def recover_held(policy,remaining):
    """Bounded alternate HEIGHT of the remaining held path, never arbitrary open."""
    held=policy.held_bottle
    if held is None or held["replans"]>=2 or not remaining:
        return False
    side=held["side"]
    if remaining[0].name not in {"bottle_lift_clear","bottle_carry"}:
        return False
    opening=float(np.asarray(policy.robot_state[f"{side}_ee_joint_state"]).reshape(-1)[0])
    if not .1<opening<.9:
        return False
    minimum=lying_clearance(held) if held.get("mode")=="lying" else held["height"]+.03
    current=tcp(policy,side)
    # Retry only while the measured bottle centre still has table clearance.
    # This is not a claim of collision/attachment feedback during transport.
    if current[2]-policy.perception.z<minimum-.005:
        return False
    for offset in (-.012,.012):
        stages=copy.deepcopy(remaining)
        moving=[w for w in stages if w.name in {"bottle_lift_clear","bottle_carry"}]
        if not moving:continue
        safe=True
        for w in moving:
            w.pose[2]+=offset
            anchor=w.pose[:3]+quat_matrix(w.pose[3:])@np.asarray(policy.calibration["tool_offset_in_ee"])
            if anchor[2]-policy.perception.z<minimum:safe=False
        if not safe:continue
        # Release at the final carry pose, so no late return to the stalled height.
        final=moving[-1]
        for w in stages:
            if w.name=="bottle_release":w.pose=final.pose.copy()
        if not policy._qualify(stages)[0]:continue
        held["replans"]+=1
        policy.plan=stages;policy.index=policy.age=policy.dwell=0
        policy.finished=False;policy.reobserve_after_plan=True
        policy.reason="bottle_held_path_bounded_height_replan"
        stat(policy,"held_path_replans")
        return True
    return False


def capture_region(policy,pending,pose):
    """Nominal upright gripper capture volume, not a relaxed global EE tolerance."""
    if pending.get("mode")=="lying" or "radius" not in pending:return False
    side=pending["side"];h=pending["height"];r=pending["radius"]
    expected=policy.pose(side,pending["xy"],h,pending["yaw"],0.)
    rotation=quat_matrix(pose[3:])
    angle=Rotation.from_matrix(rotation.T@quat_matrix(expected[3:])).magnitude()
    actual=pose[:3]+rotation@np.asarray(policy.calibration["tool_offset_in_ee"])
    # Express source centre in CURRENT gripper axes, including orientation error.
    local=rotation.T@(np.r_[pending["xy"],policy.perception.z+h]-actual)
    lateral_margin=.0444-r-.002
    palm_margin=.145-.084-r-.003
    return bool(angle<.08 and lateral_margin>0 and palm_margin>0
                and abs(local[1])<=lateral_margin
                and abs(local[0])<=min(.015,palm_margin)
                and abs(local[2])<=min(.020,.2*h))


def qualify_descent_close(policy):
    if policy.pending_bottle is None or policy.index>=len(policy.plan):return
    stage=policy.plan[policy.index]
    if stage.name!="bottle_insert_fingers" or policy.age<12:return
    pending=policy.pending_bottle
    if pending.get("capture_applied"):return
    side=stage.side
    opening=float(np.asarray(policy.robot_state[f"{side}_ee_joint_state"]).reshape(-1)[0])
    pose=np.asarray(policy.robot_state[f"{side}_ee_pose"],float)
    if opening<.95 or not capture_region(policy,pending,pose):
        # A centred wide bottle may be too close to the palm despite a small
        # EE error. Retreat along the current tool axis (away from the palm),
        # not sideways across the bottle, then recheck the same capture volume.
        if (opening>=.95 and pending.get("mode")!="lying" and
                "radius" in pending and not pending.get("axial_retreat")):
            rotation=quat_matrix(pose[3:]);r=pending["radius"]
            actual=pose[:3]+rotation@np.asarray(policy.calibration["tool_offset_in_ee"])
            local=rotation.T@(np.r_[pending["xy"],policy.perception.z+pending["height"]]-actual)
            expected=policy.pose(side,pending["xy"],pending["height"],pending["yaw"],0.)
            angle=Rotation.from_matrix(rotation.T@quat_matrix(expected[3:])).magnitude()
            shift=local[0]+.010
            if (angle<.08 and abs(local[1])<.0444-r-.002 and
                    abs(local[2])<min(.020,.2*pending["height"]) and
                    local[0]>-(.145-.084-r) and -.010<=shift<=-.004):
                target=pose.copy();target[:3]+=rotation[:,0]*shift
                if policy._qualify([Waypoint(side,target,1.,"bottle_insert_fingers",3)])[0]:
                    stage.pose=target;pending["axial_retreat"]=float(shift)
                    # A retreat shorter than the global 9 mm tolerance must
                    # not immediately advance to the old close waypoint.
                    stage.dwell=3  # B20 uses an explicit bounded capture guard
                    policy.age=policy.dwell=0
                    stat(policy,"axial_capture_retreats")
        pending["capture_count"]=0
        pending.pop("capture_pose",None)
        return
    # Require stable capture geometry in three consecutive observations.
    previous=pending.get("capture_pose")
    stable=previous is not None and np.linalg.norm(pose[:3]-np.asarray(previous)[:3])<.002
    pending["capture_count"]=pending.get("capture_count",0)+1 if stable else 1
    pending["capture_pose"]=pose.tolist()
    if pending["capture_count"]<3:return
    # Retarget descent + close only; proof lift returns to the original expected
    # raised location, so existing attachment geometry remains meaningful.
    stage.pose=pose.copy()
    stage.dwell=3
    for next_stage in policy.plan[policy.index+1:]:
        if next_stage.name=="bottle_close":
            next_stage.pose=pose.copy()
            break
    pending["capture_applied"]=True
    stat(policy,"geometry_qualified_closes")
    policy.program_notes["descent_capture_region"]=dict(actual_pose=pose.tolist(),
        source="current TCP and RGB radius; 2 mm finger/3 mm palm margins")


def plan_steps(policy,stages):
    poses={s:np.asarray(policy.robot_state.get(f"{s}_ee_pose",
           policy.pose(s,[-.3 if s=="left" else .3,-.35],.20,0.,0.)),float)
           for s in ("left","right")}
    count=0
    for w in stages:
        old=poses[w.side];offset=w.anchor_offset
        if offset is None:distance=np.linalg.norm(old[:3]-w.pose[:3])
        else:
            distance=np.linalg.norm(old[:3]+quat_matrix(old[3:])@offset-
                                    w.pose[:3]-quat_matrix(w.pose[3:])@offset)
        angle=Rotation.from_matrix(quat_matrix(old[3:]).T@quat_matrix(w.pose[3:])).magnitude()
        # Use the unchanged controller arrival tolerances and count dwell once,
        # not both a final arrival action and a duplicated whole dwell.
        count+=max(1,int(np.ceil(max(0.,distance-.009)/policy.calibration["max_translation_step"])),
                   int(np.ceil(max(0.,angle-.15)/policy.calibration["max_rotation_step"])))+max(0,w.dwell-1)+1
        poses[w.side]=w.pose
    return int(np.ceil(count*1.15))+10


def estimate_cycle(policy,source,side):
    """Waypoint-shaped estimate, including actual radial rotations and dwell."""
    from .bottle_lying import lying_parameters,upright_rotation
    xy=source.xy;h=float(source.grasp_height)
    base=np.array([-.3 if side=="left" else .3,-.45])
    yaw=float(np.arctan2(*(xy-base)[::-1]))
    from .bottle_policy import exposed_drop_xy
    # Match the executed upright radius (max XY extent), including small tilt.
    radius=float(min(source.size) if h<=.075 else max(source.size))/2
    drop=exposed_drop_xy(radius) if side=="left" else np.array([0.,-.1])
    if drop is None:return 701
    lying=h<=.075
    params=lying_parameters(source) if lying else None
    if lying and params is None:return 701
    if lying:yaw=params["yaw"]
    else:
        tracks=getattr(policy,'bottle_tracks',[])
        near=[t for t in tracks if np.linalg.norm(t['xy']-xy)<.055
              and abs(t['height']-h)<.03]
        if near:yaw+=(0.,np.pi/6,-np.pi/6)[min(int(near[0]['variant']),2)]
    pitch=np.pi/2 if lying else 0.
    hover=h+.14 if lying else 2*h+.045
    stages=[Waypoint(side,policy.pose(side,xy,z,yaw,pitch),g,n,d)
        for z,g,n,d in [(hover,1.,"bottle_approach",3),(h,1.,"bottle_insert_fingers",3),
                       (h,0.,"bottle_close",10),(h+.08,0.,"bottle_lift",5)]]
    height=lying_clearance(params) if lying else h+.08
    try:
        if lying:
            stages.append(Waypoint(side,policy.pose(side,xy,height,yaw,pitch),0.,"bottle_lift_clear",3))
            stages+=upright_rotation(policy,xy,yaw,side,height=height)
        stages+=policy._transport(xy,drop,height,side)
    except ValueError:return 701
    stages.append(Waypoint(side,stages[-1].pose.copy(),1.,"bottle_release",12))
    retreat=stages[-1].pose.copy()
    if side=='left':
        retreat[0]+=.08;retreat[2]+=.02
    else:retreat[2]+=.04
    stages.append(Waypoint(side,retreat,1.,"bottle_retract",3))
    return plan_steps(policy,stages)+(150 if side=="right" else 0)
