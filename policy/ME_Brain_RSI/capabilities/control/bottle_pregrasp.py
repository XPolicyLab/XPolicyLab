"""B16 pre-close RGB correction only; no carried-object feedback.

Independent bounded registration of the selected RGB source. Public geometry
and proprioception only. Bounds are engineering safeguards, not learned fits.
"""
from dataclasses import replace
import numpy as np
from .bottle_attachment import sample_patch,correlation,capture_upper_patch
from .bottle_geometry import fit_bottles
from .geometry import quat_matrix
from .policy import Waypoint

PHASES={"bottle_approach","bottle_insert_fingers","bottle_close"}
INTERVAL=8
MAX_SHIFT=.024
DEADBAND=.0045

def initialize(policy,source):
    p=policy.pending_bottle
    reference=p.get("lying_patch") if p.get("mode")=="lying" else p.get("upper_patch")
    p["pregrasp"]=dict(origin=np.asarray(p["xy"]).copy(),size=np.asarray(source.size).copy(),
        reference=reference,last_step=policy.step,last_phase=None,misses=0,
        corrections=0,travel=0.,start_step=policy.step,checks=0,last_tcp=None)

def texture_location(reference,rgb,camera,expected_delta):
    """Absolute original-texture registration, never rolling-template drift."""
    if reference is None:return None
    expected_delta=np.asarray(expected_delta)
    same=correlation(reference["patch"],sample_patch(rgb,camera,
        reference["quad"]+np.r_[expected_delta,0.]))
    if same>=.90:return dict(delta=expected_delta.copy(),correlation=same,kind="stationary_texture")
    scores=[]
    for x in np.arange(-.024,.025,.004):
        for y in np.arange(-.024,.025,.004):
            delta=np.array([x,y])
            if np.linalg.norm(delta)>MAX_SHIFT:continue
            score=correlation(reference["patch"],sample_patch(rgb,camera,reference["quad"]+np.r_[delta,0.]))
            scores.append((score,delta))
    scores.sort(key=lambda item:item[0],reverse=True)
    best,delta=scores[0]
    other=max((s for s,d in scores if np.linalg.norm(d-delta)>.010),default=-1.)
    if best<.80 or best-other<.05:return None
    return dict(delta=delta.copy(),correlation=best,peak_margin=best-other,kind="translated_texture")

def choose_geometry(pending,fits):
    v=pending["pregrasp"];origin=v["origin"]
    near=[f for f in fits if np.linalg.norm(np.asarray(f["xy"])-origin)<.055]
    if len(near)>1:return dict(status="unknown",reason="ambiguous_nearby_bottles")
    if not near:return dict(status="unknown",reason="silhouette_occluded_or_missing")
    f=near[0];size=np.sort(np.asarray(f["extent"][:2]))
    if abs(f["grasp_height"]-pending["height"])>.015 or np.max(abs(size-np.sort(v["size"])))>.025:
        return dict(status="changed",reason="bottle_shape_or_pose_changed")
    if pending.get("mode")=="lying":
        yaw=f["yaw"]+(np.pi/2 if f["extent"][1]>f["extent"][0] else 0.)
        error=abs((yaw-pending["yaw"]+np.pi/2)%np.pi-np.pi/2)
        if error>.15:return dict(status="changed",reason="lying_axis_changed")
    xy=np.asarray(f["xy"],float)
    if np.linalg.norm(xy-origin)>MAX_SHIFT:
        return dict(status="changed",reason="source_shift_outside_correction_bound")
    return dict(status="ok",xy=xy,kind="current_rgb_shape")

def observe(policy,rgb):
    p=policy.pending_bottle;v=p["pregrasp"]
    texture=texture_location(v["reference"],rgb,policy.perception.camera,p["xy"]-v["origin"])
    if texture is not None and texture["kind"]=="stationary_texture":
        return dict(status="ok",xy=v["origin"]+texture["delta"],kind=texture["kind"],
                    correlation=texture["correlation"])
    fits=fit_bottles(policy.perception.camera,policy.perception.z,rgb)
    shape=choose_geometry(p,fits)
    if shape["status"]=="changed":return shape
    if texture is not None:
        xy=v["origin"]+texture["delta"]
        if shape["status"]=="ok" and np.linalg.norm(shape["xy"]-xy)>.01:
            return dict(status="unknown",reason="texture_shape_disagreement")
        if shape.get("reason")=="ambiguous_nearby_bottles":return shape
        return dict(status="ok",xy=xy,kind=texture["kind"],correlation=texture["correlation"])
    return shape

def abort_open(policy,reason):
    """Vertical withdrawal only; never move sideways at bottle-body height."""
    p=policy.pending_bottle;side=p["side"]
    pose=np.asarray(policy.robot_state[f"{side}_ee_pose"],float).copy()
    opening=float(np.asarray(policy.robot_state[f"{side}_ee_joint_state"]).reshape(-1)[0])
    tcp=pose[:3]+quat_matrix(pose[3:])@np.asarray(policy.calibration["tool_offset_in_ee"])
    rise=max(.06,policy.perception.z+2*p["height"]+.045-tcp[2])
    retreat=pose.copy();retreat[2]+=rise
    stage=Waypoint(side,retreat,1.,"bottle_visual_retreat",3)
    safe=opening>=.9 and rise<=.25 and policy._qualify([stage])[0]
    policy.program_notes["pregrasp_visual"]["decision"]="withdraw" if safe else "hold"
    policy.program_notes["pregrasp_visual"]["reason"]=reason
    if safe:
        policy.plan=[stage];policy.index=policy.age=policy.dwell=0
        policy.finish_source("failed");policy.pending_bottle=None
        policy.finished=False;policy.reobserve_after_plan=True
        policy.reason="bottle_visual_withdraw_reobserve"
    else:
        policy.safe_halt=True;policy.finished=True;policy.plan=[]
        policy.reason="bottle_visual_uncertain_hold"
    return not safe

def correct(policy,xy,rgb):
    p=policy.pending_bottle;v=p["pregrasp"];delta=xy-p["xy"]
    if not np.isfinite(xy).all() or np.linalg.norm(xy-v["origin"])>MAX_SHIFT:
        return False
    if v["corrections"]>=3 or v["travel"]+np.linalg.norm(delta)>.04:return False
    stages=[]
    for w in policy.plan[policy.index:]:
        pose=w.pose.copy();pose[:2]+=delta
        stages.append(replace(w,pose=pose))
    try:
        post=[]
        if p.get("mode")=="lying":
            from .bottle_lying import upright_rotation,CARRY_HEIGHT
            post=[Waypoint(p["side"],policy.pose(p["side"],xy,CARRY_HEIGHT,p["yaw"],np.pi/2),
                           0.,"bottle_lift_clear",3)]
            post+=upright_rotation(policy,xy,p["yaw"],p["side"])
        post+=policy._transport(xy,p["drop"],.26 if p.get("mode")=="lying" else p["height"]+.08,p["side"])
        if p["side"]=="right":
            d=p["drop"]-np.array([.3,-.45]);yaw=float(np.arctan2(d[1],d[0]))
            height=p["length"]/2 if p.get("mode")=="lying" else p["height"]
            post.append(Waypoint("right",policy.pose("right",p["drop"],height,yaw,0.),
                                 0.,"bottle_stage_lower",5))
    except ValueError:return False
    if not policy._qualify(stages+post)[0]:return False
    policy.plan=policy.plan[:policy.index]+stages
    policy.dwell=0  # Age is deliberately NOT reset by corrections.
    p["xy"]=xy.copy()
    if "behind" in p:p["behind"]+=delta
    if policy.active_track is not None:policy.bottle_tracks[policy.active_track]["xy"]=xy.copy()
    if p.get("mode")=="lying":
        old=p.get("lying_patch")
        if old is not None:
            quad=old["quad"]+np.r_[delta,0.]
            patch=sample_patch(rgb,policy.perception.camera,quad)
            p["lying_patch"]=None if patch is None else dict(quad=quad,patch=patch)
    else:
        p["upper_patch"]=capture_upper_patch(rgb,policy.perception.camera,xy,p["height"],
            float(max(v["size"])/2),policy.perception.z,p["yaw"])
    v["corrections"]+=1;v["travel"]+=float(np.linalg.norm(delta))
    return True

def before_action(policy,rgb):
    """True: hold this observation. False: run the existing controller."""
    p=policy.pending_bottle
    if p is None or "pregrasp" not in p or policy.safe_halt or policy.index>=len(policy.plan):return False
    w=policy.plan[policy.index]
    if w.name not in PHASES or (w.name=="bottle_close" and policy.age>0):return False
    v=p["pregrasp"]
    pose=np.asarray(policy.robot_state[f"{w.side}_ee_pose"])
    tcp=pose[:3]+quat_matrix(pose[3:])@np.asarray(policy.calibration["tool_offset_in_ee"])
    movement=0. if v["last_tcp"] is None else np.linalg.norm(tcp-v["last_tcp"])
    trigger=.006 if w.name=="bottle_insert_fingers" else .024
    # At most eight actions; descent also checks after 6 mm actual TCP travel.
    if (v["last_phase"]==w.name and policy.step-v["last_step"]<INTERVAL
            and movement<trigger and not v["misses"]):return False
    v["last_phase"]=w.name;v["last_step"]=policy.step;v["checks"]+=1
    v["last_tcp"]=tcp.copy()
    evidence=observe(policy,rgb)
    note={k:(x.tolist() if isinstance(x,np.ndarray) else x) for k,x in evidence.items()}
    note.update(phase=w.name,step=policy.step,checks=v["checks"],corrections=v["corrections"],
                carry_feedback=False)
    policy.program_notes["pregrasp_visual"]=note
    if policy.step-v["start_step"]>180:return abort_open(policy,"pregrasp_visual_time_bound")
    if evidence["status"]=="changed":return abort_open(policy,evidence["reason"])
    if evidence["status"]!="ok":
        v["misses"]+=1
        if v["misses"]>=3:return abort_open(policy,"three_unobservable_checks")
        note["decision"]="hold_for_current_rgb";return True
    v["misses"]=0
    delta=evidence["xy"]-p["xy"];distance=float(np.linalg.norm(delta))
    note["shift_m"]=distance
    if distance<=DEADBAND:
        note["decision"]="retain_verified_target";return False
    pose=np.asarray(policy.robot_state[f"{w.side}_ee_pose"])
    tcp=pose[:3]+quat_matrix(pose[3:])@np.asarray(policy.calibration["tool_offset_in_ee"])
    if w.name=="bottle_close" or tcp[2]<policy.perception.z+2*p["height"]+.03:
        return abort_open(policy,"shift_below_clearance_replan_above")
    if not correct(policy,evidence["xy"],rgb):
        return abort_open(policy,"correction_bound_or_ik_rejected")
    note["decision"]="bounded_xy_correction";note["corrections"]=v["corrections"]
    return False
