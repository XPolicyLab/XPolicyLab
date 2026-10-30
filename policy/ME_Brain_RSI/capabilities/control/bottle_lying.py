"""B-14 RGB lying pickup: short proof lift, then rotation and disposal/transfer.

Public geometry plus current RGB and own robot state only. Top-face texture
tracking is a separate positive attachment test, not a relaxed silhouette gate.
"""
from dataclasses import replace
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import quat_matrix
from .policy import Waypoint
from .bottle_geometry import fit_bottles
from .bottle_attachment import sample_patch,correlation
from .bottle_handoff import staging_candidates
from .bottle_revision17 import lying_clearance,remember_held
from .bottle_revision18 import attachment_decision

CARRY_HEIGHT=.26

def lying_parameters(source):
    size=np.asarray(source.size,float);h=source.grasp_height
    if (size.shape!=(2,) or h is None or not np.isfinite(np.r_[size,h,source.angle]).all()
            or not .02<h<.05):return None
    width=float(size.min());length=float(size.max())
    if not .04<width<.08 or not .14<length<.27 or length/width<2.:return None
    yaw=float(source.angle+(np.pi/2 if size[1]>size[0] else 0.))
    yaw=(yaw+np.pi/2)%np.pi-np.pi/2
    return dict(height=float(h),radius=width/2,yaw=yaw,length=length)

def upright_rotation(policy,xy,yaw,side='left',height=CARRY_HEIGHT):
    base_x=-.3 if side=='left' else .3
    radial=float(np.arctan2(xy[1]+.45,xy[0]-base_x))
    change=(radial-yaw+np.pi)%(2*np.pi)-np.pi
    offset=np.asarray(policy.calibration['tool_offset_in_ee'])
    return [Waypoint(side,policy.pose(side,xy,height,yaw+t*change,
                      (1-t)*np.pi/2),0.,'bottle_carry',1,offset.copy())
            for t in np.linspace(0,1,13)[1:]]

def capture_lying_texture(policy,source,params,rgb):
    # Two strips away from the central fingers. Use the more textured patch;
    # do not use a score/layout label to choose a patch or an asset.
    axis=np.array([np.cos(params['yaw']),np.sin(params['yaw'])])
    lateral=np.array([-axis[1],axis[0]])
    candidates=[]
    for end in (-1,1):
        quad=np.array([np.r_[source.xy+axis*end*a*params['length']+
            lateral*b*params['radius'],policy.perception.z+2*params['height']]
            for a,b in [(.20,-.7),(.42,-.7),(.42,.7),(.20,.7)]])
        patch=sample_patch(rgb,policy.perception.camera,quad)
        if patch is not None:
            texture=float(np.std(patch-patch.mean(axis=(0,1))))
            if texture>=8:candidates.append((texture,dict(quad=quad,patch=patch)))
    if not candidates:return None
    best=max(candidates,key=lambda x:x[0])[1]
    return dict(best,refs=[item[1] for item in candidates])

def plan_lying_pick(policy,source,rgb):
    params=lying_parameters(source)
    if params is None:return False
    from .bottle_policy import exposed_drop_xy
    side='left' if source.xy[0]<.10 else 'right'
    if side=='right':
        upright_source=replace(source,size=np.array([2*params['radius']]*2))
        objects={k:v for k,v in policy.current_bottle_objects.items() if v is not source}
        drop=next((xy for xy in staging_candidates(upright_source,objects)
            if policy.receiver_feasible(xy,params['length']/2,params['radius'])),None)
    else:drop=exposed_drop_xy(params['radius'])
    if drop is None:return False
    h=params['height'];xy=source.xy
    carry=lying_clearance(params)
    # pi-symmetric wrist solutions close across the SAME bottle axis.
    for yaw in (params['yaw'],params['yaw']+np.pi):
        stages=[Waypoint(side,policy.pose(side,xy,z,yaw,np.pi/2),g,name,dwell)
            for z,g,name,dwell in [(h+.14,1.,'bottle_approach',3),
               (h,1.,'bottle_insert_fingers',3),(h,0.,'bottle_close',10),
               (h+.08,0.,'bottle_lift',5)]]
        try:
            post=[Waypoint(side,policy.pose(side,xy,carry,yaw,np.pi/2),0.,'bottle_lift_clear',3)]
            post+=upright_rotation(policy,xy,yaw,side,height=carry)+policy._transport(xy,drop,carry,side)
            if side=='right':
                delta=drop-np.array([.3,-.45]);angle=float(np.arctan2(delta[1],delta[0]))
                post.append(Waypoint(side,policy.pose(side,drop,params['length']/2,angle,0.),0.,'bottle_stage_lower',5))
        except ValueError:continue
        ok,reports=policy._qualify(stages+post)
        if not ok:continue
        policy.plan=stages
        policy.pending_bottle=dict(params,xy=xy.copy(),drop=drop,side=side,mode='lying')
        policy.pending_bottle['yaw']=yaw
        policy.pending_bottle['lying_patch']=capture_lying_texture(policy,source,params,rgb)
        policy.program_notes=dict(acting_arm=side,grasp_mode='lying_top_down',
            bottle_path_ik=reports,collision_checked=False,proof_lift=.08,
            attachment_required='own blocked close/TCP and raised shape or moved texture')
        policy.reobserve_after_plan=True;policy.reason='bottle_lying_picking'
        return True
    return False

def after_lying_lift(policy,rgb):
    pending=policy.pending_bottle;xy=pending['xy'];h=pending['height'];yaw=pending['yaw'];side=pending['side']
    fits=fit_bottles(policy.perception.camera,policy.perception.z+.08+pending.get('proof_extra',0.),rgb,color_proposals=False)
    matching=[f for f in fits if np.linalg.norm(f['xy']-xy)<.035 and
              abs(f['grasp_height']-h)<.01 and abs(min(f['extent'][:2])/2-pending['radius'])<.01]
    state=policy.robot_state
    pose=np.asarray(state.get(f'{side}_ee_pose',[]),float)
    opening=np.asarray(state.get(f'{side}_ee_joint_state',[]),float).reshape(-1)
    robot_ok=False;texture_ok=False;scores={}
    if pose.shape==(7,) and len(opening)==1 and np.isfinite(np.r_[pose,opening]).all() and np.linalg.norm(pose[3:])>.5:
        rotation=quat_matrix(pose[3:]);tcp=pose[:3]+rotation@np.asarray(policy.calibration['tool_offset_in_ee'])
        expected=policy.pose(side,xy,h+.08+pending.get('proof_extra',0.),yaw,np.pi/2)
        angular=Rotation.from_matrix(rotation.T@quat_matrix(expected[3:])).magnitude()
        robot_ok=bool(.1<opening[0]<.9 and np.linalg.norm(tcp[:2]-xy)<.03
                      and abs(tcp[2]-policy.perception.z-h-.08-pending.get('proof_extra',0.))<.015 and angular<.18)
        ref=pending.get('lying_patch')
        if ref is not None and robot_ok:
            delta=tcp-np.r_[xy,policy.perception.z+h]
            moved=correlation(ref['patch'],sample_patch(rgb,policy.perception.camera,ref['quad']+delta))
            stationary=correlation(ref['patch'],sample_patch(rgb,policy.perception.camera,ref['quad']))
            scores=dict(moved=moved,stationary=stationary)
            texture_ok=moved>=.75 and moved-stationary>=.20
    confirmed=attachment_decision(policy,pending,rgb,
        (pending.get('lying_patch') or {}).get('refs',[]),
        bool(matching or texture_ok) and robot_ok,robot_ok,
        scores.get('moved',-1.),scores.get('stationary',-1.))
    if confirmed is None:return True
    carry=lying_clearance(pending)
    policy.program_notes.update(raised_shape_rgb_confirmed=bool(matching),
        lying_texture=scores,lying_robot_lift_confirmed=robot_ok,grasp_rgb_confirmed=confirmed)
    if confirmed:
        policy.plan=[Waypoint(side,policy.pose(side,xy,carry,yaw,np.pi/2),0.,'bottle_lift_clear',3)]
        policy.plan+=upright_rotation(policy,xy,yaw,side,height=carry)+policy._transport(xy,pending['drop'],carry,side)
        base=np.array([-.3 if side=='left' else .3,-.45])
        end=pending['drop']-base;angle=float(np.arctan2(end[1],end[0]))
        if side=='right':
            policy.add(side,pending['drop'],pending['length']/2,0.,'bottle_stage_lower',5,angle=angle,pitch=0.)
            policy.add(side,pending['drop'],pending['length']/2,1.,'bottle_stage_open',12,angle=angle,pitch=0.)
            policy.add(side,pending['drop'],carry+.03,1.,'bottle_stage_retreat',angle=angle,pitch=0.)
            policy.staged_hint=pending['drop'].copy();policy.handoff_programs+=1
            policy.return_home(sides=['right'])
            policy.reason='bottle_lying_support_transfer_reobserve_required'
        else:
            policy.add(side,pending['drop'],carry,1.,'bottle_release',12,angle=angle,pitch=0.)
            policy.add(side,pending['drop']+[.08,0],carry+.02,1.,'bottle_retract',angle=angle,pitch=0.)
            policy.drop_programs+=1;policy.reason='bottle_lying_disposal_not_success_claim'
    else:
        policy.plan=[]
        for z,g,name,dwell in [(h,0.,'bottle_recovery_lower',5),
                (h,1.,'bottle_recovery_open',10),(h+.14,1.,'bottle_recovery_retreat',3)]:
            policy.add(side,xy,z,g,name,dwell,angle=yaw,pitch=np.pi/2)
        policy.reason='bottle_lying_lift_not_confirmed'
    policy.finish_source('staged' if confirmed and side=='right' else 'released_unverified' if confirmed else 'failed')
    if confirmed:remember_held(policy,pending)
    policy.pending_bottle=None;policy.reobserve_after_plan=True
    return True
