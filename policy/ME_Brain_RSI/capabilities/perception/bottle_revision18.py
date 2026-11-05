"""B18: bounded RGB attachment checks; uncertainty is not empty grasp.

Independent OpenCV implementation; only RGB, own pose and public geometry.
No reward, object truth, depth or scene IDs. New appearance evidence requires
multiple patches and consecutive observations; proprioception alone never
authorizes carrying. Limits are hypotheses to test, not calibrated confidence.
"""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import quat_matrix
from .policy import Waypoint
from .bottle_attachment import sample_patch
from .bottle_revision17 import tcp
from .bottle_revision19 import occluded_source_motion


def references(rgb, camera, quads):
    result=[]
    for quad in quads:
        patch=sample_patch(rgb,camera,quad)
        if patch is not None and np.std(patch-patch.mean(axis=(0,1)))>=8:
            result.append(dict(quad=np.asarray(quad).copy(),patch=patch.copy()))
    return result


def upright_references(rgb,camera,reference):
    if reference is None:return []
    q=reference['quad']
    # Separate left/right upper-body bands; no reference refresh after grasp.
    quads=[]
    for lo,hi in ((0.,.45),(.55,1.)):
        quads.append(np.array([q[0]+(q[1]-q[0])*lo,q[0]+(q[1]-q[0])*hi,
                               q[3]+(q[2]-q[3])*hi,q[3]+(q[2]-q[3])*lo]))
    return references(rgb,camera,quads)


def search_patch(reference,rgb,camera,delta):
    """Search ±5 rectified pixels, ±12 degrees and ±8% scale, not whole image."""
    uv=np.asarray(camera.project(reference['quad']+delta),np.float32)
    empty=dict(score=-1.,offset=[0.,0.],angle=0.,scale=1.,valid=False)
    if uv.shape!=(4,2) or not np.isfinite(uv).all() or abs(cv2.contourArea(uv))<100:return empty
    center=uv.mean(axis=0);template=reference['patch'].astype(np.float32)
    if not np.isfinite(template).all() or np.std(template-template.mean(axis=(0,1)))<8:return empty
    pad=5
    target=np.array([[pad,pad],[39+pad,pad],[39+pad,47+pad],[pad,47+pad]],np.float32)
    best=empty
    for angle in (-12.,0.,12.):
        theta=np.deg2rad(angle);rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        for scale in (.92,1.,1.08):
            points=((uv-center)@rot.T*scale+center).astype(np.float32)
            transform=cv2.getPerspectiveTransform(points,target)
            inv=np.linalg.inv(transform)
            outer=cv2.perspectiveTransform(np.array([[[0.,0.],[49.,0.],[49.,57.],[0.,57.]]],np.float32),inv)[0]
            if (outer.min(axis=0)<1).any() or (outer.max(axis=0)>[rgb.shape[1]-2,rgb.shape[0]-2]).any():continue
            window=cv2.warpPerspective(rgb,transform,(50,58)).astype(np.float32)
            scores=cv2.matchTemplate(window,template,cv2.TM_CCOEFF_NORMED)
            _,value,_,(x,y)=cv2.minMaxLoc(scores)
            patch=window[y:y+48,x:x+40]
            if np.max(np.abs(patch.mean(axis=(0,1))-template.mean(axis=(0,1))))>65:continue
            if not np.isfinite(value) or value<=best['score']:continue
            actual=cv2.perspectiveTransform(np.array([[[x+19.5,y+23.5]]],np.float32),inv)[0,0]
            best=dict(score=float(value),offset=(actual-center).tolist(),angle=angle,scale=scale,valid=True)
    return best


def image_evidence(refs,rgb,camera,delta):
    rows=[]
    for i,ref in enumerate(refs):
        moved=search_patch(ref,rgb,camera,delta)
        stationary=search_patch(ref,rgb,camera,np.zeros(3))
        rows.append(dict(id=i,moved=moved['score'],stationary=stationary['score'],
                         offset=moved['offset'],angle=moved['angle'],scale=moved['scale'],
                         moved_valid=moved.get('valid',True),stationary_valid=stationary.get('valid',True),
                         predicted_center=np.asarray(camera.project(ref['quad']+delta)).mean(axis=0).tolist()))
    positive=[r for r in rows if r['moved_valid'] and r['stationary_valid']
              and r['moved']>=.65 and r['moved']-r['stationary']>=.20]
    coherent=[]
    for a in positive:
        peers=[b['id'] for b in positive if np.linalg.norm(np.array(a['offset'])-b['offset'])<=7
               and abs(a['angle']-b['angle'])<=12 and abs(a['scale']-b['scale'])<=.09]
        if len(peers)>len(coherent):coherent=peers
    stationary_ids=[r['id'] for r in rows if r['stationary_valid'] and r['moved_valid'] and r['stationary']>=.80 and r['stationary']-r['moved']>=.25]
    return dict(patches=rows,positive_ids=coherent,positive=len(coherent)>=2,
                stationary=len(stationary_ids)>=2)


def single_patch_motion(history,probe):
    """One visible patch requires two actual poses, high appearance and contrast.

    Invalid stationary observations are unknown, never a score of minus one
    that makes the moving score win. At least one valid differential is needed.
    """
    if probe is None or len(history)<3:return False
    current=history[-1]
    for before in history[:-1]:
        if before['step']>probe['step'] or not before['robot_ok']:continue
        for a in before['patches']:
            for b in current['patches']:
                if a['id']!=b['id'] or not a['moved_valid'] or not b['moved_valid']:continue
                if min(a['moved'],b['moved'])<.80:continue
                contrasted=any(x['stationary_valid'] and x['moved']-x['stationary']>=.25 for x in (a,b))
                if not contrasted:continue
                # Both measured image positions must follow the predicted motion.
                motion=np.array(b['predicted_center'])-a['predicted_center']
                residual=np.array(b['offset'])-a['offset']
                measured=motion+residual
                if (np.linalg.norm(motion)>=2 and np.linalg.norm(residual)<=2
                        and measured@motion>0 and
                        .5<=np.linalg.norm(measured)/np.linalg.norm(motion)<=1.5):
                    return True
    return False


def robot_lift_ok(policy,pending):
    side=pending.get('side','left')
    try:
        pose=np.asarray(policy.robot_state[side+'_ee_pose'],float)
        opening=float(np.asarray(policy.robot_state[side+'_ee_joint_state']).reshape(-1)[0])
        current=tcp(policy,side)
        expected=policy.pose(side,pending['xy'],pending['height']+.08+pending.get('proof_extra',0.),
                             pending['yaw'],np.pi/2 if pending.get('mode')=='lying' else 0.)
        angle=Rotation.from_matrix(quat_matrix(pose[3:]).T@quat_matrix(expected[3:])).magnitude()
        return bool(np.isfinite(np.r_[current,opening,angle]).all() and .1<opening<.9
                    and np.linalg.norm(current[:2]-pending['xy'])<.03
                    and abs(current[2]-policy.perception.z-pending['height']-.08-pending.get('proof_extra',0.))<.015
                    and angle<.18)
    except (KeyError,ValueError,IndexError):return False


def attachment_decision(policy,pending,rgb,refs,strong,robot_ok,moved,stationary):
    """True confirmed, False terminal failure/unknown, None bounded closed hold."""
    side=pending.get('side','left')
    state=pending.setdefault('b18',dict(history=[],probe=None,started=policy.step))
    history=state['history']
    try:
        pose=np.asarray(policy.robot_state[side+'_ee_pose'],float)
        opening=float(np.asarray(policy.robot_state[side+'_ee_joint_state']).reshape(-1)[0])
        if pose.shape!=(7,) or not np.isfinite(np.r_[pose,opening]).all() or np.linalg.norm(pose[3:])<.5:
            raise ValueError('invalid state')
        current=tcp(policy,side)
    except (KeyError,ValueError,IndexError):
        policy.program_notes['b18_attachment']=dict(state='unknown',reason='missing_proprioception')
        return False
    evidence=dict(image_evidence(refs,rgb,policy.perception.camera,
                           current-np.r_[pending['xy'],policy.perception.z+pending['height']]))
    evidence.update(step=policy.step,tcp=current.tolist(),opening=opening,
                    robot_ok=bool(robot_ok),legacy_medium=bool(robot_ok and moved>=.60 and moved-stationary>=.30))
    # Do not count repeated calls on a single observation as temporal evidence.
    if history and history[-1]['step']==policy.step:return None
    history.append(evidence)
    note=dict(state='unknown',reason='appearance_unresolved',observations=len(history),
              probe_used=state['probe'] is not None,**evidence)
    policy.program_notes['b18_attachment']=note
    probe=state['probe']
    probe_ok=True
    if probe is not None:
        delta=current-np.array(probe['tcp'])
        probe_ok=bool(.008<=delta[2]<=.020 and np.linalg.norm(delta[:2])<=.006
                      and abs(opening-probe['opening'])<=.10)
        note['probe_delta']=delta.tolist()
    old=history[-2] if len(history)>1 else None
    temporal=bool(old and old['robot_ok'] and robot_ok and old['positive'] and evidence['positive']
                  and len(set(old['positive_ids'])&set(evidence['positive_ids']))>=2
                  and abs(old['opening']-opening)<=.10)
    legacy_probe=bool(probe and probe['legacy_medium'] and evidence['legacy_medium'] and probe_ok)
    single_motion=bool(probe_ok and single_patch_motion(history,probe))
    occluded_motion=bool(probe_ok and occluded_source_motion(history,probe))
    if robot_ok and probe_ok and (strong or temporal or legacy_probe or single_motion or occluded_motion):
        note.update(state='confirmed',reason='strong_existing' if strong else
                    'two_pose_legacy' if legacy_probe else
                    'repeated_probe_motion_source_occluded' if occluded_motion else
                    'single_visible_patch_two_pose_motion' if single_motion else 'multi_patch_temporal')
        return True
    # An unblocked near-empty close AND strong stationary evidence is a failure.
    if np.isfinite(opening) and opening<=.05 and evidence['stationary']:
        note.update(state='failed',reason='empty_close_and_stationary_source')
        return False
    def stop(reason):
        note.update(state='unknown',reason=reason)
        # One unresolved review is not a permanent target rejection.
        # Existing failed-attempt scheduling changes grasp variant and cooldown.
        # Two unresolved reviews bound repeated ambiguity; actual tries stay true.
        if policy.active_track is not None:
            track=policy.bottle_tracks[policy.active_track]
            track['attachment_unknowns']=track.get('attachment_unknowns',0)+1
            track['attachment_exhausted']=track['attachment_unknowns']>=2
        return False
    if not robot_ok:return stop('proprioception_not_qualified')
    if policy.step-state['started']>65 or len(history)>=4 or policy.remaining_steps()<100:
        return stop('bounded_review_exhausted')
    pose=np.asarray(policy.robot_state[side+'_ee_pose'],float).copy()
    if len(history)==2 and probe is None:
        # No minimum NCC required to obtain an extra observation.
        # 12mm upward only; remain closed, preserve XY/orientation, screen IK.
        target=pose.copy();target[2]+=.012
        nominal_bottom=policy.perception.z+.08
        if current[2]-pending['height']<nominal_bottom-.015:
            return stop('probe_clearance_unqualified')
        stage=Waypoint(side,target,0.,'bottle_attachment_probe',5)
        if not policy._qualify([stage])[0]:return stop('probe_ik_unqualified')
        state['probe']=dict(step=policy.step,tcp=current.tolist(),opening=opening,
                            legacy_medium=evidence['legacy_medium'])
        pending['proof_extra']=.012
        note.update(reason='bounded_probe_without_ncc_gate',probe_used=True)
    else:
        stage=Waypoint(side,pose,0.,'bottle_attachment_review',5)
        note['reason']='closed_hold_for_next_observation'
    policy.plan=[stage];policy.index=policy.age=policy.dwell=0
    policy.finished=False;policy.reobserve_after_plan=True
    policy.reason='bottle_attachment_unknown_review'
    return None
