"""RGB upper-body following evidence, not oracle attachment or task success.

Use a camera-facing public-size billboard for the unoccluded upper body.
Compare its initial texture at the proprioceptive lifted location and at the
stationary source. Require both differential image evidence and fingers that
remain partly open under a close command. No reward/layout/depth is read.
"""
import cv2
import numpy as np
from .geometry import quat_matrix

def upper_quad(camera,xy,grasp_height,radius,table_z):
    xy=np.asarray(xy,float)
    if xy.shape!=(2,) or not np.isfinite(np.r_[xy,grasp_height,radius,table_z]).all():return None
    if not .075<grasp_height<.15 or not .015<radius<.055:return None
    normal=xy-camera.c2w[:2,3];length=np.linalg.norm(normal)
    if length<.05:return None
    normal/=length;lateral=np.array([-normal[1],normal[0]])
    front=xy-normal*radius
    return np.array([np.r_[front+s*lateral*radius*1.3,table_z+z*grasp_height*2]
                     for s,z in [(-1,1.08),(1,1.08),(1,.62),(-1,.62)]])

def sample_patch(rgb,camera,quad):
    if quad is None or rgb.ndim!=3 or rgb.shape[2]!=3:return None
    uv=camera.project(quad)
    if not np.isfinite(uv).all():return None
    if (uv[:,0].min()<2 or uv[:,1].min()<2 or
        uv[:,0].max()>rgb.shape[1]-3 or uv[:,1].max()>rgb.shape[0]-3):return None
    if abs(cv2.contourArea(uv.astype(np.float32)))<100:return None
    target=np.array([[0,0],[39,0],[39,47],[0,47]],np.float32)
    transform=cv2.getPerspectiveTransform(uv.astype(np.float32),target)
    return cv2.warpPerspective(rgb,transform,(40,48)).astype(np.float32)

def partial_follow_scores(reference,rgb,camera,delta):
    """Compare the SAME visible texture coordinates, never image padding.

    Only a moved billboard clipped by the image boundary uses this fallback.
    Reference and stationary billboard still require full visibility. At least
    three quarters of the original texture must remain visible. This is a
    visibility safeguard, not a relaxation of lift/finger/NCC evidence.
    """
    quad=reference['quad']+delta
    uv=camera.project(quad)
    if (not np.isfinite(uv).all() or
        abs(cv2.contourArea(uv.astype(np.float32)))<100):return None
    stationary=sample_patch(rgb,camera,reference['quad'])
    if stationary is None:return None
    target=np.array([[0,0],[39,0],[39,47],[0,47]],np.float32)
    transform=cv2.getPerspectiveTransform(uv.astype(np.float32),target)
    valid=np.zeros(rgb.shape[:2],np.float32);valid[2:-2,2:-2]=1.
    mask=cv2.warpPerspective(valid,transform,(40,48))>=.999
    coverage=float(mask.mean())
    if coverage<.75:return None
    moved=cv2.warpPerspective(rgb,transform,(40,48)).astype(np.float32)
    a=reference['patch'][mask].astype(np.float32)
    a=a-a.mean(axis=0)
    if np.std(a)<8:return None
    def score(b):
        b=b[mask].astype(np.float32);b=b-b.mean(axis=0)
        denominator=np.linalg.norm(a)*np.linalg.norm(b)
        return -1. if denominator<1e-6 else float(np.sum(a*b)/denominator)
    return score(moved),score(stationary),coverage

def correlation(a,b):
    if a is None or b is None:return -1.
    a=a-a.mean(axis=(0,1));b=b-b.mean(axis=(0,1))
    denominator=np.linalg.norm(a)*np.linalg.norm(b)
    if denominator<1e-6:return -1.
    return float(np.sum(a*b)/denominator)

def capture_upper_patch(rgb,camera,xy,grasp_height,radius,table_z,yaw):
    quad=upper_quad(camera,xy,grasp_height,radius,table_z)
    patch=sample_patch(rgb,camera,quad)
    if patch is None or np.std(patch-patch.mean(axis=(0,1)))<8:return None
    return dict(quad=quad,patch=patch,source_tcp=np.r_[xy,table_z+grasp_height],yaw=float(yaw))

def check_upper_following(reference,rgb,camera,state,tool_offset,*,side='left'):
    result=dict(confirmed=False,kind='upper_rgb_following_and_proprioception')
    if side not in ('left','right'):
        return {**result,'reason':'invalid_arm'}
    if reference is None:return {**result,'reason':'no_textured_reference'}
    pose=np.asarray(state.get(f'{side}_ee_pose',[]),float)
    fingers=np.asarray(state.get(f'{side}_ee_joint_state',[]),float).reshape(-1)
    if pose.shape!=(7,) or len(fingers)!=1 or not np.isfinite(np.r_[pose,fingers]).all():
        return {**result,'reason':'missing_proprioception'}
    if np.linalg.norm(pose[3:])<.5:return {**result,'reason':'invalid_orientation'}
    rotation=quat_matrix(pose[3:]);tcp=pose[:3]+rotation@np.asarray(tool_offset)
    delta=tcp-reference['source_tcp']
    yaw=reference['yaw'];expected_axis=np.array([np.cos(yaw),np.sin(yaw),0.])
    angle=float(np.arccos(np.clip(rotation[:,0]@expected_axis,-1,1)))
    result.update(tcp_delta=delta.tolist(),finger_opening=float(fingers[0]),axis_error=angle)
    if not .1<fingers[0]<.9:return {**result,'reason':'no_blocked_close_evidence'}
    if not .05<delta[2]<.12 or np.linalg.norm(delta[:2])>.03 or angle>.18:
        return {**result,'reason':'not_upright_lift_geometry'}
    moved=sample_patch(rgb,camera,reference['quad']+delta)
    stationary=sample_patch(rgb,camera,reference['quad'])
    moved_score=correlation(reference['patch'],moved)
    stationary_score=correlation(reference['patch'],stationary)
    if moved is None:
        partial=partial_follow_scores(reference,rgb,camera,delta)
        if partial is not None:
            moved_score,stationary_score,coverage=partial
            result.update(visible_texture_fraction=coverage,
                          comparison='same_visible_texture_coordinates')
    result.update(moved_correlation=moved_score,stationary_correlation=stationary_score)
    # Conservative hypothesis test, NOT a fitted regressor or success reward.
    result['confirmed']=bool(moved_score>=.75 and moved_score-stationary_score>=.20)
    result['reason']='image_follows_lift' if result['confirmed'] else 'image_evidence_insufficient'
    return result
