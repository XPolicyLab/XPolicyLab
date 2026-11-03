"""Episode-observed planar key silhouette; never a supplied object mesh."""
import cv2
import numpy as np
from . import Camera, GeometryUnavailable


def observed_key_tip(observed_outline):
    """Visible distal point in the episode key frame (shaft extends along -Z)."""
    points=np.asarray(observed_outline,float)
    if (points.ndim!=2 or points.shape[1]!=3 or len(points)<3
            or not np.isfinite(points).all() or np.ptp(points[:,2])<=1e-8):
        raise GeometryUnavailable('A resolved episode key outline is required for its tip')
    return points[int(np.argmin(points[:,2]))].copy()


def observe_outline(camera, mask, key_pose):
    mask=np.asarray(mask,dtype=np.uint8)
    pose=np.asarray(key_pose,float)
    if mask.shape!=camera.image.shape[:2] or pose.shape!=(4,4) or not np.isfinite(pose).all():
        raise GeometryUnavailable('Current mask and measured key pose required')
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:raise GeometryUnavailable('No visible key outline')
    contour=max(contours,key=cv2.contourArea)
    if cv2.contourArea(contour)<8:raise GeometryUnavailable('Key outline is unresolved')
    pixels=contour.reshape(-1,2).astype(float)
    origin=camera.transform[:3,3]
    rays=camera.unproject(pixels,np.ones(len(pixels)))-origin
    normal=pose[:3,1] # The initial key face is the observed support-facing plane.
    den=rays@normal
    if np.any(np.abs(den)<1e-7):raise GeometryUnavailable('Grazing key face observation')
    depth=((pose[:3,3]-origin)@normal)/den
    if np.any(depth<=0):raise GeometryUnavailable('Key face is behind camera')
    world=origin+rays*depth[:,None]
    return (world-pose[:3,3])@pose[:3,:3]


def held_outline(rgb,relative,calibration,observed_outline):
    """Compare the episode silhouette's shaft with current wrist RGB.

    This is a planar approximation and visibility check, not full 3D shape
    reconstruction or proof of a stable grasp. Edge-on evidence is rejected.
    """
    points=np.asarray(observed_outline,float)
    if points.ndim!=2 or points.shape[1]!=3 or len(points)<3 or not np.isfinite(points).all():
        return False,{'reason':'episode_outline_unavailable'}
    low,high=np.min(points[:,2]),np.max(points[:,2])
    if high-low<=1e-8:return False,{'reason':'shaft_unresolved'}
    # Exclude the observed bow end using a dimensionless fraction of its length.
    cutoff=low+.75*(high-low)
    clipped=[]
    for start,end in zip(points,np.roll(points,-1,axis=0)):
        inside_start,inside_end=start[2]<=cutoff,end[2]<=cutoff
        if inside_start:clipped.append(start)
        if inside_start!=inside_end:
            clipped.append(start+(end-start)*((cutoff-start[2])/(end[2]-start[2])))
    shaft=np.asarray(clipped)
    if len(shaft)<3:return False,{'reason':'shaft_unresolved'}
    mount,k=calibration
    camera=Camera(rgb,k,mount)
    relative=np.asarray(relative,float)
    try:uv,depth=camera.project(shaft@relative[:3,:3].T+relative[:3,3])
    except GeometryUnavailable:return False,{'reason':'behind_camera'}
    if min(cv2.minAreaRect(uv.astype(np.float32))[1])<2.:
        return False,{'reason':'edge_on_or_unresolved'}
    pred=np.zeros(rgb.shape[:2],np.uint8)
    hull=cv2.convexHull(np.rint(uv).astype(np.int32));cv2.fillConvexPoly(pred,hull,1)
    area=int(pred.sum())
    if area<20:return False,{'reason':'insufficient_projected_area','predicted_area':area}
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    metal=(hsv[:,:,1]<85)&(hsv[:,:,2]>80)
    # Count agreement only inside the projection; surrounding robot pixels
    # cannot inflate support beyond the predicted object's area.
    near=cv2.dilate(metal.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    matched=int((near&pred.astype(bool)).sum());ratio=matched/area
    return bool(matched>=20 and ratio>.4),{'predicted_area':area,'matched_metal_pixels':matched,'ratio':ratio,'source':'episode_observed_planar_outline'}
