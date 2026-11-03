"""Triangulate a visible planar landmark from current calibrated RGB matches."""
import cv2
import numpy as np
from . import Camera,GeometryUnavailable
from .rgb_support import correspondences
from .multiview import triangulate


def observe_planar_landmark(observation,source_mask,pixel,*,matcher=correspondences):
    head=Camera.from_observation(observation,'cam_head')
    mask=np.asarray(source_mask,bool);pixel=np.asarray(pixel,float)
    if mask.shape!=head.image.shape[:2] or pixel.shape!=(2,) or not np.isfinite(pixel).all():
        raise GeometryUnavailable('Current object mask and landmark pixel required')
    xy=np.rint(pixel).astype(int)
    if np.any(xy<0) or xy[0]>=head.width or xy[1]>=head.height:
        raise GeometryUnavailable('Landmark lies outside the current camera image')
    estimates=[]
    for name in observation['vision']:
        if name=='cam_head':continue
        camera=Camera.from_observation(observation,name)
        uu,vv,confidence=matcher(head,camera)
        uu=np.asarray(uu,float);vv=np.asarray(vv,float);confidence=np.asarray(confidence,float)
        if not len(uu):continue
        indices=np.rint(uu).astype(int)
        keep=(confidence>.2)&(indices[:,0]>=0)&(indices[:,0]<head.width)&(indices[:,1]>=0)&(indices[:,1]<head.height)
        ids=np.flatnonzero(keep);ids=ids[mask[indices[ids,1],indices[ids,0]]]
        # Require real epipolar consistency before fitting a planar mapping.
        valid=[]
        for i in ids:
            try:triangulate(head,camera,uu[i],vv[i],max_error=1.)
            except GeometryUnavailable:continue
            valid.append(i)
        if len(valid)<6:continue
        a,b=uu[valid],vv[valid]
        H,inliers=cv2.findHomography(a,b,cv2.RANSAC,1.)
        if H is None or inliers is None or int(inliers.sum())<6:continue
        inliers=inliers[:,0].astype(bool)
        hull=cv2.convexHull(a[inliers].astype('float32'))
        if cv2.pointPolygonTest(hull,tuple(pixel),False)<0:continue
        mapped=cv2.perspectiveTransform(pixel[None,None,:],H)[0,0]
        try:point,error=triangulate(head,camera,pixel,mapped,max_error=1.)
        except GeometryUnavailable:continue
        projected=cv2.perspectiveTransform(a[inliers,None,:],H)[:,0]
        residual=float(np.sqrt(np.mean(np.sum((projected-b[inliers])**2,axis=1))))
        if residual>1.:continue
        estimates.append((point,{'camera':name,'inliers':int(inliers.sum()),'mapping_rms_px':residual,'ray_error_px':error}))
    if not estimates:
        raise GeometryUnavailable('Landmark lacks a calibrated, surrounding planar correspondence set')
    points=np.array([p for p,_ in estimates])
    if len(points)>1 and np.max(np.linalg.norm(points-points.mean(0),axis=1))>.005:
        raise GeometryUnavailable('Independent landmark views disagree')
    target=points.mean(0)
    return target,{'source':'current_calibrated_rgb_planar_landmark','target':target.tolist(),
                   'pixel':pixel.tolist(),'views':[e for _,e in estimates],'uses_object_geometry':False,'landmark_on_matched_plane_assumed':True}
