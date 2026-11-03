"""Supported surface height from calibrated current RGB, with pixel uncertainty."""
import cv2
import numpy as np
from . import GeometryUnavailable
from .multiview import projection,triangulate
from .rgb_support import correspondences


def measured_height(cameras,head_mask,plane,*,matcher=correspondences):
    if len(cameras)<2:raise GeometryUnavailable('Multiple calibrated views required for object height')
    head=cameras[0];mask=np.asarray(head_mask,bool);plane=np.asarray(plane,float)
    if mask.shape!=head.image.shape[:2] or plane.shape!=(3,) or not np.isfinite(plane).all():
        raise GeometryUnavailable('Current object mask and observed support plane required')
    normal=np.r_[-plane[:2],1.];length=np.linalg.norm(normal);normal/=length
    rows={}
    for camera in cameras[1:]:
        uu,vv,confidence=matcher(head,camera)
        pa,pb=projection(head),projection(camera)
        for uv,xy,score in zip(uu,vv,confidence):
            u,v=np.rint(uv).astype(int)
            if not (score>.2 and 0<=u<head.width and 0<=v<head.height and mask[v,u]):continue
            try:point,error=triangulate(head,camera,uv,xy,max_error=1.)
            except GeometryUnavailable:continue
            height=float(point@normal-plane[2]/length)
            perturbations=[]
            for shift in ([.5,0],[-.5,0],[0,.5],[0,-.5]):
                q=cv2.triangulatePoints(pa,pb,np.asarray(uv,float)[:,None],(np.asarray(xy,float)+shift)[:,None])[:,0]
                if abs(q[3])>1e-10:perturbations.append(abs(float((q[:3]/q[3]-point)@normal)))
            uncertainty=max(perturbations,default=np.inf)
            if height<=3*uncertainty:continue
            key=(u,v)
            if key not in rows or uncertainty<rows[key][1]:rows[key]=(height,uncertainty)
    if len(rows)<4:raise GeometryUnavailable('Object height is not resolved above RGB triangulation uncertainty')
    values=np.array(list(rows.values()));h=values[:,0];sigma=values[:,1]
    clusters=[np.abs(h-center)<=np.maximum(2*sigma,1e-5) for center in h]
    selected=max(clusters,key=lambda keep:int(keep.sum()))
    if selected.sum()<4:raise GeometryUnavailable('No consistent observed object top surface')
    height=float(np.median(h[selected]));uncertainty=float(max(np.median(sigma[selected]),np.median(abs(h[selected]-height))))
    if height<=3*uncertainty:raise GeometryUnavailable('Object thickness remains below measurement uncertainty')
    return height,{'source':'current_calibrated_rgb_surface_above_observed_support','points':int(selected.sum()),
                   'height_m':height,'uncertainty_m':uncertainty,'uses_asset_dimensions':False}
