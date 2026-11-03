"""Calibrated centerline reconstruction; no dimensions or depth map."""
import numpy as np
from . import GeometryUnavailable
from .multiview import projection


def centerline_cap(views, cap_camera, cap_pixel, body_pixel):
    if len(views)!=2:raise GeometryUnavailable('Exactly two centerline views required')
    cap_pixel=np.asarray(cap_pixel,float);body_pixel=np.asarray(body_pixel,float)
    if cap_pixel.shape!=(2,) or body_pixel.shape!=(2,) or not np.isfinite(np.r_[cap_pixel,body_pixel]).all():
        raise GeometryUnavailable('Finite cap and body pixels required')
    planes=[]
    for camera,center,direction in views:
        center=np.asarray(center,float);direction=np.asarray(direction,float)
        if center.shape!=(2,) or direction.shape!=(2,) or not np.isfinite(np.r_[center,direction]).all() or np.linalg.norm(direction)<1e-8:
            raise GeometryUnavailable('Resolved image centerline required')
        line=np.cross(np.r_[center,1],np.r_[center+direction,1]);plane=projection(camera).T@line
        plane/=np.linalg.norm(plane[:3]);planes.append(plane)
    if len(planes)<2:raise GeometryUnavailable('Two centerline views required')
    planes=np.asarray(planes);_,singular,vh=np.linalg.svd(planes[:,:3],full_matrices=True)
    if singular[1]<.05*singular[0]:raise GeometryUnavailable('Centerline planes have insufficient separation')
    axis=vh[-1];point=np.linalg.lstsq(planes[:,:3],-planes[:,3],rcond=None)[0]
    origin=cap_camera.transform[:3,3];ray=cap_camera.unproject([cap_pixel],[1])[0]-origin;ray/=np.linalg.norm(ray)
    if np.linalg.norm(np.cross(ray,axis))<.05:raise GeometryUnavailable('Cap ray is parallel to centerline')
    t,u=np.linalg.lstsq(np.column_stack([axis,-ray]),origin-point,rcond=None)[0]
    if u<=0:raise GeometryUnavailable('Cap is behind camera')
    cap=point+t*axis;pixel=cap_camera.project([cap])[0][0];error=float(np.linalg.norm(pixel-cap_pixel))
    if error>2.:raise GeometryUnavailable('Cap does not agree with centerline')
    # Analytic projection differential avoids choosing an object-specific probe length.
    optical=(cap-origin)@cap_camera.transform[:3,:3]*[1,-1,-1]
    delta=axis@cap_camera.transform[:3,:3]*[1,-1,-1]
    h=cap_camera.k@optical;dh=cap_camera.k@delta
    derivative=(dh[:2]*h[2]-h[:2]*dh[2])/(h[2]**2)
    toward=np.asarray(cap_pixel)-np.asarray(body_pixel)
    if np.linalg.norm(toward)<3.:raise GeometryUnavailable('Cap sign is unresolved')
    if derivative@toward<0:axis=-axis
    return {'cap':cap,'axis_to_cap':axis,'cap_error_px':error,'ray_gap_m':float(np.linalg.norm(cap-(origin+u*ray)))}
