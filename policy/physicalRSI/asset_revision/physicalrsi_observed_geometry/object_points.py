"""Within-episode object surface samples from calibrated RGB correspondences.

Only visible, matched surfaces are reconstructed; no hidden mesh is supplied.
Insufficient silhouette coverage fails rather than loading an asset profile.
"""
import cv2
import numpy as np
from . import Camera,GeometryUnavailable
from .rgb_support import correspondences
from .multiview import triangulate
from .upright_contour import pixel_scale


def observed_object_points(observation,dolls,plane,*,matcher=correspondences):
    head=Camera.from_observation(observation)
    others=[Camera.from_observation(observation,n) for n in observation['vision'] if n!='cam_head']
    if not others:raise GeometryUnavailable('Multiple calibrated object views required')
    plane=np.asarray(plane,float)
    if plane.shape!=(3,) or not np.isfinite(plane).all():raise GeometryUnavailable('Observed support required')
    matches=[(camera,matcher(head,camera)) for camera in others]
    profiles=[];evidence=[]
    for doll in dolls:
        if not doll.get('geometric_valid'):raise GeometryUnavailable('Valid observed object geometry required')
        xy=np.asarray(doll['xy'],float);height=float(doll['height_m'])
        base=np.r_[xy,plane[:2]@xy+plane[2]]
        allowance=3*pixel_scale(head,plane,xy)
        mask=np.zeros(head.image.shape[:2],np.uint8)
        cv2.fillPoly(mask,[np.asarray(doll['contour'],np.int32)],1)
        rows={}
        for camera,(uvs,vvs,scores) in matches:
            for uv,vv,score in zip(uvs,vvs,scores):
                if score<.5 or not np.isfinite(uv).all() or not np.isfinite(vv).all():continue
                u,v=np.rint(uv).astype(int)
                if not (0<=u<head.width and 0<=v<head.height and mask[v,u]):continue
                try:point,error=triangulate(head,camera,uv,vv,max_error=1.)
                except GeometryUnavailable:continue
                above=float(point[2]-plane[:2]@point[:2]-plane[2])
                if above< -allowance or above>height+allowance:continue
                key=(u,v)
                if key not in rows or error<rows[key][1]:rows[key]=(point,error)
        if len(rows)<8:raise GeometryUnavailable('Insufficient current RGB object correspondences')
        points=np.array([value[0] for value in rows.values()])
        projected=head.project(points)[0];rendered=np.zeros_like(mask)
        cv2.fillConvexPoly(rendered,cv2.convexHull(np.rint(projected).astype('int32')),1)
        overlap=float(np.count_nonzero(rendered & mask)/max(1,np.count_nonzero(rendered | mask)))
        if overlap<.65:raise GeometryUnavailable('Observed object points do not cover the visible silhouette')
        profiles.append(points-base)
        evidence.append({'source':'within_episode_calibrated_RGB_correspondences',
                         'points':len(points),'head_silhouette_iou':overlap,
                         'hidden_surfaces_reconstructed':False,'uses_asset_geometry':False})
    return profiles,evidence


def initial_size_order(dolls):
    if len(dolls)!=5:return None,{'valid':False,'reason':'five_current_objects_required'}
    ordered=sorted(dolls,key=lambda d:d['size_metric'])
    if not all(d.get('geometric_valid') for d in ordered):
        return None,{'valid':False,'reason':'invalid_observed_geometry'}
    if not all(b['size_metric']-a['size_metric']>a['size_pixel_quantum']+b['size_pixel_quantum'] for a,b in zip(ordered,ordered[1:])):
        return None,{'valid':False,'reason':'current_RGB_size_order_ambiguous'}
    return [dict(d,rank=i,calibrated_height_m=d['height_m']) for i,d in enumerate(ordered)],{'valid':True,'source':'current_RGB_size_order'}
