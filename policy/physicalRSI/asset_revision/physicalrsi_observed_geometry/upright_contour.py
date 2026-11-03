"""Existing upright-contour heuristic with current camera and observed support.

The lower silhouette midpoint is assumed to lie on the support; the upper
midpoint is assumed to lie above it. This is an inference, not mesh measurement.
"""
import cv2
import numpy as np
from . import GeometryUnavailable


def refine_mask(mask,proposal,*,camera,plane):
    plane=np.asarray(plane,float)
    if camera is None or plane.shape!=(3,) or not np.isfinite(plane).all():
        raise GeometryUnavailable('Current camera and observed support required')
    mask=np.asarray(mask,np.uint8)
    if mask.shape!=camera.image.shape[:2]:raise GeometryUnavailable('Contour image size mismatch')
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not contours:return None
    contour=max(contours,key=cv2.contourArea)
    if cv2.contourArea(contour)<25:return None
    points=contour[:,0,:];top_v,bottom_v=points[:,1].min(),points[:,1].max()
    top_u=float(np.median(points[points[:,1]<=top_v+2,0]))
    bottom_u=float(np.median(points[points[:,1]>=bottom_v-2,0]))
    if points[:,0].min()==0 or points[:,0].max()==camera.width-1 or top_v==0 or bottom_v==camera.height-1:
        return None  # Reject this incomplete proposal; other instances can remain valid.
    origin=camera.transform[:3,3]
    rays=camera.unproject([[bottom_u,bottom_v],[top_u,top_v]],[1.,1.])-origin
    normal=np.r_[-plane[:2],1.];den=float(rays[0]@normal)
    if abs(den)<1e-8:raise GeometryUnavailable('Bottom contour ray parallel to support')
    scale=(plane[2]-origin@normal)/den
    if scale<=0:raise GeometryUnavailable('Observed support behind camera')
    bottom=origin+scale*rays[0]
    horizontal=rays[1,:2];den=float(horizontal@horizontal)
    if den<1e-8:raise GeometryUnavailable('Overhead contour cannot resolve upright height')
    distance=float((bottom[:2]-origin[:2])@horizontal/den)
    top=origin+distance*rays[1];height=float(top[2]-bottom[2])
    reprojection=camera.project([[bottom[0],bottom[1],top[2]]])[0][0]
    error=float(np.linalg.norm(reprojection-[top_u,top_v]))
    valid=bool(distance>0 and np.isfinite(height) and height>0 and error<=3.)
    return {**proposal,'contour':points.tolist(),'top_pixel':[top_u,int(top_v)],
            'bottom_pixel':[bottom_u,int(bottom_v)],'xy':bottom[:2].tolist(),
            'height_m':height,'height_source':'current_calibrated_RGB_upright_contour',
            'geometric_valid':valid,'height_evidence':{'axis_reprojection_error_px':error,
                'bottom_on_support_assumed':True,'upper_midpoint_above_bottom_assumed':True}}


def pixel_scale(camera,plane,xy):
    plane=np.asarray(plane,float);xy=np.asarray(xy,float)
    z=float(plane[:2]@xy+plane[2])
    depth=float(camera.project([[*xy,z]])[1][0])
    return depth/float(np.sqrt(camera.k[0,0]*camera.k[1,1]))


def support_height(plane,xy):
    plane=np.asarray(plane,float);xy=np.asarray(xy,float)
    if plane.shape!=(3,) or xy.shape!=(2,) or not np.isfinite(plane).all() or not np.isfinite(xy).all():
        raise GeometryUnavailable('Observed support and finite XY required')
    return float(plane[:2]@xy+plane[2])
