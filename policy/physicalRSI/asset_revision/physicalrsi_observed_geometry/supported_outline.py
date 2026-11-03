"""Measure a generic upright extruded outline from calibrated RGB and support."""
import cv2
import numpy as np
from scipy.optimize import least_squares
from . import GeometryUnavailable
from .multiview import _boundary_distance
from .primitives import fixed_boundary


def fit_supported_outline(views, plane):
    """All dimensions and 16 outline radii are fitted, with no object template.

    Requires complete silhouettes in at least two separated cameras. This model
    describes an upright convex extrusion, not an arbitrary hidden 3D surface.
    Returns the current observed hull and fit disagreement, never a cached size.
    """
    plane=np.asarray(plane,float)
    if plane.shape!=(3,) or not np.isfinite(plane).all() or len(views)<2:
        raise GeometryUnavailable('Observed support and multiple silhouettes required')
    if max(np.linalg.norm(a.transform[:3,3]-b.transform[:3,3]) for a,_ in views for b,_ in views)<1e-3:
        raise GeometryUnavailable('Silhouette cameras have no baseline')
    normal=np.r_[-plane[:2],1.];norm=np.linalg.norm(normal);normal/=norm;level=plane[2]/norm
    axis=np.array([1.,0.,0.]);axis-=normal*(axis@normal);axis/=np.linalg.norm(axis)
    basis=np.column_stack([axis,np.cross(normal,axis),normal]);prepared=[]
    for camera,mask in views:
        mask=np.asarray(mask,bool)
        if mask.shape!=camera.image.shape[:2] or mask.sum()<20:raise GeometryUnavailable('Unresolved object mask')
        y,x=np.nonzero(mask)
        if x.min()==0 or y.min()==0 or x.max()==camera.width-1 or y.max()==camera.height-1:
            raise GeometryUnavailable('Supported outline requires unclipped silhouettes')
        hull=cv2.convexHull(np.c_[x,y].astype('float32')).reshape(-1,2).astype(float)
        prepared.append((camera,hull,fixed_boundary(hull)))
    camera,hull,_=prepared[0];origin=camera.transform[:3,3]
    rays=camera.unproject(hull,np.ones(len(hull)))-origin;denominator=rays@normal
    if np.any(np.abs(denominator)<1e-6):raise GeometryUnavailable('Silhouette rays parallel to support')
    distances=(level-origin@normal)/denominator
    if np.any(distances<=0):raise GeometryUnavailable('Observed support behind camera')
    footprint=(origin+rays*distances[:,None])@basis[:,:2]
    (cx,cy),sizes,_=cv2.minAreaRect(footprint.astype('float32'));seed_center=np.array([cx,cy]);scale=float(np.mean(sizes))
    if scale<=0:raise GeometryUnavailable('Degenerate observed footprint')
    angles=np.linspace(0,2*np.pi,16,endpoint=False);directions=np.c_[np.cos(angles),np.sin(angles)]
    def decode(values):
        height=np.exp(values[2]);xy=directions*np.exp(values[3:])[:,None]
        local=np.r_[np.c_[xy,np.full(16,-height/2)],np.c_[xy,np.full(16,height/2)]]
        center=basis[:,:2]@values[:2]+normal*(level+height/2)
        return center,height,xy,local@basis.T+center
    def residual(values):
        points=decode(values)[3];errors=[]
        for camera,hull,edge in prepared:
            pixels,depth=camera.project(points)
            if np.any(depth<=0):return np.full(len(prepared)*240,1e3)
            projected=cv2.convexHull(pixels.astype('float32')).reshape(-1,2).astype(float)
            errors.extend(_boundary_distance(edge,projected))
            errors.extend(_boundary_distance(fixed_boundary(projected),hull))
        return np.asarray(errors)
    low=np.r_[seed_center-2*scale,np.log(scale*.02),np.full(16,np.log(scale*.05))]
    high=np.r_[seed_center+2*scale,np.log(scale*2),np.full(16,np.log(scale*2))]
    fits=[]
    for fraction in (.2,.5,.8):
        seed=np.r_[seed_center,np.log(scale*fraction),np.full(16,np.log(scale/2))]
        try:fit=least_squares(residual,seed,bounds=(low,high),loss='soft_l1',f_scale=1.,diff_step=1e-4,max_nfev=500)
        except (GeometryUnavailable,ValueError,FloatingPointError):continue
        rms=float(np.sqrt(np.mean(fit.fun**2)))
        if fit.success and np.isfinite(rms):fits.append((rms,fit.x))
    if not fits or min(rms for rms,_ in fits)>2.:raise GeometryUnavailable('Observed silhouettes do not support an extruded outline')
    rms,values=min(fits,key=lambda item:item[0]);center,height,xy,points=decode(values)
    comparable=[decode(v)[1] for error,v in fits if error<=rms+.25]
    spread=float(np.ptp(comparable)) if comparable else 0.
    if spread>.15*height:raise GeometryUnavailable('Supported outline height is ambiguous across fits')
    return {'center':center,'rotation':basis,'height':height,'outline_local':xy,
            'dimensions':np.r_[np.ptp(xy,axis=0),height],'points':points,
            'rms_px':rms,'height_spread_m':spread,'source':'current_rgb_silhouettes_and_observed_support'}
