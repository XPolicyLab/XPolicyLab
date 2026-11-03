"""Fit an upright circular frustum from current silhouettes and observed support.

The frustum is an explicit generic shape hypothesis. Hidden surfaces are inferred,
not measured, and no object dimensions or stored mesh enter the fit.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares
from . import GeometryUnavailable
from .multiview import _boundary_distance
from .primitives import fixed_boundary


def fit_supported_frustum(views, plane):
    plane=np.asarray(plane,float)
    if plane.shape!=(3,) or not np.isfinite(plane).all() or not views:
        raise GeometryUnavailable('Observed support and a complete silhouette required')
    normal=np.r_[-plane[:2],1.];norm=np.linalg.norm(normal);normal/=norm;level=plane[2]/norm
    axis=np.array([1.,0.,0.]);axis-=normal*(axis@normal);axis/=np.linalg.norm(axis)
    basis=np.column_stack([axis,np.cross(normal,axis),normal]);prepared=[]
    for camera,mask in views:
        mask=np.asarray(mask,bool)
        if mask.shape!=camera.image.shape[:2] or mask.sum()<20:raise GeometryUnavailable('Unresolved frustum mask')
        y,x=np.nonzero(mask)
        if x.min()==0 or y.min()==0 or x.max()==camera.width-1 or y.max()==camera.height-1:
            raise GeometryUnavailable('Frustum fit requires unclipped silhouettes')
        hull=cv2.convexHull(np.c_[x,y].astype('float32')).reshape(-1,2).astype(float)
        prepared.append((camera,hull,fixed_boundary(hull)))
    camera,hull,_=prepared[0];origin=camera.transform[:3,3]
    rays=camera.unproject(hull,np.ones(len(hull)))-origin;den=rays@normal
    if np.any(np.abs(den)<1e-6):raise GeometryUnavailable('Support rays are parallel')
    distance=(level-origin@normal)/den
    if np.any(distance<=0):raise GeometryUnavailable('Support lies behind camera')
    footprint=(origin+rays*distance[:,None])@basis[:,:2]
    (cx,cy),sizes,_=cv2.minAreaRect(footprint.astype('float32'));center=np.array([cx,cy]);scale=float(np.mean(sizes))
    if scale<=0:raise GeometryUnavailable('Degenerate frustum footprint')
    angles=np.linspace(0,2*np.pi,96,endpoint=False);circle=np.c_[np.cos(angles),np.sin(angles)]
    def decode(values):
        height,bottom,top=np.exp(values[2:]);base=basis[:,:2]@values[:2]+level*normal
        points=np.r_[base+(circle*bottom)@basis[:,:2].T,
                     base+height*normal+(circle*top)@basis[:,:2].T]
        return base,height,bottom,top,points
    def residual(values):
        points=decode(values)[4];errors=[]
        for camera,hull,edge in prepared:
            projected=cv2.convexHull(camera.project(points)[0].astype('float32')).reshape(-1,2).astype(float)
            errors.extend(_boundary_distance(edge,projected));errors.extend(_boundary_distance(fixed_boundary(projected),hull))
        return np.asarray(errors)
    lower=np.r_[center-2*scale,np.log([scale*.02,scale*.03,scale*.03])]
    upper=np.r_[center+2*scale,np.log([scale*3,scale*2,scale*2])]
    fits=[]
    for factor in (.3,.7,1.4):
        seed=np.r_[center,np.log([scale*factor,scale*.35,scale*.5])]
        try:fit=least_squares(residual,seed,bounds=(lower,upper),loss='soft_l1',f_scale=.75,diff_step=1e-4,max_nfev=350)
        except (GeometryUnavailable,ValueError,FloatingPointError):continue
        rms=float(np.sqrt(np.mean(fit.fun**2)))
        if fit.success and np.isfinite(rms):fits.append((rms,fit.x))
    if not fits:raise GeometryUnavailable('No supported frustum converged')
    rms,values=min(fits,key=lambda item:item[0]);base,height,bottom,top,points=decode(values)
    if rms>1.5:raise GeometryUnavailable('Silhouettes contradict the circular frustum hypothesis')
    comparable=[decode(v)[1:4] for error,v in fits if error<=rms+.2]
    relative_spread=np.ptp(comparable,axis=0)/np.array([height,bottom,top])
    if np.any(relative_spread>.12):raise GeometryUnavailable('Frustum dimensions are ambiguous')
    # Ephemeral surface hypothesis for collision and silhouette checks. The
    # opening stays open; only the closed bottom and tapered side are sampled.
    points=np.concatenate([base+fraction*height*normal+(circle*(bottom+(top-bottom)*fraction))@basis[:,:2].T
                           for fraction in np.linspace(0,1,32)]
                          +[base+(circle*bottom*fraction)@basis[:,:2].T for fraction in np.linspace(0,1,12)])
    return {'base':base,'rotation':basis,'height':height,'bottom_radius':bottom,'top_radius':top,
            'points':points,'rms_px':rms,'relative_spread':relative_spread,
            'source':'current_rgb_outline_and_observed_support','shape_hypothesis':'upright_circular_frustum',
            'hidden_surfaces_reconstructed':True}
