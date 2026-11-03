"""Unknown upright cuboids from calibrated silhouettes, without a support height.

Upright rectangular geometry is an explicit shape hypothesis. Dimensions,
position and yaw are fitted; no benchmark mesh or object size enters the fit.
The caller must supply complete silhouettes of the same object in each view.
"""
import itertools
import cv2
import numpy as np
from scipy.optimize import least_squares
from . import GeometryUnavailable
from .multiview import projection, _boundary_distance
from .primitives import fixed_boundary


def fit_upright_cuboid(views):
    if len(views) < 2:
        raise GeometryUnavailable('Two calibrated cuboid silhouettes required')
    prepared=[]
    for camera,mask in views:
        mask=np.asarray(mask,bool)
        if mask.shape!=camera.image.shape[:2] or mask.sum()<20:
            raise GeometryUnavailable('Insufficient cuboid silhouette')
        y,x=np.nonzero(mask)
        if x.min()==0 or y.min()==0 or x.max()==camera.width-1 or y.max()==camera.height-1:
            raise GeometryUnavailable('Complete unclipped cuboid silhouettes required')
        hull=cv2.convexHull(np.c_[x,y].astype('float32')).reshape(-1,2).astype(float)
        if cv2.contourArea(hull.astype('float32'))<20:
            raise GeometryUnavailable('Degenerate cuboid outline')
        prepared.append((camera,hull,fixed_boundary(hull)))
    # Silhouette centroids need not be projections of the same physical point.
    # Their triangulation is only an initialization; all contours validate it.
    seeds=[]
    for a,b in itertools.combinations(prepared,2):
        ua,ub=np.mean(a[1],axis=0),np.mean(b[1],axis=0)
        v=cv2.triangulatePoints(projection(a[0]),projection(b[0]),ua[:,None],ub[:,None])[:,0]
        if abs(v[3])<1e-10:continue
        center=v[:3]/v[3]
        rays=[center-c.transform[:3,3] for c in (a[0],b[0])]
        cosine=np.dot(*rays)/(np.linalg.norm(rays[0])*np.linalg.norm(rays[1]))
        angle=float(np.arccos(np.clip(cosine,-1,1)))
        if angle<np.deg2rad(3):continue
        try:
            for c,_,_ in prepared:c.project([center])
        except GeometryUnavailable:continue
        seeds.append((angle,center))
    if not seeds:
        raise GeometryUnavailable('Insufficient calibrated cuboid baseline')
    center=max(seeds,key=lambda x:x[0])[1]
    camera,hull,_=prepared[0]
    depth=camera.project([center])[1][0]
    scale=float(np.max(np.ptp(hull,axis=0))*depth/min(camera.k[0,0],camera.k[1,1]))
    if not np.isfinite(scale) or scale<=0:
        raise GeometryUnavailable('Invalid observed cuboid scale proposal')
    unit=np.array(list(itertools.product([-.5,.5],repeat=3)))
    def decode(values):
        c,s=np.cos(values[3]),np.sin(values[3])
        rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        dimensions=np.exp(values[4:])
        return values[:3],rotation,dimensions,(unit*dimensions)@rotation.T+values[:3]
    def residual(values):
        points=decode(values)[3];errors=[]
        for camera,hull,edge in prepared:
            projected=cv2.convexHull(camera.project(points)[0].astype('float32')).reshape(-1,2).astype(float)
            errors.extend(_boundary_distance(edge,projected))
            errors.extend(_boundary_distance(fixed_boundary(projected),hull))
        return np.asarray(errors)
    def jacobian(values):
        # Absolute central steps survive float32 hull projection near zero XY.
        steps=np.eye(len(values))*1e-5
        return np.stack([(residual(values+d)-residual(values-d))/(2e-5) for d in steps],axis=1)
    lower=np.r_[center-3*scale,-np.pi,np.log(np.full(3,scale*.03))]
    upper=np.r_[center+3*scale,np.pi,np.log(np.full(3,scale*4))]
    fits=[]
    for yaw in np.linspace(0,np.pi/2,4,endpoint=False):
        seed=np.r_[center,yaw,np.log(np.full(3,scale*.65))]
        try:
            fit=least_squares(residual,seed,jac=jacobian,bounds=(lower,upper),loss='soft_l1',f_scale=.75,max_nfev=200)
        except (GeometryUnavailable,ValueError,FloatingPointError):continue
        rms=float(np.sqrt(np.mean(fit.fun**2)))
        if fit.success and np.isfinite(rms):fits.append((rms,fit))
    if not fits:raise GeometryUnavailable('No upright cuboid fit converged')
    rms,fit=min(fits,key=lambda item:item[0])
    view_errors=np.sqrt(np.mean(fit.fun.reshape(len(prepared),-1)**2,axis=1))
    if np.max(view_errors)>1.5:
        raise GeometryUnavailable('Silhouettes contradict the upright cuboid hypothesis')
    position,rotation,dimensions,points=decode(fit.x)
    comparable=[decode(f.x) for error,f in fits if error<=rms+.2]
    # XY permutations of a box rotated by 90 degrees are equivalent.
    sizes=np.array([np.r_[np.sort(d[:2]),d[2]] for _,_,d,_ in comparable])
    centers=np.array([c for c,_,_,_ in comparable])
    if np.any(np.ptp(sizes,axis=0)/np.r_[np.sort(dimensions[:2]),dimensions[2]]>.12) or np.max(np.ptp(centers,axis=0))>.005:
        raise GeometryUnavailable('Cuboid geometry is ambiguous across initializations')
    return {'center':position,'rotation':rotation,'dimensions':dimensions,'points':points,
            'bottom_z':float(position[2]-dimensions[2]/2),'rms_px':rms,'view_rms_px':view_errors,
            'source':'current_calibrated_rgb_silhouettes','shape_hypothesis':'upright_cuboid',
            'hidden_surfaces_reconstructed':True}
