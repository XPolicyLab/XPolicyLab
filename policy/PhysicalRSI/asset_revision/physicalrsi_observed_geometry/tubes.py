"""Resting tube pose and dimensions from a visible outline and observed support.

Generic cylinder and rounded-bottom families are hypotheses, not asset meshes.
Metric scale comes from calibrated rays and the currently observed support plane.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from . import GeometryUnavailable
from .multiview import _boundary_distance
from .primitives import fixed_boundary


def fit_resting_tube(camera, mask, cap_pixel, plane):
    mask=np.asarray(mask,bool);plane=np.asarray(plane,float);cap_pixel=np.asarray(cap_pixel,float)
    if mask.shape!=camera.image.shape[:2] or mask.sum()<20 or plane.shape!=(3,) or cap_pixel.shape!=(2,) or not np.isfinite(np.r_[plane,cap_pixel]).all():
        raise GeometryUnavailable('Resolved tube silhouette, cap direction and observed support required')
    normal=np.r_[-plane[:2],1.];norm=np.linalg.norm(normal);normal/=norm;level=plane[2]/norm
    tangent=np.array([1.,0.,0.]);tangent-=normal*(tangent@normal);tangent/=np.linalg.norm(tangent)
    basis=np.column_stack((tangent,np.cross(normal,tangent),normal))
    y,x=np.nonzero(mask);hull=cv2.convexHull(np.c_[x,y].astype('float32')).reshape(-1,2).astype(float)
    if np.any(hull<=0) or np.any(hull[:,0]>=camera.width-1) or np.any(hull[:,1]>=camera.height-1):
        raise GeometryUnavailable('Resting tube initialization requires an unclipped outline')
    def on_plane(pixels):
        rays=camera.unproject(pixels,np.ones(len(pixels)))-camera.transform[:3,3]
        den=rays@normal
        if np.any(np.abs(den)<1e-6):raise GeometryUnavailable('Support rays are parallel')
        depth=(level-camera.transform[:3,3]@normal)/den
        if np.any(depth<=0):raise GeometryUnavailable('Support is behind camera')
        return camera.transform[:3,3]+rays*depth[:,None]
    footprint=on_plane(hull)@basis[:,:2]
    (cx,cy),(width,height),angle=cv2.minAreaRect(footprint.astype('float32'))
    if width<height:width,height=height,width;angle+=90
    if height<=0 or width<=height:raise GeometryUnavailable('Tube outline is not elongated')
    yaw=np.deg2rad(angle);center=np.array([cx,cy]);radius=height/2
    cap_plane=on_plane(cap_pixel[None])[0]@basis[:,:2]
    if (cap_plane-center)@np.array([np.cos(yaw),np.sin(yaw)])<0:yaw+=np.pi
    edge=fixed_boundary(hull)
    theta=np.linspace(0,2*np.pi,48,endpoint=False)
    radial=np.c_[np.cos(theta),np.sin(theta)]
    fits=[]
    for rounded in (False,True):
        def decode(values):
            r=np.exp(values[2]);body=np.exp(values[3]);length=body+(r if rounded else 0.)
            axis=basis[:,:2]@np.array([np.cos(values[4]),np.sin(values[4])])
            cross=np.cross(axis,normal);rotation=np.column_stack((cross,axis,normal))
            midpoint=basis[:,:2]@values[:2]+normal*(level+r)
            rings=[]
            for axial in (-(length/2)+(r if rounded else 0.),length/2):
                rings.append(midpoint+axis*axial+radial[:,0,None]*cross*r+radial[:,1,None]*normal*r)
            if rounded:
                for phi in np.linspace(-np.pi/2,0,9):
                    axial=-length/2+r+r*np.sin(phi);rr=r*np.cos(phi)
                    rings.append(midpoint+axis*axial+radial[:,0,None]*cross*rr+radial[:,1,None]*normal*rr)
            pixels=camera.project(np.concatenate(rings))[0]
            poly=cv2.convexHull(pixels.astype('float32')).reshape(-1,2).astype(float)
            return midpoint-axis*length/2,rotation,length,r,poly
        def residual(values):
            poly=decode(values)[-1]
            return np.r_[_boundary_distance(edge,poly),_boundary_distance(fixed_boundary(poly),hull)]
        lower=np.r_[center-width,np.log(radius*.1),np.log(width*.1),yaw-np.pi/2]
        upper=np.r_[center+width,np.log(radius*3),np.log(width*3),yaw+np.pi/2]
        for factor in (.5,1.,1.5):
            seed=np.r_[center,np.log(radius*factor),np.log(max(width-radius if rounded else width,width*.2)),yaw]
            try:
                fit=least_squares(residual,seed,bounds=(lower,upper),loss='soft_l1',f_scale=.75,max_nfev=250,diff_step=1e-4)
                root,rotation,length,r,_=decode(fit.x)
            except (GeometryUnavailable,ValueError,FloatingPointError):continue
            rms=float(np.sqrt(np.mean(fit.fun**2)))
            if fit.success and np.isfinite(rms) and length>2*r:
                fits.append((rms,root,rotation,length,r,rounded))
    if not fits:raise GeometryUnavailable('No supported tube fit converged')
    rms,root,rotation,length,r,rounded=min(fits,key=lambda item:item[0])
    if rms>2.:raise GeometryUnavailable('Tube outline contradicts supported cylinder geometry')
    return {'pose':root.tolist()+Rotation.from_matrix(rotation).as_quat()[[3,0,1,2]].tolist(),
            'length':float(length),'radius':float(r),'rms_px':rms,'family':'rounded_bottom' if rounded else 'cylinder',
            'source':'current_rgb_outline_and_observed_support'}
