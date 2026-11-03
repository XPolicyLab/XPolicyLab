"""Fit unknown supported primitive dimensions to a calibrated RGB silhouette."""
import itertools
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from . import GeometryUnavailable
from .multiview import _boundary, _boundary_distance

def fixed_boundary(poly, count=120):
 delta=np.roll(poly,-1,axis=0)-poly
 lengths=np.linalg.norm(delta,axis=1)
 cumulative=np.r_[0,np.cumsum(lengths)]
 distance=np.linspace(0,cumulative[-1],count,endpoint=False)
 indices=np.minimum(np.searchsorted(cumulative,distance,side="right")-1,len(poly)-1)
 return poly[indices]+delta[indices]*((distance-cumulative[indices])/np.maximum(lengths[indices],1e-12))[:,None]

def fit_supported_primitive(camera, mask, plane, shape):
    if shape not in ('cube', 'straightedge'):
        raise GeometryUnavailable('Unsupported primitive family')
    mask=np.asarray(mask,bool)
    if mask.shape != camera.image.shape[:2] or mask.sum()<8:
        raise GeometryUnavailable('Insufficient observed silhouette')
    a,b,c=np.asarray(plane,float)
    normal=np.array([-a,-b,1.]);length=np.linalg.norm(normal);normal/=length;offset=-c/length
    x=np.array([1.,0,0]);x-=normal*(normal@x);x/=np.linalg.norm(x)
    basis=np.column_stack([x,np.cross(normal,x),normal])
    box=np.array(list(itertools.product([-.5,.5],repeat=3)))
    prism=np.array([[x,y,z] for x in [-.5,.5] for y,z in [(-.5,-.5),(.5,-.5),(0,.5)]])
    yy,xx=np.nonzero(mask);hull=cv2.convexHull(np.c_[xx,yy].astype('float32')).reshape(-1,2).astype(float);edge=_boundary(hull)
    rays=camera.unproject(hull,np.ones(len(hull)))-camera.transform[:3,3];depth=-(camera.transform[:3,3]@normal+offset)/(rays@normal);foot=camera.transform[:3,3]+rays*depth[:,None]
    plane_pixels=(foot@basis[:,:2]).astype('float32');rect=cv2.minAreaRect(plane_pixels);center=np.array(rect[0]);size=np.array(rect[1]);yaw=np.deg2rad(rect[2]);
    if size[0]<size[1]:size=size[::-1];yaw+=np.pi/2
    candidates=[]
    for kind,unit in ([('cube',box)] if shape=='cube' else [('box',box),('triangular_prism',prism)]):
     def decode(v):
      dims=np.repeat(v[2],3) if kind=='cube' else v[2:5]
      rot=basis@Rotation.from_euler('z',v[-1]).as_matrix();pos=basis[:,:2]@v[:2]+normal*(-offset+dims[2]/2)
      points=(unit*dims)@rot.T+pos
      poly=cv2.convexHull(camera.project(points)[0].astype('float32')).reshape(-1,2).astype(float)
      return pos,rot,dims,poly
     def residual(v):
      poly=decode(v)[3];return np.r_[_boundary_distance(edge,poly),_boundary_distance(fixed_boundary(poly),hull)]
     starts=[]
     if kind=='cube':
      side=float(np.mean(size));starts=[np.r_[center,side,angle] for angle in np.linspace(0,np.pi/2,5,endpoint=False)];lower=np.r_[center-side*2,side*.1,-np.pi];upper=np.r_[center+side*2,side*3,np.pi]
     else:
      starts=[np.r_[center,size,size[1]*factor,yaw] for factor in [.2,.5,1,2]];lower=np.r_[center-size[0],size*.1,size[1]*.01,yaw-np.pi/2];upper=np.r_[center+size[0],size*3,size[1]*4,yaw+np.pi/2]
     for seed in starts:
      try:
       fit=least_squares(residual,seed,bounds=(lower,upper),max_nfev=250,diff_step=1e-4,loss='soft_l1',f_scale=.5)
       pos,rot,dims,poly=decode(fit.x)
      except (GeometryUnavailable,ValueError,FloatingPointError):
       continue
      candidates.append({'shape':kind,'center':pos.tolist(),'dimensions':dims.tolist(),'rotation':rot.tolist(),'rms_px':float(np.sqrt(np.mean(fit.fun**2))),'success':bool(fit.success)})
    candidates=[item for item in candidates if item['success'] and np.isfinite(item['rms_px'])]
    if not candidates:
        raise GeometryUnavailable('Primitive fit did not converge')
    best=min(candidates,key=lambda item:item['rms_px'])
    if best['rms_px']>2:
        raise GeometryUnavailable('Observed silhouette does not support primitive fit')
    rotation=Rotation.from_matrix(best['rotation'])
    quat=rotation.as_quat()[[3,0,1,2]]
    return {'pose':best['center']+quat.tolist(),
            'bbox_local':(box*np.array(best['dimensions'])).tolist(),
            'dims':best['dimensions'], 'source':'current_calibrated_rgb_support_and_silhouette',
            'hidden_surfaces_reconstructed':True,'model_family':best['shape'],'rms_px':best['rms_px']}
