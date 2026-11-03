"""Episode measurements of visible object surfaces, not complete asset models."""
import numpy as np
from . import GeometryUnavailable


def measure_visible_shape(camera, depth, mask, plane, *, maximum_points=4096):
    depth=np.asarray(depth,float);mask=np.asarray(mask,bool);plane=np.asarray(plane,float)
    if (depth.shape!=(camera.height,camera.width) or mask.shape!=depth.shape
            or plane.shape!=(3,) or not np.isfinite(plane).all()
            or not isinstance(maximum_points,int) or maximum_points<20):
        raise GeometryUnavailable('Calibrated depth, instance mask and observed plane required')
    y,x=np.nonzero(mask)
    valid=np.isfinite(depth[y,x])&(depth[y,x]>0);x,y=x[valid],y[valid]
    if len(x)<20:raise GeometryUnavailable('Too few visible object pixels')
    if len(x)>maximum_points:
        keep=np.linspace(0,len(x)-1,maximum_points,dtype=int);x,y=x[keep],y[keep]
    points=camera.unproject(np.c_[x,y],depth[y,x])
    normal=np.r_[-plane[:2],1.];scale=np.linalg.norm(normal);normal/=scale;offset=-plane[2]/scale
    heights=points@normal+offset
    if np.quantile(heights,.98)<=0:raise GeometryUnavailable('Object is not resolved above observed support')
    center=np.median(points,axis=0);base=center-normal*(center@normal+offset)
    local=points-base;radial=local-heights[:,None]*normal
    return {'base':base,'points':local,'height':float(np.quantile(heights,.98)),
            'visible_radius':float(np.quantile(np.linalg.norm(radial,axis=1),.95)),
            'support_normal':normal,'support_plane':plane.copy(),
            'visible_extent':np.ptp(points,axis=0),
            'mask_touches_border':bool(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any()),
            'source':'current_rgb_depth_aligned_to_observed_support',
            'visible_only':True,'hidden_surfaces_reconstructed':False}
