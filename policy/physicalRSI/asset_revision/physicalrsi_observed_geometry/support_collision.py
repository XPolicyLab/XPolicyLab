"""Collision patches derived from a visible calibrated support plane.

The slab depth is a numerical collision representation, not measured table
thickness. Unobserved/occluded image regions do not create obstacle patches.
"""
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from . import GeometryUnavailable


def plane_clearance(vertices, plane):
    """Signed distance to an observed plane z=a*x+b*y+c.

    This preserves the controller's infinite support-plane collision model;
    it does not claim observed table extent or a reconstructed underside.
    """
    points=np.asarray(vertices,dtype=float)
    coefficients=np.asarray(plane,dtype=float)
    if (points.ndim!=2 or points.shape[1]!=3 or len(points)==0
            or coefficients.shape!=(3,) or not np.isfinite(points).all()
            or not np.isfinite(coefficients).all()):
        raise GeometryUnavailable('Finite robot vertices and observed support plane required')
    normal=np.r_[-coefficients[:2],1.]
    return float(np.min(points@normal-coefficients[2])/np.linalg.norm(normal))


def support_patches(camera, mask, plane, *, cell_size=.05):
    # A narrow visible region can be smaller than the preferred collision grid.
    # Refine the numerical grid without extending the observed surface mask.
    for divisor in (1, 2, 4, 8):
        patches = _support_patches_at_resolution(camera, mask, plane, cell_size=cell_size/divisor)
        if patches:
            return patches
    raise GeometryUnavailable('No support patch is fully visible at collision resolution')


def _support_patches_at_resolution(camera, mask, plane, *, cell_size):
    mask=np.asarray(mask,bool);plane=np.asarray(plane,float)
    if mask.shape!=camera.image.shape[:2] or plane.shape!=(3,) or not np.isfinite(plane).all() or not np.isfinite(cell_size) or cell_size<=0:
        raise GeometryUnavailable('Observed support mask, plane and positive collision resolution required')
    y,x=np.nonzero(mask)
    if len(x)<20:raise GeometryUnavailable('Insufficient visible support for collision patches')
    normal=np.r_[-plane[:2],1.];length=np.linalg.norm(normal);normal/=length
    tangent=np.array([1.,0.,0.]);tangent-=normal*(tangent@normal);tangent/=np.linalg.norm(tangent)
    basis=np.column_stack([tangent,np.cross(normal,tangent),normal]);level=plane[2]/length
    # Subsampling only controls work; all retained patches are checked against the full mask.
    stride=max(1,len(x)//5000);pixels=np.c_[x[::stride],y[::stride]]
    rays=camera.unproject(pixels,np.ones(len(pixels)))-camera.transform[:3,3]
    denominator=rays@normal
    valid=np.abs(denominator)>1e-5
    distances=np.full(len(pixels),-1.);distances[valid]=(level-camera.transform[:3,3]@normal)/denominator[valid]
    valid &= distances>0
    points=camera.transform[:3,3]+rays[valid]*distances[valid,None]
    if len(points)<20:raise GeometryUnavailable('No valid current support rays')
    xy=points@basis[:,:2]
    low=np.floor(np.percentile(xy,1,axis=0)/cell_size).astype(int)
    high=np.ceil(np.percentile(xy,99,axis=0)/cell_size).astype(int)
    if np.prod(high-low)>20000:raise GeometryUnavailable('Observed support extent is ill-conditioned')
    offsets=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.],[.5,.5]])
    occupied=[]
    for iy in range(low[1],high[1]):
        row=[]
        for ix in range(low[0],high[0]):
            samples=(np.array([ix,iy])+offsets)*cell_size
            world=samples@basis[:,:2].T+normal*level
            uv,depth=camera.project(world);uv=np.rint(uv).astype(int)
            if np.any(depth<=0) or np.any(uv<0) or np.any(uv[:,0]>=camera.width) or np.any(uv[:,1]>=camera.height):continue
            if not np.all(mask[uv[:,1],uv[:,0]]):continue
            minimum=uv[:4].min(0);maximum=uv[:4].max(0)
            footprint=np.zeros((maximum[1]-minimum[1]+1,maximum[0]-minimum[0]+1),np.uint8)
            cv2.fillConvexPoly(footprint,uv[:4]-minimum,1)
            observed=mask[minimum[1]:maximum[1]+1,minimum[0]:maximum[0]+1]
            if np.all(observed[footprint.astype(bool)]):row.append(ix)
        if row:occupied.append((iy,row))
    quaternion=Rotation.from_matrix(basis).as_quat()[[3,0,1,2]].tolist();patches=[]
    for iy,row in occupied:
        boundaries=np.flatnonzero(np.diff(row)>1)+1
        for run in np.split(np.asarray(row),boundaries):
            center_xy=np.array([(run[0]+run[-1]+1)/2,iy+.5])*cell_size
            center=basis[:,:2]@center_xy+normal*(level-cell_size/2)
            patches.append({'dims':[float(len(run)*cell_size),float(cell_size),float(cell_size)],
                            'pose':center.tolist()+quaternion})
    return patches


def patches_in_robot_frame(patches, base_pose):
    base=np.asarray(base_pose,float)
    if base.shape!=(7,) or not np.isfinite(base).all():raise GeometryUnavailable('Observed robot base pose required')
    rotation=Rotation.from_quat(base[[4,5,6,3]])
    result={}
    for index,patch in enumerate(patches):
        world=np.asarray(patch['pose'],float)
        center=rotation.inv().apply(world[:3]-base[:3])
        quat=(rotation.inv()*Rotation.from_quat(world[[4,5,6,3]])).as_quat()[[3,0,1,2]]
        result['observed_support_'+str(index)]={'dims':list(patch['dims']),'pose':center.tolist()+quat.tolist()}
    return {'cuboid':result}
