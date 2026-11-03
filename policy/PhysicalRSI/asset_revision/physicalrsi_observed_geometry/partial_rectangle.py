"""Unknown rectangular plane from calibrated, possibly image-clipped outlines."""
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from itertools import combinations
from . import GeometryUnavailable
from .multiview import _boundary_distance, triangulate


def _sample(poly, count=64):
    edges=np.roll(poly,-1,axis=0)-poly
    lengths=np.linalg.norm(edges,axis=1);distance=np.r_[0,np.cumsum(lengths)]
    if distance[-1]<=0:raise GeometryUnavailable('Degenerate rectangle outline')
    positions=np.linspace(0,distance[-1],count,endpoint=False)
    indices=np.minimum(np.searchsorted(distance,positions,side='right')-1,len(poly)-1)
    return poly[indices]+edges[indices]*((positions-distance[indices])/np.maximum(lengths[indices],1e-9))[:,None]


def fit_partial_rectangle(views, initial_center=None):
    """Fit free pose and dimensions from calibrated outlines.

    With no depth seed, triangulated outline centroids initialize the optimizer.
    Clipping can bias these centroids; only the final full-outline reprojection
    test accepts a reconstruction. The loose seed gate is not an acceptance gate.

    Only image-frame clipping is modelled. Robot/object occlusion must be rejected
    by the caller. At least two separated calibrated cameras are required.
    """
    if len(views)<2:raise GeometryUnavailable('Rectangle requires multiple views')
    if max(np.linalg.norm(a.transform[:3,3]-b.transform[:3,3]) for a,_ in views for b,_ in views)<1e-3:
        raise GeometryUnavailable('Rectangle cameras have no baseline')
    if initial_center is None:
        seeds=[]
        for (a,ha),(b,hb) in combinations(views,2):
            try:
                seed,_=triangulate(a,b,np.asarray(ha).mean(0),np.asarray(hb).mean(0),max_error=100.)
                seeds.append(seed)
            except GeometryUnavailable:
                continue
        fits=[]
        for seed in seeds:
            try:fits.append(fit_partial_rectangle(views,seed))
            except GeometryUnavailable:continue
        if not fits:raise GeometryUnavailable('No calibrated rectangle initialization converged')
        return min(fits,key=lambda item:item[3])
    center=np.asarray(initial_center,float)
    if center.shape!=(3,) or not np.isfinite(center).all():raise GeometryUnavailable('Observed rectangle seed required')
    camera,hull=views[0];depth=camera.project([center])[1][0]
    dimensions=np.sort(cv2.minAreaRect(np.asarray(hull,'float32'))[1])[::-1]*depth/camera.k[0,0]
    if depth<=0 or np.min(dimensions)<=0:raise GeometryUnavailable('Rectangle seed is behind camera or degenerate')
    unit=np.array([[-.5,-.5,0],[.5,-.5,0],[.5,.5,0],[-.5,.5,0]])
    prepared=[(c,np.asarray(h,float),_sample(np.asarray(h,float))) for c,h in views]
    def residual(values):
        with np.errstate(over='raise',invalid='raise'):
            points=(unit*np.r_[np.exp(values[6:8]),1])@Rotation.from_rotvec(values[3:6]).as_matrix().T+values[:3]
        result=[]
        for camera,hull,samples in prepared:
            try:poly,depths=camera.project(points)
            except GeometryUnavailable:return np.full(len(views)*128,1e3)
            if np.any(depths<=0):return np.full(len(views)*128,1e3)
            border=np.array([[0,0],[camera.width-1,0],[camera.width-1,camera.height-1],[0,camera.height-1]],dtype='float32')
            area,clipped=cv2.intersectConvexConvex(poly.astype('float32'),border)
            if clipped is None or area<=0:return np.full(len(views)*128,1e3)
            clipped=clipped.reshape(-1,2)
            result.extend(_boundary_distance(samples,clipped))
            result.extend(_boundary_distance(_sample(clipped),hull))
        return np.asarray(result)
    def jacobian(values):
        # Clipping uses float32 pixels. Relative perturbations near zero pose
        # coordinates disappear in that conversion, producing a zero Jacobian.
        # Use a central, absolute step floor for all free pose/size parameters.
        steps=1e-5*np.maximum(1.,np.abs(values))
        columns=[]
        for index,step in enumerate(steps):
            offset=np.zeros_like(values);offset[index]=step
            columns.append((residual(values+offset)-residual(values-offset))/(2*step))
        return np.column_stack(columns)
    fits=[]
    for yaw in (0.,np.pi/2):
        initial=np.r_[center,[0.,0.,yaw],np.log(dimensions)]
        try:
            fit=least_squares(residual,initial,loss='soft_l1',f_scale=.5,max_nfev=600,jac=jacobian)
        except (GeometryUnavailable,ValueError,FloatingPointError):continue
        rms=float(np.sqrt(np.mean(fit.fun**2)))
        if fit.success and np.isfinite(rms):fits.append((rms,fit.x))
    if not fits or min(f[0] for f in fits)>1.5:raise GeometryUnavailable('Calibrated rectangle outlines disagree')
    rms,values=min(fits,key=lambda item:item[0]);rotation=Rotation.from_rotvec(values[3:6]).as_matrix()
    if rotation[2,2]<0:rotation[:,1:]*=-1
    return values[:3],rotation,np.exp(values[6:8]),rms
