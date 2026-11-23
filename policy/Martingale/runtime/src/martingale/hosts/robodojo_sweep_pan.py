from martingale import self_play_constants as sp
# Copyright © 2026 Martingale. All Rights Reserved.
import numpy as np
from scipy.optimize import least_squares
from . import robodojo_kinematics as kin
from .robodojo_sweep_helpers import _support, DIRECTIONS
from .robodojo_sweep_shapes import mesh, iou, silhouette, THICKNESS
def fit(obs, yaw_hint=None):
    obs=[ob for ob in obs if ob.mask.sum()>=15]
    if len(obs)<2:
        return dict(kind='pan',accepted=False,reason='Fewer than two visible part masks')
    supports=[_support(ob) for ob in obs]
    r,c=np.nonzero(obs[0].mask)
    xy,ok=obs[0].view.on_plane(r,c,kin.TABLE_Z+sp.PAN_PROJECTION_PLANE_M)
    lo,hi=np.min(xy[ok],axis=0),np.max(xy[ok],axis=0)
    center=(lo+hi)/2
    size=np.clip(np.max(hi-lo),*sp.PAN_SIZE_INITIAL_BOUNDS_M)
    def expand(q):
        return np.r_[q[:2],kin.TABLE_Z+q[5]/2,q[2],0.,0.,q[3:6],q[6]]
    def residual(q):
        v,_=mesh('pan',expand(q));pieces=[]
        for ob,(target,clipped) in zip(obs,supports):
            r,c,d=ob.view.project(v)
            e=(np.c_[c,r]@DIRECTIONS.T).max(0)-target
            pieces.append(np.where(clipped,np.minimum(e,0),e))
        return np.concatenate(pieces)
    lower=[kin.TABLE_X[0],kin.TABLE_Y[0],-2*np.pi,.008,.008,.003,.35]
    upper=[kin.TABLE_X[1],kin.TABLE_Y[1],2*np.pi,.65,.65,.35,sp.PAN_CURVE_UPPER]
    candidates=[]
    for yaw in list(np.arange(8)*np.pi/4)+([] if yaw_hint is None else [yaw_hint]):
        start=np.clip(np.r_[center,yaw,size,size*.8,size*.2,.8],np.array(lower)+1e-8,np.array(upper)-1e-8)
        result=least_squares(residual,start,bounds=(lower,upper),x_scale='jac',max_nfev=150)
        p=expand(result.x);marks=[iou(ob,silhouette(ob,'pan',p)) for ob in obs]
        candidates.append((np.mean(marks),p,marks,float(np.sqrt(np.mean(residual(result.x)**2)))))
    _,p,marks,rmse=max(candidates,key=lambda item:item[0])
    v,_=mesh('pan',p)
    return dict(kind='pan',accepted=bool(min(marks)>=.62 and rmse<3.5),parameters=p.tolist(),
                per_view_iou=marks,support_rmse_px=rmse,
                entrance_edge_xyz=v[:2].tolist(),interior_floor_xyz=v[38:57].tolist(),
                assumption='Pan resting flat on the known table; generic 3mm floor shell',shell_thickness_m=THICKNESS)
