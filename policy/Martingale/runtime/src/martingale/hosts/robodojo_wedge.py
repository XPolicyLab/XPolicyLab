# Copyright © 2026 Martingale. All Rights Reserved.
from __future__ import annotations
from martingale import self_play_constants as sp

import itertools
import cv2
import numpy as np
from scipy.optimize import least_squares
from . import robodojo_kinematics as kin

_ANGLES = np.linspace(0, 2*np.pi, 128, endpoint=False)
_DIRECTIONS = np.c_[np.cos(_ANGLES), np.sin(_ANGLES)]
FACES = np.array([[0,2,1],[3,4,5],[0,1,4],[0,4,3],
                  [1,2,5],[1,5,4],[2,0,3],[2,3,5]])


def vertices(parameters, base_z=kin.TABLE_Z):
    x,y,yaw,length,width,height,apex = parameters
    u=np.array([np.cos(yaw),np.sin(yaw),0.])
    v=np.array([-np.sin(yaw),np.cos(yaw),0.])
    profile=[(-width/2,0.),(width/2,0.),(apex*width,height)]
    return np.array([np.array([x,y,base_z])+a*u+b*v+[0,0,z]
                     for a in (-length/2,length/2) for b,z in profile])


def contour(mask, allow_clipped=False):
    mask=np.asarray(mask,dtype=np.uint8)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not contours:return None
    c=max(contours,key=cv2.contourArea)[:,0].astype(float)
    if len(c)<20:return None
    _,axes,_=cv2.minAreaRect(c.astype(np.float32))
    if min(axes)<2 or max(axes)/min(axes)<3:return None
    if not allow_clipped and ((c[:,0]<=1).any() or (c[:,0]>=mask.shape[1]-2).any() or (c[:,1]<=1).any() or (c[:,1]>=mask.shape[0]-2).any()):return None
    return c


def _seed(view, pixels, base_z):
    xy,valid=view.on_plane(pixels[:,1],pixels[:,0],base_z+.01)
    if not np.all(valid):return None
    center,axes,angle=cv2.minAreaRect(xy.astype(np.float32))
    if axes[0]<axes[1]:axes=axes[::-1];angle+=90
    return [*center,np.radians(angle),axes[0],axes[1]]


def _scanlines(pixels, axis, cuts):
    """Bounds of a convex silhouette across its length at fixed image cuts."""
    normal=np.array([-axis[1],axis[0]])
    indices=cv2.convexHull(pixels.astype(np.float32),returnPoints=False).ravel()
    hull=pixels[indices]
    uv=np.c_[hull@axis,hull@normal];a=uv;b=np.roll(uv,-1,axis=0)
    den=b[:,0]-a[:,0]
    ratio=np.divide(cuts[:,None]-a[:,0],den,out=np.full((len(cuts),len(a)),np.nan),where=np.abs(den)>1e-9)
    vals=a[:,1]+ratio*(b[:,1]-a[:,1]);inside=(ratio>=0)&(ratio<=1)
    low=np.min(np.where(inside,vals,np.inf),axis=1);high=np.max(np.where(inside,vals,-np.inf),axis=1)
    return np.c_[low,high]


def fit_wedge(observations, base_z=kin.TABLE_Z, ridge_lines=None):
    """Fit (name, calibrated View, binary mask) observations of one tool.

    Head outline constrains length; wrist central scanlines constrain the
    sloping sides without interpreting an occluded end as the end of the tool.

    Seven fitted parameters: XY, heading, length, width, height, apex offset.
    Roll/pitch are constrained by the assumed resting face, not measured for a
    freely held object. The longitudinal axis has a 180-degree symmetry.
    """
    usable=[];excluded=[]
    for name,view,mask in observations:
        c=contour(mask,allow_clipped=name!='cam_head')
        if c is None:excluded.append(name)
        else:usable.append((name,view,np.asarray(mask,bool),c))
    if len(usable)<2 or not any(n=='cam_head' for n,_,_,_ in usable):
        return dict(accepted=False,reasons=['Need a complete head outline and a supporting view of the tool sides.'],excluded_views=excluded)
    usable.sort(key=lambda t:t[0]!='cam_head')
    seed=_seed(usable[0][1],usable[0][3],base_z)
    if seed is None:return dict(accepted=False,reasons=['Cannot initialize a tool on the table.'])
    x,y,yaw,length,width=seed
    if not .04<length<1.2 or not .001<width<.2:
        return dict(accepted=False,reasons=['Mask is not a supported long tool.'])
    targets=[];slices=[]
    for name,view,mask,c in usable:
        if name=='cam_head':
            targets.append(np.max(c@_DIRECTIONS.T,axis=0)+.5);slices.append(None)
        else:
            _,_,vv=np.linalg.svd(c-c.mean(0),full_matrices=False);axis=vv[0];along=c@axis
            cuts=np.linspace(np.quantile(along,sp.WEDGE_SCAN_QUANTILES[0]),np.quantile(along,sp.WEDGE_SCAN_QUANTILES[1]),48)
            target=_scanlines(c,axis,cuts)+[-.5,.5]
            targets.append(target.ravel());slices.append((axis,cuts))
    ridge_views=[(name,view,ridge_lines[name]) for name,view,_,_ in usable
                 if ridge_lines and name in ridge_lines]
    if len(ridge_views)<2:ridge_views=[]
    def ridge_residual(p):
        xyz=vertices(p,base_z)
        points=xyz[2]+np.linspace(*sp.WEDGE_RIDGE_FRACTIONS,16)[:,None]*(xyz[5]-xyz[2])
        errors=[]
        for name,view,ridge in ridge_views:
            r,c,d=view.project(points);line=np.asarray(ridge['line'])
            errors.append(np.c_[c,r,np.ones(len(r))]@line)
        return np.concatenate(errors) if errors else np.empty(0)
    def outline_residual(p):
        xyz=vertices(p,base_z);parts=[]
        for (_,view,_,_),target,section in zip(usable,targets,slices):
            r,c,d=view.project(xyz)
            if np.any(d<=.01):return np.full(sum(map(len,targets)),1e4)
            pixels=np.c_[c,r]
            predicted=(np.max(pixels@_DIRECTIONS.T,axis=0) if section is None
                       else _scanlines(pixels,*section).ravel())
            parts.append(np.nan_to_num(predicted-target,nan=1e3,posinf=1e3,neginf=-1e3))
        return np.concatenate(parts)
    def residual(p):
        return np.r_[outline_residual(p),2.*ridge_residual(p)]
    lower=[x-.08,y-.08,yaw-.4,max(.03,length*.7),.002,.0005,-.5]
    upper=[x+.08,y+.08,yaw+.4,min(1.3,length*1.3),.20,.15,.5]
    fits=[]
    for height,apex in itertools.product((.008,.03,.07),(-.4,0.,.4)):
        p=[x,y,yaw,length,np.clip(width,.003,.15),height,apex]
        fit=least_squares(residual,p,bounds=(lower,upper),loss='soft_l1',f_scale=1.,x_scale='jac',max_nfev=180)
        fits.append(fit)
    best=min(fits,key=lambda f:np.mean(residual(f.x)**2))
    errors=outline_residual(best.x);parts=np.split(errors,np.cumsum([len(t) for t in targets])[:-1]);rms=np.array([np.sqrt(np.mean(e**2)) for e in parts])
    x,y,yaw,length,width,height,apex=best.x;xyz=vertices(best.x,base_z)
    center=np.array([x,y,base_z+height/2]);rays=[center-v.pos for _,v,_,_ in usable];rays=[r/np.linalg.norm(r) for r in rays]
    separation=max(np.degrees(np.arccos(np.clip(a@b,-1,1))) for a,b in itertools.combinations(rays,2))
    eps=1e-5; jac=np.column_stack([(residual(best.x+np.eye(7)[i]*eps)-residual(best.x-np.eye(7)[i]*eps))/(2*eps) for i in range(7)])
    _,sv,vh=np.linalg.svd(jac,full_matrices=False)
    sensitivity=np.sqrt(np.diag(vh.T@np.diag(1/np.maximum(sv,1e-8)**2)@vh))
    reasons=[]
    ridge_errors=ridge_residual(best.x)
    ridge_rms=float(np.sqrt(np.mean(ridge_errors**2))) if len(ridge_errors) else None
    if ridge_rms is not None and ridge_rms>1.5:reasons.append('Central lines do not agree with one 3D ridge.')
    if separation<10:reasons.append('Insufficient camera separation.')
    if max(rms)>2.:reasons.append('Silhouettes disagree with a triangular prism.')
    if sensitivity[4]>.008 or sensitivity[5]>.008:reasons.append('Cross-section is poorly constrained; obtain another oblique view.')
    if sensitivity[6]*width>.008:reasons.append('Triangle apex is poorly constrained by these views.')
    if height<.0006 or height>.149:reasons.append('Height is at a fitting bound.')
    if width>.199:reasons.append('Width is at a fitting bound.')
    direction=np.array([np.cos(yaw),np.sin(yaw),0.]);side=np.array([-np.sin(yaw),np.cos(yaw),0.])
    normals=[]
    for a,b in ((0,1),(1,2),(2,0)):
        edge=xyz[b]-xyz[a];normal=np.cross(direction,edge);normal/=np.linalg.norm(normal)
        midpoint=(xyz[a]+xyz[b])/2
        if normal@(midpoint-xyz[:3].mean(0))<0:normal=-normal
        normals.append(normal.tolist())
    return dict(accepted=not reasons,reasons=reasons,kind='triangular_prism',parameters=best.x.tolist(),
                ridge_used=bool(ridge_views),ridge_views=[n for n,_,_ in ridge_views],ridge_rms_px=ridge_rms,
                center_xyz=center.tolist(),base_frame_xyz=[float(x),float(y),float(base_z)],
                yaw_deg=float(np.degrees(yaw)%180),long_axis_xyz=direction.tolist(),
                orientation_matrix=np.column_stack([direction,side,[0,0,1]]).tolist(),
                length_m=float(length),width_m=float(width),height_m=float(height),
                apex_offset_fraction=float(apex),base_z=float(base_z),top_z=float(base_z+height),
                vertices_xyz=xyz.tolist(),faces=FACES.tolist(),
                longitudinal_edges_xyz=[[xyz[i].tolist(),xyz[i+3].tolist()] for i in range(3)],
                side_face_normals_xyz=normals,supporting_views=[n for n,_,_,_ in usable],excluded_views=excluded,
                silhouette_rms_px=float(np.sqrt(np.mean(errors**2))),view_errors_px={n:float(e) for (n,_,_,_),e in zip(usable,rms)},
                view_separation_deg=float(separation),width_sensitivity_m=float(sensitivity[4]),height_sensitivity_m=float(sensitivity[5]),
                note='Initial tool resting on the table only. Base face is constrained horizontal. Long-axis direction has 180 degree symmetry. Use accepted geometry only; dimensions are inferred from RGB.')


def measure(primitives, *, detail=False):
    """Fit the visible wedge; omit mesh arrays unless ``detail`` is set."""
    from . import robodojo_tools as tools
    obs=tools._obs(primitives);shown=tools._shown(obs);sam=getattr(primitives,'sam3_client',None)
    if sam is None:return dict(accepted=False,reasons=['No segmentation client.'],source_observation=dict(step=shown.get('step',obs.get('step')),cameras=[]))
    from .robodojo_ridge import detect_ridge
    observations=[];ridges={}
    for name,cam in obs.get('vision',{}).items():
        view=tools._view(name,shown);color=cam.get('color')
        if view is None or color is None:continue
        result=sam.segment(color,text_prompt='the long white ruler',min_score=.2)
        if result.found:
            mask=np.asarray(result.mask,bool);observations.append((name,view,mask))
            ridge=detect_ridge(color,mask)
            if ridge is not None:ridges[name]=ridge
    result=fit_wedge(observations,ridge_lines=ridges)
    result['source_observation']=dict(step=shown.get('step',obs.get('step')),cameras=[o[0] for o in observations])
    if not detail:
        result.pop('vertices_xyz',None);result.pop('faces',None)
    return result
