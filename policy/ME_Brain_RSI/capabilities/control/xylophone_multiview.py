"""X29 held-head geometry from allowed RGB and observed link6 poses only.

The public 13mm sphere provides scale in the attached wrist camera. A clipped
image edge is NEVER a silhouette edge. Robust contour proposals must have broad
curved support and agree across color thresholds. All accepted temporal frames
and all available head-camera centers participate in confirmation; no best-frame
subset, nominal-anchor substitution, depth, episode camera matrices or rewards.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

RADIUS=.013
FOCAL=640*13./20.955
K=np.array([[FOCAL,0.,320.],[0.,FOCAL,240.],[0.,0.,1.]])
CV_TO_GL=np.diag([1.,-1.,-1.])


def wrist_mount():
    """Public X5A.urdf fixed joints + robot_config camera (intrinsic XYZ)."""
    def t(xyz,rpy,order='xyz',degrees=False):
        out=np.eye(4);out[:3,3]=xyz
        out[:3,:3]=Rotation.from_euler(order,rpy,degrees=degrees).as_matrix()
        return out
    return (t([.057,0,0],[0,0,-3.1416]) @
            t([-.0275,0,.05],[0,.3491,np.pi]) @
            t([0,0,.001],[0,-80,-90],order='XYZ',degrees=True))


MOUNT=wrist_mount()


def camera_point(anchor):
    return (np.asarray(anchor)-MOUNT[:3,3])@MOUNT[:3,:3]@CV_TO_GL


def ee_point(camera_xyz):
    return MOUNT[:3,:3]@CV_TO_GL@np.asarray(camera_xyz)+MOUNT[:3,3]


def project_wrist(anchor):
    p=camera_point(anchor)
    if p[2]<=RADIUS:return None
    uv=K@p
    return uv[:2]/uv[2]


def unit_rays(pixels):
    uv=np.asarray(pixels,float)
    rays=np.c_[uv,np.ones(len(uv))]@np.linalg.inv(K).T
    return rays/np.linalg.norm(rays,axis=1)[:,None]


def cone_residual(center,rays):
    distance=np.linalg.norm(center)
    if distance<=RADIUS:return np.full(len(rays),1e3)
    return (np.arccos(np.clip(rays@(center/distance),-1,1))-
            np.arcsin(RADIUS/distance))*FOCAL


def arc_span(center,rays):
    axis=center/np.linalg.norm(center)
    u=np.cross(axis,[0.,1.,0.]);u/=np.linalg.norm(u);v=np.cross(axis,u)
    angles=np.sort(np.arctan2(rays@v,rays@u))
    return float(2*np.pi-np.max(np.diff(np.r_[angles,angles[0]+2*np.pi])))


def fit_silhouette(pixels):
    """Deterministic cone proposals; return only broad, well-supported arcs."""
    uv=np.asarray(pixels,float)
    if uv.ndim!=2 or uv.shape[1]!=2 or len(uv)<28 or not np.isfinite(uv).all():return None
    # Bound compute and prevent thousands of adjacent pixels faking precision.
    uv=uv[np.linspace(0,len(uv)-1,min(len(uv),160),dtype=int)]
    rays=unit_rays(uv);rng=np.random.default_rng(29013);proposals=[]
    for _ in range(120):
        a,b,c=rays[rng.choice(len(rays),3,replace=False)]
        axis=np.cross(b-a,c-a);norm=np.linalg.norm(axis)
        if norm<1e-7:continue
        axis/=norm
        if axis[2]<0:axis=-axis
        cosine=float(a@axis)
        if not .80<cosine<.9998:continue
        center=axis*RADIUS/np.sqrt(1-cosine*cosine)
        errors=np.abs(cone_residual(center,rays));inside=errors<1.25
        count=int(inside.sum())
        if count<max(28,.65*len(rays)):continue
        span=arc_span(center,rays[inside])
        if span<2.0:continue
        proposals.append((count,-float(np.mean(errors[inside])),center,inside))
    if not proposals:return None
    _,_,center,inside=max(proposals,key=lambda p:p[:2])
    fit=least_squares(lambda c:cone_residual(c,rays[inside]),center,max_nfev=35)
    center=fit.x;errors=np.abs(cone_residual(center,rays))
    inside=errors<1.25
    if inside.sum()<max(28,.65*len(rays)) or arc_span(center,rays[inside])<2.0:return None
    used=rays[inside]
    fit=least_squares(lambda c:cone_residual(c,used),center,max_nfev=30)
    center=fit.x;distance=float(np.linalg.norm(center))
    rms=float(np.sqrt(np.mean(cone_residual(center,used)**2)))
    s=np.linalg.svd(fit.jac/np.sqrt(len(used)),compute_uv=False)
    sensitivity=float(.5/s[-1]) if s[-1]>1e-8 else float('inf')
    if not .035<distance<.30 or center[2]<=RADIUS or rms>.80 or sensitivity>.006:return None
    return dict(anchor=ee_point(center),boundary=uv[inside],rms_px=rms,
        sensitivity_m=sensitivity,arc_radians=arc_span(center,used),
        support_fraction=float(inside.mean()),camera_distance_m=distance)


def measure_wrist(rgb,nominal):
    """Known-radius head, with threshold consensus and explicit ambiguity."""
    if rgb is None or np.shape(rgb)!=(480,640,3):return None
    predicted=project_wrist(nominal)
    if predicted is None:return None
    hsv=cv2.cvtColor(np.asarray(rgb,np.uint8),cv2.COLOR_RGB2HSV);h,s,v=cv2.split(hsv)
    fits=[]
    for saturation in (110,140,170):
        mask=np.uint8(((h<=9)|(h>=155))&(s>=saturation)&(v>=40))*255
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        accepted=[]
        for contour in contours:
            x,y,w,height=cv2.boundingRect(contour)
            if not 20<=max(w,height)<=360 or cv2.contourArea(contour)<180:continue
            points=contour[:,0,:].astype(float)
            if np.min(np.linalg.norm(points-predicted,axis=1))>120:continue
            # Specular highlights can cut concave notches into the red mask.
            # Keep observed pixels on its outer envelope, never synthetic hull
            # chords across the highlight and never the image clipping line.
            hull=cv2.convexHull(contour)
            outer=np.array([abs(cv2.pointPolygonTest(hull,tuple(p),True))<=1.25 for p in points])
            points=points[outer]
            points=points[(points[:,0]>1)&(points[:,0]<638)&(points[:,1]>1)&(points[:,1]<478)]
            fit=fit_silhouette(points)
            if fit is None or np.linalg.norm(fit['anchor']-nominal)>.035:continue
            if np.linalg.norm(project_wrist(fit['anchor'])-predicted)>110:continue
            accepted.append(fit)
        if len(accepted)>1:return None
        if accepted:fits.append(accepted[0])
    # A low-saturation mask can merge a highlight with wood/background. Missing
    # geometry is not disagreement, but one threshold alone is insufficient.
    if len(fits)<2:return None
    anchors=np.array([f['anchor'] for f in fits]);mean=anchors.mean(0)
    spread=float(np.max(np.linalg.norm(anchors-mean,axis=1)))
    if spread>.002:return None
    # First fixed threshold supplies boundary; other thresholds validate it,
    # not an outcome- or downstream-residual-dependent best threshold choice.
    return dict(anchor=mean,boundary=fits[0]['boundary'],threshold_spread_m=spread,
                sensitivity_m=max(f['sensitivity_m'] for f in fits),
                rms_px=max(f['rms_px'] for f in fits),
                arc_radians=min(f['arc_radians'] for f in fits))


def confirm_multiview(head_camera,wrist_samples,head_samples,nominal):
    valid=[s for s in wrist_samples if s['measurement'] is not None]
    if len(valid)<5:return dict(state='unknown',reason='multiview_insufficient_wrist_observations',visible=len(valid))
    ees=[s['ee'] for s in valid]
    movement=max(np.linalg.norm(e[:3,3]-ees[0][:3,3]) for e in ees)
    if movement<.025:return dict(state='unknown',reason='multiview_insufficient_actual_motion')
    estimates=np.array([s['measurement']['anchor'] for s in valid]);mean=estimates.mean(0)
    spread=float(np.max(np.linalg.norm(estimates-mean,axis=1)))
    if spread>.003:return dict(state='inconsistent',reason='multiview_relative_head_moved',spread_m=spread)
    heads=[(ee,p) for ee,p in head_samples if p is not None]
    def residual(anchor):
        # Normalize each boundary block: adjacent pixels are correlated.
        parts=[cone_residual(camera_point(anchor),unit_rays(s['measurement']['boundary'])) /
               np.sqrt(len(s['measurement']['boundary'])) for s in valid]
        if heads:
            parts.extend((head_camera.project([ee[:3,:3]@anchor+ee[:3,3]])[0]-p) for ee,p in heads)
        return np.concatenate(parts)
    fit=least_squares(residual,mean,max_nfev=45)
    anchor=fit.x
    if np.linalg.norm(anchor-nominal)>.035:return dict(state='inconsistent',reason='multiview_outside_grasp_geometry')
    errors=[float(np.linalg.norm(head_camera.project([ee[:3,:3]@anchor+ee[:3,3]])[0]-p)) for ee,p in heads]
    wrist_errors=[float(np.sqrt(np.mean(cone_residual(camera_point(anchor),unit_rays(s['measurement']['boundary']))**2))) for s in valid]
    if errors and max(errors)>2.0:return dict(state='inconsistent',reason='multiview_head_disagreement',max_head_pixel_error=max(errors))
    if max(wrist_errors)>1.25:return dict(state='inconsistent',reason='multiview_wrist_disagreement',max_wrist_rms_px=max(wrist_errors))
    # No sqrt(frame count) precision gain for a rigidly mounted camera with
    # the same silhouette. Head rays may still improve the weak depth axis.
    singular=np.linalg.svd(fit.jac/np.sqrt(len(valid)+len(heads)),compute_uv=False)
    floor=float(.5/singular[-1]) if singular[-1]>1e-8 else float('inf')
    if floor>.003:return dict(state='unknown',reason='multiview_uncertainty',sensitivity_floor_m=floor)
    return dict(state='consistent',reason='multiview_known_sphere_rigid_binding',ee_anchor=anchor,
        wrist_views=len(valid),head_views=len(heads),actual_ee_motion_m=float(movement),
        spread_m=spread,sensitivity_floor_m=floor,max_head_pixel_error=max(errors,default=None),
        max_wrist_rms_px=max(wrist_errors),roll_observed=False)
