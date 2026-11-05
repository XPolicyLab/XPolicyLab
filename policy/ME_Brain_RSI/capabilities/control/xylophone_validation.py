"""X16 independent-view binding checks, using RGB and actual EE poses only.

Validation observations are held out from the preliminary fit. They are never
invented from projections. All old visible observations join the final fit.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from .xylophone_anchor import fit_moving_anchor
from .xylophone_geometry import rigid_transform


def distinct_view(a, b, translation=.003, rotation=.035):
    a,b=rigid_transform(a),rigid_transform(b)
    return (np.linalg.norm(a[:3,3]-b[:3,3]) >= translation or
            Rotation.from_matrix(a[:3,:3].T@b[:3,:3]).magnitude() >= rotation)


def validate_anchor(camera, old_samples, new_samples, preliminary, nominal):
    """Independent prediction plus all-point refit; no favorable subset."""
    old=[(rigid_transform(e),np.asarray(p,float)) for e,p in old_samples if p is not None]
    new=[(rigid_transform(e),np.asarray(p,float)) for e,p in new_samples if p is not None]
    result=dict(state='unknown',reason='insufficient_independent_visible_views',
                old_visible=len(old),new_visible=len(new),absolute_accuracy_verified=False)
    independent=[]
    for e,p in new:
        if all(distinct_view(e,x) for x,_ in old+independent):
            independent.append((e,p))
    result['independent_visible']=len(independent)
    if len(independent)<2:return result
    preliminary=np.asarray(preliminary,float);nominal=np.asarray(nominal,float)
    if preliminary.shape!=(3,) or nominal.shape!=(3,) or not np.isfinite([preliminary,nominal]).all():
        raise ValueError('Finite local head anchors required')
    predicted=camera.project([e[:3,:3]@preliminary+e[:3,3] for e,_ in new])
    errors=np.linalg.norm(predicted-np.array([p for _,p in new]),axis=1)
    result['held_out_max_pixel_error']=float(np.max(errors))
    if not np.isfinite(errors).all() or errors.max()>2.:
        result.update(state='inconsistent',reason='new_view_prediction_conflict')
        return result
    combined=old+new
    fit=fit_moving_anchor(camera,[e for e,_ in combined],[p for _,p in combined],stable=True)
    result['all_point_fit']={k:v for k,v in fit.items() if k!='ee_anchor'}
    if fit['state']!='consistent':
        result.update(state=fit['state'],reason='all_point_'+fit['reason']);return result
    anchor=fit['ee_anchor']
    result['anchor_shift_m']=float(np.linalg.norm(anchor-preliminary))
    if np.linalg.norm(anchor-nominal)>.035:
        result.update(state='inconsistent',reason='outside_nominal_grasp_geometry');return result
    result.update(state='consistent',reason='independent_views_and_all_point_fit',ee_anchor=anchor)
    return result


def validation_frames(current, rotation, anchor, camera):
    """Two bounded rotations at a high head position, then a translated view.

    Keep the head at its preliminary location for rotations. A successful
    sampled IK check is required elsewhere; visibility here is only predicted.
    """
    current=rigid_transform(current);anchor=np.asarray(anchor,float)
    rotation=np.asarray(rotation,float)
    if anchor.shape!=(3,) or not np.isfinite(anchor).all():raise ValueError('Finite anchor')
    head=current[:3,:3]@anchor+current[:3,3]
    if head[2]<.84:return []
    # +/-8 degrees are finite design probes, not scene-specific coefficients.
    variants=[]
    for sign in (1.,-1.):
        frames=[]
        for angle,delta in ((0.,np.zeros(3)),(sign*8.,np.zeros(3)),(0.,np.array([0.,-.025,0.]))):
            r=Rotation.from_rotvec(np.array([0.,0.,np.deg2rad(angle)])).as_matrix()@rotation
            f=np.eye(4);f[:3,:3]=r;f[:3,3]=head+delta-r@anchor
            frames.append(f)
        xyz=np.array([e[:3,:3]@anchor+e[:3,3] for e in frames])
        uv=camera.project(xyz);w,h=camera.resolution
        depth=((xyz-camera.c2w[:3,3])@camera.c2w[:3,:3]@camera.cv_to_gl)[:,2]
        if ((depth>0).all() and np.isfinite(uv).all() and np.all((uv[:,0]>20)&(uv[:,0]<w-20)&
                                                         (uv[:,1]>20)&(uv[:,1]<h-20))):
            variants.append(frames)
    return variants
