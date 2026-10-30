"""Bounded extra-probe proposals ranked by predicted image information.

Uses only the public camera, own poses and the nominal grasp offset. Predicted
views are proposals, never measurements or fit constraints. No visibility,
attachment, depth accuracy, collision or task-success claim is made here.
"""
import numpy as np
from .xylophone_geometry import rigid_transform

# Keep the old proposal available; the others stay within a 90 mm translation.
EXTRA_OFFSETS = ((-.03,0.,.03),(-.08,0.,.03),(.08,0.,.03),
                 (0.,-.08,.03),(0.,.08,.03),(-.06,-.06,.03),
                 (-.06,.06,.03),(.06,-.06,.03),(.06,.06,.03),(0.,0.,.08))


def predicted_floor(camera, poses, anchor):
    """Worst-axis .25-pixel sensitivity of HYPOTHETICAL rigid projections."""
    if len(poses)<3:
        return float('inf')
    xyz=np.array([e[:3,:3]@anchor+e[:3,3] for e in poses])
    depth=((xyz-camera.c2w[:3,3])@camera.c2w[:3,:3]@camera.cv_to_gl)[:,2]
    if (depth<=0).any():
        return float('inf')
    uv=camera.project(xyz)
    if not np.isfinite(uv).all():
        return float('inf')
    j=np.column_stack([((camera.project(np.array([
        e[:3,:3]@(anchor+np.eye(3)[i]*1e-5)+e[:3,3] for e in poses]))-uv)/1e-5).ravel()
        for i in range(3)])
    minimum=np.linalg.svd(j,compute_uv=False)[-1]
    return float(.25/minimum) if minimum>1e-8 else float('inf')


def information_probe_candidates(camera, current, anchor, samples):
    """Return finite proposals without mutating/adding any real evidence.

    Ranking takes the worst predicted sensitivity over nominal +/-10 mm local
    offsets. This is a bounded design sensitivity check, not measured covariance.
    Missing old images do not become observations. Future visibility is assumed
    only for ranking, with a 20-pixel frame margin; the actual fit stays unchanged.
    """
    current=rigid_transform(current)
    anchor=np.asarray(anchor,float)
    if anchor.shape!=(3,) or not np.isfinite(anchor).all():
        raise ValueError('Finite nominal anchor required')
    frames=[rigid_transform(e) for e,_ in samples]
    visible=[e for e,(_,pixel) in zip(frames,samples) if pixel is not None]
    old_target=current.copy();old_target[:3,3]+=EXTRA_OFFSETS[0]
    if len(visible)<3:
        return [dict(target=old_target,delta_m=list(EXTRA_OFFSETS[0]),
                     ranking='original_insufficient_visible_evidence',
                     future_observations_assumed=False)]
    variants=[anchor]+[anchor+sign*.01*np.eye(3)[i] for i in range(3) for sign in (-1,1)]
    choices=[]
    for index,delta in enumerate(EXTRA_OFFSETS):
        delta=np.asarray(delta);length=np.linalg.norm(delta)
        target=current.copy();target[:3,3]+=delta
        predicted=[]
        # Approximate the existing >=15 mm moving-view and >=3 mm endpoint spacing.
        distances=list(np.arange(.015,length,.015))+[length]
        for distance in distances:
            e=current.copy();e[:3,3]+=delta*(distance/length)
            previous=frames+predicted
            threshold=.003 if distance==length else .015
            if previous and min(np.linalg.norm(e[:3,3]-p[:3,3]) for p in previous)<threshold-1e-9:
                continue
            if len(samples)+len(predicted)>=32:
                break
            predicted.append(e)
        if not predicted:
            continue
        in_frame=True
        for a in variants:
            xyz=np.array([e[:3,:3]@a+e[:3,3] for e in predicted])
            depth=((xyz-camera.c2w[:3,3])@camera.c2w[:3,:3]@camera.cv_to_gl)[:,2]
            if (depth<=0).any():
                in_frame=False;break
            uv=camera.project(xyz);w,h=camera.resolution
            if (not np.isfinite(uv).all() or (uv[:,0]<20).any() or (uv[:,0]>w-20).any()
                    or (uv[:,1]<20).any() or (uv[:,1]>h-20).any()):
                in_frame=False;break
        if not in_frame:
            continue
        floor=max(predicted_floor(camera,visible+predicted,a) for a in variants)
        if not np.isfinite(floor):
            continue
        choices.append(dict(target=target,delta_m=delta.tolist(),ranking='predicted_worst_axis_sensitivity',
                            predicted_floor_m=floor,predicted_new_views=len(predicted),
                            future_observations_assumed=True,nominal_offset_test_m=.01,
                            original_index=index,translation_m=float(length)))
    choices.sort(key=lambda c:(c['predicted_floor_m'],c['translation_m'],c['original_index']))
    # Original fallback is kept if projections offer no usable ranking; IK and
    # budget are still mandatory in the caller. No repeated physical probing.
    return choices or [dict(target=old_target,delta_m=list(EXTRA_OFFSETS[0]),
                            ranking='original_no_rankable_proposal',
                            future_observations_assumed=False)]
