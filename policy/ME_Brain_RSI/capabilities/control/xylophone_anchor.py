"""RGB functional-point binding from camera rays and observed robot motion.

Independent geometry, no scene truth/depth/rewards. Unlike a full tool pose,
a spherical head center does not require inventing unobservable mallet roll.
Unconnected prototype: a consistent fit is evidence, never task/grasp success.
"""
import numpy as np
from scipy.optimize import least_squares
from .xylophone_geometry import rigid_transform


def fit_moving_anchor(camera, world_ees, pixels, min_motion=.015,
                      max_pixel_error=2., max_sigma_m=.02, stable=False):
    if len(world_ees) != len(pixels):
        raise ValueError("Observation lengths disagree")
    if not all(np.isfinite(x) and x > 0 for x in (min_motion,max_pixel_error,max_sigma_m)):
        raise ValueError("Positive finite evidence limits required")
    if len(pixels)<3 or any(p is None for p in pixels):
        return dict(state="unknown",reason="insufficient_visible_observations")
    ees=[rigid_transform(e) for e in world_ees]
    uv=np.asarray(pixels,float)
    if uv.shape != (len(ees),2) or not np.isfinite(uv).all():
        raise ValueError("Finite image coordinates required")
    if max(np.linalg.norm(e[:3,3]-ees[0][:3,3]) for e in ees)<min_motion:
        return dict(state="unknown",reason="insufficient_actual_motion")
    origin=camera.c2w[:3,3]
    matrices,targets=[],[]
    for e,p in zip(ees,uv):
        ray=camera.c2w[:3,:3]@camera.cv_to_gl@np.linalg.solve(camera.k,[*p,1.])
        ray/=np.linalg.norm(ray)
        perpendicular=np.eye(3)-np.outer(ray,ray)
        matrices.append(perpendicular@e[:3,:3])
        targets.append(perpendicular@(origin-e[:3,3]))
    a,b=np.vstack(matrices),np.concatenate(targets)
    singular=np.linalg.svd(a,compute_uv=False)
    if singular[-1] < 1e-5:
        return dict(state="unknown",reason="unobservable_depth")
    local=np.linalg.lstsq(a,b,rcond=None)[0]
    def projection(point):
        xyz=np.array([e[:3,:3]@point+e[:3,3] for e in ees])
        cv=(xyz-origin)@camera.c2w[:3,:3]@camera.cv_to_gl
        return camera.project(xyz),cv[:,2]
    if stable:
        # All visible points participate; no best-subset or reward selection.
        local=least_squares(lambda point:(projection(point)[0]-uv).ravel(),
                            local,max_nfev=40,xtol=1e-11,ftol=1e-11,gtol=1e-11).x
    predicted,depth=projection(local)
    if (depth<=0).any() or not np.isfinite(predicted).all():
        return dict(state="inconsistent",reason="invalid_camera_depth")
    errors=np.linalg.norm(predicted-uv,axis=1)
    if errors.max()>max_pixel_error:
        return dict(state="inconsistent",reason="nonrigid_image_motion",
                    max_pixel_error=float(errors.max()))
    # Worst local-position sensitivity for an assumed ONE pixel measurement
    # error; even zero residual does not make a weak-baseline depth accurate.
    jacobian=np.column_stack([
        ((projection(local+np.eye(3)[i]*1e-5)[0]-predicted)/1e-5).ravel()
        for i in range(3)])
    minimum=np.linalg.svd(jacobian,compute_uv=False)[-1]
    sigma=float(1/minimum) if minimum>1e-8 else float("inf")
    if sigma>max_sigma_m:
        return dict(state="unknown",reason="weak_depth_baseline",
                    one_pixel_sigma_m=sigma,max_pixel_error=float(errors.max()))
    quality={}
    if stable:
        if len(uv)<5:
            return dict(state="unknown",reason="insufficient_stable_observations")
        # Leave-one-out is a sensitivity check, not a replacement solution.
        shifts=[]
        for omit in range(len(uv)):
            keep=np.arange(len(uv))!=omit
            x=least_squares(lambda point:(projection(point)[0][keep]-uv[keep]).ravel(),
                            local,max_nfev=30).x
            shifts.append(float(np.linalg.norm(x-local)))
        quality=dict(max_leave_one_out_shift_m=max(shifts),
                     assumed_pixel_floor=.25,position_sensitivity_floor_m=.25*sigma)
        if max(shifts)>.003 or .25*sigma>.003:
            return dict(state="unknown",reason="unstable_anchor_sensitivity",
                        one_pixel_sigma_m=sigma,max_pixel_error=float(errors.max()),**quality)
    return dict(state="consistent",reason="moving_rgb_point_binding",
                ee_anchor=local,one_pixel_sigma_m=sigma,
                max_pixel_error=float(errors.max()),roll_observed=False,**quality)


def ee_translation_for_anchor(target, world_rotation, ee_anchor):
    """Position part only; caller still supplies a justified orientation and IK."""
    target,anchor=np.asarray(target,float),np.asarray(ee_anchor,float)
    frame=np.eye(4);frame[:3,:3]=world_rotation
    rigid_transform(frame)
    if target.shape!=(3,) or anchor.shape!=(3,) or not np.isfinite([target,anchor]).all():
        raise ValueError("Finite three-dimensional anchors required")
    return target-frame[:3,:3]@anchor
