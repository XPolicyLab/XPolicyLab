"""Independent wrist-camera observation of a held sphere, no scene truth.

The free wrist can change its view relative to the mallet; the holding wrist
cannot. Proposals use nominal geometry only to aim the camera. Actual RGB rays
and actual robot poses must recover attachment before any strike is planned.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .geometry import Camera
from .xylophone_multiview import K,MOUNT,CV_TO_GL
from .xylophone_measurement import round_heads,saturation_core_heads


def wrist_camera(ee):
    return Camera(dict(K=K,camera_to_world_opengl=ee@MOUNT,resolution=[640,480]))


def observer_candidates(free_ee,held_ee,nominal):
    point=held_ee[:3,:3]@nominal+held_ee[:3,3]
    original=free_ee@MOUNT;out=[]
    for delta in ([0,0,.06],[0,-.03,.06],[0,0,.10],[0,0,0]):
        camera_origin=original[:3,3]+delta
        direction=point-camera_origin;distance=np.linalg.norm(direction)
        if not .23<distance<.85:continue
        direction/=distance;forward=-original[:3,2]
        axis=np.cross(forward,direction);sine=np.linalg.norm(axis)
        if sine<1e-8:continue
        turn=Rotation.from_rotvec(axis/sine*np.arctan2(sine,forward@direction)).as_matrix()
        cam_rotation=turn@original[:3,:3]
        for roll in (0.,15.,-15.):
            c=np.eye(4);c[:3,3]=camera_origin
            c[:3,:3]=Rotation.from_rotvec(direction*np.deg2rad(roll)).as_matrix()@cam_rotation
            target=c@np.linalg.inv(MOUNT)
            if (np.linalg.norm(target[:3,3]-free_ee[:3,3])>.18 or
                np.linalg.norm(target[:3,3]-held_ee[:3,3])<.22 or target[2,3]<.86):continue
            if Rotation.from_matrix(free_ee[:3,:3].T@target[:3,:3]).magnitude()>1.8:continue
            out.append(dict(target=target,nominal_camera_distance_m=float(distance),
                            roll_deg=roll,visibility_is_prediction=True))
    return out


def observer_pixel(rgb,camera,nominal_world):
    if rgb is None:return None
    cv=(nominal_world-camera.c2w[:3,3])@camera.c2w[:3,:3]@CV_TO_GL
    if cv[2]<=.05:return None
    predicted=camera.project([nominal_world])[0]
    # The entire sphere and association uncertainty must be away from edges.
    radius=camera.k[0,0]*.013/cv[2]
    if np.any(predicted<[radius+12,radius+12]) or np.any(predicted>[628-radius,468-radius]):return None
    candidates=round_heads(rgb,radius_bounds=(3.,35.))
    if not candidates:candidates=saturation_core_heads(rgb,radius_bounds=(3.,35.))
    matches=[p['center'] for p in candidates if np.linalg.norm(p['center']-predicted)<max(18.,camera.k[0,0]*.035/cv[2])]
    return matches[0] if len(matches)==1 else None


def confirm_observer(head_camera,observer_samples,head_samples,nominal):
    external=[s for s in observer_samples if s['pixel'] is not None]
    if len(external)<5:return dict(state='unknown',reason='observer_insufficient_visible_frames',visible=len(external))
    poses=[s['ee'] for s in external]
    movement=max(np.linalg.norm(e[:3,3]-poses[0][:3,3]) for e in poses)
    if movement<.025:return dict(state='unknown',reason='observer_insufficient_held_motion')
    records=[(s['camera'],s['ee'],s['pixel']) for s in external]
    records += [(head_camera,e,p) for e,p in head_samples if p is not None]
    matrices=[];targets=[]
    for camera,e,p in records:
        ray=camera.c2w[:3,:3]@CV_TO_GL@np.linalg.solve(camera.k,[*p,1.]);ray/=np.linalg.norm(ray)
        perpendicular=np.eye(3)-np.outer(ray,ray)
        matrices.append(perpendicular@e[:3,:3]);targets.append(perpendicular@(camera.c2w[:3,3]-e[:3,3]))
    a=np.vstack(matrices);b=np.concatenate(targets)
    if np.linalg.svd(a,compute_uv=False)[-1]<1e-5:return dict(state='unknown',reason='observer_depth_unobservable')
    start=np.linalg.lstsq(a,b,rcond=None)[0]
    def residual(x,rows=records):
        return np.concatenate([cam.project([e[:3,:3]@x+e[:3,3]])[0]-p for cam,e,p in rows])
    fit=least_squares(residual,start,max_nfev=45);anchor=fit.x
    errors=np.linalg.norm(residual(anchor).reshape(-1,2),axis=1)
    if np.linalg.norm(anchor-nominal)>.035 or max(errors)>2.:
        return dict(state='inconsistent',reason='observer_nonrigid_or_disagreement',max_pixel_error=float(max(errors)))
    for cam,e,p in records:
        xyz=e[:3,:3]@anchor+e[:3,3]
        if ((xyz-cam.c2w[:3,3])@cam.c2w[:3,:3]@CV_TO_GL)[2]<=0:
            return dict(state='inconsistent',reason='observer_negative_depth')
    singular=np.linalg.svd(fit.jac/np.sqrt(len(records)),compute_uv=False)
    floor=float(.5/singular[-1]) if singular[-1]>1e-8 else float('inf')
    if floor>.003:return dict(state='unknown',reason='observer_uncertainty',sensitivity_floor_m=floor)
    shifts=[]
    for parity in (0,1):
        # Check temporal halves without borrowing the other half's wrist rays.
        half=records[:len(external)][parity::2]+records[len(external):]
        other=least_squares(lambda x:residual(x,half),anchor,max_nfev=35).x
        shifts.append(float(np.linalg.norm(other-anchor)))
    if max(shifts)>.003:return dict(state='unknown',reason='observer_temporal_disagreement',max_split_shift_m=max(shifts))
    return dict(state='consistent',reason='independent_wrist_and_head_rigid_rays',ee_anchor=anchor,
                visible=len(external),head_visible=len(records)-len(external),actual_motion_m=float(movement),
                sensitivity_floor_m=floor,max_pixel_error=float(max(errors)),max_split_shift_m=max(shifts),roll_observed=False)
