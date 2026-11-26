"""Offline cross-view consistency without fitted registration or LLM calls."""
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from .stmem_geometry import rotation


def statistics(values):
    values=np.asarray(values,float)
    return {'count':int(len(values)),'mean':float(values.mean()),'median':float(np.median(values)),
            'p95':float(np.quantile(values,.95)),'max':float(values.max())} if len(values) else {'count':0}


def compare_views(global_points,*,global_time,local_time,local_calibration,local_mask,local_points=None,identity_status='unverified',global_frame_id=None):
    """Points are global/world and optional local/camera coordinates respectively.

    Requires same-capture evidence and independently established object identity.
    No ICP, centroid alignment, depth rescaling or threshold-based identity selection.
    """
    if identity_status!='matched':return {'status':'unavailable','missing':['verified_cross_view_identity']}
    if not np.isfinite([global_time,local_time]).all() or abs(global_time-local_time)>1e-6:
        return {'status':'unavailable','missing':['same_time_observations'],'global_time':global_time,'local_time':local_time}
    if not isinstance(local_calibration,dict) or any(local_calibration.get(k) is None for k in ['K','world_from_camera']):
        return {'status':'unavailable','missing':['same_frame_wrist_intrinsics_and_extrinsics']}
    if local_calibration.get('units')!='m' or local_calibration.get('convention')!='opencv':raise ValueError('require metres/OpenCV calibration')
    if local_calibration.get('time') is None or abs(local_calibration['time']-local_time)>1e-6:
        return {'status':'unavailable','missing':['timestamped_local_calibration']}
    if not global_frame_id or local_calibration.get('frame_id')!=global_frame_id:
        return {'status':'unavailable','missing':['shared_named_world_coordinate_frame']}
    if local_calibration.get('distortion') is not None and np.any(local_calibration['distortion']) and not local_calibration.get('rectified'):
        return {'status':'unavailable','missing':['rectified_wrist_image_or_distortion_projection']}
    k=np.asarray(local_calibration['K'],float);t=np.asarray(local_calibration['world_from_camera'],float)
    if k.shape!=(3,3) or t.shape!=(4,4) or not np.isfinite(k).all() or not np.isfinite(t).all() or min(k[0,0],k[1,1])<=0 or not np.allclose(t[3],[0,0,0,1]):raise ValueError('invalid calibration')
    rotation(t[:3,:3]);mask=np.asarray(local_mask,bool)
    if mask.ndim!=2 or not mask.any():return {'status':'unavailable','missing':['nonempty_independent_wrist_mask']}
    if local_calibration.get('rgb_shape') is None or tuple(local_calibration['rgb_shape'])!=mask.shape:raise ValueError('wrist mask/calibration image lattice mismatch')
    world=np.asarray(global_points,float)
    if world.ndim!=2 or world.shape[1]!=3 or not len(world) or not np.isfinite(world).all():raise ValueError('invalid global cloud')
    camera=(world-t[:3,3])@t[:3,:3]
    front=camera[:,2]>0
    homogeneous=camera[front]@k.T
    uv=homogeneous[:,:2]/homogeneous[:,2:]
    pixel=np.rint(uv).astype(int);h,w=mask.shape
    inside=(pixel[:,0]>=0)&(pixel[:,0]<w)&(pixel[:,1]>=0)&(pixel[:,1]<h)
    pixel=pixel[inside]
    result={'status':'ok','global_time':global_time,'local_time':local_time,'input_points':len(world),
        'points_in_front':int(front.sum()),'projected_points_in_image':int(inside.sum()),
        'registration':'none','metric_scale_correction':'none',
        'visibility':'self_projection_only_external_occluders_not_removed',
        'interpretation':'Independent-view consistency; partial surfaces/occluders can also cause disagreement. Not an identity gate.'}
    if len(pixel):
        # Multiple global samples can fall on the same wrist pixel. Count pixel support once.
        pixel=np.unique(pixel,axis=0)
        hits=mask[pixel[:,1],pixel[:,0]];distances=distance_transform_edt(~mask)[pixel[:,1],pixel[:,0]]
        result['unique_projected_pixels']=len(pixel);result['projected_mask_hit_fraction']=float(hits.mean())
        result['distance_to_observed_mask_pixels']=statistics(distances)
    else:result['projected_mask_hit_fraction']=None;result['distance_to_observed_mask_pixels']={'count':0}
    if local_points is not None:
        observed=np.asarray(local_points,float)
        if observed.ndim!=2 or observed.shape[1]!=3 or not len(observed) or not np.isfinite(observed).all() or (observed[:,2]<=0).any():raise ValueError('invalid local camera-space cloud')
        predicted=camera[front]
        if len(predicted):
            result['global_to_local_nearest_surface_m']=statistics(cKDTree(observed).query(predicted)[0])
            result['local_to_global_nearest_surface_m']=statistics(cKDTree(predicted).query(observed)[0])
        result['surface_comparison_note']='Nearest-surface distances mix metric error with different visible surfaces; no centroid/ICP fit hides the raw mismatch.'
    else:result['local_3d_status']='missing_independent_local_pointcloud'
    return result
