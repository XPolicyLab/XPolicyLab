"""Bounded two-view silhouette estimate; no depth or simulator object state."""
import numpy as np
from .geometry import View, Sight, measure


def run(head_mask, wrist_mask, head_camera, wrist_camera, support_z):
    unknown={'state':'unknown','reason':'two_view_surface_unqualified'}
    masks=[]
    for mask in (head_mask,wrist_mask):
        mask=np.asarray(mask)
        if mask.shape!=(480,640) or mask.dtype!=np.bool_:
            raise ValueError('Two explicit public Boolean 480x640 masks are required')
        if mask.sum()<20:return unknown
        masks.append(mask)
    views=[]
    for pose,focal in (head_camera,wrist_camera):
        pose=np.asarray(pose,dtype=float)
        if pose.shape!=(4,4) or not np.isfinite(pose).all() or not np.isfinite(focal) or focal<=0:
            raise ValueError('Finite public pinhole camera calibration required')
        views.append(View([[focal,0,319.5],[0,focal,239.5],[0,0,1]],pose))
    estimate=measure(Sight(views[0],masks[0]),[Sight(views[1],masks[1])],base_z=float(support_z))
    if estimate is None or estimate['views']<2 or estimate['match']<.85:return unknown
    height=float(estimate['height_m'])
    if not np.isfinite(height) or not .004<=height<=.16:return unknown
    return {'state':'observed','height_above_support_m':height,
            'match':float(estimate['match']),'views':int(estimate['views']),
            'kind':estimate['kind'],'height_resolution_m':.002,
            'physical_depth_or_pose_claimed':False}
