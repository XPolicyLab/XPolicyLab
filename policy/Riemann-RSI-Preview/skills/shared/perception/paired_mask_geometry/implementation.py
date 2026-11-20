"""Current instance geometry from two public masks and a fixed X5 camera rig."""
import cv2
import numpy as np


def rotation(q):
    q = np.asarray(q, float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        raise ValueError('A finite public robot quaternion is required')
    w,x,y,z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def camera(name, ee_pose=None):
    # Fixed public X5 camera mounts; no task/object geometry.
    if name == 'cam_head':
        pose = np.eye(4)
        pose[:3,:3] = np.array([[1,0,0],[0,.8660254038,-.5],[0,.5,.8660254038]])
        pose[:3,3] = [0,-.41,1.308]
        return pose,288.1325378417969
    if name not in ('cam_left_wrist','cam_right_wrist'):
        raise ValueError('Only native public cameras are supported')
    ee_pose = np.asarray(ee_pose, float)
    if ee_pose.shape != (7,) or not np.isfinite(ee_pose).all():
        raise ValueError('A public measured end-effector pose is required')
    pose = np.eye(4);pose[:3,:3] = rotation(ee_pose[3:]);pose[:3,3] = ee_pose[:3]
    mount = np.array([[0,.5,-.8660254038,.084842],[-1,0,0,0],
                      [0,.8660254038,.5,.05094],[0,0,0,1]])
    return pose @ mount,397.0413


def rays(model, pixels):
    pose,focal = model
    pixels = np.asarray(pixels,float).reshape(-1,2)
    if not np.isfinite(pixels).all() or np.any(pixels < 0) or np.any(pixels > [639,479]):
        raise ValueError('Rays must start inside the public image')
    local = np.c_[(pixels[:,0]-319.5)/focal,-(pixels[:,1]-239.5)/focal,-np.ones(len(pixels))]
    directions = np.einsum('ij,kj->ik',local,pose[:3,:3])
    directions /= np.linalg.norm(directions,axis=1)[:,None]
    return pose[:3,3],directions


def project(model, points):
    pose,focal = model
    local = (np.asarray(points)-pose[:3,3]) @ pose[:3,:3]
    return np.c_[319.5+focal*local[:,0]/(-local[:,2]),239.5-focal*local[:,1]/(-local[:,2])]


def centroid(mask):
    y,x = np.where(mask)
    return np.array([x.mean(),y.mean()])


def pair(head_mask, wrist_mask, wrist_model, support_z):
    head_model = camera('cam_head')
    ho,hd = rays(head_model,[centroid(head_mask)])
    wo,wd = rays(wrist_model,[centroid(wrist_mask)])
    t = np.linalg.lstsq(np.c_[hd[0],-wd[0]],wo-ho,rcond=None)[0]
    first,second = ho+t[0]*hd[0],wo+t[1]*wd[0]
    centre = (first+second)/2
    separation = float(np.linalg.norm(first-second))
    if min(t) <= 0 or separation > .02 or not support_z-.01 <= centre[2] <= support_z+.16:
        return None
    y,x = np.where(head_mask);pixels = np.c_[x,y]
    origin,directions = rays(head_model,pixels)
    points = origin+(centre[2]-origin[2])/directions[:,2,None]*directions
    covariance = np.cov(points[:,:2].T)
    values,vectors = np.linalg.eigh(covariance);axis = vectors[:,-1]
    return {'centre':centre.tolist(),'yaw':float(np.arctan2(axis[1],axis[0])),
            'ray_separation_m':separation,'anisotropy':float(values[-1]/max(values[0],1e-12)),
            'head_pixel':centroid(head_mask).tolist(),'wrist_pixel':centroid(wrist_mask).tolist()}


def foreground_candidates(head_rgb, background_rgb, wrist_mask, wrist_model, support_z):
    """A semantic wrist cone selects full current head foreground components."""
    y,x = np.where(wrist_mask)
    outline = cv2.convexHull(np.c_[x,y].astype(np.float32)).reshape(-1,2)
    origin,directions = rays(wrist_model,outline)
    points = []
    for z in (support_z,support_z+.14):
        scale = (z-origin[2])/directions[:,2]
        if np.any(scale<=0):
            return []
        points.extend(origin+scale[:,None]*directions)
    projected = project(camera('cam_head'),points)
    if not np.isfinite(projected).all():
        return []
    cone = np.zeros((480,640),np.uint8)
    cv2.fillConvexPoly(cone,cv2.convexHull(np.clip(projected,-10000,10000).astype(np.int32)),1)
    foreground = (np.max(cv2.absdiff(head_rgb,background_rgb),axis=2)>45).astype(np.uint8)
    foreground = cv2.morphologyEx(foreground,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    count,labels,stats,_ = cv2.connectedComponentsWithStats(foreground)
    candidates=[]
    for i in range(1,count):
        area=stats[i,cv2.CC_STAT_AREA]
        if not 30 <= area <= 15000:
            continue
        mask=labels==i
        if np.count_nonzero(mask & (cone>0))/area >= .65:
            candidates.append(mask)
    return candidates


def run(head_rgb, background_rgb, head_regions, wrist_regions, wrist_model, support_z, excluded_head_mask=None):
    for image in (head_rgb,background_rgb):
        if np.asarray(image).shape != (480,640,3) or np.asarray(image).dtype != np.uint8:
            raise ValueError('Native public RGB arrays are required')
    if not np.isfinite(support_z) or not 0 < support_z < 2:
        raise ValueError('A finite robot-relative support calibration is required')
    for regions in (head_regions,wrist_regions):
        if not isinstance(regions,list) or len(regions)>8:
            raise ValueError('At most eight current semantic masks are accepted')
        for region in regions:
            mask=np.asarray(region['mask'])
            if mask.shape!=(480,640) or mask.dtype!=bool or mask.sum()<30:
                raise ValueError('Nonempty native Boolean instance masks are required')
    if excluded_head_mask is None:
        excluded_head_mask=np.zeros((480,640),bool)
    if np.asarray(excluded_head_mask).shape!=(480,640) or np.asarray(excluded_head_mask).dtype!=bool:
        raise ValueError('Exclusions must be an explicit public image mask')
    candidates=[]
    # A semantic detection in an excluded basket is not a current target view.
    # Filter first so that such a detection cannot suppress the wrist-cone
    # foreground fallback for the actual body on the table.
    qualified_head=[r['mask'] for r in head_regions[:8]
                    if np.count_nonzero(r['mask'] & excluded_head_mask)/np.count_nonzero(r['mask'])<=.15]
    for wrist in wrist_regions[:8]:
        masks=qualified_head
        if not masks:
            masks=foreground_candidates(head_rgb,background_rgb,wrist['mask'],wrist_model,support_z)
        for mask in masks:
            if np.count_nonzero(mask & excluded_head_mask)/np.count_nonzero(mask)>.15:
                continue
            estimate=pair(mask,wrist['mask'],wrist_model,support_z)
            if estimate is not None:
                candidates.append({**estimate,'head_mask':mask,'semantic_confidence':wrist['score']})
    candidates.sort(key=lambda r:r['ray_separation_m'])
    if not candidates:
        return {'state':'unknown','reason':'no_qualified_paired_public_instance'}
    best=candidates[0]
    if any(np.linalg.norm(np.asarray(c['centre'])[:2]-np.asarray(best['centre'])[:2])>.04
           and c['ray_separation_m']<best['ray_separation_m']+.005 for c in candidates[1:]):
        return {'state':'unknown','reason':'multiple_paired_public_instances'}
    return {'state':'observed','reason':'paired_public_mask_centres','physical_pose_claimed':False,**best}
