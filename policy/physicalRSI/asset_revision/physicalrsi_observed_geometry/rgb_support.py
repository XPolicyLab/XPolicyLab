"""Support geometry from calibrated RGB correspondences, with no supplied scale."""
import copy
import os
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import least_squares
from . import GeometryUnavailable
from .multiview import projection, triangulate

@lru_cache(maxsize=2)
def _matcher(path, device='cpu'):
    import torch
    from kornia.feature import LoFTR
    from kornia.feature.loftr.loftr import default_cfg
    config = copy.deepcopy(default_cfg)
    config['coarse']['temp_bug_fix'] = True
    model = LoFTR(pretrained=None, config=config).eval()
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['state_dict'])
    return model.to(device)

def correspondences(a, b, *, masks=None):
    import torch
    path = os.environ.get('PHYSICALRSI_LOFTR_CHECKPOINT')
    if not path or not Path(path).is_file():
        raise GeometryUnavailable('An installed generic LoFTR checkpoint is required')
    device = os.environ.get('PHYSICALRSI_LOFTR_DEVICE', 'cpu')
    def tensor(camera):
        gray = cv2.cvtColor(camera.image, cv2.COLOR_RGB2GRAY)
        return torch.from_numpy(gray).to(device=device,dtype=torch.float32)[None, None] / 255
    batch = {'image0': tensor(a), 'image1': tensor(b)}
    if masks is not None:
        if len(masks) != 2:
            raise GeometryUnavailable('Two current image masks required')
        for index, (camera, mask) in enumerate(zip((a, b), masks)):
            mask = np.asarray(mask, bool)
            if mask.shape != camera.image.shape[:2] or not mask.any():
                raise GeometryUnavailable('Nonempty matching mask must match current image')
            batch[f'mask{index}'] = torch.from_numpy(mask).to(device=device, dtype=torch.float32)[None]
    with torch.inference_mode():
        result = _matcher(path,device)(batch)
    return tuple(result[key].cpu().numpy() for key in ('keypoints0', 'keypoints1', 'confidence'))

def support_correspondences(cameras, masks, matcher=correspondences, *, unique_head_pixels=True,
                            max_reprojection_error_px=1.):
    if not np.isfinite(max_reprojection_error_px) or max_reprojection_error_px<=0:
        raise GeometryUnavailable('Positive finite correspondence reprojection tolerance required')
    points = []
    source_pixels = []
    head = cameras[0]
    matches = [matcher(head, camera) for camera in cameras[1:]]
    valid_matches = {}
    for camera, mask, (uu, vv, confidence) in zip(cameras[1:], masks[1:], matches):
        for uv, xy, score in zip(uu, vv, confidence):
            u, v = np.rint(uv).astype(int); x, y = np.rint(xy).astype(int)
            if not (score > .2 and 0 <= u < head.width and 0 <= v < head.height
                    and 0 <= x < camera.width and 0 <= y < camera.height
                    and masks[0][v,u] and mask[y,x]):
                continue
            try:
                point, _ = triangulate(head, camera, uv, xy, max_error=max_reprojection_error_px)
                if unique_head_pixels:
                    # Deduplicate only geometrically valid matches: an occluded
                    # or epipolar-invalid high-confidence view must not discard
                    # a valid observation of the same source pixel elsewhere.
                    key = tuple(np.rint(uv).astype(int))
                    if key not in valid_matches or score > valid_matches[key][0]:
                        valid_matches[key] = (score, point, uv)
                else:
                    points.append(point)
                    source_pixels.append(uv)
            except GeometryUnavailable:
                continue
    if unique_head_pixels:
        points = [entry[1] for entry in valid_matches.values()]
        source_pixels = [entry[2] for entry in valid_matches.values()]
    points = np.asarray(points)
    return points, np.asarray(source_pixels)

def initial_plane(cameras, masks, matcher=correspondences, *, unique_head_pixels=False, return_support_mask=False):
    points, source_pixels = support_correspondences(cameras, masks, matcher, unique_head_pixels=unique_head_pixels)
    head = cameras[0]
    if len(points) < 6:
        raise GeometryUnavailable('Insufficient calibrated support correspondences')
    rng = np.random.default_rng(0)
    best = np.zeros(len(points), bool)
    for _ in range(256):
        sample = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(sample[1]-sample[0], sample[2]-sample[0])
        if np.linalg.norm(n) < 1e-10: continue
        n /= np.linalg.norm(n)
        if abs(n[2]) < .9: continue
        selected = np.abs((points-sample[0]) @ n) < .005
        if selected.sum() > best.sum(): best = selected
    if best.sum() < 6:
        raise GeometryUnavailable('No consistent support plane in RGB matches')
    design = np.c_[points[best,:2], np.ones(best.sum())]
    if np.linalg.matrix_rank(design) != 3:
        raise GeometryUnavailable('Support correspondences are collinear')
    plane = np.linalg.lstsq(design, points[best,2], rcond=None)[0]
    if return_support_mask:
        # Bound the proposed surface by the convex hull of observed inliers.
        # Keep the caller's holes/occlusion exclusions inside that hull.
        hull = cv2.convexHull(np.rint(np.asarray(source_pixels)[best]).astype('int32'))
        observed = np.zeros((head.height,head.width),np.uint8)
        cv2.fillConvexPoly(observed,hull,1)
        return plane, observed.astype(bool)&np.asarray(masks[0],bool)
    return plane

def refine_plane(cameras, masks, start, *, minimum_reference_views=2):
    """Minimize local-contrast reprojection error around a measured RGB plane."""
    if type(minimum_reference_views) is not int or minimum_reference_views < 1:
        raise GeometryUnavailable('A positive number of overlapping reference views is required')
    def contrast(camera):
        gray = cv2.cvtColor(camera.image, cv2.COLOR_RGB2GRAY).astype(float)/255
        mean = gaussian_filter(gray, 3)
        return (gray-mean)/np.sqrt(gaussian_filter((gray-mean)**2,3)+.0001)
    source = cameras[0]; gray = contrast(source)
    y, x = np.nonzero(cv2.erode(masks[0].astype('uint8'),np.ones((5,5),'uint8')))
    gradient = np.hypot(cv2.Sobel(gray,cv2.CV_64F,1,0),cv2.Sobel(gray,cv2.CV_64F,0,1))
    keep = gradient[y,x] > .035; x,y = x[keep],y[keep]
    if len(x)>6000:
        keep=np.random.default_rng(139).choice(len(x),6000,replace=False); x,y=x[keep],y[keep]
    origin=source.transform[:3,3]
    rays=source.unproject(np.c_[x,y],np.ones(len(x)))-origin
    def world(plane, selected):
        a,b,c=plane
        depth=(c-origin[2]+a*origin[0]+b*origin[1])/(selected[:,2]-a*selected[:,0]-b*selected[:,1])
        return origin+selected*depth[:,None]
    pairs=[]
    initial=world(start,rays)
    for camera,mask in zip(cameras[1:],masks[1:]):
        p=projection(camera); hom=np.c_[initial,np.ones(len(initial))]@p.T; uv=hom[:,:2]/hom[:,2:]
        valid=(hom[:,2]>0)&(uv[:,0]>4)&(uv[:,0]<camera.width-5)&(uv[:,1]>4)&(uv[:,1]<camera.height-5)
        inside=np.zeros(len(valid),bool)
        inside[valid]=mask[np.rint(uv[valid,1]).astype(int),np.rint(uv[valid,0]).astype(int)]
        valid &= inside
        if valid.sum()>=100: pairs.append((p,contrast(camera),rays[valid],gray[y[valid],x[valid]]))
    if len(pairs)<minimum_reference_views:
        raise GeometryUnavailable('Insufficient overlapping reference views for support refinement')
    def residual(values):
        result=[]
        for i,(p,target,selected,reference) in enumerate(pairs):
            xyz=world(values[:3],selected);hom=np.c_[xyz,np.ones(len(xyz))]@p.T;uv=hom[:,:2]/hom[:,2:]
            predicted=map_coordinates(target,[uv[:,1],uv[:,0]],order=1,mode='nearest')
            result.extend(predicted*values[3+i*2]+values[4+i*2]-reference)
        return np.asarray(result)
    initial=np.r_[start,np.tile([1.,0.],len(pairs))]
    lower=np.r_[start-[.2,.2,.15],np.tile([.9,-.1],len(pairs))]
    upper=np.r_[start+[.2,.2,.15],np.tile([1.1,.1],len(pairs))]
    fits=[]
    for slope in [start[:2],np.zeros(2)]:
        for delta in np.linspace(-.03,.03,7):
            seed=initial.copy();seed[:2]=slope;seed[2]+=delta
            if np.any(seed<=lower) or np.any(seed>=upper):continue
            fits.append(least_squares(residual,seed,bounds=(lower,upper),loss='soft_l1',f_scale=.2,max_nfev=150,diff_step=1e-5,x_scale='jac'))
    fit=min(fits,key=lambda trial:trial.cost)
    if not np.isfinite(fit.x).all() or fit.cost>=np.sum(2*.2**2*(np.sqrt(1+(residual(initial)/.2)**2)-1))/2:
        raise GeometryUnavailable('Support photometric refinement did not improve consistency')
    return fit.x[:3]
