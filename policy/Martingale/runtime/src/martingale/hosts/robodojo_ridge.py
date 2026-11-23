from martingale import self_play_constants as sp
# Copyright © 2026 Martingale. All Rights Reserved.
import cv2
import numpy as np


def detect_ridge(image, mask):
    mask=np.asarray(mask,bool)
    rr,cc=np.nonzero(mask)
    if len(rr)<80:return None
    xy=np.c_[cc,rr].astype(float);center=xy.mean(0)
    _,_,vh=np.linalg.svd(xy-center,full_matrices=False);axis=vh[0]
    normal=np.array([-axis[1],axis[0]])
    along=(xy-center)@axis;across=(xy-center)@normal
    lo,hi=np.quantile(along,sp.RIDGE_SEARCH_QUANTILES)
    hsv=cv2.cvtColor(np.asarray(image,dtype=np.uint8),cv2.COLOR_RGB2HSV)
    dark=(hsv[rr,cc,1]<85)&(hsv[rr,cc,2]<150)
    points=[];occupied=0
    for a in np.arange(lo,hi,2.):
        band=(along>=a)&(along<a+2.)
        if band.sum()<3:continue
        low,high=across[band].min(),across[band].max()
        interior=(across>low+.15*(high-low))&(across<high-.15*(high-low))
        pick=band&interior&dark
        occupied+=1
        if pick.any():
            weights=151.-hsv[rr[pick],cc[pick],2].astype(float)
            points.append(np.average(xy[pick],axis=0,weights=weights))
    if len(points)<12 or len(points)<.6*occupied:return None
    points=np.asarray(points)
    dx,dy,cx,cy=cv2.fitLine(points.astype('float32'),cv2.DIST_HUBER,0,.01,.01).ravel()
    line=np.array([-dy,dx,dy*cx-dx*cy],float)
    errors=np.abs(points@line[:2]+line[2]);inliers=points[errors<sp.RIDGE_INLIER_DISTANCE_PX]
    if len(inliers)<.8*len(points):return None
    _,_,vv=np.linalg.svd(inliers-inliers.mean(0),full_matrices=False);direction=vv[0]
    if abs(direction@axis)<np.cos(np.radians(5)):return None
    line=np.r_[[-direction[1],direction[0]],0.];line[2]=-inliers.mean(0)@line[:2]
    errors=inliers@line[:2]+line[2]
    if np.sqrt(np.mean(errors**2))>.65:return None
    t=(inliers-inliers.mean(0))@direction
    endpoints=inliers.mean(0)+np.outer([t.min(),t.max()],direction)
    return dict(line=line.tolist(),endpoints_xy=endpoints.tolist(),rms_px=float(np.sqrt(np.mean(errors**2))),samples=len(inliers),kind='dark_central_line')
