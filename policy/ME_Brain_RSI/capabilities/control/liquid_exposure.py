"""Shadowed upright bottle-rim evidence from RGB exposure-level agreement.

Independent implementation; uses existing public mouth plane and geometry.
Only an initial-observation fallback, never a held-bottle pose estimator.
"""
import cv2
import numpy as np
from .rim_geometry import bottle_boundary_circle


def shadowed_bottle_rims(rgb, camera, plane_z):
    values=np.asarray(rgb,dtype=float)
    if values.ndim!=3 or values.shape[2]!=3:return []
    levels=[]
    for threshold in (65,85,105):
        mask=np.uint8((values.min(2)>threshold)&(np.ptp(values,axis=2)<40))*255
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        candidates=[]
        for contour in contours:
            if len(contour)<5 or not 45<cv2.contourArea(contour)<1800:continue
            points=np.array([camera.pixel_to_plane(uv,plane_z)[:2]
                             for uv in contour[:,0]],np.float32)
            metric=points*1000
            hull=cv2.convexHull(metric);area=cv2.contourArea(hull)
            perimeter=cv2.arcLength(hull,True)
            if area<=0 or perimeter<=0:continue
            circularity=4*np.pi*area/perimeter**2
            solidity=cv2.contourArea(metric)/area
            center,radius=cv2.minEnclosingCircle(hull)
            xy=np.asarray(center)/1000;diameter=radius/500
            if not (.017<diameter<.033 and circularity>.88
                    and -.48<xy[0]<.48 and -.32<xy[1]<.18):continue
            # Lower levels must retain the original solidity requirement,
            # plus a well-covered, low-residual fitted circular boundary.
            if threshold<105 and (solidity<=.85 or bottle_boundary_circle(points) is None):
                continue
            candidates.append((xy,diameter,contour))
        levels.append(candidates)
    found=[]
    for center,diameter,contour in levels[0]:
        matches=[]
        for candidates in levels[1:]:
            nearby=[c for c in candidates if np.linalg.norm(c[0]-center)<.001
                    and abs(c[1]-diameter)<.002]
            if len(nearby)!=1:break
            matches.append(nearby[0])
        if len(matches)!=2:continue
        if any(np.linalg.norm(center-other[0])<.005 for other in found):continue
        found.append((center,diameter,contour))
    return found
