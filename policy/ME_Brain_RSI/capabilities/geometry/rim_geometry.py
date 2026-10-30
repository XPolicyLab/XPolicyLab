"""Independent circular boundary evidence for a partially dark bottle opening.

Uses RGB contours projected onto the public upright mouth-height plane only.
Not valid for a held or tilted bottle, and not an object-identity classifier.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares


def bottle_boundary_circle(world_points):
    points=np.asarray(world_points,dtype=np.float32).reshape(-1,2)*1000
    if len(points)<10 or not np.isfinite(points).all():return None
    hull=cv2.convexHull(points).reshape(-1,2)
    if len(hull)<10:return None
    area=cv2.contourArea(hull);perimeter=cv2.arcLength(hull,True)
    if area<=0 or perimeter<=0:return None
    if 4*np.pi*area/perimeter**2<.92:return None
    if cv2.contourArea(points)/area<.65:return None
    center,radius=cv2.minEnclosingCircle(hull)
    fit=least_squares(lambda x:np.linalg.norm(hull-x[:2],axis=1)-x[2],
                      [*center,radius],max_nfev=40)
    if not fit.success or not 17<2*fit.x[2]<33:return None
    if np.sqrt(np.mean(fit.fun**2))>.5 or np.max(np.abs(fit.fun))>1:return None
    angles=np.sort(np.arctan2(hull[:,1]-fit.x[1],hull[:,0]-fit.x[0]))
    gaps=np.diff(np.r_[angles,angles[0]+2*np.pi])
    if gaps.max()>np.pi/3:return None
    return fit.x[:2]/1000,2*fit.x[2]/1000
