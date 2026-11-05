"""Directed static T-template fitting to visible RGB contours only."""
from functools import lru_cache
import cv2
import numpy as np


def translation_contact(footprint, xy, angle, direction):
    """Rear boundary on a line through the public uniform-density centroid.

    The RGB template fit gives xy/angle. Polygon coordinates are millimetres,
    not scene positions; no mass/pose/contact truth is consulted. This is a
    planar quasi-static heuristic, not a guarantee of pure translation.
    """
    points=np.asarray(footprint,float).reshape(-1,2)/1000.
    following=np.roll(points,-1,axis=0)
    cross=points[:,0]*following[:,1]-following[:,0]*points[:,1]
    if abs(cross.sum())<1e-12:raise ValueError('Degenerate footprint')
    centroid=((points+following)*cross[:,None]).sum(axis=0)/(3*cross.sum())
    r=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    vertices=points@r.T+np.asarray(xy);center=r@centroid+np.asarray(xy)
    d=np.asarray(direction,float);norm=np.linalg.norm(d)
    if norm<1e-9:raise ValueError('Missing push direction')
    d=d/norm
    intersections=[]
    for a,b in zip(vertices,np.roll(vertices,-1,axis=0)):
        matrix=np.column_stack([d,a-b])
        if abs(np.linalg.det(matrix))<1e-12:continue
        distance,fraction=np.linalg.solve(matrix,a-center)
        if -1e-9<=fraction<=1+1e-9:intersections.append(distance)
    if not intersections or min(intersections)>=0:raise ValueError('No rear boundary')
    return center+min(intersections)*d


@lru_cache(maxsize=4)
def templates(vertices,scale):
    points=np.array(vertices).reshape(-1,2)*scale/1000
    result=[]
    for degrees in range(-180,180,3):
        angle=np.deg2rad(degrees)
        r=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        for expansion in (1.,1.06,1.12):
            rotated=points@r.T*expansion
            low=np.floor(rotated.min(0));high=np.ceil(rotated.max(0))
            mask=np.zeros(tuple((high-low+3).astype(int)[::-1]),np.uint8)
            cv2.fillPoly(mask,[np.rint(rotated-low+1).astype(np.int32)],1)
            result.append((angle,mask.astype(np.float32),1-low))
    return result


def fit_t_contour(contour,footprint,scale):
    x,y,w,h=cv2.boundingRect(contour)
    if not (20<w<120 and 20<h<120): return None
    margin=12
    crop=np.zeros((h+2*margin,w+2*margin),np.uint8)
    cv2.drawContours(crop,[contour-np.array([[[x-margin,y-margin]]])],-1,1,-1)
    crop=crop.astype(np.float32);area=float(crop.sum())
    candidates=[]
    for angle,mask,root in templates(tuple(footprint.flatten()),scale):
        if mask.shape[0]>crop.shape[0] or mask.shape[1]>crop.shape[1]: continue
        intersection=cv2.matchTemplate(crop,mask,cv2.TM_CCORR)
        iou=intersection/np.maximum(1.,area+float(mask.sum())-intersection)
        _,score,_,location=cv2.minMaxLoc(iou)
        center=np.array([x-margin,y-margin])+location+root
        candidates.append((score,angle,center))
    if not candidates:return None
    candidates.sort(key=lambda item:-item[0])
    best=candidates[0]
    # A competing orientation over 30 degrees away with almost the same
    # overlap indicates ambiguous/occluded shape, not a confident direction.
    rivals=[c for c in candidates if abs(np.arctan2(np.sin(c[1]-best[1]),np.cos(c[1]-best[1])))>np.pi/6]
    margin=best[0]-max((c[0] for c in rivals),default=0.)
    return dict(iou=float(best[0]),angle=float(best[1]),pixel_xy=best[2],direction_margin=float(margin))
