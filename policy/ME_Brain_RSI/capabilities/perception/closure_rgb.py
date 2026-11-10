"""Current cube / independent finger inner edges for closure centering.

This supplies alignment, never an attachment claim. The plane conversion uses
public cube geometry and the disclosed static wrist prior, not measured depth.
"""
import cv2
import numpy as np
from .calibration import active_cameras
from .perception import wrist_cube_evidence,polygon_mask
from .robot_geometry import vertices,rot

def closure_alignment(rgb,ee,source):
    unknown=dict(status='unknown',reason='current_cube_or_two_finger_edges_unresolved',attachment_confirmed=False)
    if float(np.mean(np.max(rgb,axis=2)>10))<=.01:return dict(unknown,reason='missing_current_RGB')
    fit=wrist_cube_evidence(rgb,ee,source)
    if not fit['ok']:return dict(unknown,identity=fit)
    mask=fit['mask'];hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);dark=hsv[:,:,2]<95
    _,labels,_,_=cv2.connectedComponentsWithStats(dark.astype(np.uint8));bottom=set(labels[-1].tolist())-{0};fingers=[];ids=[]
    for name in ('link7','link8'):
        pts=np.concatenate([vertices(j)[name] for j in (0.,.044)]);pts=pts[pts[:,0]>.105]
        uv=active_cameras().wrist_project(rot(ee[3:]).apply(pts)+ee[:3],ee)
        possible=cv2.dilate(polygon_mask(uv,mask.shape).astype(np.uint8),np.ones((17,17),np.uint8)).astype(bool)
        chosen=[int(i) for i in bottom if np.count_nonzero((labels==i)&possible)>=100];ids.append(chosen);fingers.append(np.isin(labels,chosen)&dark)
    if any(not f.any() for f in fingers) or set(ids[0])&set(ids[1]):return dict(unknown,reason='independent_current_fingers_unresolved',finger_components=ids)
    fingers.sort(key=lambda f:np.where(f)[1].mean());yy,xx=np.where(mask);rows=[]
    for y in range(int(np.median(yy)),int(yy.max())+1):
        xs=np.where(mask[y])[0]
        if not len(xs):continue
        l=np.where(fingers[0][y,:xs.min()])[0];r=np.where(fingers[1][y,xs.max()+1:])[0]+xs.max()+1
        if len(l) and len(r):
            gaps=[int(xs.min()-l[-1]),int(r[0]-xs.max())]
            if max(gaps)<50:rows.append([y,(xs.min()+xs.max())/2,(l[-1]+r[0])/2,*gaps])
    if len(rows)<12:return dict(unknown,reason='current_inner_edges_insufficient',rows=len(rows),cube_pixels=int(mask.sum()))
    rows=np.asarray(rows);offset=float(np.median(rows[:,1]-rows[:,2]));pixel=np.median(rows[:,:2],axis=0)[::-1]
    world=active_cameras().wrist_on_plane(np.array([pixel,pixel-[offset,0.]]),ee,float(source[2]))
    delta=world[0,:2]-world[1,:2]
    return dict(status='aligned' if abs(offset)<=1.5 else 'recenter',reason='current_cube_vs_observed_finger_midline',offset_px=offset,median_gaps_px=np.median(rows[:,3:],axis=0).tolist(),delta_xy_m=delta.tolist(),cube_xy=fit['xy'],cube_pixels=int(mask.sum()),qualified_rows=len(rows),finger_components=ids,attachment_confirmed=False,plane_and_camera='public_cube_center_and_static_wrist_prior',raster_deadband_px=1.5)
