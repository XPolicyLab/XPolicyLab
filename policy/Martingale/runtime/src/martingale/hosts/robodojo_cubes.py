# Copyright © 2026 Martingale. All Rights Reserved.
from __future__ import annotations
from martingale import self_play_constants as sp
import itertools
import cv2
import numpy as np
from scipy.optimize import least_squares
from . import robodojo_kinematics as kin

ANGLES=np.linspace(0,2*np.pi,64,endpoint=False)
DIRECTIONS=np.c_[np.cos(ANGLES),np.sin(ANGLES)]
CORNERS=np.array(list(itertools.product((-.5,.5),repeat=3)))


def vertices(p,base_z=kin.TABLE_Z):
    x,y,yaw,side=p
    c,s=np.cos(yaw),np.sin(yaw)
    rotation=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    return (CORNERS*side)@rotation.T+[x,y,base_z+side/2]


def detect_blocks(image):
    """Magenta RGB connected components."""
    hsv=cv2.cvtColor(np.asarray(image,np.uint8),cv2.COLOR_RGB2HSV)
    mask=((hsv[:,:,0]>130)&(hsv[:,:,0]<175)&(hsv[:,:,1]>100)&(hsv[:,:,2]>70)).astype('uint8')
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    result=[]
    for c in contours:
        if cv2.contourArea(c)<45:continue
        x,y,w,h=cv2.boundingRect(c)
        if min(x,y)<=1 or x+w>=mask.shape[1]-1 or y+h>=mask.shape[0]-1:continue
        if min(w,h)<5 or max(w,h)>2.5*min(w,h):continue
        filled=np.zeros(mask.shape,'uint8');cv2.drawContours(filled,[c],-1,1,cv2.FILLED)
        result.append(dict(pixels=c[:,0].astype(float),mask=filled.astype(bool),pixel_xy=c[:,0].mean(0)))
    return sorted(result,key=lambda c:c['pixel_xy'][0])


def fit_cube(observations,base_z=kin.TABLE_Z):
    """Fit one unoccluded cube, including size; no known object dimensions."""
    name,view,candidate=observations[0]
    pixels=candidate['pixels'];xy,valid=view.on_plane(pixels[:,1],pixels[:,0],base_z)
    if not valid.all():return dict(accepted=False,reasons=['Invalid table projection.'])
    center,axes,_=cv2.minAreaRect(xy.astype('float32'));side=np.sqrt(max(1e-6,np.prod(axes)))
    side=float(np.clip(side,.006,.10))
    expected=[np.max(c['pixels']@DIRECTIONS.T,axis=0)+.5 for _,_,c in observations]
    def residual(p):
        parts=[]
        for (_,v,_),target in zip(observations,expected):
            r,c,depth=v.project(vertices(p,base_z))
            if np.any(depth<=.01):return np.full(len(observations)*len(DIRECTIONS),1e3)
            parts.append(np.max(np.c_[c,r]@DIRECTIONS.T,axis=0)-target)
        return np.concatenate(parts)
    bounds=([center[0]-.12,center[1]-.12,-np.pi,.005],[center[0]+.12,center[1]+.12,np.pi,.12])
    fits=[least_squares(residual,[*center,yaw,side],bounds=bounds,loss='soft_l1',f_scale=1.,max_nfev=100)
          for yaw in np.linspace(-np.pi/4,np.pi/4,7)]
    best=min(fits,key=lambda f:np.mean(residual(f.x)**2));p=best.x
    rms=np.sqrt(np.mean(residual(p).reshape(len(observations),-1)**2,axis=1))
    jac=np.column_stack([(residual(p+np.eye(4)[i]*1e-5)-residual(p-np.eye(4)[i]*1e-5))/2e-5 for i in range(4)])
    uncertainty=np.sqrt(np.maximum(0,np.diag(np.linalg.pinv(jac.T@jac))))
    ious=[]
    for _,v,candidate in observations:
        rr,cc,_=v.project(vertices(p,base_z));pred=np.zeros(candidate['mask'].shape,'uint8');cv2.fillConvexPoly(pred,cv2.convexHull(np.rint(np.c_[cc,rr]).astype('int32')),1)
        ious.append(float((pred.astype(bool)&candidate['mask']).sum()/max(1,(pred.astype(bool)|candidate['mask']).sum())))
    reasons=[]
    if max(rms)>1.:reasons.append('Outline disagrees with an upright cube; may be occluded, merged or tilted.')
    if min(ious)<sp.CUBE_MIN_IOU:reasons.append('Insufficient cube silhouette coverage.')
    if p[3]<.0051 or p[3]>.119:reasons.append('Size is at a fitting bound.')
    yaw_sensitivity=float(np.degrees(uncertainty[2]))
    if yaw_sensitivity>5:reasons.append('Orientation is poorly constrained.')
    cost=float(np.mean(residual(p)**2))
    ambiguous=any(abs((np.degrees(f.x[2]-p[2])+45)%90-45)>8 and np.mean(residual(f.x)**2)<cost+.05 for f in fits)
    if ambiguous:reasons.append('Multiple headings fit the outline; obtain another view.')
    return dict(accepted=not reasons,reasons=reasons,kind='cube',parameters=p.tolist(),center_xyz=[float(p[0]),float(p[1]),float(base_z+p[3]/2)],side_m=float(p[3]),yaw_deg=float(np.degrees(p[2])%90),base_z=float(base_z),top_z=float(base_z+p[3]),vertices_xyz=vertices(p,base_z).tolist(),supporting_views=[n for n,_,_ in observations],silhouette_rms_px=float(np.sqrt(np.mean(rms**2))),view_errors_px={n:float(e) for (n,_,_),e in zip(observations,rms)},mask_iou=min(ious),yaw_sensitivity_deg=yaw_sensitivity,position_sensitivity_m=float(np.linalg.norm(uncertainty[:2])))


def row_alignment(blocks,position_tolerance_m=.008,angle_tolerance_deg=8.):
    """Geometric evidence only; does not evaluate homing or task success."""
    if len(blocks)!=3 or not all(b.get('accepted',False) for b in blocks):
        return dict(accepted=False,aligned=None,reasons=['Need three distinct accepted current cube estimates.'])
    points=np.array([b['center_xyz'][:2] for b in blocks]);center=points.mean(0)
    if min(np.linalg.norm(a-b) for a,b in itertools.combinations(points,2))<.01:
        return dict(accepted=False,aligned=None,reasons=['Duplicate or unresolved block centres.'])
    _,sv,vh=np.linalg.svd(points-center,full_matrices=False)
    if sv[0]<.02 or sv[1]>.5*sv[0]:
        return dict(accepted=False,aligned=False,reasons=['Blocks do not define a clear row direction.'])
    axis=vh[0];heading=float(np.degrees(np.arctan2(axis[1],axis[0]))%180)
    distances=np.abs((points-center)@np.array([-axis[1],axis[0]]))
    angles=np.array([abs((b['yaw_deg']-heading+45)%90-45) for b in blocks])
    entries=[dict(id=b.get('id',i),distance_m=float(d),angle_error_deg=float(a)) for i,(b,d,a) in enumerate(zip(blocks,distances,angles))]
    pos_margin=max(.002,2*max(b.get('position_sensitivity_m',0) for b in blocks))
    angle_margin=max(2.,2*max(b.get('yaw_sensitivity_deg',0) for b in blocks))
    aligned=bool(max(distances)<position_tolerance_m-pos_margin and max(angles)<angle_tolerance_deg-angle_margin)
    if not aligned and max(distances)<=position_tolerance_m+pos_margin and max(angles)<=angle_tolerance_deg+angle_margin:aligned=None
    return dict(accepted=True,aligned=aligned,row_heading_deg=heading,blocks=entries,max_distance_m=float(max(distances)),max_angle_error_deg=float(max(angles)),position_tolerance_m=position_tolerance_m,angle_tolerance_deg=angle_tolerance_deg,position_margin_m=pos_margin,angle_margin_deg=angle_margin,reasons=[])


def fit_blocks(images,views,detections=None):
    if detections is None: detections={n:detect_blocks(im) for n,im in images.items() if n in views}
    blocks=[]
    for i,candidate in enumerate(detections.get('cam_head',[])):
        anchor=('cam_head',views['cam_head'],candidate);head=fit_cube([anchor]);fits=[head]
        if 'center_xyz' in head:
            seed=np.array(head['center_xyz']);groups=[]
            for n,candidates in detections.items():
                if n=='cam_head':continue
                matches=[]
                for j,c in enumerate(candidates):
                    pixel=c['pixel_xy'];xy,valid=views[n].on_plane(np.array([pixel[1]]),np.array([pixel[0]]),seed[2])
                    if valid[0] and np.linalg.norm(xy[0]-seed[:2])<max(.025,head['side_m']):matches.append((n,views[n],c))
                if matches:groups.append(matches)
            for count in range(1,len(groups)+1):
                for subset in itertools.combinations(groups,count):
                    fits.extend(fit_cube([anchor,*others]) for others in itertools.product(*subset))
        chosen=min(fits,key=lambda f:(not f['accepted'],-len(f.get('supporting_views',[])),f.get('silhouette_rms_px',1e6)))
        blocks.append(dict(id=i,head_pixel_xy=candidate['pixel_xy'].tolist(),**chosen))
    return dict(blocks=blocks,alignment=row_alignment(blocks),note='Current RGB estimate, not task success. Upright cubes resting on the table only. IDs follow current left-to-right head order, not persistent identity. Unknown alignment requires a clearer view; verify block identity and absence of occlusion. 90-degree cube heading symmetry is accounted for.')


def measure(primitives):
    from . import robodojo_tools as t
    obs=t._obs(primitives);views={n:t._view(n,t._shown(obs)) for n in obs.get('vision',{})};views={n:v for n,v in views.items() if v is not None}
    images={n:np.asarray(obs['vision'][n]['color']) for n in views if obs['vision'][n].get('color') is not None}
    result=fit_blocks(images,views)
    result['segmentation_method']='colour_components'
    sam=getattr(primitives,'sam3_client',None)
    if result['alignment']['accepted'] or sam is None:return result
    detections={}
    for n,image in images.items():
        candidates=[]
        for where in ('left','middle','right'):
            segmented=sam.segment(image,text_prompt=f'the {where} magenta block',min_score=.2)
            if not segmented.found:continue
            mask=np.asarray(segmented.mask,bool)
            if mask.shape!=image.shape[:2] or mask.sum()<45:continue
            if any((mask&c['mask']).sum()/max(1,(mask|c['mask']).sum())>.35 for c in candidates):continue
            contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            if not contours:continue
            contour=max(contours,key=cv2.contourArea);x,y,w,h=cv2.boundingRect(contour)
            if min(x,y)<=1 or x+w>=mask.shape[1]-1 or y+h>=mask.shape[0]-1:continue
            filled=np.zeros(mask.shape,'uint8');cv2.drawContours(filled,[contour],-1,1,cv2.FILLED)
            candidates.append(dict(pixels=contour[:,0].astype(float),mask=filled.astype(bool),pixel_xy=contour[:,0].mean(0)))
        detections[n]=sorted(candidates,key=lambda c:c['pixel_xy'][0])
    fallback=fit_blocks(images,views,detections)
    fallback['segmentation_method']='sam3_instance_fallback'
    if sum(b['accepted'] for b in fallback['blocks'])>=sum(b['accepted'] for b in result['blocks']):return fallback
    return result
