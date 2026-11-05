"""X10 RGB measurement primitives; no simulator state, reward or file inputs.

Ellipse centers use every contour point and several fixed color thresholds.
Dense shaft search keeps full-length bilateral/wood support: it does not
equate a red blob or a short stick fragment with a complete mallet.
"""
import cv2
import numpy as np


def round_heads(rgb, radius_bounds=(3.,12.), *, thresholds=((110,35),(120,40),(130,45))):
    estimates=[]
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    h,s,v=cv2.split(hsv)
    for saturation,value in thresholds:
        mask=np.uint8(((h<=7)|(h>=160))&(s>=saturation)&(v>=value))*255
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        group=[]
        for contour in contours:
            area=cv2.contourArea(contour);per=cv2.arcLength(contour,True)
            _,radius=cv2.minEnclosingCircle(contour)
            if len(contour)<8 or per<=0 or not radius_bounds[0]<=radius<=radius_bounds[1]:
                continue
            if 4*np.pi*area/per**2<.65 or area/(np.pi*radius**2)<.68:
                continue
            center,axes,angle=cv2.fitEllipse(contour)
            if min(axes)<=0 or min(axes)/max(axes)<.70:continue
            points=contour[:,0,:].astype(float)-center
            theta=np.deg2rad(angle);c,ss=np.cos(theta),np.sin(theta)
            local=points@np.array([[c,-ss],[ss,c]])
            radial=np.linalg.norm(local/(np.asarray(axes)/2),axis=1)
            residual=float(np.sqrt(np.mean((radial-1)**2))*min(axes)/2)
            if residual>.8:continue
            group.append((np.array(center),residual))
        estimates.append(group)
    result=[]
    for center,residual in estimates[1]:
        matched=[(center,residual)]
        for group in (estimates[0],estimates[2]):
            nearby=[p for p in group if np.linalg.norm(p[0]-center)<=1.]
            if len(nearby)!=1:break
            matched+=nearby
        if len(matched)!=3:continue
        centers=np.array([p[0] for p in matched])
        spread=float(np.max(np.linalg.norm(centers-centers.mean(0),axis=1)))
        if spread>.5:continue
        result.append(dict(center=centers.mean(0),threshold_spread_px=spread,
                           contour_rms_px=max(p[1] for p in matched),
                           measurement_floor_px=.25))
    return result


def saturation_core_heads(rgb, radius_bounds=(3.,12.)):
    """Failure-only bounded color-core proposals, not relaxed shape checks.

    A low-saturation reflection can join a painted sphere to its surroundings.
    Each of three fixed windows still needs three agreeing thresholds and all
    original circle/ellipse residual tests. Cross-window disagreement is not
    repaired by selecting a favorable subset. Complete shaft evidence remains
    the caller's responsibility; these are not standalone tool detections.
    """
    proposals=[]
    for saturation in (140,160,180):
        proposals.extend(round_heads(rgb,radius_bounds,thresholds=(
            (saturation-10,35),(saturation,40),(saturation+10,45))))
    groups=[]
    for item in proposals:
        nearby=[g for g in groups if any(
            np.linalg.norm(item['center']-old['center'])<=1. for old in g)]
        if len(nearby)>1:return []
        if nearby:nearby[0].append(item)
        else:groups.append([item])
    result=[]
    for group in groups:
        centers=np.array([p['center'] for p in group])
        if np.max(np.linalg.norm(centers[:,None]-centers[None,:],axis=2))>1.:
            return []
        center=centers.mean(0)
        result.append(dict(center=center,
            threshold_spread_px=max(float(np.max(np.linalg.norm(centers-center,axis=1))),
                                    max(p['threshold_spread_px'] for p in group)),
            contour_rms_px=max(p['contour_rms_px'] for p in group),
            measurement_floor_px=.25,threshold_windows=len(group)))
    return result


def dense_shaft(rgb, heads, scale=600):
    """Direction proposal without requiring a lucky Hough segment.

    All 30--190 mm samples must be in frame. Each side needs >=80% edge
    support and the center >=80% brighter than BOTH outside strips.
    Width is still 4--12 mm; ambiguity across heads/directions is rejected.
    """
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    distance=cv2.distanceTransform(255-cv2.Canny(gray,30,90),cv2.DIST_L2,3)
    angles=np.arange(0,2*np.pi,np.arctan2(.25,.19*scale))
    axes=np.c_[np.cos(angles),np.sin(angles)]
    normals=np.c_[-axes[:,1],axes[:,0]]
    outputs=[]
    for head in heads:
        base=head+axes[:,None,:]*np.linspace(.03,.19,40)[None,:,None]*scale
        def values(offset,source):
            xy=base+normals[:,None,:]*offset
            inside=(xy[:,:,0]>=0)&(xy[:,:,0]<gray.shape[1]-1)&(xy[:,:,1]>=0)&(xy[:,:,1]<gray.shape[0]-1)
            val=cv2.remap(source.astype(np.float32),xy[:,:,0].astype(np.float32),
                          xy[:,:,1].astype(np.float32),cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,borderValue=255)
            return val,inside.all(1)
        candidates=[]
        center,_=values(0,gray)
        for half in np.arange(.002*scale,.006*scale+.1,.25):
            left,il=values(-half,distance);right,ir=values(half,distance)
            outer_l,_=values(-half-2,gray);outer_r,_=values(half+2,gray)
            support=il&ir&((left<=1.5).mean(1)>=.8)&((right<=1.5).mean(1)>=.8)
            support&=((center>outer_l)&(center>outer_r)).mean(1)>=.8
            for i in np.flatnonzero(support):
                candidates.append((float((left[i]+right[i]).mean()),i,2*half))
        if not candidates:continue
        candidates.sort()
        error,index,width=candidates[0]
        if any(axes[index]@axes[i]<.98 for _,i,_ in candidates):return None
        close=[axes[i] for err,i,_ in candidates if err<=error+.05 and axes[index]@axes[i]>.999]
        axis=np.mean(close,axis=0);axis/=np.linalg.norm(axis)
        outputs.append(dict(head_px=head.copy(),axis_px=axis,
            handle_end_estimated_px=head+axis*.2006*scale,shaft_width_m=width/scale,
            observability="planar_head_and_axis_only",roll_observed=False,
            depth_observed=False,shaft_fit="dense_bilateral_subpixel"))
    if len(outputs)!=1:return None
    return outputs[0]


def elliptical_heads(rgb, radius_bounds=(3.,12.)):
    estimates=[]
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    h,s,v=cv2.split(hsv)
    for saturation,value in ((110,35),(120,40),(130,45)):
        mask=np.uint8(((h<=7)|(h>=160))&(s>=saturation)&(v>=value))*255
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        group=[]
        for contour in contours:
            area=cv2.contourArea(contour);per=cv2.arcLength(contour,True)
            _,radius=cv2.minEnclosingCircle(contour)
            if len(contour)<8 or per<=0 or not radius_bounds[0]<=radius<=radius_bounds[1]:
                continue
            if 4*np.pi*area/per**2<.65:
                continue
            center,axes,angle=cv2.fitEllipse(contour)
            if min(axes)<=0 or min(axes)/max(axes)<.70:continue
            # A rectified sphere need not fill its enclosing CIRCLE. Require
            # filled ellipse support instead, without lowering circularity,
            # axis ratio, radial residual or three-threshold consistency.
            if not .85<=area/(np.pi*np.prod(axes)/4)<=1.15:continue
            points=contour[:,0,:].astype(float)-center
            theta=np.deg2rad(angle);c,ss=np.cos(theta),np.sin(theta)
            local=points@np.array([[c,-ss],[ss,c]])
            radial=np.linalg.norm(local/(np.asarray(axes)/2),axis=1)
            residual=float(np.sqrt(np.mean((radial-1)**2))*min(axes)/2)
            if residual>.8:continue
            group.append((np.array(center),residual))
        estimates.append(group)
    result=[]
    for center,residual in estimates[1]:
        matched=[(center,residual)]
        for group in (estimates[0],estimates[2]):
            nearby=[p for p in group if np.linalg.norm(p[0]-center)<=1.]
            if len(nearby)!=1:break
            matched+=nearby
        if len(matched)!=3:continue
        centers=np.array([p[0] for p in matched])
        spread=float(np.max(np.linalg.norm(centers-centers.mean(0),axis=1)))
        if spread>.5:continue
        result.append(dict(center=centers.mean(0),threshold_spread_px=spread,
                           contour_rms_px=max(p[1] for p in matched),
                           measurement_floor_px=.25))
    return result
