"""Observed material inside current fingers; permission to continue, not held.

Finger hulls only exclude robot reflection and define an association corridor.
They never paint cube pixels or become a cube contour for a CAD fit.
"""
import cv2
import numpy as np
from .calibration import active_cameras
from .robot_geometry import vertices,rot
from .perception import polygon_mask
from .placement_rgb import observed_bracket


def visible_in_gripper(rgb,ee,closed_mask,head_evidence):
    result=dict(status='unknown',reason='current_finger_material_unresolved')
    if closed_mask is None or not closed_mask.any() or np.mean(np.max(rgb,axis=2)>10)<=.01:
        return dict(result,reason='current_RGB_or_closure_reference_missing')
    if not head_evidence.get('capture_valid'):
        return dict(result,reason='current_head_capture_missing')
    if head_evidence.get('source_still_occupied'):
        return dict(result,reason='visible_source_occupancy_contradiction')
    hsv=cv2.cvtColor(np.ascontiguousarray(rgb),cv2.COLOR_RGB2HSV)
    values=rgb.astype(np.float32);contrast=(values[:,:,0]-values[:,:,1])/(values[:,:,0]+values[:,:,1]+1.)
    hue=(hsv[:,:,0]<14)|(hsv[:,:,0]>170)
    # These are actual dark-red pixels, including a shaded cube face. Current
    # strong red anchors are mandatory; dark wood alone is not an identity.
    strong=hue&(hsv[:,:,1]>145)&(hsv[:,:,2]>135)&(contrast>.30)
    weak=hue&(hsv[:,:,1]>80)&(hsv[:,:,2]>55)&(contrast>.25)
    _,material_labels,_,_=cv2.connectedComponentsWithStats(weak.astype(np.uint8))
    seeded=set(material_labels[strong].tolist())-{0}
    dark=hsv[:,:,2]<95
    _,labels,_,_=cv2.connectedComponentsWithStats(dark.astype(np.uint8))
    bottom=set(labels[-1].tolist())-{0};hulls=[];finger_ids=[]
    for name in ('link7','link8'):
        points=np.concatenate([vertices(j)[name] for j in (0.,.044)]);points=points[points[:,0]>.105]
        uv=active_cameras().wrist_project(rot(ee[3:]).apply(points)+ee[:3],ee)
        predicted=cv2.dilate(polygon_mask(uv,dark.shape).astype(np.uint8),np.ones((17,17),np.uint8)).astype(bool)
        ids=[int(i) for i in bottom if np.count_nonzero((labels==i)&predicted)>=100]
        finger_ids.append(ids);observed=np.isin(labels,ids)&dark
        if not observed.any():return dict(result,reason='two_observed_fingers_missing',finger_components=finger_ids)
        y,x=np.where(observed);hull=np.zeros(dark.shape,np.uint8)
        cv2.fillConvexPoly(hull,cv2.convexHull(np.column_stack((x,y)).astype(np.int32)),1)
        hulls.append(hull.astype(bool))
    if set(finger_ids[0])&set(finger_ids[1]):return dict(result,reason='independent_fingers_unresolved')
    hulls.sort(key=lambda m:np.where(m)[1].mean());corridor=np.zeros(dark.shape,bool)
    for y in range(dark.shape[0]):
        left=np.where(hulls[0][y])[0];right=np.where(hulls[1][y])[0]
        if len(left) and len(right) and left.max()<right.min():corridor[y,left.max()+1:right.min()]=True
    reference=cv2.dilate(closed_mask.astype(np.uint8),np.ones((61,61),np.uint8)).astype(bool)
    selected=np.zeros(dark.shape,bool);records=[]
    # Full-frame connected material is associated only after extraction. A
    # finger-connected reflection can cross a search/image boundary without
    # vetoing the independent material still visible between both fingers.
    for label in seeded:
        full=material_labels==label;current=full&corridor
        pixels=int(current.sum());total=int(full.sum())
        if pixels<120:continue
        overlap=float((current&reference).sum()/pixels)
        anchor=int((current&strong).sum())
        valid=bool(total<.35*full.size and overlap>.70 and anchor>=120 and anchor/pixels>.15)
        records.append(dict(full_pixels=total,corridor_pixels=pixels,strong_pixels=anchor,reference_overlap=overlap,associated=valid))
        if valid:selected|=current
    # Subtractive opening cannot fill a shadow or invent occluded cube pixels.
    selected=cv2.morphologyEx(selected.astype(np.uint8),cv2.MORPH_OPEN,np.ones((3,3),np.uint8)).astype(bool)&selected
    count=int(selected.sum());reference_count=int((closed_mask&corridor).sum())
    ratio=count/max(1,reference_count);total_ratio=count/max(1,int(closed_mask.sum()))
    bracket=observed_bracket(rgb,np.asarray(ee),selected) if count else dict(ok=False)
    # This clipped support is deliberately not a silhouette fit. Require
    # substantial observed material relative to both closure and its current
    # finger-visible portion; arbitrary tiny red fragments cannot authorize.
    ok=bool(count>=120 and reference_count>=.25*closed_mask.sum() and .55<ratio<1.65 and .32<total_ratio<1.6 and bracket['ok'])
    result.update(material_components=records,finger_components=finger_ids,current_pixels=count,reference_corridor_pixels=reference_count,visible_area_ratio=ratio,total_reference_area_ratio=total_ratio,finger_bracket=bracket,attachment_confirmed=False,head_source_visibility=head_evidence.get('status'))
    if ok:result.update(status='visible_in_gripper',reason='substantial_current_material_between_observed_fingers_attachment_uncertain',mask=selected)
    return result
