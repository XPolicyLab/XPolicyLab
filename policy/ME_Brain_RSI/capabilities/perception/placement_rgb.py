"""Current RGB permission for controlled placement; never an attachment verdict."""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from .calibration import active_cameras
from .robot_geometry import vertices,rot
from .perception import BOX,MAT_TOP,polygon_mask,cube_material_candidates,head_source_crosscheck

def observed_bracket(rgb,ee,mask):
    # Independent current dark finger components anchored at image bottom and
    # intersecting the public full-opening finger geometry. No commanded jaw
    # position is used to infer actual contact.
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);dark=hsv[:,:,2]<95
    _,lab,_,_=cv2.connectedComponentsWithStats(dark.astype(np.uint8));anchored=np.isin(lab,list(set(lab[-1].tolist())-{0}))&dark
    fingers=[];ids=[]
    for name in ('link7','link8'):
        pts=np.concatenate([vertices(j)[name] for j in (0.,.044)]);pts=pts[pts[:,0]>.105]
        uv=active_cameras().wrist_project(rot(ee[3:]).apply(pts)+ee[:3],ee)
        pred=cv2.dilate(polygon_mask(uv,mask.shape).astype(np.uint8),np.ones((17,17),np.uint8)).astype(bool)
        found=[int(i) for i in np.unique(lab[anchored]) if i and np.count_nonzero((lab==i)&pred)>=100]
        ids.append(found);fingers.append(np.isin(lab,found)&anchored)
    if not all(ids) or set(ids[0])&set(ids[1]):return dict(ok=False,reason='independent_current_fingers_unresolved',components=ids)
    fingers.sort(key=lambda p:np.where(p)[1].mean());yy,xx=np.where(mask);rows=[]
    for y in range(int(np.median(yy)),int(yy.max())+1):
        xs=np.where(mask[y])[0]
        if not len(xs):continue
        left=np.where(fingers[0][y,:xs.min()])[0];right=np.where(fingers[1][y,xs.max()+1:])[0]+xs.max()+1
        if len(left) and len(right):
            gaps=[int(xs.min()-left[-1]),int(right[0]-xs.max())]
            if 0<min(gaps) and max(gaps)<50:rows.append(gaps)
    return dict(ok=len(rows)>=12,reason='current_material_between_two_observed_fingers' if len(rows)>=12 else 'current_material_not_bracketed',qualified_rows=len(rows),median_gaps_px=np.median(rows,axis=0).tolist() if rows else None,components=ids)

def placement_ready(rgb,head,own,arm,source,destination,closed_mask,closed_ee,cube_height):
    cameras=active_cameras();ee=np.asarray(own[arm+'_ee_pose']);source=np.asarray(source);destination=np.asarray(destination)
    result=dict(status='unknown',reason='placement_current_material_unresolved')
    if float(np.mean(np.max(rgb,axis=2)>10))<=.01 or closed_mask is None or not closed_mask.any():return dict(result,reason='placement_missing_current_RGB_or_reference')
    head_source=head_source_crosscheck(head,source,own);result['head_source']=head_source
    if not head_source.get('capture_valid') or not head_source.get('actionable') or head_source.get('source_still_occupied'):
        return dict(result,reason='placement_source_absence_unconfirmed')
    # Search around the gripper-local reference, not the now offscreen source.
    reference=source+ee[:3]-np.asarray(closed_ee[:3]);envelope=np.concatenate([Rotation.from_euler('z',a).apply(BOX) for a in np.linspace(-np.pi/4,np.pi/4,9)])
    area=polygon_mask(cameras.wrist_project(envelope+reference,ee),rgb.shape[:2]);area=cv2.dilate(area.astype(np.uint8),np.ones((91,91),np.uint8)).astype(bool)
    # Segment full observed components before using the search for selection.
    # A crop must not manufacture a silhouette or turn a valid shifted cube
    # into a border fragment. Image-edge/background rejection remains active.
    candidates,material=cube_material_candidates(rgb,np.ones(area.shape,bool));result['material_evidence']=material;result['candidates']=[]
    expanded=cv2.dilate(closed_mask.astype(np.uint8),np.ones((13,13),np.uint8)).astype(bool)
    for mask,record in candidates:
        if float((mask&area).sum()/max(1,mask.sum()))<.85:continue
        yy,xx=np.where(mask);contour=max(cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],key=cv2.contourArea);perimeter=cv2.arcLength(contour,True);corners=len(cv2.approxPolyDP(contour,.025*perimeter,True));shape_ok=4<=corners<=7
        ratio=float(mask.sum()/closed_mask.sum());coverage=float((mask&expanded).sum()/max(1,mask.sum()));bracket=observed_bracket(rgb,ee,mask)
        # Public-size ray envelope: consider all plausible cube-center heights
        # between destination support and the current gripper. This is an XY
        # placement envelope, not a precise depth or physical attachment fit.
        pixel=np.array([(xx.min()+xx.max())/2,(yy.min()+yy.max())/2]);width=xx.max()-xx.min()+1;plausible=[]
        for z in np.linspace(MAT_TOP+cube_height/2,max(MAT_TOP+cube_height/2,ee[2]-.10),17):
            point=cameras.wrist_on_plane([pixel],ee,z)[0];uv=cameras.wrist_project(envelope+point,ee);projected_width=float(np.ptp(uv[:,0]));size_ratio=width/max(1.,projected_width)
            if .70<size_ratio<1.25:plausible.append(dict(z=float(z),xy=point[:2].tolist(),target_xy_error_m=float(np.linalg.norm(point[:2]-destination)),size_ratio=float(size_ratio)))
        # Require the whole compatible XY interval to stay inside the target
        # neighborhood, rather than choosing one favorable depth from it.
        target_error=max((x['target_xy_error_m'] for x in plausible),default=float('inf'))
        ok=bool(shape_ok and mask.sum()>=120 and .45<ratio<1.6 and coverage>.60 and bracket['ok'] and plausible and target_error<.022)
        row=dict(material=record,polygon_corners=corners,compact_cube_shape=shape_ok,area_ratio=ratio,reference_coverage=coverage,finger_bracket=bracket,public_size_ray_envelope=plausible,maximum_target_xy_error_m=target_error if np.isfinite(target_error) else None,ready=ok);result['candidates'].append(row)
        if ok:return dict(result,status='placement_ready',reason='current_bracketed_cube_over_target_after_verified_lift',mask=mask,evidence=row)
    return dict(result,reason='placement_material_bracket_or_target_unresolved')
