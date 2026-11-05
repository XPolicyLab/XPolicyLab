"""Current RGB motion evidence; no commanded jaw value is attachment evidence."""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from .calibration import active_cameras
from .perception import BOX,polygon_mask,cube_material_candidates,fit_cube,head_source_crosscheck

def grasp_state(rgb,head,own_state,arm,source,closed_mask,closed_ee):
    """Return held / at_source / visible_in_gripper / unknown.

    A held cube may deform its projected contour slightly. Its independent
    current red component must still follow the wrist and beat the stationary
    source counterfactual. A good upright fit alone never proves attachment. Visible-in-gripper
    permits qualified continuation only and never claims stable attachment.
    """
    cameras=active_cameras();ee=np.asarray(own_state[arm+'_ee_pose']);source=np.asarray(source)
    held=source+ee[:3]-np.asarray(closed_ee[:3]);projector=lambda pts:cameras.wrist_project(pts,ee)
    envelope=np.concatenate([Rotation.from_euler('z',a).apply(BOX) for a in np.linspace(-np.pi/4,np.pi/4,9)])
    roi=polygon_mask(projector(envelope+held),rgb.shape[:2])|polygon_mask(projector(envelope+source),rgb.shape[:2])
    roi=cv2.dilate(roi.astype(np.uint8),np.ones((61,61),np.uint8))>0
    candidates,material=cube_material_candidates(rgb,roi)
    head_evidence=head_source_crosscheck(head,source,own_state)
    result=dict(status='unknown',reason='current_cube_material_unresolved',material_evidence=material,head_evidence=head_evidence,candidates=[])
    if closed_mask is None or not closed_mask.any():result['reason']='closed_reference_unavailable';return result
    old=np.column_stack(np.where(closed_mask)[::-1]);center=old.mean(0);area=len(old)
    expanded=cv2.dilate(closed_mask.astype(np.uint8),np.ones((13,13),np.uint8))>0
    static=cameras.wrist_project([source],ee)[0]-cameras.wrist_project([source],closed_ee)[0]
    for mask,record in candidates:
        if mask.sum()<120:continue
        points=np.column_stack(np.where(mask)[::-1]);current=points.mean(0)
        shift=float(np.linalg.norm(current-center));ratio=float(len(points)/area)
        coverage=float((mask&expanded).sum()/max(1,mask.sum()));static_error=float(np.linalg.norm(current-center-static));static_shift=float(np.linalg.norm(static))
        limit=max(18.,.10*np.sqrt(area))
        follows=bool(shift<limit and .55<ratio<1.6 and coverage>.60 and static_shift>18 and static_error>shift+8)
        row=dict(material=record,center_shift_px=shift,center_shift_limit_px=float(limit),area_ratio=ratio,material_coverage=coverage,stationary_error_px=static_error,stationary_shift_px=static_shift,follows=follows)
        result['candidates'].append(row)
        if follows and head_evidence.get('capture_valid',True) and not head_evidence.get('source_still_occupied',False):
            result.update(status='held',reason='independent_current_material_follows_wrist_not_stationary_source',mask=mask,motion=row);return result
        # Only a clear failed follow plus a current support-height cube fit
        # and positive head material can authorize source-directed recovery.
        at_source=fit_cube(mask,projector,center_z=float(source[2]),initial=source)
        row['source_height_fit']={k:v for k,v in at_source.items() if k!='mask'} if at_source else None
        source_positive=bool(head_evidence.get('capture_valid',True) and head_evidence.get('visible_pixels',0)>=120 and head_evidence.get('red_support_fraction',0)>.35)
        not_held=ratio<.50 or (static_error+20<shift and coverage<.60)
        if at_source and at_source['ok'] and np.linalg.norm(np.asarray(at_source['xy'])-source[:2])<.018 and source_positive and not_held:
            result.update(status='at_source',reason='current_source_height_shape_and_head_red_contradict_wrist_follow',source_fit={k:v for k,v in at_source.items() if k!='mask'},mask=mask,motion=row)
    if result['status']=='unknown' and candidates:result['reason']='visible_cube_does_not_yet_establish_held_or_source_support'
    if result['status']=='unknown':
        from .visible_grasp import visible_in_gripper
        visible=visible_in_gripper(rgb,ee,closed_mask,head_evidence)
        result['visible_evidence']={k:v for k,v in visible.items() if k!='mask'}
        if visible['status']=='visible_in_gripper':result.update(status=visible['status'],reason=visible['reason'],mask=visible['mask'])
    return result
