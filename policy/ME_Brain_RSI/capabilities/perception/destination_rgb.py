"""RGB support-height compatibility permits release, never contact truth."""
import numpy as np
import cv2
from scipy.optimize import minimize
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation
from .button_material import observed_robot_occlusion
from .robot_geometry import gripper_mask
from .calibration import active_cameras
from .perception import cube_material_candidates,fit_cube,head_source_crosscheck,MAT_TOP,BOX,polygon_mask,edges


def public_destination_region(cube_height=.035):
    """RGB support hypothesis against unchanged public30mm root-distance rule.

    Public cube mesh is centered at its root and mat mesh spans +/-2.5mm.
    Thus a supported cube's root is cube_half_height + mat_half_thickness
    above the mat root. This is geometry conditional on RGB support, not truth.
    """
    root_height_difference=.0025+float(cube_height)/2
    return dict(official_root_distance_m=.03,public_mat_half_thickness_m=.0025,
                supported_root_height_difference_m=root_height_difference,
                supported_xy_radius_m=float(np.sqrt(max(0.,.03**2-root_height_difference**2))))


def fit_visible_destination(mask,projector,occlusion,center_z,initial):
    """Fit only independently visible cube silhouette; never fill the mask.

    Existing2.8px/IoU.65 criteria apply after current robot pixels are censored.
    At least half the public silhouette/boundary must remain observable so a
    tiny visible fragment cannot explain an otherwise arbitrarily hidden cube.
    """
    if not occlusion.any():return None
    observed=mask&~occlusion
    if observed.sum()<120:return None
    boundary=edges(observed)
    distance=cv2.distanceTransform((~boundary).astype(np.uint8),cv2.DIST_L2,5)
    # One image pixel around an observed robot edge covers raster boundary
    # sampling. It censors comparisons only and does not add object material.
    border_occlusion=cv2.dilate(occlusion.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    def evaluate(p):
        uv=projector(Rotation.from_euler('z',p[2]).apply(BOX)+[p[0],p[1],center_z])
        hull=cv2.convexHull(uv.astype(np.float32)).reshape(-1,2)
        samples=np.concatenate([(1-t)*hull+t*np.roll(hull,-1,axis=0) for t in np.linspace(0,1,20)])
        visible=map_coordinates(border_occlusion.astype(float),[samples[:,1],samples[:,0]],order=0,mode='constant',cval=1)<.5
        d=map_coordinates(distance,[samples[visible,1],samples[visible,0]],order=1,mode='constant',cval=50)
        prediction=polygon_mask(uv,mask.shape);visible_prediction=prediction&~occlusion
        iou=float(np.count_nonzero(visible_prediction&observed)/max(1,np.count_nonzero(visible_prediction|observed)))
        fraction=float(visible_prediction.sum()/max(1,prediction.sum()));boundary_fraction=float(visible.mean())
        squared=float(np.mean(np.minimum(d,20)**2)) if len(d) else 400.
        return squared+18*(1-iou),iou,float(np.sqrt(squared)),fraction,boundary_fraction
    xy=np.asarray(initial[:2]);bounds=[(xy[0]-.025,xy[0]+.025),(xy[1]-.025,xy[1]+.025),(-np.pi/4,np.pi/4)]
    fits=[minimize(lambda p:evaluate(p)[0],[*xy,a],method='Powell',bounds=bounds,options={'maxiter':25,'xtol':2e-4,'ftol':1e-4}) for a in (-.5,0.,.5)]
    best=min(fits,key=lambda f:f.fun);_,iou,rmse,fraction,boundary_fraction=evaluate(best.x)
    return dict(xy=best.x[:2].tolist(),yaw=float(best.x[2]),z=float(center_z),iou=iou,edge_rmse_px=rmse,pixels=int(observed.sum()),visible_area_fraction=fraction,visible_boundary_fraction=boundary_fraction,ok=bool(iou>.65 and rmse<2.8 and fraction>=.50 and boundary_fraction>=.50),method='current_RGB_robot_occlusion_censored_support_height_cube',mask=mask)


def destination_ready(rgb,head,own,arm,source,destination,cube_height):
    result=dict(status='unknown',reason='destination_current_material_unresolved',does_not_assert_attachment_or_contact=True)
    if np.mean(np.max(rgb,axis=2)>10)<=.01:return dict(result,reason='destination_current_wrist_missing')
    head_source=head_source_crosscheck(head,source,own);result['head_source']=head_source
    if not head_source.get('capture_valid') or not head_source.get('actionable') or head_source.get('source_still_occupied'):
        return dict(result,reason='destination_source_absence_unconfirmed')
    ee=own[arm+'_ee_pose'];cameras=active_cameras();center_z=MAT_TOP+cube_height/2
    projector=lambda pts:cameras.wrist_project(pts,ee)
    possible_robot=gripper_mask(ee,rgb.shape[:2],projector,wrist=True)
    occlusion,occlusion_evidence=observed_robot_occlusion(rgb,possible_robot)
    result['current_robot_occlusion']=occlusion_evidence
    candidates,material=cube_material_candidates(rgb,np.ones(rgb.shape[:2],bool));result['material_evidence']=material;result['candidates']=[]
    for mask,record in candidates:
        if record['pixels']<120:continue
        fit=fit_cube(mask,projector,center_z=center_z,initial=[*destination,center_z])
        if fit is None:continue
        complete={k:v for k,v in fit.items() if k!='mask'}
        if not fit['ok']:
            visible=fit_visible_destination(mask,projector,occlusion,center_z,[*destination,center_z])
            if visible is not None:fit=visible
        error=float(np.linalg.norm(np.asarray(fit['xy'])-destination))
        row=dict(material=record,complete_outline_fit=complete,support_height_fit={k:v for k,v in fit.items() if k!='mask'},target_xy_error_m=error,target_xy_limit_m=public_destination_region(cube_height)['supported_xy_radius_m'],public_target_region=public_destination_region(cube_height))
        result['candidates'].append(row)
        # A support-compatible cube inside the public target region can open
        # and return home; no extra18mm proxy or held assumption is required.
        if fit['ok'] and error<row['target_xy_limit_m']:
            return dict(result,status='at_destination_release_ready',reason='current_cube_support_height_shape_at_observed_destination',evidence=row,mask=mask,xy=fit['xy'],yaw=fit['yaw'],z=center_z,height=cube_height)
    return dict(result,reason='destination_support_shape_or_target_unresolved')
