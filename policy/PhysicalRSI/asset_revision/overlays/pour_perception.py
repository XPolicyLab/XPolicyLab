"""Registration against shapes observed earlier in this episode, never asset CAD."""
import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from physicalrsi_observed_geometry import Scene, Camera, GeometryUnavailable
from physicalrsi_observed_geometry.supported_frustum import fit_supported_frustum
from physicalrsi_observed_geometry.radial_profile import observe_profile, profile_surface, render_profile


def observed_bottle_fit(mask, scene):
    """Infer an upright rotational bottle from RGB, without a depth-shaped body."""
    mask=scene.mask(mask)
    fit=observe_profile(scene.camera,mask,scene.depth_calibration_report['observed_plane'])
    transform=np.eye(4);transform[:3,3]=fit['base']
    points=profile_surface(fit['profile']).reshape(-1,3)
    rendered=render_profile(fit['profile'],transform[:3,:3],transform[:3,3],scene.camera,mask.shape).astype(bool)
    def boundary(value):
        value=value.astype('uint8')
        return value.astype(bool) & ~cv2.erode(value,np.ones((3,3),'uint8')).astype(bool)
    observed_edge=boundary(mask);projected_edge=boundary(rendered)
    if not observed_edge.any() or not projected_edge.any():
        raise GeometryUnavailable('Bottle contour is unresolved')
    distances=np.r_[cv2.distanceTransform((~observed_edge).astype('uint8'),cv2.DIST_L2,5)[projected_edge],
                    cv2.distanceTransform((~projected_edge).astype('uint8'),cv2.DIST_L2,5)[observed_edge]]
    return {'key':'bottle','pose':transform,'points_local':points,
            'error_px':float(np.sqrt(np.mean(distances**2))),
            'iou':float((mask&rendered).sum()/max(1,(mask|rendered).sum())),
            'source':fit['source'],'shape_hypothesis':'upright_rotational_profile',
            'hidden_surfaces_reconstructed':True,'hidden_geometry_verified':False,
            'evidence':fit['evidence'],'fitted_min_z':float(transform[2,3])}


def matrix(p):
    t=np.eye(4);t[:3,3]=p[:3];t[:3,:3]=Rot.from_quat(np.asarray(p)[[4,5,6,3]]).as_matrix();return t


def pose(t):
    q=Rot.from_matrix(t[:3,:3]).as_quat();return np.r_[t[:3,3],q[[3,0,1,2]]]


def observed_cup_fit(mask, scene, support):
    """Fit current cup geometry under an explicit upright frustum hypothesis."""
    normal=np.asarray(support['normal'],float);offset=float(support['offset'])
    if abs(normal[2])<.9:raise GeometryUnavailable('Cup support orientation is unresolved')
    plane=np.r_[-normal[:2]/normal[2],-offset/normal[2]]
    fit=fit_supported_frustum([(scene.camera,scene.mask(mask))],plane)
    transform=np.eye(4);transform[:3,:3]=fit['rotation']
    transform[:3,3]=fit['base']+fit['rotation'][:,2]*fit['height']/2
    return {'key':'cup','pose':transform,
            'points_local':(fit['points']-transform[:3,3])@transform[:3,:3],
            'error_px':fit['rms_px'],'source':fit['source'],
            'shape_hypothesis':fit['shape_hypothesis'],'hidden_surfaces_reconstructed':True,
            'measured_height':fit['height'],'measured_top_radius':fit['top_radius'],
            'measured_bottom_radius':fit['bottom_radius'],
            'fitted_min_z':float(fit['points'][:,2].min())}


class PixelOcclusion:
    def __init__(self,mask,camera):self.mask=np.asarray(mask,bool);self.camera=camera
    def hidden(self,world):
        uv=np.rint(self.camera.project(world)[0]).astype(int)
        h,w=self.mask.shape;valid=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
        result=np.zeros(len(world),bool);result[valid]=self.mask[uv[valid,1],uv[valid,0]];return result


def registration(mask,key,prior=None,visibility=None,extra_views=(),*,scene,template=None,inferred_template=False):
    mask=scene.mask(mask)
    if template is None:
        estimate=scene.geometry(mask)
        transform=matrix(estimate['pose']);points=estimate['points_local']
    else:
        if prior is None:raise GeometryUnavailable('Episode template needs its previous pose')
        points=np.asarray(template,float);transform=np.asarray(prior,float).copy()
        observed=scene.surface(mask);tree=cKDTree(observed)
        if visibility is None:
            transform[:3,3] += observed.mean(0)-(points@transform[:3,:3].T+transform[:3,3]).mean(0)
        sample=points[::max(1,len(points)//1200)]
        # Partial-cloud ICP. Template remains local to the episode and environment.
        for _ in range(15):
            world=sample@transform[:3,:3].T+transform[:3,3]
            keep=np.ones(len(world),bool) if visibility is None else ~visibility.hidden(world)
            distance,index=tree.query(world)
            keep &= distance<=np.quantile(distance,.7)
            if keep.sum()<8:raise GeometryUnavailable('Too few visible template correspondences')
            a=world[keep];b=observed[index[keep]];ac=a.mean(0);bc=b.mean(0)
            u,_,vt=np.linalg.svd((a-ac).T@(b-bc));rot=vt.T@u.T
            if np.linalg.det(rot)<0:vt[-1]*=-1;rot=vt.T@u.T
            delta=np.eye(4);delta[:3,:3]=rot;delta[:3,3]=bc-rot@ac
            transform=delta@transform
            if np.linalg.norm(delta[:3,3])<1e-5 and Rot.from_matrix(rot).magnitude()<1e-4:break
    world=points@transform[:3,:3].T+transform[:3,3]
    def quality(camera,mask,occlusion=None):
        h,w=mask.shape
        try:uv,depth=camera.project(world)
        except GeometryUnavailable:return float(np.hypot(h,w)),0.
        valid=(depth>0)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
        if occlusion is not None:valid &= ~occlusion.hidden(world)
        projected=uv[valid]
        # An unavailable validation view rejects this binding; it must not abort
        # the whole episode or silently count as supporting the fitted pose.
        if len(projected)<8:return float(np.hypot(h,w)),0.
        yy,xx=np.nonzero(mask);pixels=np.c_[xx,yy]
        error=float(np.sqrt(np.mean(cKDTree(projected).query(pixels)[0]**2)))
        im=np.zeros(mask.shape,'uint8');ij=np.rint(projected).astype(int);ij[:,0]=np.clip(ij[:,0],0,w-1);ij[:,1]=np.clip(ij[:,1],0,h-1);im[ij[:,1],ij[:,0]]=1
        im=cv2.morphologyEx(im,cv2.MORPH_CLOSE,np.ones((5,5),'uint8'))>0
        return error,float((im&mask).sum()/max(1,(im|mask).sum()))
    error,iou=quality(scene.camera,mask,visibility)
    other=[quality(v['camera'],v['mask'],PixelOcclusion(v['occlusion'],v['camera']))[0] for v in extra_views]
    return {'key':key,'pose':transform,'points_local':points,'error_px':error,'iou':iou,
            'extra_views':len(other),'view_errors_px':other,'source':'episode_rgb_metric_depth',
            'hidden_surfaces_reconstructed':bool(template is not None and inferred_template),'fitted_min_z':float(world[:,2].min())}


def observed_opening_fit(observation, head_mask, support, segment, *, prompts=None):
    """Associate current opening silhouettes, scoped to the selected container."""
    from physicalrsi_observed_geometry.multiview_circle import select_circle
    groups=[]
    for name in observation['vision']:
        camera=Camera.from_observation(observation,name)
        results=segment(camera.image,prompts if prompts is not None else [('vase opening',.2),('mouth of vase',.2)],timeout_s=600)
        masks=[np.asarray(m,bool).squeeze() for result in results for m in result.masks]
        if name=='cam_head':
            target=cv2.dilate(np.asarray(head_mask,'uint8'),np.ones((3,3),'uint8')).astype(bool)
            masks=[m for m in masks if m.shape==target.shape and (m&target).sum()/max(1,m.sum())>.8]
        groups.append((name,camera,masks))
    head=next((g for g in groups if g[0]=='cam_head'),None)
    if head is None or not head[2]:raise GeometryUnavailable('No opening scoped to the selected container')
    # Every proposal correspondence includes the selected object's head mask.
    fits=[]
    for name,camera,masks in groups:
        if name=='cam_head' or not masks:continue
        try:fits.append(select_circle([(head[1],head[2]),(camera,masks)],support['normal']))
        except GeometryUnavailable:continue
    if not fits:raise GeometryUnavailable('No calibrated second view resolves the selected opening')
    fits.sort(key=lambda f:f['rms_px']);best=fits[0]
    if any(np.linalg.norm(f['center']-best['center'])>.2*min(f['radius'],best['radius']) for f in fits[1:]):
        raise GeometryUnavailable('Opening estimates disagree between wrist cameras')
    return best


def observed_liquid_cup_opening(observation, head_mask, support_plane, segment):
    """Measure a cup opening from calibrated views, assuming an upright cup."""
    plane=np.asarray(support_plane,float)
    if plane.shape!=(3,) or not np.isfinite(plane).all():
        raise GeometryUnavailable('Observed cup support plane required')
    normal=np.r_[-plane[:2],1.];normal/=np.linalg.norm(normal)
    return observed_opening_fit(observation,head_mask,{'normal':normal},segment,
        prompts=[('cup opening',.2),('inside of cup',.2)])


def bind_liquid_opening(geometry,opening):
    """Use the measured mouth in the unchanged liquid planner's local frame."""
    if opening is None:raise GeometryUnavailable('Observed cup opening is required')
    transform=matrix(geometry['cup_pose'])
    center=(np.asarray(opening['center'])-transform[:3,3])@transform[:3,:3]
    radius=float(opening['radius']);vertices=np.asarray(geometry['cup_vertices'],float).copy()
    if not np.isfinite(center).all() or not np.isfinite(radius) or radius<=0 or center[2]<=vertices[:,2].min():
        raise GeometryUnavailable('Observed cup opening contradicts the visible body')
    top=vertices[:,2]>vertices[:,2].min()
    vertices[top,2]=center[2]
    geometry.update(cup_vertices=vertices.tolist(),cup_center_local=center.tolist(),
                    vessel_radius=radius,inner_rim_radius=radius,
                    cup_outer_radius=max(radius,float(geometry['cup_outer_radius'])),
                    observed_opening=opening)


def opening_in_frame(opening, frame):
    return dict(opening,center=(np.asarray(opening['center'])-frame[:3,3])@frame[:3,:3],
                normal=np.asarray(opening['normal'])@frame[:3,:3])
