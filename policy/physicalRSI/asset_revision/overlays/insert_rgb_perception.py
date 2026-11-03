"""Tube and rack geometry from current RGB, metric depth and camera calibration."""
from dataclasses import dataclass, asdict
import os
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from physicalrsi_observed_geometry import Camera, Scene, GeometryUnavailable
from physicalrsi_observed_geometry.partial_rectangle import fit_partial_rectangle
from physicalrsi_observed_geometry.support_collision import support_patches
from physicalrsi_observed_geometry.tubes import fit_resting_tube
from physicalrsi_observed_geometry.rgb_support import initial_plane, refine_plane
from .contract import InsertTubesEnvelopeError
from .geometry import _rotation, _quaternion


def _components(mask, minimum=8):
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask.astype('uint8'))
    return [(labels == i, centers[i]) for i in range(1,count) if stats[i,4] >= minimum]


def _colors(scene):
    hsv=cv2.cvtColor(scene.camera.image,cv2.COLOR_RGB2HSV)
    h,s,v=np.moveaxis(hsv,-1,0)
    return (s<60)&(v>150), (h>=3)&(h<=30)&(s>70)&(v>100), (h>=80)&(h<=115)&(s>45)&(v>100)


def _tubes(scene, excluded=None, plane=None, instance_mask=None, require_complete=False, contact_pixel=None):
    white,orange,_=_colors(scene)
    if excluded is not None:white &= ~excluded
    value=cv2.cvtColor(scene.camera.image,cv2.COLOR_RGB2HSV)[:,:,2]
    results=[]
    for mask,center in _components(white,20):
        if instance_mask is not None and np.count_nonzero(mask&instance_mask)<.8*np.count_nonzero(mask):continue
        yy,xx=np.nonzero(mask);pixels=np.c_[xx,yy].astype(float)
        eigen,axes=np.linalg.eigh(np.cov(pixels.T))
        if eigen[-1] < 4*max(eigen[0],1e-6):continue
        if contact_pixel is not None:
            # Match the visible body to the robot's current grasp contact before
            # choosing an axis; a complete resting tube is not a held tube.
            distance=float(np.linalg.norm(pixels-np.asarray(contact_pixel),axis=1).min())
            if distance>max(2.,2*np.sqrt(max(eigen[0],0.))):continue
        image_axis=axes[:,-1];projection=(pixels-center)@image_axis
        ends=[center+image_axis*t for t in np.quantile(projection,[.02,.98])]
        span=np.linalg.norm(ends[1]-ends[0])
        if span<=0 or mask.sum()/span<3.:continue  # Require a resolved body width.
        # Bright tube caps and brown support can share hue; use the current
        # body's brightness to keep the cap from merging into the whole table.
        caps=_components(orange & (value>=.9*float(np.median(value[mask]))))
        candidates=[(np.linalg.norm(c-end),index,cap,c) for cap,c in caps for index,end in enumerate(ends)]
        if not candidates:continue
        distance,index,cap,cap_center=min(candidates,key=lambda v:v[0])
        if distance>max(5.,.25*span):continue
        if not .02<=float(cap.sum())/float(mask.sum())<=.7:continue
        if require_complete:
            silhouette=mask|cap
            if (silhouette[0].any() or silhouette[-1].any()
                    or silhouette[:,0].any() or silhouette[:,-1].any()):continue
        if plane is not None:
            try:estimated=fit_resting_tube(scene.camera,mask|cap,cap_center,plane)
            except GeometryUnavailable:continue
            results.append((estimated['pose'],estimated['length']))
            continue
        cloud=scene.surface(mask);middle=cloud.mean(0)
        _,_,axes=np.linalg.svd(cloud-middle,full_matrices=False)
        axis=axes[0];cap_point=scene.surface(cap).mean(0)
        if (cap_point-middle)@axis<0:axis=-axis
        longitudinal=(cloud-middle)@axis
        root=middle+axis*np.quantile(longitudinal,.02)
        length=float((cap_point-root)@axis)
        if not np.isfinite(length) or length<=0:continue
        z=np.array([0.,0.,1.]);z-=axis*(z@axis)
        if np.linalg.norm(z)<.1:
            z=scene.camera.transform[:3,0].copy();z-=axis*(z@axis)
        z/=np.linalg.norm(z);x=np.cross(axis,z)
        results.append((root.tolist()+_quaternion(np.column_stack([x,axis,z])),length))
    return sorted(results,key=lambda item:tuple(item[0][:2]))


def _rack_openings(mask, outer_mask=None):
    """Separate observed large openings from smaller auxiliary perforations."""
    components=sorted(_components(mask),key=lambda item:int(item[0].sum()))
    if outer_mask is not None:
        outer=np.asarray(outer_mask,bool)
        boundary=outer&~cv2.erode(outer.astype('uint8'),np.ones((3,3),'uint8')).astype(bool)
        # A notch connected to the outer contour is not a closed rack aperture.
        components=[item for item in components if not np.any(item[0]&boundary)]
    if len(components)<2:return components
    areas=np.asarray([item[0].sum() for item in components],float)
    ratios=areas[1:]/areas[:-1]
    split=int(np.argmax(ratios))
    # A clear image-space size gap is required; no metric hole size is supplied.
    if ratios[split]>=1.5:return components[split+1:]
    return components


def _rack(scene):
    _,_,blue=_colors(scene)
    parts=_components(blue,20)
    if not parts:raise GeometryUnavailable('Rack is not visible')
    mask,_=max(parts,key=lambda item:item[0].sum())
    yy,xx=np.nonzero(mask)
    hull=cv2.convexHull(np.c_[xx,yy].astype('int32'))
    filled=np.zeros_like(mask,dtype='uint8');cv2.fillConvexPoly(filled,hull,1)
    # Hole positions are observed; there is no stored rack grid or slot spacing.
    holes=_rack_openings(filled.astype(bool)&~mask,filled)
    if not holes:raise GeometryUnavailable('No closed rack openings observed')
    # Fit unknown pose and size directly from calibrated RGB outlines.
    views=[(scene.camera,hull.reshape(-1,2))]
    for camera in getattr(scene,'_other_cameras',[]):
        hsv=cv2.cvtColor(camera.image,cv2.COLOR_RGB2HSV)
        h,s,v=np.moveaxis(hsv,-1,0)
        proposals=_components((h>=80)&(h<=115)&(s>45)&(v>100),20)
        if not proposals:continue
        other,_=max(proposals,key=lambda item:item[0].sum())
        oy,ox=np.nonzero(other)
        outline=cv2.convexHull(np.c_[ox,oy].astype('float32')).reshape(-1,2)
        views.append((camera,outline))
    center,rotation,_,_=fit_partial_rectangle(views)
    normal=rotation[:,2];quat=_quaternion(rotation)
    slots=[]
    for hole,pixel in holes:
        # Intersect the observed opening center ray with the reconstructed plane.
        origin=scene.camera.transform[:3,3]
        ray=scene.camera.unproject(np.asarray(pixel)[None],np.ones(1))[0]-origin
        denominator=float(ray@normal)
        if abs(denominator)<1e-6:raise GeometryUnavailable('Rack opening ray is parallel to its plane')
        distance=float((center-origin)@normal)/denominator
        if distance<=0:raise GeometryUnavailable('Rack opening lies behind camera')
        point=origin+ray*distance
        slots.append(point.tolist()+quat)
    slots.sort(key=lambda pose:tuple(pose[:2]))
    return tuple(slots),center.tolist()+quat,filled.astype(bool)


def _initial(observation):
    camera=Camera.from_observation(observation)
    others=[Camera.from_observation(observation,name) for name in observation['vision'] if name!='cam_head']
    scene=SimpleNamespace(camera=camera,_other_cameras=others)
    hsv=cv2.cvtColor(scene.camera.image,cv2.COLOR_RGB2HSV)
    # Appearance proposes table pixels; calibrated multi-view geometry sets scale.
    support_mask=(hsv[:,:,0]<35)&(hsv[:,:,1]>35)&(hsv[:,:,2]>40)
    cameras=[camera,*others]
    masks=[support_mask]+[np.ones(c.image.shape[:2],bool) for c in others]
    start,support_mask=initial_plane(cameras,masks,unique_head_pixels=True,return_support_mask=True)
    masks[0]=support_mask
    # Calibrated stereo already supplies a metric plane from six independent
    # source pixels. One overlapping reference can refine it when the other
    # wrist cannot see the same patch; a third view is not necessary for scale.
    plane=refine_plane(cameras,masks,start,minimum_reference_views=1)
    normal=np.r_[-plane[:2],1.];normal/=np.linalg.norm(normal)
    if abs(normal[2])<.9:raise GeometryUnavailable('Support is not sufficiently resolved')
    support=support_patches(scene.camera,support_mask,plane)
    if not support:raise GeometryUnavailable('No observed support collision patches')
    slots,rack,excluded=_rack(scene)
    from PourBallsRGB.sam.sam3_http import segment_many_http
    instances=segment_many_http(camera.image,[('test tube',.3)])[0]
    if not len(instances.masks):
        # A lower-confidence semantic proposal must still pass the observed
        # body/cap and support-plane geometry checks below.
        instances=segment_many_http(scene.camera.image,[('test tube',.2)])[0]
    if not len(instances.masks):raise GeometryUnavailable('No current RGB test-tube instances observed')
    instance_mask=np.any(np.asarray(instances.masks,bool),axis=0)
    instance_mask=cv2.dilate(instance_mask.astype('uint8'),np.ones((5,5),'uint8')).astype(bool)
    tubes=_tubes(scene,excluded,plane,instance_mask)
    if len(tubes)<3:
        # The instruction requests three tubes. A partial semantic proposal must
        # not prevent a lower-confidence mask from passing the same RGB/shape fit.
        extra=segment_many_http(scene.camera.image,[('test tube',.2)])[0]
        if len(extra.masks):
            more=np.any(np.asarray(extra.masks,bool),axis=0)
            more=cv2.dilate(more.astype('uint8'),np.ones((5,5),'uint8')).astype(bool)
            tubes=_tubes(scene,excluded,plane,instance_mask|more)
    if not 1<=len(tubes)<=3:raise GeometryUnavailable('One to three distinct tubes must be observed')
    if len(slots)<len(tubes):raise GeometryUnavailable('Not enough observed rack openings for the tubes')
    return _Estimate(tuple(p for p,_ in tubes),slots,rack,tuple(length for _,length in tubes),support)


def estimate_initial_geometry(observation):
    try:
        result=_initial(observation)
        return result.tube_poses,result.slot_poses,result.rack_pose
    except GeometryUnavailable as exc:raise InsertTubesEnvelopeError(str(exc)) from exc


def estimate_held_axis(observation,side,estimated_yaw,robot_geometry=None):
    if side not in {'left','right'} or robot_geometry is None:
        raise InsertTubesEnvelopeError('Known robot geometry and arm are required')
    try:
        scene=Scene.from_observation(observation,f'cam_{side}_wrist')
        from .geometry import _ee_pose_to_semantic
        from .planning import _PICK_OBJECT_IN_GRIPPER
        semantic_pose=_ee_pose_to_semantic(robot_geometry,observation['state'][f'{side}_ee_pose'])
        contact=np.asarray(semantic_pose[:3])+_rotation(semantic_pose[3:])@_PICK_OBJECT_IN_GRIPPER
        pixel=scene.camera.project([contact])[0][0]
        tubes=_tubes(scene,require_complete=True,contact_pixel=pixel)
        if len(tubes)!=1:raise GeometryUnavailable('Held tube is ambiguous or occluded')
        axis=_rotation(tubes[0][0][3:])[:,1]
        ee=_rotation(observation['state'][f'{side}_ee_pose'][3:])
        semantic=ee@np.asarray(robot_geometry.delta_matrix).reshape(3,3)
        return (semantic.T@axis).tolist()
    except GeometryUnavailable as exc:raise InsertTubesEnvelopeError(str(exc)) from exc


def estimate_aligned_cap_xy(observation,expected_cap):
    try:
        return cap_xy_from_rgb(Camera.from_observation(observation),expected_cap)
    except GeometryUnavailable as exc:raise InsertTubesEnvelopeError(str(exc)) from exc


@dataclass
class _Estimate:
    tube_poses: tuple
    slot_poses: tuple
    rack_pose: list
    tube_lengths: tuple
    support: list


class RgbInsertTubesPerception:
    def __init__(self,robot_geometry=None):
        self.robot_geometry=robot_geometry;self.reset()

    def reset(self):
        self._slots={};self._latest={};self._traced=set();self._trace_paths={}
        self._failed_observations={};self._errors={}
        self._trace_sequence=getattr(self,'_trace_sequence',0)

    def support_geometry(self,env_idx):return self._slots[env_idx].support

    def is_ready(self,env_idx):return env_idx in self._slots

    def observe(self,env_idx,observation):
        self._latest[env_idx]=observation
        trace=os.environ.get('PHYSICALRSI_INSERT_TRACE')
        if trace and env_idx not in self._traced:
            directory=Path(trace);directory.mkdir(parents=True,exist_ok=True)
            arrays={f'vision/{name}/{key}':np.asarray(camera[key])
                for name,camera in observation['vision'].items()
                for key in ('color','intrinsic_matrix','extrinsic_matrix')}
            arrays.update({f'state/{key}':np.asarray(value) for key,value in observation['state'].items()})
            capture=directory/f'capture-{self._trace_sequence:06d}-env-{env_idx}.npz'
            np.savez_compressed(capture,**arrays)
            self._trace_paths[env_idx]=capture
            self._trace_sequence+=1
            self._traced.add(env_idx)
        if env_idx not in self._slots:
            digest=hashlib.sha256()
            for name,camera in sorted(observation['vision'].items()):
                digest.update(name.encode())
                digest.update(np.ascontiguousarray(camera['color']).tobytes())
                for key in ('intrinsic_matrix','extrinsic_matrix'):
                    digest.update(np.round(np.asarray(camera[key],float),6).tobytes())
            fingerprint=digest.digest()
            if self._failed_observations.get(env_idx)==fingerprint:return
            try:self._slots[env_idx]=_initial(observation)
            except GeometryUnavailable as exc:
                # A missing geometric observation is local to this environment.
                # Preserve the reason; the planner reports this trial infeasible.
                # New RGB/calibration permits another estimate, identical input
                # cannot justify repeatedly running the same expensive fit.
                self._failed_observations[env_idx]=fingerprint
                self._errors[env_idx]=str(exc)
                print(f'[insert_tubes rgb] env={env_idx} geometry_unavailable={exc}',flush=True)
                return
            self._errors.pop(env_idx,None);self._failed_observations.pop(env_idx,None)
            if env_idx in self._trace_paths:
                self._trace_paths[env_idx].with_suffix('.geometry.json').write_text(json.dumps(asdict(self._slots[env_idx]), default=lambda value: np.asarray(value).tolist(), indent=2)+'\n')

    def initial_estimate(self,env_idx):
        if env_idx not in self._slots:
            raise InsertTubesEnvelopeError(self._errors.get(env_idx,'Initial RGB geometry is unavailable'))
        item=self._slots[env_idx]
        return {'env_idx':env_idx,'tube_poses':list(item.tube_poses),'slot_poses':list(item.slot_poses),
                'rack_pose':item.rack_pose,'tube_lengths':list(item.tube_lengths),'provenance':'rgb'}

    def held_axis(self,env_idx,side,estimated_yaw):
        try:axis=estimate_held_axis(self._latest[env_idx],side,estimated_yaw,self.robot_geometry)
        except InsertTubesEnvelopeError:axis=None
        return {'env_idx':env_idx,'side':side,'axis_in_gripper':axis,'visible':axis is not None,'provenance':'rgb'}

    def held_pose(self,env_idx,side,tube_length):
        from PourBallsRGB.sam.sam3_http import segment_many_http
        try:
            fit=stereo_held_pose(self._latest[env_idx],side,tube_length,self.robot_geometry,segment_many_http)
            return dict(fit,env_idx=env_idx,side=side,visible=True,provenance='rgb',retry_at_visible_stage=True)
        except GeometryUnavailable:
            return dict(env_idx=env_idx,side=side,visible=False,provenance='rgb',retry_at_visible_stage=True)

    def aligned_cap(self,env_idx,expected_cap):
        try:xy=estimate_aligned_cap_xy(self._latest[env_idx],expected_cap)
        except InsertTubesEnvelopeError:xy=None
        return {'env_idx':env_idx,'cap_xy':xy,'visible':xy is not None,'provenance':'rgb'}


def cap_xy_from_rgb(camera, expected_cap):
    expected=np.asarray(expected_cap,float)
    if expected.shape!=(3,) or not np.isfinite(expected).all():raise GeometryUnavailable('Finite predicted cap required')
    pixel,depth=camera.project([expected])
    if depth[0]<=0:raise GeometryUnavailable('Predicted cap is behind camera')
    hsv=cv2.cvtColor(camera.image,cv2.COLOR_RGB2HSV);h,s,v=np.moveaxis(hsv,-1,0)
    white=(s<60)&(v>150)
    if np.count_nonzero(white)<20:raise GeometryUnavailable('No observed bright tube appearance')
    # Shadows and the dark robot occupy much of the low-saturation population.
    # Use its illuminated upper quartile so the cap does not merge with wood.
    bright=.9*float(np.quantile(v[white], .75))
    orange=(h>=3)&(h<=30)&(s>70)&(v>=bright)
    count,labels,stats,centers=cv2.connectedComponentsWithStats(orange.astype('uint8'))
    candidates=[]
    for i in range(1,count):
        x,y,width,height,area=stats[i]
        if area<4 or area>.01*orange.size or x<=0 or y<=0 or x+width>=camera.width or y+height>=camera.height:continue
        if max(width,height)>4*min(width,height):continue
        distance=float(np.linalg.norm(centers[i]-pixel[0]))
        # Association range follows the observed cap size, not a fixture offset.
        if distance>max(5.,2*np.sqrt(area)) or distance>.05*camera.width:continue
        candidates.append((distance,centers[i],int(area)))
    if not candidates:raise GeometryUnavailable('No visible cap near the predicted held position')
    _,center,area=min(candidates,key=lambda item:item[0])
    # XY servo uses the planned cap height from measured tube/slot geometry.
    # Do not infer depth from a tiny foreground mask or substitute a table height.
    origin=camera.transform[:3,3];ray=camera.unproject(center[None],np.ones(1))[0]-origin
    if abs(ray[2])<1e-6:raise GeometryUnavailable('Cap ray is parallel to its predicted height')
    distance=(expected[2]-origin[2])/ray[2]
    if distance<=0:raise GeometryUnavailable('Cap height lies behind camera')
    return (origin+ray*distance)[:2].tolist()


def stereo_held_pose(observation, side, tube_length, robot_geometry, segment):
    """Recover the held centerline and cap from two current RGB silhouettes."""
    from .geometry import _ee_pose_to_semantic
    from .planning import _PICK_OBJECT_IN_GRIPPER
    from physicalrsi_observed_geometry.centerline import centerline_cap
    if not np.isfinite(tube_length) or tube_length<=0:raise GeometryUnavailable('Observed tube length required')
    pose=_ee_pose_to_semantic(robot_geometry,observation['state'][f'{side}_ee_pose'])
    rotation=_rotation(pose[3:]);origin=np.asarray(pose[:3]);contact=origin+rotation@_PICK_OBJECT_IN_GRIPPER
    cameras=[];groups=[]
    for name in ['cam_head',f'cam_{side}_wrist']:
        camera=Camera.from_observation(observation,name);cameras.append(camera);pixel=camera.project([contact])[0][0]
        prompts=[('test tube',.2)] if name=='cam_head' else [('white tube',.15),('cylinder',.15),('white cylindrical object',.15)]
        proposals=segment(camera.image,prompts,timeout_s=600);candidates=[]
        for group in proposals:
            for raw in group.masks:
                mask=np.asarray(raw,bool).squeeze()
                if mask.shape!=(camera.height,camera.width) or mask.sum()<20:continue
                yy,xx=np.nonzero(mask);pixels=np.c_[xx,yy];center=pixels.mean(0);values,axes=np.linalg.eigh(np.cov(pixels.T))
                if values[-1]<4*max(values[0],1e-6):continue
                if np.linalg.norm(pixels-pixel,axis=1).min()>max(2.,2*np.sqrt(max(values[0],0))):continue
                candidates.append((center,axes[:,-1],mask))
        groups.append(candidates)
    head,wrist=cameras;hsv=cv2.cvtColor(head.image,cv2.COLOR_RGB2HSV)
    orange=(hsv[:,:,0]>=3)&(hsv[:,:,0]<=30)&(hsv[:,:,1]>70)&(hsv[:,:,2]>150)
    fits=[]
    for center,direction,mask in groups[0]:
        yy,xx=np.nonzero(mask&orange)
        if len(xx)<8:continue
        cap_pixel=np.array([xx.mean(),yy.mean()])
        for wc,wd,_ in groups[1]:
            try:fits.append(centerline_cap([(head,center,direction),(wrist,wc,wd)],head,cap_pixel,center))
            except GeometryUnavailable:continue
    if not fits:raise GeometryUnavailable('Held cap and centerline are not jointly visible')
    best=min(fits,key=lambda f:f['cap_error_px'])
    if any(np.linalg.norm(f['cap']-best['cap'])>.05*tube_length or f['axis_to_cap']@best['axis_to_cap']<np.cos(.05) for f in fits):
        raise GeometryUnavailable('Held tube correspondence is ambiguous')
    root=best['cap']-tube_length*best['axis_to_cap']
    return {'axis_in_gripper':(rotation.T@best['axis_to_cap']).tolist(),
            'root_in_gripper':(rotation.T@(root-origin)).tolist(),
            'cap_world':best['cap'].tolist(),'cap_error_px':best['cap_error_px']}
