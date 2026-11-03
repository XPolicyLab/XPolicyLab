"""Language-bound block tracking using current RGB and measured episode geometry."""
import re
import os
import json
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from physicalrsi_observed_geometry import Camera, GeometryUnavailable
from physicalrsi_observed_geometry.rgb_support import initial_plane, refine_plane
from physicalrsi_observed_geometry.support_observation import observe_support_geometry
from physicalrsi_observed_geometry.primitives import fit_supported_primitive
from physicalrsi_observed_geometry.boxes import fit_box_views, fit_supported_box_pose
from physicalrsi_observed_geometry.occlusion import verify_partial_box
from rdj_v1_stack_blocks_by_language.vendor.rdj_master.perception.stack_blocks import RgbBlockDetection, StackBlocksRgbError

HUES={'red':0.,'orange':15.,'yellow':30.,'green':60.,'cyan':90.,'blue':120.}


def parse(text):
    found=re.fullmatch(r'stack the blocks from bottom to top in the order of\s*(blue|green|red|yellow|orange|cyan), (blue|green|red|yellow|orange|cyan), and (blue|green|red|yellow|orange|cyan), then reset the robot arm\.',str(text))
    if not found or len(set(found.groups()))!=3:
        raise StackBlocksRgbError('Instruction color binding is ambiguous')
    return found.groups()


def _components(camera, color, floors=(120,175,220), *, allow_border=False, value_floor=75):
    hsv=cv2.cvtColor(camera.image,cv2.COLOR_RGB2HSV)
    delta=np.abs(hsv[:,:,0].astype(float)-HUES[color])
    hue_mask=np.minimum(delta,180-delta)<(8 if color in ('red','orange','yellow') else 13)
    for floor in floors:
        raw=(hue_mask&(hsv[:,:,1]>floor)&(hsv[:,:,2]>value_floor)).astype('uint8')
        count,labels,stats,_=cv2.connectedComponentsWithStats(raw)
        for index in sorted(range(1,count),key=lambda i:stats[i,cv2.CC_STAT_AREA],reverse=True)[:8]:
            x,y,width,height,area=stats[index]
            if area<12:continue
            if not allow_border and (x<=0 or y<=0 or x+width>=camera.width or y+height>=camera.height):continue
            yield labels==index,int(floor),int(area)


def _initialize(camera,color,plane,instance_mask=None):
    candidates=[]
    for mask,floor,area in _components(camera,color):
        if instance_mask is not None and np.count_nonzero(mask&instance_mask)<.8*area:continue
        try:result=fit_supported_primitive(camera,mask,plane,'cube')
        except GeometryUnavailable:continue
        # Absolute pixel error favors tiny reflections over the object.
        candidates.append((result['rms_px']/np.sqrt(area),result,mask,floor))
    if not candidates:raise GeometryUnavailable('No supported RGB cube observed: '+color)
    _,result,mask,floor=min(candidates,key=lambda row:row[0])
    quat=np.asarray(result['pose'][3:])
    return (np.asarray(result['pose'][:3]),Rotation.from_quat(quat[[1,2,3,0]]).as_matrix(),
            np.asarray(result['dims'])/2,mask,floor)


def _refine_resting_shape(camera,color,previous,shape):
    """Re-measure a nearby resting box from its current close RGB view."""
    choices=[];old_dimensions=2*np.asarray(shape['half_extents'])
    for value_floor in (75,125,175,200):
        for mask,floor,area in _components(camera,color,value_floor=value_floor):
            if area<400:continue
            try:result=fit_supported_primitive(camera,mask,shape['support_plane'],'cube')
            except GeometryUnavailable:continue
            center=np.asarray(result['pose'][:3]);dimensions=np.asarray(result['dims'])
            if np.linalg.norm(center-np.asarray(previous.center))>np.max(old_dimensions):continue
            if np.any(dimensions<old_dimensions*.5) or np.any(dimensions>old_dimensions*1.5):continue
            choices.append((result['rms_px']/np.sqrt(area),result,floor,value_floor))
    if not choices:raise GeometryUnavailable('No consistent close RGB shape refinement')
    _,result,floor,value_floor=min(choices,key=lambda row:row[0])
    quat=np.asarray(result['pose'][3:]);rotation=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
    return SimpleNamespace(center=np.asarray(result['pose'][:3]),rotation=rotation),{
        **shape,'half_extents':(np.asarray(result['dims'])/2).tolist(),
        'saturation_floor':floor,'value_floor':value_floor,'wrist_refined':True}


def _track(camera,color,previous,shape,*,supported=False,stationary=False,predicted_motion=False,occluder_provider=None):
    center=np.asarray(previous.center);rotation=np.asarray(previous.rotation)
    pixel,depth=camera.project([center])
    if depth[0]<=0:raise GeometryUnavailable('Previous object is behind the current camera')
    choices=[]
    # A resting bright face can enter shadow while held; test both the measured
    # bright-face threshold and the broad color mask against current geometry.
    for value_floor in dict.fromkeys((shape.get('value_floor',75),75)):
        for mask,floor,area in _components(camera,color,(shape['saturation_floor'],),allow_border=True,value_floor=value_floor):
            y,x=np.nonzero(mask);distance=np.linalg.norm(np.array([x.mean(),y.mean()])-pixel[0])
            choices.append((distance/np.sqrt(area),mask))
    if supported or stationary or predicted_motion:
        occluders=np.zeros(camera.image.shape[:2],bool)
        for other in HUES:
            if other==color:continue
            for other_mask,_,_ in _components(camera,other,(120,),allow_border=True):
                occluders|=other_mask
        for _,mask in sorted(choices,key=lambda row:row[0])[:3]:
            try:
                verify_partial_box(camera,center,rotation,2*np.asarray(shape['half_extents']),mask,occluders)
            except GeometryUnavailable:continue
            return center.copy(),rotation.copy(),mask
        if stationary and choices and occluder_provider is not None:
            robot_mask=np.asarray(occluder_provider(camera),bool)
            if robot_mask.shape!=occluders.shape:raise GeometryUnavailable('Robot occlusion mask has wrong image shape')
            for _,mask in sorted(choices,key=lambda row:row[0])[:3]:
                try:
                    verify_partial_box(camera,center,rotation,2*np.asarray(shape['half_extents']),mask,occluders|robot_mask)
                except GeometryUnavailable:continue
                return center.copy(),rotation.copy(),mask
            # A cropped color fragment under a robot cannot establish a new
            # stationary depth. Keep the last observation stale until visible.
            from itertools import product
            corners=np.asarray(list(product([-.5,.5],repeat=3)))*(2*np.asarray(shape['half_extents']))
            pixels,_=camera.project(corners@rotation.T+center)
            silhouette=np.zeros(robot_mask.shape,np.uint8)
            cv2.fillConvexPoly(silhouette,cv2.convexHull(np.rint(pixels).astype('int32')),1)
            if np.count_nonzero(robot_mask & silhouette.astype(bool))>.1*np.count_nonzero(silhouette):
                raise GeometryUnavailable('Stationary box depth is occluded by the current robot')
    for _,mask in sorted(choices,key=lambda row:row[0])[:3]:
        y,x=np.nonzero(mask)
        try:
            dimensions=2*np.asarray(shape['half_extents'])
            if supported:
                result=fit_supported_box_pose(camera,np.c_[x,y],shape['support_plane'],dimensions,center,rotation)
            else:
                result=fit_box_views([(camera,np.c_[x,y])],previous_rotation=rotation,
                    previous_center=center,measured_dimensions=dimensions,
                    max_rms_px=max(2.,.01*float(np.ptp(np.c_[x,y],axis=0).max())))
            plane=np.asarray(shape['support_plane'])
            normal=np.r_[-plane[:2],1.];length=np.linalg.norm(normal);normal/=length
            clearance=result['center']@normal-plane[2]/length-np.abs(normal@result['rotation'])@(dimensions/2)
            if clearance < -.1*float(np.min(dimensions)):
                raise GeometryUnavailable('Tracked box penetrates the episode-observed support plane')
        except GeometryUnavailable:continue
        return result['center'],result['rotation'],mask
    raise GeometryUnavailable('No consistent current RGB box tracking: '+color)


def _robot_occluder(camera,cache):
    # Cache only within this observation. The model supplies image occlusion,
    # never object position, dimensions, or freshness without visible RGB proof.
    if camera not in cache:
        from PourBallsRGB.sam.sam3_http import segment_many_http
        try:instances=segment_many_http(camera.image,[('robot arm',.3)])[0]
        except (RuntimeError,OSError,ValueError) as exc:
            raise GeometryUnavailable('Current robot occlusion could not be observed') from exc
        cache[camera]=(np.any(np.asarray(instances.masks,bool),axis=0) if len(instances.masks)
                       else np.zeros(camera.image.shape[:2],bool))
    return cache[camera]


def detect_language_blocks(policy, env_idx, obs):
    colors=parse(obs['instruction'])
    context=policy._rdj_stack_blocks_rgb_context.get(env_idx)
    if not hasattr(policy,'rgb_observed_shapes'):policy.rgb_observed_shapes={}
    if context is None:policy.rgb_observed_shapes[env_idx]={}
    if not hasattr(policy,'rgb_observed_support'):policy.rgb_observed_support={}
    if context is None:policy.rgb_observed_support.pop(env_idx,None)
    history=policy.rgb_observed_shapes.setdefault(env_idx,{})
    if not hasattr(policy,'visual_freshness'):policy.visual_freshness={}
    freshness={};policy.visual_freshness[env_idx]=freshness
    out=[]
    robot_occluders={}
    slot=policy.executor._slots.get(env_idx)
    previous_robot=getattr(policy,'rgb_previous_robot',{}).get(env_idx)
    trace=os.environ.get('PHYSICALRSI_RGB_TRACE')
    step=policy.action_counts.get(env_idx,0)
    if trace and step%max(1,int(os.environ.get('PHYSICALRSI_RGB_TRACE_EVERY','20')))==0:
        directory=Path(trace);directory.mkdir(parents=True,exist_ok=True)
        arrays={f'vision/{name}/{key}':np.asarray(camera[key]) for name,camera in obs['vision'].items()
                for key in ('color','intrinsic_matrix','extrinsic_matrix')}
        arrays.update({f'state/{key}':np.asarray(value) for key,value in obs['state'].items()})
        arrays['phase']=np.asarray('initial' if slot is None else slot.phase)
        np.savez_compressed(directory/f'episode-{policy.episode}-env-{env_idx}-step-{step:05d}.npz',**arrays)
    try:
        head=Camera.from_observation(obs,'cam_head')
        plane=None
        if context is None:
            from PourBallsRGB.sam.sam3_http import segment_many_http
            observed_support=observe_support_geometry(obs,segment_many_http,occluders=('robot arm','cube'))
            plane=observed_support['plane']
            policy.rgb_observed_support[env_idx]=observed_support['patches']
            # Generic semantic instances reject wood/robot color reflections.
            # SAM supplies only current-image masks, never object dimensions.
            from PourBallsRGB.sam.sam3_http import segment_many_http
            instances=segment_many_http(head.image,[('cube',.3)])[0]
            if not len(instances.masks):raise GeometryUnavailable('No RGB cube instances observed')
            instance_mask=np.any(np.asarray(instances.masks,bool),axis=0)
            instance_mask=cv2.dilate(instance_mask.astype('uint8'),np.ones((5,5),'uint8')).astype(bool)
        for i,color in enumerate(colors):
            label='block_'+str(i)
            if context is None:
                center,rotation,half,mask,floor=_initialize(head,color,plane,instance_mask)
                history[label]={'half_extents':half.tolist(),'saturation_floor':floor,'support_plane':plane.tolist()}
                freshness[label]=True
            else:
                half=np.asarray(history[label]['half_extents'])
                previous=SimpleNamespace(center=context.geometries[label].center,
                    rotation=np.asarray(history[label].get('last_rotation',context.geometries[label].rotation)))
                if (slot is not None and i==context.bridge.target_index and slot.phase=='close'
                        and not history[label].get('wrist_refined',False)):
                    arm=slot.moves[slot.attempt%len(slot.moves)].arm
                    try:
                        previous,refined=_refine_resting_shape(Camera.from_observation(obs,'cam_'+arm+'_wrist'),color,previous,history[label])
                        history[label]=refined;half=np.asarray(refined['half_extents'])
                    except GeometryUnavailable:pass
                moving=(slot is not None and i==context.bridge.target_index and
                        slot.phase in ('lift','transfer','place','release'))
                predicted=previous
                if moving and previous_robot is not None and slot.moves:
                    arm=slot.moves[slot.attempt%len(slot.moves)].arm
                    # The object pose remains at its last verified observation on a gap.
                    # Its robot reference must remain at that same instant.
                    reference=history[label].get('observed_robot',previous_robot)
                    old=np.asarray(reference[arm+'_ee_pose']);new=np.asarray(obs['state'][arm+'_ee_pose'])
                    old_rotation=Rotation.from_quat(old[[4,5,6,3]]).as_matrix()
                    new_rotation=Rotation.from_quat(new[[4,5,6,3]]).as_matrix()
                    delta=new_rotation@old_rotation.T
                    predicted=SimpleNamespace(center=new[:3]+delta@(np.asarray(previous.center)-old[:3]),
                                              rotation=delta@previous.rotation)
                plane=np.asarray(history[label]['support_plane'])
                normal=np.r_[-plane[:2],1.];length=np.linalg.norm(normal);normal/=length
                clearance=np.asarray(previous.center)@normal-plane[2]/length-np.abs(normal@previous.rotation)@half
                supported=not moving and abs(clearance)<.2*float(np.min(half*2))
                try:
                    tracking_cameras=[head]
                    if slot is not None and i==context.bridge.target_index and slot.phase in ('close','lift','transfer','place','release','retreat'):
                        arm=slot.moves[slot.attempt%len(slot.moves)].arm
                        tracking_cameras=[Camera.from_observation(obs,'cam_'+arm+'_wrist'),head]
                    last_error=None
                    for tracking_camera in tracking_cameras:
                        try:
                            center,rotation,mask=_track(tracking_camera,color,predicted,history[label],supported=supported,stationary=not moving,predicted_motion=moving and previous_robot is not None,
                                occluder_provider=lambda camera:_robot_occluder(camera,robot_occluders))
                            break
                        except GeometryUnavailable as error:last_error=error
                    else:raise last_error
                    freshness[label]=True
                except GeometryUnavailable:
                    center=np.asarray(previous.center);rotation=np.asarray(previous.rotation)
                    freshness[label]=False;mask=None
            if freshness[label]:
                history[label]['last_rotation']=np.asarray(rotation).tolist()
                history[label]['observed_robot']={key:np.asarray(value).tolist()
                    for key,value in obs['state'].items() if key.endswith('_ee_pose')}
            if mask is None:pixel=(0.,0.)
            else:
                pixel=tuple(head.project([center])[0][0])
            out.append(RgbBlockDetection(tuple(center),tuple(half),tuple(map(tuple,rotation)),
                (HUES[color],255.,255.,0.,0.,0.),float(freshness[label]),pixel))
    except GeometryUnavailable as exc:
        raise StackBlocksRgbError(str(exc)) from exc
    if trace:
        directory=Path(trace);directory.mkdir(parents=True,exist_ok=True)
        with (directory/f'episode-{policy.episode}-env-{env_idx}-geometry.jsonl').open('a') as stream:
            stream.write(json.dumps({'step':step,'shapes':history,'freshness':freshness,
                'detections':[{'center':list(d.center),'rotation':[list(row) for row in d.rotation]} for d in out]})+'\n')
    if not hasattr(policy,'rgb_previous_robot'):policy.rgb_previous_robot={}
    policy.rgb_previous_robot[env_idx]={key:np.asarray(value).tolist() for key,value in obs['state'].items()}
    return tuple(out)
