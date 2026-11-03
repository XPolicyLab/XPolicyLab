"""Exercise production color/geometry helpers without importing robot-controller vendors."""
import ast
from pathlib import Path
import sys
import itertools
import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]/'asset_revision'
sys.path.insert(0,str(ROOT))
from physicalrsi_observed_geometry import Camera,GeometryUnavailable
from physicalrsi_observed_geometry.primitives import fit_supported_primitive
from physicalrsi_observed_geometry.boxes import fit_box_views,fit_supported_box_pose
from physicalrsi_observed_geometry.occlusion import verify_partial_box
from types import SimpleNamespace


def helpers():
    tree=ast.parse((ROOT/'overlays/language_rgb.py').read_text())
    selected=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in ('_components','_initialize','_track','_refine_resting_shape')]
    namespace=dict(cv2=cv2,np=np,Rotation=Rotation,SimpleNamespace=SimpleNamespace,GeometryUnavailable=GeometryUnavailable,
                   fit_supported_primitive=fit_supported_primitive,fit_box_views=fit_box_views,
                   verify_partial_box=verify_partial_box,fit_supported_box_pose=fit_supported_box_pose,HUES={'blue':120.})
    exec(compile(ast.Module(body=selected,type_ignores=[]),'<production-language-helpers>','exec'),namespace)
    return namespace


@pytest.mark.parametrize('height,side',[(.57,.029),(.91,.061)])
def test_initialization_measures_size_and_ignores_small_reflection(height,side):
    transform=np.eye(4);transform[:3,3]=[.1,-.18,1.8]
    transform[:3,:3]=Rotation.from_euler('x',12,degrees=True).as_matrix()
    image=np.full((480,640,3),100,np.uint8)
    camera=Camera(image,[[650,0,320],[0,650,240],[0,0,1]],transform)
    center=np.array([.02,.01,height+side/2])
    rotation=Rotation.from_euler('z',.31).as_matrix()
    points=np.array(list(itertools.product([-.5,.5],repeat=3)))*side
    pixels=camera.project(points@rotation.T+center)[0]
    cv2.fillConvexPoly(image,cv2.convexHull(np.rint(pixels).astype('int32')),(0,0,255))
    image[30:34,30:34]=[0,0,255]
    actual,_,half,mask,_=helpers()['_initialize'](camera,'blue',np.array([0.,0.,height]))
    np.testing.assert_allclose(actual,center,atol=.003)
    np.testing.assert_allclose(half*2,side,atol=.003)
    assert not mask[31,31]


def test_absent_color_has_no_asset_fallback():
    camera=Camera(np.full((100,100,3),100,np.uint8),[[100,0,50],[0,100,50],[0,0,1]],np.eye(4))
    with pytest.raises(GeometryUnavailable):helpers()['_initialize'](camera,'blue',[0.,0.,.7])


@pytest.mark.parametrize('height,valid',[(.92,True),(.65,False)])
def test_tracking_allows_lift_but_rejects_below_observed_support(height,valid):
    transform=np.eye(4);transform[:3,3]=[.1,-.18,1.8]
    transform[:3,:3]=Rotation.from_euler('x',12,degrees=True).as_matrix()
    image=np.full((480,640,3),100,np.uint8)
    camera=Camera(image,[[650,0,320],[0,650,240],[0,0,1]],transform)
    center=np.array([.02,.01,height]);side=.05
    rotation=Rotation.from_euler('xyz',[.08,-.1,.31]).as_matrix()
    points=np.array(list(itertools.product([-.5,.5],repeat=3)))*side
    pixels=camera.project(points@rotation.T+center)[0]
    cv2.fillConvexPoly(image,cv2.convexHull(np.rint(pixels).astype('int32')),(0,0,255))
    previous=SimpleNamespace(center=center,rotation=rotation)
    shape={'half_extents':[side/2]*3,'saturation_floor':120,'support_plane':[0.,0.,.75]}
    track=helpers()['_track']
    if valid:
        actual,_,_=track(camera,'blue',previous,shape,supported=False)
        np.testing.assert_allclose(actual,center,atol=.015)
    else:
        with pytest.raises(GeometryUnavailable):track(camera,'blue',previous,shape,supported=False)


def test_observed_rotation_survives_controller_context_that_only_updates_centers(monkeypatch):
    monkeypatch.delenv('PHYSICALRSI_RGB_TRACE',raising=False)
    import os
    tree=ast.parse((ROOT/'overlays/language_rgb.py').read_text())
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='detect_language_blocks')
    measured=Rotation.from_euler('xyz',[.2,.1,.3]).as_matrix()
    seen=[]
    seen_cameras=[]
    def track(camera,color,previous,shape,**kwargs):
        seen.append(np.asarray(previous.rotation).copy())
        seen_cameras.append(int(camera.image[0,0,0]))
        return np.asarray(previous.center),measured,np.ones(camera.image.shape[:2],bool)
    ns=dict(np=np,os=os,Path=Path,SimpleNamespace=SimpleNamespace,Camera=Camera,Rotation=Rotation,
            GeometryUnavailable=GeometryUnavailable,StackBlocksRgbError=RuntimeError,
            parse=lambda _:('blue','yellow','orange'),HUES={'blue':120.,'yellow':30.,'orange':15.},
            _track=track,RgbBlockDetection=lambda *args:args)
    exec(compile(ast.Module(body=[function],type_ignores=[]),'<production-language-detection>','exec'),ns)
    labels=['block_'+str(i) for i in range(3)]
    context=SimpleNamespace(geometries={label:SimpleNamespace(center=np.array([0.,0.,.9]),rotation=np.eye(3)) for label in labels})
    policy=SimpleNamespace(_rdj_stack_blocks_rgb_context={0:context},executor=SimpleNamespace(_slots={}),
        action_counts={0:0},episode=1,rgb_observed_shapes={0:{label:{'half_extents':[.025]*3,'support_plane':[0,0,.75],'saturation_floor':120} for label in labels}})
    transform=np.eye(4);transform[2,3]=2.
    observation={'instruction':'unused','state':{},'vision':{'cam_head':{'color':np.zeros((100,100,3),np.uint8),'intrinsic_matrix':np.array([[100,0,50],[0,100,50],[0,0,1]]),'extrinsic_matrix':transform}}}
    ns['detect_language_blocks'](policy,0,observation)
    ns['detect_language_blocks'](policy,0,observation)
    for rotation in seen[:3]:np.testing.assert_allclose(rotation,np.eye(3))
    for rotation in seen[3:]:np.testing.assert_allclose(rotation,measured)

    # During retreat the object may already be stacked while the head is occluded.
    context.bridge=SimpleNamespace(target_index=1)
    policy.executor._slots[0]=SimpleNamespace(phase='retreat',attempt=0,moves=[SimpleNamespace(arm='left')])
    observation['vision']['cam_left_wrist']=dict(observation['vision']['cam_head'],color=np.ones((100,100,3),np.uint8))
    ns['detect_language_blocks'](policy,0,observation)
    assert seen_cameras[-3:]==[0,1,0]


@pytest.mark.parametrize('side,height',[(.029,.57),(.061,.91)])
def test_close_view_remeasures_episode_shape_and_rejects_remote_reflections(side,height):
    transform=np.eye(4);transform[:3,3]=[.02,-.01,height+.3]
    transform[:3,:3]=Rotation.from_euler('x',12,degrees=True).as_matrix()
    image=np.full((480,640,3),100,np.uint8)
    camera=Camera(image,[[650,0,320],[0,650,240],[0,0,1]],transform)
    center=np.array([.02,.01,height+side/2]);rotation=Rotation.from_euler('z',.31).as_matrix()
    points=np.array(list(itertools.product([-.5,.5],repeat=3)))*side
    pixels=camera.project(points@rotation.T+center)[0]
    cv2.fillConvexPoly(image,cv2.convexHull(np.rint(pixels).astype('int32')),(0,0,255))
    previous=SimpleNamespace(center=center,rotation=rotation)
    shape={'half_extents':[side*.55]*3,'support_plane':[0.,0.,height],'saturation_floor':120}
    measured,refined=helpers()['_refine_resting_shape'](camera,'blue',previous,shape)
    np.testing.assert_allclose(measured.center,center,atol=.002)
    np.testing.assert_allclose(np.asarray(refined['half_extents'])*2,side,atol=.002)
    previous.center=center+[.3,0,0]
    with pytest.raises(GeometryUnavailable):helpers()['_refine_resting_shape'](camera,'blue',previous,shape)


@pytest.mark.parametrize('pixel_shift',[0,25])
@pytest.mark.parametrize('brightness',[150,255])
def test_rigid_prediction_requires_current_rgb_agreement(pixel_shift,brightness):
    transform=np.eye(4);transform[:3,3]=[.1,-.18,1.8]
    transform[:3,:3]=Rotation.from_euler('x',12,degrees=True).as_matrix()
    image=np.full((480,640,3),100,np.uint8)
    camera=Camera(image,[[650,0,320],[0,650,240],[0,0,1]],transform)
    center=np.array([.02,.01,.95]);side=.05
    rotation=Rotation.from_euler('xyz',[.08,-.1,.31]).as_matrix()
    points=np.array(list(itertools.product([-.5,.5],repeat=3)))*side
    pixels=camera.project(points@rotation.T+center)[0]+[pixel_shift,0]
    cv2.fillConvexPoly(image,cv2.convexHull(np.rint(pixels).astype('int32')),(0,0,brightness))
    previous=SimpleNamespace(center=center,rotation=rotation)
    shape={'half_extents':[side/2]*3,'saturation_floor':120,'support_plane':[0.,0.,.75],'value_floor':200}
    ns=helpers()
    def unavailable(*args,**kwargs):raise GeometryUnavailable('Unresolved pose optimization')
    ns['fit_box_views']=unavailable
    if pixel_shift:
        with pytest.raises(GeometryUnavailable):ns['_track'](camera,'blue',previous,shape,predicted_motion=True)
    else:
        actual,_,_=ns['_track'](camera,'blue',previous,shape,predicted_motion=True)
        np.testing.assert_allclose(actual,center)


def test_reacquisition_accumulates_robot_motion_since_last_verified_pose(monkeypatch):
    """A missing image must not advance the reference for a frozen object pose."""
    import os
    monkeypatch.delenv('PHYSICALRSI_RGB_TRACE',raising=False)
    tree=ast.parse((ROOT/'overlays/language_rgb.py').read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='detect_language_blocks')
    seen=[];visible=False
    def track(camera,color,previous,shape,**kwargs):
        if color=='yellow':
            seen.append(np.asarray(previous.center).copy())
            if not visible:raise GeometryUnavailable('Temporary occlusion')
        return previous.center,previous.rotation,np.ones(camera.image.shape[:2],bool)
    ns=dict(np=np,os=os,Path=Path,SimpleNamespace=SimpleNamespace,Camera=Camera,Rotation=Rotation,
        GeometryUnavailable=GeometryUnavailable,StackBlocksRgbError=RuntimeError,
        parse=lambda _:('blue','yellow','orange'),HUES={'blue':120.,'yellow':30.,'orange':15.},
        _track=track,RgbBlockDetection=lambda *args:args)
    exec(compile(ast.Module(body=[function],type_ignores=[]),'<production-language-detection>','exec'),ns)
    labels=['block_'+str(i) for i in range(3)];origin=[0.,0.,1.,1.,0.,0.,0.]
    context=SimpleNamespace(geometries={label:SimpleNamespace(center=np.array([0.,0.,.9]),rotation=np.eye(3)) for label in labels},bridge=SimpleNamespace(target_index=1))
    slot=SimpleNamespace(phase='transfer',attempt=0,moves=[SimpleNamespace(arm='left')])
    policy=SimpleNamespace(_rdj_stack_blocks_rgb_context={0:context},executor=SimpleNamespace(_slots={0:slot}),action_counts={0:0},episode=1,
        rgb_previous_robot={0:{'left_ee_pose':origin}},rgb_observed_shapes={0:{label:{'half_extents':[.025]*3,'support_plane':[0.,0.,.75],'saturation_floor':120,'observed_robot':{'left_ee_pose':origin}} for label in labels}})
    transform=np.eye(4);transform[2,3]=2.
    camera={'color':np.zeros((100,100,3),np.uint8),'intrinsic_matrix':np.array([[100,0,50],[0,100,50],[0,0,1]]),'extrinsic_matrix':transform}
    observation={'instruction':'unused','state':{'left_ee_pose':[.01,0.,1.,1.,0.,0.,0.]},'vision':{'cam_head':camera,'cam_left_wrist':camera}}
    ns['detect_language_blocks'](policy,0,observation)
    assert not policy.visual_freshness[0]['block_1']
    np.testing.assert_allclose(policy.rgb_observed_shapes[0]['block_1']['observed_robot']['left_ee_pose'],origin)
    visible=True;observation['state']['left_ee_pose']=[.02,0.,1.,1.,0.,0.,0.]
    ns['detect_language_blocks'](policy,0,observation)
    np.testing.assert_allclose(seen[-1],[.02,0.,.9])
    assert policy.visual_freshness[0]['block_1']



def test_semantic_instances_filter_background_without_supplying_geometry():
    transform=np.eye(4);transform[:3,3]=[.1,-.18,1.8]
    transform[:3,:3]=Rotation.from_euler('x',12,degrees=True).as_matrix()
    image=np.full((480,640,3),100,np.uint8)
    camera=Camera(image,[[650,0,320],[0,650,240],[0,0,1]],transform)
    rotation=Rotation.from_euler('z',.31).as_matrix();height=.72
    target=np.array([.02,.01,height+.025]);instance=np.zeros(image.shape[:2],np.uint8)
    for i,(center,side) in enumerate([(target,.05),(np.array([.2,.01,height+.045]),.09)]):
        points=np.array(list(itertools.product([-.5,.5],repeat=3)))*side
        pixels=camera.project(points@rotation.T+center)[0]
        hull=cv2.convexHull(np.rint(pixels).astype('int32'))
        cv2.fillConvexPoly(image,hull,(0,0,255))
        if i==0:cv2.fillConvexPoly(instance,hull,1)
    actual,_,half,_,_=helpers()['_initialize'](camera,'blue',[0,0,height],instance.astype(bool))
    np.testing.assert_allclose(actual,target,atol=.003)
    np.testing.assert_allclose(half*2,.05,atol=.003)
    with pytest.raises(GeometryUnavailable):
        helpers()['_initialize'](camera,'blue',[0,0,height],np.zeros_like(instance,bool))


def test_stationary_pose_requires_visible_evidence_under_robot_occlusion():
    transform=np.eye(4);transform[:3,3]=[.1,-.18,1.8]
    transform[:3,:3]=Rotation.from_euler('x',12,degrees=True).as_matrix()
    image=np.full((480,640,3),100,np.uint8)
    camera=Camera(image,[[650,0,320],[0,650,240],[0,0,1]],transform)
    center=np.array([.02,.01,.90]);side=.05
    rotation=Rotation.from_euler('z',.31).as_matrix()
    points=np.array(list(itertools.product([-.5,.5],repeat=3)))*side
    pixels=camera.project(points@rotation.T+center)[0]
    hull=cv2.convexHull(np.rint(pixels).astype('int32'))
    cv2.fillConvexPoly(image,hull,(0,0,255))
    robot=np.zeros(image.shape[:2],bool)
    x0,y0=hull.reshape(-1,2).min(0);x1,y1=hull.reshape(-1,2).max(0)
    robot[y0-3:(y0+y1)//2,x0-3:x1+4]=True
    image[robot]=[20,20,20]
    previous=SimpleNamespace(center=center,rotation=rotation)
    shape={'half_extents':[side/2]*3,'saturation_floor':120,'support_plane':[0.,0.,.7]}
    calls=[]
    def provider(c):calls.append(c);return robot
    actual,actual_rotation,_=helpers()['_track'](camera,'blue',previous,shape,
        stationary=True,occluder_provider=provider)
    np.testing.assert_array_equal(actual,center)
    np.testing.assert_array_equal(actual_rotation,rotation)
    assert calls==[camera]
    # A small remaining fragment must not be interpreted as a smaller/farther
    # stationary box when the current robot explains the missing silhouette.
    robot[y0-3:y1-3,x0-3:x1+4]=True
    image[robot]=20
    with pytest.raises(GeometryUnavailable,match='occluded'):
        helpers()['_track'](camera,'blue',previous,shape,stationary=True,occluder_provider=provider)
    # A fully hidden object cannot become a fresh observation from the mask alone.
    image[:]=20
    with pytest.raises(GeometryUnavailable):
        helpers()['_track'](camera,'blue',previous,shape,stationary=True,occluder_provider=lambda _:np.ones(image.shape[:2],bool))
