"""Behavior tests for the new shape registration and RGB tube geometry."""
import importlib.util
from pathlib import Path
import sys
import types
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]/'asset_revision'
sys.path.insert(0,str(ROOT))
from physicalrsi_observed_geometry import Camera, Scene, GeometryUnavailable


def load(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/'overlays'/file)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def make_scene(width=16,translation=0):
    rgb=np.zeros((80,100,3),np.uint8)
    k=np.array([[100.,0,50],[0,100.,40],[0,0,1]])
    t=np.eye(4);t[:3,3]=[translation,0,2]
    mask=np.zeros((80,100),bool);mask[25:55,50-width//2:50+width//2]=True
    depth=np.full(mask.shape,1.4);depth[mask]=np.linspace(1.,1.15,mask.sum())
    return Scene(Camera(rgb,k,t),predictor=lambda _:depth),mask


def test_registration_uses_observed_shape_and_current_camera():
    module=load('observed_pour_test','pour_perception.py')
    a,ma=make_scene(12);b,mb=make_scene(30,.4)
    fa=module.registration(ma,'cup',scene=a);fb=module.registration(mb,'cup',scene=b)
    assert fb['pose'][0,3]-fa['pose'][0,3] == pytest.approx(.4,abs=.02)
    assert np.ptp(fb['points_local'],axis=0).max()>np.ptp(fa['points_local'],axis=0).max()
    assert fa['source']=='episode_rgb_metric_depth'
    assert not fa['hidden_surfaces_reconstructed']
    assert not hasattr(module,'SHAPES')


def test_tracking_keeps_template_explicit_and_episode_local():
    module=load('observed_pour_tracking_test','pour_perception.py')
    a,mask=make_scene();initial=module.registration(mask,'cup',scene=a)
    template=initial['points_local'].copy()
    moved,_=make_scene(16,.015)
    tracked=module.registration(mask,'cup',scene=moved,prior=initial['pose'],template=template)
    assert np.array_equal(template,initial['points_local'])
    assert tracked['pose'][0,3]-initial['pose'][0,3] == pytest.approx(.015,abs=.008)
    inferred=module.registration(mask,'cup',scene=moved,prior=initial['pose'],template=template,inferred_template=True)
    assert inferred['hidden_surfaces_reconstructed']
    assert not tracked['hidden_surfaces_reconstructed']
    with pytest.raises(GeometryUnavailable,match='previous pose'):
        module.registration(mask,'cup',scene=a,template=template)


@pytest.mark.parametrize('mode',['outside','behind','occluded'])
def test_unavailable_wrist_view_rejects_binding_without_aborting_episode(mode):
    module=load('observed_pour_missing_view_test','pour_perception.py')
    scene,mask=make_scene()
    wrist,_=make_scene(translation=10 if mode=='outside' else 0)
    if mode=='behind':wrist.camera.transform[2,3]=-2
    occlusion=np.full(mask.shape,mode=='occluded',bool)
    fit=module.registration(mask,'cup',scene=scene,extra_views=[
        {'camera':wrist.camera,'mask':mask,'occlusion':occlusion}])
    assert fit['extra_views']==1
    assert fit['error_px']<3
    assert not all(e<4 for e in fit['view_errors_px'])
    assert np.isfinite(fit['view_errors_px']).all()


def insert_module(monkeypatch):
    package=types.ModuleType('insert_observation_test');package.__path__=[]
    contract=types.ModuleType(package.__name__+'.contract');contract.InsertTubesEnvelopeError=ValueError
    geometry=types.ModuleType(package.__name__+'.geometry')
    geometry._rotation=lambda q:Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix()
    geometry._quaternion=lambda r:Rotation.from_matrix(r).as_quat()[[3,0,1,2]].tolist()
    for module in (package,contract,geometry):monkeypatch.setitem(sys.modules,module.__name__,module)
    return load(package.__name__+'.rgb_perception','insert_rgb_perception.py')


def test_tube_length_changes_with_rgb_extent(monkeypatch):
    module=insert_module(monkeypatch)
    lengths=[]
    for right in (55,75):
        rgb=np.zeros((80,100,3),np.uint8);rgb[35:45,20:right]=230;rgb[35:45,right:right+6]=[240,100,0]
        camera=Camera(rgb,[[100,0,50],[0,100,40],[0,0,1]],np.eye(4))
        scene=Scene(camera,predictor=lambda _:np.ones((80,100)))
        tubes=module._tubes(scene)
        assert len(tubes)==1
        lengths.append(tubes[0][1])
    assert lengths[1]-lengths[0]==pytest.approx(.2,abs=.02)


def test_missing_rack_openings_are_not_filled_from_fixed_grid(monkeypatch):
    module=insert_module(monkeypatch)
    rgb=np.zeros((80,100,3),np.uint8);rgb[20:60,20:80]=[0,180,255]
    scene=Scene(Camera(rgb,[[100,0,50],[0,100,40],[0,0,1]],np.eye(4)),predictor=lambda _:np.ones((80,100)))
    with pytest.raises(GeometryUnavailable,match='No closed rack openings'):
        module._rack(scene)


@pytest.mark.parametrize('clipped_part',['body','cap'])
def test_held_axis_does_not_accept_image_truncated_tube(monkeypatch,clipped_part):
    module=insert_module(monkeypatch)
    def observation(clipped):
        rgb=np.zeros((100,100,3),np.uint8)
        start=0 if clipped and clipped_part=='cap' else 10
        end=100 if clipped and clipped_part=='body' else 90
        rgb[start+10:end,45:55]=230
        rgb[start:start+10,45:55]=[240,100,0]
        return Scene(Camera(rgb,[[100,0,50],[0,100,50],[0,0,1]],np.eye(4)),predictor=lambda _:np.ones((100,100)))
    assert len(module._tubes(observation(False),require_complete=True))==1
    assert module._tubes(observation(True),require_complete=True)==[]


def test_complete_resting_tube_is_not_selected_as_held(monkeypatch):
    module=insert_module(monkeypatch)
    rgb=np.zeros((100,100,3),np.uint8);rgb[20:90,45:55]=230;rgb[10:20,45:55]=[240,100,0]
    scene=Scene(Camera(rgb,[[100,0,50],[0,100,50],[0,0,1]],np.eye(4)),predictor=lambda _:np.ones((100,100)))
    assert len(module._tubes(scene,require_complete=True,contact_pixel=[50,50]))==1
    assert module._tubes(scene,require_complete=True,contact_pixel=[80,50])==[]
