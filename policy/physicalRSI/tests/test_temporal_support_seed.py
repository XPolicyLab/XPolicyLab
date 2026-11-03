from pathlib import Path
import sys
import itertools
import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry import Camera, GeometryUnavailable
from physicalrsi_observed_geometry.rgb_support import initial_plane
from physicalrsi_observed_geometry.primitives import fit_supported_primitive


def camera(x=0):
    t=np.eye(4);t[:3,3]=[x,-.2,2.]
    t[:3,:3]=Rotation.from_euler('x',15,degrees=True).as_matrix()
    return Camera(np.zeros((480,640,3),np.uint8),np.array([[650,0,320],[0,650,240],[0,0,1]]),t)


@pytest.mark.parametrize('plane', [[.02,-.03,.6],[-.04,.02,1.05]])
@pytest.mark.parametrize('has_texture',[True,False,'inconsistent'])
def test_prior_seed_requires_fresh_geometry_from_current_images(monkeypatch, plane, has_texture):
    from scipy.ndimage import map_coordinates, gaussian_filter
    from physicalrsi_observed_geometry import semantic_support as module
    plane=np.asarray(plane)
    texture=gaussian_filter(np.random.default_rng(2).uniform(0,1,(1024,1024)),1)
    cameras=[]
    for shift in (0,.25,-.25):
        transform=np.eye(4);transform[:3,3]=[shift,0,2]
        cam=Camera(np.zeros((240,320,3),np.uint8),np.array([[300,0,160],[0,300,120],[0,0,1]]),transform)
        yy,xx=np.indices((240,320));uv=np.c_[xx.ravel(),yy.ravel()]
        origin=transform[:3,3];rays=cam.unproject(uv,np.ones(len(uv)))-origin
        distance=(plane[2]-origin[2]+plane[0]*origin[0]+plane[1]*origin[1])/(rays[:,2]-plane[0]*rays[:,0]-plane[1]*rays[:,1])
        xyz=origin+rays*distance[:,None]
        values=map_coordinates(texture,[xyz[:,1]*300+512,xyz[:,0]*300+512],order=1,mode='nearest')
        rgb=np.repeat(np.clip(values.reshape(240,320)*255,0,255).astype('uint8')[:,:,None],3,axis=2)
        cameras.append(Camera(rgb,cam.k,cam.transform))
    if not has_texture:
        for cam in cameras:cam.image[:]=127
    if has_texture=='inconsistent':
        for index,cam in enumerate(cameras[1:]):
            cam.image[:]=np.random.default_rng(401+index).integers(0,256,cam.image.shape,dtype=np.uint8)
    masks=[np.ones((240,320),bool)]*len(cameras)
    point=np.array([[0.,0.,plane[2]]])
    def matches(a,b,*,masks):return np.empty((0,2)),np.empty((0,2)),np.empty(0)
    monkeypatch.setattr(module,'correspondences',matches)
    monkeypatch.setattr(module,'observe_support_masks',lambda *a,**kw:(cameras,masks))
    if not has_texture or has_texture=='inconsistent':
        with pytest.raises(GeometryUnavailable,match='validated photometric'):
            module.observe_semantic_plane({},None,prior_plane=plane-[0,0,.015])
        return
    result,observed=module.observe_semantic_plane({},None,prior_plane=plane-[0,0,.015])
    np.testing.assert_allclose(result,plane,atol=.003)
    assert observed.any()




def test_prior_seed_is_private_to_episode_and_copied():
    from types import SimpleNamespace
    from physicalrsi_observed_geometry.language_support import prior_language_support_plane
    plane=np.array([.01,.02,.6])
    first=SimpleNamespace(_observed_relay_support=(object(),SimpleNamespace(plane=plane)))
    second=SimpleNamespace()
    assert prior_language_support_plane(second) is None
    returned=prior_language_support_plane(first);returned[2]=9
    assert plane[2]==.6


def test_invalid_prior_is_rejected_before_matching():
    from physicalrsi_observed_geometry.semantic_support import observe_semantic_plane
    for invalid in [[0,1], [0,np.nan,.5], [0,0,np.inf]]:
        with pytest.raises(GeometryUnavailable,match='Finite previously observed'):
            observe_semantic_plane({},None,prior_plane=invalid)


def test_episode_initializer_clears_prior_and_calibration_receives_it():
    import ast
    from classify_language_depth import bind_language_depth_observation,bind_language_support_segmentation,replace_language_contact_floor,bind_language_prior_support
    from classify_language_camera import bind_language_camera_calls
    root=Path('/cpfs/physicalrsi-asset-audit/v4/implementations/code-skills/implementations/rdj_v1_classify_objects_by_language')
    paths=list(root.glob('*/rdj_v1_classify_objects_by_language/model.py'))
    if not paths:pytest.skip('Release fixture unavailable')
    for path in paths:
        source=bind_language_prior_support(replace_language_contact_floor(bind_language_support_segmentation(bind_language_camera_calls(bind_language_depth_observation(path.read_text()),'model'))))
        tree=ast.parse(source)
        episode=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Episode')
        init=next(n for n in episode.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        assert ast.unparse(init.body[-1])=='self._observed_relay_support = None'
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='calibrated_world_height']
        assert len(calls)==1
        prior=next(k.value for k in calls[0].keywords if k.arg=='prior_plane')
        assert ast.unparse(prior)=='prior_language_support_plane(self)'
