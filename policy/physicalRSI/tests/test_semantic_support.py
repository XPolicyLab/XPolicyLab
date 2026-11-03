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
def test_semantic_multistart_requires_independent_texture(monkeypatch, plane, has_texture):
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
    def matches(a,b,*,masks):return a.project(point)[0],b.project(point)[0],np.ones(1)
    monkeypatch.setattr(module,'correspondences',matches)
    monkeypatch.setattr(module,'observe_support_masks',lambda *a,**kw:(cameras,masks))
    if not has_texture or has_texture=='inconsistent':
        with pytest.raises(GeometryUnavailable,match='validated photometric'):
            module.observe_semantic_plane({},None)
        return
    result,observed=module.observe_semantic_plane({},None)
    np.testing.assert_allclose(result,plane,atol=.003)
    assert observed.any()



def test_language_model_passes_its_resident_segmenter():
    from classify_language_depth import bind_language_depth_observation, bind_language_support_segmentation
    import ast
    root=Path('/cpfs/physicalrsi-asset-audit/v4/implementations/code-skills/implementations/rdj_v1_classify_objects_by_language')
    paths=list(root.glob('*/rdj_v1_classify_objects_by_language/model.py'))
    if not paths:pytest.skip('Release fixture unavailable')
    for path in paths:
        bound=ast.parse(bind_language_depth_observation(path.read_text()))
        after=ast.parse(bind_language_support_segmentation(ast.unparse(bound)))
        calls=[n for n in ast.walk(after) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='calibrated_world_height']
        assert len(calls)==1
        keyword=calls[0].keywords.pop()
        assert keyword.arg=='segment_fn' and ast.unparse(keyword.value)=='self.inventory.sam.segment_many'
        assert ast.dump(after)==ast.dump(bound)
