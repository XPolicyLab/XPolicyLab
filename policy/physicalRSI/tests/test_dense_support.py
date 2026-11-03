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
@pytest.mark.parametrize('proposal_error_px',[1.,3.])
def test_sparse_plane_proposal_requires_held_out_image_validation(plane, has_texture, proposal_error_px):
    from scipy.ndimage import map_coordinates, gaussian_filter
    from physicalrsi_observed_geometry.dense_support import photometric_initial_plane
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
    def matches(a,b):return a.project(point)[0],b.project(point)[0],np.ones(1)
    if not has_texture or has_texture=='inconsistent':
        with pytest.raises(GeometryUnavailable,match='validated photometric'):
            photometric_initial_plane(cameras,masks,matcher=matches,proposal_error_px=proposal_error_px)
        return
    result,observed=photometric_initial_plane(cameras,masks,matcher=matches,proposal_error_px=proposal_error_px)
    np.testing.assert_allclose(result,plane,atol=.003)
    assert observed.any()


def test_correspondence_proposal_tolerance_is_explicit_and_default_is_unchanged():
    from physicalrsi_observed_geometry.rgb_support import support_correspondences
    first,second=camera(0),camera(.25)
    point=np.array([[0.,0.,1.]])
    def matches(a,b):return a.project(point)[0],b.project(point)[0]+[0,3],np.ones(1)
    masks=[np.ones(first.image.shape[:2],bool),np.ones(second.image.shape[:2],bool)]
    strict,_=support_correspondences([first,second],masks,matches)
    proposals,_=support_correspondences([first,second],masks,matches,max_reprojection_error_px=3.)
    assert len(strict)==0 and len(proposals)==1
