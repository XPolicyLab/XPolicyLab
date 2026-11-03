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


@pytest.mark.parametrize('plane', [[.05,-.04,.55],[-.06,.08,1.1]])
@pytest.mark.parametrize('unique', [False, True])
def test_plane_recovers_changed_height_and_tilt(plane, unique):
    cameras=[camera(),camera(.3),camera(-.3)]
    xy=np.array(list(itertools.product(np.linspace(-.15,.15,4),repeat=2)))
    points=np.c_[xy,xy@plane[:2]+plane[2]]
    def matches(a,b):
        return a.project(points)[0],b.project(points)[0],np.ones(len(points))
    actual=initial_plane(cameras,[np.ones((480,640),bool)]*3,matcher=matches,unique_head_pixels=unique)
    np.testing.assert_allclose(actual,plane,atol=1e-8)


@pytest.mark.parametrize('height,side',[(.55,.048),(1.05,.075)])
def test_cube_size_is_measured_from_changed_silhouette(height,side):
    cam=camera();center=np.array([.04,0,height+side/2])
    box=np.array(list(itertools.product([-.5,.5],repeat=3)))
    rotation=Rotation.from_euler('z',.3).as_matrix()
    pixels=cam.project((box*side)@rotation.T+center)[0]
    mask=np.zeros((480,640),np.uint8)
    cv2.fillConvexPoly(mask,cv2.convexHull(np.rint(pixels).astype('int32')),1)
    result=fit_supported_primitive(cam,mask,[0,0,height],'cube')
    np.testing.assert_allclose(result['dims'],side,atol=.003)
    np.testing.assert_allclose(result['pose'][:3],center,atol=.003)


def test_no_support_matches_is_not_replaced_by_fixed_height():
    cam=camera()
    with pytest.raises(GeometryUnavailable):
        initial_plane([cam,cam],[np.ones((480,640),bool)]*2,
                      matcher=lambda a,b:(np.empty((0,2)),np.empty((0,2)),np.empty(0)))


def test_observed_support_extent_excludes_distant_nonplane_matches_and_mask_holes():
    cameras=[camera(),camera(.3),camera(-.3)]
    xy=np.array(list(itertools.product(np.linspace(-.15,.15,4),repeat=2)))
    points=np.r_[np.c_[xy,np.full(len(xy),.7)],[[.3,.2,1.1]]]
    def matches(a,b):return a.project(points)[0],b.project(points)[0],np.ones(len(points))
    mask=np.ones((480,640),bool)
    hole=np.rint(cameras[0].project([[0,0,.7]])[0][0]).astype(int)
    mask[hole[1]-2:hole[1]+3,hole[0]-2:hole[0]+3]=False
    plane,observed=initial_plane(cameras,[mask,np.ones_like(mask),np.ones_like(mask)],
                                matcher=matches,unique_head_pixels=True,return_support_mask=True)
    np.testing.assert_allclose(plane,[0,0,.7],atol=1e-8)
    far=np.rint(cameras[0].project([points[-1]])[0][0]).astype(int)
    assert not observed[far[1],far[0]]
    assert not observed[hole[1],hole[0]]
    assert observed.any() and not observed[0].any()

@pytest.mark.parametrize('plane', [[.02,-.03,.6],[-.04,.02,1.05]])
@pytest.mark.parametrize('reference_views',[1,2])
def test_photometric_plane_refinement_on_changed_textured_support(plane,reference_views):
    from scipy.ndimage import map_coordinates, gaussian_filter
    from physicalrsi_observed_geometry.rgb_support import refine_plane
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
    cameras=cameras[:reference_views+1]
    masks=[np.ones((240,320),bool)]*len(cameras)
    result=refine_plane(cameras,masks,plane+[.01,-.01,-.015],minimum_reference_views=reference_views)
    np.testing.assert_allclose(result,plane,atol=.003)


def test_repeated_views_cannot_supply_six_independent_support_pixels():
    cameras=[camera(),camera(.2),camera(.3),camera(-.3)]
    points=np.array([[-.1,-.1,.7],[.1,-.1,.7],[0,.1,.7]])
    def matches(a,b):
        return a.project(points)[0],b.project(points)[0],np.ones(len(points))
    with pytest.raises(GeometryUnavailable, match='Insufficient calibrated'):
        initial_plane(cameras,[np.ones((480,640),bool)]*len(cameras),
                      matcher=matches,unique_head_pixels=True)


@pytest.mark.parametrize('invalid_reason', ['mask', 'epipolar'])
def test_invalid_high_confidence_match_does_not_hide_valid_other_view(invalid_reason):
    cameras=[camera(),camera(.3),camera(-.3)]
    xy=np.array(list(itertools.product(np.linspace(-.12,.12,4),repeat=2)))
    points=np.c_[xy,np.full(len(xy),.7)]
    masks=[np.ones((480,640),bool) for _ in cameras]
    if invalid_reason=='mask':masks[1][:]=False
    def matches(a,b):
        target=b.project(points)[0].copy()
        if b is cameras[1] and invalid_reason=='epipolar':target[:,1]+=20
        return a.project(points)[0],target,np.full(len(points),.9 if b is cameras[1] else .8)
    actual=initial_plane(cameras,masks,matcher=matches,unique_head_pixels=True)
    np.testing.assert_allclose(actual,[0,0,.7],atol=1e-8)
