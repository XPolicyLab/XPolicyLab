from pathlib import Path
import sys,time
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry import Camera
from physicalrsi_observed_geometry.observed_depth_alignment import ObservedPlaneDepthAlignment

@pytest.mark.parametrize('plane,scale,offset', [([.03,-.02,.45],.17,.06), ([-.04,.06,1.15],.4,-.1)])
def test_alignment_uses_measured_plane_height_and_tilt(plane,scale,offset):
    transform=np.eye(4);transform[:3,3]=[0,-.2,2]
    transform[:3,:3]=Rotation.from_euler('x',20,degrees=True).as_matrix()
    camera=Camera(np.zeros((80,100,3),np.uint8),np.array([[100,0,50],[0,100,40],[0,0,1]]),transform)
    y,x=np.indices((80,100));pixels=np.c_[x.ravel(),y.ravel()]
    origin=transform[:3,3];rays=camera.unproject(pixels,np.ones(len(pixels)))-origin
    a,b,c=plane;depth=((c-origin[2]+a*origin[0]+b*origin[1])/(rays[:,2]-a*rays[:,0]-b*rays[:,1])).reshape(80,100)
    prediction=(depth-offset)/scale
    mask=np.ones_like(depth,dtype=bool)
    aligner=ObservedPlaneDepthAlignment(camera=camera,plane=plane,target='axial')
    fitted,report=aligner.align(prediction,mask,deadline=time.monotonic()+5)
    np.testing.assert_allclose(fitted,depth,atol=1e-9)
    np.testing.assert_allclose(report['observed_plane'],plane)
    assert report['holdout']['inlier_fraction']==1
    # A biased supplied plane must change the answer: there is no hidden fixed-height fallback.
    biased=ObservedPlaneDepthAlignment(camera=camera,plane=[a,b,c+.04],target='axial')
    other,_=biased.align(prediction,mask,deadline=time.monotonic()+5)
    assert np.max(np.abs(other-fitted))>.03


def test_observed_plane_rejects_missing_or_invalid_geometry():
    for plane in ([0,0,np.nan],[0,0],None):
        with pytest.raises((ValueError,TypeError)):
            ObservedPlaneDepthAlignment(camera=None,plane=plane,target='axial')

def test_scene_calibrates_before_unprojecting_object_pixels(monkeypatch):
    from physicalrsi_observed_geometry import Scene
    from physicalrsi_observed_geometry import rgb_support
    cameras=[]
    for shift in (0,.25,-.25):
        transform=np.eye(4);transform[:3,3]=[shift,-.2,2]
        transform[:3,:3]=Rotation.from_euler('x',20,degrees=True).as_matrix()
        cameras.append(Camera(np.zeros((80,100,3),np.uint8),np.array([[100,0,50],[0,100,40],[0,0,1]]),transform))
    head=cameras[0];height=.57
    y,x=np.indices((80,100));uv=np.c_[x.ravel(),y.ravel()]
    rays=head.unproject(uv,np.ones(len(uv)))-head.transform[:3,3]
    object_mask=np.zeros((80,100),bool);object_mask[30:45,40:60]=True
    z=np.full((80,100),height);z[object_mask]+=.06
    depth=((z.ravel()-2)/rays[:,2]).reshape(80,100)
    observation={'vision':{name:{'color':cam.image,'intrinsic_matrix':cam.k,'extrinsic_matrix':cam.transform} for name,cam in zip(['cam_head','cam_left_wrist','cam_right_wrist'],cameras)}}
    # Supply perfect synthetic RGB matches; the real calibrated triangulation determines scale.
    support_pixels=np.array([[u,v] for u in np.linspace(25,75,4) for v in np.linspace(10,25,4)])
    support_rays=head.unproject(support_pixels,np.ones(len(support_pixels)))-head.transform[:3,3]
    points=head.transform[:3,3]+support_rays*((height-2)/support_rays[:,2])[:,None]
    original=rgb_support.initial_plane;calls=[]
    def measured(cams,masks):
        calls.append(True)
        return original(cams,masks,matcher=lambda a,b:(a.project(points)[0],b.project(points)[0],np.ones(len(points))))
    monkeypatch.setattr(rgb_support,'initial_plane',measured)
    monkeypatch.setattr(rgb_support,'refine_plane',lambda cameras,masks,start:start)
    scene=Scene.from_observation(observation,predictor=lambda rgb:(depth-.07)/.2)
    item=scene.geometry(object_mask,support_mask=~object_mask)
    np.testing.assert_allclose(scene.depth,depth,atol=1e-8)
    corners=np.array(item['bbox_local'])+np.array(item['pose'][:3])
    assert abs(corners[:,2].min()-height)<1e-6
    assert abs(corners[:,2].max()-(height+.06))<1e-6
    scene.geometry(object_mask,support_mask=~object_mask)
    assert len(calls)==1
