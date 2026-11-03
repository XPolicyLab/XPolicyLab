from pathlib import Path
import sys
import cv2
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry import Camera,GeometryUnavailable
from physicalrsi_observed_geometry.key_outline import observe_outline,held_outline


@pytest.mark.parametrize('width,length',[(.008,.04),(.02,.11)])
def test_observed_shapes_of_different_sizes_are_checked_without_mesh(width,length):
    k=np.array([[400,0,128],[0,400,128],[0,0,1.]])
    mount=np.eye(4);relative=np.eye(4)
    relative[:3,:3]=np.array([[1,0,0],[0,0,-1],[0,1,0]])
    relative[:3,3]=[0,0,-.4]
    points=np.array([[-width/2,0,-length],[width/2,0,-length],[width/2,0,0],[-width/2,0,0]])
    image=np.zeros((256,256,3),np.uint8);cam=Camera(image,k,mount)
    uv,_=cam.project(points@relative[:3,:3].T+relative[:3,3])
    cv2.fillConvexPoly(image,np.rint(uv).astype(np.int32),(200,200,200))
    accepted,info=held_outline(image,relative,(mount,k),points)
    assert accepted and info['ratio']>.95
    blank=np.zeros_like(image)
    assert not held_outline(blank,relative,(mount,k),points)[0]
    blank[:30,:30]=200 # A robot-colored region away from the object cannot certify a grasp.
    assert not held_outline(blank,relative,(mount,k),points)[0]
    edge=relative.copy();edge[:3,:3]=np.eye(3)
    assert not held_outline(image,edge,(mount,k),points)[0]


def test_initial_rgb_outline_roundtrips_through_measured_plane():
    image=np.zeros((180,240,3),np.uint8);mask=np.zeros(image.shape[:2],np.uint8)
    cv2.rectangle(mask,(110,50),(117,110),1,-1);cv2.circle(mask,(114,120),14,1,-1)
    t=np.eye(4);t[:3,3]=[.2,-.1,1.7]
    camera=Camera(image,np.array([[220,0,120],[0,230,90],[0,0,1]]),t)
    pose=np.eye(4);pose[:3,:3]=np.array([[1,0,0],[0,0,-1],[0,1,0]]);pose[:3,3]=[.2,-.1,.9]
    points=observe_outline(camera,mask,pose)
    assert np.max(np.abs(points[:,1]))<1e-12
    uv,_=camera.project(points@pose[:3,:3].T+pose[:3,3])
    recovered=np.zeros_like(mask);cv2.fillPoly(recovered,[np.rint(uv).astype(np.int32)],1)
    assert np.array_equal(recovered,mask)
    with pytest.raises(GeometryUnavailable):observe_outline(camera,np.zeros_like(mask),pose)
