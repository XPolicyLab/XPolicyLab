from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry import Camera, GeometryUnavailable
from physicalrsi_observed_geometry.slot_servo import transverse_correction

@pytest.mark.parametrize('height,angle',[(.6,0),(1.1,25)])
@pytest.mark.parametrize('side',[-1,1])
def test_eye_in_hand_translation_reduces_pixel_error_without_changing_plane_height(height,angle,side):
    transform=np.eye(4);transform[:3,3]=[0,0,height+.4]
    transform[:3,:3]=Rotation.from_euler('x',angle,degrees=True).as_matrix()
    camera=Camera(np.zeros((480,640,3),np.uint8),np.array([[400.,0,320],[0,400,240],[0,0,1]]),transform)
    plane=np.array([0.,0.,height]);normal=np.array([0.,0.,1.])
    slot_world=np.array([[0,-.02,height],[0,.02,height]])
    slot_pixels=camera.project(slot_world)[0];center=slot_pixels.mean(0);axis=slot_pixels[1]-slot_pixels[0];axis/=np.linalg.norm(axis)
    across=np.array([-axis[1],axis[0]])
    coin=center+side*12*across+np.arange(-3,4)[:,None]*axis
    result=transverse_correction(camera,plane,normal,center,axis,coin)
    assert np.linalg.norm(result['translation'])<=.003000001
    assert abs(result['translation']@normal)<1e-10
    moved=transform.copy();moved[:3,3]+=result['translation']
    updated=Camera(camera.image,camera.k,moved).project(slot_world)[0]
    error=(np.median(coin,axis=0)-updated.mean(0))@across
    assert abs(error)<abs(result['transverse_error_px'])
    assert not result['release_verified']


def test_missing_plane_is_not_replaced_with_a_table_constant():
    camera=Camera(np.zeros((80,100,3),np.uint8),np.array([[100.,0,50],[0,100,40],[0,0,1]]),np.eye(4))
    with pytest.raises(GeometryUnavailable):
        transverse_correction(camera,[0,0,0],[0,0,0],[50,40],[1,0],[[50,40],[50,41],[50,42]])
