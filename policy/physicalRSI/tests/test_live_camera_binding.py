import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

REVISION=Path(__file__).resolve().parents[1]/'asset_revision'
sys.path.insert(0,str(REVISION))
from physicalrsi_observed_geometry import GeometryUnavailable
spec=importlib.util.spec_from_file_location('live_camera_binding',REVISION/'overlays/live_camera_binding.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def observation(offset):
    vision={}
    for index,name in enumerate(('cam_head','cam_left_wrist','cam_right_wrist')):
        transform=np.eye(4);transform[:3,3]=np.array([index*.2,-.3,1.4])+offset
        vision[name]={'color':np.zeros((24,32,3),np.uint8),
                      'intrinsic_matrix':np.array([[80.+index,0,16],[0,90.+index,12],[0,0,1]]),
                      'extrinsic_matrix':transform}
    return {'vision':vision,'state':{},'instruction':'test','env_idx':3}


@pytest.mark.parametrize('offset',[[0,0,0],[2,-1,.7]])
def test_current_calibration_is_preserved_for_every_camera(offset):
    obs=observation(offset)
    originals={name:{key:value.copy() for key,value in camera.items()} for name,camera in obs['vision'].items()}
    assert module.bind_camera_calibration(obs) is obs
    for name,camera in obs['vision'].items():
        for key,value in camera.items():np.testing.assert_array_equal(value,originals[name][key])


def test_missing_live_calibration_does_not_fall_back_to_saved_file():
    obs=observation([0,0,0]);del obs['vision']['cam_left_wrist']['extrinsic_matrix']
    with pytest.raises(GeometryUnavailable):module.bind_camera_calibration(obs)
