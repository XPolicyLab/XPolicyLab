from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry.wrist_camera import ee_relative_camera
from physicalrsi_observed_geometry import GeometryUnavailable
from key_camera import edit_key_camera


def observed(translation):
    obs={'state':{},'vision':{}}
    for i,arm in enumerate(('left','right')):
        ee=np.eye(4);ee[:3,:3]=Rotation.from_euler('xyz',[.3,-.4,i*.5]).as_matrix();ee[:3,3]=np.asarray(translation)+[i*.4,0,0]
        relative=np.eye(4);relative[:3,:3]=Rotation.from_euler('y',.6).as_matrix();relative[:3,3]=[.03+i*.01,.04,.09]
        k=np.array([[120+i*40,0,32],[0,140+i*10,24],[0,0,1.]])
        obs['state'][arm+'_ee_pose']=np.r_[ee[:3,3],np.roll(Rotation.from_matrix(ee[:3,:3]).as_quat(),1)]
        obs['vision']['cam_'+arm+'_wrist']={'color':np.zeros((48,64,3),np.uint8),'intrinsic_matrix':k,'extrinsic_matrix':ee@relative}
    return obs


def test_live_relative_calibration_is_frame_invariant_and_per_observation():
    first=observed([.2,.4,.7]);translated=observed([20.,-5.,3.])
    a,ka=ee_relative_camera(first,first['state']['left_ee_pose'])
    b,kb=ee_relative_camera(translated,translated['state']['left_ee_pose'])
    np.testing.assert_allclose(a,b,atol=1e-14)
    translated['vision']['cam_left_wrist']['extrinsic_matrix'][0,3]+=.07
    translated['vision']['cam_left_wrist']['intrinsic_matrix'][0,0]+=23
    c,kc=ee_relative_camera(translated,translated['state']['left_ee_pose'])
    assert not np.allclose(a,c) and kc[0,0]==ka[0,0]+23
    again,kagain=ee_relative_camera(first,first['state']['left_ee_pose'])
    np.testing.assert_array_equal(a,again);np.testing.assert_array_equal(ka,kagain)
    ka[0,0]=0;assert first['vision']['cam_left_wrist']['intrinsic_matrix'][0,0]>0
    stale=first['state']['left_ee_pose'].copy();stale[0]+=.1
    with pytest.raises(GeometryUnavailable):ee_relative_camera(first,stale)


def test_hole_wrapper_forwards_live_calibration_without_loading_a_file():
    source="""import json, numpy as np
from pathlib import Path
cal=json.loads(Path(__file__).with_name('wrist_calibration.json').read_text())
REL=np.array(cal['ee_to_camera'])
K=np.array(cal['intrinsic'])
def centre(rgb,ee,hole):return observe(rgb,ee,hole,REL,K)
"""
    ns={'observe':lambda *args:args}
    exec(edit_key_camera('implementations/insert_key/hash/insert_key/hole_servo.py',source),ns)
    obs=observed([.1,.2,.3]);cal=ee_relative_camera(obs,obs['state']['right_ee_pose'])
    args=ns['centre']('rgb','ee','hole',calibration=cal)
    assert args[:3]==('rgb','ee','hole')
    np.testing.assert_array_equal(args[3],cal[0]);np.testing.assert_array_equal(args[4],cal[1])
    with pytest.raises(TypeError):ns['centre']('rgb','ee','hole')


def test_head_rays_use_live_pose_intrinsics_and_reject_grazing():
    from physicalrsi_observed_geometry import Camera
    source='''import numpy as np
K = np.eye(3)
KI = np.linalg.inv(K)
C = np.eye(3)
O = np.zeros(3)
def ray(uv,z):
    d = C @ (KI @ np.r_[uv, 1] * [1,-1,-1])
    return O + d * ((z-O[2])/d[2])
def perceive(rgb):return ray([32,24], .8)
'''
    ns={};exec(edit_key_camera('/insert_key/model.py',source),ns)
    assert not any(k in ns for k in ['K','KI','C','O'])
    for focal,shift,angle,height in [(140,.3,.15,.8),(240,-.2,-.1,1.2)]:
        t=np.eye(4);t[:3,:3]=Rotation.from_euler('y',angle).as_matrix();t[:3,3]=[shift,.1,2]
        k=np.array([[focal,0,32],[0,focal*.9,24],[0,0,1.]])
        camera=Camera(np.zeros((48,64,3),np.uint8),k,t)
        point=camera.unproject([[34,26]],[(2-height)/np.cos(angle)])[0]
        uv,_=camera.project([point])
        np.testing.assert_allclose(ns['ray'](uv[0],point[2],camera=camera),point,atol=1e-12)
    t=np.eye(4);t[:3,:3]=Rotation.from_euler('y',np.pi/2).as_matrix();t[2,3]=2
    camera=Camera(np.zeros((48,64,3),np.uint8),np.array([[100,0,32],[0,100,24],[0,0,1]]),t)
    with pytest.raises(GeometryUnavailable):ns['ray']([32,24],.8,camera=camera)
