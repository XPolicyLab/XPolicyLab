from pathlib import Path
import sys, ast
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry import GeometryUnavailable
from physicalrsi_observed_geometry.wrist_camera import current_wrist_camera
from source import controller_edit


def observation():
    obs={'vision':{},'state':{}}
    for index,arm in enumerate(('left','right')):
        t=np.eye(4);t[:3,3]=[index*.3,0,2]
        obs['vision']['cam_'+arm+'_wrist']={'color':np.zeros((120,160,3),np.uint8),'intrinsic_matrix':np.array([[100+index*20,0,75],[0,110,55],[0,0,1]]),'extrinsic_matrix':t}
        obs['state'][arm+'_ee_pose']=np.array([index*.3,0,1.8,1,0,0,0])
    return obs


@pytest.mark.parametrize('arm',['left','right'])
def test_live_calibration_and_quaternion_sign(arm):
    obs=observation();pose=obs['state'][arm+'_ee_pose'].copy();pose[3:]*=-1
    c=current_wrist_camera(obs,pose)
    np.testing.assert_array_equal(c.k,obs['vision']['cam_'+arm+'_wrist']['intrinsic_matrix'])
    assert (c.width,c.height)==(160,120)
    obs['vision']['cam_'+arm+'_wrist']['extrinsic_matrix'][0,3]+=.1
    c=current_wrist_camera(obs,pose)
    np.testing.assert_array_equal(c.transform,obs['vision']['cam_'+arm+'_wrist']['extrinsic_matrix'])


def test_stale_ambiguous_and_missing_calibration_rejected():
    obs=observation();pose=obs['state']['left_ee_pose'].copy();pose[0]+=.02
    with pytest.raises(GeometryUnavailable,match='stale'):current_wrist_camera(obs,pose)
    obs['state']['right_ee_pose']=obs['state']['left_ee_pose'].copy()
    with pytest.raises(GeometryUnavailable,match='uniquely'):current_wrist_camera(obs,obs['state']['left_ee_pose'])
    obs=observation();del obs['vision']['cam_left_wrist']['intrinsic_matrix']
    with pytest.raises(GeometryUnavailable,match='calibration'):current_wrist_camera(obs,obs['state']['left_ee_pose'])


def test_transformed_hull_uses_live_projection():
    import cv2
    text='''import numpy as np, cv2
F=500
MOUNT_T=np.zeros(3)
MOUNT_R=np.eye(3)
class WristGeometry:
 def hull(self,xy,ee,rank):raise AssertionError('legacy hull')
'''
    edited=controller_edit('/nesting_rgb/wrist_geometry.py',text,None)
    ns={};exec(edited,ns);g=ns['WristGeometry']();g.calibrated_obs=observation()
    g.points=[np.array([[-.1,-.1,0],[.1,-.1,0],[.1,.1,0],[-.1,.1,0]])]
    assert all(k not in ns for k in ('F','MOUNT_T','MOUNT_R'))
    ee=g.calibrated_obs['state']['right_ee_pose'];xyz=[.3,0,.9]
    got=g.hull(xyz,ee,0)
    cam=current_wrist_camera(g.calibrated_obs,ee);uv,_=cam.project(g.points[0]+xyz)
    np.testing.assert_allclose(cv2.boundingRect(got),cv2.boundingRect(uv.astype('float32')),atol=0)
    g.calibrated_obs=None
    with pytest.raises(GeometryUnavailable):g.hull(xyz,ee,0)
