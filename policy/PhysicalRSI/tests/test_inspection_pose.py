from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry import GeometryUnavailable
from physicalrsi_observed_geometry.inspection_pose import conditioned_wrist_pose

def scene(x=0.):
 k=np.array([[100.,0,400],[0,100.,400],[0,0,1]])
 head=np.eye(4);head[:3,3]=[0,-1,2]
 wrist=np.eye(4);wrist[:3,3]=[x,-.2,1.2]
 pose=np.array([x,-.2,1.1,1.,0,0,0])
 obs={'vision':{n:{'color':np.zeros((800,800,3),np.uint8),'intrinsic_matrix':k,'extrinsic_matrix':t} for n,t in [('cam_head',head),('cam_left_wrist',wrist)]},'state':{'left_ee_pose':pose}}
 return obs,pose,np.array([-x,.2,-.1])

def test_degenerate_view_gets_bounded_orientation_only_change():
 obs,pose,root=scene();result=conditioned_wrist_pose(obs,'left',pose,root,[0,0,1],.1)
 assert result is not None
 np.testing.assert_array_equal(result['pose'][:3],pose[:3])
 assert result['condition_before']<.05 and result['condition_after']>=.075
 from scipy.spatial.transform import Rotation
 rotation=Rotation.from_quat(np.array(result['pose'])[[4,5,6,3]])
 assert rotation.magnitude()<=.300001
 np.testing.assert_array_equal(obs['state']['left_ee_pose'],pose)

def test_resolved_view_is_unchanged():
 obs,pose,root=scene(.3)
 assert conditioned_wrist_pose(obs,'left',pose,root,[0,0,1],.1) is None

def test_unknown_geometry_is_rejected():
 obs,pose,root=scene()
 with pytest.raises(GeometryUnavailable):conditioned_wrist_pose(obs,'left',pose,root,[0,0,0],.1)
