from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry import Camera,GeometryUnavailable
from physicalrsi_observed_geometry.delivery_observability import centerline_condition
from source_edits import SLOT_ORIENTATIONS


def camera(position):
 t=np.eye(4);t[:3,3]=position
 return Camera(np.zeros((800,800,3),np.uint8),np.array([[100.,0,400],[0,100.,400],[0,0,1.]]),t)


def test_coplanar_views_rejected_and_lateral_baseline_resolves_axis():
 head=camera([0,-1,2]);same_plane=camera([0,-.2,1.2]);lateral=camera([.3,-.2,1.2])
 assert centerline_condition([head,same_plane],[0,0,1],[0,0,1],.1)<.05
 assert centerline_condition([head,lateral],[0,0,1],[0,0,1],.1)>.075


def test_behind_camera_and_invalid_axis_cannot_pass():
 views=[camera([0,-1,2]),camera([.3,-.2,1.2])]
 assert centerline_condition(views,[0,0,3],[0,0,1],.1)==0
 with pytest.raises(GeometryUnavailable):centerline_condition(views,[0,0,1],[0,0,0],.1)


def test_eight_axial_poses_preserve_opening_center_and_axis():
 from scipy.spatial.transform import Rotation
 namespace={'np':np,'_rotation':lambda q:Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix(),
 '_quaternion':lambda r:Rotation.from_matrix(r).as_quat()[[3,0,1,2]].tolist()}
 exec(SLOT_ORIENTATIONS,namespace)
 q=Rotation.from_rotvec([.2,.1,-.3]).as_quat()[[3,0,1,2]];pose=[.1,-.2,.9,*q];variants=list(namespace['_slot_orientations'](pose));assert len(variants)==8
 axis=namespace['_rotation'](q)[:,2]
 for p in variants:
  np.testing.assert_allclose(p[:3],pose[:3]);np.testing.assert_allclose(namespace['_rotation'](p[3:])[:,2],axis,atol=1e-12)
 assert len({tuple(np.round(p[3:],8)) for p in variants})==8
