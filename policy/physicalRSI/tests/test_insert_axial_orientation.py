"""Circular insertion search must preserve the measured opening axis and position."""
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from source import SLOT_ORIENTATIONS


def test_axial_search_preserves_tilted_slot_geometry_and_input():
    def rotation(q):return Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix()
    def quaternion(r):return Rotation.from_matrix(r).as_quat()[[3,0,1,2]].tolist()
    ns={'np':np,'_rotation':rotation,'_quaternion':quaternion}
    exec(SLOT_ORIENTATIONS,ns)
    r=Rotation.from_euler('xyz',[.23,-.17,1.3]).as_matrix()
    slot=[.18,-.07,.91]+quaternion(r);original=slot.copy()
    poses=list(ns['_slot_orientations'](slot))
    assert slot==original and len(poses)==8
    for p in poses:
        np.testing.assert_allclose(p[:3],slot[:3])
        np.testing.assert_allclose(rotation(p[3:])[:,2],r[:,2],atol=1e-12)
    assert len({tuple(np.round(rotation(p[3:])[:,0],6)) for p in poses})==8
