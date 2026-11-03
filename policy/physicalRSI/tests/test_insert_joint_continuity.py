from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry.robot import nearest_revolute_equivalents
from source import CONTINUOUS_INSERT_SOLVE


def test_equivalent_angles_preserve_rotation_and_honor_robot_limits():
    reference=np.array([.1,2.5,1.5,1.8,1.2,-.7])
    raw=reference+np.array([2,-2,2,-2,2,0])*np.pi+.05
    limits=np.array([[-10,10]]*5+[[-3.14,3.14]])
    actual=np.array(nearest_revolute_equivalents(raw,reference,limits))
    np.testing.assert_allclose(actual,reference+.05,atol=1e-12)
    for old,new in zip(raw,actual):
        np.testing.assert_allclose(Rotation.from_rotvec([0,0,old]).as_matrix(),Rotation.from_rotvec([0,0,new]).as_matrix(),atol=1e-12)
    assert np.all(actual>=limits[:,0]) and np.all(actual<=limits[:,1])
    assert nearest_revolute_equivalents([.1],[6.2],[[-.5,.5]])==[.1]
    with pytest.raises(ValueError):nearest_revolute_equivalents([2.],[0.],[[-.5,.5]])


def test_waypoints_remain_on_seeded_branch_and_delivery_uses_lift_seed():
    runtime=SimpleNamespace(observations={0:{'state':{'left_arm_joint_state':[0.]}}})
    seeds=[]
    def solve(runtime,env,arm,pose,seed):
        seeds.append(list(seed))
        # Disconnected IK branches have the same endpoint but different paths.
        return [pose[0] if seed[0]<3 else pose[0]+2*np.pi]
    ns={'_joint_target_seeded':solve};exec(CONTINUOUS_INSERT_SOLVE,ns)
    result=ns['_solve'](runtime,0,'left',{'above':[.2],'insert':[.25]},seed_joints=[6.4])
    assert seeds[0]==[6.4]
    np.testing.assert_allclose(seeds[1],result['above'])
    assert abs(result['insert'][0]-result['above'][0])<.1
