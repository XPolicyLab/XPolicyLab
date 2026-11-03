"""Exercise the installed overlay's IK dispatch with an instrumented solver."""
import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

REVISION=Path(__file__).resolve().parents[1]/'asset_revision'
sys.path.insert(0,str(REVISION))
from physicalrsi_observed_geometry import GeometryUnavailable


@pytest.fixture
def frames(monkeypatch):
    # Only omit imports and the constructor requiring installed URDF assets.
    tree=ast.parse((REVISION/'overlays/deposit_robot_frames.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    namespace={'np':np,'__name__':'deposit_probe.robot_frames','__package__':'deposit_probe'}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'<real deposit overlay>','exec'),namespace)
    obj=namespace['RobotFrames'].__new__(namespace['RobotFrames'])
    obj.states={};obj.observations={};obj.supports={};obj.solvers={}
    calls=[]
    module=ModuleType('deposit_probe.policy_ik')
    class Solver:
        def __init__(self,scene):
            self.scene=scene
            calls.append(self)
        def solve_ik_to_joint(self,state,target,real_robot_pose):
            return {'state':state,'target':target,'base':real_robot_pose,'scene':self.scene}
    module.PolicyIK=Solver
    monkeypatch.setitem(sys.modules,module.__name__,module)
    return obj,calls


def test_deposit_rejects_missing_observation_before_solver(frames):
    obj,calls=frames
    with pytest.raises(GeometryUnavailable,match='calibrated RGB'):
        obj.solve_ik([0]*7,0,SimpleNamespace(arm='left'))
    assert calls==[]


def test_deposit_support_isolates_environments_and_robot_bases(frames):
    obj,calls=frames
    target=[.1,.2,.8,1.,0.,0.,0.]
    for env,height in enumerate((.61,.94)):
        obj.observations[env]={'vision':{}}
        obj.states[env]={'left_arm_joint_state':np.full(6,env)}
        obj.supports[env]=[{'dims':[.3,.2,.04],'pose':[.1,.2,height-.02,1.,0.,0.,0.]}]
        rotation=Rotation.from_euler('xyz',[.1,-.2,env*.6])
        base=np.r_[.2,-.3,.9,rotation.as_quat()[[3,0,1,2]]]
        robot=SimpleNamespace(arm='left',entity_origin_pose=base)
        result=obj.solve_ik(target,env,robot)
        local=next(iter(result['scene']['cuboid'].values()))
        np.testing.assert_allclose(rotation.apply(local['pose'][:3])+base[:3],[.1,.2,height-.02],atol=1e-12)
        assert result['target'] is target
        np.testing.assert_array_equal(result['state'],np.full(6,env))
        obj.solve_ik(target,env,robot)
        assert len(calls)==env+1
    robot.entity_origin_pose=base+np.r_[0.,0.,.1,np.zeros(4)]
    obj.solve_ik(target,1,robot)
    assert len(calls)==3
    obj.reset()
    assert not any((obj.states,obj.observations,obj.supports,obj.solvers))
