"""Recover robot placement without a fixed world mounting pose."""
import ast
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('robot_geometry',ROOT/'runtime/PhysicalRSI_baselines/robodojo/skills/geometry.py')
geometry=importlib.util.module_from_spec(spec);spec.loader.exec_module(geometry)
tree=ast.parse((ROOT/'asset_revision/overlays/insert_ik.py').read_text())
namespace={'np':np,**{name:getattr(geometry,name) for name in ('matrix','pose','forward_kinematics')}}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='observed_base'],type_ignores=[]),'<insert-base>','exec'),namespace)


@pytest.mark.parametrize('arm,position,angles',[('left',[.12,.8,.56],[.2,-.1,.7]),('right',[-.55,.2,1.1],[-.1,.3,-1.])])
def test_robot_base_follows_observed_rigid_placement(arm,position,angles):
    chain=[]
    for i in range(6):
        origin=np.eye(4);origin[:3,3]=[.02*(i%2),0,.06]
        chain.append((origin,np.eye(3)[i%3]))
    joints=np.array([.1,-.4,.2,.7,-.2,.5])
    base=np.eye(4);base[:3,3]=position;base[:3,:3]=Rotation.from_euler('xyz',angles).as_matrix()
    observed=geometry.forward_kinematics(chain,geometry.pose(base),joints)
    state={arm+'_ee_pose':observed,arm+'_arm_joint_state':joints}
    actual=namespace['observed_base'](chain,arm,state)
    np.testing.assert_allclose(geometry.matrix(actual),base,atol=1e-12)
    state[arm+'_ee_pose'][2]+=0.25
    shifted=geometry.matrix(namespace['observed_base'](chain,arm,state))
    np.testing.assert_allclose(shifted[:3,3],np.asarray(position)+[0,0,.25],atol=1e-12)


def test_missing_or_nonfinite_proprioception_has_no_fixed_mount_fallback():
    with pytest.raises(KeyError):namespace['observed_base']([], 'left', {})
    with pytest.raises(ValueError):namespace['observed_base']([], 'left', {'left_ee_pose':[0,0,np.nan,1,0,0,0],'left_arm_joint_state':[0]*6})
