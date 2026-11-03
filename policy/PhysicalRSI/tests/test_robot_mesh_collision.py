from pathlib import Path
import sys
import numpy as np
import pytest
pytest.importorskip('pinocchio')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry.robot_mesh_collision import RobotArmMeshCollision


@pytest.fixture
def arm(tmp_path):
    # The distal collision box folds into the base at q2=0, and moves away
    # at q2=pi. Base/proximal overlap is intentional adjacent-link geometry.
    path=tmp_path/'arm.urdf'
    path.write_text('''<robot name="arm">
<link name="base"><collision><geometry><box size="0.2 0.2 0.2"/></geometry></collision></link>
<link name="proximal"><collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision></link>
<link name="distal"><collision><origin xyz="-0.4 0 0"/><geometry><box size="0.1 0.1 0.1"/></geometry></collision></link>
<joint name="joint1" type="revolute"><parent link="base"/><child link="proximal"/><axis xyz="0 0 1"/><limit lower="-4" upper="4" effort="10" velocity="10"/></joint>
<joint name="joint2" type="revolute"><parent link="proximal"/><child link="distal"/><origin xyz="0.4 0 0"/><axis xyz="0 0 1"/><limit lower="-4" upper="4" effort="10" velocity="10"/></joint>
</robot>''')
    return RobotArmMeshCollision(path,['joint1','joint2'])


def test_rejects_distal_base_collision_but_not_adjacent_overlap(arm):
    assert not arm.is_collision_free([0.,0.])
    assert arm.is_collision_free([0.,np.pi])
    assert not arm.is_collision_free([.5,0.])


@pytest.mark.parametrize('joints',[[0.],[0.,np.nan]])
def test_invalid_joint_observation_is_rejected(arm,joints):
    with pytest.raises(ValueError):arm.is_collision_free(joints)
