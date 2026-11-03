from pathlib import Path
import sys
import numpy as np
import pytest
pytest.importorskip('pinocchio')
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry.robot_mesh_collision import RobotArmMeshCollision
from physicalrsi_observed_geometry.insert_obstacles import PlacedTubeScene
from physicalrsi_observed_geometry.finger_envelope import FingerEnvelopeScene

@pytest.fixture
def scene(tmp_path):
 mesh=tmp_path/'link.stl';trimesh.creation.box(extents=[.02,.02,.02]).export(mesh)
 links=''.join(f'<link name="{name}"><collision><geometry><mesh filename="{mesh}"/></geometry></collision></link>' for name in ['base','proximal','distal','finger'])
 joints='''<joint name="arm1" type="revolute"><parent link="base"/><child link="proximal"/><origin xyz="1 0 0"/><axis xyz="0 0 1"/><limit lower="-4" upper="4" effort="10" velocity="10"/></joint>
<joint name="arm2" type="revolute"><parent link="proximal"/><child link="distal"/><origin xyz="1 0 0"/><axis xyz="0 0 1"/><limit lower="-4" upper="4" effort="10" velocity="10"/></joint>
<joint name="finger_slide" type="prismatic"><parent link="distal"/><child link="finger"/><origin xyz="1 0 0"/><axis xyz="0 1 0"/><limit lower="0" upper="0.1" effort="10" velocity="10"/></joint>'''
 urdf=tmp_path/'robot.urdf';urdf.write_text('<robot name="test">'+links+joints+'</robot>');guard=RobotArmMeshCollision(urdf,['arm1','arm2']);bases={'left':[0,0,0,1,0,0,0]};objects={'near':(.008,.08,[3,.1,0,1,0,0,0]),'far':(.008,.08,[3,.3,0,1,0,0,0])}
 return guard,bases,objects

def test_finger_travel_rejects_collision_missed_by_arm_only(scene):
 guard,bases,objects=scene;arm=PlacedTubeScene(guard,bases,objects);full=FingerEnvelopeScene(guard,bases,objects)
 assert arm.clear(arm.path('left',[0,0],[[0,0]]),['near'])
 assert not full.clear(full.path('left',[0,0],[[0,0]]),['near'])
 assert full.clear(full.path('left',[0,0],[[0,0]]),['far'])
 assert full.envelopes[0]['upper']==.1

def test_finger_envelope_tracks_parent_arm_rotation(scene):
 guard,bases,objects=scene;full=FingerEnvelopeScene(guard,bases,objects)
 assert full.clear(full.path('left',[0,np.pi/2],[[0,np.pi/2]]),['near'])
