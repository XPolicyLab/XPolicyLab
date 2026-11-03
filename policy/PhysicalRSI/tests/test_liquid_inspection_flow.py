import ast,json,runpy,sys,types
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]/'asset_revision'
sys.path.insert(0,str(ROOT))
from physicalrsi_observed_geometry import GeometryUnavailable

def inspection_type():
    tree=ast.parse((ROOT/'overlays/pour_liquid_inspection.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    ns={'np':np,'Path':Path,'json':json,'GeometryUnavailable':GeometryUnavailable,
        'KEYS':('left_ee_pose','right_ee_pose','left_ee_joint_state','right_ee_joint_state'),
        'pose_error':lambda a,b:(float(np.linalg.norm(np.array(a[:3])-b[:3])),0.),'bounded_pose':lambda a,b:list(b)}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'actual inspection','exec'),ns)
    return ns['CupInspection']

def observation():
    return {'env_idx':0,'state':{**{a+'_ee_pose':[0,0,1,1,0,0,0] for a in ['left','right']},
        **{a+'_ee_joint_state':[1] for a in ['left','right']},**{a+'_arm_joint_state':[0]*6 for a in ['left','right']}}}

def test_actual_inspection_return_handoff_and_joint_branch_guard(tmp_path):
    cls=inspection_type();obj=cls(tmp_path);obs=observation()
    obj.phase='return';obj.plan={'arm':'right'};obj.poses=[obs['state']['right_ee_pose']];obj.joints=[[0]*6];obj.index=0
    assert obj.step(obs) is None and obj.phase=='complete'
    other=cls(tmp_path/'other');other.phase='view';other.plan=obj.plan;other.poses=obj.poses;other.joints=[[1]*6];other.index=0
    action=other.step(obs)
    assert other.phase=='failed' and action[0]['right_ee_pose']==obs['state']['right_ee_pose']
    assert other.opening is None

def test_opening_binding_replaces_mouth_xyz_and_radius_in_rotated_frame():
    ns=runpy.run_path(str(ROOT/'overlays/pour_perception.py'))
    rotation=Rotation.from_euler('z',.7).as_matrix();transform=np.eye(4);transform[:3,:3]=rotation;transform[:3,3]=[.2,-.1,.8]
    center=np.array([.24,-.12,.92]);geometry={'cup_pose':ns['pose'](transform),'cup_vertices':[[x,y,z] for x in [-.03,.03] for y in [-.03,.03] for z in [-.04,.04]],'cup_outer_radius':.04}
    ns['bind_liquid_opening'](geometry,{'center':center,'radius':.025})
    np.testing.assert_allclose(rotation@geometry['cup_center_local']+transform[:3,3],center)
    assert max(v[2] for v in geometry['cup_vertices'])==pytest.approx(geometry['cup_center_local'][2])
    assert geometry['vessel_radius']==.025
    with pytest.raises(GeometryUnavailable):ns['bind_liquid_opening'](geometry,None)

def test_batch_wrapper_keeps_inspecting_and_ready_environments_separate(monkeypatch,tmp_path):
    tree=ast.parse((ROOT/'overlays/pour_liquid_controller.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ObservedCupModel')
    class Base:
        def __init__(self,cfg):self.out=tmp_path
        def reset(self):pass
        def update_obs_batch(self,obs):self.ready=[o['env_idx'] for o in obs]
        def get_action_batch(self,ids):assert ids==self.ready;return [[{'ready':i}] for i in ids]
    class Inspect:
        def __init__(self,out):pass
        def step(self,o):return [{'inspecting':o['env_idx']}] if o['env_idx']==2 else None
    module=types.ModuleType('inspection_test.cup_inspection');module.CupInspection=Inspect
    monkeypatch.setitem(sys.modules,'inspection_test.cup_inspection',module)
    ns={'StructuredModel':Base,'__package__':'inspection_test'};exec(compile(ast.Module(body=[cls],type_ignores=[]),'actual wrapper','exec'),ns)
    model=ns['ObservedCupModel']({});model.update_obs_batch([{'env_idx':2},{'env_idx':7}])
    assert model.get_action_batch([7,2])==[[{'ready':7}],[{'inspecting':2}]]
    with pytest.raises(ValueError):model.get_action_batch([2])
    model.reset();assert model.inspections=={}


def test_dense_qualified_view_executes_bounded_rotation_without_station_stops(monkeypatch,tmp_path):
    from scipy.spatial.transform import Slerp
    from math import acos
    from physicalrsi_observed_geometry import Camera
    fixture=Path('/cpfs/physicalrsi-asset-audit/v4/implementations/code-skills/implementations/rdj_pour/2a7b80f914e85dd1/rdj_pour/controller.py')
    if not fixture.exists():pytest.skip('Release controller fixture not installed')
    tree=ast.parse(fixture.read_text());helpers=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('pose_error','bounded_pose')]
    scope={'np':np,'acos':acos,'Rotation':Rotation,'Slerp':Slerp}
    exec(compile(ast.Module(body=helpers,type_ignores=[]),'released controller limits','exec'),scope)
    cls=inspection_type();globals_=cls.step.__globals__;globals_.update(scope)
    obs=observation();obs['vision']={'cam_head':{'color':np.zeros((16,16,3),np.uint8),'intrinsic_matrix':np.eye(3),'extrinsic_matrix':np.eye(4)}}
    camera=Camera.from_observation(obs);scene=SimpleNamespace(camera=camera,mask=lambda m:m,depth_calibration_report={'observed_plane':[0,0,.7]})
    start=obs['state']['right_ee_pose'];poses=[start[:3]+Rotation.from_rotvec([0,0,angle]).as_quat()[[3,0,1,2]].tolist() for angle in np.linspace(0,.68,17)[1:]]
    plan={'arm':'right','poses':poses,'joints':[[0]*6]*16,'original_pose':start,'original_joints':[0]*6}
    globals_.update(Camera=Camera,calibrated_depth_scene=lambda o:scene,
        segment_many_http=lambda *a,**kw:[SimpleNamespace(masks=[np.ones((16,16),bool)])],
        support_point=lambda *a,**kw:np.array([0,0,.7]),qualify_view=lambda *a:plan,
        observed_liquid_cup_opening=lambda *a:{'center':np.array([0,0,.8]),'radius':.03})
    module=types.ModuleType('rdj_insert_policy.ik');module.X5IK=lambda:object()
    monkeypatch.setitem(sys.modules,'rdj_insert_policy.ik',module)
    obj=cls(tmp_path);angles=[]
    for tick in range(30):
        action=obj.step(obs)
        if action is None:break
        old=np.array(obs['state']['right_ee_pose']);new=np.array(action[0]['right_ee_pose'])
        angles.append(scope['pose_error'](old,new)[1]);obs['state'].update(action[0])
    assert obj.phase=='complete' and obj.opening is not None
    assert obj.calls<=20 and max(angles)<=.12+1e-8
    np.testing.assert_allclose(obs['state']['right_ee_pose'],start,atol=1e-8)
    assert len(plan['poses'])==16  # Qualification density stays intact.
