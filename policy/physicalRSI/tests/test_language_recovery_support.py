import ast
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry.finger_support_clearance import observed_finger_clearance
from classify_language_recovery_edits import replace_language_return_floor,replace_language_open_recovery
ROTATION=np.array([[0.,0,1],[0,1,0],[-1,0,0]])
ROOT=Path('/cpfs/physicalrsi-asset-audit/v4/implementations/code-skills/implementations/rdj_v1_classify_objects_by_language')


def test_finger_clearance_accounts_for_observed_slope():
    transform=SimpleNamespace(translation=np.array([.1,.2,1.]),rotation=ROTATION)
    support=SimpleNamespace(height=lambda x,y:.4+.2*y)
    assert observed_finger_clearance(transform,support)==pytest.approx(1.-.15757000232458113-(.4+.2*.24))


def test_actual_joint_return_preserves_relative_clearance_under_translation():
    paths=list(ROOT.glob('*/rdj_v1_classify_objects_by_language/model.py'))
    if not paths:pytest.skip('Release fixture unavailable')
    for path in paths:
        text=replace_language_return_floor(path.read_text())
        fn=next(n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.FunctionDef) and n.name=='joint_return')
        for height in [-.2,.765,1.3]:
            for offset,passed in [(.25,True),(.20,False)]:
                support=SimpleNamespace(height=lambda x,y:height)
                scope={'np':np,'observed_language_support':lambda ep:support,'observed_finger_clearance':observed_finger_clearance}
                exec(compile(ast.Module(body=[fn],type_ignores=[]),'actual return','exec'),scope)
                records=[];frozen=[]
                ep=SimpleNamespace(obs={'state':{'left_arm_joint_state':np.zeros(6),'right_arm_joint_state':np.zeros(6)}},initial={'left_arm_joint_state':np.full(6,.1)},branch=SimpleNamespace(fk=lambda *args:SimpleNamespace(translation=np.array([0,0,height+offset]),rotation=ROTATION)),queue=[],record=lambda **kw:records.append(kw),freeze=lambda why:frozen.append(why))
                scope['joint_return'](ep,'left','park')
                assert records[0]['passed']==passed
                assert bool(ep.queue)==passed and bool(frozen)!=passed
                assert records[0]['min_finger_clearance']==pytest.approx(offset-.15757000232458113)


def test_actual_open_recovery_has_no_absolute_height_cap_or_floor():
    paths=list(ROOT.glob('*/rdj_v1_classify_objects_by_language/open_recovery.py'))
    if not paths:pytest.skip('Release fixture unavailable')
    for path in paths:
        source=replace_language_open_recovery(path.read_text());scope={};exec(compile(source,'actual recovery','exec'),scope)
        for height in [-.2,.765,1.3]:
            for clearance,passed in [(.04,True),(.001,False)]:
                start_z=height+.15757000232458113+clearance
                q=np.zeros(6);q[0]=start_z
                state={'left_arm_joint_state':q,'right_arm_joint_state':np.zeros(6),'left_ee_pose':[0,0,start_z,1,0,0,0]}
                def solve(arm,target,previous):
                    out=previous.copy();out[0]=target[2];return out,0.,0.
                branch=SimpleNamespace(fk=lambda q,arm:SimpleNamespace(translation=np.array([0,0,q[0]]),rotation=ROTATION),solve=solve)
                actions,report=scope['plan_open_retreat'](branch,state,'left',support=SimpleNamespace(height=lambda x,y:height))
                assert report['passed']==passed
                assert report['target_z']==pytest.approx(start_z+.08)
                if passed:assert len(actions)==12
                else:assert actions is None
