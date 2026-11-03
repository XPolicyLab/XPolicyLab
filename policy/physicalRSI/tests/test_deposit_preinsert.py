from pathlib import Path
import sys
from copy import deepcopy
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry import GeometryUnavailable
from physicalrsi_observed_geometry.slot_servo import stage_window
from source_edits import DEPOSIT_NEAR_FEEDBACK, deposit_preinsert_feedback


def plan():
    return {'context':{'delivery_bound':True,'bank_arm':'right','measured_coin_local':[.1,0,0],
                       'slot_poses':[[0,0,.9,1,0,0,0]],'slot_index':0},
            'stages':[{'name':name,'steps':steps,'target':{'original':name}}
                      for name,steps in [('pickup',100),('slot_align',15),('slot_insert',10),('release',10)]]}


def scope(fail_ik=False):
    calls=[]
    def target(frames,index,arm,pose):
        calls.append((index,pose))
        return None if fail_ik and len(calls)==2 else pose[:3]
    env={'np':np,'stage_window':stage_window,'GeometryUnavailable':GeometryUnavailable,
         '_ee_pose_to_semantic':lambda robot,pose:pose,'_rotation':lambda q:np.eye(3),
         'near_slot_correction':lambda *args:{'translation':np.array([.002,0,0]),'transverse_error_px':8},
         '_rgb_bank_waypoints':lambda *args:{'slot_align':[0,0,.92,1,0,0,0],'slot_insert':[0,0,.9,1,0,0,0]},
         '_joint_target':target,'_target':lambda old,arm_joints:{**old,'arm_joints':arm_joints}}
    exec(DEPOSIT_NEAR_FEEDBACK,env)
    return env,calls

class Frames:
    def get_robot_by_arm_name(self,name):return name

OBS={'state':{'right_ee_pose':[0,0,1,1,0,0,0]}}

@pytest.mark.parametrize('action',[109,115,125,135])
def test_near_feedback_cannot_run_before_window_during_descent_or_after_release(action):
    env,calls=scope();p=plan();before=deepcopy(p)
    env['_near_insert_feedback'](Frames(),3,p,OBS,action)
    assert p==before and not calls


def test_commit_once_and_keep_environment_plans_independent():
    env,calls=scope();a=plan();b=plan();before=deepcopy(b)
    for action in [110,111,114]:env['_near_insert_feedback'](Frames(),3,a,OBS,action)
    assert len(calls)==2 and all(index==3 for index,_ in calls)
    assert b==before
    assert a['stages'][1]['target']['arm_joints']['right']==[.002,0,.92]
    assert a['stages'][2]['target']['arm_joints']==a['stages'][3]['target']['arm_joints']
    assert a['stages'][0]==b['stages'][0]


def test_failed_second_ik_does_not_commit_half_a_plan():
    env,calls=scope(True);p=plan();before=deepcopy(p)
    env['_near_insert_feedback'](Frames(),3,p,OBS,110)
    assert p==before and len(calls)==2


def test_missing_geometry_keeps_plan_and_does_not_invent_a_correction():
    env,calls=scope();p=plan();before=deepcopy(p)
    def missing(*args):raise GeometryUnavailable('occluded')
    env['near_slot_correction']=missing
    env['_near_insert_feedback'](Frames(),3,p,OBS,110)
    assert p==before and not calls


def test_half_open_stage_intervals_cover_transition_boundaries():
    stages=plan()['stages']
    assert stage_window(stages,'slot_insert')==(115,125)
    assert stage_window(stages,'release')==(125,135)
    assert stage_window(stages,'missing')==(135,135)
