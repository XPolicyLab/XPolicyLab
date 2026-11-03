import importlib.util
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[1]/'runtime/PhysicalRSI_baselines/robodojo/evaluation.py'
spec=importlib.util.spec_from_file_location('native_evaluation',path)
evaluation=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluation)


def test_frozen_insert_key_alias_preserves_values_and_original_chunk():
    original={'left_gripper':[.23],'right_gripper':[.81],'left_ee_pose':list(range(7))}
    native=evaluation._native_action(original)
    assert native=={'left_ee_joint_state':[.23],'right_ee_joint_state':[.81],'left_ee_pose':list(range(7))}
    assert 'left_gripper' in original and 'left_ee_joint_state' not in original


def test_conflicting_gripper_commands_are_not_silently_overwritten():
    with pytest.raises(ValueError,match='Ambiguous'):
        evaluation._native_action({'left_gripper':[0],'left_ee_joint_state':[1]})


def test_batch_evaluator_passes_native_gripper_fields():
    class Environment:
        ended=False
        def is_episode_end(self):return self.ended
        def get_running_env_idx_list(self):return [0]
        def get_obs_batch(self,indices):return [{}]
        def take_action_batch(self,actions,indices):
            assert actions==[{'right_ee_joint_state':[.4]}]
            assert indices==[0]
            self.ended=True
    class Client:
        def call(self,func_name,**kwargs):
            if func_name=='get_action_batch':return [[{'right_gripper':[.4]}]]
    evaluation.eval_one_episode_batch(Environment(),Client())
