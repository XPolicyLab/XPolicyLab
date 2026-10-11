import copy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from conveyor_reward_feedback import ConveyorRewardFeedback, attach_command_feedback, enabled


def environment():
    def forbidden(*args, **kwargs):
        raise AssertionError('Mutating evaluator function invoked')
    rm = SimpleNamespace(check_list=[[[
        ('is_A_on_B_bottom', dict(label_A='target', label_B='basket', min_z_gap=-.01, max_z_gap=.04)),
        ('is_lift', dict(label='target', z_threshold=.08))]]],
        score_meta=[{'mode': None}], final_score_meta=[{'mode': None}],
        final_check_list=[[]], trigger_check_list=[[]], query_list=[[]], trigger_query_list=[[]],
        get_reward=forbidden, get_score=forbidden, step=forbidden, check_once=forbidden)
    return SimpleNamespace(reward_manager=rm, take_action_cnt=[0], step_lim=700,
                           success=[True], end_flag=[False], is_episode_end=forbidden)


class FeedbackTests(unittest.TestCase):
    def test_default_and_benchmark_off(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(enabled())
        with patch.dict(os.environ, {'ROBODAWN_CONVEYOR_REWARD_FEEDBACK': '1', 'ROBODAWN_RUN_MODE': 'benchmark'}):
            self.assertFalse(enabled())
        with patch.dict(os.environ, {'ROBODAWN_CONVEYOR_REWARD_FEEDBACK': '1', 'ROBODAWN_RUN_MODE': 'collection'}):
            self.assertTrue(enabled())

    def test_readonly_baseline_delta_latching_and_action_order(self):
        with TemporaryDirectory() as root:
            env = environment()
            before = copy.deepcopy(vars(env.reward_manager))
            adapter = ConveyorRewardFeedback(env, root)
            baseline = adapter.snapshot(env, 'baseline', 'initial')
            self.assertEqual(before, vars(env.reward_manager))
            self.assertEqual(baseline['current_score'], 0)
            self.assertIsNone(baseline['delta_score'])
            self.assertFalse(baseline['native_success'])  # success=True at initialization is not success
            session, report = {'conveyor_reward_feedback': adapter}, {'command_results': []}
            for step in (12, 12, 30):  # includes zero-step IK failure
                env.take_action_cnt[0] = step
                attach_command_feedback(env, session, report, {'command': 'probe', 'note': 'local correction'})
            self.assertEqual([r['privileged_reward_feedback']['native_step'] for r in report['command_results']], [12, 12, 30])
            self.assertEqual(report['privileged_reward_feedback']['delta_score'], 0)
            env.take_action_cnt[0] = 40
            env.end_flag[0] = True
            env.reward_manager.check_list[0] = []  # native update, not adapter evaluation
            packet = adapter.snapshot(env, 'terminal', 'native')
            self.assertEqual(packet['delta_score'], 1)
            self.assertTrue(packet['phase_conditions']['native_and_group']['historical'])
            self.assertIsNone(packet['phase_conditions']['native_and_group']['current'])
            self.assertEqual(packet['remaining_steps'], 660)
            self.assertEqual(len(Path(root, 'conveyor_reward_feedback.jsonl').read_text().splitlines()), 5)
            self.assertTrue(json.loads(Path(root, 'conveyor_reward_feedback_metadata.json').read_text())['uses_privileged_reward_feedback'])

    def test_failure_at_cap(self):
        with TemporaryDirectory() as root:
            env = environment()
            adapter = ConveyorRewardFeedback(env, root)
            adapter.snapshot(env, 'baseline', 'initial')
            env.take_action_cnt[0], env.end_flag[0], env.success[0] = 700, True, False
            packet = adapter.snapshot(env, 'end', 'native_budget')
            self.assertEqual((packet['current_score'], packet['delta_score'], packet['remaining_steps']), (0, 0, 0))

    def test_unregistered_or_new_scoring_is_unknown(self):
        for variant in ('empty', 'process_score', 'late_attach'):
            with self.subTest(variant=variant), TemporaryDirectory() as root:
                env = environment()
                if variant == 'empty':
                    env.reward_manager.check_list = [[]]
                elif variant == 'process_score':
                    env.reward_manager.score_meta[0]['mode'] = 'transition'
                else:
                    env.take_action_cnt[0] = 10
                packet = ConveyorRewardFeedback(env, root).snapshot(env, 'initial', variant)
                self.assertIsNone(packet['current_score'])
                self.assertIsNone(packet['native_success'])

    def test_reset_rejected(self):
        with TemporaryDirectory() as root:
            env = environment()
            adapter = ConveyorRewardFeedback(env, root)
            adapter.snapshot(env, 'initial', 'initial')
            env.take_action_cnt[0] = 10
            adapter.snapshot(env, 'next', 'next')
            env.take_action_cnt[0] = 0
            with self.assertRaises(ValueError):
                adapter.snapshot(env, 'reset', 'reset')


if __name__ == '__main__':
    unittest.main(verbosity=2)
