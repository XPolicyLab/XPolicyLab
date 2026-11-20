"""Expected exhausted skills reach native termination; bugs remain visible."""
from types import SimpleNamespace
import unittest

import numpy as np

from skills.shared.motion.public_dense_execution.implementation import NativeEpisodeEnded
from tasks.sort_nesting_dolls_by_size.bindings.domain_failure import HANDLED_REFUSALS, run_with_native_final_hold
from tasks.sort_nesting_dolls_by_size.skills.domain_refusal.implementation import (
    HomePlanningExhausted, PerceptionEvidenceRejected, RefreshPerceptionAmbiguous, TransferPlanRejected,
)
from tasks.sort_nesting_dolls_by_size.skills.scene_recovery.implementation import SceneRecoveryMixin
from tasks.sort_nesting_dolls_by_size.skills.row_execution.implementation import RowExecutionMixin
from tasks.sort_nesting_dolls_by_size.skills.atomic_transfer.implementation import AtomicTransferMixin


def fail_with(error):
    raise error


def episode(error=None):
    env = SimpleNamespace(task_name='sort_nesting_dolls_by_size', num_envs=1,
        step_lim=1050, dt=.004, obs_manager=SimpleNamespace(collect_freq=25, collect_interval=10),
        take_action_cnt=[1047], end_flag=[False])
    actor = SimpleNamespace(phase='existing_skill', run=lambda: fail_with(error) if error else None)
    calls = []

    def execute(controls, phase):
        calls.append((controls, phase))
        for command in controls:
            if command != {}:
                raise AssertionError('Final hold must retain the last command')
            env.take_action_cnt[0] += 1
            if env.take_action_cnt[0] == env.step_lim:
                env.end_flag[0] = True
                raise NativeEpisodeEnded('Public terminal acknowledgement')

    actor._official_bridge = SimpleNamespace(execute=execute)
    return actor, env, calls


class NestingOfficialCompletionTests(unittest.TestCase):
    def test_reviewed_exhaustions_preserve_reason_then_hold_to_native_end(self):
        for exception in HANDLED_REFUSALS:
            with self.subTest(exception=exception.__name__):
                actor, env, calls = episode(exception('original reason'))
                refusal = run_with_native_final_hold(actor, env)
                self.assertEqual(refusal, actor._integrated_source_refusal)
                self.assertEqual(refusal['message'], 'original reason')
                self.assertEqual(refusal['action_count_at_refusal'], 1047)
                self.assertEqual(refusal['phase_at_refusal'], 'existing_skill')
                self.assertEqual(calls, [([{}] * 10, 'official_final_hold')])
                self.assertEqual(env.take_action_cnt, [1050])
                self.assertEqual(env.end_flag, [True])

    def test_completed_motion_still_requires_native_termination(self):
        actor, env, calls = episode()
        self.assertIsNone(run_with_native_final_hold(actor, env))
        self.assertEqual(env.end_flag, [True])
        self.assertEqual(len(calls), 1)

    def test_unknown_bugs_and_unreviewed_internal_rejections_are_not_masked(self):
        class Unreviewed(RefreshPerceptionAmbiguous):
            pass
        for error in [KeyError('files'), TypeError('bad reply'), RuntimeError('backend'),
                      PerceptionEvidenceRejected('internal'), TransferPlanRejected('internal'),
                      Unreviewed('new type')]:
            with self.subTest(error=type(error).__name__):
                actor, env, calls = episode(error)
                with self.assertRaises(type(error)):
                    run_with_native_final_hold(actor, env)
                self.assertEqual(calls, [])
                self.assertEqual(env.take_action_cnt, [1047])

    def test_transport_failure_is_not_a_scored_domain_outcome(self):
        actor, env, _ = episode(RefreshPerceptionAmbiguous('occluded'))
        actor._official_bridge.execute = lambda *args: fail_with(ConnectionError('disconnected'))
        with self.assertRaises(ConnectionError):
            run_with_native_final_hold(actor, env)
        self.assertEqual(env.end_flag, [False])

    def test_terminal_signal_requires_actual_public_terminal_flag(self):
        actor, env, calls = episode(NativeEpisodeEnded('terminal'))
        with self.assertRaisesRegex(RuntimeError, 'actual native termination'):
            run_with_native_final_hold(actor, env)
        env.end_flag[0] = True
        self.assertIsNone(run_with_native_final_hold(actor, env))
        self.assertEqual(calls, [])

    def test_wrong_clock_is_rejected_before_issuing_commands(self):
        actor, env, calls = episode()
        env.step_lim = 100
        with self.assertRaises(ValueError):
            run_with_native_final_hold(actor, env)
        self.assertEqual(calls, [])

    def test_original_refresh_candidates_exhaust_into_typed_task_failure(self):
        attempts = []
        pose = np.array([0., 0., 1., 1., 0., 0., 0.])
        robots = {name: SimpleNamespace(arm_name=name) for name in ['left_arm', 'right_arm']}
        context = SimpleNamespace(dolls=[{'rank': 0, 'center': [0., 0., 1.]}], robots=robots,
            args=SimpleNamespace(gripper_forward_offset=.145),
            public_project=lambda points: np.zeros((len(points), 2)),
            _actual_ee_pose=lambda robot: pose.copy(), public_open_finger_spheres=lambda robot: [],
            projected_finger_occlusion=lambda *args: {'score': 0},
            _robot_base_pose=lambda robot: pose, lateral_view_targets=lambda *args, **kwargs: [],
            _configure_obstacles=lambda rank: None, _events=[])

        def reject(robot, target, mode):
            attempts.append((robot.arm_name, float(target[2]), mode))
            raise TransferPlanRejected('original collision rejection')

        context._plan_constrained = reject
        with self.assertRaisesRegex(RefreshPerceptionAmbiguous, 'single-arm view clearing'):
            SceneRecoveryMixin._recover_refresh_view(context, PerceptionEvidenceRejected('rank 0 occluded'))
        self.assertEqual([v[0] for v in attempts], ['left_arm'] * 3 + ['right_arm'] * 3)
        np.testing.assert_allclose([v[1] for v in attempts], [1.08, 1.16, 1.24] * 2)

    def test_home_exhaustion_keeps_cause_and_restores_home_scope(self):
        original = TransferPlanRejected('all original home paths rejected')
        context = SimpleNamespace(configure_rgb_obstacles=lambda rank: None,
            _return_native_both_home=lambda: fail_with(original))
        with self.assertRaises(HomePlanningExhausted) as caught:
            RowExecutionMixin._return_both_home(context)
        self.assertIs(caught.exception.__cause__, original)
        self.assertFalse(context._require_native_home)

    def test_deferred_home_rejection_keeps_arm_and_issues_no_new_motion(self):
        original = TransferPlanRejected('native home unavailable')
        context = SimpleNamespace(_deferred_home_arm='left_arm', robots={'left_arm': object()},
            configure_rgb_obstacles=lambda rank: None, _events=[],
            _execute=lambda *args: self.fail('Rejected home must not issue motion'),
            _plan_joint_home=lambda robot: fail_with(original))
        with self.assertRaises(HomePlanningExhausted):
            AtomicTransferMixin._complete_deferred_home(context)
        self.assertEqual(context._deferred_home_arm, 'left_arm')
        self.assertEqual(context._events[0]['reason'], str(original))

    def test_home_backend_bugs_remain_unexpected(self):
        context = SimpleNamespace(configure_rgb_obstacles=lambda rank: None,
            _return_native_both_home=lambda: fail_with(KeyError('planner reply')))
        with self.assertRaises(KeyError):
            RowExecutionMixin._return_both_home(context)
        self.assertFalse(context._require_native_home)


if __name__ == '__main__':
    unittest.main()
