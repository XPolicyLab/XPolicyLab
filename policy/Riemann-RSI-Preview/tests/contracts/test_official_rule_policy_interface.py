"""Nine bounded policy-side tests, including the real official WS/deploy loop."""
from copy import deepcopy
import asyncio
import importlib
from pathlib import Path
import sys
import threading
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rsi.deployment import locations
SDK = Path(locations()['official_root'])
sys.path[:0] = [str(SDK), str(SDK / 'XPolicyLab')]
from skills.shared.orchestration.public_rule_policy_interface.implementation import (
    BlockingRulePolicy, RulePolicyFailure, normalized_action,
)


def observation(value=0.0):
    state = {side + '_arm_joint_state': np.full(6, value, np.float32) for side in ('left', 'right')}
    state.update({side + '_ee_joint_state': np.array([0.5], np.float32) for side in ('left', 'right')})
    state.update({side + '_ee_pose': np.array([0, 0, 0, 1, 0, 0, 0], np.float32) for side in ('left', 'right')})
    return {
        'state': state,
        'vision': {camera: {'color': np.full((3, 4, 3), [11, 23, 47], np.uint8), 'depth': 'not-public'}
                   for camera in ('cam_head', 'cam_left_wrist', 'cam_right_wrist')},
        'layout': 'not-public', 'reward': 100, 'env_idx': 0,
    }


def action(value):
    return normalized_action({key: data for key, data in observation(value)['state'].items() if not key.endswith('_pose')})


class Interface(unittest.TestCase):
    def make_policy(self, runner):
        policy = BlockingRulePolicy(runner, wait_timeout=2, join_timeout=1)
        self.addCleanup(policy.close)
        policy.reset()
        return policy

    def test_01_exactly_one_command_gets_the_next_current_frame(self):
        received = []
        def runner(initial, exchange):
            received.append(initial['state']['left_arm_joint_state'][0])
            received.append(exchange(action(.1))['state']['left_arm_joint_state'][0])
            exchange(action(float(received[-1]) + .1))
        p = self.make_policy(runner)
        p.update_obs(observation(.2))
        self.assertAlmostEqual(p.get_action()[0]['left_arm_joint_state'][0], .1)
        p.update_obs(observation(.4))
        self.assertAlmostEqual(p.get_action()[0]['left_arm_joint_state'][0], .5)
        np.testing.assert_allclose(received, [.2, .4])
        self.assertEqual(p.diagnostics()['actions_returned'], 2)

    def test_02_no_action_replay_without_a_new_observation(self):
        p = self.make_policy(lambda initial, exchange: exchange(action(.1)))
        with self.assertRaisesRegex(RuntimeError, 'fresh official observation'):
            p.get_action()
        p.update_obs(observation())
        p.get_action()
        with self.assertRaisesRegex(RuntimeError, 'fresh official observation'):
            p.get_action()
        self.assertEqual(p.diagnostics()['actions_returned'], 1)

    def test_03_normal_completion_keeps_last_finite_target_until_official_end(self):
        p = self.make_policy(lambda initial, exchange: exchange(action(.3)))
        p.update_obs(observation())
        expected = p.get_action()
        for _ in range(3):
            p.update_obs(observation(.6))
            self.assertEqual(p.get_action(), expected)
        self.assertTrue(p.diagnostics()['workflow_completed'])
        self.assertFalse(p.diagnostics()['official_success_claimed'])

    def test_04_no_command_completion_uses_only_public_initial_targets(self):
        p = self.make_policy(lambda initial, exchange: None)
        p.update_obs(observation(.4))
        self.assertEqual(p.get_action(), [action(float(np.float32(.4)))])

    def test_05_reset_joins_blocked_old_worker_before_starting_another(self):
        started = []
        suffix = []
        def runner(initial, exchange):
            started.append(initial['state']['left_arm_joint_state'][0])
            try:
                exchange(action(.1))
            except Exception:
                suffix.append('legacy-recovery-must-not-run')
            suffix.append('cancelled-suffix-must-not-run')
        p = self.make_policy(runner)
        p.update_obs(observation(.2)); p.get_action()
        old_worker = p._thread
        p.reset()
        self.assertFalse(old_worker.is_alive())
        p.update_obs(observation(.7)); p.get_action()
        np.testing.assert_allclose(started, [.2, .7])
        self.assertEqual(suffix, [])
        # A runner stuck outside exchange must not be silently reused.
        entered = threading.Event(); release = threading.Event()
        def stuck(initial, exchange):
            entered.set(); release.wait(1)
        isolated = BlockingRulePolicy(stuck, wait_timeout=.05, join_timeout=.01)
        try:
            isolated.update_obs(observation())
            with self.assertRaises(TimeoutError): isolated.get_action()
            self.assertTrue(entered.is_set())
            with self.assertRaisesRegex(RuntimeError, 'refusing episode reuse'): isolated.reset()
            with self.assertRaisesRegex(RuntimeError, 'closed'): isolated.update_obs(observation())
        finally:
            release.set(); isolated._join_timeout = 1; isolated.close()

    def test_06_private_fields_are_never_passed_to_runner_and_rgb_is_not_swapped(self):
        seen = []
        p = self.make_policy(lambda initial, exchange: seen.append(initial))
        raw = observation(); p.update_obs(raw)
        raw['state']['left_arm_joint_state'][0] = 99
        p.get_action()
        self.assertEqual(set(seen[0]), {'state', 'vision'})
        self.assertEqual(len(seen[0]['state']), 6)
        self.assertEqual(seen[0]['state']['left_arm_joint_state'][0], 0)
        for frame in seen[0]['vision'].values():
            self.assertEqual(set(frame), {'colors'})
            np.testing.assert_equal(frame['colors'][0, 0], [11, 23, 47])

    def test_07_malformed_action_or_feedback_cannot_be_returned_as_a_command(self):
        bad = action(0); bad['left_ee_joint_state'] = [.08, .08]
        p = self.make_policy(lambda initial, exchange: exchange(bad))
        p.update_obs(observation())
        with self.assertRaises(RulePolicyFailure) as raised:
            p.get_action()
        self.assertIsInstance(raised.exception.__cause__, ValueError)
        self.assertEqual(p.diagnostics()['actions_returned'], 0)
        frame = observation(); frame['state']['right_arm_joint_state'][0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Invalid public robot observation'):
            p.update_obs(frame)

    def test_08_unexpected_errors_are_not_converted_to_success_or_silenced(self):
        def fail(initial, exchange):
            raise LookupError('unexpected-workflow-bug')
        p = self.make_policy(fail)
        p.update_obs(observation())
        with self.assertRaises(RulePolicyFailure) as raised:
            p.get_action()
        self.assertIsInstance(raised.exception.__cause__, LookupError)
        self.assertIn('unexpected-workflow-bug', p.diagnostics()['workflow_error'])
        self.assertFalse(p.diagnostics()['official_success_claimed'])

    def test_09_real_official_websocket_client_server_and_unmodified_deploy(self):
        # No fake client or patched eval_one_episode. Only the environment is a
        # CPU fixture; this validates the policy protocol, not task performance.
        from client_server.ws.model_client import WsModelClient
        from client_server.ws.model_server import PolicyServer, PolicyServerConfig
        deploy = importlib.import_module('XPolicyLab.policy.DP.deploy')
        self.assertEqual(Path(deploy.__file__).resolve(), SDK / 'XPolicyLab/policy/DP/deploy.py')
        seen = []
        def runner(initial, exchange):
            seen.append(initial)
            frame = exchange(action(.1))
            exchange(action(float(frame['state']['left_arm_joint_state'][0]) + .1))
        p = self.make_policy(runner)
        loop = asyncio.new_event_loop()
        worker = threading.Thread(target=loop.run_forever, name='official-ws-test', daemon=True)
        worker.start()
        server = PolicyServer(p, PolicyServerConfig(host='127.0.0.1', port=0))
        client = None
        try:
            asyncio.run_coroutine_threadsafe(server.start(), loop).result(5)
            client = WsModelClient(url=server.url, evaluation_id='r33-cpu-only', trial_id='one',
                                   request_timeout_s=5, close_timeout_s=2, max_connect_attempts=1)
            class Environment:
                def __init__(self): self.applied = []
                def is_episode_end(self): return len(self.applied) == 3
                def get_obs(self): return observation(.4 if self.applied else .2)
                def take_action(self, target): self.applied.append(deepcopy(target))
            env = Environment()
            deploy.eval_one_episode(TASK_ENV=env, model_client=client)
            self.assertEqual(len(env.applied), 3)
            np.testing.assert_allclose([a['left_arm_joint_state'][0] for a in env.applied], [.1, .5, .5])
            self.assertEqual(len(seen), 1)
            self.assertEqual(set(seen[0]), {'state', 'vision'})
            self.assertEqual(p.diagnostics()['actions_returned'], 3)
        finally:
            p.close()
            if client is not None: client.close()
            asyncio.run_coroutine_threadsafe(server.stop(), loop).result(5)
            loop.call_soon_threadsafe(loop.stop); worker.join(5); loop.close()
        self.assertFalse(p.diagnostics()['worker_alive'])
        self.assertNotIn('src.eval_client.eval_env', sys.modules)


if __name__ == '__main__':
    unittest.main()
