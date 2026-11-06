"""Ten targeted checks of official counts, policy clocks and workflow reuse."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import ast
import json
import sys
import tempfile
import unittest
from rsi.deployment import locations

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from skills.shared.motion.public_observation_command_clock.implementation import PublicHistory, PublicCommandClock
from skills.shared.orchestration.public_rule_policy_interface.implementation import BlockingRulePolicy, normalized_action
from skills.shared.orchestration.public_mature_workflow.implementation import run
from tests.contracts.test_official_rule_policy_interface import observation, action


def environment():
    robots = [SimpleNamespace(type='target', arm_name=s + '_arm', gripper_name=s + '_ee',
                              gripper_scale=(-.01, .044), gripper_move={'sign': 1}) for s in ('left', 'right')]
    manager = SimpleNamespace(robot_list=robots, process_name=lambda name: name + '_joint_state', update=lambda frame: None)
    return SimpleNamespace(robot_manager=manager, dt=.004, step_lim=1900, take_action_cnt=[0])


class Adapter(unittest.TestCase):
    def history(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        return PublicHistory(observation(.2), Path(temp.name) / 'policy')

    def test_01_all_rule_modes_use_actual_official_native_counts(self):
        import yaml
        plan = json.loads((ROOT / 'configs/official-rule-evaluation.json').read_text())
        actual = yaml.safe_load((Path(locations()['official_root']) / 'task/RoboDojo/config/_task.yml').read_text())
        self.assertEqual(len(plan['tasks']), 44)
        self.assertEqual(sum(v['official_eval_nums'] for v in plan['tasks'].values()), 1775)
        for name, record in plan['tasks'].items():
            self.assertEqual(record['official_eval_nums'], actual['tasks'].get(name, {}).get('eval_nums', actual['common']['eval_nums']))

    def test_02_dense_sampling_matches_independent_numpy_reference(self):
        # 24 cases include fractional strides, sparse holds, short final tails,
        # and jaw-only paths whose duration must not follow arm motion speed.
        for speed in (1., .9, 1.25, 2.):
            for arm in (True, False):
                for length in (1, 17, 53):
                    history = self.history(); env = environment(); history.public_environment = env
                    issued = []
                    def exchange(command):
                        issued.append(deepcopy(command)); return observation(.4)
                    clock = PublicCommandClock(env, history, exchange, time_scale=speed)
                    values = np.linspace(.001, .021, length)
                    key = 'left_arm_joint_state' if arm else 'left_ee_joint_state'
                    dense = [{key: {'position': np.full(6 if arm else 2, value)}} for value in values]
                    clock.execute(dense, 'original')
                    stride = 10 * speed if arm else 10
                    samples = np.minimum(np.arange(stride, length + stride, stride), length) - 1
                    expected = np.interp(samples, np.arange(length), values)
                    actual = np.array([a[key][0] for a in issued])
                    if not arm: expected = (expected + .01) / .054
                    np.testing.assert_allclose(actual, expected, atol=1e-12)
                    self.assertEqual(env.take_action_cnt[0], len(issued))

    def test_03_callbacks_run_after_the_current_public_feedback(self):
        history = self.history(); env = environment(); history.public_environment = env
        clock = PublicCommandClock(env, history, lambda command: observation(.7), time_scale=1)
        seen = []
        clock.after_action = lambda phase: seen.append((float(history.state_left_arm[-1][0]), env.take_action_cnt[0], phase))
        clock.execute([{}] * 20, 'wait')
        self.assertEqual(len(seen), 2)
        self.assertEqual([v[1] for v in seen], [1, 2])
        self.assertTrue(all(abs(v[0] - .7) < 1e-6 for v in seen))

    def test_04_already_sampled_actions_are_not_sampled_again(self):
        history = self.history(); env = environment(); issued = []
        clock = PublicCommandClock(env, history, lambda command: issued.append(command) or observation(), time_scale=2.4)
        clock.execute_normalized(action(.3), 'checked_path')
        self.assertEqual(issued, [action(.3)])

    def test_05_observation_native_pose_actions_keep_the_official_format(self):
        a = {side + suffix: [0, 0, 0, 1, 0, 0, 0] if suffix == '_ee_pose' else [.4]
             for side in ('left', 'right') for suffix in ('_ee_pose', '_ee_joint_state')}
        self.assertEqual(normalized_action(a), a)
        a['left_ee_pose'][3] = 0
        with self.assertRaises(ValueError): normalized_action(a)

    def test_06_only_the_public_instruction_is_preserved(self):
        seen = []
        p = BlockingRulePolicy(lambda frame, exchange: seen.append(frame), wait_timeout=1)
        self.addCleanup(p.close)
        obs = observation(); obs['instruction'] = 'Pick up the red tool.'
        p.update_obs(obs); p.get_action()
        self.assertEqual(seen[0]['instruction'], obs['instruction'])
        self.assertNotIn('reward', seen[0]); self.assertNotIn('layout', seen[0])

    def test_07_existing_controller_factory_and_workflow_are_called_unchanged(self):
        seen = []
        env = environment()
        def factory(frame, recorder, args):
            seen.append(('factory', args.public_instruction)); return env
        def create(e, recorder, args, clock):
            def actor_run():
                seen.append(('run', e is env)); clock.execute([{}] * 10, 'original_wait')
            return SimpleNamespace(run=actor_run, result=lambda error: {'error': error})
        ctrl = SimpleNamespace(create_public_environment=factory, create=create)
        with tempfile.TemporaryDirectory() as d:
            obs = observation(); obs['instruction'] = 'public only'
            issued = []
            run(ctrl, {'parameters': {'motion_speed': 1}}, obs,
                lambda a: issued.append(a) or observation(.4), Path(d) / 'episode', horizon=1900, prepare=lambda full: None)
            self.assertEqual(seen, [('factory', 'public only'), ('run', True)])
            self.assertEqual(len(issued), 1)
            record = json.loads((Path(d) / 'episode/policy-diagnostics.json').read_text())
            self.assertIsNone(record['policy_error'])

    def test_08_workflow_bug_is_recorded_and_returns_to_official_finite_hold(self):
        def fail(*args): raise TypeError('reconstruction-bug')
        ctrl = SimpleNamespace(create_public_environment=fail)
        with tempfile.TemporaryDirectory() as d:
            def runner(frame, exchange):
                run(ctrl, {'parameters': {'motion_speed': 1}}, frame, exchange,
                    Path(d) / 'episode', horizon=400, prepare=lambda full: None)
            p = BlockingRulePolicy(runner, wait_timeout=2); self.addCleanup(p.close)
            p.update_obs(observation(.2)); held = p.get_action()
            p.update_obs(observation(.7)); self.assertEqual(p.get_action(), held)
            record = json.loads((Path(d) / 'episode/policy-diagnostics.json').read_text())
            self.assertIn('reconstruction-bug', record['policy_error'])
            self.assertFalse(record['official_score_assigned'])

    def test_09_official_entry_does_not_replace_any_evaluator_function(self):
        text = (ROOT / 'tools/run_official_eval.py').read_text()
        tree = ast.parse(text)
        assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)]
        self.assertFalse(any(isinstance(t, ast.Attribute) and t.attr in ('eval_one_episode', 'WsModelClient', '__init__') for n in assignments for t in n.targets))
        self.assertIn('runpy.run_path', text)
        plugin = (ROOT / 'rsi/adapters/Riemann-RSI-Preview/deploy.py').read_text()
        self.assertIn('XPolicyLab.policy.demo_policy.deploy import eval_one_episode', plugin)

    def test_10_existing42_closures_remain_unchanged_and_new_closure_is_complete(self):
        from rsi.policy_catalog import catalog
        from rsi.task_contract import verify
        from tools.official_rule_integrity import verify_policy
        for mode in catalog():
            self.assertGreater(verify(ROOT, mode), 0, mode)
        self.assertGreater(verify_policy(), 0)


if __name__ == '__main__': unittest.main()
