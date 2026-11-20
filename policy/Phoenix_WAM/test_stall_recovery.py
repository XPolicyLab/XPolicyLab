"""CPU regression tests; the adapter tests use a fake sampler, not model weights."""
from collections import deque
from pathlib import Path
from types import ModuleType, SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .stall_recovery import StallCover, StallSettings


def settings(**overrides):
    config = dict(phoenix_stall_fix='cover', phoenix_stall_steps=4)
    config.update(overrides)
    return StallSettings.from_config(config, 30)


def adapted():
    head = np.full((9, 13, 3), 157, dtype=np.uint8)
    head.setflags(write=False)
    return {'observation': {
        'head_camera': {'rgb': head},
        'left_camera': {'rgb': np.full((8, 10, 3), 73, dtype=np.uint8)},
        'right_camera': {'rgb': np.full((8, 10, 3), 96, dtype=np.uint8)}},
        'joint_action': {'vector': np.zeros(14, dtype=np.float32)}}


class DetectorTests(unittest.TestCase):
    def make(self, **config):
        return StallCover(settings(**config), 14, [slice(6, 7), slice(13, 14)])

    def fill(self, tracker, state=None, start=0):
        state = np.zeros(14) if state is None else state
        for step in range(start, start + tracker.settings.steps + 1):
            tracker.observe(state, step)

    def test_rates_and_explicit_override(self):
        for hz, expected in [(30, 90), (25, 75), (10, 30)]:
            self.assertEqual(StallSettings.from_config(
                {'phoenix_stall_fix': 'cover', 'action_hz': hz}, 30).steps, expected)
        self.assertEqual(settings(phoenix_stall_steps=17, action_hz=25).steps, 17)

    def test_invalid_configuration(self):
        for key, value in [('phoenix_stall_fix', 'mask'), ('phoenix_stall_steps', -1),
                           ('phoenix_stall_steps', 1.5), ('phoenix_stall_steps', True),
                           ('phoenix_stall_cover_chunks', 0),
                           ('phoenix_stall_arm_threshold', float('nan')),
                           ('phoenix_stall_gripper_threshold', -.1),
                           ('phoenix_stall_seed', -2), ('phoenix_stall_seed', 1.5),
                           ('phoenix_stall_seed', True), ('phoenix_stall_seed', 2**63),
                           ('action_hz', float('inf'))]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                settings(**{key: value})

    def test_disabled_identity(self):
        for config in [dict(phoenix_stall_fix='none'), dict(phoenix_stall_steps=0)]:
            tracker = self.make(**config)
            self.fill(tracker)
            obs = adapted()
            supplied, decision = tracker.plan(obs)
            self.assertIs(supplied, obs)
            self.assertFalse(decision.covered)

    def test_wait_for_full_acknowledged_window(self):
        tracker = self.make()
        for step in range(4):
            tracker.observe(np.zeros(14), step)
            self.assertFalse(tracker.plan(adapted())[1].covered)
        tracker.observe(np.zeros(14), 4)
        self.assertTrue(tracker.plan(adapted())[1].covered)

    def test_duplicate_uploads_are_not_steps(self):
        tracker = self.make()
        for _ in range(100):
            tracker.observe(np.zeros(14), 0)
        self.assertEqual(len(tracker.history), 1)
        self.assertFalse(tracker.plan(adapted())[1].stalled)

    def test_head_only_black_without_mutating_input(self):
        tracker = self.make()
        self.fill(tracker)
        obs = adapted()
        supplied, decision = tracker.plan(obs)
        self.assertTrue(decision.covered)
        self.assertEqual(supplied['observation']['head_camera']['rgb'].dtype, np.uint8)
        self.assertFalse(supplied['observation']['head_camera']['rgb'].any())
        self.assertTrue((obs['observation']['head_camera']['rgb'] == 157).all())
        for view in ['left_camera', 'right_camera']:
            self.assertIs(supplied['observation'][view], obs['observation'][view])
        self.assertIs(supplied['joint_action'], obs['joint_action'])

    def test_two_cover_one_restore_cycle(self):
        tracker = self.make()
        self.fill(tracker)
        pattern = []
        for _ in range(9):
            _, decision = tracker.plan(adapted())
            pattern.append(decision.covered)
            tracker.commit(decision)
        self.assertEqual(pattern, [True, True, False] * 3)
        self.assertEqual(tracker.snapshot()['changed'], 6)
        self.assertEqual(tracker.snapshot()['stalled'], 9)

    def test_arm_or_gripper_motion_restores_real_image(self):
        for dim, change in [(1, .03), (6, .06), (13, .06)]:
            tracker = self.make()
            self.fill(tracker)
            tracker.commit(tracker.plan(adapted())[1])
            state = np.zeros(14)
            state[dim] = change
            tracker.observe(state, 5)
            obs = adapted()
            supplied, decision = tracker.plan(obs)
            self.assertIs(supplied, obs)
            self.assertFalse(decision.stalled)
            tracker.commit(decision)
            self.assertEqual(tracker.phase, 0)

    def test_out_and_back_is_not_stall(self):
        tracker = self.make()
        for step in range(5):
            state = np.zeros(14)
            state[1] = .1 if step == 2 else 0.
            tracker.observe(state, step)
        self.assertFalse(tracker.plan(adapted())[1].stalled)

    def test_reset_and_invalid_observation(self):
        tracker = self.make()
        self.fill(tracker)
        tracker.commit(tracker.plan(adapted())[1])
        with self.assertRaises(ValueError):
            tracker.observe(np.full(14, np.nan), 5)
        with self.assertRaises(ValueError):
            tracker.observe(np.zeros(14), 3)
        tracker.reset()
        self.assertEqual(tracker.snapshot()['changed'], 0)
        self.assertFalse(tracker.plan(adapted())[1].covered)

    def test_inference_plan_does_not_advance_cycle(self):
        tracker = self.make()
        self.fill(tracker)
        before = tracker.snapshot()
        for _ in range(3):
            self.assertTrue(tracker.plan(adapted())[1].covered)
        self.assertEqual(tracker.snapshot(), before)

    def test_random_seed_is_fresh_without_changing_phase_before_commit(self):
        tracker = self.make()
        self.fill(tracker)
        decision = tracker.plan(adapted())[1]
        before = tracker.snapshot()
        with patch(__package__ + '.stall_recovery.secrets.randbits',
                   side_effect=[7, 101, 101, 102]):
            self.assertEqual(tracker.inference_seed(decision, 7), (-1, 101))
            self.assertEqual(tracker.snapshot(), before)
            tracker.commit(decision, requested_seed=-1, effective_seed=101)
            self.assertEqual(tracker.inference_seed(tracker.plan(adapted())[1], 7),
                             (-1, 102))

    def test_fixed_null_and_normal_seeds(self):
        for configured, expected in [(None, (None, 7)), ('null', (None, 7)),
                                     (0, (0, 0)), (23, (23, 23)),
                                     (2**63 - 1, (2**63 - 1, 2**63 - 1)),
                                     (str(2**63 - 1), (2**63 - 1, 2**63 - 1))]:
            tracker = self.make(phoenix_stall_seed=configured)
            with patch(__package__ + '.stall_recovery.secrets.randbits',
                       side_effect=AssertionError('unexpected random draw')):
                self.assertEqual(tracker.inference_seed(tracker.plan(adapted())[1], 7),
                                 (None, 7))
                self.fill(tracker)
                self.assertEqual(tracker.inference_seed(tracker.plan(adapted())[1], 7),
                                 expected)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        from .model import Model
        from . import _adapter_base
        self.Model = Model
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.calls = []
        self.seed_calls = []
        self.fail = False
        audit_dir = Path(self.directory.name)
        runtime = SimpleNamespace(audit_dir=audit_dir)

        class Session:
            def __init__(self, seed):
                self.runtime = runtime
                self.seed = seed
                self.reset_model()

            def reset_model(self):
                self.pending_model_actions = deque()
                self.steps = self.calls = 0

            def update_obs(self, observation):
                self.pending_model_actions.popleft()
                self.steps += 1

            def get_timing_rollout(self):
                return dict(executed_action_steps=self.steps, replan_calls=self.calls)

        def load(model):
            model.model_cfg['runtime_mode'] = 'joint96_real_phoenix'
            model._stall_settings = StallSettings.from_config(model.model_cfg, 30)
            model.runtime = runtime
            model.session_factory = lambda: Session(model.model_cfg['seed'])
            model.session = model.session_factory()

        def predict(model, session, obs, instruction):
            self.seed_calls.append(session.seed)
            if self.fail:
                raise RuntimeError('injected sampler failure')
            self.calls.append((session, obs, instruction, model.model_cfg['seed']))
            session.pending_model_actions.extend(range(model.replan_steps))
            session.calls += 1
            action = np.tile(obs['joint_action']['vector'], (model.replan_steps, 1))
            return _adapter_base.unpack_robot_state(
                action, 'joint', model.robot_action_dim_info, source_type='obs')

        backend = ModuleType(__package__ + '.phoenix_backend')
        backend.predict_phoenix = predict
        for context in [patch.object(Model, '_load_real_policy', load),
                        patch.object(_adapter_base, 'get_robot_action_dim_info',
                                     return_value={'arm_dim': [6, 6], 'ee_dim': [1, 1]}),
                        patch.dict('sys.modules', {backend.__name__: backend})]:
            context.start()
            self.addCleanup(context.stop)

    def model(self, **config):
        cfg = dict(env_cfg_type='arx_x5', action_type='joint', seed=7,
                   replan_steps=4, phoenix_stall_fix='cover', phoenix_stall_steps=4)
        cfg.update(config)
        return self.Model(cfg)

    def obs(self, env_idx=None, movement=0.):
        a = adapted()
        state = np.zeros(6, dtype=np.float32)
        state[0] = movement
        obs = dict(vision={key: {'color': a['observation'][view]['rgb']}
                           for key, view in [('cam_head', 'head_camera'),
                                             ('cam_left_wrist', 'left_camera'),
                                             ('cam_right_wrist', 'right_camera')]},
                   state=dict(left_arm_joint_state=state, left_ee_joint_state=np.zeros(1),
                              right_arm_joint_state=np.zeros(6), right_ee_joint_state=np.zeros(1)),
                   instruction='Cover the blocks')
        if env_idx is not None:
            obs['env_idx'] = env_idx
        return obs

    def ack(self, model, **kwargs):
        for _ in range(model.replan_steps):
            model.update_obs(self.obs(**kwargs))

    def test_real_adapter_hook_and_pending_queue(self):
        m = self.model()
        m.update_obs(self.obs())
        m.get_action()
        with self.assertRaises(RuntimeError):
            m.get_action()
        self.ack(m)
        actions = m.get_action()
        self.assertEqual(len(actions), 4)
        self.assertFalse(self.calls[-1][1]['observation']['head_camera']['rgb'].any())
        self.assertTrue(m.last_obs['observation']['head_camera']['rgb'].any())
        self.assertEqual(self.calls[-1][2:], ('Cover the blocks', 7))
        self.assertEqual(m.get_stall_rollout()['single']['changed'], 1)
        self.ack(m, movement=.1)
        m.get_action()
        self.assertTrue(self.calls[-1][1]['observation']['head_camera']['rgb'].any())

    def test_disabled_inference_passes_same_observation(self):
        m = self.model(phoenix_stall_fix='none')
        m.update_obs(self.obs())
        m.get_action()
        self.assertIs(self.calls[-1][1], m.last_obs)
        self.ack(m)
        m.get_action()
        self.assertIs(self.calls[-1][1], m.last_obs)
        self.assertEqual(self.seed_calls, [7, 7])

    def test_black_and_random_seed_cycle_then_motion_restore(self):
        m = self.model()
        m.update_obs(self.obs())
        m.get_action()
        with patch(__package__ + '.stall_recovery.secrets.randbits',
                   side_effect=[101, 102, 103]):
            for expected_cover in (True, True, False):
                self.ack(m)
                m.get_action()
                status = m.get_stall_rollout()['single']
                self.assertEqual(status['last_covered'], expected_cover)
                self.assertEqual(status['last_requested_seed'], -1)
                self.assertEqual(status['last_effective_seed'], self.seed_calls[-1])
                self.assertEqual(m.session.seed, 7)
        self.ack(m, movement=.1)
        m.get_action()
        status = m.get_stall_rollout()['single']
        self.assertFalse(status['last_stalled'])
        self.assertIsNone(status['last_requested_seed'])
        self.assertEqual(status['last_effective_seed'], 7)
        self.assertEqual(status['reseeded'], 3)
        self.assertEqual(self.seed_calls, [7, 101, 102, 103, 7])
        self.assertEqual(m.model_cfg['seed'], 7)
        m.reset()
        self.assertEqual(m.session.seed, 7)
        self.assertEqual(m.get_stall_rollout()['single']['reseeded'], 0)

    def test_cover_only_and_fixed_seed_override(self):
        for configured, expected in [(None, 7), (23, 23)]:
            m = self.model(phoenix_stall_seed=configured)
            m.update_obs(self.obs())
            m.get_action()
            self.ack(m)
            m.get_action()
            self.assertTrue(m.get_stall_rollout()['single']['last_covered'])
            self.assertEqual(self.seed_calls[-1], expected)
            self.assertEqual(m.session.seed, 7)

    def test_failed_inference_does_not_consume_cover(self):
        m = self.model()
        m.update_obs(self.obs())
        m.get_action()
        self.ack(m)
        before = m.get_stall_rollout()
        self.fail = True
        with self.assertRaises(RuntimeError):
            m.get_action()
        self.assertEqual(m.get_stall_rollout(), before)
        self.assertEqual(m.session.seed, 7)
        self.assertNotEqual(self.seed_calls[-1], 7)
        self.fail = False
        m.get_action()
        self.assertEqual(m.get_stall_rollout()['single']['changed'], 1)

    def test_batches_are_isolated_and_reset(self):
        m = self.model()
        m.update_obs_batch([self.obs(10), self.obs(20)])
        m.get_action_batch()
        for step in range(4):
            m.update_obs_batch([self.obs(10), self.obs(20, movement=.1 * step)])
        m.get_action_batch([20, 10])
        by_env = m.get_stall_rollout()
        self.assertEqual(by_env['10']['changed'], 1)
        self.assertEqual(by_env['20']['changed'], 0)
        self.assertTrue(self.calls[-2][1]['observation']['head_camera']['rgb'].any())
        self.assertFalse(self.calls[-1][1]['observation']['head_camera']['rgb'].any())
        self.assertEqual(self.seed_calls[-2], 7)
        self.assertNotEqual(self.seed_calls[-1], 7)
        self.assertTrue(all(session.seed == 7 for session in m.env_sessions.values()))
        m.reset()
        self.assertTrue(all(x['stalled'] == 0 for x in m.get_stall_rollout().values()))
        m.update_obs_batch([self.obs(10), self.obs(20)])
        m.get_action_batch()
        self.assertTrue(self.calls[-1][1]['observation']['head_camera']['rgb'].any())
        events = [json.loads(line) for line in
                  (Path(self.directory.name) / 'stall_recovery.jsonl').read_text().splitlines()]
        self.assertEqual(sum(row['event'] == 'episode_end' for row in events), 2)

    def test_audit_write_failure_does_not_lose_action_chunk(self):
        m = self.model()
        m.update_obs(self.obs())
        with patch.object(Path, 'open', side_effect=OSError('test full disk')):
            with self.assertLogs(level='ERROR'):
                actions = m.get_action()
        self.assertEqual(len(actions), 4)
        self.assertEqual(len(m.session.pending_model_actions), 4)


if __name__ == '__main__':
    unittest.main()
