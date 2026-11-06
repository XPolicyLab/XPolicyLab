import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from rsi.official_runtime_monitor import RuntimeMonitor, OfficialRuntimeFault


class RuntimeMonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.now = 0.
        self.client = SimpleNamespace(pid=99999999, poll=lambda: None)
        self.server = SimpleNamespace(pid=99999998, poll=lambda: None)
        self.monitor = RuntimeMonitor(self.client, self.server, self.root/'result.json',
                                      self.root/'health.json', clock=lambda: self.now)

    def test_actual_official_ansi_action_counter_and_split_marker(self):
        self.monitor.feed('env0 step: \x1b[92m512 / 800\x1b[0m\r')
        self.assertEqual(self.monitor.actions, (512, 800))
        self.monitor.feed('env0 step: \x1b[92m800 / 800\x1b[0m\rHang dete')
        self.monitor.feed('cted\n')
        self.now = 31.
        with self.assertRaises(OfficialRuntimeFault):
            self.monitor.check()

    def test_recovered_native_progress_after_old_fatal_line_is_not_killed(self):
        self.monitor.feed('Cuda context manager error\nenv0 step: \x1b[92m10 / 800\x1b[0m\r')
        self.now = 20.
        self.monitor.check()
        self.assertIsNone(self.monitor.pending_fault)

    def test_server_signal_is_reported_while_client_is_still_alive(self):
        self.server.poll = lambda: -11
        with self.assertRaises(OfficialRuntimeFault) as caught:
            self.monitor.check()
        self.assertEqual(caught.exception.kind, 'policy_server_exited')
        self.assertEqual(json.loads((self.root/'health.json').read_text())['server_returncode'], -11)

    def test_slow_planning_and_log_chatter_never_trigger_a_wall_cap(self):
        self.monitor.feed('512/800 [12:00<1:00]\n')
        self.now = 100000.
        self.monitor.feed('still planning\n')
        self.monitor.check()
        self.assertEqual(self.monitor.last_progress, 0.)

    def test_explicit_cuda_fault_waits_for_sdk_recovery_then_reports(self):
        self.monitor.feed('PhysX Synchronizing/Fetching GPU Narrowphase failed! 700\n')
        self.now = 6.
        with self.assertRaises(OfficialRuntimeFault) as caught:
            self.monitor.check()
        self.assertEqual(caught.exception.kind, 'client_cuda_context_failed')

    def test_real_action_progress_delays_hang_confirmation(self):
        self.monitor.feed('Hang detected\n')
        self.now = 20.
        self.monitor.feed('513/800 [12:00<1:00]\n')
        self.now = 35.
        self.monitor.check()

    def test_open_kit_log_hang_and_terminal_unscored_are_not_official_success(self):
        log = self.root/'kit.log'; log.write_text('Hang detected\nKit appears to be hanging\n')
        self.monitor = RuntimeMonitor(self.client, self.server, self.root/'result.json',
                                      self.root/'health.json', clock=lambda:self.now, auxiliary_logs=[log])
        self.monitor.feed('800/800 [12:00<0:00]\n')
        self.monitor.check()
        self.now = 31.
        with self.assertRaises(OfficialRuntimeFault) as caught:
            self.monitor.check()
        self.assertEqual(caught.exception.kind, 'client_kit_hang')
        self.assertEqual(self.monitor.graded, 0)

    def test_kit_ui_warning_during_unfinished_slow_policy_is_not_killed(self):
        self.monitor.feed('512/800 [12:00<1:00]\nHang detected\n')
        self.now = 100000.
        self.monitor.check()
        self.assertIsNotNone(self.monitor.pending_fault)

    def test_native_exec_discards_closed_kit_context_faults(self):
        old = self.root/'closed-kit.log';old.write_text('Cuda context manager error\n')
        self.monitor.logs[old] = 0
        self.monitor.pending_fault = (0., 'old context', 'client_cuda_context_failed', 5.)
        self.now = 100.
        self.monitor.check()
        self.assertNotIn(old, self.monitor.logs)
        self.assertIsNone(self.monitor.pending_fault)

    def test_exited_client_preserves_native_restart_returncode_contract(self):
        self.monitor.feed('Cuda context manager error\n')
        self.client.poll = lambda: 99
        self.now = 100.
        self.monitor.check()

    def test_scored_episode_terminal_does_not_mark_the_next_reset_unscored(self):
        self.monitor.policy_episode_path = self.root / 'policy/episode-0000'
        self.monitor.feed('800/800 [12:00<0:00]\n')
        result = dict(details={'0': dict(layout_id=0, success=False, score=0.)},
                      eval_time=1, success_rate=0., score=0.)
        (self.root/'result.json').write_text(json.dumps(result))
        self.now = 1.
        self.monitor.check()
        self.assertEqual(self.monitor.graded, 1)
        self.assertIsNone(self.monitor.actions)
        self.monitor.feed('Resetting all environments\nHang detected\n')
        self.now = 100000.
        self.monitor.check()
        self.assertEqual(json.loads((self.root/'health.json').read_text())['status'], 'hang_suspected')

    def test_next_episode_terminal_still_requires_official_scoring(self):
        self.monitor.feed('800/800 [12:00<0:00]\n')
        result = dict(details={'0': dict(layout_id=0, success=False, score=0.)},
                      eval_time=1, success_rate=0., score=0.)
        (self.root/'result.json').write_text(json.dumps(result))
        self.monitor.check()
        self.now = 1.
        self.monitor.feed('800/800 [12:00<0:00]\nHang detected\n')
        self.now = 32.
        with self.assertRaises(OfficialRuntimeFault) as caught:
            self.monitor.check()
        self.assertEqual(caught.exception.kind, 'client_kit_hang')
        self.assertEqual(self.monitor.graded, 1)

    def test_explicit_kit_hang_before_the_first_rule_episode_is_reported(self):
        self.monitor.policy_episode_path = self.root / 'policy/episode-0000'
        self.monitor.feed('[INFO] Resetting all environments\nHang detected\n')
        self.now = 299.
        self.monitor.check()
        self.now = 301.
        with self.assertRaises(OfficialRuntimeFault) as caught:
            self.monitor.check()
        self.assertEqual(caught.exception.kind, 'client_kit_bootstrap_hang')
        health = json.loads((self.root / 'health.json').read_text())
        self.assertFalse(health['official_episode_started'])
        self.assertEqual(health['official_graded'], 0)

    def test_first_rule_perception_without_an_action_counter_is_not_a_bootstrap_hang(self):
        self.monitor.policy_episode_path = self.root / 'policy/episode-0000'
        self.monitor.policy_episode_path.mkdir(parents=True)
        self.monitor.feed('[INFO] Resetting all environments\nHang detected\n')
        self.now = 100000.
        self.monitor.check()
        self.assertEqual(json.loads((self.root / 'health.json').read_text())['status'], 'hang_suspected')

    def test_bootstrap_detection_requires_positive_reset_and_no_episode_evidence(self):
        self.monitor.policy_episode_path = self.root / 'policy/episode-0000'
        self.monitor.feed('Hang detected\n')
        self.now = 100000.
        self.monitor.check()  # Elapsed time and Kit alone are not enough.
        self.monitor.feed('[INFO] Resetting all environments\n[INFO] Running Episode 0\n')
        self.now = 200000.
        self.monitor.check()  # An episode can start before its first action.
        self.assertTrue(self.monitor.episode_started)

    def test_reset_discards_old_terminal_before_first_next_action(self):
        self.monitor.feed('1050/1050 [12:00<0:00]\n[INFO] Resetting all environments\n')
        self.assertIsNone(self.monitor.actions)
        self.monitor.feed('still planning the next episode\nHang detected\n')
        self.now = 100000.
        self.monitor.check()
        self.assertEqual(self.monitor.graded, 0)
        self.assertEqual(json.loads((self.root/'health.json').read_text())['status'], 'hang_suspected')

    def test_episode_marker_is_consumed_before_later_action_chunks(self):
        self.monitor.feed('1050/1050 [12:00<0:00]\n')
        self.monitor.feed('[INFO] Running Episode 20\n')
        self.assertIsNone(self.monitor.actions)
        self.monitor.feed('env0 step: 1 / 1050\n')
        self.assertEqual(self.monitor.actions, (1, 1050))
        self.monitor.feed('still planning\n')
        self.assertEqual(self.monitor.actions, (1, 1050))

    def test_reset_without_action_does_not_clear_explicit_cuda_failure(self):
        self.monitor.feed('Cuda context manager error\n[INFO] Resetting all environments\n')
        self.now = 6.
        with self.assertRaises(OfficialRuntimeFault) as caught:
            self.monitor.check()
        self.assertEqual(caught.exception.kind, 'client_cuda_context_failed')


if __name__ == '__main__':
    unittest.main()
