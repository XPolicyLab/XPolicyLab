import json
import os
import signal
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rsi.official_unit_supervisor import owned_peers, ProcessView, recover, client_transcript_path


class OfficialUnitSupervisor(unittest.TestCase):
    def test_another_users_owner_is_rejected_before_any_signal(self):
        spec={'output':'/owned/campaign/runs/unit'}
        with patch('rsi.official_unit_supervisor.read',return_value={'owner_pid':123}), \
             patch('rsi.official_unit_supervisor.proc',return_value=dict(pid=123,uid=os.getuid()+1,state='S',pgid=123,cmdline='/owned/campaign')), \
             patch('rsi.official_unit_supervisor.os.kill') as signal:
            with self.assertRaisesRegex(ValueError,'Unverified unit owner'):
                owned_peers(spec)
            signal.assert_not_called()

    def test_changed_standard_lock_descriptor_is_rejected(self):
        spec={'output':'/owned/campaign/runs/unit'}
        with patch('rsi.official_unit_supervisor.read',return_value={'owner_pid':123,'held_descriptors':[4]}), \
             patch('rsi.official_unit_supervisor.proc',return_value=dict(pid=123,uid=os.getuid(),state='S',pgid=123,cmdline='/owned/campaign')), \
             patch('rsi.official_unit_supervisor.os.readlink',return_value='/unrelated/file'):
            with self.assertRaisesRegex(ValueError,'descriptor identity'):
                owned_peers(spec)

    def test_verified_zombie_server_keeps_its_real_signal_diagnosis(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'campaign/runs/unit';out.mkdir(parents=True)
            owner=dict(pid=100,uid=os.getuid(),state='S',pgid=100,start='10',cmdline=str(out.parent.parent))
            client=dict(pid=101,uid=os.getuid(),state='S',pgid=101,start='11',cmdline='run_official_eval.py --task_name task')
            server=dict(pid=102,uid=os.getuid(),state='Z',pgid=102,start='12',exit_code=11,cmdline='')
            (out/'lease.json').write_text(json.dumps(dict(owner_pid=100,held_descriptors=[4])))
            (out/'supervisor-owner-proof.json').write_text(json.dumps(dict(owner=owner,client=client,server=server)))
            with patch('rsi.official_unit_supervisor.proc',side_effect=lambda pid:{100:owner,101:client,102:server}.get(pid)), \
                 patch('rsi.official_unit_supervisor.os.readlink',return_value='/x/robodojo-shared-gpu-locks/node/gpu0.lock'), \
                 patch('rsi.official_unit_supervisor.subprocess.run',return_value=SimpleNamespace(stdout='101\n')):
                peers=owned_peers(dict(output=str(out),mode='task'))
                self.assertEqual(ProcessView(peers['server']).poll(),-11)

    def test_native_shell_restart_uses_highest_attempt_not_log_mtime(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);first=root/'official-client-attempt-1.log';second=root/'official-client-attempt-2.log'
            first.write_text('old fatal context');second.write_text('new healthy client')
            os.utime(first,(2000000000,2000000000));os.utime(second,(1,1))
            self.assertEqual(client_transcript_path(root),second)

    def test_reused_pid_is_not_treated_as_the_original_peer(self):
        view=ProcessView(dict(pid=123,start='old'))
        with patch('rsi.official_unit_supervisor.proc',return_value=dict(start='new',state='S')):
            self.assertEqual(view.poll(),1)

    def test_unscored_fault_never_starts_a_replacement_episode(self):
        with tempfile.TemporaryDirectory() as folder:
            r=Path(folder);out=r/'output';out.mkdir();(out/'lease.json').write_text(json.dumps({'owner_pid':123}))
            spec=dict(output=str(out),official_result=str(r/'missing-result.json'),expected=50)
            with patch('rsi.official_unit_supervisor.proc',return_value=None), \
                 patch('rsi.official_unit_supervisor.subprocess.Popen') as start:
                self.assertIsNone(recover(r/'registry.json',r/'unit.json',spec,'client_kit_hang',r))
                start.assert_not_called()

    def test_embedded_fault_cleanup_is_not_interrupted_by_a_second_signal(self):
        for reported_server, cleanup in [(125,True),(999,False)]:
            with self.subTest(reported_server=reported_server), tempfile.TemporaryDirectory() as folder:
                root=Path(folder);out=root/'output';out.mkdir()
                (out/'lease.json').write_text(json.dumps({'owner_pid':123}))
                (out/'runtime-health.json').write_text(json.dumps(dict(
                    status='fault',fault='policy_server_exited',client_pid=124,server_pid=reported_server)))
                owner=dict(pid=123,state='S')
                peers=dict(owner=owner,client={'pid':124},server={'pid':125})
                spec=dict(output=str(out),official_result=str(root/'missing.json'),expected=50)
                with patch('rsi.official_unit_supervisor.proc',side_effect=[owner,None,None]), \
                     patch('rsi.official_unit_supervisor.owned_peers',return_value=peers), \
                     patch('rsi.official_unit_supervisor.os.kill') as stop, \
                     patch('rsi.official_unit_supervisor.subprocess.Popen') as start:
                    self.assertIsNone(recover(root/'registry.json',root/'unit.json',spec,'policy_server_exited',root))
                    if cleanup:stop.assert_not_called()
                    else:stop.assert_called_once_with(123,signal.SIGTERM)
                    start.assert_not_called()
                self.assertEqual(json.loads((out/'supervisor-stop-proof.json').read_text())['launcher_cleanup_observed'],cleanup)


if __name__ == '__main__':
    unittest.main()
