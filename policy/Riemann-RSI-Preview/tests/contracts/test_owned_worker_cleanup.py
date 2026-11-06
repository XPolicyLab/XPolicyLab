"""A real reaped session leader must not leave workers holding GPU leases."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from rsi.gpu_resources import remember_owned_group, stop_owned, _process_identity, _live_owned_group


@unittest.skipUnless(Path('/proc/self/stat').is_file(), 'Linux evaluator process contract')
class OwnedWorkerCleanup(unittest.TestCase):
    def test_foreign_group_member_prevents_any_group_signal(self):
        identity=dict(pid=123,pgid=123,sid=123,uid=os.getuid(),start='old')
        member=dict(pid=124,pgid=123,sid=123,uid=os.getuid()+1,start='child',state='S')
        with patch('rsi.gpu_resources.subprocess.run',return_value=SimpleNamespace(returncode=0,stdout='124\n')), \
             patch('rsi.gpu_resources._process_identity',return_value=member), \
             patch('rsi.gpu_resources.os.killpg') as stop:
            with self.assertRaisesRegex(RuntimeError,'identity changed'):
                _live_owned_group(identity)
            stop.assert_not_called()

    def test_reaped_leader_worker_is_stopped_before_the_lease_is_released(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); lock = root/'gpu0.lock'; pid_file = root/'worker.pid'
            descriptor = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            code = ('import subprocess,sys,time;from pathlib import Path;'
                    'worker=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"],'
                    'pass_fds=(int(sys.argv[2]),));'
                    'Path(sys.argv[1]).write_text(str(worker.pid));time.sleep(.1)')
            leader = subprocess.Popen([sys.executable,'-c',code,str(pid_file),str(descriptor)],
                                      start_new_session=True,pass_fds=(descriptor,))
            remember_owned_group(leader)
            try:
                leader.wait(timeout=5)
                worker = int(pid_file.read_text())
                self.assertNotEqual(_process_identity(worker)['state'], 'Z')
                stop_owned(leader)
                state = _process_identity(worker)
                self.assertTrue(state is None or state['state'] == 'Z')
            finally:
                stop_owned(leader)
                os.close(descriptor)
            # A leaked worker inherits this same open-file description and
            # would keep the EX lock after the parent's descriptor closes.
            descriptor = os.open(lock, os.O_RDWR)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(descriptor)


if __name__ == '__main__':
    unittest.main()
