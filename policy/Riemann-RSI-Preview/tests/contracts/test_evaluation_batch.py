from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rsi import evaluation_batch as campaign


class EvaluationBatchTests(unittest.TestCase):
    def test_batch_cli_works_outside_the_repository_without_sdk(self):
        import subprocess
        import sys
        entry=Path(__file__).resolve().parents[2]/'tools/run_evaluation_batch.py'
        with tempfile.TemporaryDirectory() as folder:
            result=subprocess.run([sys.executable,'-I','-S','-B',str(entry),'--help'],
                                  cwd=folder,capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--campaign',result.stdout)
        self.assertIn('prepare',result.stdout)

    def test_default_task_invocations_do_not_add_a_whole_cohort_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); plan = self.prepare(folder)
            for backend in ('rule', 'learned'):
                key, unit = next((k,v) for k,v in plan['tasks'].items() if v['backend']==backend)
                slot = next(s for s in plan['slots'] if s['node']==unit['assigned_node'])
                def result(*args, **kwargs):
                    campaign.write(Path(unit['official_result']), dict(eval_time=unit['expected'],
                        details={str(i): {} for i in range(unit['expected'])}))
                with patch.object(campaign.socket, 'gethostname', return_value=slot['node']), \
                        patch.dict(campaign.os.environ, SLURM_JOB_ID=slot['job']), \
                        patch('rsi.skills.git_versioning.require_clean', return_value={'commit':'fixture'}), \
                        patch('tools.official_launch.evaluate', side_effect=result) as evaluate:
                    campaign.run_task(folder,key,'0,1' if backend=='learned' else '0','')
                self.assertNotIn('timeout', evaluate.call_args.kwargs)

    def test_worker_default_wall_budget_is_unbounded(self):
        import inspect
        self.assertIsNone(inspect.signature(campaign.worker).parameters['wall_seconds'].default)
        with patch.object(campaign, 'worker') as worker:
            campaign.main(['worker','--campaign','/tmp/synthetic-campaign','--seed','0'])
        self.assertIsNone(worker.call_args.args[-1])

    def test_explicit_invalid_wall_budgets_are_rejected(self):
        import io
        for value in ('0','-1'):
            with patch('sys.stderr',io.StringIO()), self.assertRaises(SystemExit) as error:
                campaign.main(['monitor','--campaign','/tmp/synthetic-campaign','--wall-seconds',value])
            self.assertEqual(error.exception.code,2)


    def prepare(self, folder):
        from rsi.policy_catalog import runtime_settings
        common = dict(runtime_settings(), official_root=str(folder / 'sdk'))
        slots = ['%d:%d:node%d' % (i // 3, 100 + i, i) for i in range(9)]
        with patch('rsi.skills.git_versioning.require_clean', return_value={'commit': 'fixture'}), \
                patch.object(campaign, 'runtime_settings', return_value=common):
            campaign.prepare(folder, slots)
        return campaign.read(folder / 'plan.json')

    def test_all_54_modes_each_seed_and_no_probe_or_manual_layout_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            plan = self.prepare(folder)
            for seed in range(3):
                units = [u for u in plan['tasks'].values() if u['seed'] == seed]
                self.assertEqual(len(units), 54)
                self.assertEqual(sum(u['expected'] for u in units), 2100)
                self.assertTrue(all('layout_ids' not in u for u in units))
            for seed in range(3):
                slots = [s for s in plan['slots'] if s['seed'] == seed]
                for backend in ('rule','learned'):
                    counts = [sum(u['backend'] == backend and u['assigned_node'] == s['node']
                                  for u in plan['tasks'].values()) for s in slots]
                    expected = sum(u['seed'] == seed and u['backend'] == backend
                                   for u in plan['tasks'].values())
                    self.assertEqual(sum(counts),expected)
                    self.assertLessEqual(max(counts)-min(counts),1)
            report = campaign.collect(folder)
            self.assertEqual(report['done'], 0)
            self.assertIsNone(report['final_success_rate'])
            self.assertFalse(plan['probe_results_imported'])
            with self.assertRaises(FileExistsError):
                campaign.prepare(folder, [])

    def test_twelve_nodes_keep_exactly_three_seeds_and_6300_default_episodes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            slots=['%d:%d:node%d' % (i // 4, 200+i, i) for i in range(12)]
            with patch('rsi.skills.git_versioning.require_clean', return_value={'commit':'fixture'}):
                campaign.prepare(root,slots)
            plan=campaign.read(root/'plan.json')
            self.assertEqual(len(plan['tasks']),162)
            self.assertEqual(sum(u['expected'] for u in plan['tasks'].values()),6300)
            for seed in range(3):
                self.assertEqual(len([s for s in plan['slots'] if s['seed']==seed]),4)
                owners={s['node'] for s in plan['slots'] if s['seed']==seed}
                self.assertTrue(all(u['assigned_node'] in owners for u in plan['tasks'].values() if u['seed']==seed))
            self.assertFalse(plan['probe_results_imported'])

    def test_node_writes_are_disjoint_and_cannot_drop_or_take_another_partition(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            plan = self.prepare(folder)
            first, second = plan['slots'][:2]
            path = folder / ('node-' + first['node'] + '-state.json')
            untouched = folder / ('node-' + second['node'] + '-state.json')
            original = untouched.read_bytes()
            snapshot = campaign.read(path)
            key = next(iter(snapshot['tasks']))
            snapshot['worker_pid'] = 123
            size = 2 if plan['tasks'][key]['backend'] == 'learned' else 1
            snapshot['tasks'][key] = dict(status='running', node=first['node'],
                                          worker_pid=123, gpus=list(range(size)))
            campaign.write(path, snapshot)
            self.assertEqual(untouched.read_bytes(), original)
            self.assertEqual(sum(r['status'] == 'running' for r in campaign.collect(folder)['tasks']), 1)
            del snapshot['tasks'][key]['gpus']
            campaign.write(path, snapshot)
            with self.assertRaisesRegex(ValueError, 'lost its node/GPU'):
                campaign.collect(folder)

    def test_fixed_node_worker_guard_refuses_a_second_owner(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with campaign.worker_lock(folder, 'node0'):
                code = ('import sys; from pathlib import Path; '
                        'from rsi.evaluation_batch import worker_lock; '
                        'guard=worker_lock(Path(sys.argv[1]),"node0"); guard.__enter__()')
                other = subprocess.run([sys.executable, '-B', '-c', code, str(folder)],
                                       capture_output=True, text=True)
                self.assertNotEqual(other.returncode, 0)
                self.assertIn('BlockingIOError', other.stderr)
            with campaign.worker_lock(folder, 'node0'):
                pass

    def test_all_owned_tasks_receive_stop_before_waiting_for_nested_cleanup(self):
        import subprocess
        import sys
        import time
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            code = '''import signal,sys,time
from pathlib import Path
root=Path(sys.argv[1]); name=sys.argv[2]; stopping=False
def stop(signum,frame):
 global stopping
 stopping=True
signal.signal(signal.SIGTERM,stop)
(root/('ready-'+name)).touch()
while not stopping:time.sleep(.01)
(root/('stopped-'+name)).touch()
while len(list(root.glob('stopped-*')))<2:time.sleep(.01)
'''
            children = [subprocess.Popen([sys.executable, '-B', '-c', code, str(folder), str(i)],
                                         start_new_session=True) for i in range(2)]
            try:
                deadline = time.monotonic() + 5
                while len(list(folder.glob('ready-*'))) != 2:
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(.01)
                campaign.stop_task_processes(children, grace_seconds=3)
                self.assertEqual([child.returncode for child in children], [0, 0])
            finally:
                for child in children:
                    if child.poll() is None:
                        child.kill()
                        child.wait(timeout=5)

    def test_official_success_boolean_wins_over_a_high_partial_reward(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            plan = self.prepare(folder)
            key = next(iter(plan['tasks']))
            unit = plan['tasks'][key]
            campaign.write(Path(unit['official_result']), dict(eval_time=1, success_rate=0., score=100.,
                details={'0': dict(layout_id=2, success=False, score=1.)}))
            report = campaign.collect(folder)
            self.assertEqual((report['done'], report['successes'], report['failures']), (1, 0, 1))
            self.assertEqual(report['provisional_mean_score'], 100.)
            self.assertIsNone(report['final_mean_score'])
            self.assertFalse(report['complete'])

    def test_duplicate_official_layouts_are_reported_without_fabricating_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            plan = self.prepare(folder)
            key = next(iter(plan['tasks']))
            campaign.write(Path(plan['tasks'][key]['official_result']),
                dict(eval_time=2, success_rate=.5, score=50., details={
                    '0': dict(layout_id=7, success=True, score=1.),
                    '1': dict(layout_id=7, success=False, score=0.)}))
            report = campaign.collect(folder)
            row = next(r for r in report['tasks'] if r['identity'] == key)
            self.assertIn('more than once', row['read_error'])
            self.assertEqual(report['done'], 0)
            self.assertIsNone(report['provisional_mean_score'])


if __name__ == '__main__':
    unittest.main()
