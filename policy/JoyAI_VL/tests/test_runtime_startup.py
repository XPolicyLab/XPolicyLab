"""Startup regressions without starting Docker or a policy server."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FAKE_DOCKER = '''#!/usr/bin/env python3
import ipaddress,json,os,sys
from pathlib import Path
p=Path(os.environ['FAKE_NETWORKS'])
state=json.loads(p.read_text()) if p.exists() else {}
a=sys.argv[1:]
if a[:2]==['network','inspect']:
    name=a[2]
    if os.environ.get('FAKE_EMPTY'):
        raise SystemExit(0)
    if name not in state: raise SystemExit(1)
    if '--format' in a: print(state[name]['gateway'])
    raise SystemExit(0)
if a[:2]==['network','create']:
    if '--subnet' not in a: raise SystemExit(1)
    if '--gateway' not in a: raise SystemExit(2)
    subnet=a[a.index('--subnet')+1]
    gateway=a[a.index('--gateway')+1]
    assert ipaddress.ip_address(gateway) in ipaddress.ip_network(subnet)
    state[a[-1]]={'subnet':subnet,'gateway':gateway}
    p.write_text(json.dumps(state))
    print(a[-1]); raise SystemExit(0)
raise SystemExit(3)
'''


class Startup(unittest.TestCase):
    def test_exhausted_address_pool_creates_explicit_gateways(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            docker = directory / 'docker'
            docker.write_text(FAKE_DOCKER)
            docker.chmod(0o755)
            state = directory / 'networks.json'
            env = dict(os.environ, PATH=str(directory)+':'+os.environ['PATH'],
                       FAKE_NETWORKS=str(state), LANE='startup-test')
            result = subprocess.run(['bash', str(ROOT/'agent/api/run_agent.sh'), '--gateway'],
                                    env=env, capture_output=True, text=True, check=True)
            networks = json.loads(state.read_text())
            self.assertEqual(len(networks), 2)
            self.assertEqual(result.stdout.strip(), networks['roboshell-ext-startup-test']['gateway'])

    def test_missing_gateway_stops_before_starting_agent(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            docker = directory / 'docker'
            docker.write_text(FAKE_DOCKER)
            docker.chmod(0o755)
            env = dict(os.environ, PATH=str(directory)+':'+os.environ['PATH'],
                       FAKE_NETWORKS=str(directory/'networks.json'), FAKE_EMPTY='1', LANE='startup-test')
            result = subprocess.run(['bash', str(ROOT/'agent/api/run_agent.sh'), '--gateway'],
                                    env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('no IPv4 gateway', result.stderr)
            self.assertEqual(result.stdout.strip(), '')

    def test_selected_checkout_precedes_other_benchmark_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            other = directory/'RoboDojo'
            (other/'XPolicyLab').mkdir(parents=True)
            (other/'XPolicyLab/__init__.py').write_text('raise RuntimeError("wrong checkout")\n')
            python = directory/'python'
            python.write_text('#!/usr/bin/env python3\nimport XPolicyLab\nprint(XPolicyLab.__file__)\n')
            python.chmod(0o755)
            env = dict(os.environ, ISAAC_PYTHON=str(python), ROBODOJO_REPO=str(other))
            command = ['bash',str(ROOT/'setup_eval_policy_server.sh'),'RoboDojo','synthetic',
                       'fixture','arx_x5','joint','0','0',str(python),'19999']
            result = subprocess.run(command,env=env,capture_output=True,text=True,check=True)
            self.assertIn(str(ROOT.parents[1]/'__init__.py'),result.stdout)


if __name__ == '__main__':
    unittest.main()
