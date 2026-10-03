"""Owned process boundary for a frozen executable code skill.

The configured command starts exactly one skill service. It receives no benchmark
identifier from this adapter and cannot be substituted after episode creation.
"""
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import time
import uuid

from PhysicalRSI_core.infra.storage import file_digest


def isolate_transport(command, output):
    """Give each frozen launcher an empty policy namespace to bind its program."""
    command = list(command)
    if '--xpolicylab' not in command:
        return command
    index = command.index('--xpolicylab') + 1
    if index >= len(command):
        raise ValueError('Missing frozen transport source')
    source = Path(command[index]).resolve(strict=True)
    target = output / 'XPolicyLab'
    shutil.copytree(source, target,
                    ignore=shutil.ignore_patterns('policy', '__pycache__', '.git'))
    (target / 'policy').mkdir()
    (target / 'policy' / '__init__.py').write_text('')
    command[index] = str(target)
    return command


class Model:
    def __init__(self, configuration):
        self.process = self.client = self.log = None
        self.closed = False
        command = configuration.get('command')
        files = configuration.get('files', {})
        if not isinstance(command, list) or not command or not files:
            raise ValueError('A frozen command and dependency hashes are required')
        for name, expected in files.items():
            if file_digest(Path(name)) != expected:
                raise ValueError('Code skill dependency changed: ' + name)
        self.output = Path(configuration['output']) / uuid.uuid4().hex
        self.output.mkdir(parents=True, exist_ok=False)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        values = {'port': str(port), 'output': str(self.output)}
        def expand(value):
            for key, replacement in values.items():
                value = value.replace('{' + key + '}', replacement)
            return value
        env = dict(os.environ)
        env.update({k: expand(v) for k, v in configuration.get('environment', {}).items()})
        try:
            self.log = (self.output / 'service.log').open('w')
            launch = isolate_transport([expand(v) for v in command], self.output)
            self.process = subprocess.Popen(launch,
                cwd=configuration['cwd'], env=env, stdout=self.log,
                stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + configuration.get('startup_timeout_s', 600)
            while True:
                if self.process.poll() is not None:
                    raise RuntimeError('Code skill exited during startup; inspect ' + str(self.output))
                try:
                    with socket.create_connection(('127.0.0.1', port), timeout=0.2):
                        break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Code skill startup timed out: ' + str(self.output))
                    time.sleep(0.1)
            from client_server.ws.model_client import WsModelClient
            self.client = WsModelClient(url=f'ws://127.0.0.1:{port}',
                evaluation_id=self.output.name, trial_id=self.output.name,
                max_connect_seconds=configuration.get('startup_timeout_s', 600),
                request_timeout_s=configuration.get('request_timeout_s', 120))
            self.reset()
        except BaseException:
            self.close()
            raise

    def _call(self, method, obs=None):
        if self.closed or self.process.poll() is not None:
            raise RuntimeError('Code skill service is not running')
        return self.client.call(func_name=method, **({'obs': obs} if obs is not None else {}))

    def update_obs_batch(self, observations):
        return self._call('update_obs_batch', observations)

    def get_action_batch(self, indices):
        chunks = self._call('get_action_batch', indices)
        # Preserve the original primitive client's gripper aliases at the
        # standard XPolicyLab boundary; joint and EEF coordinates are unchanged.
        aliases = {'left_gripper': 'left_ee_joint_state', 'right_gripper': 'right_ee_joint_state'}
        result = []
        for chunk in chunks:
            actions = []
            for action in chunk:
                normalized = {}
                for key, value in action.items():
                    target = aliases.get(key, key)
                    if target in normalized:
                        raise ValueError('Conflicting code skill action keys')
                    normalized[target] = value
                actions.append(normalized)
            result.append(actions)
        return result

    def reset(self):
        return self._call('reset')

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.client is not None:
                self.client.close()
        finally:
            try:
                if self.process is not None and self.process.poll() is None:
                    os.killpg(self.process.pid, signal.SIGTERM)
                    try:
                        self.process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(self.process.pid, signal.SIGKILL)
                        self.process.wait(timeout=10)
            finally:
                if self.log is not None:
                    self.log.close()
