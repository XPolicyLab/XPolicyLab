"""Owned code-skill process using a private pipe and the installed framework."""
import json
from multiprocessing import Pipe
import os
from pathlib import Path
import signal
import subprocess
import uuid

from PhysicalRSI_core.infra.storage import file_digest


def worker_configuration(configuration, output):
    """Read the explicit worker descriptor from a frozen launch declaration."""
    command = configuration['command']
    if len(command) < 2:
        raise ValueError('A descriptor-based code skill command is required')
    options = {}
    for index in range(2, len(command), 2):
        if index + 1 >= len(command) or command[index] in options:
            raise ValueError('Malformed code skill launch declaration')
        options[command[index]] = command[index + 1]
    required = {'--descriptor', '--assets', '--output', '--python', '--port'}
    if not required <= options.keys() or set(options) - required - {'--framework-root', '--xpolicylab'}:
        raise ValueError('Migrate the code skill to an explicit worker descriptor')
    framework = options.get('--framework-root', options.get('--xpolicylab'))
    if not framework:
        raise ValueError('The installed framework is required')
    for path in (command[1], options['--descriptor']):
        if path not in configuration['files']:
            raise ValueError('Worker launcher and descriptor must be verified dependencies')
    return dict(launcher=command[1], descriptor=options['--descriptor'],
                assets=options['--assets'], framework=framework, output=str(output),
                startup_timeout_s=configuration.get('startup_timeout_s', 600))


class Model:
    def __init__(self, configuration):
        self.process = self.pipe = self.log = None
        self.closed = self.failed = False
        # Batch initialization can create several isolated code-policy models.
        # Match the declared public client budget instead of timing out inside
        # the worker while that public request is still allowed to run.
        self.timeout = configuration.get('request_timeout_s', 1800)
        command = configuration.get('command')
        files = configuration.get('files', {})
        if not isinstance(command, list) or not command or not files:
            raise ValueError('A frozen command and dependency hashes are required')
        for name, expected in files.items():
            if file_digest(Path(name)) != expected:
                raise ValueError('Code skill dependency changed: ' + name)
        self.output = Path(configuration['output']) / uuid.uuid4().hex
        settings = worker_configuration(configuration, self.output)
        self.output.mkdir(parents=True, exist_ok=False)
        settings_path = self.output / 'process.json'
        settings_path.write_text(json.dumps(settings, indent=2) + '\n')
        env = dict(os.environ)
        env.update({key: value.replace('{output}', str(self.output))
                    for key, value in configuration.get('environment', {}).items()})
        env['PYTHONUNBUFFERED'] = '1'
        parent, child = Pipe()
        self.pipe = parent
        try:
            self.log = (self.output / 'service.log').open('w')
            self.process = subprocess.Popen(
                [command[0], '-u', str(Path(__file__).with_name('program_worker.py')),
                 str(settings_path), str(child.fileno())],
                pass_fds=(child.fileno(),), cwd=configuration['cwd'], env=env,
                stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
            child.close()
            self._receive(settings['startup_timeout_s'])
            self.reset()
        except BaseException:
            child.close()
            self.close()
            raise

    def _receive(self, timeout):
        if not self.pipe.poll(timeout):
            raise TimeoutError('Code skill response timeout: ' + str(self.output))
        status, value = self.pipe.recv()
        if status != 'ok':
            raise RuntimeError(value)
        return value

    def _call(self, method, obs=None):
        if self.closed or self.failed or self.process.poll() is not None:
            raise RuntimeError('Code skill worker is not running')
        try:
            self.pipe.send((method, () if obs is None else (obs,)))
            return self._receive(self.timeout)
        except BaseException:
            self.failed = True
            raise

    def update_obs_batch(self, observations):
        # Frozen programs control fixed-base manipulators. Newer clients also
        # include optional mobile-base metadata, outside that input contract.
        prepared = [dict(obs, state={key: value for key, value in obs['state'].items()
                                    if key != 'mobile'})
                    if isinstance(obs.get('state'), dict) and 'mobile' in obs['state']
                    else obs for obs in observations]
        return self._call('update_obs_batch', prepared)

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
            if self.process is not None and self.process.poll() is None:
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait(timeout=10)
        finally:
            if self.pipe is not None:
                self.pipe.close()
            if self.log is not None:
                self.log.close()
