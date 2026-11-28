"""Run the verified upstream deployment loop without reshaping its actions."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .physicalrsi_bridge import VENDOR, _module, normalize_native_action, prepare_vendor


NATIVE_TASKS = frozenset({'insert_key', 'insert_tubes'})


def plain(value):
    """Serialize evidence without changing the objects sent to the evaluator."""
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_snapshot(observation, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name, camera in observation.get('vision', {}).items():
        Image.fromarray(np.asarray(camera['color'], dtype=np.uint8)).save(directory / (name+'.png'))
    (directory / 'observation.json').write_text(json.dumps(plain({
        key: value for key, value in observation.items() if key != 'vision'})))


def load_original_deploy(task):
    if task not in NATIVE_TASKS:
        raise ValueError('Unsupported native PhysicalRSI task')
    return _module('insert_key.deploy' if task == 'insert_key' else 'rdj_insert_policy.deploy')


def create_original_model(task):
    """Load the release model in the ordinary policy server, with no IK fallback."""
    if task not in NATIVE_TASKS:
        raise ValueError('Unsupported native PhysicalRSI task')
    prepare_vendor()
    descriptor = json.loads((VENDOR / 'runtime/descriptors' / (task+'.json')).read_text())
    output = Path(os.environ['COAP_EVIDENCE_DIR']) / 'native_policy_artifacts'
    output.mkdir(parents=True, exist_ok=True)
    for key, value in descriptor['environment'].items():
        os.environ[key] = value.replace('{output}', str(output)).replace('{assets}', str(VENDOR))
    if task == 'insert_key':
        os.environ['IK_RESULT'] = str(output)
        module = _module('insert_key.model')
        model = module.Model()
    else:
        module = _module('rdj_rgb_adapters.insert_tubes')
        model = module.Model(deepcopy(descriptor['model_config']))
    source = Path(module.__file__).resolve()
    metadata = {
        'task': task, 'execution_backend': 'original_model_and_original_batch_deploy',
        'action_type': descriptor['model_config']['action_type'],
        'loaded_model_source': str(source.relative_to(VENDOR)),
        'loaded_model_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'original_descriptor_source': descriptor['implementation']['source'],
        'original_source_sha256': descriptor['implementation']['sha256'],
        'original_timing_preserved': True, 'original_task_geometry_preserved': True,
        'extra_coap_ik': False, 'extra_coap_motion_limits': False,
        'extra_coap_geometry_gate': False, 'extra_coap_tracking_gate': False,
        'ik_fallback_allowed': False, 'policy_reads_simulator_truth': False,
    }
    if metadata['loaded_model_sha256'] != metadata['original_source_sha256']:
        raise ValueError('Native model does not match the fixed upstream descriptor')
    try:
        (Path(os.environ['COAP_EVIDENCE_DIR']) / 'native_model_metadata.json').write_text(
            json.dumps(metadata, indent=2)+'\n')
    except OSError as error:
        metadata['evidence_write_error'] = type(error).__name__+': '+str(error)
    return model, metadata


def tracking_diagnostics(observation, action, previous):
    """Record public target errors without turning ordinary lag into a failure."""
    report = {'diagnostics_only': True, 'extra_geometry_gate_applied': False}
    state = observation['state']
    joint = 'left_arm_joint_state' in action
    report['action_type'] = 'joint' if joint else 'ee'
    if joint:
        report['next_target_lead_rad'] = max(float(np.max(np.abs(
            np.asarray(action[arm+'_arm_joint_state']) - np.asarray(state[arm+'_arm_joint_state']))))
            for arm in ('left', 'right'))
        if previous is not None:
            report['previous_target_tracking_error_rad'] = max(float(np.max(np.abs(
                np.asarray(previous[arm+'_arm_joint_state']) - np.asarray(state[arm+'_arm_joint_state']))))
                for arm in ('left', 'right'))
    else:
        report['next_eef_target_distance_m'] = max(float(np.linalg.norm(
            np.asarray(action[arm+'_ee_pose'])[:3] - np.asarray(state[arm+'_ee_pose'])[:3]))
            for arm in ('left', 'right'))
        if previous is not None:
            report['previous_eef_target_distance_m'] = max(float(np.linalg.norm(
                np.asarray(previous[arm+'_ee_pose'])[:3] - np.asarray(state[arm+'_ee_pose'])[:3]))
                for arm in ('left', 'right'))
    return report


def diagnose_measured_geometry(records, stride=25):
    """Sample recorded public joints offline; never gate or predict execution."""
    from .kinematics import X5Kinematics
    from .motion import TABLE_Z
    from .robot_motion_geometry import check_joint_path
    if not records or type(stride) is not int or stride < 1:
        raise ValueError('Recorded public states and a positive sample stride are required')
    kinematics = X5Kinematics(records[0]['state'])
    indices = sorted(set(range(0, len(records), stride)) | {len(records)-1})
    samples = []
    for index in indices:
        joints = np.concatenate([np.asarray(records[index]['state'][arm+'_arm_joint_state'])
                                 for arm in ('left', 'right')])
        geometry = check_joint_path(kinematics, np.stack([joints, joints]), TABLE_Z)
        samples.append({'action_index': records[index]['index'], 'geometry': geometry})
    return {'schema': 'coap.native-measured-geometry-diagnostic/v1',
        'diagnostics_only': True, 'affects_policy': False, 'affects_scoring': False,
        'sample_stride_actions': stride, 'sampled_states': len(samples),
        'rejected_sampled_states': sum(not row['geometry']['accepted'] for row in samples),
        'limits': 'Public measured joint samples and static COAP hull model; '
                  'a clearance rejection is not evidence of actual contact or of an illegal upstream action.',
        'samples': samples}


class NativeEnvironment:
    """Add evidence around the original batch API without adding control frames."""
    def __init__(self, environment, task, evidence_root):
        self.environment = environment
        self.task = task
        self.root = Path(evidence_root)
        self.episodes = {}
        self.sidecar_errors = []
        self.last_episode_end = False

    def __getattr__(self, name):
        return getattr(self.environment, name)

    def is_episode_end(self):
        self.last_episode_end = self.environment.is_episode_end()
        return self.last_episode_end

    def _sidecar(self, phase, operation):
        """Keep optional recording failures outside the upstream control loop."""
        try:
            return operation()
        except Exception as error:
            self.sidecar_errors.append({'phase': phase,
                'error': type(error).__name__+': '+str(error)})
            return None

    def get_obs_batch(self, indices):
        observations = self.environment.get_obs_batch(indices)
        if len(observations) != len(indices):
            raise ValueError('Native batch observation count mismatch')
        for index, observation in zip(indices, observations, strict=True):
            if index not in self.episodes:
                directory = self.root / ('episode_'+str(time.time_ns())+'_'+str(index))
                self.episodes[index] = {'directory': directory, 'actions': 0,
                    'previous_action': None, 'observation': observation}
                self._sidecar('episode_directory', lambda: directory.mkdir(parents=True, exist_ok=False))
                self._sidecar('initial_snapshot', lambda: save_snapshot(observation, directory / 'initial'))
            episode = self.episodes[index]
            episode['observation'] = observation
            episode['observation_pending'] = True
            if episode['actions'] and episode['actions'] % 25 == 0:
                self._sidecar('snapshot', lambda: save_snapshot(observation,
                    episode['directory'] / ('step_'+str(episode['actions']).zfill(4))))
        return observations

    def take_action_batch(self, actions, indices):
        if len(actions) != len(indices):
            raise ValueError('Native batch action count mismatch')
        for action, index in zip(actions, indices, strict=True):
            episode = self.episodes[index]
            if not episode.get('observation_pending'):
                raise ValueError('Native action has no fresh original observation')
            # Validate the wire contract, but send the original object unchanged.
            normalize_native_action(action)
            expected = 'left_ee_pose' if self.task == 'insert_key' else 'left_arm_joint_state'
            if expected not in action:
                raise ValueError('Native action representation differs from its descriptor')
            self._sidecar('trace', lambda: self._record_action(episode, action))
        result = self.environment.take_action_batch(actions, indices)
        for action, index in zip(actions, indices, strict=True):
            episode = self.episodes[index]
            episode['previous_action'] = deepcopy(action)
            episode['actions'] += 1
            episode['observation_pending'] = False
        return result

    def _record_action(self, episode, action):
        diagnostics = self._sidecar('tracking_diagnostics', lambda: tracking_diagnostics(
            episode['observation'], action, episode['previous_action'])) or {}
        record = {'index': episode['actions'], 'action': plain(action),
            'stage': self.task+'_physicalrsi_native',
            'state': plain(episode['observation']['state']),
            'motion_plan': {'mode': 'physicalrsi_original_deploy',
                'phase': 'original_action_passthrough', 'native_timing_preserved': True,
                'model_action_rate_preserved': True, **diagnostics},
            'memory': {'physicalrsi_backend': {
                'execution_backend': 'original_model_and_original_batch_deploy',
                'original_timing_preserved': True, 'extra_coap_ik': False,
                'extra_coap_motion_limits': False, 'policy_reads_simulator_truth': False}}}
        with (episode['directory'] / 'trace.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')

    def _finish_episode(self, episode, error):
        directory = episode['directory']
        terminal = {'index': episode['actions'],
            'finish_kind': 'policy_error' if error else 'environment_end',
            'stage': self.task+'_physicalrsi_native',
            'state': plain(episode['observation']['state']),
            'last_action': plain(episode['previous_action']),
            'observation_phase': 'before_last_native_action',
            'memory': {'native_deployment': True, 'extra_control_frames': 0}}
        (directory / 'terminal_policy_state.json').write_text(json.dumps(terminal)+'\n')
        if error is not None:
            (directory / 'policy_error.json').write_text(json.dumps({
                **terminal, 'error': type(error).__name__+': '+str(error)})+'\n')
            return
        (directory / 'execution_complete.json').write_text(json.dumps({
            'reached_episode_end': bool(self.last_episode_end),
            'finish_kind': 'environment_end', 'policy_actions': episode['actions'],
            'settling_actions': 0, 'total_actions': episode['actions'],
            'original_deployment_loop': True,
            'development_truth_enabled': os.environ.get('COAP_DEVELOPMENT_TRUTH') == '1'})+'\n')

    def finish(self, error=None):
        for episode in self.episodes.values():
            self._sidecar('terminal_record', lambda: self._finish_episode(episode, error))
        if self.sidecar_errors:
            self._sidecar('sidecar_error_record', lambda: (self.root /
                ('native_sidecar_errors_'+str(time.time_ns())+'.json')).write_text(
                    json.dumps(self.sidecar_errors, indent=2)+'\n'))


def evaluate_native_episode(environment, model_client, task, evidence_root):
    """Delegate observation filtering, reset and stepping to the upstream file."""
    if task not in NATIVE_TASKS:
        raise ValueError('Unsupported native PhysicalRSI task')
    wrapper = NativeEnvironment(environment, task, evidence_root)
    deployment = load_original_deploy(task)
    try:
        deployment.eval_one_episode_batch(wrapper, model_client)
    except Exception as error:
        wrapper.finish(error)
        raise
    wrapper.finish()
