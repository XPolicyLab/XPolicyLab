"""One standard XPolicyLab Model, dispatching the declared Rule and WAM backends."""
from pathlib import Path
import os
import tempfile

import numpy as np
from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info

from .policy_catalog import resolve_task


def create_backend(mode, declaration, config):
    if config.get('debug_hold'):
        if os.environ.get('EVAL_ENV_TYPE') != 'debug':
            raise ValueError('debug_hold is forbidden in simulator evaluation')
        return DebugTransportPolicy(config.get('action_type', 'joint'))
    if declaration['backend'] == 'rule':
        from .rule_policy import create
    else:
        from .wam_policy import create
    return create(mode, declaration, config)


class DebugTransportPolicy:
    """Explicit synthetic interface exercise; never a model/performance result."""
    def __init__(self, action_type):
        self.action_type = action_type
        self.obs = None

    def update_obs(self, obs):
        self.obs = obs

    def get_action(self):
        if self.obs is None:
            raise RuntimeError('Missing debug observation')
        keys = [side + suffix for side in ('left', 'right')
                for suffix in (('_ee_pose', '_ee_joint_state') if self.action_type == 'ee'
                               else ('_arm_joint_state', '_ee_joint_state'))]
        action = {key: np.asarray(self.obs['state'][key]).copy() for key in keys}
        if self.action_type == 'ee':
            # The official synthetic client can send an all-zero quaternion.
            # Only this explicit transport fixture supplies a valid identity;
            # real WAM/Rule commands never pass through this code.
            for side in ('left', 'right'):
                pose = action[side + '_ee_pose'].astype(float)
                norm = np.linalg.norm(pose[3:])
                pose[3:] = pose[3:] / norm if norm > 1e-8 else [1., 0., 0., 0.]
                action[side + '_ee_pose'] = pose
        return [action]

    def reset(self):
        self.obs = None


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        super().__init__()
        from .deployment import model_options
        self.cfg = dict(model_options(), **model_cfg)
        if self.cfg.get('env_cfg_type', 'arx_x5') != 'arx_x5':
            raise ValueError('The existing policies support arx_x5')
        if self.cfg.get('strict_boundary') is False:
            raise ValueError('The public observation boundary cannot be disabled')
        if self.cfg.get('bench_name') not in (None, 'RoboDojo'):
            raise ValueError('The registered policies support RoboDojo')
        self.action_dims = get_robot_action_dim_info('arx_x5')
        if self.cfg.get('protocol', 'ws') != 'ws':
            raise ValueError('The official unified adapter uses protocol: ws')
        self.task_name = self.cfg.get('task_name') or None
        if self.task_name is not None:
            resolve_task(None, self.task_name)
        if not self.cfg.get('output_dir'):
            self.cfg['output_dir'] = str(Path(tempfile.mkdtemp(prefix='riemann-model-')) / 'policy')
        self.backend = None
        self.route = None
        self._backend_kind = None
        self._batch_index = None
        self._batch_obs = None
        self._route_generation = 0
        if self.task_name is not None:
            _, declaration = resolve_task(None, self.task_name)
            self.backend = create_backend(self.task_name, declaration, self.cfg)
            self.route = self.task_name
            self._backend_kind = declaration['backend']

    def update_obs(self, obs):
        if not isinstance(obs, dict):
            raise ValueError('Expected the official decoded observation')
        instruction = obs.get('instruction', '')
        mode, declaration = resolve_task(instruction, self.task_name)
        if self.route is not None and self.route != mode:
            raise ValueError('A running skill cannot change its task route before reset')
        if self.backend is None:
            config = self.cfg
            if self.task_name is None:
                config = dict(self.cfg, output_dir=str(Path(self.cfg['output_dir']) / ('route-%04d' % self._route_generation)))
                self._route_generation += 1
            self.backend = create_backend(mode, declaration, config)
            self.route = mode
            self._backend_kind = declaration['backend']
        # Both actors receive only the same projected RGB/own-state contract.
        from skills.shared.motion.normalized_pair_joint_execution.implementation import project_observation, STATE_WIDTHS, CAMERAS
        if self._backend_kind == 'learned':
            # The original WAM adapter consumes color (not the rule recorder's
            # colors). Keep original dtypes and leave EE/image validation to it.
            frame = {'state': {key: np.asarray(value).copy() for key, value in obs.get('state', {}).items()
                               if key in STATE_WIDTHS}, 'vision': {}}
            for name in CAMERAS:
                if name not in obs.get('vision', {}):
                    continue
                view = obs['vision'][name]
                image = view.get('color', view.get('colors')) if isinstance(view, dict) else view
                frame['vision'][name] = {'color': np.asarray(image).copy()}
        else:
            frame = project_observation(obs)
        frame['instruction'] = instruction
        self.backend.update_obs(frame)

    def get_action(self):
        if self.backend is None:
            raise RuntimeError('update_obs must precede get_action')
        actions = self.backend.get_action()
        if not isinstance(actions, list):
            raise ValueError('An official action chunk must be a list')
        # Preserve the native joint/EE protocol and numerical values. No IK,
        # interpolation, gripper conversion or image decode belongs here.
        from skills.shared.orchestration.public_rule_policy_interface.implementation import normalized_action
        result = []
        for action in actions:
            if self._backend_kind == 'rule':
                normalized_action(action)
            result.append({key: np.asarray(value).copy() for key, value in action.items()})
        return result

    def reset(self):
        if self.backend is not None:
            self.backend.reset()
        if self.task_name is None:
            self.close()
            self.route = None
        self._batch_index = None
        self._batch_obs = None

    def update_obs_batch(self, obs_list):
        if not isinstance(obs_list, list) or len(obs_list) != 1:
            raise ValueError('This release supports one environment per policy process')
        self._batch_obs = list(obs_list)

    def get_action_batch(self, env_idx_list=None):
        if self._batch_obs is None:
            raise RuntimeError('update_obs_batch must precede get_action_batch')
        ids = list(range(len(self._batch_obs))) if env_idx_list is None else list(env_idx_list)
        if len(ids) != 1 or type(ids[0]) is not int or ids[0] < 0:
            raise ValueError('Running environment indices must align with observations')
        if self._batch_index is not None and self._batch_index != ids[0]:
            raise ValueError('Reset the policy before changing its active environment')
        self._batch_index = ids[0]
        observation = self._batch_obs[0]
        self._batch_obs = None
        self.update_obs(observation)
        return [self.get_action()]

    def close(self):
        if self.backend is not None:
            close = getattr(self.backend, 'close', None)
            if close is not None:
                close()
            self.backend = None
        self._batch_index = None
        self._batch_obs = None

    def prepare_case(self, case_meta=None):
        # Case metadata may contain evaluator-only labels/layouts. Ignore it.
        return None

    def on_trial_end(self, result=None):
        # Rewards and success never enter either online actor.
        return None
