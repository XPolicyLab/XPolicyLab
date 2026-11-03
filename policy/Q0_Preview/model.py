"""Standard remote joint policy; no simulator imports or environment handles."""
import os
import threading
import uuid
from pathlib import Path
import numpy as np
from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info, get_batch_size
from .controller.backend import Backend
from .runtime.remote_observation import clean_observation
from .runtime.episode import Episode
from .runtime.task_context import task_step_limit


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        self.dims = get_robot_action_dim_info(model_cfg['env_cfg_type'])
        self.batch_size = get_batch_size(model_cfg['env_cfg_type'])
        if model_cfg['env_cfg_type'] != 'arx_x5' or model_cfg['action_type'] != 'joint':
            raise ValueError('Supported interface: arx_x5 / joint')
        if len(self.dims['arm_dim']) != 2 or self.dims['ee_dim'] != [1,1]:
            raise ValueError('Expected dual X5 arms and scalar gripper commands')
        self.debug = os.environ.get('EVAL_ENV_TYPE') == 'debug'
        self.control_step_limit = (None if self.debug else task_step_limit(
            self.model_cfg.get('task_name'), self.model_cfg.get('max_control_steps')))
        self.backend = self.worker = self.episode = self.obs = None
        self.fresh = False

    def reset(self):
        if self.backend is not None:
            self.backend.cancel()
        if self.worker is not None:
            self.worker.join(timeout=30)
            if self.worker.is_alive():
                raise RuntimeError('Previous policy worker did not stop; refusing overlapping episodes')
        if self.backend is not None:
            self.backend.close()
        self.backend = self.worker = self.episode = self.obs = None
        self.fresh = False

    def on_trial_end(self, result=None):
        # Lifecycle notification only. Never inspect or pass evaluation results to Agent.
        self.reset()

    def update_obs(self, obs):
        # The shared WS server has already decoded RGB. Debug observations have
        # placeholder poses/calibration, so only their transport/action shapes apply.
        if self.debug:
            for camera in obs['vision'].values():
                image = np.asarray(camera['color'])
                if image.ndim != 3 or image.shape[-1] != 3:
                    raise ValueError('Expected decoded RGB images')
            self.obs = {'state':{}}
            for i,a in enumerate(('left','right')):
                for part,size in [('arm',self.dims['arm_dim'][i]),('ee',self.dims['ee_dim'][i])]:
                    key=f'{a}_{part}_joint_state'
                    value=np.asarray(obs['state'][key],dtype=np.float32).reshape(-1)
                    if value.size!=size or not np.isfinite(value).all(): raise ValueError('Invalid action dimension/value')
                    self.obs['state'][key]=value.copy()
        else:
            self.obs = clean_observation(obs,self.dims)
            if self.backend is None:
                self.backend = Backend(self.obs,self.control_step_limit)
            else:
                self.backend.update(self.obs)
        self.fresh = True

    def get_action(self):
        if self.obs is None or not self.fresh:
            raise RuntimeError('A fresh update_obs is required before each get_action')
        self.fresh = False
        if self.debug:
            return [{k:v.copy() for k,v in self.obs['state'].items()}]
        if self.worker is None:
            run_root = self.model_cfg.get('run_root') or os.environ.get('Q0_PREVIEW_RUN_ROOT')
            if not run_root:
                raise ValueError('Set run_root and start tools/policy_service.py to run the Agent')
            self.episode = Episode(self.backend,Path(run_root)/uuid.uuid4().hex,self.model_cfg)
            self.worker = threading.Thread(target=self.episode.run,name='q0-preview-policy',daemon=True)
            self.worker.start()
        return [self.backend.next_action()]

    def update_obs_batch(self, obs_list):
        if len(obs_list) != 1:
            raise ValueError('Use one environment per policy server; launch independent servers for parallel trials')
        self.update_obs(obs_list[0])

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is not None and list(env_idx_list) != [0]:
            raise ValueError('Only environment index 0 is supported')
        return [self.get_action()]
