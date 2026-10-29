"""XPolicyLab adapter for the standalone WAM_policy runtime."""
from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path
import sys

import cv2
import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root

POLICY_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(POLICY_ROOT))
sys.path.insert(0, str(POLICY_ROOT / 'go3_nano_preview'))


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        if model_cfg['action_type'] != 'joint':
            raise ValueError('The released Robotwin checkpoint supports joint actions only')
        self.config = dict(model_cfg)
        self.dims = get_robot_action_dim_info(model_cfg['env_cfg_type'])
        if len(self.dims['arm_dim']) != 2 or len(self.dims['ee_dim']) != 2:
            raise ValueError('This checkpoint requires a dual-arm robot')
        self.keys = ('left_arm_joint_state','left_ee_joint_state',
                     'right_arm_joint_state','right_ee_joint_state')
        self.widths = (self.dims['arm_dim'][0],self.dims['ee_dim'][0],
                       self.dims['arm_dim'][1],self.dims['ee_dim'][1])
        self.camera_keys = tuple(model_cfg['camera_keys'])
        if len(self.camera_keys) != 3:
            raise ValueError('The checkpoint requires head, left-wrist, right-wrist cameras')
        self.horizon = int(model_cfg['action_horizon'])
        self.steps = int(model_cfg['num_ae_steps'])
        if self.horizon < 1 or self.steps < 1:
            raise ValueError('action_horizon and num_ae_steps must be positive')
        checkpoint = str(resolve_checkpoint_root(model_cfg,
            POLICY_ROOT/'checkpoints',policy_dir=POLICY_ROOT))
        # import_module supports the retained hyphenated service filename.
        service = import_module('go3_nano_preview.serve_GO-3')
        service._set_seed(int(model_cfg['seed']))
        self.inferrer = service.build_inferrer(
            str(POLICY_ROOT/'configs/robotwin_infer.py'), checkpoint,
            robot_name=os.environ.get('ROBOT_NAME', model_cfg['robot_name']),
            stat_path=str(POLICY_ROOT/'assets/stats/robotwin.json'),
            dense_action_steps=self.horizon)
        stats = self.inferrer._taskflow.cfg.infer
        if self.widths != (stats.joint_dim, 1, stats.joint_dim, 1):
            raise ValueError('Action profile does not match the checkpoint statistics')
        self.reset()

    def _prepare(self, obs):
        state = []
        for key,width in zip(self.keys,self.widths):
            value = np.asarray(obs['state'][key], dtype=np.float32).reshape(-1)
            if value.shape != (width,) or not np.isfinite(value).all():
                raise ValueError(f'Invalid state field {key}: expected {width} finite values')
            state.append(value)
        images=[]
        height,width=self.config['image_size']
        for key in self.camera_keys:
            frame=obs['vision'][key]['color']
            if not isinstance(frame,np.ndarray) or frame.ndim!=3 or frame.shape[-1]!=3 or frame.dtype!=np.uint8:
                raise ValueError(f'{key}.color must be a decoded uint8 RGB HWC image')
            images.append(cv2.resize(frame,(width,height),interpolation=cv2.INTER_LINEAR))
        instruction=obs['instruction']
        if not isinstance(instruction,str):
            raise TypeError('instruction must be a string')
        return {'obs':np.stack(images),'state':np.concatenate(state),'prompt':instruction,
                'num_ae_steps':self.steps,'dense_action_steps':self.horizon}

    def update_obs(self, obs):
        self.obs = self._prepare(obs)

    def update_obs_batch(self, obs_list):
        self.batch = [self._prepare(obs) for obs in obs_list]

    def _infer(self, request):
        import torch
        with torch.inference_mode():
            result = self.inferrer.infer(data=request)['result']
        values=(result['left_arm']['values'], result['left_effector'],
                result['right_arm']['values'], result['right_effector'])
        arrays=[np.asarray(v,dtype=np.float32) for v in values]
        lengths={len(v) for v in arrays}
        if len(lengths)!=1 or not lengths or next(iter(lengths)) < 1:
            raise ValueError('Model returned an empty or inconsistent action chunk')
        for key,width,array in zip(self.keys,self.widths,arrays):
            if array.ndim!=2 or array.shape[1]!=width or not np.isfinite(array).all():
                raise ValueError(f'Invalid model output for {key}')
        return [{key:array[i].copy() for key,array in zip(self.keys,arrays)}
                for i in range(min(self.horizon,len(arrays[0])))]

    def get_action(self):
        if self.obs is None:
            raise RuntimeError('Call update_obs before get_action')
        return self._infer(self.obs)

    def get_action_batch(self, env_idx_list=None):
        if self.batch is None:
            raise RuntimeError('Call update_obs_batch before get_action_batch')
        if env_idx_list is not None and len(env_idx_list)!=len(self.batch):
            raise ValueError('Active environment IDs must match the compact observation batch')
        # The released checkpoint is stateless; each environment gets its own request.
        return [self._infer(request) for request in self.batch]

    def reset(self):
        self.obs=None
        self.batch=None
