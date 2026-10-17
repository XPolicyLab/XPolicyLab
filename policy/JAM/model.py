"""Real Piper X adapter shared by separately trained JAM (Wan) and FastWAM."""
import json
import os
from pathlib import Path
import sys

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import get_robot_action_dim_info, pack_robot_state, unpack_robot_state
from .delivery import load_deployment, validate_overrides, verify_assets

POLICY_DIR = Path(__file__).resolve().parent


def observation_for_policy(obs, dimensions):
    """The server has already decoded RGB; retain its pixels and camera order."""
    result = {'qpos': pack_robot_state(obs, 'joint', dimensions, source_type='obs').astype(np.float32)}
    for destination, source in [('cam_high', 'cam_head'), ('cam_left_wrist', 'cam_left_wrist'),
                                ('cam_right_wrist', 'cam_right_wrist')]:
        image = np.asarray(obs['vision'][source]['color'])
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(f'{source}: expected decoded uint8 HWC RGB, got {image.shape}/{image.dtype}')
        result[destination] = image
    instruction = obs.get('instruction', obs.get('instructions', obs.get('task_instruction')))
    if isinstance(instruction, (list, tuple)):
        instruction = instruction[0] if instruction else None
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('A real task instruction is required')
    return result, instruction.strip()


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        if model_cfg.get('action_type') != 'joint' or model_cfg.get('env_cfg_type') != 'piper_x':
            raise ValueError('This checkpoint supports Piper X joint targets only')
        self.dimensions = get_robot_action_dim_info(model_cfg['env_cfg_type'])
        checkpoint_root = resolve_checkpoint_root(model_cfg, POLICY_DIR / 'checkpoints')
        manifest_path = checkpoint_root / 'deployment.json' if checkpoint_root.is_dir() else checkpoint_root
        if manifest_path.name != 'deployment.json':
            raise ValueError('Pass the deployment bundle directory, containing deployment.json')
        manifest = load_deployment(manifest_path)
        validate_overrides(model_cfg, manifest)
        verify_assets(method=manifest['method'])
        if manifest['robot'] != 'piper_x' or manifest['action_dim'] != sum(self.dimensions['arm_dim'] + self.dimensions['ee_dim']):
            raise ValueError('Checkpoint/robot dimension mismatch')
        code_root = Path(manifest['code_root'])
        for path in (str(code_root), str(code_root / 'src'), str(code_root / 'scripts')):
            if path not in sys.path:
                sys.path.insert(0, path)
        from robodojo_piperx_infer import load_policy
        self.policy = load_policy(manifest)
        self.replan_steps = int(manifest['replan_steps'])
        if not 1 <= self.replan_steps <= self.policy.action_horizon:
            raise ValueError('Invalid replan length')
        self.reset()

    def update_obs(self, obs):
        self.current = observation_for_policy(obs, self.dimensions)

    def update_obs_batch(self, obs_list):
        if not obs_list:
            raise ValueError('Empty batch')
        self.batch = {int(obs['env_idx']): observation_for_policy(obs, self.dimensions) for obs in obs_list}

    def _predict(self, prepared):
        actions, self.last_latency_s = self.policy.predict(*prepared)
        return unpack_robot_state(actions[:self.replan_steps], 'joint', self.dimensions, source_type='obs')

    def get_action(self):
        if self.current is None:
            raise RuntimeError('update_obs must precede get_action')
        return self._predict(self.current)

    def get_action_batch(self, env_idx_list=None):
        ids = list(self.batch) if env_idx_list is None else list(env_idx_list)
        return [self._predict(self.batch[int(idx)]) for idx in ids]

    def reset(self):
        self.current, self.batch, self.last_latency_s = None, {}, None
