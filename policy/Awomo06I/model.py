"""XPolicyLab adapter for Awomo-0.6I RoboTwin joint-action inference."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import get_robot_action_dim_info, pack_robot_state, unpack_robot_state

POLICY_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = POLICY_DIR / 'Awomo06I'


def _enabled(value: Any) -> bool:
    return str(value or '').strip().lower() in {'1', 'true', 'yes', 'on'}


def _image(vision: dict, key: str) -> np.ndarray:
    value = np.asarray(vision[key]['color'])
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f'{key} must be a decoded HWC uint8 RGB image')
    return np.ascontiguousarray(value)


def _instruction(obs: dict, fallback: str) -> str:
    value = obs.get('instruction', obs.get('instructions', obs.get('task_instruction', fallback)))
    if isinstance(value, (tuple, list)):
        value = value[0] if value else fallback
    text = str(value or fallback).strip()
    if not text:
        raise ValueError('RoboTwin instruction is required')
    return text


def _require_path(value: Any, label: str) -> Path:
    if value is None or not str(value).strip():
        raise ValueError(f'{label} is required; see policy/Awomo06I/README.md')
    path = Path(str(value)).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f'{label} does not exist: {path}')
    return path


def _verify_runtime(runtime: Path, manifest_path: Path, expected_count: int = 48) -> None:
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict) or len(manifest) != expected_count:
        raise ValueError(f'expected {expected_count} historical runtime hashes')
    runtime_source = (runtime / 'Automodel').resolve()
    for name, expected in manifest.items():
        path = (runtime_source / name).resolve()
        if not path.is_relative_to(runtime_source) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'historical runtime file missing or different: {name}')


def _make_policy(cfg: dict, checkpoint: Path):
    runtime = _require_path(cfg.get('runtime_root') or os.environ.get('AWOMO06I_RUNTIME_ROOT'), 'runtime_root')
    base = _require_path(cfg.get('flux2_base_dir') or os.environ.get('AWOMO06I_FLUX2_BASE'), 'flux2_base_dir')
    qwen = _require_path(cfg.get('qwen_vl_path') or os.environ.get('AWOMO06I_QWEN_VL'), 'qwen_vl_path')
    for path in (runtime / 'Automodel', base / 'transformer', base / 'vae', checkpoint / 'config.yaml'):
        if not path.exists():
            raise FileNotFoundError(f'required model asset is missing: {path}')
    _verify_runtime(runtime, RUNTIME_DIR / 'runtime_snapshot' / 'SHA256.json')
    from .Awomo06I.scripts.verify_checkpoint import verify_checkpoint
    verify_checkpoint(checkpoint)
    os.environ.update({'FLUX2_TRANSFORMER_DIR': str(base / 'transformer'),
                       'FLUX2_VAE_DIR': str(base / 'vae'), 'FLUX2_BASE_DIR': str(base),
                       'QWEN3_VL_DIR': str(qwen)})
    if str(runtime / 'Automodel') not in sys.path:
        sys.path.insert(0, str(runtime / 'Automodel'))
    from .Awomo06I.scripts.server import RobotTwinBackend
    out = Path(tempfile.mkdtemp(prefix='awomo06i-inference-'))
    config = {'checkpoint': str(checkpoint), 'model_config': str(checkpoint / 'config.yaml'),
              'runtime_root': str(runtime)}
    return RobotTwinBackend(out, config)


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        cfg = dict(model_cfg)
        self.action_type = cfg.get('action_type') or 'joint'
        if self.action_type != 'joint':
            raise ValueError('Awomo-0.6I RoboTwin supports joint actions only')
        self.robot_info = get_robot_action_dim_info(cfg['env_cfg_type'])
        dim = sum(self.robot_info['arm_dim']) + sum(self.robot_info['ee_dim'])
        if dim != 14:
            raise ValueError(f'Awomo-0.6I requires dual-arm 14-D joint actions, got {dim}')
        self.default_instruction = str(cfg.get('default_instruction') or '')
        self.dummy = _enabled(cfg.get('allow_dummy_policy', False))
        self.policy = None
        if self.dummy:
            print('[Awomo06I] dummy policy enabled for protocol-only debugging')
        else:
            if not cfg.get('checkpoint_path') and os.environ.get('AWOMO06I_CHECKPOINT_PATH'):
                cfg['checkpoint_path'] = os.environ['AWOMO06I_CHECKPOINT_PATH']
            checkpoint = resolve_checkpoint_root(cfg, POLICY_DIR / 'checkpoints', policy_dir=POLICY_DIR)
            self.policy = _make_policy(cfg, checkpoint)
        self._observations = {}
        self._order = []

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        if isinstance(obs_list, dict):
            obs_list = [obs_list]
        if not obs_list:
            raise ValueError('at least one observation is required')
        self._observations = {}
        self._order = []
        for index, obs in enumerate(obs_list):
            env_idx = int(obs.get('env_idx', index))
            state = np.asarray(pack_robot_state(obs, self.action_type, self.robot_info, source_type='obs', state_type='state'), dtype=np.float32)
            if state.shape != (14,) or not np.isfinite(state).all():
                raise ValueError('RoboTwin state must contain 14 finite joint/gripper values')
            vision = obs['vision']
            self._observations[env_idx] = {
                'state': state, 'head': _image(vision, 'cam_head'),
                'left': _image(vision, 'cam_left_wrist'), 'right': _image(vision, 'cam_right_wrist'),
                'instruction': np.asarray(_instruction(obs, self.default_instruction)),
            }
            self._order.append(env_idx)

    def _infer(self, env_idx):
        if env_idx not in self._observations:
            raise ValueError('call update_obs before get_action')
        action = np.zeros((16, 14), np.float32) if self.dummy else np.asarray(self.policy.predict(self._observations[env_idx]))
        if action.shape != (16, 14) or not np.isfinite(action).all():
            raise ValueError('policy must return 16 finite 14-D actions')
        return unpack_robot_state(action[:8], self.action_type, self.robot_info, source_type='obs')

    def get_action(self):
        if not self._order:
            raise ValueError('call update_obs before get_action')
        return self._infer(self._order[0])

    def get_action_batch(self, env_idx_list=None):
        indices = self._order if env_idx_list is None else [int(i) for i in env_idx_list]
        return [self._infer(index) for index in indices]

    def reset(self):
        self._observations = {}
        self._order = []
