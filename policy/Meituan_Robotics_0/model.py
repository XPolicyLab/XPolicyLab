"""MR0 evaluation adapter with causal first-frame and CD-LAM memory."""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any
import sys

import numpy as np
import torch
import yaml

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)

from .deploy import (
    ALIASES, FIRST_KEY, FPS, HORIZON, SCHEMA, STRIDE, advance_state,
    frame_key, integer, model_memory, source_indices, validate_state,
)
from .runtime_config import install_paths


def _ensure_hwc_uint8(image: Any) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected an image array with ndim=3, got {image.shape}.")
    if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected three RGB channels, got {image.shape}.")
    if np.issubdtype(image.dtype, np.floating):
        image = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return image


def _extract_camera(observation: dict[str, Any], names: tuple[str, ...]) -> np.ndarray:
    vision = observation.get("vision", {})
    for name in names:
        if name not in vision:
            continue
        value = vision[name]
        if isinstance(value, dict):
            for key in ("color", "rgb", "colors"):
                if key in value:
                    return _ensure_hwc_uint8(value[key])
        else:
            return _ensure_hwc_uint8(value)
    raise KeyError(f"Missing RGB camera; tried {names}.")


def _unnormalize_q99(values: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    low = np.asarray(stats["q01"], dtype=np.float32)
    high = np.asarray(stats["q99"], dtype=np.float32)
    mask = np.asarray(stats.get("mask", np.ones_like(low)), dtype=bool)
    clipped = np.clip(values, -1.0, 1.0)
    return np.where(mask, 0.5 * (clipped + 1.0) * (high - low) + low, clipped).astype(np.float32)


def _normalize_state(values, statistics):
    """Use the checkpoint's q99 normalization and float16 state input."""
    low = np.asarray(statistics['q01'], dtype=np.float32)
    high = np.asarray(statistics['q99'], dtype=np.float32)
    normalized = np.asarray(values, dtype=np.float32).copy()
    varying = high != low
    normalized[..., varying] = (normalized[..., varying] - low[varying]) / (high[varying] - low[varying])
    normalized[..., varying] = 2 * normalized[..., varying] - 1
    return np.clip(normalized, -1, 1).astype(np.float16)


class Model(ModelTemplate):
    """Continuous 14D joint actions, with 50 predicted and 16 executed steps."""

    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        if model_cfg.get('env_cfg_type') != 'arx_x5' or model_cfg.get('action_type') != 'joint':
            raise ValueError('This checkpoint supports env_cfg_type=arx_x5 and action_type=joint')
        policy_dir = Path(__file__).resolve().parent
        asset_root = (
            os.environ.get('ROBODOJO_ASSET_ROOT')
            or model_cfg.get('asset_root')
            or policy_dir / 'checkpoints/mr0_width_130k'
        )
        paths = install_paths(asset_root, os.environ.get('ROBODOJO_RUN_ROOT') or model_cfg.get('run_root'))
        selection = dict(model_cfg)
        if not any(selection.get(key) for key in ('checkpoint_path', 'model_path', 'ckpt_path')):
            selection['checkpoint_path'] = str(paths['checkpoint'])
        self.checkpoint_path = resolve_checkpoint_root(selection, policy_dir / 'checkpoints', policy_dir=policy_dir)
        if self.checkpoint_path.resolve() != paths['checkpoint']:
            raise ValueError('The checkpoint must be checkpoints/model.pt under the asset root')

        config = yaml.safe_load((paths['assets'] / 'config.yaml').read_text())
        statistics = json.loads((paths['assets'] / 'dataset_statistics.json').read_text())
        self.data_config = config['datasets']['vla_data']
        if (self.data_config['lam_memory']['window_sec'] != 3
                or self.data_config['normalization_mode'] != 'q99'
                or config['framework']['action_model']['action_horizon'] != 50
                or config['framework']['action_model']['compute_dtype'] != 'float32'):
            raise ValueError('Inference configuration does not match this checkpoint')

        self.action_type = model_cfg['action_type']
        self.robot_action_dim_info = get_robot_action_dim_info(model_cfg['env_cfg_type'])
        if self.robot_action_dim_info['arm_dim'] != [6, 6] or self.robot_action_dim_info['ee_dim'] != [1, 1]:
            raise ValueError('This checkpoint requires dual 6-joint arms and two scalar grippers')
        self.action_dim = sum(self.robot_action_dim_info['arm_dim']) + sum(self.robot_action_dim_info['ee_dim'])
        self.action_horizon = int(config['framework']['action_model']['action_horizon'])
        self.execute_horizon = HORIZON
        self.state_stats = statistics['new_embodiment']['state']
        self.action_stats = statistics['new_embodiment']['action']
        loaded = sys.modules.get('starVLA')
        if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(paths['source']):
            raise RuntimeError('A different starVLA runtime is already imported; use a separate process')
        from starVLA.model.framework.base_framework import build_framework
        from starVLA.model.framework.share_tools import dict_to_namespace
        from starVLA.visual_history import VisualHistoryConfig, image_metadata
        from starVLA.cdlam_teacher import OnlineTeacher

        if not torch.cuda.is_available():
            raise RuntimeError('Meituan_Robotics_0 evaluation requires CUDA')
        self.device = torch.device('cuda')
        config['framework']['qwenvl']['base_vlm'] = str(paths['base_vlm'])
        config['framework']['qwenvl']['checkpoint_path'] = str(self.checkpoint_path)
        # Keep initialization order: action model first, then the frozen teacher.
        # This also preserves the generator state used by diffusion inference.
        torch.manual_seed(int(model_cfg.get('seed', 0)))
        self.model = build_framework(dict_to_namespace(config))
        weights = torch.load(self.checkpoint_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(weights, strict=True)
        del weights
        self.model = self.model.eval().to(self.device)
        self.teacher = OnlineTeacher(paths['teacher_code'], paths['teacher'], device=str(self.device), batch_size=16)
        self.lam_spec = VisualHistoryConfig.from_data_config(self.data_config, for_inference=True)
        self._image_metadata = image_metadata
        self.reset()
        print(f'[Meituan_Robotics_0] {self.checkpoint_path}; '
              'RGB main/left/right, 25Hz, abs interleaved q99, predict50/execute16, Euler10', flush=True)

    def _convert_obs(self, obs):
        images = [_extract_camera(obs, names).copy() for names in ALIASES]
        if any(image.shape != (480, 640, 3) for image in images):
            raise ValueError('This checkpoint requires original 640x480 RGB observations for all cameras')
        instruction = obs.get('instruction') or obs.get('instructions') or self.model_cfg.get('task_name', '')
        if isinstance(instruction, (tuple, list)):
            instruction = instruction[0] if instruction else ''
        raw = pack_robot_state(obs, self.action_type, self.robot_action_dim_info, source_type='obs').astype(np.float32)
        if raw.shape != (self.action_dim,) or not np.isfinite(raw).all():
            raise ValueError('Observation must contain a finite 14D interleaved joint state')
        return images, str(instruction), _normalize_state(raw, self.state_stats)[None, :]

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        from starVLA.cdlam_teacher import center_crop_resize
        from starVLA.memory_inputs import MemoryInputs
        if not obs_list:
            raise ValueError('Empty observation batch')
        indices = [int(obs.get('env_idx', 0)) for obs in obs_list]
        if len(indices) != len(set(indices)):
            raise ValueError('Each batched observation needs a unique env_idx')
        pending = []
        for index, obs in zip(indices, obs_list):
            images, language, state = self._convert_obs(obs)
            bank = self.memory_by_env.get(index)
            if bank is None:
                bank = MemoryInputs(self.data_config)
            frame, fresh, clock_mode = bank.observation_tick(obs, images[0])
            teacher_image = center_crop_resize(images[0]) if fresh else None
            if fresh:
                bank.pair(frame, teacher_image)
            pending.append((index, bank, frame, fresh, clock_mode, teacher_image, images, language, state))
        for index, bank, frame, fresh, clock_mode, teacher_image, images, language, state in pending:
            if fresh:
                bank.append(frame, images[0], teacher_image)
            bank.clock_mode = clock_mode
            self.memory_by_env[index] = bank
            self.obs_by_env[index] = (images, language, state)
        self._latest_env_idx_list = indices
        self._flush_teacher([index for index in indices if len(self.memory_by_env[index].pending_pairs) >= 32])

    def get_action(self):
        if not self._latest_env_idx_list:
            raise ValueError('Call update_obs before inference')
        return self.get_action_batch([self._latest_env_idx_list[0]])[0]

    @torch.inference_mode()
    def get_action_batch(self, env_idx_list=None):
        indices = self._latest_env_idx_list if env_idx_list is None else [int(index) for index in env_idx_list]
        if not indices or any(index not in self.obs_by_env for index in indices):
            raise ValueError('Call update_obs for every requested environment before inference')
        if len(indices) != len(set(indices)):
            raise ValueError('Requested environment IDs must be unique')
        for index in indices:
            self.memory_by_env[index].validate_action_tick(self.execute_horizon)
        self._flush_teacher(indices)
        examples = [self.memory_by_env[index].example(*self.obs_by_env[index]) for index in indices]
        chunks = self._predict_examples(examples)
        for index in indices:
            self.memory_by_env[index].last_action_frame = self.memory_by_env[index].last
        return chunks

    def _predict_examples(self, examples):
        if self.model.action_model.compute_dtype != torch.float32:
            raise ValueError('This checkpoint requires the native FP32 action head')
        # The LAM projection uses BF16; the native action head enters its own
        # FP32 context. Moving this boundary changes the evaluated policy.
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            prediction = np.asarray(
                self.model.predict_action(examples=examples)['normalized_actions'], dtype=np.float32,
            )
        expected_shape = (len(examples), self.action_horizon, self.action_dim)
        if prediction.shape != expected_shape or not np.isfinite(prediction).all():
            raise ValueError(f'Invalid predicted action chunk: {prediction.shape}')
        actions = _unnormalize_q99(prediction, self.action_stats)[:, :self.execute_horizon]
        return [unpack_robot_state(action, self.action_type, self.robot_action_dim_info, source_type='obs')
                for action in actions]

    def _flush_teacher(self, indices):
        pending = [(index, end, pair) for index in indices for end, pair in self.memory_by_env[index].pending_pairs]
        if pending:
            values = self.teacher.encode([pair for _, _, pair in pending])
            if len(values) != len(pending):
                raise ValueError('Teacher batch length differs from completed pairs')
            for (index, end, _), value in zip(pending, values):
                self.memory_by_env[index].lam.append(end, value)
            for index in indices:
                self.memory_by_env[index].pending_pairs.clear()

    @torch.inference_mode()
    def predict_lam(self, obs):
        """Infer from one complete request, including its causal memory state."""
        batched = isinstance(obs, list)
        items = obs if batched else [obs]
        if not items or any(not isinstance(item, dict) for item in items):
            raise ValueError('predict_lam needs an observation or nonempty list')
        responses = []
        # Match the validated server's ordered batch splitting and RNG use.
        for start in range(0, len(items), 4):
            responses.extend(self._predict_lam_batch(items[start:start + 4]))
        return responses if batched else responses[0]

    def _predict_lam_batch(self, items):
        indices = [integer(item.get('env_idx', 0)) for item in items]
        if len(set(indices)) != len(indices):
            raise ValueError('LAM batch env_idx values must be unique')
        prepared, pairs = [], []
        for item, index in zip(items, indices):
            meta = item.get('lam_request')
            if not isinstance(meta, dict) or meta.get('schema') != SCHEMA:
                raise ValueError(f'predict_lam requires {SCHEMA}')
            episode = meta.get('episode_id')
            if not isinstance(episode, str) or not episode:
                raise ValueError('Missing LAM episode ID')
            frame = integer(meta.get('frame_index'))
            timestamp = meta.get('timestamp_sec')
            if (not isinstance(timestamp, (int, float))
                    or not np.isfinite(timestamp)
                    or abs(timestamp-frame/FPS) > 1e-6):
                raise ValueError('LAM timestamps must use the 25 Hz simulation clock')
            state = validate_state(meta.get('state'))
            selected = source_indices(state, frame)
            if meta.get('source_frame_indices') != selected:
                raise ValueError('LAM request must carry every required source frame in order')
            source = {}
            for number in selected:
                image = item.get('vision', {}).get(frame_key(number), {}).get('color')
                if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.shape != (240, 320, 3):
                    raise ValueError(f'Missing lossless 320x240 LAM source image {number}')
                source[number] = image
            new_pairs = [(source[end-STRIDE], source[end])
                         for end in range(max(STRIDE, state['through_frame']+1), frame+1)]
            images, language, robot_state = self._convert_obs(item)
            if frame:
                first = item.get('vision', {}).get(FIRST_KEY, {}).get('color')
                if not isinstance(first, np.ndarray) or first.dtype != np.uint8 or first.shape != (480, 640, 3):
                    raise ValueError('Missing lossless native 640x480 episode-first image')
                images = [first] + images
            example = dict(image=images, lang=language, state=robot_state,
                           visual_history=asdict(self.lam_spec),
                           image_metas=self._image_metadata(self.lam_spec, frame, FPS,
                                                     ['cam_high', 'cam_left_wrist', 'cam_right_wrist']))
            prepared.append((index, episode, frame, state, example, len(new_pairs)))
            pairs.extend(new_pairs)
        values = self.teacher.encode(pairs) if pairs else []
        if len(values) != len(pairs):
            raise ValueError('LAM teacher returned an unexpected number of latents')
        responses, examples, offset = [], [], 0
        for index, episode, frame, state, example, count in prepared:
            updated = advance_state(state, frame, values[offset:offset+count])
            offset += count
            example['lam_memory'] = model_memory(updated)
            examples.append(example)
            responses.append(dict(schema=SCHEMA, env_idx=index, episode_id=episode,
                                  frame_index=frame, state=updated))
        for response, actions in zip(responses, self._predict_examples(examples)):
            response['actions'] = actions
        return responses

    def reset_envs(self, env_idx_list):
        for index in env_idx_list:
            self.memory_by_env.pop(int(index), None)
            self.obs_by_env.pop(int(index), None)
        self._latest_env_idx_list = [index for index in self._latest_env_idx_list if index in self.obs_by_env]

    def reset(self):
        self.memory_by_env = {}
        self.obs_by_env = {}
        self._latest_env_idx_list = []
