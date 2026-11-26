"""Joint T5 and VLM conditioning with visual history."""
from collections import OrderedDict
import numpy as np
import torch
from .robodojo_history_model import RoboDojoLeRobotMemoryModel
from .robodojo_model import _rgb_camera


def history_ids(anchor):
    """Real past control steps, oldest first, sampled at the trained 25-step period."""
    return [anchor - offset for offset in range(500, 0, -25) if anchor >= offset]


class RoboDojoMemoryPolicy(RoboDojoLeRobotMemoryModel):
    _expected_mode = 'auto'

    def _validate_checkpoint_contract(self, config):
        expected = self.model_cfg.get('vlm_conditioning', 'joint_t5_vlm')
        if (expected != 'joint_t5_vlm' or config.wam.plugin != 'wan22'
                or config.condition_router.auto.feature_mode != expected
                or self.model_cfg.get('history_capture_mode') != 'dense'):
            raise ValueError('VLM adapter requires a 5B joint_t5_vlm checkpoint and dense observations')
        trained_color = config.condition_router.auto.vlm.parameters.get('input_color_order')
        if self.model_cfg.get('vlm_input_color_order', 'rgb') != trained_color:
            raise ValueError('vlm_input_color_order must explicitly match the checkpoint color contract')
        super()._validate_checkpoint_contract(config)

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self._vlm_frames = {}

    def update_obs_batch(self, obs_list):
        staged = []
        for fallback, obs in enumerate(obs_list):
            env = int(obs.get('env_idx', fallback))
            frame = _rgb_camera(obs, 'cam_head')
            if frame.dtype != np.uint8:
                raise ValueError('VLM requires native uint8 RGB input')
            staged.append((env, obs['wsp_control_step'], frame.copy()))
        super().update_obs_batch(obs_list)
        for env, step, frame in staged:
            frames = self._vlm_frames.setdefault(env, OrderedDict())
            if step not in frames:
                frames[step] = frame
            while frames and next(iter(frames)) < step - 500:
                frames.popitem(last=False)

    def _adapt_observation(self, env_idx, *, cameras=None):
        adapted = super()._adapt_observation(env_idx, cameras=cameras)
        if self.model_cfg.get('vlm_conditioning', 'joint_t5_vlm') == 'joint_t5_vlm':
            # Generic AUTO adapters only populate the VLM prompt. Fusion also
            # needs the same raw instruction for T5, as during training.
            instructions = adapted.prompts.vlm_planning_text
            if not instructions:
                raise ValueError('T5 + VLM evaluation requires a raw task instruction')
            adapted.prompts.language_instruction = list(instructions)
        step = self._observation_index_by_env[env_idx]
        frames = self._vlm_frames[env_idx]
        current = torch.from_numpy(frames[step]).permute(2,0,1)
        history = torch.zeros(1,20,*current.shape,dtype=torch.uint8,device=self.device)
        valid = torch.zeros(1,20,dtype=torch.bool,device=self.device)
        ids = history_ids(step)
        if ids:
            history[0,:len(ids)] = torch.from_numpy(np.stack([frames[i] for i in ids])).permute(0,3,1,2).to(self.device)
            valid[0,:len(ids)] = True
        adapted.observation.head_view = current[None,None].to(self.device)
        adapted.observation.vlm_history_images = history
        adapted.observation.vlm_history_mask = valid
        return adapted

    def commit_action(self, execution=None):
        super().commit_action(execution)
        if execution:
            for env in execution.get('finished_env_idx_list', []):
                self._vlm_frames.pop(int(env), None)

    def discard_action(self, execution=None):
        super().discard_action(execution)
        if execution and execution.get('execution_started', False):
            self._vlm_frames = {}
