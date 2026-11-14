from __future__ import annotations
from collections import deque
from pathlib import Path
import time
from typing import Any
import numpy as np
import torch
from .processing import prepare_observation
from .runtime import load_runtime
PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"

class CloudRoboPolicy:

    def __init__(self, checkpoint_path: str | Path, *, model_config_path: str | Path | None=None, normalization_path: str | Path | None=None, asset_root: str | Path | None=None, device: str='cuda', num_inference_steps: int=4, replan_steps: int=32, seed: int=42):
        if int(replan_steps) != 32:
            raise ValueError('replan_steps must equal 32')
        if int(num_inference_steps) != 4:
            raise ValueError('num_inference_steps must equal 4')
        self.model, self.normalizer, self.condition_id, self._time_kwargs = load_runtime(checkpoint_path, model_config_path=model_config_path, normalization_path=normalization_path, asset_root=asset_root, device=device)
        self.num_inference_steps = int(num_inference_steps)
        self.replan_steps = int(replan_steps)
        self.seed = int(seed)
        self.pending_actions: deque[np.ndarray] = deque()
        from .history_runtime import SelectedHistory
        self.history = SelectedHistory()
        self.reset()

    def reset(self):
        self.history.clear()
        self.pending_actions.clear()
        self._observation = None
        self._prompt = None
        self._context = self._context_mask = None
        self.replan_count = 0
        self.inference_seconds = 0.0
        self.last_prediction = None

    def observe(self, observation: dict[str, Any]):
        self.history.observe(observation, device=self.model.device, dtype=self.model.torch_dtype)
        self._observation = observation

    def should_request_observation(self) -> bool:
        return not self.pending_actions

    @torch.inference_mode()
    def predict_with_eef(self, observation: dict[str, Any] | None, instruction: str) -> dict:
        if observation is None:
            observation = self._observation
        if observation is None:
            raise ValueError('A current observation is required before inference')
        state, valid, anchor = self.normalizer.encode_observation(observation)
        device, dtype = (self.model.device, self.model.torch_dtype)
        inputs = prepare_observation(observation, device=device, dtype=dtype)
        if self.history.step is None:
            self.observe(observation)
        history_inputs = self.history.inputs(inputs['input_image'])
        prompt = PROMPT.format(task=instruction)
        if prompt != self._prompt:
            self._context, self._context_mask = self.model.encode_prompt(prompt)
            self._prompt = prompt
        started = time.perf_counter()
        out = self.model.infer_action(**inputs, **history_inputs, context=self._context, context_mask=self._context_mask, proprio=torch.from_numpy(state).to(device), state_valid=torch.from_numpy(valid)[None].to(device), action_valid=torch.from_numpy(np.broadcast_to(self.normalizer.action_valid, (1, 32, 90)).copy()).to(device), condition_id=torch.tensor([self.condition_id], device=device), action_horizon=32, num_video_frames=9, num_inference_steps=self.num_inference_steps, seed=self.seed, **self._time_kwargs)
        action = out['action'].detach().float().cpu().numpy()
        if action.ndim == 3 and action.shape[0] == 1:
            action = action[0]
        self.inference_seconds += time.perf_counter() - started
        self.replan_count += 1
        decoded = self.normalizer.decode_with_eef(action[:self.replan_steps], anchor)
        self.last_prediction = decoded
        from .postprocessing.ik_device_candidate import BoundedIKDevice
        from .postprocessing.robotwin_fk import RobotWinUrdfFK
        if not hasattr(self, '_bounded_ik'):
            original=self.normalizer._load_fk()
            fk=RobotWinUrdfFK(original.urdf_path,original.embodiment_config_path,
                             expected_urdf_sha256=original.urdf_sha256)
            self._bounded_ik=BoundedIKDevice(fk,device='cpu',batch_backtracks=True)
        corrected,diagnostics=self._bounded_ik(decoded['joint14'],decoded['eef_relative'],decoded['anchor_pose'])
        self.last_raw_joint14=decoded['joint14'].copy()
        decoded=dict(decoded,joint14=corrected,ik_diagnostics=diagnostics)
        self.last_prediction=decoded
        return decoded

    def predict(self, observation: dict[str, Any] | None, instruction: str) -> np.ndarray:
        return self.predict_with_eef(observation, instruction)['joint14']

    def action(self, observation: dict[str, Any] | None, instruction: str) -> np.ndarray:
        if not self.pending_actions:
            self.pending_actions.extend(self.predict(observation, instruction))
        return self.pending_actions.popleft().copy()
