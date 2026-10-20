"""RGB-D policy with Joint14 + FK EEF state and jointly predicted EEF actions."""
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


class CRWMPolicy:
    """Returns absolute left joints/gripper followed by right joints/gripper."""

    def __init__(self, checkpoint_path: str | Path, *, model_config_path: str | Path | None = None,
                 normalization_path: str | Path | None = None, device: str = "cuda",
                 num_inference_steps: int = 4, replan_steps: int = 32, seed: int = 42):
        if not 1 <= int(replan_steps) <= 32:
            raise ValueError("replan_steps must be in [1,32]")
        if int(num_inference_steps) < 1:
            raise ValueError("num_inference_steps must be positive")
        self.model, self.normalizer, self.condition_id, self._time_kwargs = load_runtime(
            checkpoint_path, model_config_path=model_config_path, normalization_path=normalization_path,
            device=device)
        self.num_inference_steps = int(num_inference_steps)
        self.replan_steps = int(replan_steps)
        self.seed = int(seed)
        self.pending_actions: deque[np.ndarray] = deque()
        self.reset()

    def reset(self):
        self.pending_actions.clear()
        self._observation = None
        self._prompt = None
        self._context = self._context_mask = None
        self.replan_count = 0
        self.inference_seconds = 0.0
        self.last_prediction = None

    def observe(self, observation: dict[str, Any]):
        """Keep the latest acquisition as the anchor; this checkpoint needs no past frames."""
        self._observation = observation

    def should_request_observation(self) -> bool:
        return not self.pending_actions

    @torch.inference_mode()
    def predict_with_eef(self, observation: dict[str, Any] | None, instruction: str) -> dict:
        """Predict Joint14 and raw-link absolute EEF poses from one fixed anchor."""
        if observation is None:
            observation = self._observation
        if observation is None:
            raise ValueError("A current observation is required before inference")
        state, valid, anchor = self.normalizer.encode_observation(observation)
        device, dtype = self.model.device, self.model.torch_dtype
        inputs = prepare_observation(observation, device=device, dtype=dtype)
        prompt = PROMPT.format(task=instruction)
        if prompt != self._prompt:
            self._context, self._context_mask = self.model.encode_prompt(prompt)
            self._prompt = prompt
        started = time.perf_counter()
        out = self.model.infer_action(
            **inputs, context=self._context, context_mask=self._context_mask,
            proprio=torch.from_numpy(state).to(device),
            state_valid=torch.from_numpy(valid)[None].to(device),
            action_valid=torch.from_numpy(np.broadcast_to(self.normalizer.action_valid, (1, 32, 90)).copy()).to(device),
            condition_id=torch.tensor([self.condition_id], device=device),
            action_horizon=32, num_video_frames=9, num_inference_steps=self.num_inference_steps,
            seed=self.seed, **self._time_kwargs)
        action = out["action"].detach().float().cpu().numpy()
        if action.ndim == 3 and action.shape[0] == 1:
            action = action[0]
        self.inference_seconds += time.perf_counter() - started
        self.replan_count += 1
        decoded = self.normalizer.decode_with_eef(action[:self.replan_steps], anchor)
        self.last_prediction = decoded
        return decoded

    def predict(self, observation: dict[str, Any] | None, instruction: str) -> np.ndarray:
        """Keep the executable Joint14 API; EEF is exposed by predict_with_eef()."""
        return self.predict_with_eef(observation, instruction)["joint14"]

    def action(self, observation: dict[str, Any] | None, instruction: str) -> np.ndarray:
        """Consume one queued native command, replanning only after the chunk ends."""
        if not self.pending_actions:
            self.pending_actions.extend(self.predict(observation, instruction))
        return self.pending_actions.popleft().copy()
