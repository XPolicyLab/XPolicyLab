"""RoboDojo observation history and inference."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from .robodojo_lerobot_model import RoboDojoLeRobotModel
from .robodojo_model import _rgb_camera


class RoboDojoLeRobotMemoryModel(RoboDojoLeRobotModel):
    _expected_observation_interval = 1
    _expected_observation_history_segments = 24
    _expected_history_memory = True

    def _validate_checkpoint_contract(self, checkpoint_config):
        capture_mode = self.model_cfg.get("history_capture_mode", "dense")
        if capture_mode not in ("dense", "chunk_boundary"):
            raise ValueError("history_capture_mode must be dense or chunk_boundary")
        # At H24 replans the nominal stride-24 sampler only uses real episode
        # frames 0,24,48,... . Capture those exact frames without rendering,
        # transporting, and resizing unused intermediate observations.
        self._history_observation_stride = 24 if capture_mode == "chunk_boundary" else 1
        self._expected_observation_interval = self._history_observation_stride
        self._expected_observation_history_segments = 24 // self._history_observation_stride
        super()._validate_checkpoint_contract(checkpoint_config)
        memory = checkpoint_config.history_memory
        if (memory.history_frames, memory.stride) != (8, 24):
            raise ValueError("History requires nine RGB history slots with stride=24")
        if checkpoint_config.wam.core.parameters.get("visual_memory") is not True:
            raise ValueError("History requires checkpoint visual_memory=true")
        self._history_config = memory

    def reset(self):
        super().reset()
        self._history_archives = {}
        self._history_prediction_steps = {}
        self._history_finished_envs = set()
        self._history_requires_reset = False

    def update_obs_batch(self, obs_list):
        from worldscape_policy.memory.visual.history_prefix import EpisodeHistoryArchive

        if self._history_requires_reset:
            raise RuntimeError("History execution was interrupted; reset the simulator episode and policy first")
        if not obs_list:
            raise ValueError("update_obs_batch requires at least one observation")
        staged = []
        envs = []
        for fallback_idx, obs in enumerate(obs_list):
            if not isinstance(obs, Mapping):
                raise TypeError("History observations must be mappings")
            env = int(obs.get("env_idx", fallback_idx))
            if env in self._history_finished_envs:
                raise ValueError("Finished History environments cannot be reused before reset()")
            step = obs.get("wsp_control_step")
            episode = obs.get("wsp_episode_id")
            if type(step) is not int or step < 0:
                raise ValueError("History requires wsp_control_step as a nonnegative integer")
            if not isinstance(episode, str) or not episode:
                raise ValueError("History requires a nonempty wsp_episode_id")
            archive = self._history_archives.get(env)
            if archive is None:
                if step != 0:
                    raise ValueError("History requires every real control step in order, starting at zero")
                archive_options = (
                    {"observation_stride": self._history_observation_stride}
                    if self._history_observation_stride != 1 else {}
                )
                archive = EpisodeHistoryArchive(self._history_config, **archive_options)
                archive.reset(episode)
                previous = -1
            else:
                if archive.episode_id != episode:
                    raise ValueError("History episode changed without reset(); refusing to reuse old history")
                previous = next(reversed(archive.frames))
            expected = 0 if previous < 0 else previous + self._history_observation_stride
            if step not in (previous, expected):
                if self._history_observation_stride == 1:
                    raise ValueError("History requires every real control step in order, starting at zero")
                raise ValueError("History requires consecutive real chunk-boundary steps 0,24,48,...")
            if step > previous:
                pending = self._pending_prediction
                if previous >= 0 and (pending is None or env not in pending.env_idx_list):
                    raise ValueError("History control steps may advance only while executing a pending action chunk")
                if pending is not None and env in pending.env_idx_list:
                    start = self._history_prediction_steps[env]
                    if step > start + pending.horizons[env]:
                        raise ValueError("History observations exceed the pending action horizon")
                views = np.stack([_rgb_camera(obs, camera) for camera in self._camera_order])
                if views.dtype != np.uint8:
                    raise ValueError("History requires native uint8 RGB camera frames")
                views = torch.from_numpy(np.ascontiguousarray(views))
            else:
                views = None
            envs.append(env)
            staged.append((env, step, obs, archive, views))
        if len(set(envs)) != len(envs):
            raise ValueError("update_obs_batch received duplicate env indices")
        for env, step, obs, archive, views in staged:
            if views is not None:
                archive.append(step, views)
                self._history_archives[env] = archive
                # The parent retains only the latest native observation. Its
                # old stride-window list is unrelated to History episode history.
                super().update_obs_batch([{**obs, "env_idx": env}])
                self._observation_index_by_env[env] = step
        # Repeated observations of the current step are idempotent, including
        # state/current-camera inputs, rather than another control tick.
        self._latest_env_idx_list = envs

    def _adapt_observation(self, env_idx, *, cameras=None):
        adapted = super()._adapt_observation(env_idx, cameras=cameras)
        archive = self._history_archives[env_idx]
        # The archive stores CPU FP16 images exactly like the training payload.
        # Keep CPU packing outside BF16 autocast (the VAE owns mixed precision).
        with torch.autocast(device_type="cpu", enabled=False):
            adapted.observation.visual_history = archive.snapshot(
                self._observation_index_by_env[env_idx], device=self.device,
            )
        return adapted

    @torch.no_grad()
    def get_action_batch(self, env_idx_list=None):
        if self._history_requires_reset:
            raise RuntimeError("History execution was interrupted; reset the simulator episode and policy first")
        envs = self._latest_env_idx_list if env_idx_list is None else [int(env) for env in env_idx_list]
        actions = super().get_action_batch(envs)
        self._history_prediction_steps = {
            env: self._observation_index_by_env[env] for env in envs
        }
        return actions

    def commit_action(self, execution=None):
        pending = self._pending_prediction
        if pending is None:
            return super().commit_action(execution)
        finished = set()
        if execution is not None:
            if not isinstance(execution, Mapping) or not isinstance(execution.get("executed_steps"), Mapping):
                raise ValueError("commit_action requires an executed_steps mapping")
            finished = {int(env) for env in execution.get("finished_env_idx_list", [])}
        for env in pending.env_idx_list:
            if env in finished:
                continue
            expected = self._history_prediction_steps[env] + pending.horizons[env]
            if self._observation_index_by_env[env] != expected:
                if self._history_observation_stride == 1:
                    raise ValueError("History commit requires the real observation after every executed control step")
                raise ValueError(f"History commit requires the real chunk-boundary observation at step {expected}")
        super().commit_action(execution)
        self._history_prediction_steps = {}
        for env in finished:
            self._history_archives.pop(env, None)
            self._latest_obs_by_env.pop(env, None)
            self._obs_history_by_env.pop(env, None)
            self._observation_index_by_env.pop(env, None)
            self._history_finished_envs.add(env)
        self._latest_env_idx_list = [env for env in self._latest_env_idx_list if env not in finished]

    def discard_action(self, execution=None):
        pending = self._pending_prediction
        super().discard_action(execution)
        if pending is None:
            return
        self._history_prediction_steps = {}
        if execution is not None and execution.get("execution_started", False):
            self._history_archives = {}
            self._history_requires_reset = True
