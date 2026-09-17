from __future__ import annotations

import copy
import json
import sys
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
for _path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from evals.robotwin2.adapter import RoboTwin2Adapter
from evals.robotwin2.checkpoint import load_robotwin2_checkpoint_transform
from worldscape_policy.native_builder import (
    build_wan22_policy_from_checkpoint,
    checkpoint_mode,
    checkpoint_supports_mode,
)
from worldscape_policy.rollout.session import PolicyRuntime
from worldscape_policy.types import InteractionMode, WorldActionOutput

DEFAULT_ACTION_HORIZON = 24
JOINT_DIM = 14
OBSERVATION_HISTORY_FRAMES = 9
DEFAULT_VLM_COT_PROMPT = (
    "You are a robot planner. Instructions: {task}. Given the current high-level "
    "task instruction and current head-view observation, predict the next atomic "
    "action subtask for the next second."
)
DEFAULT_T5_PROMPT_TEMPLATE = (
    "A video shows that a robot {instruction} The robot {instruction}"
)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return bool(value)


class WSP2RoboTwinPolicy:
    """RoboTwin ``eval_policy.py`` bridge backed by one persistent WSP2 model."""

    def __init__(self, usr_args: dict[str, Any]) -> None:
        checkpoint = Path(str(usr_args["ckpt_setting"])).expanduser().resolve()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"WSP2 checkpoint not found: {checkpoint}")
        self.device = str(usr_args.get("device", "cuda"))
        primary_mode = checkpoint_mode(checkpoint, validate_artifacts=False)
        self.mode = InteractionMode.parse(usr_args.get("mode", primary_mode.value))
        if not checkpoint_supports_mode(primary_mode, self.mode):
            raise ValueError(
                f"checkpoint mode is {primary_mode.value!r}, "
                f"not requested {self.mode.value!r}"
            )
        self.action_horizon = int(
            usr_args.get("action_horizon", DEFAULT_ACTION_HORIZON)
        )
        if self.action_horizon not in {24, 48}:
            raise ValueError("WSP2 RoboTwin action_horizon must be 24 or 48")
        self.replan_steps = int(
            usr_args.get("replan_steps", self.action_horizon)
        )
        if not 1 <= self.replan_steps <= self.action_horizon:
            raise ValueError(
                "replan_steps must be between 1 and action_horizon"
            )
        requested_layout = usr_args.get("diffusion_view_layout")
        if requested_layout is not None:
            requested_layout = str(requested_layout)
            if requested_layout not in {"mosaic_2x2", "robotwin_concat"}:
                raise ValueError(
                    "diffusion_view_layout must be 'mosaic_2x2' or "
                    "'robotwin_concat'"
                )
        if str(usr_args.get("action_type", "qpos")) != "qpos":
            raise ValueError("WSP2 RoboTwin policy supports qpos actions only")
        self.observation_interval = int(
            usr_args.get(
                "observation_interval",
                self.action_horizon // (OBSERVATION_HISTORY_FRAMES - 1),
            )
        )
        if (
            self.observation_interval <= 0
            or self.action_horizon % self.observation_interval
            or self.action_horizon // self.observation_interval
            != OBSERVATION_HISTORY_FRAMES - 1
        ):
            raise ValueError(
                "observation_interval must produce nine frames over one "
                f"action block; expected {self.action_horizon // 8} for "
                f"action_horizon={self.action_horizon}"
            )
        if self.replan_steps % self.observation_interval:
            raise ValueError(
                "replan_steps must be divisible by observation_interval"
            )
        self.memory_reset_chunks = int(usr_args.get("memory_reset_chunks", 2))
        if self.memory_reset_chunks < 0:
            raise ValueError("memory_reset_chunks cannot be negative")
        self.frames_per_replan = OBSERVATION_HISTORY_FRAMES
        self.skip_get_obs_within_replan = _parse_bool(
            usr_args.get("skip_get_obs_within_replan", True)
        )
        self.manages_action_chunks = True

        transform = load_robotwin2_checkpoint_transform(checkpoint)
        self.vlm_history_num_frames = int(
            usr_args.get("vlm_history_num_frames", 8)
        )
        if self.vlm_history_num_frames < 1:
            raise ValueError("vlm_history_num_frames must be positive")
        self.vlm_history_stride = int(
            usr_args.get("vlm_history_stride", self.action_horizon)
        )
        if self.vlm_history_stride <= 0:
            raise ValueError("vlm_history_stride must be positive")
        if self.vlm_history_stride % self.observation_interval:
            raise ValueError(
                "vlm_history_stride must be divisible by observation_interval"
            )
        self._vlm_history_cache_size = (
            (self.vlm_history_num_frames - 1)
            * self.vlm_history_stride
            // self.observation_interval
            + 1
        )
        self.vlm_cot_prompt = str(
            usr_args.get("vlm_cot_prompt", DEFAULT_VLM_COT_PROMPT)
        )
        self.t5_prompt_template = str(
            usr_args.get("t5_prompt_template", DEFAULT_T5_PROMPT_TEMPLATE)
        )
        try:
            self.vlm_cot_prompt.format(task="test")
            self.t5_prompt_template.format(instruction="test")
        except (KeyError, ValueError) as exc:
            raise ValueError(
                "RoboTwin prompt templates must contain valid {task} and "
                "{instruction} placeholders"
            ) from exc
        policy = build_wan22_policy_from_checkpoint(
            checkpoint,
            visual_input_range="zero_one",
            diffusion_view_layout=requested_layout,
            device=self.device,
            expected_mode=self.mode,
            expected_action_horizon=self.action_horizon,
            vlm_cot_prompt=self.vlm_cot_prompt,
            validate_checkpoint_artifacts=_parse_bool(
                usr_args.get("validate_checkpoint_artifacts", False)
            ),
        )
        self.diffusion_view_layout = str(
            policy.visual_memory.codec.diffusion_view_layout
        )
        if self.mode is InteractionMode.AUTO:
            kernel = getattr(getattr(policy, "wam", None), "_numerical_kernel", None)
            kernel_config = getattr(kernel, "config", None)
            if kernel_config is None:
                raise RuntimeError("Auto mode could not locate the WAM kernel config")
            kernel.config = replace(kernel_config, cfg_scale=1.0)
        self.runtime = PolicyRuntime(policy)
        self.adapter = RoboTwin2Adapter(
            checkpoint_transform=transform,
            vlm_history_num_frames=self.vlm_history_num_frames,
            preserve_camera_resolution=(
                self.diffusion_view_layout == "robotwin_concat"
            ),
            action_horizon=self.action_horizon,
        )
        self.generator = torch.Generator(device=torch.device(self.device)).manual_seed(
            int(usr_args.get("seed", 0))
        )
        self.log_inference = _parse_bool(usr_args.get("log_inference", True))
        self.pending_actions: deque[np.ndarray] = deque()
        self.pending_output: WorldActionOutput | None = None
        self.executed_in_chunk = 0
        self.chunk_frames: tuple[deque[np.ndarray], ...] | None = None
        self.inference_seconds: list[float] = []
        self.simulation_seconds = 0.0
        self.episode_started = time.perf_counter()
        self.reset()
        print(
            "[wsp2-config] "
            + json.dumps(
                {
                    "checkpoint": str(checkpoint),
                    "mode": self.mode.value,
                    "device": self.device,
                    "action_horizon": self.action_horizon,
                    "replan_steps": self.replan_steps,
                    "observation_interval": self.observation_interval,
                    "memory_reset_chunks": self.memory_reset_chunks,
                    "diffusion_view_layout": self.diffusion_view_layout,
                    "vlm_history_stride": self.vlm_history_stride,
                },
                sort_keys=True,
            )
        )

    @staticmethod
    def _camera_frames(
        observation: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = observation["observation"]
        frames = tuple(
            np.asarray(data[name]["rgb"], dtype=np.uint8)
            for name in ("head_camera", "left_camera", "right_camera")
        )
        if any(frame.ndim != 3 or frame.shape[-1] != 3 for frame in frames):
            raise ValueError(
                "RoboTwin camera observations must be individual HWC RGB frames"
            )
        return frames  # type: ignore[return-value]

    def _record_vlm_history_observation(
        self,
        observation: dict[str, Any],
    ) -> None:
        step = self.episode_action_steps
        frame = self._camera_frames(observation)[0].copy()
        if self._vlm_head_history:
            latest_step = self._vlm_head_history[-1][0]
            if latest_step > step:
                raise RuntimeError(
                    "VLM observation history moved backwards from "
                    f"step {latest_step} to {step}"
                )
            if latest_step == step:
                self._vlm_head_history[-1] = (step, frame)
                return
        self._vlm_head_history.append((step, frame))

    def _sample_vlm_history_frames(self) -> np.ndarray:
        if not self._vlm_head_history:
            raise RuntimeError("VLM observation history is empty")
        current_step = self.episode_action_steps
        frames_by_step = dict(self._vlm_head_history)
        earliest_step, earliest_frame = self._vlm_head_history[0]
        targets = [
            current_step
            - self.vlm_history_stride
            * (self.vlm_history_num_frames - 1 - index)
            for index in range(self.vlm_history_num_frames)
        ]
        frames: list[np.ndarray] = []
        for target in targets:
            if target < earliest_step:
                frames.append(earliest_frame)
                continue
            frame = frames_by_step.get(target)
            if frame is None:
                raise RuntimeError(
                    "Missing stride-aligned VLM history frame at action "
                    f"step {target}; cached steps are "
                    f"{tuple(frames_by_step)}"
                )
            frames.append(frame)
        return np.stack(frames, axis=0)

    def _policy_observation(
        self,
        current: dict[str, Any],
        *,
        append_current: bool = True,
    ) -> dict[str, Any]:
        if self.chunk_frames is None:
            result = copy.deepcopy(current)
            cameras = result["observation"]
            for name, frame in zip(
                ("head_camera", "left_camera", "right_camera"),
                self._camera_frames(current),
            ):
                cameras[name]["rgb"] = np.repeat(
                    frame[None, ...],
                    self.frames_per_replan,
                    axis=0,
                )
            return result
        if append_current:
            self._append_history_observation(current)
        counts = tuple(len(history) for history in self.chunk_frames)
        if len(set(counts)) != 1 or counts[0] > self.frames_per_replan:
            raise RuntimeError(
                "Invalid rolling observation history counts: "
                f"{counts}"
            )
        result = copy.deepcopy(current)
        cameras = result["observation"]
        for name, history in zip(
            ("head_camera", "left_camera", "right_camera"),
            self.chunk_frames,
        ):
            values = list(history)
            padded = [values[0]] * (
                self.frames_per_replan - len(values)
            ) + values
            cameras[name]["rgb"] = np.stack(padded, axis=0)
        return result

    def _append_history_observation(self, observation: dict[str, Any]) -> None:
        if self.chunk_frames is None:
            raise RuntimeError("Missing observation buffer for active chunk")
        for history, frame in zip(
            self.chunk_frames,
            self._camera_frames(observation),
        ):
            history.append(frame.copy())

    def _fill_action_queue(
        self,
        task_env: Any,
        observation: dict[str, Any],
        *,
        append_current: bool = True,
    ) -> None:
        self._record_vlm_history_observation(observation)
        native_observation = self._policy_observation(
            observation,
            append_current=append_current,
        )
        policy_observation = self.adapter.observation(
            native_observation,
            device=self.device,
            vlm_history_frames=self._sample_vlm_history_frames(),
        )
        instruction = str(task_env.get_instruction())
        if self.mode is InteractionMode.INTERACTIVE:
            instruction = self.t5_prompt_template.format(
                instruction=instruction.lower()
            )
        prompts = self.adapter.prompt(instruction, mode=self.mode)
        started = time.perf_counter()
        output = self.runtime.predict(
            observation=policy_observation,
            prompts=prompts,
            generator=self.generator,
        )
        inference_s = time.perf_counter() - started
        actions = self.adapter.action(output)
        if actions.shape != (self.action_horizon, JOINT_DIM):
            self.runtime.discard()
            raise ValueError(
                "WSP2 policy produced "
                f"{actions.shape}, expected ({self.action_horizon}, 14)"
            )
        executed_actions = actions[: self.replan_steps]
        self.pending_output = output
        self.pending_actions.extend(executed_actions)
        self.executed_in_chunk = 0
        # Keep a rolling, training-aligned nine-frame window. The observation
        # interval is 3 for horizon 24 and 6 for horizon 48.
        if self.chunk_frames is None:
            self.chunk_frames = tuple(
                deque([frame.copy()], maxlen=self.frames_per_replan)
                for frame in self._camera_frames(observation)
            )
        self.inference_seconds.append(inference_s)
        if self.log_inference:
            print(
                f"[wsp2-infer] count={len(self.inference_seconds)} "
                f"infer_s={inference_s:.3f}"
            )

    def step(self, task_env: Any, observation: dict[str, Any] | None) -> None:
        if self.pending_actions or self.pending_output is not None:
            raise RuntimeError("A policy step must start at an action-chunk boundary")

        append_current = True
        if observation is None:
            if self._latest_observation is None:
                observation = task_env.get_obs()
            else:
                observation = self._latest_observation
                append_current = not self._latest_observation_buffered
        self._latest_observation = None
        self._latest_observation_buffered = False
        self._fill_action_queue(
            task_env,
            observation,
            append_current=append_current,
        )

        while self.pending_actions:
            action = self.pending_actions.popleft()
            started = time.perf_counter()
            task_env.take_action(action, action_type="qpos")
            self.simulation_seconds += time.perf_counter() - started
            self.executed_in_chunk += 1
            self.episode_action_steps += 1

            if self.executed_in_chunk % self.observation_interval == 0:
                sampled = task_env.get_obs()
                self._append_history_observation(sampled)
                self._record_vlm_history_observation(sampled)
                self._latest_observation = sampled
                self._latest_observation_buffered = True
            elif not self.skip_get_obs_within_replan:
                # Preserve the legacy option to refresh simulator observations
                # after every action. Only stride-aligned frames enter the
                # model's training-aligned nine-frame history.
                task_env.get_obs()

        step_limit = getattr(task_env, "step_lim", None)
        episode_finished = bool(getattr(task_env, "eval_success", False))
        if step_limit is not None:
            episode_finished = episode_finished or int(
                getattr(task_env, "take_action_cnt", 0)
            ) >= int(step_limit)
        if episode_finished:
            self.pending_actions.clear()
            if self.pending_output is not None:
                self.runtime.discard()
                self.pending_output = None
            return

        if self.pending_output is None:
            raise RuntimeError("Missing transactional output at chunk boundary")
        # The WAM candidate memory does not store an unexecuted action suffix;
        # the next call prefills it from the real rolling observation window.
        self.runtime.commit(self.pending_output)
        self.pending_output = None
        self.memory_chunks_since_reset += 1
        if (
            self.memory_reset_chunks
            and self.memory_chunks_since_reset >= self.memory_reset_chunks
        ):
            self.runtime.reset(self.mode.value)
            # Model memory and real observation history have different
            # lifetimes. Keep the episode-scoped rolling frames so the first
            # inference after a model-memory reset can still prefill from the
            # observations that were actually executed.
            self.memory_chunks_since_reset = 0

    def reset(self) -> None:
        if getattr(self, "runtime", None) is not None:
            if self.runtime.has_pending_prediction:
                self.runtime.discard()
            self.runtime.reset(self.mode.value)
        adapter_reset = getattr(getattr(self, "adapter", None), "reset", None)
        if callable(adapter_reset):
            adapter_reset()
        self.pending_actions.clear()
        self.pending_output = None
        self.executed_in_chunk = 0
        self.chunk_frames = None
        self._latest_observation: dict[str, Any] | None = None
        self._latest_observation_buffered = False
        self.episode_action_steps = 0
        self._vlm_head_history: deque[tuple[int, np.ndarray]] = deque(
            maxlen=self._vlm_history_cache_size
        )
        self.memory_chunks_since_reset = 0
        self.inference_seconds = []
        self.simulation_seconds = 0.0
        self.episode_started = time.perf_counter()

    def record_episode_result(self, **record: Any) -> dict[str, Any]:
        count = len(self.inference_seconds)
        timing = {
            "infer_count": count,
            "infer_s": sum(self.inference_seconds),
            "infer_s_mean": (
                sum(self.inference_seconds) / count if count else 0.0
            ),
            "sim_s": self.simulation_seconds,
            "episode_s": time.perf_counter() - self.episode_started,
        }
        print("[wsp2-timing] " + json.dumps(timing, sort_keys=True))
        return {**record, "timing": timing}

    def prepare_evaluation_job(self) -> None:
        self.reset()


def get_model(usr_args: dict[str, Any]) -> WSP2RoboTwinPolicy:
    return WSP2RoboTwinPolicy(usr_args)


def eval(
    task_env: Any,
    model: WSP2RoboTwinPolicy,
    observation: dict[str, Any] | None,
) -> None:
    model.step(task_env, observation)


def reset_model(model: WSP2RoboTwinPolicy) -> None:
    model.reset()


def prepare_model_for_evaluation_job(
    model: WSP2RoboTwinPolicy,
) -> None:
    """Reset job-scoped state without reloading checkpoint weights."""

    model.prepare_evaluation_job()


__all__ = [
    "WSP2RoboTwinPolicy",
    "eval",
    "get_model",
    "prepare_model_for_evaluation_job",
    "reset_model",
]
