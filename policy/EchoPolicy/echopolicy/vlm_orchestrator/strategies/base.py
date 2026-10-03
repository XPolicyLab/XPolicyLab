# SPDX-License-Identifier: Apache-2.0

"""Shared state and transform-aware helpers for the VLA proxy."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import numpy as np

from vlm_orchestrator.transforms import (
    ImageTransform,
    ImageTransformPipeline,
    PromptTransform,
    PromptTransformPipeline,
)

@dataclass
class SessionState:
    """Per-environment state. Actions are always sourced from the VLA."""

    infer_count: int = 0
    episode_step: int = 0
    action_step_count: int = 0
    episode_id: int = 0
    episode_marker: object | None = None
    original_instruction: str | None = None
    rewritten_instruction: str | None = None
    vla_prompt: str | None = None
    subgoals: list[str] = field(default_factory=list)
    subgoal_adapters: list[str] = field(default_factory=list)
    subgoal_atomic_actions: list[str] = field(default_factory=list)
    subgoal_coordinates: list[list[list[int]]] = field(default_factory=list)
    subgoal_arms: list[str] = field(default_factory=list)
    subgoals_ordered: bool = False
    current_subgoal_idx: int = 0
    last_advanced_subgoal: str | None = None
    target_revision: int = 0
    # A newly selected subgoal must observe one fresh action chunk before its
    # first progress decision. This prevents the tail of the previous motion
    # (for example, its retraction) from being consumed as new evidence.
    progress_transition_pending: bool = False
    task_completed: bool = False
    action_cache: Any = field(default=None, repr=False)
    action_cache_offset: int = 0
    action_cache_target_revision: int = -1
    action_cache_video_overlay: Any = field(default=None, repr=False)
    flush_actions: bool = False
    step_at_last_progress_check: int = 0
    group_progress_checked: bool = False
    before_vlm_images: list[Any] = field(default_factory=list, repr=False)
    check_memory: str = ""
    last_executed_ee_trajectory: dict[str, Any] | None = None
    # Actual EE chunks are dispatched in action chunks (normally 10 steps).
    # Keep enough chunks to build the recent VLM evidence window without ever
    # including an unexecuted VLA candidate tail.
    executed_ee_trajectory_history: list[dict[str, Any]] = field(
        default_factory=list, repr=False,
    )
    # The last five actually dispatched joint targets. These are used as
    # temporal conditioning for the next VLA candidate batch.
    last_dispatched_joint_states: list[list[float]] = field(default_factory=list)
    # Set by trajectory selection when the VLA appears to be using an
    # intermediate placement/transfer.  SubgoalStrategy consumes it on the
    # next observation and asks the VLM to replan from the new state.
    pending_replan_reason: str | None = None
    log_entries: list[dict] = field(default_factory=list)
    episode_log_dir: str | None = None
    transform_metadata: list[dict] = field(default_factory=list)

    def recent_executed_ee_trajectory(self, max_steps: int = 20) -> dict[str, Any] | None:
        """Return the latest actual EE steps, preserving action-chunk boundaries."""
        if max_steps <= 0:
            return None
        history = self.executed_ee_trajectory_history
        if not history:
            return deepcopy(self.last_executed_ee_trajectory)

        flattened: list[dict[str, Any]] = []
        latest_chunk_index = len(history) - 1
        for chunk_index, trajectory in enumerate(history):
            chunk_label = (
                "latest_executed_chunk"
                if chunk_index == latest_chunk_index
                else "previous_executed_chunk"
            )
            for chunk_step_index, step in enumerate(trajectory.get("steps", [])):
                item = deepcopy(step)
                item["chunk_index"] = chunk_index
                item["execution_chunk"] = chunk_label
                item["chunk_step_index"] = int(
                    step.get("step_index", chunk_step_index)
                    if isinstance(step, dict) else chunk_step_index
                )
                flattened.append(item)
        if not flattened:
            return deepcopy(self.last_executed_ee_trajectory)

        start = max(0, len(flattened) - max_steps)
        steps = flattened[start:]
        for step_index, step in enumerate(steps):
            step["step_index"] = step_index
        boundaries: list[dict[str, int]] = []
        for chunk_index in sorted({step["chunk_index"] for step in steps}):
            indices = [
                index for index, step in enumerate(steps)
                if step["chunk_index"] == chunk_index
            ]
            boundaries.append({
                "chunk_index": int(chunk_index),
                "label": steps[indices[0]]["execution_chunk"],
                "start_step": min(indices),
                "end_step": max(indices),
                "executed_steps": len(indices),
            })
        latest = history[-1]
        return {
            "executed_steps": len(steps),
            "window_steps": max_steps,
            "chunk_boundaries": boundaries,
            "atomic_action": latest.get("atomic_action", ""),
            "target_points": latest.get("target_points", []),
            "coordinate_frame": latest.get("coordinate_frame", "world_m"),
            "steps": steps,
        }

    def log(self, entry: dict) -> None:
        item = dict(entry)
        item.setdefault("timestamp", time.time())
        item.setdefault("episode_id", self.episode_id)
        self.log_entries.append(item)

class VLMClientLike(Protocol):
    """Structural marker for VLM clients used by strategies."""


@dataclass
class StrategyContext:
    vlm: Any
    image_key: str = "observation/exterior_image_1_left"
    prompt_key: str = "prompt"
    extra_image_keys: list[str] = field(
        default_factory=lambda: [
            "observation/wrist_image_left",
            "observation/wrist_image_right",
        ]
    )
    vla_host: str = "127.0.0.1"
    vla_port: int = 8000
    prompt_transforms: list[PromptTransform] = field(default_factory=list)
    prompt_transform_consumers: set[str] = field(default_factory=set)
    image_transforms: list[ImageTransform] = field(default_factory=list)
    image_transform_consumers: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self._prompt_pipeline = PromptTransformPipeline(self.prompt_transforms)
        self._image_pipeline = ImageTransformPipeline(
            self.image_transforms, set(self.image_transform_consumers)
        )

    def get_prompt(self, obs: Mapping[str, Any]) -> str:
        return str(obs.get(self.prompt_key, ""))

    def set_prompt(self, obs: Mapping[str, Any], prompt: str) -> dict:
        result = dict(obs)
        result[self.prompt_key] = prompt
        return result

    def get_image(self, obs: Mapping[str, Any]) -> np.ndarray | None:
        for key in (self.image_key, "observation/image", "observation/image_raw", "observation/exterior_image_1_left"):
            image = obs.get(key)
            if isinstance(image, np.ndarray):
                return image
        return None

    def get_extra_images(self, obs: Mapping[str, Any]) -> list[np.ndarray]:
        return [
            image
            for key in self.extra_image_keys
            if isinstance(image := obs.get(key + "_raw", obs.get(key)), np.ndarray)
        ]

    def transform_prompt(self, prompt: str, obs: dict, state: SessionState, phase: str, consumer: str | None = None) -> tuple[str, list[dict]]:
        if consumer and self.prompt_transform_consumers and consumer not in self.prompt_transform_consumers:
            return prompt, []
        return self._prompt_pipeline.apply(prompt, obs, state, phase)

    def transform_image(self, image: np.ndarray, obs: dict, state: SessionState, consumer: str) -> tuple[np.ndarray, list[dict]]:
        return self._image_pipeline.apply(image, obs, state, consumer)

    def get_vlm_images(self, obs: dict, state: SessionState, phase: str) -> tuple[np.ndarray | None, list[np.ndarray], list[dict]]:
        primary = self.get_image(obs)
        extras = self.get_extra_images(obs)
        consumer = {
            "episode_start": "vlm_decompose",
            "progress": "vlm_progress",
        }.get(phase, f"vlm_{phase}")
        metadata: list[dict] = []
        if primary is not None:
            primary, items = self.transform_image(primary, obs, state, consumer)
            metadata.extend(items)
        edited_extras = []
        for image in extras:
            image, items = self.transform_image(image, obs, state, consumer)
            edited_extras.append(image)
            metadata.extend(items)
        return primary, edited_extras, metadata

    def transform_vla_observation(self, obs: dict, state: SessionState) -> tuple[dict, list[dict]]:
        result = dict(obs)
        if "vla" not in self.image_transform_consumers:
            return result, []
        metadata: list[dict] = []
        for key in [self.image_key, *self.extra_image_keys]:
            image = result.get(key)
            if isinstance(image, np.ndarray):
                result[key], items = self.transform_image(image, result, state, "vla")
                metadata.extend(items)
        return result, metadata


class OrchestrationStrategy(ABC):
    def __init__(self, ctx: StrategyContext):
        self.ctx = ctx

    @abstractmethod
    def process(self, obs: dict, state: SessionState) -> tuple[dict, SessionState]: ...

    def is_new_episode(self, obs: dict, state: SessionState) -> bool:
        marker = obs.get("__episode_id")
        prompt = self.ctx.get_prompt(obs)
        changed = (
            state.original_instruction is None
            or prompt not in {state.original_instruction, state.rewritten_instruction, *state.subgoals}
            or (marker is not None and marker != state.episode_marker)
        )
        if changed:
            state.episode_marker = marker
            if marker is not None:
                try:
                    state.episode_id = int(marker)
                except (TypeError, ValueError):
                    state.episode_id = max(1, state.episode_id + 1)
            else:
                state.episode_id = max(1, state.episode_id + (state.infer_count > 0))
        return changed

    def current_step(self, obs: dict, state: SessionState) -> int:
        return state.episode_step
