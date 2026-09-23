"""VLM orchestration proxy with VLA-only action execution."""

from __future__ import annotations

import asyncio
import contextvars
import threading
from contextlib import nullcontext
import json
import logging
import math
import os
import re
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from XPolicyLab.policy.EchoPolicy.vlm_orchestrator.protocols.base import Backend, Frontend, FrontendSession
from XPolicyLab.policy.EchoPolicy.vlm_orchestrator.strategies.base import (
    OrchestrationStrategy,
    SessionState,
    StrategyContext,
)
from XPolicyLab.policy.EchoPolicy.vlm_orchestrator.strategies.subgoal_base import SubgoalBaseStrategy
from XPolicyLab.policy.EchoPolicy.vlm_orchestrator.trajectory_selection import (
    DEFAULT_FAR_DISTANCE_M,
    DEFAULT_FAR_TRAJECTORY_STEPS,
    DEFAULT_NEAR_TRAJECTORY_STEPS,
    action_steps,
    average_most_similar_chunks,
    average_consistent_chunks,
    current_ee_pose,
    fk_trajectory,
    group_trajectory_endpoints,
    normalize_points,
    project_points,
    select_candidate,
    target_pixel_to_world,
)
from XPolicyLab.policy.EchoPolicy.vlm_orchestrator.vlm.api import vla_coordinate_payload
from XPolicyLab.policy.EchoPolicy.vlm_orchestrator.vlm.task_descriptions import find_task_reference
from XPolicyLab.policy.EchoPolicy.vlm_orchestrator.robodojo_camera import fixed_head_calibration
from XPolicyLab.policy.EchoPolicy.vlm_orchestrator.robodojo_camera import ROBODOJO_DEFAULT_TABLE_HEIGHT_M

logger = logging.getLogger(__name__)

VLA_STATE_HISTORY_COUNT = 5
VLA_STATE_HISTORY_STRIDE = 1
VLA_STATE_HISTORY_STEPS = 1 + (VLA_STATE_HISTORY_COUNT - 1) * VLA_STATE_HISTORY_STRIDE
# Endpoint grouping uses one fixed 5 cm world-coordinate radius.  The
# near/far classifier remains available for diagnostics and VLM context, but
# it does not change how candidate groups are formed.
GROUP_ENDPOINT_RADIUS_M = 0.05
GROUP_TRAJECTORY_STEPS = 30
OVERLAY_TRAJECTORY_STEPS = 30

async def _run_vlm_call(function, *args, **kwargs):
    """Run blocking VLM work without the default executor's worker limit."""
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    context = contextvars.copy_context()

    def complete(result, error):
        if not future.done():
            if error is not None:
                future.set_exception(error)
            else:
                future.set_result(result)

    def work():
        try:
            result, error = context.run(function, *args, **kwargs), None
        except BaseException as exc:
            result, error = None, exc
        try:
            loop.call_soon_threadsafe(complete, result, error)
        except RuntimeError:
            pass  # The caller was cancelled and its event loop has closed.

    threading.Thread(target=work, name="vlm-request", daemon=True).start()
    return await future


def _slug(value: str) -> str:
    result = re.sub(r"[^\w\s-]", "", value).strip().replace(" ", "_")
    return re.sub(r"_+", "_", result)[:80] or "unknown"


def _action_count(actions: Any) -> int:
    if actions is None:
        return 0
    if hasattr(actions, "ndim") and hasattr(actions, "shape"):
        return 1 if actions.ndim == 1 else int(actions.shape[0])
    if isinstance(actions, (list, tuple)):
        return len(actions)
    if isinstance(actions, dict):
        return 1
    return 0


def _response_actions(response: Any) -> Any:
    if isinstance(response, dict):
        return response.get("actions", response.get("__robodojo_actions"))
    return response


def _joint_state_from_action(step: Any) -> list[float] | None:
    """Normalize one dispatched action to the canonical 14-D robot state."""
    import numpy as np

    if isinstance(step, dict):
        try:
            left_arm = np.asarray(step["left_arm_joint_state"], dtype=np.float64).reshape(-1)
            right_arm = np.asarray(step["right_arm_joint_state"], dtype=np.float64).reshape(-1)
            left_ee = np.asarray(step["left_ee_joint_state"], dtype=np.float64).reshape(-1)
            right_ee = np.asarray(step["right_ee_joint_state"], dtype=np.float64).reshape(-1)
            values = np.concatenate((left_arm[:6], left_ee[:1], right_arm[:6], right_ee[:1]))
        except (KeyError, TypeError, ValueError):
            return None
    else:
        try:
            values = np.asarray(step, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return None
        if values.size < 14:
            return None
        # RoboDojo joint action layout is [left arm(6), left ee, right arm(6), right ee].
        values = np.concatenate((values[:6], values[6:7], values[7:13], values[13:14]))
    if values.shape != (14,) or not np.isfinite(values).all():
        return None
    return values.tolist()


def _pi05_training_image_augmentation(
    image: Any,
    *,
    geometric: bool,
    color_factors: tuple[float, float, float],
    rotation: float,
    crop_top: int,
    crop_left: int,
) -> Any:
    """Apply Pi05's training-time RGB preprocessing to one candidate image.

    Pi05 resizes with padding to 224x224, applies a 95% crop and +/-5 degree
    rotation only to the non-wrist camera, and applies the same brightness,
    contrast, and saturation ranges to every camera.  The caller samples the
    factors once per candidate so the three camera views remain appearance-
    consistent, as they are in the training pipeline.
    """
    import numpy as np
    from PIL import Image, ImageEnhance

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
        return array.copy()

    target = 224
    height, width = array.shape[:2]
    ratio = max(width / target, height / target)
    resized_width = max(1, int(width / ratio))
    resized_height = max(1, int(height / ratio))
    pil = Image.fromarray(array, mode="RGB").resize(
        (resized_width, resized_height), Image.Resampling.BILINEAR,
    )
    padded = Image.new("RGB", (target, target), (0, 0, 0))
    pad_left = (target - resized_width) // 2
    pad_top = (target - resized_height) // 2
    padded.paste(pil, (pad_left, pad_top))
    pil = padded

    if geometric:
        crop_size = int(target * 0.95)
        # The sampled offsets are in the 224x224 post-resize frame.  Keeping
        # them explicit makes the transform deterministic across camera keys.
        top = min(max(int(crop_top), 0), target - crop_size)
        left = min(max(int(crop_left), 0), target - crop_size)
        pil = pil.crop((left, top, left + crop_size, top + crop_size)).resize(
            (target, target), Image.Resampling.BILINEAR,
        )
        pil = pil.rotate(
            float(rotation), resample=Image.Resampling.BILINEAR,
            expand=False, fillcolor=(0, 0, 0),
        )

    brightness, contrast, saturation = color_factors
    pil = ImageEnhance.Brightness(pil).enhance(brightness)
    pil = ImageEnhance.Contrast(pil).enhance(contrast)
    pil = ImageEnhance.Color(pil).enhance(saturation)
    return np.asarray(pil, dtype=np.uint8)


@dataclass
class ProxyConfig:
    vla_host: str = "127.0.0.1"
    vla_port: int = 6000
    host: str = "0.0.0.0"
    port: int = 8001
    strategy: OrchestrationStrategy | None = None
    image_key: str = "observation/exterior_image_1_left"
    prompt_key: str = "prompt"
    extra_image_keys: list[str] = field(
        default_factory=lambda: ["observation/wrist_image_left", "observation/wrist_image_right"]
    )
    log_dir: str | None = None
    cfg_scale: float | None = None
    cfg_uncond_prompt: str | None = None
    cfg_enabled: bool = True
    action_chunk_size: int = 10
    vla_candidates: int = 16
    vla_image_noise_std: float = 2.0
    vla_image_augmentation: bool = True
    vla_joint_state_noise_std: float = 0.01
    far_distance_m: float = DEFAULT_FAR_DISTANCE_M
    table_height_m: float = ROBODOJO_DEFAULT_TABLE_HEIGHT_M
    near_trajectory_steps: int = DEFAULT_NEAR_TRAJECTORY_STEPS
    far_trajectory_steps: int = DEFAULT_FAR_TRAJECTORY_STEPS
    batch_vlm_concurrency: int = 0
    global_vlm_concurrency: int | None = None
    frontend: Frontend | None = None
    backend: Backend | None = None

    def __post_init__(self) -> None:
        if self.action_chunk_size <= 0:
            raise ValueError("action_chunk_size must be positive")
        if self.vla_candidates <= 0:
            raise ValueError("vla_candidates must be positive")
        if not math.isfinite(self.vla_image_noise_std) or self.vla_image_noise_std < 0:
            raise ValueError("vla_image_noise_std must be non-negative")
        if not math.isfinite(self.vla_joint_state_noise_std) or self.vla_joint_state_noise_std < 0:
            raise ValueError("vla_joint_state_noise_std must be non-negative")
        if self.far_distance_m < 0:
            raise ValueError("far_distance_m must be non-negative")
        if not math.isfinite(self.table_height_m):
            raise ValueError("table_height_m must be finite")
        if self.near_trajectory_steps <= 0 or self.far_trajectory_steps <= 0:
            raise ValueError("trajectory comparison steps must be positive")
        if self.batch_vlm_concurrency < 0:
            raise ValueError("batch_vlm_concurrency must be non-negative")
        if self.global_vlm_concurrency is not None and self.global_vlm_concurrency < 0:
            raise ValueError("global_vlm_concurrency must be non-negative")
        if self.frontend is None or self.backend is None:
            from XPolicyLab.policy.EchoPolicy.vlm_orchestrator.protocols.robodojo_ws import RoboDojoWsFrontend
            from XPolicyLab.policy.EchoPolicy.vlm_orchestrator.protocols.xpolicylab_ws import XPolicyLabWsBackend

            self.frontend = self.frontend or RoboDojoWsFrontend()
            self.backend = self.backend or XPolicyLabWsBackend(self.vla_host, self.vla_port)
        if self.strategy is None:
            from XPolicyLab.policy.EchoPolicy.vlm_orchestrator.strategies.passthrough import PassthroughStrategy
            from XPolicyLab.policy.EchoPolicy.vlm_orchestrator.vlm import PassthroughVLM

            self.strategy = PassthroughStrategy(
                StrategyContext(
                    vlm=PassthroughVLM(),
                    image_key=self.image_key,
                    prompt_key=self.prompt_key,
                    extra_image_keys=self.extra_image_keys,
                )
            )


class OrchestratorProxy:
    def __init__(self, config: ProxyConfig):
        self.config = config
        concurrency = config.global_vlm_concurrency
        self._global_vlm_semaphore = asyncio.Semaphore(concurrency) if concurrency else nullcontext()

    def serve_forever(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        assert self.config.frontend is not None
        await self.config.frontend.serve(self.config.host, self.config.port, self._handle_session)

    def _new_episode(self, obs: dict, state: SessionState) -> bool:
        prompt = str(obs.get(self.config.prompt_key, ""))
        marker = obs.get("__episode_id")
        known = {state.original_instruction, state.rewritten_instruction, *state.subgoals}
        return state.infer_count == 0 or prompt not in known or (
            marker is not None and marker != state.episode_marker
        )

    def _rotate_log(self, state: SessionState, prompt: str, observation: dict, slot: str) -> None:
        if not self.config.log_dir:
            return
        base = os.path.join(self.config.log_dir, f"env_{_slug(str(slot))}", f"episode_{state.episode_id}")
        path, attempt = base, 2
        while True:
            try:
                os.makedirs(path, exist_ok=False)
                break
            except FileExistsError:
                path, attempt = f"{base}_attempt{attempt}", attempt + 1
        state.episode_log_dir = path
        with open(os.path.join(path, "metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "episode_id": state.episode_id,
                    "episode_marker": observation.get("__episode_id"),
                    "original_instruction": prompt,
                    "env_idx": observation.get("env_idx"),
                    "cfg_scale": self.config.cfg_scale if self.config.cfg_enabled else None,
                    "start_timestamp": time.time(),
                },
                handle,
                indent=2,
            )

    @staticmethod
    def _rgb_image(value: Any):
        """Convert an observation image to a PIL RGB image for diagnostics."""
        import numpy as np
        from PIL import Image

        array = np.asarray(value)
        if array.ndim != 3:
            raise ValueError(f"expected a 3-D image, got {array.shape}")
        if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = np.moveaxis(array, 0, -1)
        if np.issubdtype(array.dtype, np.floating):
            finite = array[np.isfinite(array)]
            scale = 255.0 if finite.size and float(finite.max()) <= 1.0 else 1.0
            array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0) * scale
        array = np.clip(array, 0, 255).astype(np.uint8)
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        elif array.shape[-1] == 4:
            array = array[..., :3]
        elif array.shape[-1] != 3:
            raise ValueError(f"expected 1, 3, or 4 image channels, got {array.shape[-1]}")
        return Image.fromarray(array, mode="RGB")

    @staticmethod
    def _save_group_selection_image(
        state: SessionState, group_images: list[Any], selected_group: int, reason: str,
        panel_labels: list[str] | None = None, memory: str = "",
        before_to_now_summary: str = "",
        distance_class: str | None = None,
    ) -> str | None:
        """Save group inputs side by side with selection reason and current memory."""
        if not state.episode_log_dir or not group_images:
            return None
        try:
            from PIL import Image, ImageDraw, ImageFont

            panels = [OrchestratorProxy._rgb_image(value) for value in group_images]
            width = sum(panel.width for panel in panels)
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 14)
                title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
            except OSError:
                font = title_font = ImageFont.load_default()
            labels = panel_labels or [f"G{index}" for index in range(len(panels))]
            if selected_group >= 0:
                selected_panel = labels.index(f"G{selected_group}") if f"G{selected_group}" in labels else selected_group
            else:
                selected_panel = labels.index("PREVIOUS") if "PREVIOUS" in labels else len(panels) - 1
            probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
            max_text_width = max(1, width - 16)

            def wrap_text(prefix: str, value: str) -> list[str]:
                """Wrap both normal prose and unspaced (e.g. CJK) memory text."""
                text = str(value or "").strip() or "(none)"
                result, line = [], prefix
                for token in text.split() if " " in text else list(text):
                    candidate = f"{line} {token}" if line != prefix else f"{prefix}{token}"
                    if line != prefix and probe.textlength(candidate, font=font) > max_text_width:
                        result.append(line)
                        line = f"{prefix}{token}" if not result else token
                    elif line == prefix and probe.textlength(candidate, font=font) > max_text_width:
                        result.append(line.rstrip())
                        line = token
                    else:
                        line = candidate
                result.append(line)
                return result

            distance_label = str(distance_class or "").strip().upper()
            far_label = {"FAR": "YES", "NEAR": "NO"}.get(distance_label, "UNKNOWN")
            lines = wrap_text(
                f"FAR: {far_label} | SELECTED: {labels[selected_panel]} | REASON: ",
                reason or "(no reason returned)",
            )
            lines.extend(wrap_text("BEFORE->NOW: ", before_to_now_summary))
            lines.extend(wrap_text("MEMORY: ", memory))
            header_height = 10 + 18 * len(lines)
            height = header_height + max(panel.height for panel in panels)
            canvas = Image.new("RGB", (width, height), (20, 20, 20))
            draw = ImageDraw.Draw(canvas)
            for line_index, text in enumerate(lines):
                draw.text((8, 5 + 18 * line_index), text, fill=(130, 255, 160), font=font)
            x = 0
            for group_id, panel in enumerate(panels):
                y = header_height
                canvas.paste(panel, (x, y))
                color = (0, 255, 120) if group_id == selected_panel else (150, 150, 150)
                draw.rectangle((x + 2, y + 2, x + panel.width - 3, y + panel.height - 3), outline=color, width=5)
                label = f"{labels[group_id]} SELECTED" if group_id == selected_panel else labels[group_id]
                label_width = int(draw.textlength(label, font=title_font)) + 12
                draw.rectangle((x + 6, y + 28, x + 6 + label_width, y + 52), fill=(0, 0, 0))
                draw.text((x + 12, y + 32), label, fill=color, font=title_font)
                x += panel.width
            path = os.path.join(state.episode_log_dir, f"vla_group_selection_{state.infer_count:06d}.png")
            canvas.save(path)
            return path
        except (ImportError, OSError, TypeError, ValueError) as exc:
            logger.warning("Failed to save VLM group selection image: %s", exc)
            return None

    @staticmethod
    def _strip_internal_keys(obs: dict) -> dict:
        return {key: value for key, value in obs.items() if not key.startswith("__") and not key.endswith("_raw")}

    @staticmethod
    def _prepare_vla_observation(
        obs: dict,
        state: SessionState,
        cfg_scale: float | None,
        cfg_uncond_prompt: str | None,
        cfg_enabled: bool,
    ) -> dict:
        result = OrchestratorProxy._strip_internal_keys(obs)
        prompt = state.original_instruction or result.get("instruction", result.get("prompt", ""))
        points, _ = (
            ([], "")
            if state.task_completed
            else SubgoalBaseStrategy._current_grounding(state)
        )
        result["target_revision"] = state.target_revision
        result["subgoals_ordered"] = bool(state.subgoals_ordered)
        if prompt is not None:
            prompt = str(prompt)
            result["prompt"] = prompt
            result["instruction"] = prompt
        if points:
            result["coordinates"] = vla_coordinate_payload(points)
        else:
            result.pop("coordinates", None)
        result.pop("arm", None)

        has_subgoal = bool(
            not state.task_completed
            and state.subgoals
            and 0 <= state.current_subgoal_idx < len(state.subgoals)
        )
        if not has_subgoal:
            result.pop("atomic_action", None)
            result.pop("adapter_id", None)
        planned_adapter = (
            str(state.subgoal_adapters[state.current_subgoal_idx]).strip()
            if has_subgoal and state.current_subgoal_idx < len(state.subgoal_adapters)
            else ""
        )
        planned_action = (
            str(state.subgoal_atomic_actions[state.current_subgoal_idx]).strip().lower()
            if has_subgoal and state.current_subgoal_idx < len(state.subgoal_atomic_actions)
            else ""
        )
        if has_subgoal:
            if planned_action:
                result["atomic_action"] = planned_action
            if planned_adapter:
                result["adapter_id"] = planned_adapter
            if state.current_subgoal_idx < len(state.subgoal_coordinates):
                current_points = state.subgoal_coordinates[state.current_subgoal_idx]
                if current_points:
                    result["coordinates"] = deepcopy(current_points)
        if not cfg_enabled:
            result.pop("cfg_scale", None)
            result.pop("cfg_uncond_prompt", None)
            return result
        result.setdefault("cfg_scale", cfg_scale)
        if cfg_uncond_prompt == "highlevel":
            uncond = state.original_instruction
        else:
            uncond = cfg_uncond_prompt or result.get("cfg_uncond_prompt") or state.original_instruction
        if uncond is not None:
            result["cfg_uncond_prompt"] = str(uncond).strip().strip("'\"")
        return result

    @staticmethod
    def _clear_action_cache(state: SessionState) -> None:
        state.action_cache = None
        state.action_cache_offset = 0
        state.action_cache_target_revision = -1
        state.action_cache_video_overlay = None

    @staticmethod
    def _previous_candidate_remainder(state: SessionState) -> list[Any]:
        if (
            state.flush_actions
            or state.action_cache is None
        ):
            return []
        remainder = action_steps(state.action_cache)[state.action_cache_offset:]
        # Keep the unexecuted tail even when it is shorter than the normal
        # comparison horizon.  It remains useful across a subgoal transition
        # as a continuation candidate; a newly selected response refreshes the
        # cache for the new target before the next observation.
        return remainder if remainder else []

    @classmethod
    def _cached_action_chunk(cls, response: dict, state: SessionState, size: int) -> dict:
        actions = _response_actions(response)
        count = _action_count(actions)
        if count <= 0:
            raise RuntimeError("VLA returned an empty action chunk")
        if (
            state.action_cache is None
            or state.action_cache_target_revision != state.target_revision
            or state.action_cache_offset >= len(action_steps(state.action_cache))
        ):
            state.action_cache = deepcopy(actions)
            state.action_cache_offset = 0
            state.action_cache_target_revision = state.target_revision
            state.action_cache_video_overlay = deepcopy(response.get("video_overlay"))
        cached = action_steps(state.action_cache)
        start = state.action_cache_offset
        end = min(start + max(1, size), len(cached))
        if start >= end:
            raise RuntimeError("VLA action cache is exhausted")
        result = dict(response)
        result["actions"] = cached[start:end]
        result["video_overlay"] = deepcopy(state.action_cache_video_overlay)
        result["orchestrator_action_cache_offset"] = [start, end]
        result["orchestrator_action_cache_refreshed"] = start == 0
        state.action_cache_offset = end
        return result

    @staticmethod
    def _executed_chunk_ee_trajectory(actions: Any, selection: dict) -> dict:
        import numpy as np

        steps = action_steps(actions)
        trajectory = fk_trajectory(steps)

        def gripper_value(step: Any, arm: str) -> float | None:
            if isinstance(step, dict):
                value = step.get(f"{arm}_ee_joint_state")
                if value is None:
                    return None
                array = np.asarray(value, dtype=np.float64).reshape(-1)
            else:
                array = np.asarray(step, dtype=np.float64).reshape(-1)
                index = 6 if arm == "left" else 13
                if len(array) <= index:
                    return None
                array = array[index:index + 1]
            if not len(array) or not np.isfinite(array[0]):
                return None
            return float(array[0])

        ee_steps = []
        for index, action in enumerate(steps):
            item = {"step_index": index}
            for arm in ("left", "right"):
                arm_item = {
                    "position_world_m": np.asarray(
                        trajectory[arm][index], dtype=np.float64,
                    ).tolist(),
                }
                gripper = gripper_value(action, arm)
                if gripper is not None:
                    arm_item["gripper_position"] = gripper
                item[arm] = arm_item
            ee_steps.append(item)
        return {
            "executed_steps": len(steps),
            "atomic_action": selection.get("atomic_action") or "",
            "target_points": selection.get("target_points") or [],
            "coordinate_frame": "world_m",
            "steps": ee_steps,
        }

    @classmethod
    def _record_executed_chunk(
        cls, state: SessionState, response: dict, sent_count: int,
    ) -> None:
        available_steps = action_steps(_response_actions(response))
        steps = available_steps[:min(sent_count, len(available_steps))]
        dispatched_states = [
            normalized for step in steps
            if (normalized := _joint_state_from_action(step)) is not None
        ]
        # Keep only the actions that were actually sent, never the unexecuted
        # tail of a VLA action chunk.
        state.last_dispatched_joint_states = (
            state.last_dispatched_joint_states + dispatched_states
        )[-VLA_STATE_HISTORY_STEPS:]
        selection = response.get("video_overlay", {}).get(
            "trajectory_selection", {}
        )
        try:
            trajectory = cls._executed_chunk_ee_trajectory(steps, selection)
        except (TypeError, ValueError, IndexError) as exc:
            trajectory = {
                "executed_steps": len(steps),
                "error": f"EE trajectory unavailable: {exc}",
            }
        state.last_executed_ee_trajectory = trajectory
        state.executed_ee_trajectory_history = (
            state.executed_ee_trajectory_history + [deepcopy(trajectory)]
        )[-8:]
        state.log({
            "type": "executed_action_ee_trajectory",
            "trajectory": trajectory,
        })

    @staticmethod
    def _candidate_vla_observations(
        vla_obs: dict,
        state: SessionState,
        count: int,
        image_noise_std: float = 0.0,
        joint_state_noise_std: float = 0.0,
        far_target: bool = False,
        rng: Any = None,
        image_augmentation: bool = False,
    ) -> list[dict]:
        """Condition each sampled candidate on one of the last dispatched states.

        The Pi05 transport still receives its normal 14-D state.  This avoids a
        protocol/model shape change while giving the candidate batch temporal
        state context from the previous executed chunk.
        """
        import numpy as np

        current_joint = np.asarray(
            vla_obs.get("observation/joint_position", []), dtype=np.float32,
        ).reshape(-1)
        current_gripper = np.asarray(
            vla_obs.get("observation/gripper_position", []), dtype=np.float32,
        ).reshape(-1)
        conditioning_states = [None] * count
        if current_joint.size == 12 and current_gripper.size == 2:
            current = np.concatenate(
                (current_joint[:6], current_gripper[:1], current_joint[6:12], current_gripper[1:2])
            ).tolist()
            dense_history = list(
                state.last_dispatched_joint_states[-VLA_STATE_HISTORY_STEPS:]
            )
            if len(dense_history) < VLA_STATE_HISTORY_STEPS:
                dense_history = (
                    [current] * (VLA_STATE_HISTORY_STEPS - len(dense_history))
                    + dense_history
                )
            history = dense_history[::VLA_STATE_HISTORY_STRIDE]
            # Distribute samples evenly over the last five consecutive actions.
            # Fifteen new samples therefore use each temporal state three times.
            base, remainder = divmod(count, len(history))
            conditioning_states = [
                value
                for index, value in enumerate(history)
                for _ in range(base + (index >= len(history) - remainder))
            ]
        noise_rng = rng if rng is not None else np.random.default_rng()
        result = []
        for values in conditioning_states:
            item = dict(vla_obs)
            if values is not None:
                values = list(values)
                if far_target and joint_state_noise_std > 0:
                    values[:6] = (
                        np.asarray(values[:6], dtype=np.float64)
                        + noise_rng.normal(0.0, joint_state_noise_std, 6)
                    ).tolist()
                    values[7:13] = (
                        np.asarray(values[7:13], dtype=np.float64)
                        + noise_rng.normal(0.0, joint_state_noise_std, 6)
                    ).tolist()
                item["observation/joint_position"] = np.asarray(
                    values[:6] + values[7:13], dtype=np.float32,
                )
                item["observation/gripper_position"] = np.asarray(
                    [values[6], values[13]], dtype=np.float32,
                )
                item["vla_state_source"] = "last_dispatched_action"
            if image_augmentation:
                color_factors = (
                    float(noise_rng.uniform(0.7, 1.3)),
                    float(noise_rng.uniform(0.6, 1.4)),
                    float(noise_rng.uniform(0.5, 1.5)),
                )
                rotation = float(noise_rng.uniform(-5.0, 5.0))
                crop_top = int(noise_rng.integers(0, 13))
                crop_left = int(noise_rng.integers(0, 13))
                for key, value in tuple(item.items()):
                    image = np.asarray(value)
                    if (
                        "image" not in key
                        or image.dtype != np.uint8
                        or image.ndim != 3
                        or image.shape[-1] != 3
                    ):
                        continue
                    item[key] = _pi05_training_image_augmentation(
                        image,
                        geometric=("wrist" not in key),
                        color_factors=color_factors,
                        rotation=rotation,
                        crop_top=crop_top,
                        crop_left=crop_left,
                    )
            if image_noise_std > 0:
                for key, value in tuple(item.items()):
                    image = np.asarray(value)
                    if (
                        "image" not in key
                        or image.dtype != np.uint8
                        or image.ndim != 3
                        or image.shape[-1] != 3
                    ):
                        continue
                    noise = noise_rng.normal(0.0, image_noise_std, image.shape)
                    item[key] = np.rint(np.clip(
                        image.astype(np.float32) + noise, 0, 255,
                    )).astype(np.uint8)
            result.append(item)
        return result

    def _target_distance_m(self, obs: dict, state: SessionState) -> float | None:
        """Return the nearest 3-D EE-to-target distance for the active subgoal."""
        import numpy as np

        if state.task_completed:
            return None
        index = state.current_subgoal_idx
        if index >= len(state.subgoal_coordinates):
            return None
        points = normalize_points(state.subgoal_coordinates[index])
        image = obs.get(self.config.image_key)
        depth = obs.get("observation/depth_exterior_image_1_left")
        if len(points) == 0 or image is None:
            return None
        intrinsic, extrinsic, shape = self._selection_inputs(obs)
        if intrinsic is None or extrinsic is None or shape is None:
            return None
        try:
            targets = np.asarray([
                target_pixel_to_world(
                    point, depth, intrinsic, extrinsic, shape,
                    default_plane_height_m=self.config.table_height_m,
                )
                for point in points
            ], dtype=np.float64)
            joints = obs.get("observation/joint_position")
            if joints is None:
                return None
            ee = np.asarray([
                current_ee_pose(joints, arm)[0] for arm in ("left", "right")
            ], dtype=np.float64)
            distances = np.linalg.norm(targets[:, None, :] - ee[None, :, :], axis=2)
            value = float(np.nanmin(distances))
            return value if math.isfinite(value) else None
        except (TypeError, ValueError, np.linalg.LinAlgError):
            return None

    @staticmethod
    def _selection_inputs(obs: dict) -> tuple[Any, Any, tuple[int, int] | None]:
        image = obs.get("observation/exterior_image_1_left")
        if image is None:
            return None, None, None
        intrinsic, camera_to_world, image_shape = fixed_head_calibration(image)
        return intrinsic, camera_to_world, image_shape

    @staticmethod
    def _project_candidate_trajectories(
        chunks: list[Any],
        intrinsic: Any,
        camera_to_world: Any,
        image_shape: tuple[int, int] | None,
        trajectory_steps: int | None = None,
    ) -> list[dict[str, list[list[float]]]]:
        """Project every sampled candidate, independently of selection policy."""
        if intrinsic is None or camera_to_world is None or image_shape is None:
            return []
        try:
            import numpy as np

            height, width = map(int, image_shape)
            if height <= 1 or width <= 1:
                return []
            scale = np.asarray((255.0 / (width - 1), 255.0 / (height - 1)))
            projected = []
            for chunk in chunks:
                if trajectory_steps is not None:
                    chunk = action_steps(chunk)[:max(1, int(trajectory_steps))]
                trajectory = fk_trajectory(chunk)
                projected.append(
                    {
                        arm: (project_points(trajectory[arm], intrinsic, camera_to_world) * scale).tolist()
                        for arm in ("left", "right")
                    }
                )
            return projected
        except (TypeError, ValueError, np.linalg.LinAlgError):
            return []

    def _select_grouped_vla_response(
        self, candidates, obs, state, before_images=None, *, selection_only=False,
        preserved_progress=None,
    ):
        import numpy as np
        from PIL import ImageDraw

        candidates = list(candidates)
        new_candidate_count = len(candidates)
        # Candidate groups are judged on exactly the prefix that can be
        # dispatched in this observation.  The remaining tail stays in
        # ``full_chunks`` for the action cache, but is not exposed to VLM.
        steps = self.config.action_chunk_size
        previous_candidate_index = None
        previous_steps_available = 0
        remainder = self._previous_candidate_remainder(state)
        if remainder:
            previous_candidate_index = len(candidates)
            previous_steps_available = len(remainder)
            candidates.append({"actions": deepcopy(remainder)})
        candidate_sources = ["new"] * len(candidates)
        if previous_candidate_index is not None:
            candidate_sources[previous_candidate_index] = "previous_remaining"
        idx = state.current_subgoal_idx
        points = state.subgoal_coordinates[idx] if idx < len(state.subgoal_coordinates) else []
        atomic = state.subgoal_atomic_actions[idx] if idx < len(state.subgoal_atomic_actions) else ""
        chunks = [_response_actions(candidate) for candidate in candidates]
        intrinsic, extrinsic, shape = self._selection_inputs(obs)
        full_chunks = chunks
        chunks = [action_steps(chunk)[:steps] for chunk in chunks]
        ee_trajectories = [fk_trajectory(chunk) for chunk in chunks]
        # Keep the far/near classifier for diagnostics and VLM context.  It
        # does not change the fixed-radius 3-D endpoint grouping policy.
        target_distance_m = self._target_distance_m(obs, state)
        distance_class = (
            "near" if target_distance_m is not None
            and target_distance_m <= self.config.far_distance_m
            else "far" if target_distance_m is not None
            else "unavailable"
        )
        group_endpoint_radius_m = GROUP_ENDPOINT_RADIUS_M
        # Endpoint clustering uses the longer planning horizon, while the
        # trajectory data exposed to VLM remains limited to the executable
        # 10-step prefix.
        group_chunks = [
            action_steps(chunk)[:GROUP_TRAJECTORY_STEPS]
            for chunk in full_chunks
        ]
        group_ee_trajectories = [fk_trajectory(chunk) for chunk in group_chunks]
        groups = group_trajectory_endpoints(
            group_ee_trajectories[:new_candidate_count], radius=group_endpoint_radius_m,
        )
        if previous_candidate_index is not None:
            previous_group = group_trajectory_endpoints(
                [group_ee_trajectories[previous_candidate_index]], radius=group_endpoint_radius_m,
            )[0]
            previous_group["group_id"] = len(groups)
            previous_group["candidate_indices"] = [previous_candidate_index]
            groups.append(previous_group)
        # Each group is represented by the average of its most consistent
        # full action sequences.  VLM never chooses an individual raw sample;
        # the selected group average is the action that is executed.
        group_average_chunks = []
        group_average_indices = []
        group_average_distances = []
        for group in groups:
            averaged, consistent_members, consistency_distance = average_consistent_chunks(
                full_chunks, group["candidate_indices"], compute_average=True,
            )
            group_average_chunks.append(averaged)
            group_average_indices.append(consistent_members)
            group_average_distances.append(consistency_distance)
        average_ee_trajectories = [
            fk_trajectory(action_steps(chunk)[:steps])
            for chunk in group_average_chunks
        ]
        full_average_ee_trajectories = [
            fk_trajectory(action_steps(chunk))
            for chunk in group_average_chunks
        ]
        projected_for_overlay = self._project_candidate_trajectories(
            group_average_chunks, intrinsic, extrinsic, shape, OVERLAY_TRAJECTORY_STEPS,
        )
        overlay_projection_available = len(projected_for_overlay) == len(groups)
        if not overlay_projection_available:
            # Calibration is optional in some RoboDojo payloads. Keep the
            # FK-based grouping and VLM decision alive, while making the
            # missing 2-D projection explicit in diagnostics.
            projected_for_overlay = [{"left": [], "right": []} for _ in groups]
        projection_available = overlay_projection_available
        current_ee = {}
        joints = obs.get("observation/joint_position")
        gripper_values = obs.get("observation/gripper_position")
        grippers = (
            np.asarray(gripper_values, dtype=np.float64).reshape(-1)
            if gripper_values is not None else np.asarray([])
        )
        if joints is not None:
            for arm_index, arm_name in enumerate(("left", "right")):
                ee_position, ee_rotation = current_ee_pose(joints, arm_name)
                ee_pixel = None
                if intrinsic is not None and extrinsic is not None and shape is not None:
                    projected_ee = project_points([ee_position], intrinsic, extrinsic)[0]
                    projected_ee *= np.asarray(
                        (255.0 / (shape[1] - 1), 255.0 / (shape[0] - 1))
                    )
                    if np.isfinite(projected_ee).all():
                        ee_pixel = projected_ee.tolist()
                gripper_position = None
                if len(grippers) > arm_index and np.isfinite(grippers[arm_index]):
                    gripper_position = float(grippers[arm_index])
                current_ee[arm_name] = {
                    "position_world_m": ee_position.tolist(),
                    "rotation_world_matrix": ee_rotation.tolist(),
                    "main_view_pixel_0_255": ee_pixel,
                    "gripper_position": gripper_position,
                }
        targets = normalize_points(points)
        group_members = []
        for group_index, group in enumerate(groups):
            consistent_members = group_average_indices[group_index]
            consistency_distance = group_average_distances[group_index]
            member_entries = []
            for index in group["candidate_indices"]:
                comparison_actions = action_steps(full_chunks[index])[:steps]
                execution_actions = action_steps(full_chunks[index])[:self.config.action_chunk_size]
                member_entries.append({
                    "candidate_index": index,
                    "comparison_actions": comparison_actions,
                    "ee": ee_trajectories[index],
                    "execution_steps": len(execution_actions),
                    "execution_ee": fk_trajectory(execution_actions),
                })
            group_members.append({
                "members": list(group["candidate_indices"]),
                "consistent_members": consistent_members,
                "distance": consistency_distance,
                "average_actions": group_average_chunks[group_index],
                "entries": member_entries,
            })

        for group, member_group in zip(groups, group_members, strict=True):
            member_endpoints = []
            member_z_ranges = []
            member_execution_endpoints = []
            member_execution_z_ranges = []
            for member in member_group["entries"]:
                member_endpoints.append({
                    "candidate_index": member["candidate_index"],
                    **{
                        arm: member["ee"][arm][-1].tolist()
                        for arm in ("left", "right")
                    },
                })
                member_z_ranges.append({
                    "candidate_index": member["candidate_index"],
                    **{
                        arm: {
                            "min": float(np.min(member["ee"][arm][:, 2])),
                            "max": float(np.max(member["ee"][arm][:, 2])),
                        }
                        for arm in ("left", "right")
                    },
                })
                member_execution_endpoints.append({
                    "candidate_index": member["candidate_index"],
                    **{
                        arm: member["execution_ee"][arm][-1].tolist()
                        for arm in ("left", "right")
                    },
                })
                member_execution_z_ranges.append({
                    "candidate_index": member["candidate_index"],
                    **{
                        arm: {
                            "min": float(np.min(member["execution_ee"][arm][:, 2])),
                            "max": float(np.max(member["execution_ee"][arm][:, 2])),
                        }
                        for arm in ("left", "right")
                    },
                })
            group["endpoint_space"] = "world_xyz_meters"
            group["endpoint_radius_m"] = group_endpoint_radius_m
            group["member_candidate_indices"] = member_group["members"]
            group["consistent_candidate_indices"] = member_group["consistent_members"]
            group["member_endpoints"] = member_endpoints
            group["member_trajectory_z_ranges_m"] = member_z_ranges
            group["member_execution_steps"] = [
                {
                    "candidate_index": member["candidate_index"],
                    "steps": member["execution_steps"],
                }
                for member in member_group["entries"]
            ]
            group["member_execution_endpoints"] = member_execution_endpoints
            group["member_execution_trajectory_z_ranges_m"] = member_execution_z_ranges
            mean_ee = average_ee_trajectories[group["group_id"]]
            full_mean_ee = full_average_ee_trajectories[group["group_id"]]
            group["mean_ee_endpoints"] = {
                arm: mean_ee[arm][-1].tolist() for arm in ("left", "right")
            }
            group["mean_ee_trajectory_z_ranges_m"] = {
                arm: {
                    "min": float(np.min(mean_ee[arm][:, 2])),
                    "max": float(np.max(mean_ee[arm][:, 2])),
                }
                for arm in ("left", "right")
            }
            group["mean_ee_trajectory"] = {
                arm: full_mean_ee[arm].tolist() for arm in ("left", "right")
            }
            group["mean_ee_trajectory_steps"] = {
                arm: int(len(full_mean_ee[arm])) for arm in ("left", "right")
            }
            group["mean_action_steps"] = len(action_steps(group_average_chunks[group["group_id"]]))
            group["atomic_action"] = atomic
            group["target_distance_m"] = target_distance_m
            group["far_distance_m"] = self.config.far_distance_m
            group["distance_class"] = distance_class
            group["candidate_sources"] = [
                candidate_sources[index] for index in group["candidate_indices"]
            ]
            group["contains_previous_remaining"] = (
                previous_candidate_index in group["candidate_indices"]
                if previous_candidate_index is not None else False
            )
        image = obs[self.config.image_key]
        def render_group_image(
            title: str, paths: list[tuple[str, dict]], grasp_axis_points: Any = None,
        ) -> Any:
            canvas = self._rgb_image(image)
            width, height = canvas.size
            draw = ImageDraw.Draw(canvas)
            def pixels(values):
                return [(float(x) * (width - 1) / 255, float(y) * (height - 1) / 255) for x, y in values]
            draw.rectangle((0, 0, width, 24), fill="black")
            draw.text((6, 6), title, fill="white")
            for label, trajectory in paths:
                for name, color in (("left", "red"), ("right", "lime")):
                    path = pixels(trajectory[name])
                    path = [(x, y) for x, y in path if np.isfinite((x, y)).all()]
                    if len(path) >= 2:
                        draw.line(path, fill=color, width=2)
                    if path:
                        x, y = path[-1]
                        draw.ellipse((x-4, y-4, x+4, y+4), outline=color, width=2)
                        draw.text((x+5, y), label, fill=color)
            for x, y in pixels(points):
                draw.ellipse((x-6, y-6, x+6, y+6), outline="cyan", width=3)
            if grasp_axis_points:
                axis = pixels(grasp_axis_points)
                draw.line(axis, fill="magenta", width=4)
                for x, y in axis:
                    draw.ellipse((x-4, y-4, x+4, y+4), fill="magenta")
            return np.asarray(canvas)

        group_images = []
        for group, member_group in zip(groups, group_members, strict=True):
            previous_group = group["contains_previous_remaining"]
            paths = [("MEAN", projected_for_overlay[group["group_id"]])]
            group_images.append(render_group_image(
                f"Group {group['group_id']} | MEAN | "
                f"{'PREVIOUS' if previous_group else 'CANDIDATES'} | "
                f"overlay {OVERLAY_TRAJECTORY_STEPS} steps | {distance_class.upper()}",
                paths,
            ))
            # A retained previous tail can be shorter after partial execution;
            # expose its real length instead of claiming that it has a full
            # execution prefix.
            group["trajectory_steps"] = steps
            group["overlay_trajectory_steps"] = OVERLAY_TRAJECTORY_STEPS
            group["target_points"] = points
            group["current_ee"] = current_ee or None
        vlm = self.config.strategy.ctx.vlm
        instruction = state.original_instruction or str(obs.get("prompt", ""))
        task_hint = str(obs.get("task_name") or obs.get("__task_name") or "")
        task_reference = find_task_reference(task_hint) or find_task_reference(instruction)
        task_kwargs = (
            {
                "task_description": task_reference.description,
                "full_score_condition": task_reference.full_score_condition,
            }
            if task_reference is not None else {}
        )
        extras = [obs[key] for key in self.config.extra_image_keys if obs.get(key) is not None]
        # Expose only one representative per group: its averaged action,
        # including the executable 10-step EE prefix and the complete EE
        # trajectory for the available action horizon. Raw member candidates
        # remain internal diagnostics.
        vlm_groups = [
            {
                "group_id": group["group_id"],
                "endpoint_space": group["endpoint_space"],
                "mean_ee_trajectory": group["mean_ee_trajectory"],
                "mean_ee_trajectory_steps": group["mean_ee_trajectory_steps"],
                "mean_action_steps": group["mean_action_steps"],
                "trajectory_steps": group["trajectory_steps"],
                "overlay_trajectory_steps": group["overlay_trajectory_steps"],
                "atomic_action": group["atomic_action"],
                "endpoint_radius_m": group["endpoint_radius_m"],
                "target_distance_m": group["target_distance_m"],
                "far_distance_m": group["far_distance_m"],
                "distance_class": group["distance_class"],
                "contains_previous_remaining": group["contains_previous_remaining"],
                "target_points": group["target_points"],
                "current_ee": group["current_ee"],
            }
            for group in groups
        ]
        started = time.perf_counter()
        next_subgoal = ""
        next_index = idx + 1
        if next_index < len(state.subgoals):
            next_subgoal = state.subgoals[next_index]
            next_atomic = (
                state.subgoal_atomic_actions[next_index]
                if next_index < len(state.subgoal_atomic_actions)
                else ""
            )
            next_points = (
                state.subgoal_coordinates[next_index]
                if next_index < len(state.subgoal_coordinates)
                else []
            )
            next_subgoal = (
                f"{next_subgoal} | atomic_action={next_atomic or '(unknown)'}"
                f" | target_coordinates={next_points or '(none)'}"
            )
        choice = vlm.select_trajectory_group(
            instruction,
            state.subgoals[idx] if idx < len(state.subgoals) else "",
            vlm_groups, image,
            [*extras, *group_images],
            next_subgoal=next_subgoal,
            before_images=before_images, previous_available=False,
            memory=state.check_memory,
            last_executed_ee_trajectory=state.recent_executed_ee_trajectory(),
            selection_only=selection_only,
            **task_kwargs,
        )
        # A target transition can trigger a second, selection-only VLM call in
        # the same observation.  That call intentionally returns only the new
        # group selection, so retain the progress result already produced by
        # the first call when writing the final diagnostics and overlay.
        if selection_only and preserved_progress:
            choice = dict(choice)
            choice.update({
                key: preserved_progress[key]
                for key in (
                    "decision", "decision_reason", "progress_reason", "memory",
                    "before_to_now_summary", "replan",
                )
                if key in preserved_progress
            })
        group_id = choice.get("group_id")
        valid_ids = set(range(len(groups)))
        if type(group_id) is not int or group_id not in valid_ids:
            raise ValueError(f"Invalid VLM trajectory group: {group_id!r}")
        diagnostic_images = []
        for group, member_group in zip(groups, group_members, strict=True):
            paths = [("MEAN", projected_for_overlay[group["group_id"]])]
            diagnostic_images.append(render_group_image(
                f"Group {group['group_id']} | MEAN | "
                f"{'PREVIOUS' if group['contains_previous_remaining'] else 'CANDIDATES'} | "
                f"overlay {OVERLAY_TRAJECTORY_STEPS} steps | {distance_class.upper()}",
                paths,
            ))
        diagnostic_labels = [
            "PREVIOUS" if group.get("contains_previous_remaining") else f"G{index}"
            for index, group in enumerate(groups)
        ]
        pool = groups[group_id]["candidate_indices"]
        member_group = group_members[group_id]
        average_actions = member_group["average_actions"]
        if not action_steps(average_actions):
            raise ValueError(f"Trajectory group {group_id} has no candidates")
        representative_candidate_index = member_group["consistent_members"][0]
        actions = average_actions
        actual_ee = fk_trajectory(action_steps(actions)[:steps])
        group_selection_path = self._save_group_selection_image(
            state, diagnostic_images, group_id, str(choice.get("reason", "")), diagnostic_labels,
            str(choice.get("memory") or state.check_memory or ""),
            str(choice.get("before_to_now_summary") or ""),
            distance_class,
        )
        diagnostics = {
            "selection_reason": "vlm_endpoint_group",
            "groups": groups, "selected_group_id": group_id, "vlm_reason": choice.get("reason", ""),
            "selected_candidate_index": None,
            "representative_candidate_index": representative_candidate_index,
            "averaged_candidate_indices": member_group["consistent_members"],
            "group_selection_image_path": group_selection_path,
            "candidate_pool": pool,
            "consistent_candidate_indices": member_group["consistent_members"],
            "candidate_selection_distance": member_group["distance"],
            "action_similarity_rms_threshold": 0.05, "action_similarity_max_difference": 0.2,
            "action_similarity_sequence_lengths": [len(action_steps(chunk)) for chunk in full_chunks],
            "executed_ee_endpoint": {name: actual_ee[name][-1].tolist() for name in ("left", "right")},
            "previous_option_available": False,
            "previous_cache_offset": (
                state.action_cache_offset
                if previous_candidate_index is not None else None
            ),
            "previous_steps_available": previous_steps_available,
            "previous_option_selected": False,
            "previous_candidate_index": previous_candidate_index,
            "previous_candidate_group_id": next(
                (
                    group["group_id"] for group in groups
                    if previous_candidate_index is not None
                    and previous_candidate_index in group["candidate_indices"]
                ),
                None,
            ),
            "previous_candidate_selected": previous_candidate_index in member_group["consistent_members"],
            "candidate_sources": candidate_sources,
            "new_candidate_count": new_candidate_count,
            "grouped_candidate_count": len(candidates),
            "projected_trajectories": projected_for_overlay,
            "target_points": points, "atomic_action": atomic,
            "trajectory_steps_requested": steps,
            "grouping_trajectory_steps": GROUP_TRAJECTORY_STEPS,
            "overlay_trajectory_steps": OVERLAY_TRAJECTORY_STEPS,
            "selected_action_steps": len(action_steps(actions)),
            "execution_steps": self.config.action_chunk_size,
            "executed_trajectory_index": group_id,
            "distance_class": distance_class,
            "target_distance_m": target_distance_m,
            "group_endpoint_radius_m": group_endpoint_radius_m,
            "projection_available": projection_available,
            "overlay_projection_available": overlay_projection_available,
            "vlm_replan": bool(choice.get("replan", False)),
            "progress_decision": choice.get("decision", "stay"),
            "decision_reason": choice.get(
                "decision_reason", choice.get("progress_reason", "")
            ),
            "progress_reason": choice.get("progress_reason", ""),
            "progress_memory": choice.get("memory", ""),
            "before_to_now_summary": choice.get("before_to_now_summary", ""),
            "selection_only": bool(selection_only),
        }
        state.log({"type": "vlm_call", "operation": "select_trajectory_group",
                   "infer_count": state.infer_count, "latency_s": round(time.perf_counter()-started, 4), **choice})
        result = dict(candidates[representative_candidate_index])
        overlay = dict(result.get("video_overlay") or {})
        overlay["trajectory_selection"] = diagnostics
        result["video_overlay"], result["actions"] = overlay, actions
        self._save_overlay_image(obs, state, overlay, group_id)
        return result

    def _select_vla_response(self, candidates: list[dict], obs: dict, state: SessionState) -> dict:
        import numpy as np

        candidates = list(candidates)
        retained_index = None
        remainder = self._previous_candidate_remainder(state)
        if remainder:
            retained_index = len(candidates)
            candidates.append({"actions": deepcopy(remainder)})
        chunks = [_response_actions(item) for item in candidates]
        idx = state.current_subgoal_idx
        atomic = ""
        points = None
        if not state.task_completed and 0 <= idx < len(state.subgoal_atomic_actions):
            atomic = str(state.subgoal_atomic_actions[idx]).strip().lower()
        if not state.task_completed and 0 <= idx < len(state.subgoal_coordinates):
            points = state.subgoal_coordinates[idx]
        intrinsic, camera_to_world, image_shape = self._selection_inputs(obs)
        selected, diagnostics = select_candidate(
            chunks,
            ordered=bool(state.subgoals_ordered),
            observe=atomic == "observe",
            points=points,
            arm="",
            intrinsic=intrinsic,
            camera_to_world=camera_to_world,
            image_shape=image_shape,
            current_joints=obs.get("observation/joint_position"),
            depth_image=obs.get("observation/depth_exterior_image_1_left"),
            default_plane_height_m=self.config.table_height_m,
            far_distance_m=self.config.far_distance_m,
            near_trajectory_steps=30,
            far_trajectory_steps=30,
            execution_steps=self.config.action_chunk_size,
        )
        scores = diagnostics.get("geometry_scores", [])
        costs = list(scores)
        if scores and retained_index is not None:
            reference = fk_trajectory(chunks[retained_index])
            reference = np.concatenate((reference["left_joints"], reference["right_joints"]), axis=1)
            for index, chunk in enumerate(chunks):
                trajectory = fk_trajectory(action_steps(chunk)[:self.config.action_chunk_size])
                joints = np.concatenate((trajectory["left_joints"], trajectory["right_joints"]), axis=1)
                length = min(len(joints), len(reference))
                deviation = float(np.sqrt(np.mean((joints[:length] - reference[:length]) ** 2)))
                costs[index] += 10.0 * deviation
            best = int(np.argmin(costs))
            improvement = max(1.0, abs(costs[retained_index]) * 0.1)
            if costs[best] >= costs[retained_index] - improvement:
                selected = retained_index
                diagnostics["candidate_pool"] = [retained_index]
                diagnostics["selection_reason"] = "retain_previous_trajectory"
            else:
                selected = best
                diagnostics["candidate_pool"] = [
                    index for index, cost in enumerate(costs)
                    if cost <= costs[best] + max(1.0, abs(costs[best]) * 0.1)
                    and cost < costs[retained_index] - improvement
                ]
                diagnostics["selection_reason"] = "switch_improved_trajectory"
        elif retained_index is not None and atomic != "observe":
            selected = retained_index
            diagnostics["candidate_pool"] = [retained_index]
            diagnostics["selection_reason"] = "retain_previous_trajectory"
        diagnostics["continuity_costs"] = costs
        diagnostics["retained_candidate_index"] = retained_index
        diagnostics["retained_cache_offset"] = state.action_cache_offset if retained_index is not None else None
        projected = self._project_candidate_trajectories(
            chunks,
            intrinsic,
            camera_to_world,
            image_shape,
            diagnostics.get("trajectory_steps_requested"),
        )
        if projected:
            diagnostics["projected_trajectories"] = projected
        candidate_pool = diagnostics.get("candidate_pool", [selected])
        averaged_actions, averaged_indices, average_distance = average_most_similar_chunks(
            chunks, candidate_pool
        )
        if len(averaged_indices) > 1 and scores:
            _, validation = select_candidate(
                [averaged_actions], ordered=True, points=points, arm="",
                intrinsic=intrinsic, camera_to_world=camera_to_world, image_shape=image_shape,
                current_joints=obs.get("observation/joint_position"),
                depth_image=obs.get("observation/depth_exterior_image_1_left"),
                default_plane_height_m=self.config.table_height_m,
                far_distance_m=self.config.far_distance_m,
                near_trajectory_steps=30,
                far_trajectory_steps=30,
                execution_steps=self.config.action_chunk_size,
            )
            average_scores = validation.get("geometry_scores", [])
            best_original = min(averaged_indices, key=lambda index: scores[index])
            diagnostics["average_geometry_score"] = average_scores[0] if average_scores else None
            average_switch_valid = True
            if average_scores and retained_index is not None:
                trajectory = fk_trajectory(action_steps(averaged_actions)[:self.config.action_chunk_size])
                joints = np.concatenate((trajectory["left_joints"], trajectory["right_joints"]), axis=1)
                length = min(len(joints), len(reference))
                average_cost = average_scores[0] + 10.0 * float(
                    np.sqrt(np.mean((joints[:length] - reference[:length]) ** 2))
                )
                average_switch_valid = average_cost < costs[retained_index] - improvement
            if not average_switch_valid or not average_scores or average_scores[0] > scores[best_original] + max(1.0, abs(scores[best_original]) * 0.1):
                accepted_original = min(averaged_indices, key=lambda index: costs[index])
                averaged_actions = chunks[accepted_original]
                averaged_indices = [accepted_original]
                diagnostics["average_rejected"] = True
        diagnostics["averaged_candidate_indices"] = averaged_indices
        diagnostics["average_action_distance"] = average_distance
        if len(averaged_indices) > 1:
            diagnostics["selection_reason"] = (
                f"{diagnostics.get('selection_reason', 'candidate')}_similarity_average"
            )
        result = dict(candidates[selected])
        overlay = result.get("video_overlay")
        overlay = dict(overlay) if isinstance(overlay, dict) else {}
        overlay["trajectory_selection"] = {
            "selected_index": selected,
            "target_points": deepcopy(points) if points is not None else [],
            "target_arm": "",
            "atomic_action": atomic,
            **diagnostics,
        }
        result["video_overlay"] = overlay
        actual = self._project_candidate_trajectories(
            [averaged_actions], intrinsic, camera_to_world, image_shape,
            diagnostics.get("trajectory_steps_requested", self.config.far_trajectory_steps),
        )
        if actual:
            trajectories = overlay["trajectory_selection"].setdefault("projected_trajectories", [])
            trajectories.append(actual[0])
            actual_index = len(trajectories) - 1
            overlay["trajectory_selection"]["executed_trajectory_index"] = actual_index
            overlay["trajectory_selection"]["execution_steps"] = self.config.action_chunk_size
            self._save_overlay_image(obs, state, overlay, actual_index)
        result["actions"] = averaged_actions
        return result

    @staticmethod
    def _draw_dashed_line(draw: Any, points: list[tuple[float, float]], *, fill: tuple, width: int) -> None:
        """Draw a selected path without covering coincident candidates completely."""
        import math

        dash_length = 8.0
        gap_length = 5.0
        drawing = True
        remaining = dash_length
        for start, end in zip(points, points[1:]):
            x0, y0 = start
            x1, y1 = end
            segment_length = math.hypot(x1 - x0, y1 - y0)
            if segment_length == 0:
                continue
            consumed = 0.0
            while consumed < segment_length:
                step = min(remaining, segment_length - consumed)
                start_ratio = consumed / segment_length
                end_ratio = (consumed + step) / segment_length
                if drawing:
                    draw.line(
                        (
                            x0 + (x1 - x0) * start_ratio,
                            y0 + (y1 - y0) * start_ratio,
                            x0 + (x1 - x0) * end_ratio,
                            y0 + (y1 - y0) * end_ratio,
                        ),
                        fill=fill,
                        width=width,
                    )
                consumed += step
                remaining -= step
                if remaining <= 1e-6:
                    drawing = not drawing
                    remaining = dash_length if drawing else gap_length

    @staticmethod
    def _save_overlay_image(obs: dict, state: SessionState, overlay: dict, selected: int) -> None:
        """Render proxy-owned projected trajectories and persist a diagnostic PNG."""
        if not state.episode_log_dir:
            return
        selection = overlay.get("trajectory_selection")
        if not isinstance(selection, dict):
            return
        projected = selection.get("projected_trajectories")
        image = obs.get("observation/exterior_image_1_left")
        if projected is None or image is None:
            return
        try:
            import numpy as np
            from PIL import ImageDraw

            canvas = OrchestratorProxy._rgb_image(image)
            draw = ImageDraw.Draw(canvas, "RGBA")
            width, height = canvas.size
            colors = (
                (30, 144, 255, 150),
                (255, 165, 0, 150),
                (160, 80, 255, 150),
                (0, 210, 120, 150),
                (255, 215, 0, 150),
                (0, 220, 220, 150),
                (255, 105, 180, 150),
                (120, 220, 40, 150),
                (70, 100, 255, 150),
                (220, 220, 220, 150),
            )
            arms = ("left", "right")
            draw_order = [index for index in range(len(projected)) if index != selected]
            if 0 <= selected < len(projected):
                draw_order.append(selected)
            for candidate_index in draw_order:
                candidate = projected[candidate_index]
                if not isinstance(candidate, dict):
                    continue
                color = colors[candidate_index % len(colors)]
                line_width = 3 if candidate_index == selected else 1
                for arm in arms:
                    points = candidate.get(arm)
                    if not isinstance(points, list):
                        continue
                    pixels = []
                    for point in points:
                        if isinstance(point, (list, tuple)) and len(point) >= 2:
                            x, y = float(point[0]) * (width - 1) / 255.0, float(point[1]) * (height - 1) / 255.0
                            if np.isfinite((x, y)).all():
                                pixels.append((x, y))
                    if len(pixels) >= 2:
                        if candidate_index == selected:
                            OrchestratorProxy._draw_dashed_line(
                                draw, pixels, fill=(255, 45, 45, 255), width=line_width
                            )
                            executed = pixels[:selection.get("execution_steps", 0)]
                            if len(executed) >= 2:
                                draw.line(executed, fill=(0, 255, 120, 255), width=4)
                        else:
                            draw.line(pixels, fill=color, width=line_width)
                    if pixels:
                        radius = 4 if candidate_index == selected else 1
                        x, y = pixels[-1]
                        endpoint_color = (255, 60, 60, 255) if candidate_index == selected else color
                        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=endpoint_color)
            targets = selection.get("target_points")
            for group in selection.get("groups", []):
                display_index = group.get("group_id")
                if not isinstance(display_index, int) or not (0 <= display_index < len(projected)):
                    continue
                for arm in ("left", "right"):
                    path = projected[display_index].get(arm, [])
                    if path and np.isfinite(path[-1]).all():
                        x, y = path[-1][0] * (width-1) / 255, path[-1][1] * (height-1) / 255
                        draw.text((x+4, y+4), f"G{group['group_id']}", fill=(255, 255, 255, 255))
            if isinstance(targets, list):
                for point in targets:
                    if not isinstance(point, (list, tuple)) or len(point) < 2:
                        continue
                    x = float(point[0]) * (width - 1) / 255.0
                    y = float(point[1]) * (height - 1) / 255.0
                    if np.isfinite((x, y)).all():
                        radius = 6
                        draw.ellipse(
                            (x - radius, y - radius, x + radius, y + radius),
                            outline=(0, 255, 255, 255),
                            width=3,
                        )
                        draw.line((x - 9, y, x + 9, y), fill=(0, 255, 255, 255), width=2)
                        draw.line((x, y - 9, x, y + 9), fill=(0, 255, 255, 255), width=2)
            if "executed_trajectory_index" in selection:
                label = f"groups: {len(projected)}  red: selected  green: execute"
            else:
                label = f"batch: {len(projected)}  selected: {selected}  arms: both"
            distance_class = str(selection.get("distance_class") or "unavailable").upper()
            far_label = {"FAR": "YES", "NEAR": "NO"}.get(distance_class, "UNKNOWN")
            label += f"  FAR: {far_label}"
            label_width = min(width - 8, max(82, 7 * len(label) + 8))
            draw.rectangle((4, 4, 4 + label_width, 20), fill=(0, 0, 0, 180))
            draw.text((8, 7), label, fill=(255, 255, 255, 255))
            filename = os.path.join(state.episode_log_dir, f"vla_overlay_{state.infer_count:06d}.png")
            canvas.save(filename)
            selection["overlay_image_path"] = filename
            selection["overlay_image_selected_index"] = int(selected)
        except (ImportError, OSError, TypeError, ValueError) as exc:
            logger.warning("Failed to save proxy overlay image: %s", exc)

    @staticmethod
    def _attach_metadata(response: dict, obs: dict, state: SessionState) -> dict:
        result = dict(response)
        overlay = result.get("video_overlay")
        overlay = dict(overlay) if isinstance(overlay, dict) else {}
        if state.rewritten_instruction is not None:
            result["orchestrator_instruction"] = state.rewritten_instruction
        if state.original_instruction is not None:
            result["orchestrator_original_instruction"] = state.original_instruction
        result["orchestrator_target_revision"] = state.target_revision
        if state.subgoals:
            result["orchestrator_subgoals"] = list(state.subgoals)
            result["orchestrator_subgoal_idx"] = state.current_subgoal_idx
            result["orchestrator_subgoal_adapters"] = list(state.subgoal_adapters)
            result["orchestrator_subgoal_atomic_actions"] = list(state.subgoal_atomic_actions)
            result["orchestrator_subgoal_coordinates"] = list(state.subgoal_coordinates)
            result["orchestrator_subgoal_arms"] = list(state.subgoal_arms)
            idx = state.current_subgoal_idx
            if not state.task_completed and 0 <= idx < len(state.subgoals):
                result["orchestrator_active_subgoal"] = state.subgoals[idx]
            if not state.task_completed and 0 <= idx < len(state.subgoal_atomic_actions):
                overlay["atomic_action"] = state.subgoal_atomic_actions[idx]
        if state.flush_actions:
            result["orchestrator_flush_actions"] = True
            state.flush_actions = False
        if overlay:
            result["video_overlay"] = overlay
        return result

    @staticmethod
    def _drain_logs(state: SessionState) -> None:
        if not state.episode_log_dir or not state.log_entries:
            state.log_entries.clear()
            return
        with open(os.path.join(state.episode_log_dir, "events.jsonl"), "a", encoding="utf-8") as handle:
            for entry in state.log_entries:
                handle.write(json.dumps(entry, default=str) + "\n")
        state.log_entries.clear()

    async def _handle_session(self, session: FrontendSession) -> None:
        assert self.config.backend is not None
        states: dict[str, SessionState] = {}
        episode_counters: dict[str, int] = {}
        backend_connection = await self.config.backend.connect()
        limit = self.config.batch_vlm_concurrency
        semaphore = asyncio.Semaphore(limit) if limit else nullcontext()
        try:
            await session.send_metadata(await backend_connection.recv_metadata())
            while True:
                recv_batch = getattr(session, "recv_batch", None)
                observations = await recv_batch() if callable(recv_batch) else await session.recv_obs()
                if observations is None:
                    break
                if isinstance(observations, dict):
                    observations = [observations]
                if not observations:
                    raise ValueError("frontend returned an empty observation batch")

                items = []
                for index, original_obs in enumerate(observations):
                    slot = str(original_obs.get("env_idx", index))
                    state = states.get(slot)
                    if state is None or self._new_episode(original_obs, state):
                        if state is not None:
                            self._finalize_log(state)
                        episode_counters[slot] = episode_counters.get(slot, 0) + 1
                        state = SessionState(
                            episode_id=episode_counters[slot],
                            episode_marker=original_obs.get("__episode_id"),
                        )
                        states[slot] = state
                        self._rotate_log(state, str(original_obs.get(self.config.prompt_key, "")), original_obs, slot)
                    items.append({"index": index, "slot": slot, "original_obs": original_obs, "state": state})

                async def process_item(item: dict) -> dict:
                    item["before_images"] = deepcopy(item["state"].before_vlm_images)
                    async with semaphore:
                        async with self._global_vlm_semaphore:
                            processed, state = await _run_vlm_call(
                                self.config.strategy.process, item["original_obs"], item["state"]
                            )
                    ctx = getattr(self.config.strategy, "ctx", None)
                    if isinstance(ctx, StrategyContext):
                        processed, metadata = ctx.transform_vla_observation(processed, state)
                        if metadata:
                            state.log({"type": "transform", "phase": "vla", "items": metadata})
                    item["obs"], item["state"] = processed, state
                    item["vla_obs"] = self._prepare_vla_observation(
                        processed,
                        state,
                        self.config.cfg_scale,
                        self.config.cfg_uncond_prompt,
                        self.config.cfg_enabled,
                    )
                    return item

                items = list(await asyncio.gather(*(process_item(item) for item in items)))
                expanded: list[dict] = []
                for item in items:
                    target_distance_m = self._target_distance_m(item["obs"], item["state"])
                    far_target = (
                        target_distance_m is not None
                        and target_distance_m > self.config.far_distance_m
                    )
                    item["joint_state_target_distance_m"] = target_distance_m
                    item["joint_state_noise_enabled"] = (
                        far_target and self.config.vla_joint_state_noise_std > 0
                    )
                    previous_available = (
                        not item["vla_obs"].get("orchestrator_flush_actions")
                        and bool(self._previous_candidate_remainder(item["state"]))
                    )
                    new_candidate_count = max(
                        1, self.config.vla_candidates - int(previous_available)
                    )
                    item["sampled_candidate_count"] = new_candidate_count
                    expanded.extend(
                        self._candidate_vla_observations(
                            item["vla_obs"], item["state"], new_candidate_count,
                            self.config.vla_image_noise_std,
                            self.config.vla_joint_state_noise_std,
                            far_target,
                            image_augmentation=self.config.vla_image_augmentation,
                        )
                    )
                started = time.perf_counter()
                raw_responses = await backend_connection.infer_batch(expanded)
                expected = len(expanded)
                if len(raw_responses) != expected:
                    raise RuntimeError(f"VLA batch response length mismatch: {len(raw_responses)} != {expected}")

                def parse_candidates(values: list[Any]) -> list[dict]:
                    parsed = []
                    for raw in values:
                        candidate = raw if isinstance(raw, dict) else {"actions": raw}
                        if _action_count(_response_actions(candidate)) <= 0:
                            raise RuntimeError("VLA returned an empty candidate action chunk")
                        parsed.append(candidate)
                    return parsed

                responses: list[dict] = [None] * len(items)  # type: ignore[assignment]
                response_offset = 0
                for item_index, item in enumerate(items):
                    sampled_candidate_count = item["sampled_candidate_count"]
                    candidate_responses = parse_candidates(
                        raw_responses[
                            response_offset : response_offset + sampled_candidate_count
                        ]
                    )
                    response_offset += sampled_candidate_count
                    state = item["state"]
                    if item["vla_obs"].get("orchestrator_flush_actions"):
                        self._clear_action_cache(state)
                    atomic = (
                        state.subgoal_atomic_actions[state.current_subgoal_idx]
                        if not state.task_completed
                        and state.current_subgoal_idx < len(state.subgoal_atomic_actions)
                        else ""
                    )
                    grouped = bool(atomic and atomic != "observe")
                    if grouped:
                        async with self._global_vlm_semaphore:
                            selected = await _run_vlm_call(
                                self._select_grouped_vla_response, candidate_responses,
                                item["obs"], state, item["before_images"],
                            )
                    else:
                        selected = self._select_vla_response(
                            candidate_responses, item["obs"], state,
                        )
                    selection = selected.get("video_overlay", {}).get(
                        "trajectory_selection", {}
                    )
                    preserved_progress = None
                    apply_progress = getattr(
                        self.config.strategy, "apply_group_progress", None,
                    )
                    changed = False
                    if grouped and callable(apply_progress):
                        progress = {
                            "decision": selection.get("progress_decision", "stay"),
                            "decision_reason": selection.get(
                                "decision_reason", selection.get("progress_reason", "")
                            ),
                            "progress_reason": selection.get("progress_reason", ""),
                            "memory": selection.get("progress_memory", ""),
                            "before_to_now_summary": selection.get(
                                "before_to_now_summary", ""
                            ),
                        }
                        async with self._global_vlm_semaphore:
                            updated_obs, state, changed = await _run_vlm_call(
                                apply_progress, item["obs"], state, progress,
                            )
                        item["state"] = state
                        item["obs"] = updated_obs
                    if changed and not state.task_completed:
                        # A progress decision may advance exactly one subgoal.
                        # Re-sample once for the new target, but never apply a
                        # second progress decision until a new observation arrives.
                        state.log({
                            "type": "vla_candidates_discarded",
                            "reason": "group_progress_changed_target",
                            "progress_decision": selection.get("progress_decision"),
                            "progress_reason": selection.get("progress_reason"),
                        })
                        item["vla_obs"] = self._prepare_vla_observation(
                            updated_obs,
                            state,
                            self.config.cfg_scale,
                            self.config.cfg_uncond_prompt,
                            self.config.cfg_enabled,
                        )
                        retry_target_distance_m = self._target_distance_m(item["obs"], state)
                        retry_previous_available = bool(
                            self._previous_candidate_remainder(state)
                        )
                        retry_candidate_count = max(
                            1,
                            self.config.vla_candidates - int(retry_previous_available),
                        )
                        retry_expanded = self._candidate_vla_observations(
                            item["vla_obs"], state, retry_candidate_count,
                            self.config.vla_image_noise_std,
                            self.config.vla_joint_state_noise_std,
                            retry_target_distance_m is not None
                            and retry_target_distance_m > self.config.far_distance_m,
                            image_augmentation=self.config.vla_image_augmentation,
                        )
                        # Keep the first call's progress decision separate from
                        # the selection-only retry.  The retry may choose a
                        # different group for the new target, but it must not
                        # erase the progress evidence already established for
                        # this observation.
                        preserved_progress = {
                            "decision": selection.get("progress_decision", "stay"),
                            "decision_reason": selection.get(
                                "decision_reason", selection.get("progress_reason", "")
                            ),
                            "progress_reason": selection.get("progress_reason", ""),
                            "memory": selection.get("progress_memory", ""),
                            "before_to_now_summary": selection.get(
                                "before_to_now_summary", ""
                            ),
                            "replan": bool(selection.get("vlm_replan", False)),
                        }
                        item["joint_state_target_distance_m"] = retry_target_distance_m
                        item["joint_state_noise_enabled"] = (
                            retry_target_distance_m is not None
                            and retry_target_distance_m > self.config.far_distance_m
                            and self.config.vla_joint_state_noise_std > 0
                        )
                        retry_raw = await backend_connection.infer_batch(retry_expanded)
                        if len(retry_raw) != retry_candidate_count:
                            raise RuntimeError(
                                "VLA retry response length mismatch: "
                                f"{len(retry_raw)} != {retry_candidate_count}"
                            )
                        candidate_responses = parse_candidates(retry_raw)
                        item["sampled_candidate_count"] = retry_candidate_count
                        item["before_images"] = deepcopy(state.before_vlm_images)
                        atomic = (
                            state.subgoal_atomic_actions[state.current_subgoal_idx]
                            if state.current_subgoal_idx < len(state.subgoal_atomic_actions)
                            else ""
                        )
                        if atomic and atomic != "observe":
                            async with self._global_vlm_semaphore:
                                selected = await _run_vlm_call(
                                    self._select_grouped_vla_response, candidate_responses,
                                    item["obs"], state, item["before_images"],
                                    selection_only=True,
                                    preserved_progress=preserved_progress,
                                )
                            selection = selected.get("video_overlay", {}).get(
                                "trajectory_selection", {}
                            )
                    if selection.get("vlm_replan") and not state.task_completed:
                        state.pending_replan_reason = (
                            "VLM requested a replan for the current trajectory selection: "
                            f"{selection.get('vlm_reason') or 'the current scene/task state requires replanning.'}"
                        )
                        state.log({"type": "trajectory_replan_requested", "reason": state.pending_replan_reason})
                    self._clear_action_cache(state)
                    action_chunk_size = int(
                        selected.pop("orchestrator_action_chunk_size", self.config.action_chunk_size)
                    )
                    response = self._cached_action_chunk(selected, state, action_chunk_size)
                    responses[item_index] = self._attach_metadata(response, item["obs"], state)
                    state.log({
                        "type": "vla_infer",
                        "infer_count": state.infer_count,
                        "episode_step": state.episode_step,
                        "batch_size": len(expanded),
                        "candidate_count": self.config.vla_candidates,
                        "sampled_candidate_count": item["sampled_candidate_count"],
                        "vla_image_noise_std": self.config.vla_image_noise_std,
                        "vla_image_augmentation": self.config.vla_image_augmentation,
                        "vla_joint_state_noise_std": self.config.vla_joint_state_noise_std,
                        "joint_state_target_distance_m": item.get("joint_state_target_distance_m"),
                        "joint_state_noise_enabled": item.get("joint_state_noise_enabled", False),
                        "vla_state_history_count": len(state.last_dispatched_joint_states),
                        "vla_state_conditioning_count": VLA_STATE_HISTORY_COUNT,
                        "vla_state_history_stride": VLA_STATE_HISTORY_STRIDE,
                        "env_idx": item["obs"].get("env_idx"),
                        "latency_s": round(time.perf_counter() - started, 4),
                        "target_revision": state.target_revision,
                        "coordinates": item["vla_obs"].get("coordinates"),
                        "action_count": _action_count(response.get("actions")),
                        "trajectory_selection": {
                            key: value for key, value in response.get("video_overlay", {}).get("trajectory_selection", {}).items()
                            if key not in {"projected", "projected_trajectories", "trajectories"}
                        },
                    })

                send_action_batch = getattr(session, "send_action_batch", None)
                sent_counts = await send_action_batch(responses) if callable(send_action_batch) else [
                    await session.send_action(response) for response in responses
                ]
                if len(sent_counts) != len(items):
                    raise RuntimeError(f"frontend action response length mismatch: {len(sent_counts)} != {len(items)}")
                for item, response, sent_count in zip(items, responses, sent_counts, strict=True):
                    state = item["state"]
                    count = sent_count if isinstance(sent_count, int) else _action_count(response.get("actions"))
                    if count <= 0:
                        raise RuntimeError("frontend sent an empty VLA action chunk")
                    self._record_executed_chunk(state, response, count)
                    state.episode_step += count
                    state.action_step_count += count
                    state.infer_count += 1
                    self._drain_logs(state)
        finally:
            self._drain_logs_for(states.values())
            for state in states.values():
                self._finalize_log(state)
            await backend_connection.close()

    @classmethod
    def _drain_logs_for(cls, states) -> None:
        for state in states:
            cls._drain_logs(state)

    @staticmethod
    def _finalize_log(state: SessionState) -> None:
        if not state.episode_log_dir:
            return
        path = os.path.join(state.episode_log_dir, "metadata.json")
        try:
            with open(path, encoding="utf-8") as handle:
                metadata = json.load(handle)
            metadata.update(
                {
                    "end_timestamp": time.time(),
                    "infer_count": state.infer_count,
                    "subgoals": list(state.subgoals),
                    "subgoal_coordinates": deepcopy(state.subgoal_coordinates),
                    "subgoal_arms": list(state.subgoal_arms),
                    "subgoals_ordered": state.subgoals_ordered,
                }
            )
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)
        except OSError:
            logger.exception("Failed to finalize episode metadata")
