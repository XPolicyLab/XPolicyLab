"""VLM task decomposition, progress decisions, and visual replanning."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from .base import SessionState
from .subgoal_base import SubgoalBaseStrategy, parse_subgoals
from vlm_orchestrator.vlm.api import extract_arm, extract_coordinates_from_text, infer_atomic_action
from vlm_orchestrator.vlm.task_descriptions import find_task_reference

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SubgoalConfig:
    progress_interval: int = 20

    def __post_init__(self) -> None:
        if self.progress_interval <= 0:
            raise ValueError("progress_interval must be positive")


class SubgoalStrategy(SubgoalBaseStrategy):
    def __init__(self, ctx, config: SubgoalConfig | None = None):
        super().__init__(ctx)
        self.config = config or SubgoalConfig()

    def _images(self, obs: dict, state: SessionState, phase: str):
        image, extras, metadata = self.ctx.get_vlm_images(obs, state, phase)
        if metadata:
            state.transform_metadata.extend(metadata)
            state.log({"type": "transform", "phase": phase, "items": metadata})
        return image, extras

    @staticmethod
    def _task_reference(obs: dict, instruction: str):
        task_hint = str(obs.get("task_name") or obs.get("__task_name") or "")
        return find_task_reference(task_hint) or find_task_reference(instruction)


    def _start_episode(self, obs: dict, state: SessionState) -> dict:
        instruction = self.ctx.get_prompt(obs)
        state.original_instruction = instruction
        state.episode_step = state.action_step_count = 0
        state.current_subgoal_idx = state.target_revision = 0
        state.last_advanced_subgoal = None
        state.progress_transition_pending = False
        state.task_completed = False
        state.check_memory = ""
        state.last_executed_ee_trajectory = None
        state.executed_ee_trajectory_history = []
        state.pending_replan_reason = None
        state.step_at_last_progress_check = 0
        image, extras = self._images(obs, state, "episode_start")
        state.before_vlm_images = [value.copy() for value in [image, *extras] if value is not None]
        fallback = (
            [instruction],
            True,
            [""],
            [infer_atomic_action(instruction)],
            [extract_coordinates_from_text(instruction)],
            [extract_arm(text=instruction)],
        )
        if image is None:
            plan = fallback
        else:
            started = time.perf_counter()
            try:
                prompt, _ = self.ctx.transform_prompt(
                    instruction, obs, state, "episode_start", "vlm_decompose"
                )
                task_reference = self._task_reference(obs, instruction)
                decompose_kwargs = (
                    {
                        "task_description": task_reference.description,
                        "full_score_condition": task_reference.full_score_condition,
                    }
                    if task_reference is not None
                    else {}
                )
                plan = parse_subgoals(
                    self.ctx.vlm.decompose(
                        prompt, image, extras or None, **decompose_kwargs
                    )
                )
                state.log(
                    {
                        "type": "vlm_call",
                        "operation": "decompose",
                        "latency_s": round(time.perf_counter() - started, 4),
                        "task_reference": (
                            task_reference.slug if task_reference is not None else None
                        ),
                        "subgoals": plan[0],
                        "coordinates": plan[4],
                        "arms": plan[5],
                        "ordered": plan[1],
                    }
                )
            except Exception as exc:
                logger.warning("VLM decomposition failed: %s", exc)
                state.log({"type": "vlm_error", "operation": "decompose", "error": str(exc)})
                plan = fallback
        self._install_plan(state, plan)
        state.rewritten_instruction = state.vla_prompt = instruction
        return self._set_subgoal_prompt(obs, state, "episode_start")

    @staticmethod
    def _install_plan(state: SessionState, plan) -> None:
        (
            state.subgoals,
            state.subgoals_ordered,
            state.subgoal_adapters,
            state.subgoal_atomic_actions,
            state.subgoal_coordinates,
            state.subgoal_arms,
        ) = plan

    def _advance(self, obs: dict, state: SessionState) -> dict:
        if state.current_subgoal_idx + 1 >= len(state.subgoals):
            state.step_at_last_progress_check = state.episode_step
            state.task_completed = True
            state.log({"type": "task_completed", "subgoal_index": state.current_subgoal_idx})
            return obs
        state.last_advanced_subgoal = state.subgoals[state.current_subgoal_idx]
        state.current_subgoal_idx += 1
        state.target_revision += 1
        state.progress_transition_pending = True
        state.step_at_last_progress_check = state.episode_step
        state.log(
            {
                "type": "subgoal_advance",
                "subgoal_index": state.current_subgoal_idx,
            }
        )
        return self._set_subgoal_prompt(obs, state)

    def _replan(
        self,
        obs: dict,
        state: SessionState,
        image,
        extras,
        reason: str,
    ) -> dict:
        index = min(state.current_subgoal_idx, len(state.subgoals) - 1)
        try:
            task_reference = self._task_reference(
                obs, state.original_instruction or ""
            )
            task_kwargs = (
                {
                    "task_description": task_reference.description,
                    "full_score_condition": task_reference.full_score_condition,
                }
                if task_reference is not None
                else {}
            )
            result = self.ctx.vlm.replan(
                state.original_instruction or "",
                state.subgoals[index],
                state.subgoals[index + 1 :],
                reason,
                image,
                extras or None,
                before_images=state.before_vlm_images or None,
                memory=state.check_memory,
                last_executed_ee_trajectory=state.recent_executed_ee_trajectory(),
                **task_kwargs,
            )
            plan = parse_subgoals(result)
            if not plan[0]:
                raise ValueError("replanned subgoals must not be empty")
            previous_plan = (
                state.subgoals[index:],
                state.subgoals_ordered,
                state.subgoal_adapters[index:],
                state.subgoal_atomic_actions[index:],
                state.subgoal_coordinates[index:],
                state.subgoal_arms[index:],
            )
            if plan == previous_plan:
                state.step_at_last_progress_check = state.episode_step
                state.log({"type": "subgoal_replan_unchanged", "reason": reason})
                return self._set_subgoal_prompt(obs, state, "replan")
            target_changed = any(new[0] != old[0] for new, old in zip(
                (plan[0], plan[2], plan[3], plan[4], plan[5]),
                (
                    previous_plan[0], previous_plan[2], previous_plan[3],
                    previous_plan[4], previous_plan[5],
                ),
                strict=True,
            ))
            self._install_plan(state, plan)
            state.current_subgoal_idx = 0
            state.last_advanced_subgoal = None
            if target_changed:
                state.target_revision += 1
            state.step_at_last_progress_check = state.episode_step
            state.log(
                {
                    "type": "subgoal_replan",
                    "reason": reason,
                    "memory": state.check_memory,
                    "subgoals": list(state.subgoals),
                    "coordinates": list(state.subgoal_coordinates),
                    "arms": list(state.subgoal_arms),
                }
            )
            return self._set_subgoal_prompt(obs, state, "replan")
        except Exception as exc:
            logger.warning("VLM replan failed: %s", exc)
            state.log({"type": "vlm_error", "operation": "replan", "error": str(exc)})
            return obs

    @staticmethod
    def _progress_value(result, key: str, default):
        if isinstance(result, dict):
            return result.get(key, default)
        return getattr(result, key, default)

    def _apply_progress_result(
        self, obs: dict, state: SessionState, result, image, extras, operation: str,
    ) -> tuple[dict, bool]:
        state.step_at_last_progress_check = state.episode_step
        index = min(state.current_subgoal_idx, len(state.subgoals) - 1)
        decision = str(self._progress_value(result, "decision", "stay")).strip().lower()
        if decision not in {"stay", "advance", "replan"}:
            decision = "stay"
        if isinstance(result, dict):
            reason = str(
                result.get("decision_reason")
                or result.get("progress_reason")
                or result.get("reason", "")
            ).strip()
        else:
            reason = str(
                getattr(result, "decision_reason", "")
                or getattr(result, "reason", "")
            ).strip()
        updated_memory = str(self._progress_value(result, "memory", "")).strip()
        before_to_now_summary = str(
            self._progress_value(result, "before_to_now_summary", "")
        ).strip()
        requested_decision = decision
        if updated_memory:
            state.check_memory = updated_memory
        log_entry = {
            "type": "vlm_call" if operation == "assess_progress" else "progress_decision",
            "operation": operation,
            "subgoal_index": index,
            "decision": decision,
            "requested_decision": requested_decision,
            "reason": reason,
            "decision_reason": reason,
            "memory": state.check_memory,
            "before_to_now_summary": before_to_now_summary,
        }
        before_target = (
            state.current_subgoal_idx,
            state.target_revision,
            state.task_completed,
        )
        if state.progress_transition_pending:
            # The first chunk after a target transition belongs to the new
            # subgoal's approach/transition. Requiring one more fresh chunk
            # makes a split motion unable to complete two subgoals at once.
            if decision == "advance":
                decision = "stay"
                reason = (
                    f"{reason} Progress held for one fresh action chunk after the "
                    "subgoal transition."
                ).strip()
                log_entry["decision"] = decision
                log_entry["transition_guard_blocked"] = True
            state.progress_transition_pending = False
        state.log(log_entry)
        if decision == "replan":
            state.progress_transition_pending = False
            obs = self._replan(obs, state, image, extras, reason)
        else:
            state.last_advanced_subgoal = None
            if decision == "advance":
                obs = self._advance(obs, state)
        state.before_vlm_images = [value.copy() for value in [image, *extras]]
        return obs, before_target != (
            state.current_subgoal_idx,
            state.target_revision,
            state.task_completed,
        )

    def apply_group_progress(
        self, obs: dict, state: SessionState, result: dict,
    ) -> tuple[dict, SessionState, bool]:
        image, extras = self._images(obs, state, "progress")
        if image is None or not state.subgoals:
            return obs, state, False
        obs, changed = self._apply_progress_result(
            obs, state, result, image, extras, "select_trajectory_group",
        )
        state.group_progress_checked = True
        return obs, state, changed

    def _check_progress(self, obs: dict, state: SessionState) -> dict:
        image, extras = self._images(obs, state, "progress")
        state.step_at_last_progress_check = state.episode_step
        if image is None or not state.subgoals:
            return obs
        index = min(state.current_subgoal_idx, len(state.subgoals) - 1)
        try:
            task_reference = self._task_reference(
                obs, state.original_instruction or ""
            )
            task_kwargs = (
                {
                    "task_description": task_reference.description,
                    "full_score_condition": task_reference.full_score_condition,
                }
                if task_reference is not None else {}
            )
            result = self.ctx.vlm.assess_progress(
                state.original_instruction or "",
                state.subgoals[index],
                state.subgoals[index + 1 :],
                state.last_advanced_subgoal,
                image,
                extras or None,
                before_images=state.before_vlm_images or None,
                memory=state.check_memory,
                last_executed_ee_trajectory=state.recent_executed_ee_trajectory(),
                **task_kwargs,
            )
            obs, _ = self._apply_progress_result(
                obs, state, result, image, extras, "assess_progress",
            )
        except Exception as exc:
            logger.warning("VLM progress check failed: %s", exc)
            state.log(
                {"type": "vlm_error", "operation": "assess_progress", "error": str(exc)}
            )
            state.before_vlm_images = [value.copy() for value in [image, *extras]]
        return obs

    def process(self, obs: dict, state: SessionState) -> tuple[dict, SessionState]:
        if self.is_new_episode(obs, state):
            obs = self._start_episode(obs, state)
        elif state.pending_replan_reason and state.subgoals and not state.task_completed:
            # A trajectory selector can observe that the policy is staging an
            # object at an intermediate location.  Replan against the next
            # real observation so the new scene, gripper state and memory are
            # part of the VLM decision.
            reason = state.pending_replan_reason
            state.pending_replan_reason = None
            image, extras = self._images(obs, state, "progress")
            if image is not None:
                obs = self._replan(obs, state, image, extras, reason)
        elif state.group_progress_checked:
            state.group_progress_checked = False
        elif (
            not state.task_completed
            and
            state.subgoals
            and state.episode_step - state.step_at_last_progress_check
            >= self.config.progress_interval
        ):
            obs = self._check_progress(obs, state)
        result = dict(obs)
        if (
            not state.task_completed
            and state.subgoals
            and 0 <= state.current_subgoal_idx < len(state.subgoal_atomic_actions)
        ):
            result["atomic_action"] = state.subgoal_atomic_actions[state.current_subgoal_idx]
        return self._attach_coordinates(result, state), state
