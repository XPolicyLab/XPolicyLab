"""Prompt construction, response parsing and closed-loop bookkeeping.

One turn is:

1. :meth:`VLMAgent.observe` takes the fresh RoboDojo observation;
2. :meth:`VLMAgent.act` renders the state, the camera frames, the in-context
   demonstration and the recent history into a prompt, asks the VLM for a few
   discrete commands, and ships them to the simulation client as a
   ``sequence`` envelope (:mod:`motion`);
3. :meth:`VLMAgent.note_execution` folds the per-command results the client
   reports back into the next prompt.

Everything the model saw and answered is written to a per-episode log
directory so a run can be replayed offline.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import commands as command_grammar
from . import icl_demos
from . import main_prompts
from . import main_route
from . import motion
from .action_parser import parse_action_response
from .controller import ArmState, ControllerConfig, ControllerError, read_arm_states
from .vlm_client import VLMClient, VLMError
from .grounding import annotate_head, annotate_wrist, autocontrast

# Only a wrong key or endpoint ends a run; every other HTTP failure is a turn error after its retries.
PERMANENT_API_STATUSES = (401, 403, 404)

_MAIN_DEFAULTS = {
    "profile": None,                 # profile YAML; None = profiles/robodojo_x5_main.yaml
    "max_commands_per_turn": 4,
    "history_turns": 12,
    "memory_history": True,
    "memory_scratchpad": True,
    "done_limit": 3,                 # consecutive "done" turns without success that end the episode
    "home_plan_steps": 10000,        # step cap for a planned 'home' (it travels far)
    "max_plan_steps": 10000,         # step cap for any other planned motion
    "motion_settle_steps": 10,       # hold the final joint target for 0.4 s before judging tracking
    "tcp_offset_m": main_route.TCP_OFFSET_M,
}

# In-context demonstrations: the expert demonstration of the task being run,
# read from ``<demo_bank>/<task>/demo.json`` (see icl_demos.py).
_ICL_DEFAULTS = {
    "demo_bank": None,               # bank directory with one <task>/ entry per task (required)
    "demo_count": 1,                 # entries per task: <task>/, then <task>+2/, ...
}


STEP_COST_NOTE = ("STEP BUDGET: the step counter in the state counts 25 Hz control steps - a 10-20 cm move costs "
                  "roughly 10-20 steps, a gripper command about 10 - so prefer few, decisive commands.")


def _main_config(raw: Any) -> dict:
    """Knobs of the control loop (the ``main`` block of deploy.yml), with the profile loaded."""
    raw = dict(raw or {})
    unknown = sorted(set(raw) - set(_MAIN_DEFAULTS))
    if unknown:
        raise ValueError(f"unknown main option(s): {unknown}")
    cfg = {**_MAIN_DEFAULTS, **raw}
    for key in ("max_commands_per_turn", "history_turns", "done_limit",
                "home_plan_steps", "max_plan_steps", "motion_settle_steps"):
        value = cfg[key]
        minimum = 0 if key == "history_turns" else 1
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or value != int(value) or value < minimum):
            raise ValueError(f"main.{key} must be an integer >= {minimum}")
        cfg[key] = int(value)
    for key in ("memory_history", "memory_scratchpad"):
        cfg[key] = bool(cfg[key])
    cfg["tcp_offset_m"] = float(cfg["tcp_offset_m"])
    if not math.isfinite(cfg["tcp_offset_m"]) or cfg["tcp_offset_m"] < 0:
        raise ValueError("main.tcp_offset_m must be finite and nonnegative")
    cfg["profile_data"] = main_prompts.load_profile(cfg["profile"])
    cfg["profile"] = cfg["profile_data"].get("_source")
    return cfg


def _icl_config(raw: Any) -> dict:
    """The ICL block (``vlm_agent.icl``): which demonstration bank to read."""
    raw = dict(raw or {})
    unknown = sorted(set(raw) - set(_ICL_DEFAULTS))
    if unknown:
        raise ValueError(f"unknown icl option(s): {unknown}")
    cfg = {**_ICL_DEFAULTS, **raw}
    value = str(cfg["demo_bank"] or "").strip()
    if not value:
        raise ValueError("icl.demo_bank is required: the bank holding <task>/demo.json")
    path = icl_demos.resolve_icl_path(value)
    if not path.exists():
        raise ValueError(f"icl.demo_bank not found: {value} (looked at {path})")
    cfg["demo_bank"] = str(path)
    count = cfg["demo_count"]
    if isinstance(count, bool) or not isinstance(count, (int, float)) or count != int(count) or int(count) < 1:
        raise ValueError("icl.demo_count must be an integer >= 1")
    cfg["demo_count"] = int(count)
    return cfg


class AgentConfig:
    """Runtime knobs, read from the ``vlm_agent`` block of ``deploy.yml``."""

    def __init__(self, raw: dict | None = None):
        raw = dict(raw or {})
        self.vlm_profile: str | None = raw.get("vlm_profile")
        # The head camera and both wrist views, every turn.
        self.cameras: list[str] = list(raw.get("cameras") or ["cam_head", "cam_left_wrist", "cam_right_wrist"])
        self.image_max_width: int = int(raw.get("image_max_width", 640))
        self.jpeg_quality: int = int(raw.get("jpeg_quality", 80))
        self.gripper_settle = {"enabled": True, "min_s": 0.5, "stable_s": 0.2, "max_s": 2.5, "tolerance": 0.002}
        self.gripper_settle.update(dict(raw.get("gripper_settle") or {}))
        self.gripper_settle["enabled"] = bool(self.gripper_settle["enabled"])
        for key in ("min_s", "stable_s", "max_s", "tolerance"):
            value = float(self.gripper_settle[key])
            if not math.isfinite(value) or value < 0 or (key != "min_s" and value == 0):
                raise ValueError(f"gripper_settle.{key} must be finite and positive (min_s may be zero)")
            self.gripper_settle[key] = value
        if max(self.gripper_settle["min_s"], self.gripper_settle["stable_s"]) > self.gripper_settle["max_s"]:
            raise ValueError("gripper_settle.max_s must cover min_s and stable_s")
        self.max_tokens: int = int(raw.get("max_tokens", 3000))
        self.temperature: float | None = raw.get("temperature")
        if self.temperature is not None:
            self.temperature = float(self.temperature)
        # Endpoint-specific; low does not guarantee that reasoning is disabled.
        self.reasoning_effort: str | None = raw.get("reasoning_effort", "medium")
        self.request_timeout_s: float = float(raw.get("request_timeout_s", 60.0))
        self.max_attempts: int = int(raw.get("max_attempts", 6))
        self.retry_delay_s: float = float(raw.get("retry_delay_s", 2.0))
        # Cover the initial call plus five retries. deploy configures the
        # simulation RPC timeout above this budget; the episode deadline still applies.
        self.decision_budget_s: float = float(raw.get("decision_budget_s", 660.0))
        self.max_consecutive_errors = int(raw.get("max_consecutive_errors", 2))
        self.log_dir: str = str(raw.get("log_dir", "eval_result/_vlm_logs"))
        self.log_images: bool = bool(raw.get("log_images", True))
        # Drawn on the head image (see grounding.annotate_head).
        aids = dict(raw.get("visual_aids") or {})
        self.visual_aids = {
            "markers": bool(aids.get("markers", True)),
            "grid": bool(aids.get("grid", True)),
            "enhance": bool(aids.get("enhance", True)),
        }
        # Finger position can suggest obstructed closure, but cannot identify
        # the obstacle or confirm that the target object was grasped.
        self.measure_gripper: bool = bool(raw.get("measure_gripper", True))
        self.grasp_open_threshold: float = float(raw.get("grasp_open_threshold", 0.05))
        self.main = _main_config(raw.get("main"))
        self.icl = _icl_config(raw.get("icl"))
        self.controller = ControllerConfig(**dict(raw.get("controller") or {}))

    def as_dict(self) -> dict:
        return {
            "vlm_profile": self.vlm_profile,
            "cameras": self.cameras,
            "image_max_width": self.image_max_width,
            "jpeg_quality": self.jpeg_quality,
            "gripper_settle": dict(self.gripper_settle),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
            "request_timeout_s": self.request_timeout_s,
            "max_attempts": self.max_attempts,
            "retry_delay_s": self.retry_delay_s,
            "decision_budget_s": self.decision_budget_s,
            "max_consecutive_errors": self.max_consecutive_errors,
            "log_dir": self.log_dir,
            "log_images": self.log_images,
            "visual_aids": dict(self.visual_aids),
            "measure_gripper": self.measure_gripper,
            "grasp_open_threshold": self.grasp_open_threshold,
            "main": {key: value for key, value in self.main.items() if key != "profile_data"},
            "icl": dict(self.icl),
            "controller": self.controller.as_dict(),
        }


class VLMAgent:
    """Closed-loop VLM policy over RoboDojo observations."""

    def __init__(self, cfg: AgentConfig, client: VLMClient, run_tag: str = "", task_name: str = ""):
        self.cfg = cfg
        self.client = client
        self.task_name = str(task_name or "").strip()
        self.icl_msgs: list[dict] | None = None
        self.icl_demos: list = []
        self.run_root = _resolve_log_root(cfg.log_dir) / (run_tag or time.strftime("%Y-%m-%d_%H-%M-%S"))
        self.episode_index = -1
        self._missing_cameras_warned = False
        self._missing_calibration_warned = False
        self._wrist_overlay_warned: set[str] = set()
        self.fatal_api_error = False
        self.reset()

    # ------------------------------------------------------------------ loop

    def reset(self) -> None:
        """Start a new episode: clear history, plan and per-episode counters."""
        self.episode_index += 1
        self.decision_index = 0
        self._missing_wrist_warned: set[str] = set()
        self.obs: dict | None = None
        self.states: dict[str, ArmState] | None = None
        self.execution: dict = {}
        self.episode_meta: dict = {}
        self.consecutive_errors = 0
        # Turn bookkeeping (see _act_main): the model's own notes plus the
        # harness-written history, the results the model has not seen yet, the
        # height at which each arm last closed on something, consecutive "done"s.
        self.main_memory = main_prompts.AgentMemory(
            use_history=self.cfg.main["memory_history"], use_scratchpad=self.cfg.main["memory_scratchpad"],
            history_turns=self.cfg.main["history_turns"])
        self.main_last_results: list[dict] = []
        self.main_pending_results: list[dict] = []
        self.main_turn_commands: list[str] = []
        self.main_grasp_facts: dict[str, dict] = {}
        self.main_done_count = 0
        self.main_system_prompt: str | None = None
        self.stop_reason: str | None = "permanent_api_error" if self.fatal_api_error else None
        self.observed_at = time.monotonic()
        self.log_dir = self.run_root / f"episode_{self.episode_index:03d}"

    def observe(self, obs: dict) -> None:
        """Store the new observation."""
        self.obs = obs
        self.observed_at = time.monotonic()
        try:
            self.states = read_arm_states(obs)
        except ControllerError:
            # Surfaced with full context when act() needs the state.
            self.states = None
            raise

    def act(self) -> dict:
        """Ask the VLM for one turn of commands and return its motion envelope (see :mod:`motion`)."""
        if self.states is None or self.obs is None:
            raise RuntimeError("act() called before a successful observe()")
        if self.stop_reason:
            return motion.stop(self.stop_reason)
        deadline = time.monotonic() + self.cfg.decision_budget_s
        if "policy_budget_s" in self.obs:
            deadline = min(deadline, self.observed_at + max(0, float(self.obs["policy_budget_s"])))
        if time.monotonic() >= deadline:
            self.stop_reason = "wall_budget"
            return motion.stop(self.stop_reason)
        return self._act_main(deadline)

    def note_execution(self, report: dict) -> None:
        """Record what the simulator actually ran for the last chunk."""
        if not isinstance(report, dict):
            return
        if report.get("phase") == "episode_start":
            self.episode_meta = dict(report)
            control_dt = report.get("control_dt_s")
            if isinstance(control_dt, (int, float)) and math.isfinite(control_dt) and control_dt > 0:
                self.cfg.controller.control_hz = 1.0 / control_dt
            return
        if report.get("phase") == "episode_end":
            summary = dict(report)
            summary.update({"model": self.client.model, "vlm_profile": self.cfg.vlm_profile,
                            "policy_stop_reason": self.stop_reason})
            try:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                (self.log_dir / "episode_summary.json").write_text(
                    json.dumps(summary, indent=2, default=str), encoding="utf-8"
                )
            except OSError as exc:
                print(f"[vlm_agent] cannot write episode summary: {type(exc).__name__}", flush=True)
            return
        self.execution = dict(report)
        if isinstance(report.get("command_results"), list):
            self._note_main_execution(report)
        path = self.log_dir / f"decision_{self.decision_index:04d}.json"
        if path.is_file():
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                record["execution"] = report
                path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
            except (OSError, ValueError) as exc:
                print(f"[vlm_agent] cannot append execution log: {type(exc).__name__}", flush=True)

    # ------------------------------------------------------------------ turn

    def _main_state(self) -> dict:
        """The state dict main_prompts.state_text renders: fingertip centres in cm, measured openings."""
        main = self.cfg.main
        used = self.execution.get("steps_used", self.episode_meta.get("steps_used"))
        state: dict[str, Any] = {"steps_used": used,
                                 "table_z_cm": round(self.cfg.controller.table_z * 100.0, 1)}
        for prefix, arm in (self.states or {}).items():
            state[prefix] = main_route.arm_state(prefix, arm.pos, arm.quat, arm.gripper_real, arm.gripper_closed_floor,
                                                 main["tcp_offset_m"], self.cfg.grasp_open_threshold)
        return state

    def _collect_main_frames(self) -> tuple[list[dict], list[str]]:
        """Every configured camera the observation carries, the head view annotated, each with a caption."""
        vision = (self.obs or {}).get("vision") or {}
        aids = self.cfg.visual_aids
        table_z = int(round(self.cfg.controller.table_z * 100.0))
        grid_note = (f"; the thin grid is drawn ON THE TABLE TOP (z = {table_z}) every 10 cm with a fainter line every 5 cm, labelled x... along the "
                     "near edge and y... along the left edge - read object x/y positions from it" if aids["grid"] else "")
        marker_note = ("; markers: circle = wrist, cross = fingertip centre (orange L = left arm, blue R = right arm), "
                       "line = wrist to fingertips" if aids["markers"] else "")
        frames: list[dict] = []
        captions: list[str] = []
        for name in dict.fromkeys(self.cfg.cameras):
            color = (vision.get(name) or {}).get("color")
            if color is None:
                if name not in self._missing_wrist_warned:
                    self._missing_wrist_warned.add(name)
                    print(f"[vlm_agent] camera {name} is not in the observation; skipped", flush=True)
                continue
            if name == "cam_head":
                color = self._annotate(color, units="cm")
                caption = ("cam_head (the robot's head camera, above and behind the torso looking forward and down at "
                           "the table; the robot body is at the bottom, +y is up the image, the left arm appears on the "
                           "left" + marker_note + grid_note + ")")
            elif name.endswith("_wrist"):
                prefix = name.split("_")[1]
                color, marked = self._annotate_wrist(color, name, prefix)
                caption = (f"{name} (wrist camera of the {prefix} arm, looking along its fingers"
                           + ("; the white cross marks the FINGERTIP CENTRE, i.e. exactly where the fingers will "
                              "close - to grasp, the object must sit ON the cross, not merely between the two "
                              "finger silhouettes; the arrows from the cross show the world +x/+y/+z directions "
                              "at fingertip depth and are each 5 cm long with a tick every 1 cm; the cyan jaw axis marks the finger "
                              "opening/closing direction. Use these to estimate a move command"
                              if marked else "; its image axes are not world axes") + ")")
            else:
                caption = name
            if aids["enhance"]:
                color = autocontrast(color)
            jpeg = encode_jpeg(color, self.cfg.image_max_width, self.cfg.jpeg_quality)
            if jpeg is None:
                continue
            url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
            frames.append({"camera": name, "jpeg": jpeg, "part": {"type": "image_url", "image_url": {"url": url}}})
            captions.append(caption)
        if not frames and vision and not self._missing_cameras_warned:
            self._missing_cameras_warned = True
            print(f"[vlm_agent] none of the configured cameras {self.cfg.cameras} are in the observation "
                  f"{sorted(vision)}", flush=True)
        return frames, captions

    def _ensure_icl(self) -> None:
        """Load the demonstration messages once per run."""
        if self.icl_msgs is not None:
            return
        task = (self.task_name or str(self.episode_meta.get("task_name") or "")
                or str((self.obs or {}).get("task_name") or ""))
        demos = icl_demos.load_demos(self.cfg.icl, task=task)
        self.icl_demos = demos
        self.icl_msgs = icl_demos.demo_messages(demos)
        print(
            f"[vlm_agent] ICL: {len(demos)} demo(s) for {task or '?'} "
            f"({sum(demo.n_images() for demo in demos)} images, "
            f"{sum(len(demo.frames) for demo in demos)} phases)",
            flush=True,
        )
        if self.log_dir:
            for index, demo in enumerate(demos, start=1):
                icl_demos.save_demo(demo, self.log_dir / ("icl_demo" if index == 1 else f"icl_demo{index}"))

    def _act_main(self, deadline: float) -> dict:
        """One turn: images + state in, a sequence of discrete commands out."""
        cfg = self.cfg
        main = cfg.main
        self.decision_index += 1
        turn = self.decision_index
        instruction = main_prompts.strip_reset_clause((self.obs or {}).get("instruction")) or "(none given)"
        self._ensure_icl()
        if self.main_system_prompt is None:
            self.main_system_prompt = main_prompts.system_prompt(
                main["profile_data"], main["max_commands_per_turn"], main["memory_scratchpad"],
                context={"table_z": round(cfg.controller.table_z * 100.0, 1)}, extra_note=STEP_COST_NOTE)
        frames, captions = self._collect_main_frames()
        state = self._main_state()
        text = main_prompts.turn_text(turn, instruction, state, self.main_last_results, self.main_memory.render(),
                                      captions, grasp_facts=self.main_grasp_facts)
        messages = (
            [{"role": "system", "content": self.main_system_prompt}]
            + list(self.icl_msgs or [])
            + [{"role": "user", "content": [frame["part"] for frame in frames] + [{"type": "text", "text": text}]}]
        )
        record: dict[str, Any] = {
            "vlm_profile": cfg.vlm_profile, "model": self.client.model,
            "episode": self.episode_index, "decision": turn, "turn": turn,
            "prompt": text, "system_prompt": self.main_system_prompt, "config": cfg.as_dict(),
            "previous_execution": self.execution, "calibration": (self.obs or {}).get("calibration"),
            "state": state, "last_results": list(self.main_last_results),
            "grasp_facts": {name: dict(fact) for name, fact in self.main_grasp_facts.items()},
            "cameras": [frame["camera"] for frame in frames], "requests": [],
        }
        pending: list[dict] = []
        executed: list[command_grammar.Command] = []
        done = False
        request_started = time.monotonic()
        try:
            response = self.client.complete(messages, max_tokens=cfg.max_tokens, temperature=cfg.temperature,
                                            reasoning_effort=cfg.reasoning_effort, deadline=deadline)
        except VLMError as exc:
            record["error"] = str(exc)
            record["latency_s"] = round(exc.latency_s if exc.latency_s is not None else time.monotonic() - request_started, 3)
            record["attempts"] = exc.attempts
            record["retry_history"] = exc.retry_history
            record["api_error"] = {"status_code": exc.status_code, "retryable": exc.retryable}
            record["requests"].append({key: record[key] for key in ("error", "latency_s", "attempts", "api_error", "retry_history")})
            if exc.status_code in PERMANENT_API_STATUSES:
                self.fatal_api_error = True
                self.stop_reason = "permanent_api_error"
            print(f"[vlm_agent] turn {turn}: {exc}; waiting", flush=True)
            pending.append({"command": "(no command)", "kind": "error", "arm": None, "ok": False,
                            "note": "your previous reply could not be obtained; reply with the JSON object"})
        else:
            record.update(response=response["text"], usage=response["usage"], finish_reason=response["finish_reason"],
                          latency_s=round(response["latency_s"], 2), attempts=response["attempts"])
            record["requests"].append(dict(response))
            parsed, parse_info = parse_action_response(response["text"])
            record["parse"] = parse_info
            if parsed is None:
                record["error"] = "reply parse error: " + str(parse_info.get("error", "no response"))
                pending.append({"command": "(unparseable reply)", "kind": "error", "arm": None, "ok": False,
                                "note": f"your reply was not a valid JSON object ({parse_info.get('error')}); "
                                        "reply with the JSON object only"})
            else:
                self.main_memory.update_scratchpad(parsed.get("memory"))
                commands_parsed, errors = command_grammar.parse_command_list(parsed.get("commands"))
                record.update(scene=_short(parsed.get("scene")), progress=_short(parsed.get("progress")),
                              plan=_short(parsed.get("plan"), 400), memory=self.main_memory.scratchpad)
                for command in commands_parsed[: main["max_commands_per_turn"]]:
                    if command.kind == "done":
                        done = True
                        break
                    executed.append(command)
                if len(commands_parsed) > main["max_commands_per_turn"]:
                    errors.append(f"only the first {main['max_commands_per_turn']} commands of a turn are executed")
                record["command_errors"] = errors
                pending.extend({"command": "(invalid)", "kind": "invalid", "arm": None, "ok": False, "note": err}
                               for err in errors)
                if done:
                    if self.main_done_count == 0:
                        # 45 of RoboDojo's 54 tasks only count success with both arms back at their origin,
                        # and 4 say so; the harness returns the arms on the model's first "done" so that the
                        # instruction need not (main_prompts.strip_reset_clause removes the sentence).
                        executed.extend(command_grammar.Command("home", arm) for arm in command_grammar.ARMS)
                        record["auto_home"] = True
                    pending.append(dict(main_prompts.DONE_RESULT))
        record["api_calls"] = sum(request.get("attempts", 1) for request in record["requests"])
        self.consecutive_errors = self.consecutive_errors + 1 if "error" in record else 0
        if self.consecutive_errors >= cfg.max_consecutive_errors:
            self.stop_reason = self.stop_reason or "consecutive_model_errors"
        if done:
            self.main_done_count += 1
            if self.main_done_count >= main["done_limit"]:
                # after done_limit consecutive "done" turns without success the episode ends as agent_done
                self.stop_reason = self.stop_reason or "agent_done"
        else:
            self.main_done_count = 0

        wire = [command.to_dict() for command in executed]
        if not wire:
            # Nothing executable this turn: hold still so the episode loop keeps going.
            wire = [command_grammar.Command("wait").to_dict()]
            record["held"] = True
        options = {"hold_steps": cfg.controller.hold_steps, "gripper_steps": cfg.controller.gripper_steps,
                   "tcp_offset_m": main["tcp_offset_m"], "grasp_open_threshold": cfg.grasp_open_threshold,
                   "home_plan_steps": main["home_plan_steps"], "max_plan_steps": main["max_plan_steps"],
                   "motion_settle_steps": main["motion_settle_steps"]}
        plan = {"turn": turn, "commands": [command.text() for command in executed], "done": done,
                "stop_reason": self.stop_reason}
        envelope = motion.sequence(wire, plan, options)
        self.main_turn_commands = plan["commands"]
        self.main_pending_results = pending
        record["commands"] = plan["commands"]
        record["pending_results"] = pending
        record["envelope"] = {"kind": envelope["kind"], "commands": len(wire), "options": options}
        self._write_log(record, frames)
        if self.stop_reason == "agent_done":
            return motion.stop(self.stop_reason)
        return envelope

    def _note_main_execution(self, report: dict) -> None:
        """Fold the simulation client's per-command results into memory, feedback and grasp facts."""
        executed = [result for result in report.get("command_results") or [] if isinstance(result, dict)]
        results = executed + list(self.main_pending_results)
        self.main_pending_results = []
        self.main_last_results = results
        state = self._main_state()
        for result in reversed(executed):
            if isinstance(result.get("after"), dict):
                state.update({prefix: arm for prefix, arm in result["after"].items() if isinstance(arm, dict)})
                break
        if report.get("steps_used") is not None:
            state["steps_used"] = report["steps_used"]
        self.main_memory.record_turn(self.decision_index, self.main_turn_commands, results, state)
        for result in executed:
            if result.get("kind") != "gripper" or not result.get("ok") or result.get("arm") not in ("left", "right"):
                continue
            arm = result["arm"]
            value = result.get("value")
            if value is None:
                continue
            if float(value) >= 0.5:
                self.main_grasp_facts.pop(arm, None)
                continue
            if result.get("held"):
                position = ((result.get("after") or {}).get(arm) or {}).get("position_cm") or []
                if len(position) == 3:
                    self.main_grasp_facts[arm] = {"grasp_z": round(float(position[2]), 1),
                                                  "opening": result.get("opening")}

    # ---------------------------------------------------------------- images

    def _annotate(self, color: Any, units: str = "m") -> Any:
        """Draw the configured overlay on the head frame.

        A missing calibration costs the aid and one warning; the run continues
        on the raw image.
        """
        aids = self.cfg.visual_aids
        if aids["markers"] or aids["grid"]:
            calibration = (self.obs or {}).get("calibration", {}).get("cam_head")
            if calibration is None:
                if not self._missing_calibration_warned:
                    self._missing_calibration_warned = True
                    print("[vlm_agent] no head-camera calibration in the observation; "
                          "drawing no grid or markers", flush=True)
            else:
                color = annotate_head(color, calibration, self.states,
                                      table_z=self.cfg.controller.table_z,
                                      grid=aids["grid"], markers=aids["markers"], units=units)
        return color

    def _annotate_wrist(self, color: Any, camera: str, prefix: str) -> tuple[Any, bool]:
        """Mark the fingertip centre and world axes on a wrist frame; ``(image, drawn)``.

        Needs this camera's calibration in the observation and the arm's pose.
        The cross and the 5 cm arrows make the offset between the fingers and
        the object measurable in the image.
        """
        if not self.cfg.visual_aids["markers"]:
            return color, False
        calibration = ((self.obs or {}).get("calibration") or {}).get(camera)
        state = (self.states or {}).get(prefix)
        if calibration is None or state is None:
            if camera not in self._wrist_overlay_warned:
                self._wrist_overlay_warned.add(camera)
                print(f"[vlm_agent] no calibration for {camera} in the observation; wrist view left unmarked",
                      flush=True)
            return color, False
        try:
            return annotate_wrist(color, calibration, state.pos, state.quat,
                                  offset=float(self.cfg.main["tcp_offset_m"])), True
        except Exception as exc:  # noqa: BLE001 - an aid must not fail the turn
            if camera not in self._wrist_overlay_warned:
                self._wrist_overlay_warned.add(camera)
                print(f"[vlm_agent] wrist overlay failed for {camera}: {type(exc).__name__}: {exc}", flush=True)
            return color, False

    # ------------------------------------------------------------------- logs

    def _write_log(self, record: dict, frames: list[dict]) -> None:
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            stem = f"decision_{self.decision_index:04d}"
            with open(self.log_dir / f"{stem}.json", "w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2, ensure_ascii=False, default=str)
            if self.cfg.log_images:
                for frame in frames:
                    (self.log_dir / f"{stem}_{frame['camera']}.jpg").write_bytes(frame["jpeg"])
        except Exception as exc:  # noqa: BLE001 - logging must never kill an eval
            print(f"[vlm_agent] failed to write decision log: {exc}", flush=True)


def encode_jpeg(color: Any, max_width: int, quality: int) -> bytes | None:
    """Encode an HxWx3 RGB array as JPEG, downscaled to ``max_width``."""
    array = np.asarray(color)
    if array.ndim != 3 or array.shape[2] < 3:
        return None
    array = np.ascontiguousarray(array[:, :, :3])
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    from PIL import Image  # imported lazily: the sim client never needs it

    image = Image.fromarray(array, mode="RGB")
    if max_width and image.width > max_width:
        height = max(1, round(image.height * max_width / image.width))
        image = image.resize((max_width, height), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality))
    return buffer.getvalue()


def _short(value: Any, limit: int = 160) -> str:
    text = str(value or "").strip().replace("\n", " ")
    return text[:limit]


def _resolve_log_root(log_dir: str) -> Path:
    """Resolve ``log_dir`` against the RoboDojo root when it is relative."""
    path = Path(log_dir)
    if path.is_absolute():
        return path
    for start in (Path.cwd(), Path(__file__).absolute().parent, Path(__file__).resolve().parent):
        for parent in (start, *start.parents):
            if (parent / "scripts" / "robodojo.sh").is_file():
                return parent / path
    return Path(os.getcwd()) / path
