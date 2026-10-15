"""Run the standalone RoboDojo L3 inspect-inspired policy inside RoboDojo."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import numpy as np
import yaml

from XPolicyLab.utils.live_frames import LiveFrameRecorder
from XPolicyLab.utils.process_data import decode_image_bit

from .embodiment_docs import ARX_X5_JOINT_DOCS, build_arx_x5_docs
from .policy import (
    CapabilityFailure,
    InfrastructureFailure,
    JointAgentPolicy,
    RoboDojoActionSpec,
    client_config_from_env,
    validate_rgb_only_depth,
    vision_model_view_array,
)
from .probe import probe_joint_directions
from .trace import (
    SCHEMA_VERSION,
    env_step,
    frame_ranges,
    measured_flange_poses,
    named_state,
    video_frame_counts,
)
from .types import Observation

_GRIPPER_EPS_PATTERN = "gripper_eps="


def _robodojo_root() -> Path:
    raw = os.environ.get("ROBODOJO_ROOT")
    if raw:
        return Path(raw)
    raise InfrastructureFailure(
        "ROBODOJO_ROOT is unset; point it at the RoboDojo checkout so the "
        "adapter can read the embodiment limits."
    )


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _gripper_eps(root: Path) -> float:
    text = (root / "env/robot_manager/control_manager.py").read_text(encoding="utf-8")
    for token in text.split():
        if token.startswith(_GRIPPER_EPS_PATTERN):
            return float(token.split("=", 1)[1].rstrip(",)"))
    raise RuntimeError("RoboDojo MetaControl gripper_eps was not found")


# What one env step actually delivers, which is well under what the actuators
# could sweep in the same time. A step target only lands if the joint can
# accelerate to it and brake again inside the step, and RoboDojo's position
# controller falls behind without complaining, so the velocity ceiling is an
# upper bound on the declaration rather than the declaration itself.
#
# Measured over the first 42-task sweep, 3494 planned moves, against CuRobo's
# path resolution of 0.0027 rad per waypoint: under 1% of moves land more than
# 100 mm off target up to 0.054 rad per step, 62% miss at 0.081 and 88% at
# 0.108, so 0.05 is the last measured-clean value. The ceiling would declare
# 0.2 (5 rad/s at 25 Hz), four times too permissive; step counts derived from
# that compress long paths into jumps the arm cannot follow, which was the
# dominant failure mode of that sweep.
_TRACKED_ARM_STEP_RAD = 0.05
# The jaw's ceiling works out to its whole range in one step. Every recorded
# opening -- the clean signal, since nothing blocks an open the way a grasped
# object blocks a close -- took four env steps to cover that range instead.
_TRACKED_GRIPPER_STEP = 0.25


def _declared_step(span: float, per_step: float) -> float | None:
    """The per-step budget to declare for one arm joint, in radians.

    ``per_step`` is the actuator ceiling; what gets declared is the smaller of
    that and what the controller demonstrably tracks.
    """
    if not np.isfinite(per_step) or per_step <= 0:
        return None
    per_step = min(per_step, _TRACKED_ARM_STEP_RAD)
    if per_step >= span:
        return None
    return float(per_step)


def _gripper_step(span: float, per_step: float) -> float | None:
    if not np.isfinite(per_step) or per_step <= 0:
        return None
    return float(min(span, per_step, _TRACKED_GRIPPER_STEP))


def _arm_velocity_limit_sim(root: Path) -> float:
    text = (root / "env/robot_manager/robot_config/x5.py").read_text(encoding="utf-8")
    for line in text.splitlines():
        if "velocity_limit_sim" in line:
            return float(line.split("=", 1)[1].strip().rstrip(","))
    raise RuntimeError("RoboDojo X5 velocity_limit_sim was not found")


def _positive_rate(name: str, value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        raise InfrastructureFailure(
            f"RoboDojo reported {name}={value!r}; it must be finite and > 0. "
            "Check the observation section of the RoboDojo env config."
        )
    return float(value)


def _is_decoded_rgb_image(color: Any) -> bool:
    return isinstance(color, np.ndarray) and color.ndim >= 3


def _camera_color_rgb(color: Any) -> np.ndarray:
    if _is_decoded_rgb_image(color):
        return color
    if os.environ.get("EVAL_ENV_TYPE") == "debug":
        return np.asarray(decode_image_bit(color))
    return np.asarray(color)


def _observation(task_env: Any, policy_step: int) -> Observation:
    raw = task_env.get_obs()
    camera_views = raw.get("vision") or {}
    images = {
        name: _camera_color_rgb(view["color"])
        for name, view in camera_views.items()
        if isinstance(view, dict) and "color" in view
    }
    extra: dict[str, Any] = {}
    for key in ("additional_info", "data_format_version", "env_idx"):
        if key in raw:
            extra[key] = raw[key]
    env_idx = raw.get("env_idx", 0)
    env_seeds = getattr(task_env, "env_seeds", None)
    if isinstance(env_idx, int) and env_seeds is not None:
        try:
            layout_id = env_seeds[env_idx]
        except (IndexError, KeyError, TypeError):
            layout_id = None
        if layout_id is not None:
            extra["layout_id"] = int(layout_id)
    task_name = getattr(task_env, "task_name", None)
    if isinstance(task_name, str) and task_name.strip():
        extra["task"] = task_name.strip()
    current_env_step = env_step(task_env)
    if current_env_step is not None:
        extra["env_step"] = current_env_step
    return Observation(
        images=images,
        state=raw.get("state") or {},
        instruction=(
            raw.get("instruction")
            or raw.get("instructions")
            or getattr(task_env, "instruction", None)
        ),
        step=policy_step,
        remaining_steps=_remaining_env_steps(task_env),
        extra=extra,
    )


def _remaining_env_steps(task_env: Any) -> int | None:
    """Env steps left before RoboDojo ends the episode.

    This, not the LLM call budget, is what actually runs out: it is the
    ``take_action_cnt >= step_lim`` arm of RoboDojo's ``is_episode_end``. The
    model is told the number so it can spend steps where they matter, coarsely
    while approaching and finely on contact.
    """
    limit = getattr(task_env, "step_lim", None)
    used = env_step(task_env)
    if used is None:
        return None
    try:
        return max(0, int(limit) - used)
    except (TypeError, ValueError):
        return None


def _action_spec(task_env: Any) -> RoboDojoActionSpec:
    """Read bounds and control rate from the live RoboDojo embodiment."""
    action_type = os.environ.get("L3_INSPECT_ACTION_TYPE", "joint")
    if action_type != "joint":
        raise InfrastructureFailure(
            f"L3_INSPECT_ACTION_TYPE={action_type!r} is unsupported; set it to "
            "'joint' or unset it. RoboDojo's EE action is quaternion+IK, and "
            "this adapter rejects quaternion pose spaces as unsafe for "
            "per-dimension interpolation."
        )
    manager = getattr(task_env, "robot_manager", None)
    if manager is None:
        if os.environ.get("EVAL_ENV_TYPE") != "debug":
            raise RuntimeError("RoboDojo robot_manager is required to derive action bounds")
        return _debug_action_spec()

    labels: list[str] = []
    low: list[float] = []
    high: list[float] = []
    max_step: list[float | None] = []
    sides: list[str] = []
    mount_poses: dict[str, Any] = {}
    control_hz = _positive_rate(
        "collect_freq",
        float(getattr(task_env.obs_manager, "collect_freq", 0.0)),
    )
    collect_interval = _positive_rate(
        "collect_interval",
        float(getattr(task_env.obs_manager, "collect_interval", 0.0)),
    )
    gripper_step = _gripper_step(1.0, _gripper_eps(_robodojo_root()) * collect_interval)
    for index, robot in enumerate(manager.robot_list):
        if robot.type != "target":
            continue
        side = str(robot.arm_name).removesuffix("_arm")
        sides.append(side)
        mount_pose = getattr(robot, "entity_origin_pose", None)
        if mount_pose is not None:
            mount_poses[side] = mount_pose
        asset = manager.robot_key[index]
        limits = _as_numpy(asset.data.soft_joint_pos_limits[0, robot.arm_joint_indices])
        names = tuple(str(name) for name in robot.arm_joints_name)
        if limits.shape != (len(names), 2):
            raise RuntimeError(
                f"{side} joint limits have shape {limits.shape}, expected {(len(names), 2)}"
            )
        velocities = _as_numpy(asset.data.joint_vel_limits[0, robot.arm_joint_indices])
        if velocities.shape != (len(names),):
            raise RuntimeError(
                f"{side} joint velocity limits have shape {velocities.shape}, "
                f"expected {(len(names),)}"
            )
        labels.extend(f"{side}_{name}" for name in names)
        low.extend(float(value) for value in limits[:, 0])
        high.extend(float(value) for value in limits[:, 1])
        max_step.extend(
            _declared_step(float(hi - lo), float(vel) / control_hz)
            for lo, hi, vel in zip(limits[:, 0], limits[:, 1], velocities, strict=True)
        )
        labels.append(f"{side}_gripper")
        low.append(0.0)
        high.append(1.0)
        max_step.append(gripper_step)

    spec = RoboDojoActionSpec(
        labels=tuple(labels),
        low=np.asarray(low, dtype=np.float64),
        high=np.asarray(high, dtype=np.float64),
        control_hz=control_hz,
        docs=build_arx_x5_docs(mount_poses),
        max_step=tuple(max_step),
    )
    if tuple(sides) != ("left", "right") or len(spec.labels) != 14:
        raise RuntimeError(
            "RoboDojo_Agent_L3_Inspect expected a dual 6-DoF ARX X5 enumerated left "
            f"then right, matching the joint channel order, got sides={sides}, "
            f"dimensions={len(spec.labels)}"
        )
    return spec


def _debug_action_spec() -> RoboDojoActionSpec:
    root_value = os.environ.get("ROBODOJO_ROOT")
    if not root_value:
        raise InfrastructureFailure(
            "ROBODOJO_ROOT is unset; L3 inspect debug evaluation reads the joint "
            "limits from the RoboDojo checkout."
        )
    root = Path(root_value)
    urdf = ElementTree.parse(root / "Assets/Robots/x5/X5A.urdf")
    arm_joints = []
    for joint in urdf.getroot().findall("joint"):
        name = joint.get("name", "")
        if name not in {f"joint{i}" for i in range(1, 7)}:
            continue
        limit = joint.find("limit")
        if limit is None:
            raise RuntimeError(f"URDF joint {name} has no limit")
        arm_joints.append(
            (name, float(limit.get("lower", "")), float(limit.get("upper", "")))
        )
    arm_joints.sort(key=lambda item: int(item[0].removeprefix("joint")))
    if len(arm_joints) != 6:
        raise RuntimeError(f"expected six X5 arm joints in URDF, got {arm_joints}")
    env_config = yaml.safe_load(
        (root / "env_cfg/arx_x5.yml").read_text(encoding="utf-8")
    )
    sim_config = yaml.safe_load(
        (root / "env_cfg/sim/sim_config.yml").read_text(encoding="utf-8")
    )
    control_hz = _positive_rate(
        "collect_freq", float(env_config["observation"]["collect_freq"])
    )
    collect_interval = _positive_rate(
        "collect_interval", 1.0 / (float(sim_config["dt"]) * control_hz)
    )
    arm_velocity = _arm_velocity_limit_sim(root)
    gripper_step = _gripper_step(1.0, _gripper_eps(root) * collect_interval)
    labels: list[str] = []
    low: list[float] = []
    high: list[float] = []
    max_step: list[float | None] = []
    for side in ("left", "right"):
        for name, lower, upper in arm_joints:
            labels.append(f"{side}_{name}")
            low.append(lower)
            high.append(upper)
            max_step.append(_declared_step(upper - lower, arm_velocity / control_hz))
        labels.append(f"{side}_gripper")
        low.append(0.0)
        high.append(1.0)
        max_step.append(gripper_step)
    return RoboDojoActionSpec(
        labels=tuple(labels),
        low=np.asarray(low, dtype=np.float64),
        high=np.asarray(high, dtype=np.float64),
        control_hz=control_hz,
        docs=ARX_X5_JOINT_DOCS,
        max_step=tuple(max_step),
    )


def _mark_incomplete_episode_failed(task_env: Any) -> None:
    if task_env.is_episode_end():
        return
    running = (
        task_env.get_running_env_idx_list()
        if hasattr(task_env, "get_running_env_idx_list")
        else [0]
    )
    success = getattr(task_env, "success", None)
    if success is None:
        raise RuntimeError(
            "L3 inspect policy stopped before official termination and the "
            "environment does not expose a failure state."
        )
    for env_idx in running:
        success[env_idx] = False
    task_env.is_episode_end()
    print(
        f"[L3 inspect] policy stopped early; marked envs failed: {running}",
        flush=True,
    )


_MIN_SECRET_LEN = 4


def _transcript_secrets(env: Mapping[str, str]) -> set[str]:
    secrets: set[str] = set()
    # A comma-separated list, because the run rotates over several keys when one
    # is throttled. Each named variable is redacted by value, and by name too:
    # the name is not a secret, but a transcript has no reason to carry it.
    key_envs = str(client_config_from_env(env)["api_key_env"])
    for key_env in (part.strip() for part in str(key_envs).split(",")):
        if not key_env:
            continue
        secrets.add(key_env)
        api_key_value = env.get(key_env, "")
        if api_key_value:
            secrets.add(str(api_key_value))
    for name, value in env.items():
        if not value or len(str(value)) < _MIN_SECRET_LEN:
            continue
        lowered = name.lower()
        if "api_key" in lowered or lowered.endswith("_key") or "secret" in lowered:
            secrets.add(str(value))
    return secrets


def _sanitize_for_transcript(value: Any, secrets: set[str]) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in ("api-key", "api_key", "authorization")):
                sanitized[key] = "[redacted]"
                continue
            sanitized[key] = _sanitize_for_transcript(item, secrets)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_for_transcript(item, secrets) for item in value]
    if isinstance(value, str):
        if value.startswith("data:image/"):
            return "[image omitted]"
        lowered = value.lower()
        if any(token in lowered for token in ("api-key", "api_key", "authorization", "bearer ")):
            return "[redacted]"
        for secret in secrets:
            if len(secret) >= _MIN_SECRET_LEN and secret in value:
                return "[redacted]"
        return value
    return value


def _transcript_path(trace_dir: str, run_id: str, layout_id: int | None) -> Path:
    layout_segment = f"layout-{layout_id}" if layout_id is not None else "layout-unknown"
    return Path(trace_dir) / run_id / layout_segment / "l3_inspect_transcript.json"


def _write_json_atomically(path: Path, payload: Mapping[str, Any]) -> None:
    """Replace the transcript in one step so a reader never sees half of it.

    The transcript is rewritten whole after every turn, and the console reads it
    while the episode runs, so a plain write would expose truncated JSON.
    """
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temp, path)


def _live_frame_recorder(
    trace_dir: str | None, env: Mapping[str, str], policy: JointAgentPolicy
) -> LiveFrameRecorder | None:
    """Record frames beside the transcript, for the console's live panel."""
    if not trace_dir:
        return None
    try:
        run_id = env.get("ROBODOJO_RUN_ID", "l3-inspect")
        layout_id = policy.audit_config().get("scene", {}).get("init_seed")
        return LiveFrameRecorder(_transcript_path(trace_dir, run_id, layout_id).parent)
    except Exception as error:
        print(f"[L3 inspect] live frames unavailable: {error}", flush=True)
        return None


def _model_view_images(
    images: Mapping[str, Any], env: Mapping[str, str]
) -> dict[str, Any]:
    """Match the planner's camera flips/masks when dumping live frames."""
    return {
        name: vision_model_view_array(image, name, env)
        for name, image in images.items()
    }

def _write_transcript(
    policy: JointAgentPolicy | None,
    task_env: Any,
    env: Mapping[str, str],
    *,
    termination_reason: str | None,
    failure_kind: str | None,
    error_message: str | None = None,
    instruction: str | None = None,
    turns: list[dict[str, Any]] | None = None,
    in_progress: bool = False,
) -> None:
    directory = env.get("L3_INSPECT_TRACE_DIR")
    if not directory:
        return
    run_id = env.get("ROBODOJO_RUN_ID", "l3-inspect")
    layout_id: int | None = None
    if policy is not None and hasattr(policy, "audit_config"):
        layout_id = policy.audit_config().get("scene", {}).get("init_seed")
    path = _transcript_path(directory, run_id, layout_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    secrets = _transcript_secrets(env)
    if policy is None or not hasattr(policy, "audit_config"):
        result = _sanitize_for_transcript(
            {
                "schema_version": SCHEMA_VERSION,
                "task": getattr(task_env, "task_name", None),
                "run_id": run_id,
                "in_progress": in_progress,
                "instruction": instruction,
                "llm_calls": getattr(policy, "calls", 0) if policy is not None else 0,
                "termination_reason": termination_reason,
                "failure_kind": failure_kind,
                "error_message": error_message,
                "official_success": list(getattr(task_env, "success", [])),
                "turns": turns or [],
                "transcript": (
                    policy.transcript() if policy is not None and hasattr(policy, "transcript") else []
                ),
            },
            secrets,
        )
        _write_json_atomically(path, result)
        return
    policy_config = policy.audit_config()
    layout_id = policy_config.get("scene", {}).get("init_seed")
    result = _sanitize_for_transcript(
        {
            "schema_version": SCHEMA_VERSION,
            "task": getattr(task_env, "task_name", None),
            "run_id": run_id,
            "layout_id": layout_id,
            "in_progress": in_progress,
            "instruction": instruction,
            "llm_calls": policy.calls,
            "hindsight": policy.hindsight,
            "termination_reason": termination_reason,
            "failure_kind": failure_kind,
            "error_message": error_message,
            "official_success": list(getattr(task_env, "success", [])),
            "policy_config": policy_config,
            "turns": turns or [],
            "transcript": policy.transcript(),
        },
        secrets,
    )
    _write_json_atomically(path, result)


def _write_probe_report(
    report: Mapping[str, Any], task_env: Any, env: Mapping[str, str]
) -> Path:
    """Persist one diagnostic probe outside the scored transcript schema."""
    trace_dir = env.get("L3_INSPECT_TRACE_DIR")
    if not trace_dir:
        raise InfrastructureFailure(
            "L3_INSPECT_EMBODIMENT_PROBE=1 requires L3_INSPECT_TRACE_DIR"
        )
    run_id = env.get("ROBODOJO_RUN_ID", "l3-inspect-probe")
    path = Path(trace_dir) / run_id / "arx_x5_joint_probe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(report)
    payload["task"] = getattr(task_env, "task_name", None)
    payload["seed"] = getattr(task_env, "seed", None)
    payload["run_id"] = run_id
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def eval_one_episode(
    TASK_ENV: Any,
    model_client: Any,
    *,
    policy_factory: Callable[..., JointAgentPolicy] | None = None,
) -> None:
    del model_client  # The standalone loop generates actions through its local Agent.
    env = dict(os.environ)
    trace_dir = env.get("L3_INSPECT_TRACE_DIR")
    if not trace_dir:
        print(
            "[L3 inspect] L3_INSPECT_TRACE_DIR is unset; transcript is not archived "
            "for this episode",
            flush=True,
        )
    policy: JointAgentPolicy | None = None
    policy_step = 0
    stopped = False
    termination_reason: str | None = None
    failure_kind: str | None = None
    infrastructure_error: InfrastructureFailure | None = None
    turns: list[dict[str, Any]] = []
    instruction: str | None = None
    recorder: LiveFrameRecorder | None = None

    def flush_transcript() -> None:
        """Publish the transcript mid-episode so the console can follow along.

        Without this the file appears only in the `finally` block, and a running
        job has nothing structured on disk to read. A flush that fails must not
        end an episode that is otherwise fine.
        """
        if not trace_dir:
            return
        try:
            _write_transcript(
                policy,
                TASK_ENV,
                env,
                termination_reason=None,
                failure_kind=None,
                instruction=instruction,
                turns=turns,
                in_progress=True,
            )
        except (Exception, InfrastructureFailure) as error:
            print(f"[L3 inspect] transcript flush failed: {error}", flush=True)

    try:
        validate_rgb_only_depth(env)
        step_limit_raw = str(env.get("L3_INSPECT_ENV_STEP_LIMIT", "") or "").strip()
        if step_limit_raw:
            try:
                step_limit_override = int(step_limit_raw)
            except ValueError as error:
                raise InfrastructureFailure(
                    f"L3_INSPECT_ENV_STEP_LIMIT={step_limit_raw!r} is not an int"
                ) from error
            if step_limit_override <= 0:
                raise InfrastructureFailure(
                    "L3_INSPECT_ENV_STEP_LIMIT must be a positive int"
                )
            # Probe-only: leave the task's default step_lim alone unless a
            # launch explicitly asks for a longer official budget.
            TASK_ENV.step_lim = step_limit_override
            print(
                f"[L3 inspect] env step_lim overridden to {step_limit_override}",
                flush=True,
            )
        action_spec = _action_spec(TASK_ENV)
        if env.get("L3_INSPECT_EMBODIMENT_PROBE") == "1":
            report = probe_joint_directions(TASK_ENV, delta_rad=0.05)
            report["embodiment_docs"] = action_spec.docs
            path = _write_probe_report(report, TASK_ENV, env)
            termination_reason = "embodiment_probe"
            failure_kind = "diagnostic"
            stopped = True
            print(f"[L3 inspect] embodiment probe written to {path}", flush=True)
            return
        policy = (
            JointAgentPolicy(action_spec=action_spec, env=env)
            if policy_factory is None
            else policy_factory(
                action_spec=action_spec,
                env=env,
                task_env=TASK_ENV,
            )
        )
        while not TASK_ENV.is_episode_end():
            observation_frame_start = video_frame_counts(TASK_ENV)
            observation = _observation(TASK_ENV, policy_step)
            observation_frame_end = video_frame_counts(TASK_ENV)
            instruction = observation.instruction
            if policy_step == 0:
                policy.prepare(observation)
                recorder = _live_frame_recorder(trace_dir, env, policy)
            if recorder is not None:
                recorder.record(_model_view_images(observation.images, env), step=policy_step)
            transcript_before = (
                len(policy.transcript())
                if hasattr(policy, "transcript") and policy.transcript() is not None
                else 0
            )
            observation_record = {
                "env_step": env_step(TASK_ENV),
                "state": named_state(action_spec.labels, observation.state),
                "flange": measured_flange_poses(observation.state),
                "cameras": frame_ranges(
                    observation_frame_start,
                    observation_frame_end,
                    include_frame=True,
                ),
            }
            try:
                chunk = policy.act(observation)
            except (Exception, InfrastructureFailure) as error:
                calls = (
                    policy.transcript()[transcript_before:]
                    if hasattr(policy, "transcript") and policy.transcript() is not None
                    else []
                )
                turns.append(
                    {
                        "policy_step": policy_step,
                        "observation": observation_record,
                        "llm_calls": calls,
                        "decision": None,
                        "execution": None,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                raise
            policy_step += 1
            calls = (
                policy.transcript()[transcript_before:]
                if hasattr(policy, "transcript") and policy.transcript() is not None
                else []
            )
            execution_frame_start = video_frame_counts(TASK_ENV)
            env_step_start = env_step(TASK_ENV)
            played = 0
            episode_end_mid_chunk = False
            for action in chunk.actions:
                TASK_ENV.take_action(dict(action.data))
                played += 1
                # RoboDojo appends the official mp4 only inside get_obs. Flush
                # after every waypoint so interpolated motion is visible.
                executed = _observation(TASK_ENV, observation.step)
                if recorder is not None:
                    recorder.record(
                        _model_view_images(executed.images, env),
                        step=observation.step,
                    )
                if action.meta.get("request_stop"):
                    stopped = True
                    reason = action.meta.get("stop_reason")
                    termination_reason = str(reason)
                    print(
                        f"[L3 inspect] policy {reason}: {action.meta.get('stop_detail')}",
                        flush=True,
                    )
                    break
                if TASK_ENV.is_episode_end():
                    episode_end_mid_chunk = played < len(chunk.actions)
                    break
            policy.confirm_executed(played)
            execution_frame_end = video_frame_counts(TASK_ENV)
            decision = dict(chunk.meta.get("trace") or {})
            turns.append(
                {
                    "policy_step": observation.step,
                    "observation": observation_record,
                    "llm_calls": calls,
                    "decision": decision,
                    "execution": {
                        "env_step_start": env_step_start,
                        "env_step_end": env_step(TASK_ENV),
                        "planned_waypoints": int(
                            decision.get("planned_waypoints", len(chunk.actions))
                        ),
                        "executed_waypoints": played,
                        "stopped_early": played < len(chunk.actions),
                        "episode_end_mid_chunk": episode_end_mid_chunk,
                        "cameras": frame_ranges(
                            execution_frame_start,
                            execution_frame_end,
                        ),
                    },
                }
            )
            flush_transcript()
            if stopped:
                break
    except CapabilityFailure as error:
        print(f"[L3 inspect] {error}", flush=True)
        failure_kind = "capability"
        termination_reason = "policy_error"
        stopped = True
    except InfrastructureFailure as error:
        infrastructure_error = error
    except Exception as error:
        # The message can come from an SDK that echoes the request, so it goes
        # through the same redaction the transcript uses before being surfaced.
        detail = _sanitize_for_transcript(str(error), _transcript_secrets(env))
        infrastructure_error = InfrastructureFailure(
            f"unexpected infrastructure failure: {type(error).__name__}: {detail}"
        )
    finally:
        finalize_error: BaseException | None = None
        try:
            if infrastructure_error is None:
                if stopped or not TASK_ENV.is_episode_end():
                    _mark_incomplete_episode_failed(TASK_ENV)
            if trace_dir:
                _write_transcript(
                    policy,
                    TASK_ENV,
                    env,
                    termination_reason=termination_reason,
                    failure_kind=(
                        "infrastructure"
                        if infrastructure_error is not None
                        else failure_kind
                    ),
                    error_message=(
                        str(infrastructure_error) if infrastructure_error is not None else None
                    ),
                    instruction=instruction,
                    turns=turns,
                )
            if infrastructure_error is None and policy is not None:
                print(f"[L3 inspect] {policy.calls} llm calls", flush=True)
        # InfrastructureFailure is not an Exception, and a finalization step must
        # not be able to escape and replace the error the episode already carries.
        except (Exception, InfrastructureFailure) as error:
            finalize_error = error
            print(f"[L3 inspect] finalize error: {error}", flush=True)
    if infrastructure_error is not None:
        raise infrastructure_error
    if finalize_error is not None:
        if isinstance(finalize_error, InfrastructureFailure):
            raise finalize_error
        detail = _sanitize_for_transcript(str(finalize_error), _transcript_secrets(env))
        raise InfrastructureFailure(
            "failed to finalize the episode: "
            f"{type(finalize_error).__name__}: {detail}"
        )


def eval_one_episode_batch(TASK_ENV: Any, model_client: Any) -> None:
    num_envs = int(getattr(TASK_ENV, "num_envs", 1) or 1)
    if num_envs > 1:
        raise InfrastructureFailure(
            f"RoboDojo started {num_envs} environments; RoboDojo_Agent_L3_Inspect is "
            "one conversation per environment. Set eval_batch=false so RoboDojo runs "
            "a single env."
        )
    eval_one_episode(TASK_ENV, model_client)
