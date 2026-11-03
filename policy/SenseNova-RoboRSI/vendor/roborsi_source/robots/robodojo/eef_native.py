# Copyright 2026 The HEART Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""RoboProbe native observation/specification conversion and episode execution.

RoboDojo's run_eval remains responsible for scoring and official result files.
This module supplies only the policy callback on the simulator owner thread.
"""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any
from xml.etree import ElementTree

import numpy as np
import yaml

from roborsi.evaluation.result import write_json_atomic
from roborsi.utils.logging import get_logger

from .eef_contracts import CapabilityFailure, InfrastructureFailure, RoboDojoActionSpec
from .eef_docs import eef_docs_from_joint_docs
from .eef_trace import env_step, frame_ranges, video_frame_counts
from .eef_types import Observation
from .episode import mark_incomplete_episode_failed as _mark_incomplete_episode_failed
from .joint_docs import ARX_X5_JOINT_DOCS, build_arx_x5_docs

if TYPE_CHECKING:
    from .eef_policy import EefEpisodePolicy

logger = get_logger("robodojo.eef")

_GRIPPER_EPS_PATTERN = "gripper_eps="
_TRACKED_ARM_STEP_RAD = 0.05
_TRACKED_GRIPPER_STEP = 0.25


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
            raise RuntimeError(
                "RoboDojo robot_manager is required to derive action bounds"
            )
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


def _decode_single_image_bit(image_bit):
    """Decode one encoded image buffer into an HWC uint8 RGB array."""
    if isinstance(image_bit, np.ndarray) and image_bit.dtype.kind in {"S", "U"}:
        image_bit = image_bit.item() if image_bit.ndim == 0 else image_bit.tobytes()

    if isinstance(image_bit, str):
        image_bit = image_bit.encode("utf-8")
    elif isinstance(image_bit, memoryview):
        image_bit = image_bit.tobytes()

    if isinstance(image_bit, (bytes, bytearray)):
        # Fixed-width HDF5 byte columns pad the tail with NUL.
        image_bit = image_bit.rstrip(b"\0")
    elif isinstance(image_bit, np.ndarray):
        image_bit = np.ascontiguousarray(image_bit)

    # The returned array is RGB. This is the conclusion for this repo — do not
    # "correct" it with the usual "cv2 means BGR" rule. cv2.imencode/imdecode
    # only carry channels in the order they were handed in, so the round trip
    # is an identity on channel order, and XPolicyLab trajectory files and
    # runtime observations are encoded from RGB arrays. Adding a COLOR_BGR2RGB
    # swap here (or in any caller) is what actually breaks the channel order.
    import cv2

    image = cv2.imdecode(np.frombuffer(image_bit, np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(
            f"Failed to decode image bits (type={type(image_bit).__name__}, "
            f"size={getattr(image_bit, 'size', len(image_bit) if hasattr(image_bit, '__len__') else '?')})."
        )

    return image


def _decode_image_bit_sequence(image_bits):
    frames = []

    for index, image_bit in enumerate(image_bits):
        try:
            frames.append(decode_image_bit(image_bit))
        except ValueError as exc:
            raise ValueError(f"Frame {index}: {exc}") from exc

    if not frames:
        return np.zeros((0,), dtype=np.uint8)

    return np.stack(frames, axis=0)


def decode_image_bit(image_bits):
    """
    Decode encoded image bit stream(s) into uint8 RGB image array(s).

    The output is RGB. Treat that as settled and do not apply the usual
    "OpenCV returns BGR" rule: XPolicyLab trajectory files and runtime
    observations store buffers encoded from RGB arrays, and encode/decode
    round trips preserve the channel order they were given. Never add a
    COLOR_BGR2RGB after this function to "correct" the output — there is
    nothing to correct, and the swap is what breaks the channel order.

    Deliberately converting the RGB result to BGR is a different thing and is
    allowed where a checkpoint was trained on BGR data; that must be an opt-in
    documented in the adapter (see Dexora_1B's `input_color_order`), never a
    silent fix applied at the decode site.

    Values that are already decoded are returned unchanged, so this function is
    safe to call on an observation or trajectory field without knowing whether
    the producer encoded it.

    Dispatch is on dtype first, then ndim:
        - bytes / bytearray / memoryview / str  -> one encoded buffer
        - ndarray of dtype kind 'S', 'U', 'O'   -> sequence of encoded buffers,
                                                   or one buffer when 0-d
        - uint8 ndarray, ndim == 1              -> one encoded buffer
        - uint8 ndarray, ndim == 2              -> (T, N) stack of encoded buffers
        - uint8 ndarray, ndim >= 3              -> already decoded, returned as is
        - ndarray of any other dtype            -> already decoded, returned as is
        - list / tuple                          -> element-wise, stacked on axis 0

    Grayscale (H, W) uint8 images are not supported: a 2-D uint8 array is always
    read as a stack of encoded buffers.

    Invalid encoded buffers propagate ValueError from the decoder.
    """
    if isinstance(image_bits, (bytes, bytearray, memoryview, str)):
        return _decode_single_image_bit(image_bits)

    if isinstance(image_bits, np.ndarray):
        if image_bits.dtype.kind in {"S", "U", "O"}:
            if image_bits.ndim == 0:
                return _decode_single_image_bit(image_bits.item())
            return _decode_image_bit_sequence(image_bits)

        if image_bits.dtype == np.uint8:
            if image_bits.ndim == 1:
                return _decode_single_image_bit(image_bits)
            if image_bits.ndim == 2:
                return _decode_image_bit_sequence(image_bits)

        return image_bits

    if isinstance(image_bits, (list, tuple)):
        return _decode_image_bit_sequence(image_bits)

    return _decode_single_image_bit(image_bits)


def action_spec(task_env: Any) -> RoboDojoActionSpec:
    """Read the live joint limits and render the EEF embodiment instructions."""
    spec = _action_spec(task_env)
    return replace(spec, docs=eef_docs_from_joint_docs(spec.docs))


def native_planner(task_env: Any):
    """Bind CuRobo planning to the same live environment that owns execution."""
    manager = getattr(task_env, "robot_manager", None)
    if manager is None:
        raise InfrastructureFailure(
            "RoboDojo robot_manager is required for EEF planning"
        )

    def plan(*, arm: str, target_pose):
        robot = manager.get_robot_by_arm_name(f"{arm}_arm")
        planner = manager.planner.get(robot.robot_name)
        if planner is None:
            return {"status": "Unavailable"}
        env_idx = int(getattr(task_env, "env_idx", 0))
        current = manager.get_joint(robot, env_idx_list=[env_idx])[env_idx]
        return planner.plan_path(
            current,
            target_pose,
            real_robot_pose=copy.deepcopy(robot.entity_origin_pose),
        )

    return plan


@dataclass(frozen=True)
class EpisodeExecution:
    """Execution accounting; the enclosing native evaluator owns score/success."""

    termination_reason: str | None
    failure_kind: str | None
    llm_calls: int
    env_steps: int | None
    chunks: int


def run_episode(
    task_env: Any,
    policy: EefEpisodePolicy,
    *,
    output_dir: str | Path,
    native_trace_dir: str | Path | None = None,
) -> EpisodeExecution:
    """Execute native chunks and finish each deferred tool with its actual view.

    Give-up and budget exhaustion execute their one hold step before stopping.
    Capability errors stop without a hold. Infrastructure errors bypass failure
    scoring and escape the native layout loop via InfrastructureFailure.
    """
    from .eef_artifacts import NativeTrace, sanitize

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    execution_path = output / "execution.jsonl"
    if execution_path.exists() or (output / "episode.json").exists():
        raise FileExistsError("EEF execution output already exists")
    if policy.ctx.get("benchmark", task_env) is not task_env:
        raise ValueError("EEF episode is already bound to another benchmark")
    policy.ctx["benchmark"] = task_env
    policy_step = 0
    termination = None
    stop_detail = None
    failure_kind = None
    infrastructure = None
    error_message = None
    started = time.monotonic()
    trace = (
        NativeTrace(Path(native_trace_dir), task_env, policy)
        if native_trace_dir is not None
        else None
    )
    try:
        limit = policy.controller._env.get("L3_INSPECT_ENV_STEP_LIMIT", "").strip()
        if limit:
            try:
                override = int(limit)
            except ValueError as error:
                raise InfrastructureFailure(
                    "L3_INSPECT_ENV_STEP_LIMIT must be a positive int"
                ) from error
            if override <= 0:
                raise InfrastructureFailure(
                    "L3_INSPECT_ENV_STEP_LIMIT must be a positive int"
                )
            task_env.step_lim = override
        with execution_path.open("x", encoding="utf-8") as stream:
            while not task_env.is_episode_end():
                frame_start = video_frame_counts(task_env)
                observation = _observation(task_env, policy_step)
                if policy_step == 0:
                    policy.prepare(observation)
                if trace is not None:
                    trace.begin(
                        observation,
                        frame_ranges(
                            frame_start,
                            video_frame_counts(task_env),
                            include_frame=True,
                        ),
                    )
                chunk = policy.act(observation)
                policy_step += 1
                played = 0
                stop = False
                start = env_step(task_env)
                execution_frame_start = video_frame_counts(task_env)
                for action in chunk.actions:
                    policy.ctx["benchmark"].take_action(dict(action.data))
                    played += 1
                    # Native video capture happens in get_obs, including holds.
                    executed = _observation(task_env, policy_step)
                    if trace is not None:
                        trace.record_frame(executed)
                    if action.meta.get("request_stop"):
                        termination = str(action.meta.get("stop_reason"))
                        stop_detail = action.meta.get("stop_detail")
                        stop = True
                        break
                    if task_env.is_episode_end():
                        break
                if played == 0:
                    raise InfrastructureFailure("EEF returned an empty action chunk")
                policy.confirm_executed(played)
                native_done = bool(task_env.is_episode_end())
                policy.complete_post_action(executed, episode_done=native_done)
                stream.write(
                    json.dumps(
                        {
                            "policy_step": observation.step,
                            "env_step_start": start,
                            "env_step_end": env_step(task_env),
                            "planned_waypoints": len(chunk.actions),
                            "executed_waypoints": played,
                            "native_end": native_done,
                            "decision": chunk.meta.get("trace"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                stream.flush()
                if trace is not None:
                    trace.finish_turn(
                        chunk.meta.get("trace"),
                        {
                            "env_step_start": start,
                            "env_step_end": env_step(task_env),
                            "planned_waypoints": len(chunk.actions),
                            "executed_waypoints": played,
                            "stopped_early": played < len(chunk.actions),
                            "episode_end_mid_chunk": native_done
                            and played < len(chunk.actions),
                            "cameras": frame_ranges(
                                execution_frame_start, video_frame_counts(task_env)
                            ),
                        },
                    )
                if stop:
                    break
    except CapabilityFailure as error:
        failure_kind = "capability"
        termination = "policy_error"
        error_message = str(error)
    except InfrastructureFailure as error:
        infrastructure = error
    except Exception as error:
        infrastructure = InfrastructureFailure(
            f"unexpected infrastructure failure: {type(error).__name__}: {error}"
        )
    finally:
        if infrastructure is None:
            try:
                _mark_incomplete_episode_failed(task_env)
            except Exception as error:
                infrastructure = InfrastructureFailure(
                    f"native termination failed: {error}"
                )
        try:
            policy.close()
        except (Exception, InfrastructureFailure) as error:
            if infrastructure is None:
                infrastructure = InfrastructureFailure(
                    f"policy cleanup failed: {error}"
                )
            else:
                logger.warning("policy cleanup also failed: %s", type(error).__name__)
        if infrastructure is not None:
            failure_kind = "infrastructure"
            error_message = str(infrastructure)
        if trace is not None:
            try:
                trace.write(
                    termination_reason=termination,
                    failure_kind=failure_kind,
                    error_message=error_message,
                )
            except (Exception, InfrastructureFailure) as error:
                if infrastructure is None:
                    infrastructure = InfrastructureFailure(
                        f"native trace failed: {error}"
                    )
                    failure_kind = "infrastructure"
                    error_message = str(infrastructure)
                else:
                    logger.warning("native trace also failed: %s", type(error).__name__)
        try:
            observed = policy.ctx.get("observation")
            write_json_atomic(
                output / "episode.json",
                sanitize(
                    {
                        "task": getattr(task_env, "task_name", None),
                        "run_id": policy.controller._env.get("ROBODOJO_RUN_ID"),
                        "layout_id": policy.controller._layout_id,
                        "instruction": observed.instruction
                        if observed is not None
                        else None,
                        "termination_reason": termination,
                        "stop_detail": stop_detail,
                        "planner_stop_reason": policy.result.stop_reason
                        if policy.result is not None
                        else None,
                        "failure_kind": failure_kind,
                        "llm_calls": policy.calls,
                        "requests": policy.calls,
                        "env_steps": env_step(task_env),
                        "chunks": policy_step,
                        "config": policy.audit_config(),
                        "error": error_message,
                        "agent_error": error_message,
                        "agent_elapsed_s": time.monotonic() - started,
                        "model_artifacts": str(policy.agent.output_dir),
                    },
                    policy.controller._env,
                ),
            )
        except (Exception, InfrastructureFailure) as error:
            if infrastructure is None:
                infrastructure = InfrastructureFailure(
                    f"execution artifact failed: {error}"
                )
            else:
                logger.warning(
                    "execution artifact also failed: %s", type(error).__name__
                )
        finally:
            policy.ctx.pop("benchmark", None)
    if infrastructure is not None:
        raise infrastructure
    return EpisodeExecution(
        termination, failure_kind, policy.calls, env_step(task_env), policy_step
    )
