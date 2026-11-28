"""RoboDojo episode loop for AR and single-frame VAP inference."""

from __future__ import annotations

import hashlib
import math
import os
import uuid
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# This file is intentionally self-contained: RoboDojo's policy dropbox only
# synchronizes deploy.py and deploy.yml, not server-side helper modules.
ACTION_STEPS = 40
VIDEO_FRAMES = 20
CAMERA_NAMES = ("cam_head", "cam_left_wrist", "cam_right_wrist")
INFERENCE_MODE_AR = "ar"
INFERENCE_MODE_SINGLE_FRAME = "single_frame"
STATE_DIMS = {
    "left_ee_pose": 7,
    "left_ee_joint_state": 1,
    "right_ee_pose": 7,
    "right_ee_joint_state": 1,
}

LEGACY_PROFILE = "legacy_a40v20"
A32V16_320X256_PROFILE = "a32v16_320x256"
EE_V3_LFU_STATE_PROFILE = "a32v16_320x256_eev3_lfu_state"
EE_V3_LFU_STATE_MINMAX_PROFILE = "a32v16_320x256_eev3_lfu_state_minmax"
EE_V3_LFU_STATE_GLOBALNORM_PROFILE = "a32v16_320x256_eev3_lfu_state_globalnorm"
EE_V3_LFU_NOSTATE_PROFILE = "a32v16_320x256_eev3_lfu_nostate"
EE_V3_LFU_NOSTATE_GLOBALNORM_PROFILE = "a32v16_320x256_eev3_lfu_nostate_globalnorm"
EE_V3_LFU_NOSTATE_GLOBALNORM_FOLLOWER_PROFILE = (
    "a32v16_320x256_eev3_lfu_nostate_globalnorm_follower"
)
EE_V3_LFU_NOSTATE_GLOBALNORM_MEMORY_PROFILE = (
    "a32v16_320x256_eev3_lfu_nostate_globalnorm_memory"
)
EE_V3_LFU_NOSTATE_GLOBALNORM_A32V8_MEMORY_PROFILE = (
    "a32v8_320x256_eev3_lfu_nostate_globalnorm_memory"
)
EE_V3_LFU_NOSTATE_GLOBALNORM_A64V16_PROFILE = (
    "a64v16_320x256_eev3_lfu_nostate_globalnorm"
)


@dataclass(frozen=True)
class _EvaluationContract:
    name: str
    protocol_version: int
    action_steps: int
    video_frames: int
    image_height: int
    image_width: int

    @property
    def observation_interval(self) -> int:
        if self.action_steps % self.video_frames:
            raise ValueError(
                f"profile {self.name!r} has a non-integral action/video ratio"
            )
        return self.action_steps // self.video_frames


_EVALUATION_CONTRACTS = {
    LEGACY_PROFILE: _EvaluationContract(
        LEGACY_PROFILE, 2, ACTION_STEPS, VIDEO_FRAMES, 192, 224
    ),
    A32V16_320X256_PROFILE: _EvaluationContract(
        A32V16_320X256_PROFILE, 3, 32, 16, 256, 320
    ),
    EE_V3_LFU_STATE_PROFILE: _EvaluationContract(
        EE_V3_LFU_STATE_PROFILE, 4, 32, 16, 256, 320
    ),
    EE_V3_LFU_STATE_MINMAX_PROFILE: _EvaluationContract(
        EE_V3_LFU_STATE_MINMAX_PROFILE, 5, 32, 16, 256, 320
    ),
    EE_V3_LFU_STATE_GLOBALNORM_PROFILE: _EvaluationContract(
        EE_V3_LFU_STATE_GLOBALNORM_PROFILE, 6, 32, 16, 256, 320
    ),
    EE_V3_LFU_NOSTATE_PROFILE: _EvaluationContract(
        EE_V3_LFU_NOSTATE_PROFILE, 4, 32, 16, 256, 320
    ),
    EE_V3_LFU_NOSTATE_GLOBALNORM_PROFILE: _EvaluationContract(
        EE_V3_LFU_NOSTATE_GLOBALNORM_PROFILE, 6, 32, 16, 256, 320
    ),
    EE_V3_LFU_NOSTATE_GLOBALNORM_FOLLOWER_PROFILE: _EvaluationContract(
        EE_V3_LFU_NOSTATE_GLOBALNORM_FOLLOWER_PROFILE, 7, 32, 16, 256, 320
    ),
    EE_V3_LFU_NOSTATE_GLOBALNORM_MEMORY_PROFILE: _EvaluationContract(
        EE_V3_LFU_NOSTATE_GLOBALNORM_MEMORY_PROFILE, 6, 32, 16, 256, 320
    ),
    EE_V3_LFU_NOSTATE_GLOBALNORM_A64V16_PROFILE: _EvaluationContract(
        EE_V3_LFU_NOSTATE_GLOBALNORM_A64V16_PROFILE, 8, 64, 16, 256, 320
    ),
    EE_V3_LFU_NOSTATE_GLOBALNORM_A32V8_MEMORY_PROFILE: _EvaluationContract(
        EE_V3_LFU_NOSTATE_GLOBALNORM_A32V8_MEMORY_PROFILE, 9, 32, 8, 256, 320
    ),
}


def _evaluation_contract() -> _EvaluationContract:
    raw = os.environ.get("VAP_ROBODOJO_PROFILE", EE_V3_LFU_NOSTATE_GLOBALNORM_MEMORY_PROFILE)
    name = raw.strip().lower().replace("-", "_")
    name = {
        "legacy": LEGACY_PROFILE,
        "a40v20": LEGACY_PROFILE,
        "a40v20_192x224": LEGACY_PROFILE,
        "a32v16": A32V16_320X256_PROFILE,
        "ivava16a32_320x256": A32V16_320X256_PROFILE,
        "eev3": EE_V3_LFU_STATE_PROFILE,
        "ee_v3": EE_V3_LFU_STATE_PROFILE,
        "a32v16_eev3": EE_V3_LFU_STATE_PROFILE,
        "eev3_minmax": EE_V3_LFU_STATE_MINMAX_PROFILE,
        "ee_v3_minmax": EE_V3_LFU_STATE_MINMAX_PROFILE,
        "a32v16_eev3_minmax": EE_V3_LFU_STATE_MINMAX_PROFILE,
        "eev3_state_globalnorm": EE_V3_LFU_STATE_GLOBALNORM_PROFILE,
        "ee_v3_state_globalnorm": EE_V3_LFU_STATE_GLOBALNORM_PROFILE,
        "a32v16_eev3_state_globalnorm": EE_V3_LFU_STATE_GLOBALNORM_PROFILE,
        "eev3_nostate": EE_V3_LFU_NOSTATE_PROFILE,
        "ee_v3_nostate": EE_V3_LFU_NOSTATE_PROFILE,
        "a32v16_eev3_nostate": EE_V3_LFU_NOSTATE_PROFILE,
        "eev3_nostate_globalnorm": EE_V3_LFU_NOSTATE_GLOBALNORM_PROFILE,
        "ee_v3_nostate_globalnorm": EE_V3_LFU_NOSTATE_GLOBALNORM_PROFILE,
        "a32v16_eev3_nostate_globalnorm": EE_V3_LFU_NOSTATE_GLOBALNORM_PROFILE,
        "eev3_nostate_globalnorm_follower": (
            EE_V3_LFU_NOSTATE_GLOBALNORM_FOLLOWER_PROFILE
        ),
        "ee_v3_nostate_globalnorm_follower": (
            EE_V3_LFU_NOSTATE_GLOBALNORM_FOLLOWER_PROFILE
        ),
        "a32v16_eev3_nostate_globalnorm_follower": (
            EE_V3_LFU_NOSTATE_GLOBALNORM_FOLLOWER_PROFILE
        ),
        "eev3_nostate_globalnorm_memory": (EE_V3_LFU_NOSTATE_GLOBALNORM_MEMORY_PROFILE),
        "ee_v3_nostate_globalnorm_memory": (
            EE_V3_LFU_NOSTATE_GLOBALNORM_MEMORY_PROFILE
        ),
        "a32v16_eev3_nostate_globalnorm_memory": (
            EE_V3_LFU_NOSTATE_GLOBALNORM_MEMORY_PROFILE
        ),
        "a64v16_eev3_nostate_globalnorm": (EE_V3_LFU_NOSTATE_GLOBALNORM_A64V16_PROFILE),
        "a32v8_eev3_nostate_globalnorm_memory": (
            EE_V3_LFU_NOSTATE_GLOBALNORM_A32V8_MEMORY_PROFILE
        ),
        "eev3_nostate_globalnorm_a64v16": (EE_V3_LFU_NOSTATE_GLOBALNORM_A64V16_PROFILE),
    }.get(name, name)
    if name not in _EVALUATION_CONTRACTS:
        raise ValueError(
            f"unsupported VAP_ROBODOJO_PROFILE={raw!r}; "
            f"available={sorted(_EVALUATION_CONTRACTS)}"
        )
    return _EVALUATION_CONTRACTS[name]


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _active_layout_id(task_env: Any) -> int:
    """Read the active RoboDojo layout without mutating evaluation state."""

    if os.environ.get("EVAL_ENV_TYPE") == "debug":
        return int(getattr(task_env, "episode_num", 0))
    get_running = getattr(task_env, "get_running_env_idx_list", None)
    if not callable(get_running):
        raise AttributeError(
            "model-noise seeding requires TASK_ENV.get_running_env_idx_list()"
        )
    running = [int(value) for value in get_running()]
    if len(running) != 1:
        raise ValueError(
            "VAP's non-batched evaluator requires exactly one active environment, "
            f"got indices={running}"
        )
    env_idx = running[0]

    current_map = getattr(task_env, "current_env_seed_map", None)
    if isinstance(current_map, Mapping) and env_idx in current_map:
        return int(current_map[env_idx])

    env_seeds = getattr(task_env, "env_seeds", None)
    if env_seeds is None:
        raise AttributeError(
            "model-noise seeding requires TASK_ENV.env_seeds or "
            "TASK_ENV.current_env_seed_map"
        )
    try:
        return int(env_seeds[env_idx])
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError(
            f"cannot resolve layout id for active env index {env_idx}"
        ) from exc


def model_noise_identity(task_env: Any) -> str:
    """Build a stable identity used only by the model's flow-noise sampler.

    This function only reads RoboDojo's task/layout metadata. It does not set
    or otherwise modify the benchmark environment seed.
    """

    task_name = str(getattr(task_env, "task_name", getattr(task_env, "deploy_cfg", {}).get("task_name", ""))).strip()
    if not task_name:
        raise ValueError("model-noise seeding requires TASK_ENV.task_name")
    eval_seed = int(getattr(task_env, "eval_seed", 0))
    layout_id = _active_layout_id(task_env)
    key = f"vap-model-noise|{task_name}|{eval_seed}|{layout_id}".encode("utf-8")
    return hashlib.sha256(key).hexdigest()[:32]


def _inference_mode() -> str:
    raw = os.environ.get("VAP_INFERENCE_MODE", INFERENCE_MODE_AR)
    mode = raw.strip().lower().replace("-", "_")
    mode = {
        "autoregressive": INFERENCE_MODE_AR,
        "single": INFERENCE_MODE_SINGLE_FRAME,
        "singleframe": INFERENCE_MODE_SINGLE_FRAME,
    }.get(mode, mode)
    if mode not in {INFERENCE_MODE_AR, INFERENCE_MODE_SINGLE_FRAME}:
        raise ValueError(
            f"unsupported VAP_INFERENCE_MODE={raw!r}; "
            "expected 'ar' or 'single_frame'"
        )
    return mode


def _execute_actions_per_chunk(
    *, inference_mode: str, contract: _EvaluationContract
) -> int:
    raw = os.environ.get("VAP_EXECUTE_ACTIONS_PER_CHUNK", str(contract.action_steps))
    try:
        steps = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"VAP_EXECUTE_ACTIONS_PER_CHUNK={raw!r} must be an integer"
        ) from exc
    if not 1 <= steps <= contract.action_steps:
        raise ValueError(
            f"VAP_EXECUTE_ACTIONS_PER_CHUNK={steps} must be in "
            f"[1, {contract.action_steps}]"
        )
    if inference_mode == INFERENCE_MODE_AR and steps != contract.action_steps:
        raise ValueError(
            "AR inference requires VAP_EXECUTE_ACTIONS_PER_CHUNK="
            f"{contract.action_steps} for profile {contract.name!r} so the next "
            f"window receives exactly {contract.action_steps} executed states "
            f"and {contract.video_frames} video frames"
        )
    return steps


class _IKSuccessStats:
    """Count RoboDojo IK outcomes without changing the public environment."""

    def __init__(self, task_env: Any, *, episode_id: str):
        self.episode_id = episode_id
        self.enabled = _env_flag("VAP_IK_SUCCESS_STATS")
        self._manager = getattr(task_env, "robot_manager", None)
        self._original_solve_ik = None
        self._had_instance_override = False
        self._statuses: Counter[str] = Counter()
        self._arm_statuses: defaultdict[str, Counter[str]] = defaultdict(Counter)

    def install(self) -> None:
        if not self.enabled:
            return
        if self._manager is None or not callable(
            getattr(self._manager, "solve_ik", None)
        ):
            raise RuntimeError(
                "VAP_IK_SUCCESS_STATS=1 requires TASK_ENV.robot_manager.solve_ik"
            )

        self._had_instance_override = "solve_ik" in vars(self._manager)
        self._original_solve_ik = self._manager.solve_ik

        def wrapped_solve_ik(*args: Any, **kwargs: Any) -> Any:
            robot = kwargs.get("robot")
            if robot is None and len(args) >= 3:
                robot = args[2]
            arm = str(
                getattr(robot, "arm_name", getattr(robot, "robot_name", "unknown"))
            )
            try:
                result = self._original_solve_ik(*args, **kwargs)
            except Exception as exc:
                status = f"Exception:{type(exc).__name__}"
                self._statuses[status] += 1
                self._arm_statuses[arm][status] += 1
                raise
            status = (
                str(result.get("status", "MissingStatus"))
                if isinstance(result, Mapping)
                else f"InvalidResult:{type(result).__name__}"
            )
            self._statuses[status] += 1
            self._arm_statuses[arm][status] += 1
            return result

        self._manager.solve_ik = wrapped_solve_ik

    def restore(self) -> None:
        if self._original_solve_ik is None:
            return
        if self._had_instance_override:
            self._manager.solve_ik = self._original_solve_ik
        else:
            delattr(self._manager, "solve_ik")

    @staticmethod
    def _format_counts(counts: Counter[str]) -> str:
        total = sum(counts.values())
        success = counts.get("Success", 0)
        rate = success / total if total else 0.0
        statuses = ",".join(
            f"{status}:{count}" for status, count in sorted(counts.items())
        )
        return (
            f"total={total} success={success} failure={total - success} "
            f"rate={rate:.6f} statuses={statuses or 'none'}"
        )

    def report(self) -> None:
        if not self.enabled:
            return
        print(
            f"[VAP_IK_SUCCESS] episode={self.episode_id} "
            f"{self._format_counts(self._statuses)}",
            flush=True,
        )
        for arm in sorted(self._arm_statuses):
            print(
                f"[VAP_IK_SUCCESS_ARM] episode={self.episode_id} arm={arm} "
                f"{self._format_counts(self._arm_statuses[arm])}",
                flush=True,
            )


def _copy_raw_camera_frames(observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
    vision = observation.get("vision")
    if not isinstance(vision, Mapping):
        raise TypeError("observation['vision'] must be a mapping")
    result = {}
    for camera in CAMERA_NAMES:
        camera_data = vision.get(camera)
        if not isinstance(camera_data, Mapping) or "color" not in camera_data:
            raise KeyError(f"vision[{camera!r}]['color'] is required")
        from XPolicyLab.utils.process_data import decode_image_bit
        frame = np.asarray(decode_image_bit(camera_data["color"]))
        if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
            raise ValueError(
                f"vision[{camera!r}]['color'] must be uint8 [H,W,3], got "
                f"shape={frame.shape} dtype={frame.dtype}"
            )
        result[camera] = np.array(frame, copy=True, order="C")
    return result


def _stack_frame_history(
    frames: Mapping[str, list[np.ndarray]],
    *,
    contract: _EvaluationContract,
) -> dict[str, np.ndarray]:
    result = {}
    for camera in CAMERA_NAMES:
        camera_frames = frames.get(camera, [])
        if len(camera_frames) != contract.video_frames:
            raise ValueError(
                f"camera {camera!r} has {len(camera_frames)} frames; "
                f"expected {contract.video_frames}"
            )
        result[camera] = np.stack(camera_frames, axis=0)
    return result


def _copy_robot_state(observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
    state = observation.get("state")
    if not isinstance(state, Mapping):
        raise TypeError("observation['state'] must be a mapping")
    result = {}
    for key, dimension in STATE_DIMS.items():
        value = np.asarray(state.get(key), dtype=np.float64).reshape(-1)
        if value.shape != (dimension,) or not np.isfinite(value).all():
            raise ValueError(
                f"state[{key!r}] must be finite shape ({dimension},), got {value.shape}"
            )
        result[key] = np.array(value, copy=True)
    return result


def _stack_state_history(
    states: list[Mapping[str, np.ndarray]],
    *,
    contract: _EvaluationContract,
) -> dict[str, np.ndarray]:
    if len(states) != contract.action_steps:
        raise ValueError(
            f"executed state history has {len(states)} rows; "
            f"expected {contract.action_steps}"
        )
    return {
        key: np.stack([state[key] for state in states], axis=0) for key in STATE_DIMS
    }


def _policy_observation(
    observation: Mapping[str, Any],
    *,
    episode_id: str,
    model_noise_id: str,
    window_index: int,
    inference_mode: str,
    contract: _EvaluationContract,
    executed_video: Mapping[str, Any] | None,
    executed_state: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(observation, Mapping):
        raise TypeError("RoboDojo observation must be a mapping")
    payload = dict(observation)
    metadata: dict[str, Any] = {
        "protocol_version": contract.protocol_version,
        "episode_id": episode_id,
        "model_noise_id": model_noise_id,
        "window_index": window_index,
        "inference_mode": inference_mode,
    }
    if contract.protocol_version >= 3:
        metadata.update(
            {
                "evaluation_profile": contract.name,
                "action_steps": contract.action_steps,
                "video_frames": contract.video_frames,
                "image_height": contract.image_height,
                "image_width": contract.image_width,
            }
        )
    if executed_video is not None:
        metadata["executed_video"] = executed_video
    if executed_state is not None:
        metadata["executed_state"] = executed_state
    if (executed_video is None) != (executed_state is None):
        raise ValueError("executed_video and executed_state must be supplied together")
    payload["_vap"] = metadata
    return payload


def _validate_action_chunk(
    actions: Any, *, contract: _EvaluationContract
) -> list[Mapping[str, Any]]:
    if not isinstance(actions, list):
        raise TypeError(f"VAP policy must return a list, got {type(actions).__name__}")
    if len(actions) != contract.action_steps:
        raise ValueError(
            f"VAP policy returned {len(actions)} actions; "
            f"expected {contract.action_steps}"
        )
    if not all(isinstance(action, Mapping) for action in actions):
        raise TypeError("every VAP action must be a mapping")
    return actions


def eval_one_episode(TASK_ENV: Any, model_client: Any) -> None:
    """Run one non-batched episode with checkpoint-defined replanning windows."""

    episode_id = uuid.uuid4().hex
    noise_id = model_noise_identity(TASK_ENV)
    print(
        "[VAP_MODEL_NOISE] "
        f"task={str(getattr(TASK_ENV, 'task_name', getattr(TASK_ENV, 'deploy_cfg', {}).get('task_name', 'debug'))).strip()} "
        f"eval_seed={int(getattr(TASK_ENV, 'eval_seed', 0))} "
        f"layout_id={_active_layout_id(TASK_ENV)} identity={noise_id}",
        flush=True,
    )
    ik_stats = _IKSuccessStats(TASK_ENV, episode_id=episode_id)
    ik_stats.install()
    try:
        _eval_one_episode(
            TASK_ENV,
            model_client,
            episode_id=episode_id,
            model_noise_id=noise_id,
        )
    finally:
        ik_stats.restore()
        ik_stats.report()


def _configure_transport_timeouts(model_client: Any) -> None:
    """Apply VAP's wait budgets to the official already-connected WS client.

    The pinned RoboDojo client constructs WsModelClient without forwarding these
    deploy.yml fields. Update the public protocol config before this adapter's
    first RPC; do not wrap calls or retry stateful inference after a timeout.
    Debug clients and injected test transports without that config keep their
    existing contracts.
    """

    config = getattr(getattr(model_client, "_client", None), "config", None)
    if config is None:
        return
    deploy = yaml.safe_load(Path(__file__).with_name("deploy.yml").read_text())
    timeouts = {}
    for name in ("request_timeout_s", "max_connect_seconds"):
        value = deploy.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{name} must be a non-negative finite number, got {value!r}")
        timeouts[name] = value
    for name, value in timeouts.items():
        setattr(config, name, value)
    print(
        "[VAP_TRANSPORT] "
        f"request_timeout_s={config.request_timeout_s} "
        f"max_connect_seconds={config.max_connect_seconds}",
        flush=True,
    )


def _eval_one_episode(
    TASK_ENV: Any,
    model_client: Any,
    *,
    episode_id: str,
    model_noise_id: str,
) -> None:
    _configure_transport_timeouts(model_client)
    model_client.call(func_name="reset")
    contract = _evaluation_contract()
    inference_mode = _inference_mode()
    execute_actions = _execute_actions_per_chunk(
        inference_mode=inference_mode, contract=contract
    )
    print(
        f"[VAP_CLIENT_MODE] episode={episode_id} inference_mode={inference_mode} "
        f"evaluation_profile={contract.name}",
        flush=True,
    )
    print(
        f"[VAP_CLIENT_EXECUTION] episode={episode_id} "
        f"predicted_actions={contract.action_steps} "
        f"executed_actions={execute_actions} video_frames={contract.video_frames} "
        f"observation_interval={contract.observation_interval}",
        flush=True,
    )
    window_index = 0
    observation = TASK_ENV.get_obs()
    from XPolicyLab.policy.vap.recording import EpisodeRecorder
    recorder = EpisodeRecorder(TASK_ENV, episode_id)
    recorder.capture(observation, 0)
    total_steps = 0
    executed_video = None
    executed_state = None

    while not TASK_ENV.is_episode_end():
        payload = _policy_observation(
            observation,
            episode_id=episode_id,
            model_noise_id=model_noise_id,
            window_index=window_index,
            inference_mode=inference_mode,
            contract=contract,
            executed_video=executed_video,
            executed_state=executed_state,
        )
        model_client.call(func_name="update_obs", obs=payload)
        actions = _validate_action_chunk(
            model_client.call(func_name="get_action"), contract=contract
        )

        sampled_frames = {camera: [] for camera in CAMERA_NAMES}
        sampled_states: list[dict[str, np.ndarray]] = []
        completed_steps = 0
        latest_observation = observation
        for action_index, action in enumerate(actions[:execute_actions], start=1):
            TASK_ENV.take_action(action)
            total_steps += 1
            completed_steps = action_index
            if TASK_ENV.is_episode_end():
                break

            # RoboDojo records evaluation frames while observations are
            # collected. Keep this per-action cadence in both modes so a
            # single-frame policy still produces a complete 801-frame video.
            # In single-frame mode only the final observation becomes the next
            # I_t; none of these intermediate frames are sent to the policy.
            latest_observation = TASK_ENV.get_obs()
            recorder.capture(latest_observation, total_steps)
            if inference_mode == INFERENCE_MODE_AR:
                # Capture after every action. In particular, step 32 becomes
                # history frame 16 and the next current observation.
                sampled_states.append(_copy_robot_state(latest_observation))
                if action_index % contract.observation_interval == 0:
                    frames = _copy_raw_camera_frames(latest_observation)
                    for camera, frame in frames.items():
                        sampled_frames[camera].append(frame)

        if TASK_ENV.is_episode_end():
            break
        if completed_steps != execute_actions:
            raise RuntimeError(
                f"VAP window executed {completed_steps} steps without episode end; "
                f"expected {execute_actions}"
            )
        if inference_mode == INFERENCE_MODE_AR:
            executed_video = _stack_frame_history(sampled_frames, contract=contract)
            executed_state = _stack_state_history(sampled_states, contract=contract)
            if any(
                video.shape[0] != contract.video_frames
                for video in executed_video.values()
            ):
                raise RuntimeError(
                    "VAP history does not contain exactly "
                    f"{contract.video_frames} frames per camera"
                )
        else:
            executed_video = None
            executed_state = None
        observation = latest_observation
        window_index += 1

    recorder.finish(total_steps)


def eval_one_episode_batch(TASK_ENV: Any, model_client: Any) -> None:
    raise NotImplementedError(
        "VAP's stateful sliding-window protocol currently requires eval_batch=false"
    )


__all__ = ["eval_one_episode", "eval_one_episode_batch", "model_noise_identity"]
