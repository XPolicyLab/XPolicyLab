"""Native AgileX adapters, runner, and guarded command entrypoint."""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from evals.agilex.action_adapter import AgileXActionAdapter
from evals.agilex.safety import SafetyError, SafetyGuard
from evals.agilex.robot import (
    AgileXActionCommand,
    AgileXExecutionResult,
    AgileXReadRequest,
    AgileXRobot,
    HDF5ReplayRobot,
    LegacyAgileXRobotAdapter,
    ensure_agilex_robot,
)
from evals.agilex.observation_adapter import (
    AgileXObservationAdapter,
    _STATE_KEY_ORDER,
    _ensure_video,
    resize_image,
    sample_video,
)
from evals.common.checkpoint_runtime import (
    CheckpointTransformBundle,
)
from evals.common.evaluator import (
    EvaluationEpisodeResult,
    EvaluationResult,
)
from evals.common.schemas import (
    EpisodeLatencyMetrics,
    EpisodeRecord,
    StepLatencyMetrics,
    StepRecord,
)
from evals.common.suite import EpisodeSpec, EvaluationTask
from worldscape_policy.action_space import parse_action_mode
from worldscape_policy.embodiment import AGILEX, coalesce_embodiment, is_agilex_embodiment
from worldscape_policy.rollout.session import PolicyRuntime
from worldscape_policy.types import (
    InteractionMode,
    WorldActionOutput,
)

VisualPromptKind = Literal["text", "goal", "uniform"]
LOGGER = logging.getLogger(__name__)
AGILEX_PRETRAIN_ADAPTER_ROW = 2


@dataclass(frozen=True)
class AgileXVisualPromptConfig:
    """Strict visual-conditioning contract for migrated real-robot recipes."""

    kind: VisualPromptKind
    source: str = "none"
    path: str | None = None
    ctx_head_only: bool = True
    context_frames: int = 50
    goal_from_first_observation: bool = False
    poll_transport: bool = False

    @classmethod
    def from_config(cls, value: Any) -> AgileXVisualPromptConfig:
        if not isinstance(value, Mapping):
            raise TypeError("visual_prompt must be a mapping")
        allowed = {
            "kind",
            "source",
            "path",
            "ctx_head_only",
            "context_frames",
            "goal_from_first_observation",
            "poll_transport",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"visual_prompt has unknown fields: {sorted(unknown)}")
        kind = str(value.get("kind", ""))
        if kind not in {"text", "goal", "uniform"}:
            raise ValueError("visual_prompt.kind must be text, goal, or uniform")
        source = str(value.get("source", "none"))
        path = value.get("path")
        if path is not None and not isinstance(path, str):
            raise TypeError("visual_prompt.path must be a string")
        result = cls(
            kind=kind,  # type: ignore[arg-type]
            source=source,
            path=path,
            ctx_head_only=bool(value.get("ctx_head_only", True)),
            context_frames=int(value.get("context_frames", 50)),
            goal_from_first_observation=bool(
                value.get("goal_from_first_observation", False)
            ),
            poll_transport=bool(value.get("poll_transport", False)),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.kind == "text":
            if (
                self.source != "none"
                or self.path is not None
                or self.goal_from_first_observation
                or self.poll_transport
            ):
                raise ValueError("text visual_prompt must not configure visual input")
            return
        if self.kind == "goal":
            allowed = {"path", "hdf5", "upload", "first_observation"}
            if self.source not in allowed:
                raise ValueError(f"goal visual_prompt source must be one of {sorted(allowed)}")
            if self.source == "first_observation" and not self.goal_from_first_observation:
                raise ValueError(
                    "first-observation goals require the explicit "
                    "goal_from_first_observation=true opt-in"
                )
            if self.goal_from_first_observation and self.source != "first_observation":
                raise ValueError(
                    "goal_from_first_observation is valid only with source=first_observation"
                )
            if self.source == "path" and not self.path:
                raise ValueError("goal path source requires visual_prompt.path")
            if self.context_frames != 1:
                raise ValueError("goal recipes require context_frames=1")
            if not self.ctx_head_only:
                raise ValueError("goal recipes require ctx_head_only=true (V=1)")
            if self.poll_transport:
                raise ValueError("goal recipes do not support transport polling")
            return
        if self.source not in {"path", "hdf5", "upload", "transport"}:
            raise ValueError(
                "uniform visual_prompt.source must be path, hdf5, upload, or transport"
            )
        if self.source == "path" and not self.path:
            raise ValueError("uniform path source requires visual_prompt.path")
        if self.context_frames <= 0:
            raise ValueError("uniform recipes require positive context_frames")
        if self.goal_from_first_observation:
            raise ValueError("uniform cannot use goal_from_first_observation")
        if self.poll_transport != (self.source == "transport"):
            raise ValueError(
                "uniform transport source requires poll_transport=true, and other "
                "sources require poll_transport=false"
            )


class RobotResetRequiredError(RuntimeError):
    """Raised when a failed candidate physically executed a partial prefix."""

    def __init__(self, accepted_segments: int, cause: Exception) -> None:
        self.accepted_segments = accepted_segments
        self.cause = cause
        super().__init__(
            f"Robot reset required after {accepted_segments} accepted segment(s): "
            f"{cause}"
        )


TransportExecutionResult = AgileXExecutionResult


@dataclass(frozen=True)
class RealRobotRunnerConfig:
    mode: InteractionMode | str
    action_rate: int = 24
    action_mode: str = "eef"
    max_steps: int = 0
    rollout_steps: int = 8
    refresh_horizon: int = 0
    use_history: bool = False
    num_history_frames: int = 4
    observation_timeout_s: float = 20.0
    observation_timeout_policy: Literal["raise", "retry"] = "raise"
    send_timeout_s: float | None = None
    dry_run: bool = False
    goal_from_first_observation: bool = False
    context_poll: bool = False
    context_frames: int = 50
    ctx_head_only: bool = True
    context_poll_timeout_s: float = 30.0
    episode_id: str = "agilex-0000"
    task_id: str = "agilex"
    instruction: str = "AgileX real-robot rollout"
    seed: int = 0
    capture_frames: bool = False
    max_recorded_frames: int = 256

    def __post_init__(self) -> None:
        if self.max_recorded_frames < 1:
            raise ValueError("max_recorded_frames must be positive")
        if self.context_poll and self.context_frames != 50:
            raise ValueError("Transport context polling requires exactly 50 frames")


@dataclass(frozen=True)
class ActionPreview:
    """Structured pre-execution trajectory preview."""

    episode_id: str
    step_index: int
    action_mode: str
    dry_run: bool
    left: tuple[tuple[float, ...], ...]
    right: tuple[tuple[float, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": "agilex_action_preview",
            "schema_version": 1,
            **asdict(self),
        }


class RealRobotRunner:
    """Segmented native rollout with one transaction per complete action."""

    def __init__(
        self,
        runtime: PolicyRuntime,
        robot: AgileXRobot | Any,
        observation_adapter: AgileXObservationAdapter,
        action_adapter: AgileXActionAdapter,
        *,
        safety_guard: SafetyGuard | None = None,
        artifact_writer: Any | None = None,
        action_preview: Callable[[ActionPreview], None] | None = None,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.runtime = runtime
        self.robot = ensure_agilex_robot(robot)
        self.observation_adapter = observation_adapter
        self.action_adapter = action_adapter
        self.safety_guard = safety_guard or SafetyGuard()
        self.artifact_writer = artifact_writer
        self.action_preview = action_preview
        self.clock = clock
        self.sleep = sleep
        self.last_result: EvaluationResult | None = None
        self.last_previews: tuple[ActionPreview, ...] = ()

    def run(
        self,
        config: RealRobotRunnerConfig,
        *,
        generator: torch.Generator,
    ) -> list[WorldActionOutput]:
        if config.rollout_steps < 1:
            raise ValueError("rollout_steps must be positive")
        mode = InteractionMode.parse(config.mode)
        outputs: list[WorldActionOutput] = []
        records: list[StepRecord] = []
        frames: list[np.ndarray] = []
        previews: list[ActionPreview] = []
        episode_started = self.clock()
        step = 0
        accepted_segments = 0
        context_session = 0
        raw: tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], Any] | None = None
        error: Exception | None = None
        try:
            self.runtime.reset(mode.value)
            self.robot.reset_episode()
            self.observation_adapter.reset_episode()
            raw = self._read(config, step=step)
            self._record_frame(config, frames, raw)
            if config.goal_from_first_observation:
                self.observation_adapter.reset_episode(
                    goal_image=_ensure_video(raw[0])[-1]
                )
            while config.max_steps <= 0 or step < config.max_steps:
                context = self._poll_context(
                    config,
                    wait=step == 0,
                )
                if context is not None:
                    context_session += 1
                    LOGGER.info(
                        "[WSP eval] Activating context video session %d",
                        context_session,
                    )
                    demo = context[0] if config.ctx_head_only else context
                    self.runtime.reset(mode.value)
                    self.observation_adapter.reset_episode(
                        goal_image=None,
                        demo_video=demo,
                        session_id=f"{config.episode_id}-ctx-{context_session}",
                    )
                    raw = None
                    LOGGER.info(
                        "[WSP eval] Reading fresh observation/state for context "
                        "session %d",
                        context_session,
                    )
                    raw = self._read(config, step=step)
                    self._record_frame(config, frames, raw)
                step_started = self.clock()
                adapted = self.observation_adapter(raw)
                observation_finished = self.clock()
                output: WorldActionOutput | None = None
                accepted_segments = 0
                output = self.runtime.predict(
                    observation=adapted.observation,
                    prompts=adapted.prompts,
                    generator=generator,
                )
                prediction_finished = self.clock()
                if isinstance(self.action_adapter, AgileXActionAdapter):
                    left, right = self.action_adapter(
                        output, current_state=adapted.raw_state
                    )
                else:
                    left, right = self.action_adapter(output)
                horizon = len(left) if config.refresh_horizon <= 0 else min(
                    len(left), config.refresh_horizon
                )
                left, right = left[:horizon], right[:horizon]
                try:
                    safety_warnings = self.safety_guard.validate(
                        left,
                        right,
                        action_mode=config.action_mode,
                        current_state=adapted.raw_state,
                        enforce_motion_limits=False,
                    )
                except SafetyError as safety_error:
                    # Structurally invalid/non-finite commands cannot be sent to
                    # the transport. Motion-limit violations are warning-only.
                    LOGGER.warning(
                        "Invalid AgileX prediction skipped episode=%s step=%d: %s",
                        config.episode_id,
                        step,
                        safety_error,
                    )
                    if self.runtime.has_pending_prediction:
                        self.runtime.discard()
                    rejected = self.clock()
                    records.append(
                        StepRecord(
                            episode_id=config.episode_id,
                            step_index=step,
                            status="rejected",
                            latency=_step_latency(
                                step_started,
                                observation_finished,
                                prediction_finished,
                                rejected,
                                rejected,
                            ),
                            error_type=type(safety_error).__name__,
                            error_message=str(safety_error),
                        )
                    )
                    step += 1
                    if config.max_steps <= 0 or step < config.max_steps:
                        raw = self._read(config, step=step)
                        self._record_frame(config, frames, raw)
                    continue
                for safety_warning in safety_warnings:
                    LOGGER.warning(
                        "AgileX motion safety warning; command will still be sent "
                        "episode=%s step=%d: %s",
                        config.episode_id,
                        step,
                        safety_warning,
                    )
                preview = ActionPreview(
                    episode_id=config.episode_id,
                    step_index=step,
                    action_mode=config.action_mode,
                    dry_run=config.dry_run,
                    left=tuple(tuple(float(value) for value in row) for row in left),
                    right=tuple(tuple(float(value) for value in row) for row in right),
                )
                previews.append(preview)
                if self.action_preview is not None:
                    self.action_preview(preview)
                LOGGER.info(
                    "AgileX action preview episode=%s step=%d horizon=%d dry_run=%s",
                    config.episode_id,
                    step,
                    horizon,
                    config.dry_run,
                )
                segments = _segment_bounds(horizon, config.rollout_steps)
                # Preserve the complete executed GT observation chunk for the
                # next causal prefill. This is independent of use_history,
                # which only controls transport-side sampling before a plan.
                next_frames = [
                    [_ensure_video(camera)[-1]] for camera in raw[:3]
                ]
                segment_timestamp = adapted.timestamp
                for sub, (start, end) in enumerate(segments, start=1):
                    execution = (
                        AgileXExecutionResult(
                            accepted=True, detail="dry-run: command not sent"
                        )
                        if config.dry_run
                        else self._send(
                            segment_timestamp,
                            config,
                            left[start:end],
                            right[start:end],
                        )
                    )
                    if execution.accepted:
                        accepted_segments += 1
                    sampled = self._read(config, step=step, sub=sub)
                    for camera_frames, camera in zip(next_frames, sampled[:3]):
                        camera_frames.append(_ensure_video(camera)[-1])
                    self._record_frame(config, frames, sampled)
                    segment_timestamp = sampled[4]
                raw = (
                    np.stack(next_frames[0]),
                    np.stack(next_frames[1]),
                    np.stack(next_frames[2]),
                    sampled[3],
                    segment_timestamp,
                )
                execution_finished = self.clock()
                self.runtime.commit(output)
                step_finished = self.clock()
                # Keep only a detached CPU result. The committed runtime owns
                # the live KV cache; retaining each candidate output would
                # otherwise keep every historical GPU cache alive.
                outputs.append(output.result_snapshot())
                records.append(
                    StepRecord(
                        episode_id=config.episode_id,
                        step_index=step,
                        status="completed",
                        latency=_step_latency(
                            step_started,
                            observation_finished,
                            prediction_finished,
                            execution_finished,
                            step_finished,
                        ),
                    )
                )
                step += 1
        except Exception as exc:
            if self.runtime.has_pending_prediction:
                self.runtime.discard()
            error = (
                RobotResetRequiredError(accepted_segments, exc)
                if accepted_segments and not config.dry_run
                else exc
            )
            LOGGER.exception(
                "AgileX rollout failed episode=%s step=%d accepted_segments=%d",
                config.episode_id,
                step,
                accepted_segments,
            )
            failed = self.clock()
            records.append(
                StepRecord(
                    episode_id=config.episode_id,
                    step_index=step,
                    status="timed_out" if isinstance(exc, TimeoutError) else "failed",
                    latency=StepLatencyMetrics(
                        observation_ms=0.0,
                        prediction_ms=0.0,
                        execution_ms=0.0,
                        total_ms=max(0.0, (failed - episode_started) * 1000.0),
                    ),
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
            )
        finished = self.clock()
        self.last_previews = tuple(previews)
        self.last_result = self._result(
            config,
            mode,
            outputs,
            records,
            frames,
            error,
            episode_ms=max(0.0, (finished - episode_started) * 1000.0),
        )
        self._write_artifacts(config, self.last_result, previews)
        if error is not None:
            raise error from (error.cause if isinstance(error, RobotResetRequiredError) else None)
        return outputs

    def _poll_context(
        self,
        config: RealRobotRunnerConfig,
        *,
        wait: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if not config.context_poll:
            return None
        if wait:
            LOGGER.info("[WSP eval] Waiting for initial context video upload")
        poll = getattr(self.robot, "try_read_context_video", None)
        if not callable(poll):
            raise TypeError(
                "Configured context polling requires try_read_context_video()"
            )
        started = self.clock()
        while True:
            value = poll(config.context_frames)
            if not isinstance(value, tuple) or len(value) != 2:
                raise TypeError(
                    "try_read_context_video() must return (context, is_complete)"
                )
            context, complete = value
            if complete:
                if context is None:
                    raise RuntimeError("Context upload completed without video")
                return _normalize_context(context, frames=config.context_frames)
            if not wait:
                return None
            if self.clock() - started > config.context_poll_timeout_s:
                raise TimeoutError("Initial context video upload timed out")
            self.sleep(0.05)

    @staticmethod
    def _record_frame(
        config: RealRobotRunnerConfig,
        frames: list[np.ndarray],
        raw: tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], Any],
    ) -> None:
        if not config.capture_frames or len(frames) >= config.max_recorded_frames:
            return
        # Artifacts use the primary (head) camera. Copy the latest image because
        # live transports may reuse their receive buffer on the next sample.
        frames.append(np.array(_ensure_video(raw[0])[-1], copy=True))

    def _result(
        self,
        config: RealRobotRunnerConfig,
        mode: InteractionMode,
        outputs: list[WorldActionOutput],
        records: list[StepRecord],
        frames: list[np.ndarray],
        error: Exception | None,
        *,
        episode_ms: float,
    ) -> EvaluationResult:
        status = (
            "timed_out"
            if isinstance(error, TimeoutError)
            else "failed"
            if error is not None
            else "completed"
        )
        record = EpisodeRecord(
            episode_id=config.episode_id,
            mode=mode.value,
            status=status,
            requested_steps=config.max_steps,
            completed_steps=len(outputs),
            latency=EpisodeLatencyMetrics.from_steps(
                (item.latency for item in records), episode_ms=episode_ms
            ),
            steps=tuple(records),
            error_type=type(error).__name__ if error is not None else None,
            error_message=str(error) if error is not None else None,
            task_id=config.task_id,
            success=None if error is None else False,
            seed=config.seed,
        )
        spec = EpisodeSpec(
            EvaluationTask(config.task_id, config.instruction),
            episode_index=0,
            seed=config.seed,
        )
        return EvaluationResult(
            (
                EvaluationEpisodeResult(
                    spec=spec,
                    outputs=tuple(outputs),
                    record=record,
                    frames=tuple(frames),
                    error=error,
                ),
            )
        )

    def _write_artifacts(
        self,
        config: RealRobotRunnerConfig,
        result: EvaluationResult,
        previews: list[ActionPreview],
    ) -> None:
        if self.artifact_writer is None:
            return
        self.artifact_writer.write(
            {
                **asdict(config),
                "backend": "agilex",
                "visual_input_range": self.observation_adapter.visual_input_range,
                "session_id": self.observation_adapter.session_id,
            },
            result,
        )
        output_dir = Path(self.artifact_writer.output_dir)
        with (output_dir / "action_previews.jsonl").open(
            "w", encoding="utf-8"
        ) as stream:
            for preview in previews:
                stream.write(json.dumps(preview.to_dict(), sort_keys=True) + "\n")

    def _read(
        self,
        config: RealRobotRunnerConfig,
        *,
        step: int,
        sub: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], Any]:
        del step, sub
        started = self.clock()
        request = AgileXReadRequest(
            use_history=config.use_history,
            num_history_frames=config.num_history_frames,
            action_mode=config.action_mode,
        )
        while True:
            value = self.robot.try_observe(request)
            if value is not None:
                return value.as_tuple()
            if self.clock() - started > config.observation_timeout_s:
                if config.observation_timeout_policy == "raise":
                    raise TimeoutError("Robot observation timed out")
                if config.observation_timeout_policy != "retry":
                    raise ValueError(
                        "observation_timeout_policy must be 'raise' or 'retry'"
                    )
                logging.warning(
                    "Robot observation timed out after %.1fs; continuing to wait",
                    config.observation_timeout_s,
                )
                started = self.clock()
            self.sleep(0.05)

    def _send(
        self,
        timestamp: Any,
        config: RealRobotRunnerConfig,
        left: list[list[float]],
        right: list[list[float]],
    ) -> AgileXExecutionResult:
        result = self.robot.execute(
            AgileXActionCommand(
                timestamp=timestamp,
                rate=config.action_rate,
                left=left,
                right=right,
                action_mode=config.action_mode,
            ),
            timeout_s=config.send_timeout_s,
        )
        if result.accepted:
            return result
        if result.timed_out:
            raise TimeoutError(result.detail or "Robot action send timed out")
        raise RuntimeError(result.detail or "Robot action send was rejected")


def reject_distributed_native_runtime() -> None:
    """Native real-robot ownership is intentionally single-process."""

    import os

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
    if world_size > 1:
        raise RuntimeError("Native AgileX runtime requires world_size=1")


def _step_latency(
    started: float,
    observation_finished: float,
    prediction_finished: float,
    execution_finished: float,
    finished: float,
) -> StepLatencyMetrics:
    def milliseconds(value: float) -> float:
        return max(0.0, value * 1000.0)

    return StepLatencyMetrics(
        observation_ms=milliseconds(observation_finished - started),
        prediction_ms=milliseconds(prediction_finished - observation_finished),
        execution_ms=milliseconds(execution_finished - prediction_finished),
        total_ms=milliseconds(finished - started),
    )


def _segment_bounds(horizon: int, count: int) -> list[tuple[int, int]]:
    if horizon < 1:
        raise ValueError("Action horizon must be positive")
    count = min(horizon, count)
    boundaries = np.linspace(0, horizon, count + 1, dtype=np.int64)
    return [
        (int(boundaries[index]), int(boundaries[index + 1]))
        for index in range(count)
    ]


_EEF_STATE_FIELDS = _STATE_KEY_ORDER[:6]
_EEF_ACTION_FIELDS = tuple(key.replace("state.", "action.") for key in _EEF_STATE_FIELDS)


def validate_agilex_transform_bundle(
    bundle: CheckpointTransformBundle,
    *,
    action_mode: str,
    visual_input_range: str | None = None,
) -> None:
    """Reject a checkpoint whose embodiment or ordered fields differ."""

    if not is_agilex_embodiment(bundle.embodiment):
        raise ValueError(
            f"Migrated AgileX recipes require embodiment={AGILEX!r}, "
            f"got {bundle.embodiment!r}"
        )
    if action_mode != "eef":
        raise ValueError("Migrated AgileX recipes require action_mode='eef'")
    embodiment = getattr(bundle.transform, "embodiment", None)
    if embodiment is None:
        raise ValueError("AgileX evaluation requires a native transform bundle")
    state = tuple(field.key for field in embodiment.state_fields)
    action = tuple(field.key for field in embodiment.action_fields)
    if state != _EEF_STATE_FIELDS:
        raise ValueError(
            f"Transform state field order mismatch: expected {_EEF_STATE_FIELDS}, got {state}"
        )
    if action != _EEF_ACTION_FIELDS:
        raise ValueError(
            f"Transform action field order mismatch: expected {_EEF_ACTION_FIELDS}, got {action}"
        )
    expected_sizes = (3, 6, 1, 3, 6, 1)
    state_sizes = tuple(field.size for field in embodiment.state_fields)
    action_sizes = tuple(field.size for field in embodiment.action_fields)
    if state_sizes != expected_sizes or action_sizes != expected_sizes:
        raise ValueError(
            "Transform EEF field widths mismatch: "
            f"state={state_sizes}, action={action_sizes}, expected={expected_sizes}"
        )
    bundle_range = getattr(bundle.transform, "image_input_range", None)
    if visual_input_range is not None and bundle_range != visual_input_range:
        raise ValueError(
            "Evaluation visual_input_range does not match transform bundle: "
            f"{visual_input_range!r} != {bundle_range!r}"
        )


def _single_adapter_model_configs(
    checkpoint: str | Path,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Build a one-row eval model so AgileX selects row 2 into local row 0."""

    directory = Path(checkpoint)
    model_config_path = directory / "model_config.yaml"
    generation_config_path = directory / "generation_config.yaml"
    if not model_config_path.is_file() or not generation_config_path.is_file():
        return None, None

    from omegaconf import OmegaConf

    model_config = OmegaConf.to_container(
        OmegaConf.load(model_config_path),
        resolve=True,
    )
    generation_config = OmegaConf.to_container(
        OmegaConf.load(generation_config_path),
        resolve=True,
    )
    if not isinstance(model_config, dict) or not isinstance(generation_config, dict):
        raise TypeError("Native checkpoint model/generation configs must be mappings")
    core = model_config["model"]["wam"]["core"]["parameters"]
    if not isinstance(core, dict):
        raise TypeError("model.wam.core.parameters must be a mapping")
    core["max_num_embodiments"] = 1
    return model_config, generation_config


def _normalize_context(
    value: Any,
    *,
    frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError("Context video must contain cameras in high,left,right order")
    result = []
    for camera in value:
        video = _ensure_video(np.asarray(camera))
        result.append(sample_video(list(video), frames))
    return tuple(result)  # type: ignore[return-value]


def _load_visual_path(path: str | Path) -> np.ndarray | tuple[np.ndarray, ...]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Visual prompt path does not exist: {source}")
    if source.is_dir():
        files = sorted(
            item
            for item in source.iterdir()
            if item.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        )
        if not files:
            raise ValueError(f"Visual prompt directory has no supported images: {source}")
        import imageio.v3 as iio

        return np.stack([np.asarray(iio.imread(item))[:, :, :3] for item in files])
    if source.suffix.lower() == ".npy":
        return np.load(source, allow_pickle=False)
    if source.suffix.lower() == ".npz":
        with np.load(source, allow_pickle=False) as archive:
            keys = tuple(archive.files)
            if keys == ("high", "left", "right"):
                return tuple(np.asarray(archive[key]) for key in keys)
            if keys == ("frames",):
                return np.asarray(archive["frames"])
            raise ValueError("Visual NPZ must contain frames or high,left,right arrays")
    import imageio.v3 as iio

    return np.asarray(iio.imread(source))


def _prepare_visual_prompt(
    prompt: AgileXVisualPromptConfig,
    robot: AgileXRobot,
    *,
    uploaded_visual_prompt: Any | None,
) -> tuple[np.ndarray | None, np.ndarray | tuple[np.ndarray, ...] | None]:
    if prompt.kind == "text" or prompt.source == "first_observation":
        if uploaded_visual_prompt is not None:
            raise ValueError("This recipe does not accept an uploaded visual prompt")
        return None, None
    if prompt.source == "upload":
        if uploaded_visual_prompt is None:
            raise ValueError("visual_prompt.source=upload requires uploaded_visual_prompt")
        source = uploaded_visual_prompt
    elif prompt.source == "path":
        source = _load_visual_path(prompt.path or "")
    elif prompt.source == "hdf5":
        source = robot.read_context_video(prompt.context_frames)  # type: ignore[attr-defined]
        if source is None:
            raise RuntimeError("HDF5 transport did not provide visual context")
    else:
        if uploaded_visual_prompt is not None:
            raise ValueError("Transport prompt recipes do not accept external uploads")
        return None, None
    if prompt.kind == "goal":
        if isinstance(source, (tuple, list)):
            source = source[0]
        video = _ensure_video(np.asarray(source))
        return resize_image(video[-1]), None
    if isinstance(source, (tuple, list)):
        context = _normalize_context(source, frames=prompt.context_frames)
    else:
        if not prompt.ctx_head_only:
            raise ValueError(
                "ctx_head_only=false requires explicit high,left,right demo streams"
            )
        video = _ensure_video(np.asarray(source))
        context = (
            sample_video(list(video), prompt.context_frames),
            sample_video(list(video), prompt.context_frames),
            sample_video(list(video), prompt.context_frames),
        )
    return None, context[0] if prompt.ctx_head_only else context


def run_agilex_recipe(
    config: Mapping[str, Any],
    *,
    checkpoint: str | Path,
    output_dir: str | Path,
    live_hardware: bool = False,
    uploaded_visual_prompt: Any | None = None,
) -> dict[str, Any]:
    """Run one guarded native AgileX episode from a common evaluation recipe."""

    LOGGER.info(
        "[WSP eval] Starting AgileX evaluation: checkpoint=%s, live_hardware=%s",
        checkpoint,
        live_hardware,
    )
    reject_distributed_native_runtime()
    backend_config = config.get("backend_config", {})
    if not isinstance(backend_config, Mapping):
        raise TypeError("backend_config must be a mapping")
    configured_dry_run = bool(config.get("dry_run", True))
    if not configured_dry_run and not live_hardware:
        raise ValueError(
            "AgileX evaluation is dry-run by default; pass --live-hardware "
            "to authorize physical command execution"
        )
    transport = str(backend_config.get("transport", "hdf5"))
    LOGGER.info(
        "[WSP eval] Initializing %s transport (host=%s, port=%s)",
        transport,
        backend_config.get("host", "0.0.0.0"),
        backend_config.get("port", 8887),
    )
    if live_hardware and transport != "manifold":
        raise ValueError("--live-hardware requires backend_config.transport=manifold")
    if transport == "hdf5":
        path = backend_config.get("path")
        if not path:
            raise ValueError("AgileX HDF5 transport requires backend_config.path")
        robot: AgileXRobot = HDF5ReplayRobot(
            path,
            video_columns=backend_config.get("video_columns"),
        )
    elif transport == "manifold":
        if not live_hardware:
            raise ValueError(
                "Manifold transport requires explicit --live-hardware opt-in"
            )
        from evals.agilex.server import NativeManifoldRobot

        robot = LegacyAgileXRobotAdapter(
            NativeManifoldRobot(
                host=str(backend_config.get("host", "0.0.0.0")),
                port=int(backend_config.get("port", 8887)),
                node_name=str(backend_config.get("node_name", "wsp")),
                model_width=int(backend_config.get("model_width", 640)),
                model_height=int(backend_config.get("model_height", 480)),
                use_wma_layout=bool(
                    backend_config.get("use_wma_layout", False)
                ),
            )
        )
    else:
        raise ValueError(f"Unknown AgileX transport: {transport!r}")

    from evals.common.artifacts import EvaluationArtifactWriter
    from evals.common.checkpoint_runtime import (
        load_checkpoint_transform_bundle,
        without_state_action_normalization,
    )
    from evals.common.suite import TaskSuite
    from worldscape_policy.native_builder import (
        build_wan22_policy_from_checkpoint,
        checkpoint_mode,
        checkpoint_supports_mode,
    )

    # Full artifact validation is performed once by the policy builder.
    LOGGER.info("[WSP eval] Resolving checkpoint mode and task configuration")
    checkpoint_primary_mode = checkpoint_mode(
        checkpoint, validate_artifacts=False
    )
    configured_mode = config.get("mode")
    mode = InteractionMode.parse(
        configured_mode
        if configured_mode is not None
        else checkpoint_primary_mode
    )
    if not checkpoint_supports_mode(checkpoint_primary_mode, mode):
        raise ValueError(
            f"checkpoint mode is {checkpoint_primary_mode.value!r}, "
            f"not requested {mode.value!r}"
        )
    suite = TaskSuite.from_config(config)
    episodes = list(suite.episodes())
    if len(episodes) != 1:
        raise ValueError("AgileX recipes must resolve to exactly one episode")
    spec = episodes[0]
    adapter_config = backend_config.get("adapter", {})
    if not isinstance(adapter_config, Mapping):
        raise TypeError("backend_config.adapter must be a mapping")
    visual_prompt = AgileXVisualPromptConfig.from_config(config.get("visual_prompt"))
    embodiment = coalesce_embodiment(adapter_config, default=AGILEX)
    device = torch.device(str(config.get("device", "cuda")))
    visual_input_range = str(config.get("visual_input_range", "zero_one"))
    LOGGER.info("[WSP eval] Loading checkpoint observation/action transforms")
    bundle = load_checkpoint_transform_bundle(
        checkpoint,
        embodiment,
    )
    state_action_normalization = str(
        backend_config.get("state_action_normalization", "identity")
    )
    if state_action_normalization == "identity":
        bundle = without_state_action_normalization(bundle)
    elif state_action_normalization != "checkpoint":
        raise ValueError(
            "backend_config.state_action_normalization must be identity or checkpoint"
        )
    action_mode = parse_action_mode(str(backend_config.get("action_mode", "eef")))
    validate_agilex_transform_bundle(
        bundle,
        action_mode=action_mode,
        visual_input_range=visual_input_range,
    )
    LOGGER.info("[WSP eval] Preparing %s visual prompt", visual_prompt.kind)
    goal_image, demo_video = _prepare_visual_prompt(
        visual_prompt,
        robot,
        uploaded_visual_prompt=uploaded_visual_prompt,
    )
    LOGGER.info("[WSP eval] Building policy on device: %s", device)
    model_config, generation_config = _single_adapter_model_configs(checkpoint)
    pretrained_action_adapter_index = int(
        backend_config.get(
            "pretrained_action_adapter_index",
            AGILEX_PRETRAIN_ADAPTER_ROW,
        )
    )
    LOGGER.info(
        "[WSP eval] Loading AgileX adapter row %d into local row 0",
        pretrained_action_adapter_index,
    )
    policy = build_wan22_policy_from_checkpoint(
        checkpoint,
        visual_input_range=visual_input_range,
        device=device,
        expected_mode=mode,
        model_config=model_config,
        generation_config=generation_config,
        pretrained_action_adapter_index=pretrained_action_adapter_index,
    )
    LOGGER.info("[WSP eval] Policy ready; preparing rollout runtime")
    rollout_steps = int(backend_config.get("rollout_steps", 8))
    max_steps = int(config.get("max_steps", 0))
    if max_steps <= 0 and isinstance(robot, HDF5ReplayRobot):
        max_steps = robot.default_max_steps(rollout_steps)
    if max_steps <= 0:
        raise ValueError("AgileX live evaluation requires a positive max_steps")
    seed = int(config.get("seed", spec.seed))
    artifact_writer = EvaluationArtifactWriter(
        output_dir,
        config_format=str(config.get("config_format", "yaml")),
        video_fps=int(config.get("video_fps", 10)),
    )
    runner = RealRobotRunner(
        PolicyRuntime(policy),
        robot,
        AgileXObservationAdapter(
            bundle,
            mode=mode,
            instruction=spec.task.instruction,
            visual_input_range=visual_input_range,  # type: ignore[arg-type]
            device=device,
            goal_image=goal_image,
            demo_video=demo_video,
            session_id=str(backend_config.get("session_id", spec.episode_id)),
        ),
        AgileXActionAdapter(
            bundle,
            action_mode=action_mode,
        ),
        artifact_writer=artifact_writer,
    )
    LOGGER.info("[WSP eval] Starting AgileX rollout")
    runner.run(
        RealRobotRunnerConfig(
            mode,
            action_rate=int(backend_config.get("action_rate", 24)),
            action_mode=action_mode,
            max_steps=max_steps,
            rollout_steps=rollout_steps,
            refresh_horizon=int(backend_config.get("refresh_horizon", 0)),
            use_history=bool(backend_config.get("use_history", False)),
            num_history_frames=int(
                backend_config.get("num_history_frames", 4)
            ),
            observation_timeout_s=float(
                backend_config.get("observation_timeout_s", 20.0)
            ),
            observation_timeout_policy=str(
                backend_config.get(
                    "observation_timeout_policy",
                    "retry" if transport == "manifold" else "raise",
                )
            ),
            send_timeout_s=(
                float(backend_config["send_timeout_s"])
                if backend_config.get("send_timeout_s") is not None
                else None
            ),
            dry_run=not live_hardware,
            goal_from_first_observation=visual_prompt.goal_from_first_observation,
            context_poll=visual_prompt.poll_transport,
            context_frames=visual_prompt.context_frames,
            ctx_head_only=visual_prompt.ctx_head_only,
            context_poll_timeout_s=float(
                backend_config.get("context_poll_timeout_s", 30.0)
            ),
            episode_id=spec.episode_id,
            task_id=spec.task.task_id,
            instruction=spec.task.instruction,
            seed=seed,
            capture_frames=bool(backend_config.get("capture_frames", False)),
            max_recorded_frames=int(
                backend_config.get("max_recorded_frames", 256)
            ),
        ),
        generator=torch.Generator(device=device).manual_seed(seed),
    )
    if runner.last_result is None:
        raise RuntimeError("AgileX runner did not produce an evaluation result")
    effective_config = dict(config)
    effective_config.update(
        {
            "backend": "agilex",
            "checkpoint": str(checkpoint),
            "output_dir": str(output_dir),
            "mode": mode.value,
            "dry_run": not live_hardware,
            "live_hardware": live_hardware,
        }
    )
    return artifact_writer.write(effective_config, runner.last_result)


def _native_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evals.agilex.evaluate",
        description="Native WorldScape AgileX evaluation (safe dry-run by default)",
    )
    parser.add_argument("--model-path")
    parser.add_argument("--hdf5-path")
    parser.add_argument("--artifacts-dir", default="outputs/eval/agilex")
    parser.add_argument("--language-instruction", default="AgileX rollout")
    parser.add_argument("--embodiment", default=AGILEX)
    parser.add_argument(
        "--embodiment-tag",
        dest="embodiment",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--recipe",
        choices=("text", "goal", "uniform"),
        default="text",
        help="Strict migrated training/evaluation recipe",
    )
    parser.add_argument("--visual-prompt-path")
    parser.add_argument(
        "--goal-from-first-observation",
        action="store_true",
        help="Explicit opt-in fallback for goal recipe only",
    )
    parser.add_argument(
        "--ctx-head-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Demo50 uses V=1 when true and V=3 when false",
    )
    parser.add_argument(
        "--visual-input-range",
        choices=("uint8", "zero_one", "minus_one_one"),
        default="zero_one",
    )
    parser.add_argument("--action-mode", choices=("eef",), default="eef")
    parser.add_argument("--action-rate", type=int, default=24)
    parser.add_argument("--rollout-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--live-hardware", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8887)
    parser.add_argument("--node-name", default="wsp")
    return parser


def main(argv: list[str] | None = None) -> int:
    import sys

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = _native_parser()
    args = parser.parse_args(raw_argv)
    if not args.model_path:
        parser.error("--model-path is required for native evaluation")
    if not args.live_hardware and not args.hdf5_path:
        parser.error("safe native dry-run requires --hdf5-path")
    transport = "manifold" if args.live_hardware else "hdf5"
    if args.recipe == "text":
        visual_prompt = {"kind": "text", "source": "none"}
    elif args.recipe == "goal":
        source = (
            "first_observation"
            if args.goal_from_first_observation
            else "path"
            if args.visual_prompt_path
            else "hdf5"
        )
        visual_prompt = {
            "kind": "goal",
            "source": source,
            "path": args.visual_prompt_path,
            "context_frames": 1,
            "ctx_head_only": True,
            "goal_from_first_observation": args.goal_from_first_observation,
        }
    else:
        source = (
            "path"
            if args.visual_prompt_path
            else "transport"
            if args.live_hardware
            else "hdf5"
        )
        visual_prompt = {
            "kind": "uniform",
            "source": source,
            "path": args.visual_prompt_path,
            "context_frames": 50,
            "ctx_head_only": args.ctx_head_only,
            "poll_transport": source == "transport",
        }
    config = {
        "backend": "agilex",
        "checkpoint": args.model_path,
        "output_dir": args.artifacts_dir,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "visual_input_range": args.visual_input_range,
        "dry_run": not args.live_hardware,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "episodes_per_task": 1,
        "visual_prompt": visual_prompt,
        "backend_config": {
            "transport": transport,
            "path": args.hdf5_path,
            "host": args.host,
            "port": args.port,
            "node_name": args.node_name,
            "action_mode": args.action_mode,
            "action_rate": args.action_rate,
            "rollout_steps": args.rollout_steps,
            "adapter": {"embodiment": args.embodiment},
        },
        "tasks": [
            {"id": "agilex", "instruction": args.language_instruction}
        ],
    }
    summary = run_agilex_recipe(
        config,
        checkpoint=args.model_path,
        output_dir=args.artifacts_dir,
        live_hardware=args.live_hardware,
    )
    print(
        f"Completed {summary['episodes']} AgileX episode(s); "
        f"artifacts: {args.artifacts_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
