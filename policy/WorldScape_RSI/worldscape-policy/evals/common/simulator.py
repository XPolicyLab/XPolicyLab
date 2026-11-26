from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from evals.common.protocols import ActionAdapter, ObservationAdapter
from evals.common.suite import EvaluationTask
from worldscape_policy.types import (
    InteractionMode,
    ObservationBatch,
    PromptBatch,
    WorldActionOutput,
)


class OptionalSimulatorDependencyError(ImportError):
    """Raised when an optional simulator package is requested but unavailable."""


@runtime_checkable
class SimulatorEnvironment(Protocol):
    """Minimal environment API consumed by simulator adapters."""

    def reset(self, **kwargs: Any) -> Any: ...

    def step(self, action: np.ndarray) -> Any: ...


@runtime_checkable
class SimulatorAdapterProtocol(
    ObservationAdapter[Any, ObservationBatch],
    ActionAdapter[np.ndarray],
    Protocol,
):
    """Backend-neutral boundary between a simulator and WorldScape."""

    def prompt(
        self,
        instruction: str,
        *,
        mode: InteractionMode | str,
    ) -> PromptBatch: ...

@dataclass(frozen=True)
class ObservationMapping:
    """Keys used to translate a simulator observation into public schemas."""

    camera_keys: tuple[str, ...]
    state_keys: tuple[str, ...]
    head_camera_key: str | None = None
    embodiment_id: int = 0


class SimulatorAdapter:
    """Reusable schema mapping for dictionary-like simulator observations."""

    def __init__(
        self,
        mapping: ObservationMapping,
        *,
        action_transform: Callable[[np.ndarray], np.ndarray] | None = None,
        image_size: tuple[int, int] | None = None,
        image_resize_interpolation: str = "linear",
        preserve_camera_resolution: bool = False,
        head_video_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        if not mapping.camera_keys:
            raise ValueError("At least one camera key is required")
        if not mapping.state_keys:
            raise ValueError("At least one state key is required")
        if image_size is not None and (
            len(image_size) != 2 or any(int(value) <= 0 for value in image_size)
        ):
            raise ValueError("image_size must be a positive (height, width) pair")
        if image_resize_interpolation not in {"linear", "area"}:
            raise ValueError(
                "image_resize_interpolation must be 'linear' or 'area'"
            )
        self.mapping = mapping
        self._action_transform = action_transform
        self._image_resize_interpolation = image_resize_interpolation
        self._preserve_camera_resolution = bool(preserve_camera_resolution)
        self._head_video_transform = head_video_transform
        self._image_size = (
            None
            if image_size is None
            else (int(image_size[0]), int(image_size[1]))
        )

    def observation(
        self,
        value: Any,
        *,
        device: torch.device | str = "cpu",
    ) -> ObservationBatch:
        observation = _as_mapping(_unwrap_reset(value))
        videos = [
            (
                _camera_video(_lookup(observation, key), key=key)
                if self._preserve_camera_resolution
                else _resize_camera_video(
                    _camera_video(_lookup(observation, key), key=key),
                    self._image_size,
                    interpolation=self._image_resize_interpolation,
                )
            )
            for key in self.mapping.camera_keys
        ]
        history_length = max(video.shape[0] for video in videos)
        videos = [
            np.repeat(video, history_length, axis=0)
            if video.shape[0] == 1
            else video
            for video in videos
        ]
        if any(video.shape[0] != history_length for video in videos):
            raise ValueError("Simulator camera histories must have equal lengths")
        image_array = np.stack(videos, axis=1)

        head_key = self.mapping.head_camera_key or self.mapping.camera_keys[0]
        head = _camera_video(_lookup(observation, head_key), key=head_key)
        if self._head_video_transform is not None:
            head = _camera_video(
                self._head_video_transform(head),
                key=head_key,
            )
        else:
            head = _resize_camera_video(
                head,
                self._image_size,
                interpolation=self._image_resize_interpolation,
            )
        # Simulator histories are chronological, whereas training's Qwen
        # head_view is the temporal anchor. The current anchor is therefore
        # the final frame of an evaluation history, not the first.
        head = head[-1:]
        state_values = [
            np.asarray(_lookup(observation, key), dtype=np.float32).reshape(-1)
            for key in self.mapping.state_keys
        ]
        if not state_values:
            raise ValueError("Simulator observation did not contain any state values")

        batch = ObservationBatch(
            images=torch.from_numpy(image_array)
            .permute(0, 1, 4, 2, 3)
            .unsqueeze(0)
            .float()
            .div(255.0)
            .to(device),
            head_view=torch.from_numpy(head)
            .permute(0, 3, 1, 2)
            .unsqueeze(0)
            .float()
            .div(255.0)
            .to(device),
            proprioception=torch.from_numpy(np.concatenate(state_values))
            .view(1, 1, -1)
            .to(device),
            embodiment_id=torch.tensor(
                [self.mapping.embodiment_id], dtype=torch.long, device=device
            ),
        )
        batch.validate()
        return batch

    def prompt(
        self,
        instruction: str,
        *,
        mode: InteractionMode | str,
    ) -> PromptBatch:
        parsed_mode = InteractionMode.parse(mode)
        if parsed_mode is InteractionMode.AUTO:
            return PromptBatch(
                vlm_planning_text=[instruction],
                negative_vlm_text=[""],
            )
        return PromptBatch(
            language_instruction=[instruction],
            negative_language_instruction=[""],
        )

    def action(self, output: WorldActionOutput) -> np.ndarray:
        action = output.require_action().detach().to(device="cpu", dtype=torch.float32)
        if action.ndim == 0:
            raise ValueError("WorldScape action must have at least one dimension")
        if action.ndim >= 2:
            if action.shape[0] != 1:
                raise ValueError(
                    "Simulator adapters support one environment at a time; "
                    f"got action batch size {action.shape[0]}"
                )
            action = action[0]
        result = np.asarray(action.numpy())
        if self._action_transform is not None:
            result = np.asarray(self._action_transform(result))
        return result


class TaskFactoryEvaluationEnvironment:
    """Construct a fresh optional simulator environment for each trial."""

    def __init__(
        self,
        *,
        module_name: str,
        factory_name: str,
        backend_name: str,
        environment_args: tuple[Any, ...] = (),
        environment_kwargs: Mapping[str, Any] | None = None,
        metadata_to_factory: Mapping[str, str] | None = None,
        capture_frames: bool = False,
    ) -> None:
        self.module_name = module_name
        self.factory_name = factory_name
        self.backend_name = backend_name
        self.environment_args = environment_args
        self.environment_kwargs = dict(environment_kwargs or {})
        self.metadata_to_factory = dict(metadata_to_factory or {})
        self.capture_frames = capture_frames
        self._environment: Any = None

    def reset(self, task: EvaluationTask, *, seed: int | None = None) -> Any:
        self.close()
        kwargs = dict(self.environment_kwargs)
        task_kwargs = task.metadata.get("environment_kwargs", {})
        if not isinstance(task_kwargs, Mapping):
            raise TypeError("Task environment_kwargs must be a mapping")
        kwargs.update(task_kwargs)
        for metadata_name, factory_name in self.metadata_to_factory.items():
            if metadata_name in task.metadata:
                kwargs[factory_name] = task.metadata[metadata_name]
        factory = load_optional_factory(
            self.module_name,
            self.factory_name,
            backend_name=self.backend_name,
        )
        native = factory(*self.environment_args, **kwargs)
        from evals.common.environment import (
            SimulatorEvaluationEnvironment,
        )

        self._environment = SimulatorEvaluationEnvironment(
            native,
            capture_frames=self.capture_frames,
        )
        return self._environment.reset(task, seed=seed)

    def step(self, action: np.ndarray) -> Any:
        return self._require_environment().step(action)

    def success(self) -> bool:
        return self._require_environment().success()

    def metrics(self) -> dict[str, float]:
        return self._require_environment().metrics()

    def close(self) -> None:
        if self._environment is not None:
            self._environment.close()
            self._environment = None

    def _require_environment(self) -> Any:
        if self._environment is None:
            raise RuntimeError("Call reset before using the task environment")
        return self._environment


def load_optional_factory(
    module_name: str,
    factory_name: str,
    *,
    backend_name: str,
) -> Callable[..., Any]:
    """Resolve an optional simulator factory only when construction is requested."""

    try:
        module = import_module(module_name)
    except ImportError as exc:
        raise OptionalSimulatorDependencyError(
            f"{backend_name} support requires optional module {module_name!r}. "
            f"Install the {backend_name} simulator dependencies before creating "
            "the environment."
        ) from exc
    try:
        factory = getattr(module, factory_name)
    except AttributeError as exc:
        raise OptionalSimulatorDependencyError(
            f"Optional module {module_name!r} does not expose factory "
            f"{factory_name!r} required by the {backend_name} adapter."
        ) from exc
    if not callable(factory):
        raise TypeError(f"{module_name}.{factory_name} is not callable")
    return factory


def _unwrap_reset(value: Any) -> Any:
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], Mapping):
        return value[0]
    return value


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "__dict__"):
        return vars(value)
    raise TypeError(
        "Simulator observation must be mapping-like or expose attributes, "
        f"got {type(value).__name__}"
    )


def _lookup(value: Mapping[str, Any], key: str) -> Any:
    current: Any = value
    for part in key.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            raise KeyError(f"Simulator observation is missing required key {key!r}")
    return current


def _camera_video(value: Any, *, key: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim == 3:
        image = image[None]
    if image.ndim != 4:
        raise ValueError(
            f"Camera {key!r} must have shape [H,W,C], [C,H,W], "
            f"[T,H,W,C], or [T,C,H,W], got {image.shape}"
        )
    if image.shape[-1] in (1, 3, 4):
        video = image[..., :3]
    elif image.shape[1] in (1, 3, 4):
        video = np.moveaxis(image[:, :3], 1, -1)
    else:
        raise ValueError(f"Camera {key!r} has no recognizable channel dimension")
    if video.shape[-1] == 1:
        video = np.repeat(video, 3, axis=-1)
    if np.issubdtype(video.dtype, np.floating):
        maximum = float(video.max()) if video.size else 0.0
        if maximum <= 1.0:
            video = video * 255.0
    return np.ascontiguousarray(np.clip(video, 0, 255).astype(np.uint8))


def _resize_camera_video(
    video: np.ndarray,
    image_size: tuple[int, int] | None,
    *,
    interpolation: str = "linear",
) -> np.ndarray:
    if image_size is None or video.shape[1:3] == image_size:
        return video
    import cv2

    height, width = image_size
    cv2_interpolation = (
        cv2.INTER_AREA if interpolation == "area" else cv2.INTER_LINEAR
    )
    return np.stack(
        [
            cv2.resize(
                frame,
                (width, height),
                interpolation=cv2_interpolation,
            )
            for frame in video
        ],
        axis=0,
    ).astype(np.uint8, copy=False)
