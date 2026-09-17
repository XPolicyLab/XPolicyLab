from __future__ import annotations

import base64
import binascii
import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from worldscape_policy.data.sampling import (
    AuditedVisualPromptOverride,
    EventChunkSampler,
    HistorySampler,
    VisualPromptSampler,
)
from worldscape_policy.data.schema import EventSample, VisualPromptMetadata
from worldscape_policy.data.temporal import (
    ContextSampler,
    LanguageTemporalPacker,
    VLMHistorySampler,
)
from worldscape_policy.geometry import quaternion_pose_to_rotation6d


def _profile_sample_enabled() -> bool:
    return os.environ.get("WSP_DATALOADER_PROFILE_SAMPLE", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _profile_log(event: str, payload: dict[str, Any]) -> None:
    if not _profile_sample_enabled():
        return
    print(
        json.dumps({"dataloader_profile": event, **payload}, sort_keys=True),
        flush=True,
    )


def _optional_dependency(module: str, extra: str) -> Any:
    try:
        return import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"{module!r} is required to read this dataset; install "
            f"worldscape-policy[{extra}]"
        ) from exc


def _text(value: object) -> str | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.size == 0:
        return None
    item = array.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8", errors="replace")
    result = str(item)
    return result if result else None


def _first(arrays: Mapping[str, np.ndarray], names: Sequence[str]) -> np.ndarray | None:
    for name in names:
        if name in arrays:
            return np.asarray(arrays[name])
    return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _resolve_episode_path(
    episode: Mapping[str, Any],
    roots: Sequence[Path],
    info: Mapping[str, Any] | None = None,
) -> Path:
    raw = episode.get("path")
    if raw is None and info is not None:
        pattern = info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        )
        index = int(episode["episode_index"])
        raw = str(pattern).format(
            episode_index=index,
            episode_chunk=int(episode.get("episode_chunk", index // 1000)),
        )
    if raw is None:
        raise ValueError("episode metadata needs 'path' or info.json needs 'data_path'")
    path = Path(str(raw))
    if path.is_absolute() and path.is_file():
        return path
    for root in roots:
        candidate = root / path
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"episode data file does not exist: {path}")


def _decode_image(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray) and value.ndim == 3:
        return np.asarray(value, dtype=np.uint8)
    payload: bytes | None = None
    if isinstance(value, Mapping):
        payload = value.get("bytes")  # type: ignore[assignment]
        if payload is None and value.get("path"):
            with Path(str(value["path"])).open("rb") as handle:
                payload = handle.read()
    elif isinstance(value, (bytes, bytearray, np.bytes_, np.void)):
        payload = bytes(value)
    if isinstance(value, str):
        text = value.partition(",")[2] if value.startswith("data:") else value
        try:
            payload = base64.b64decode(text, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("encoded image string is not valid base64") from exc
    if payload is None:
        raise TypeError(f"unsupported encoded image value: {type(value).__name__}")
    # GEAR/HDF5 may store either raw JPEG/PNG bytes or their base64 encoding.
    # Decode only a syntactically valid base64 payload whose result has a known
    # image signature, so ordinary encoded image bytes are preserved verbatim.
    if not payload.startswith((b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF8", b"RIFF")):
        try:
            decoded = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError):
            decoded = b""
        if decoded.startswith((b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF8", b"RIFF")):
            payload = decoded
    image = _optional_dependency("PIL.Image", "train")
    io = import_module("io")
    return np.asarray(image.open(io.BytesIO(payload)).convert("RGB"), dtype=np.uint8)


def _image_sequence(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim == 4:
        return values.astype(np.uint8, copy=False)
    if values.ndim == 1 or values.dtype.kind in {"O", "S", "U", "V"}:
        return np.stack([_decode_image(value) for value in values])
    raise ValueError(
        f"image sequence must be [T,H,W,C] or encoded images, got {values.shape}"
    )


def _read_video(path: Path) -> np.ndarray:
    av = _optional_dependency("av", "core")
    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise ValueError(f"video contains no decodable frames: {path}")
    return np.stack(frames).astype(np.uint8, copy=False)


def _observations(
    arrays: Mapping[str, np.ndarray],
    *,
    mask: np.ndarray | None = None,
    indices: np.ndarray | None = None,
    return_full_head: bool = False,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    selected_indices = (
        None if indices is None else np.asarray(indices, dtype=np.int64).reshape(-1)
    )
    camera_prefixes = (
        "observations.images.",
        "observation.images.",
        "observations.camera.",
        "observation.camera.",
    )
    cameras: list[tuple[str, np.ndarray, np.ndarray]] = []
    for key, value in arrays.items():
        if key.startswith(camera_prefixes):
            name = key.rsplit(".", 1)[-1]
            frames = _image_sequence(value)
            full_frames = frames if mask is None else frames[mask]
            selected = (
                full_frames
                if selected_indices is None
                else full_frames[selected_indices]
            )
            cameras.append((name, selected, full_frames))
    if not cameras:
        direct = _first(arrays, ("observations.video", "observation.video", "video"))
        if direct is None:
            raise ValueError("dataset episode contains no image observation")
        video = np.asarray(direct)
        if mask is not None:
            video = video[mask]
        full_video = video
        if selected_indices is not None:
            video = video[selected_indices]
        if video.ndim == 4:
            video = video[:, None]
        if full_video.ndim == 4:
            full_video = full_video[:, None]
        if video.ndim != 5 or full_video.ndim != 5:
            raise ValueError("video observations must have shape [T,V,H,W,C]")
        head = full_video[:, 0] if return_full_head else video[:, 0]
        return {"video": video.astype(np.uint8, copy=False)}, head
    lengths = {len(frames) for _, frames, _ in cameras}
    trailing = {frames.shape[1:] for _, frames, _ in cameras}
    if len(lengths) != 1 or len(trailing) != 1:
        raise ValueError("camera observations must have matching time and image shapes")
    cameras.sort(key=lambda item: (item[0] not in {"cam_high", "head", "top"}, item[0]))
    video = np.stack([frames for _, frames, _ in cameras], axis=1)
    head = cameras[0][2] if return_full_head else cameras[0][1]
    return {"video": video}, head


def _observation_length(
    arrays: Mapping[str, np.ndarray], *, mask: np.ndarray | None = None
) -> int:
    lengths: list[int] = []
    for key, value in arrays.items():
        if key.startswith(
            (
                "observations.images.",
                "observation.images.",
                "observations.camera.",
                "observation.camera.",
            )
        ):
            lengths.append(len(np.asarray(value)))
    if not lengths:
        direct = _first(arrays, ("observations.video", "observation.video", "video"))
        if direct is not None:
            lengths.append(len(np.asarray(direct)))
    if not lengths:
        raise ValueError("dataset episode contains no image observation")
    length = min(lengths)
    if mask is not None:
        length = min(length, len(mask))
        return int(np.asarray(mask[:length]).astype(bool).sum())
    return int(length)


def _state_and_action(
    arrays: Mapping[str, np.ndarray], mask: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    state = _first(
        arrays,
        (
            "robot_state",
            "observation.state",
            "observations.state",
            "observation.eef6d",
            "state",
        ),
    )
    action = _first(arrays, ("action.eef6d", "actions.eef6d", "actions", "action"))
    converted_legacy_pose = False
    if state is None:
        end_pose = _first(
            arrays,
            (
                "observations.end_pose",
                "observation.end_pose",
                "observations.end_effector_pose",
                "observation.end_effector_pose",
                "end_pose",
            ),
        )
        qpos = _first(
            arrays,
            (
                "observations.qpos",
                "observation.qpos",
                "observations.joint_positions",
                "observation.joint_positions",
                "qpos",
            ),
        )
        if end_pose is None or qpos is None:
            raise ValueError("episode contains no robot state")
        state = _legacy_eef6d(end_pose, qpos)
        converted_legacy_pose = True
    state = _numeric_sequence(state)
    if mask is not None:
        state = state[mask]
    if state.ndim == 1:
        state = state[:, None]
    if action is None:
        action = np.concatenate((state[1:], state[-1:]), axis=0)
    else:
        action = _numeric_sequence(action)
        if mask is not None:
            action = action[mask]
        if action.ndim == 1:
            action = action[:, None]
        if converted_legacy_pose and action.shape[-1] != 20:
            action = np.concatenate((state[1:], state[-1:]), axis=0)
    if len(state) != len(action):
        length = min(len(state), len(action))
        state, action = state[:length], action[:length]
    return state.astype(np.float32, copy=False), action.astype(np.float32, copy=False)


def _legacy_eef6d(end_pose: np.ndarray, qpos: np.ndarray) -> np.ndarray:
    poses = _numeric_sequence(np.asarray(end_pose))
    joints = _numeric_sequence(np.asarray(qpos))
    if poses.ndim != 2 or poses.shape[-1] < 14:
        raise ValueError("legacy two-arm end pose must have shape [T,>=14]")
    if joints.ndim != 2 or joints.shape[-1] < 14:
        raise ValueError("legacy two-arm qpos must have shape [T,>=14]")
    left = quaternion_pose_to_rotation6d(poses[:, :7])
    right = quaternion_pose_to_rotation6d(poses[:, 7:14])
    return np.concatenate(
        (left, joints[:, 6:7], right, joints[:, 13:14]), axis=-1
    ).astype(np.float32, copy=False)


def _numeric_sequence(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind == "O":
        try:
            array = np.stack(array.tolist())
        except ValueError as exc:
            raise ValueError(
                "low-dimensional episode values have inconsistent shapes"
            ) from exc
    return array


def _frame_text_series(
    arrays: Mapping[str, np.ndarray],
    length: int,
    mask: np.ndarray | None,
    field_names: Sequence[str],
) -> np.ndarray | None:
    raw = _first(arrays, field_names)
    if raw is None:
        return None
    values = np.asarray(raw).reshape(-1)
    if values.size <= 1:
        return None
    if mask is not None and len(values) == len(mask):
        values = values[mask]
    if len(values) < length:
        return None
    return np.asarray([_text(value) or "" for value in values[:length]], dtype=object)


def _eligible_temporal_anchors(
    arrays: Mapping[str, np.ndarray],
    max_chunk_size: int,
    *,
    action_horizon: int = 24,
    respect_trajectory_segments: bool = True,
    anchor_start: int = 0,
    step_stride: int = 1,
) -> list[int]:
    if step_stride <= 0:
        raise ValueError("step_stride must be positive")
    action = _first(arrays, ("action.eef6d", "actions.eef6d", "actions", "action"))
    state = _first(
        arrays,
        (
            "robot_state",
            "observation.state",
            "observations.state",
            "observation.eef6d",
            "state",
            "observations.end_pose",
            "observation.end_pose",
        ),
    )
    source = action if action is not None else state
    if source is None:
        return []
    length = len(_numeric_sequence(source))
    is_exec = _first(arrays, ("meta.is_exec", "is_exec"))
    mask = None if is_exec is None else np.asarray(is_exec).astype(bool).reshape(-1)
    if mask is not None:
        length = min(length, int(mask.sum()))
    labels, segment_ids = _temporal_labels(arrays, length, mask)
    packer = LanguageTemporalPacker(
        max_chunk_size=max_chunk_size,
        chunk_size=action_horizon,
    )
    anchors: list[int] = []
    start = int(anchor_start) % int(step_stride)
    for anchor in range(start, length, int(step_stride)):
        try:
            packer.indices(
                anchor,
                labels,
                trajectory_ids=segment_ids if respect_trajectory_segments else None,
            )
        except ValueError:
            continue
        anchors.append(anchor)
    return anchors


def _task_records(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    return {
        int(item.get("task_index", index)): dict(item)
        for index, item in enumerate(_read_jsonl(path))
    }


def _segment_bounds(segment: Mapping[str, Any], length: int) -> tuple[int, int]:
    start = int(
        segment.get(
            "start",
            segment.get("start_index", segment.get("start_frame", segment.get("from", 0))),
        )
    )
    stop = int(
        segment.get(
            "end",
            segment.get("end_index", segment.get("end_frame", segment.get("to", length))),
        )
    )
    return max(0, start), min(length, stop)


def _task_metadata(
    arrays: Mapping[str, np.ndarray],
    episode: Mapping[str, Any],
    tasks: Mapping[int, Mapping[str, Any]],
    length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return real per-frame task IDs and hard temporal segment IDs."""

    raw = _first(arrays, ("task_index", "meta.task_index", "annotation.task_index"))
    fallback = episode.get("task_index", 0)
    task_indices = np.full(length, int(fallback), dtype=np.int64)
    if raw is not None:
        values = np.asarray(raw).reshape(-1)
        if len(values) == 1:
            task_indices.fill(int(values[0]))
        elif len(values) >= length:
            task_indices = values[:length].astype(np.int64, copy=False)

    explicit_segments = np.zeros(length, dtype=np.int64)
    segments = episode.get("task_segments")
    if isinstance(segments, Sequence) and not isinstance(segments, (str, bytes)):
        for segment_id, segment in enumerate(segments, start=1):
            if not isinstance(segment, Mapping):
                continue
            start, stop = _segment_bounds(segment, length)
            if stop <= start:
                continue
            explicit_segments[start:stop] = segment_id
            if raw is None and "task_index" in segment:
                task_indices[start:stop] = int(segment["task_index"])

    event_labels = _frame_text_series(
        arrays,
        length,
        None,
        ("event_instruction", "subtask", "annotation.subtask"),
    )

    # Every explicit boundary and every real task-index transition is a hard
    # trajectory boundary, even if two task records happen to share text.
    boundary = np.zeros(length, dtype=np.int64)
    for index in range(1, length):
        boundary[index] = boundary[index - 1] + int(
            task_indices[index] != task_indices[index - 1]
            or explicit_segments[index] != explicit_segments[index - 1]
            or (
                event_labels is not None
                and event_labels[index] != event_labels[index - 1]
            )
        )
    return task_indices, boundary


def _record_text(record: Mapping[str, Any] | None, *, event: bool) -> str:
    if record is None:
        return ""
    explicit_names = (
        ("event_instruction", "subtask", "event")
        if event
        else ("high_level_instruction", "language")
    )
    explicit = next(
        (str(record[name]) for name in explicit_names if record.get(name)),
        None,
    )
    if explicit is not None:
        return explicit
    task = str(record.get("task", ""))
    high_level, subtask = _split_combined_task_text(task)
    if event and subtask is not None:
        return subtask
    if not event and high_level is not None:
        return high_level
    return task or str(record.get("language", ""))


def _split_combined_task_text(text: str) -> tuple[str | None, str | None]:
    """Parse legacy ``task: ..., sub_task: ..., embodiment_tag: ...`` records."""

    separator = ", sub_task: "
    if separator not in text:
        return None, None
    high_level, subtask = text.split(separator, 1)
    if high_level.startswith("task: "):
        high_level = high_level[len("task: ") :]
    subtask = subtask.split(", embodiment_tag: ", 1)[0]
    return high_level.strip() or None, subtask.strip() or None


def _refine_segment_ids_with_events(
    segment_ids: np.ndarray, event_labels: np.ndarray, length: int
) -> np.ndarray:
    if len(event_labels) != length or len(np.unique(event_labels)) <= 1:
        return segment_ids
    refined = np.zeros(length, dtype=np.int64)
    for index in range(1, length):
        refined[index] = refined[index - 1] + int(
            segment_ids[index] != segment_ids[index - 1]
            or event_labels[index] != event_labels[index - 1]
        )
    return refined


def _metadata_task(
    arrays: dict[str, np.ndarray],
    episode: Mapping[str, Any],
    tasks: Mapping[int, Mapping[str, Any]],
    *,
    length: int | None = None,
) -> None:
    if length is None:
        source = _first(
            arrays,
            (
                "action.eef6d",
                "actions.eef6d",
                "actions",
                "action",
                "robot_state",
                "observation.state",
                "observations.state",
            ),
        )
        length = 1 if source is None else len(_numeric_sequence(source))
    has_real_task_index = (
        _first(arrays, ("task_index", "meta.task_index", "annotation.task_index"))
        is not None
        or "task_index" in episode
        or bool(episode.get("task_segments"))
    )
    task_indices, segment_ids = _task_metadata(arrays, episode, tasks, length)
    arrays["task_index"] = task_indices
    arrays["_task_index_is_real"] = np.asarray(
        has_real_task_index, dtype=np.bool_
    )
    arrays["_task_segment_id"] = segment_ids

    raw_high_level = _first(
        arrays, ("high_level_instruction", "language", "task", "annotation.task")
    )
    if raw_high_level is None or (
        np.asarray(raw_high_level).size == 1 and len(np.unique(task_indices)) > 1
    ):
        episode_text = episode.get("task") or episode.get("language")
        arrays["high_level_instruction"] = np.asarray(
            [
                _record_text(tasks.get(int(task_index)), event=False)
                or (str(episode_text) if episode_text else _text(raw_high_level) or "")
                for task_index in task_indices
            ],
            dtype=object,
        )
    raw_event = _first(
        arrays, ("event_instruction", "subtask", "annotation.subtask")
    )
    if raw_event is None or (
        np.asarray(raw_event).size == 1 and len(np.unique(task_indices)) > 1
    ):
        episode_text = episode.get("event_instruction") or episode.get("subtask")
        arrays["event_instruction"] = np.asarray(
            [
                _record_text(tasks.get(int(task_index)), event=True)
                or (str(episode_text) if episode_text else _text(raw_event) or "")
                for task_index in task_indices
            ],
            dtype=object,
        )
    event_labels = _frame_text_series(
        arrays,
        length,
        None,
        ("event_instruction", "subtask", "annotation.subtask"),
    )
    if event_labels is None:
        event_labels = np.asarray(
            arrays.get("event_instruction", np.full(length, "", dtype=object)),
            dtype=object,
        ).reshape(-1)
    arrays["_task_segment_id"] = _refine_segment_ids_with_events(
        segment_ids, event_labels[:length], length
    )


def _high_level_label_series(
    arrays: Mapping[str, np.ndarray],
    length: int,
    mask: np.ndarray | None,
) -> np.ndarray:
    raw_language = _first(
        arrays, ("high_level_instruction", "language", "task", "annotation.task")
    )
    if (
        raw_language is None
        or np.asarray(raw_language).ndim == 0
        or np.asarray(raw_language).size == 1
    ):
        return np.full(length, _text(raw_language) or "", dtype=object)
    labels = np.asarray(raw_language).reshape(-1)
    if mask is not None and len(labels) == len(mask):
        labels = labels[mask]
    labels = labels[:length]
    if len(labels) != length:
        return np.full(length, _text(raw_language) or "", dtype=object)
    return labels


def _temporal_labels(
    arrays: Mapping[str, np.ndarray],
    length: int,
    mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    event_labels = _frame_text_series(
        arrays,
        length,
        mask,
        ("event_instruction", "subtask", "annotation.subtask"),
    )
    if event_labels is not None and len(np.unique(event_labels)) > 1:
        labels = event_labels
    else:
        labels = None
    raw_language = _first(
        arrays, ("high_level_instruction", "language", "task", "annotation.task")
    )
    if labels is None:
        if (
            raw_language is None
            or np.asarray(raw_language).ndim == 0
            or np.asarray(raw_language).size == 1
        ):
            labels = np.full(length, _text(raw_language) or "", dtype=object)
        else:
            labels = np.asarray(raw_language).reshape(-1)
            if mask is not None and len(labels) == len(mask):
                labels = labels[mask]
            labels = labels[:length]
            if len(labels) != length:
                labels = np.full(length, _text(raw_language) or "", dtype=object)
    raw_segments = _first(arrays, ("_task_segment_id",))
    segments = (
        np.zeros(length, dtype=np.int64)
        if raw_segments is None
        else np.asarray(raw_segments).reshape(-1)
    )
    if mask is not None and len(segments) == len(mask):
        segments = segments[mask]
    if len(segments) != length:
        segments = np.zeros(length, dtype=np.int64)
    return labels[:length], segments[:length]


def _prompt_metadata(
    arrays: Mapping[str, np.ndarray],
    *,
    prompt_name: str,
    task_id: str,
    embodiment: str,
    episode_id: str,
    session_id: str,
    trusted_same_sample: bool,
) -> VisualPromptMetadata:
    def identity(field: str, fallback: str) -> str:
        value = _text(
            _first(
                arrays,
                (
                    f"prompt.{prompt_name}.{field}",
                    f"{prompt_name}_prompt.{field}",
                    f"{prompt_name}_{field}",
                ),
            )
        )
        return fallback if value is None else value

    return VisualPromptMetadata(
        task_id=identity("task_id", task_id),
        embodiment=identity("embodiment", embodiment),
        source_episode_id=identity("source_episode_id", episode_id),
        source_session_id=identity("source_session_id", session_id),
        trusted_same_sample=trusted_same_sample,
    )


def _build_sample(
    arrays: Mapping[str, np.ndarray],
    *,
    episode_id: str,
    session_id: str,
    visual_prompt: str,
    embodiment: str,
    action_horizon: int,
    max_event_chunks: int,
    history_sampler: HistorySampler,
    history_window: int | None,
    seed: int,
    visual_prompt_override: AuditedVisualPromptOverride | None,
    temporal_packing: bool,
    temporal_anchor_index: int,
    max_chunk_size: int,
    respect_trajectory_segments: bool,
    context_sampling_mode: str | None,
    context_video_len: int,
    ctx_head_only: bool,
    wo_norm: bool,
    use_native_history_sampler: bool = False,
) -> EventSample:
    profile_start = time.perf_counter()
    if action_horizon <= 0:
        raise ValueError("action_horizon must be positive")
    is_exec = _first(arrays, ("meta.is_exec", "is_exec"))
    mask = None if is_exec is None else np.asarray(is_exec).astype(bool).reshape(-1)
    state_action_start = time.perf_counter()
    state, action = _state_and_action(arrays, mask)
    if embodiment == "robotwin2":
        if state.ndim != 2 or state.shape[-1] != 14:
            raise ValueError(
                f"RoboTwin2 state must have shape [T,14], got {state.shape}"
            )
        if action.ndim != 2 or action.shape[-1] != 14:
            raise ValueError(
                f"RoboTwin2 action must have shape [T,14], got {action.shape}"
            )
    state_action_s = time.perf_counter() - state_action_start
    length = min(len(action), _observation_length(arrays, mask=mask))
    if length == 0:
        raise ValueError(f"episode {episode_id!r} has no execution samples")
    state, action = state[:length], action[:length]
    observations_s = 0.0
    if not temporal_packing:
        observations_start = time.perf_counter()
        observations, head = _observations(arrays, mask=mask)
        if embodiment == "robotwin2" and (
            set(observations) != {"video"}
            or observations["video"].ndim != 5
            or observations["video"].shape[1] != 3
        ):
            raise ValueError(
                "RoboTwin2 samples require video with shape [T,3,H,W,C], "
                f"got { {key: value.shape for key, value in observations.items()} }"
            )
        observations_s = time.perf_counter() - observations_start
        observations = {name: value[:length] for name, value in observations.items()}

    metadata_start = time.perf_counter()
    packing_labels, segment_ids = _temporal_labels(arrays, length, mask)
    high_level_values = _high_level_label_series(arrays, length, mask)
    if len(packing_labels) != length:
        packing_labels = np.full(
            length,
            _text(
                _first(
                    arrays,
                    ("high_level_instruction", "language", "task", "annotation.task"),
                )
            )
            or "",
            dtype=object,
        )
    event_values = _first(
        arrays, ("event_instruction", "subtask", "annotation.subtask")
    )
    if (
        event_values is None
        or np.asarray(event_values).ndim == 0
        or np.asarray(event_values).size == 1
    ):
        event_values = np.full(length, _text(event_values) or "", dtype=object)
    else:
        event_values = np.asarray(event_values).reshape(-1)
        if mask is not None and len(event_values) == len(mask):
            event_values = event_values[mask]
        event_values = event_values[:length]
    task_indices = _first(arrays, ("task_index", "meta.task_index"))
    if task_indices is not None:
        task_indices = np.asarray(task_indices).reshape(-1)
        if mask is not None and len(task_indices) == len(mask):
            task_indices = task_indices[mask]
        task_indices = task_indices[:length]

    source_indices: dict[str, np.ndarray] | None = None
    if temporal_packing:
        packed = LanguageTemporalPacker(
            max_chunk_size=max_chunk_size,
            chunk_size=action_horizon,
        ).indices(
            temporal_anchor_index,
            packing_labels,
            trajectory_ids=segment_ids if respect_trajectory_segments else None,
        )
        observations_start = time.perf_counter()
        observations, head = _observations(
            arrays,
            mask=mask,
            indices=packed.video,
            return_full_head=history_sampler.num_frames > 0,
        )
        if embodiment == "robotwin2" and (
            set(observations) != {"video"}
            or observations["video"].ndim != 5
            or observations["video"].shape[1] != 3
        ):
            raise ValueError(
                "RoboTwin2 samples require video with shape [T,3,H,W,C], "
                f"got { {key: value.shape for key, value in observations.items()} }"
            )
        if embodiment == "robotwin2":
            video_frames = int(observations["video"].shape[0])
            if video_frames != len(packed.video) or (video_frames - 1) % 4 != 0:
                raise ValueError(
                    "RoboTwin2 packed video must preserve every frame and have "
                    f"VAE-compatible length 1+4k; got {video_frames} frames for "
                    f"{len(packed.video)} indices"
                )
        observations_s = time.perf_counter() - observations_start
        action = action[packed.action]
        state = state[packed.state]
        source_indices = {
            "video": packed.video,
            "action": packed.action,
            "state": packed.state,
            "anchors": packed.anchors,
        }

    label_index = int(np.clip(temporal_anchor_index, 0, length - 1))
    high_level = _text(high_level_values[label_index])
    event_instruction = (
        _text(event_values[label_index]) if len(event_values) == length else None
    ) or high_level
    raw_planning_text = _first(
        arrays,
        ("planning_labels_text", "planning_label_text", "annotation.plan"),
    )
    planning_values = None
    if raw_planning_text is not None and np.asarray(raw_planning_text).size > 1:
        planning_values = np.asarray(raw_planning_text).reshape(-1)
        if mask is not None and len(planning_values) == len(mask):
            planning_values = planning_values[mask]
    planning_labels_text = (
        _text(planning_values[label_index])
        if planning_values is not None and len(planning_values) == length
        else _text(raw_planning_text)
    ) or event_instruction
    task_id = _text(_first(arrays, ("task_id", "meta.task_id")))
    real_task_index = bool(
        np.asarray(arrays.get("_task_index_is_real", False)).reshape(-1)[0]
    )
    if (
        task_id is None
        and real_task_index
        and task_indices is not None
        and len(task_indices) == length
    ):
        task_id = str(task_indices[label_index])
    task_id = task_id or high_level or f"episode:{episode_id}"
    planning_labels = _first(
        arrays, ("planning_labels", "planning_token_ids", "annotation.planning_labels")
    )
    semantic_target = _first(
        arrays,
        ("semantic_target", "event_semantic_target", "annotation.semantic_target"),
    )
    semantic_mask = _first(
        arrays, ("semantic_mask", "event_semantic_mask", "annotation.semantic_mask")
    )
    action_dim_mask = _first(
        arrays,
        ("action_dim_mask", "action_dimension_mask", "action.dim_mask"),
    )
    has_real_action_value = _first(
        arrays,
        ("has_real_action", "action.has_real_action"),
    )
    has_real_action = (
        True
        if has_real_action_value is None
        else bool(np.asarray(has_real_action_value).reshape(-1)[0])
    )
    event_id = f"{episode_id}:0"
    metadata_s = time.perf_counter() - metadata_start

    prompt_context_start = time.perf_counter()
    explicit_demo = _first(arrays, ("demo_video", "prompt.demo_video"))
    explicit_goal = _first(arrays, ("goal_image", "prompt.goal_image"))
    context_head: np.ndarray | None = None
    context_video: np.ndarray | None = None
    if mask is not None and np.any(~mask):
        context_observations, context_head = _observations(arrays, mask=~mask)
        context_video = context_observations["video"]
    if context_sampling_mode is None:
        demo_video = (
            _image_sequence(explicit_demo) if explicit_demo is not None else context_head
        )
        goal_image = (
            np.asarray(explicit_goal, dtype=np.uint8)
            if explicit_goal is not None
            else (None if context_head is None else context_head[-1])
        )
    else:
        sampler = ContextSampler(
            context_sampling_mode,
            context_video_len,
            ctx_head_only=ctx_head_only,
        )
        sampled_context = sampler.sample(
            (
                np.empty((0, *head.shape[1:]), dtype=np.uint8)
                if context_head is None
                else context_head
            )
            if ctx_head_only
            else (
                np.empty((0, 1, *head.shape[1:]), dtype=np.uint8)
                if context_video is None
                else context_video
            )
        )
        demo_video = (
            sampled_context if context_sampling_mode == "uniform" else None
        )
        goal_image = (
            sampled_context[-1]
            if context_sampling_mode == "last" and sampled_context is not None
            else None
        )
    demo_metadata = (
        None
        if demo_video is None
        else _prompt_metadata(
            arrays,
            prompt_name="demo",
            task_id=task_id,
            embodiment=embodiment,
            episode_id=episode_id,
            session_id=session_id,
            trusted_same_sample=explicit_demo is None and context_head is not None,
        )
    )
    goal_metadata = (
        None
        if goal_image is None
        else _prompt_metadata(
            arrays,
            prompt_name="goal",
            task_id=task_id,
            embodiment=embodiment,
            episode_id=episode_id,
            session_id=session_id,
            trusted_same_sample=explicit_goal is None and context_head is not None,
        )
    )
    prompt_context_s = time.perf_counter() - prompt_context_start

    chunk_start = time.perf_counter()
    sample_length = len(action)
    chunks: list[EventSample] = []
    iteration_horizon = sample_length if temporal_packing else action_horizon
    for start in range(0, sample_length, iteration_horizon):
        stop = min(start + iteration_horizon, sample_length)
        chunk = EventSample(
            episode_id=episode_id,
            event_id=event_id,
            observations={
                name: value[start:stop] for name, value in observations.items()
            },
            actions=action[start:stop],
            robot_state=state[start:stop],
            high_level_instruction=high_level,
            event_instruction=event_instruction,
            goal_image=goal_image,
            demo_video=demo_video,
            history_head_frames=None,
            embodiment=embodiment,
            task_id=task_id,
            session_id=session_id,
            goal_prompt_metadata=goal_metadata,
            demo_prompt_metadata=demo_metadata,
            planning_labels_text=planning_labels_text,
            planning_labels=(
                None
                if planning_labels is None
                else np.asarray(planning_labels, dtype=np.int64).reshape(-1)
            ),
            semantic_target=(
                None
                if semantic_target is None
                else np.asarray(semantic_target, dtype=np.float32)
            ),
            semantic_mask=(
                None
                if semantic_mask is None
                else np.asarray(semantic_mask, dtype=np.bool_).reshape(-1)
            ),
            action_dim_mask=(
                None
                if action_dim_mask is None
                else np.asarray(action_dim_mask, dtype=np.float32).reshape(-1)
            ),
            has_real_action=has_real_action,
            source_indices=source_indices,
            provenance={
                "temporal_packing": temporal_packing,
                "max_chunk_size": max_chunk_size,
                "context_sampling_mode": context_sampling_mode or "legacy",
                "ctx_head_only": ctx_head_only,
                "wo_norm": wo_norm,
                "source_start": start,
                "source_stop": stop,
            },
        )
        chunk.validate()
        chunks.append(chunk)
    selected = EventChunkSampler(max_event_chunks).sample(
        chunks, rng=np.random.default_rng(seed)
    )
    if source_indices is not None:
        history_anchor = int(source_indices["anchors"][0])
    elif embodiment == "robotwin2":
        history_anchor = int(selected[0].provenance["source_start"])
    else:
        history_anchor = len(head) - 1
    if history_sampler.num_frames == 0:
        history = head[:0].copy()
    elif use_native_history_sampler:
        history = history_sampler.sample(head, anchor_index=history_anchor)
    else:
        history = VLMHistorySampler(
            num_frames=history_sampler.num_frames,
            stride=history_sampler.stride,
            window=history_window
            if history_window is not None
            else history_sampler.num_frames * history_sampler.stride,
        ).sample(head, history_anchor)
    if (
        embodiment == "robotwin2"
        and history_sampler.num_frames > 0
        and len(history) < history_sampler.num_frames
    ):
        padding = np.repeat(
            head[:1],
            history_sampler.num_frames - len(history),
            axis=0,
        )
        history = np.concatenate((padding, history), axis=0)
    sample = replace(
        selected[0],
        observations={
            name: np.concatenate([chunk.observations[name] for chunk in selected])
            for name in selected[0].observations
        },
        actions=np.concatenate([chunk.actions for chunk in selected]),
        robot_state=np.concatenate([chunk.robot_state for chunk in selected]),
        history_head_frames=history if len(history) else None,
        observation_valid_masks={
            name: np.ones(
                sum(len(chunk.observations[name]) for chunk in selected), dtype=np.bool_
            )
            for name in selected[0].observations
        },
        action_valid_mask=np.ones(
            sum(len(chunk.actions) for chunk in selected), dtype=np.bool_
        ),
        robot_state_valid_mask=np.ones(
            sum(len(chunk.robot_state) for chunk in selected), dtype=np.bool_
        ),
    )
    chunk_history_s = time.perf_counter() - chunk_start
    visual_prompt_start = time.perf_counter()
    prompt = VisualPromptSampler(visual_prompt, override=visual_prompt_override).sample(
        sample, rng=np.random.default_rng(seed)
    )
    sample = replace(
        sample,
        goal_image=prompt.goal_image,
        demo_video=prompt.demo_video,
        goal_prompt_metadata=prompt.goal_metadata,
        demo_prompt_metadata=prompt.demo_metadata,
    )
    sample.validate()
    visual_prompt_s = time.perf_counter() - visual_prompt_start
    _profile_log(
        "hdf5_build_sample",
        {
            "episode_id": episode_id,
            "anchor": int(temporal_anchor_index),
            "length": int(length),
            "temporal_packing": bool(temporal_packing),
            "observations_s": round(observations_s, 6),
            "state_action_s": round(state_action_s, 6),
            "metadata_s": round(metadata_s, 6),
            "prompt_context_s": round(prompt_context_s, 6),
            "chunk_history_s": round(chunk_history_s, 6),
            "visual_prompt_s": round(visual_prompt_s, 6),
            "total_s": round(time.perf_counter() - profile_start, 6),
            "observation_shapes": {
                name: list(value.shape) for name, value in observations.items()
            },
        },
    )
    return sample


def _effective_length(arrays: Mapping[str, np.ndarray]) -> int:
    source = _first(
        arrays,
        (
            "action.eef6d",
            "actions.eef6d",
            "actions",
            "action",
            "robot_state",
            "observation.state",
            "observations.state",
            "observation.eef6d",
            "state",
        ),
    )
    if source is None:
        return 0
    length = len(_numeric_sequence(source))
    is_exec = _first(arrays, ("meta.is_exec", "is_exec"))
    if is_exec is not None:
        length = min(length, int(np.asarray(is_exec).astype(bool).reshape(-1).sum()))
    return length


def _selected_chunk_length(
    length: int,
    *,
    action_horizon: int,
    max_event_chunks: int,
    seed: int,
) -> int:
    chunk_lengths = [
        min(action_horizon, length - start)
        for start in range(0, length, action_horizon)
    ]
    count = min(len(chunk_lengths), max_event_chunks)
    if not chunk_lengths:
        return 0
    start = (
        0
        if len(chunk_lengths) == count
        else int(np.random.default_rng(seed).integers(0, len(chunk_lengths) - count + 1))
    )
    return sum(chunk_lengths[start : start + count])


