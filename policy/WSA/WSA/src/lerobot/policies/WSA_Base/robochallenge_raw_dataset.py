#!/usr/bin/env python

from __future__ import annotations

import bisect
import hashlib
import json
import logging
import os
import re
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from lerobot.datasets.utils import cast_stats_to_numpy, load_json
from lerobot.transforms.core import (
    DataTransformFn,
    compose,
    filter_image_features,
    hydrate_compose_field_transform,
    hydrate_delta_action_transform,
    hydrate_normalize_transform,
    hydrate_remap_image_key_transform,
)
from lerobot.utils.constants import ACTION, OBS_STATE

ROBOCHALLENGE_W1_ROBOT_TYPE = "DOS-W1"
ROBOCHALLENGE_W1_STATE_DIM = 14
ROBOCHALLENGE_W1_CAMERA_KEYS = (
    "observation.images.head",
    "observation.images.left",
    "observation.images.right",
)
ROBOCHALLENGE_W1_VIDEO_FILES = {
    "observation.images.head": "cam_high_rgb.mp4",
    "observation.images.left": "cam_left_wrist_rgb.mp4",
    "observation.images.right": "cam_right_wrist_rgb.mp4",
}
ROBOCHALLENGE_W1_DELTA_MASK = np.asarray(
    [True] * 6 + [False] + [True] * 6 + [False],
    dtype=bool,
)
ROBOCHALLENGE_W1_TABLE30V2_REGULAR_TASKS = (
    "fold_the_clothes",
    "tidy_up_the_makeup_table",
    "put_in_pen_container",
    "hold_the_tray_with_both_hands",
    "stack_bowls",
    "place_objects_into_desk_drawer",
)
ROBOCHALLENGE_W1_TABLE30V2_EXTRA_TASKS = (
    "sweep_the_trash",
    "put_the_shoes_back",
    "tie_a_knot",
    "untie_the_shoelaces",
)
ROBOCHALLENGE_W1_TASK_PRESET_TABLE30V2 = "table30v2_w1"
ROBOCHALLENGE_W1_TABLE30V2_REGULAR_TOTAL_WEIGHT = 6.0
ROBOCHALLENGE_W1_TABLE30V2_EXTRA_TOTAL_WEIGHT = 3.0
ROBOCHALLENGE_W1_TASK_SAMPLING_NONE = "none"
ROBOCHALLENGE_W1_TASK_SAMPLING_PER_TASK = "per_task"
ROBOCHALLENGE_W1_TASK_SAMPLING_GROUP_FRAMES_POW = "group_frames_pow"

ROBOCHALLENGE_ALOHA_ROBOT_TYPE = "ALOHA"
ROBOCHALLENGE_ALOHA_STATE_DIM = 14
ROBOCHALLENGE_ALOHA_CAMERA_KEYS = ROBOCHALLENGE_W1_CAMERA_KEYS
ROBOCHALLENGE_ALOHA_VIDEO_FILES = ROBOCHALLENGE_W1_VIDEO_FILES
ROBOCHALLENGE_ALOHA_TABLE30V2_REGULAR_TASKS = (
    "put_the_books_back",
    "stamp_positioning",
    "wipe_the_blackboard",
    "scoop_with_a_small_spoon",
)
ROBOCHALLENGE_ALOHA_TABLE30V2_EXTRA_TASKS = (
    "wrap_with_a_soft_cloth",
    "paint_jam",
    "pack_the_items",
    "put_the_pencil_case_into_the_schoolbag",
    "pack_the_toothbrush_holder",
    "lint_roller_remove_dirt",
)
ROBOCHALLENGE_ALOHA_TASK_PRESET_TABLE30V2 = "table30v2_aloha"
ROBOCHALLENGE_ALOHA_TABLE30V2_REGULAR_TOTAL_WEIGHT = 4.0
ROBOCHALLENGE_ALOHA_TABLE30V2_EXTRA_TOTAL_WEIGHT = 4.0


@dataclass(frozen=True)
class RoboChallengeRawEmbodimentSpec:
    key: str
    robot_type: str
    state_dim: int
    camera_keys: tuple[str, ...]
    video_files: dict[str, str]
    regular_tasks: tuple[str, ...]
    extra_tasks: tuple[str, ...]
    task_preset: str
    regular_total_weight: float
    extra_total_weight: float

    @property
    def task_names(self) -> tuple[str, ...]:
        return self.regular_tasks + self.extra_tasks


ROBOCHALLENGE_RAW_W1_SPEC = RoboChallengeRawEmbodimentSpec(
    key="w1",
    robot_type=ROBOCHALLENGE_W1_ROBOT_TYPE,
    state_dim=ROBOCHALLENGE_W1_STATE_DIM,
    camera_keys=ROBOCHALLENGE_W1_CAMERA_KEYS,
    video_files=ROBOCHALLENGE_W1_VIDEO_FILES,
    regular_tasks=ROBOCHALLENGE_W1_TABLE30V2_REGULAR_TASKS,
    extra_tasks=ROBOCHALLENGE_W1_TABLE30V2_EXTRA_TASKS,
    task_preset=ROBOCHALLENGE_W1_TASK_PRESET_TABLE30V2,
    regular_total_weight=ROBOCHALLENGE_W1_TABLE30V2_REGULAR_TOTAL_WEIGHT,
    extra_total_weight=ROBOCHALLENGE_W1_TABLE30V2_EXTRA_TOTAL_WEIGHT,
)
ROBOCHALLENGE_RAW_ALOHA_SPEC = RoboChallengeRawEmbodimentSpec(
    key="aloha",
    robot_type=ROBOCHALLENGE_ALOHA_ROBOT_TYPE,
    state_dim=ROBOCHALLENGE_ALOHA_STATE_DIM,
    camera_keys=ROBOCHALLENGE_ALOHA_CAMERA_KEYS,
    video_files=ROBOCHALLENGE_ALOHA_VIDEO_FILES,
    regular_tasks=ROBOCHALLENGE_ALOHA_TABLE30V2_REGULAR_TASKS,
    extra_tasks=ROBOCHALLENGE_ALOHA_TABLE30V2_EXTRA_TASKS,
    task_preset=ROBOCHALLENGE_ALOHA_TASK_PRESET_TABLE30V2,
    regular_total_weight=ROBOCHALLENGE_ALOHA_TABLE30V2_REGULAR_TOTAL_WEIGHT,
    extra_total_weight=ROBOCHALLENGE_ALOHA_TABLE30V2_EXTRA_TOTAL_WEIGHT,
)
ROBOCHALLENGE_RAW_SPECS_BY_PRESET = {
    ROBOCHALLENGE_RAW_W1_SPEC.task_preset: ROBOCHALLENGE_RAW_W1_SPEC,
    ROBOCHALLENGE_RAW_ALOHA_SPEC.task_preset: ROBOCHALLENGE_RAW_ALOHA_SPEC,
}
ROBOCHALLENGE_RAW_SPECS_BY_EMBODIMENT = {
    "DOSW1": ROBOCHALLENGE_RAW_W1_SPEC,
    "W1": ROBOCHALLENGE_RAW_W1_SPEC,
    "ALOHA": ROBOCHALLENGE_RAW_ALOHA_SPEC,
}


@dataclass(frozen=True)
class RoboChallengeRawEpisode:
    task_name: str
    prompt: str
    episode_dir: Path
    left_states_path: Path
    right_states_path: Path
    video_paths: dict[str, Path]
    n_states: int
    sample_start: int
    sample_end: int
    frame_interval: int

    @property
    def sample_count(self) -> int:
        return self.sample_end - self.sample_start


def _normalize_embodiment_name(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def _normalize_task_preset(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"", "all", "none", "null"}:
        return None
    return normalized


def get_robochallenge_raw_spec(
    embodiment: str | None = None,
    task_preset: str | None = None,
) -> RoboChallengeRawEmbodimentSpec:
    preset = _normalize_task_preset(task_preset)
    if preset is not None:
        spec = ROBOCHALLENGE_RAW_SPECS_BY_PRESET.get(preset)
        if spec is None:
            raise ValueError(f"Unknown RoboChallenge raw task_preset={task_preset!r}")
        return spec

    normalized_embodiment = _normalize_embodiment_name(embodiment or ROBOCHALLENGE_W1_ROBOT_TYPE)
    spec = ROBOCHALLENGE_RAW_SPECS_BY_EMBODIMENT.get(normalized_embodiment)
    if spec is None:
        available = ", ".join(sorted(spec.robot_type for spec in ROBOCHALLENGE_RAW_SPECS_BY_PRESET.values()))
        raise ValueError(f"Unsupported RoboChallenge raw embodiment={embodiment!r}. Available: {available}")
    return spec


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _count_jsonl_lines(path: Path) -> int:
    count = 0
    with path.open("rb") as f:
        for _ in f:
            count += 1
    return count


def _is_robochallenge_task_dir(path: Path) -> bool:
    return (path / "meta" / "task_info.json").is_file() and (path / "data").is_dir()


def _resolve_robochallenge_task_dir(path: Path) -> Path | None:
    if _is_robochallenge_task_dir(path):
        return path

    nested = path / path.name
    if nested.is_dir() and _is_robochallenge_task_dir(nested):
        return nested

    return None


def resolve_robochallenge_raw_task_names(task_preset: str | None) -> tuple[str, ...] | None:
    if _normalize_task_preset(task_preset) is None:
        return None
    return get_robochallenge_raw_spec(task_preset=task_preset).task_names


def resolve_robochallenge_w1_task_names(task_preset: str | None) -> tuple[str, ...] | None:
    return resolve_robochallenge_raw_task_names(task_preset)


def resolve_robochallenge_raw_task_weights(
    task_preset: str | None,
    *,
    regular_task_weight: float = 1.0,
    extra_task_weight: float = 0.8,
) -> dict[str, float] | None:
    if resolve_robochallenge_raw_task_names(task_preset) is None:
        return None
    spec = get_robochallenge_raw_spec(task_preset=task_preset)
    return {
        **{task: float(regular_task_weight) for task in spec.regular_tasks},
        **{task: float(extra_task_weight) for task in spec.extra_tasks},
    }


def resolve_robochallenge_w1_task_weights(
    task_preset: str | None,
    *,
    regular_task_weight: float = 1.0,
    extra_task_weight: float = 0.8,
) -> dict[str, float] | None:
    return resolve_robochallenge_raw_task_weights(
        task_preset,
        regular_task_weight=regular_task_weight,
        extra_task_weight=extra_task_weight,
    )


def _table30v2_group_frame_pow_task_weights(
    task_sample_counts: dict[str, int],
    *,
    regular_tasks: Sequence[str],
    extra_tasks: Sequence[str],
    regular_total_weight: float,
    extra_total_weight: float,
    gamma: float,
) -> dict[str, float]:
    if regular_total_weight <= 0:
        raise ValueError("regular_total_weight must be positive")
    if extra_total_weight <= 0:
        raise ValueError("extra_total_weight must be positive")
    if gamma < 0:
        raise ValueError("task_sampling_gamma must be non-negative")

    task_weights: dict[str, float] = {}
    groups = (
        (tuple(regular_tasks), float(regular_total_weight)),
        (tuple(extra_tasks), float(extra_total_weight)),
    )
    for task_names, group_weight in groups:
        present = [
            (task_name, int(task_sample_counts[task_name]))
            for task_name in task_names
            if int(task_sample_counts.get(task_name, 0)) > 0
        ]
        if not present:
            continue

        scores = [float(sample_count) ** float(gamma) for _, sample_count in present]
        total_score = sum(scores)
        if total_score <= 0:
            continue
        for (task_name, _), score in zip(present, scores):
            task_weights[task_name] = group_weight * score / total_score

    return task_weights


def _task_prompt(task_dir: Path, task_info: dict[str, Any]) -> str:
    task_desc = task_info.get("task_desc") or {}
    prompt = task_desc.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()

    task_desc_path = task_dir / "task_desc.json"
    if task_desc_path.is_file():
        payload = _read_json(task_desc_path)
        for key in ("prompt", "description", "task_desc"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return task_dir.name.replace("_", " ")


def _task_matches_embodiment(task_info: dict[str, Any], embodiment: str) -> bool:
    wanted = _normalize_embodiment_name(embodiment)
    aliases = {wanted}
    if wanted in {"DOSW1", "W1"}:
        aliases.update({"DOSW1", "W1"})

    tags = (task_info.get("task_desc") or {}).get("task_tag") or []
    if isinstance(tags, str):
        tags = [tags]
    normalized_tags = {_normalize_embodiment_name(tag) for tag in tags}
    return bool(aliases & normalized_tags)


def _sample_count_from_states(n_states: int, frame_interval: int) -> int:
    return len(range(frame_interval, int(n_states), int(frame_interval)))


def discover_robochallenge_w1_episodes(
    raw_root: str | Path,
    *,
    embodiment: str = ROBOCHALLENGE_W1_ROBOT_TYPE,
    frame_interval: int = 1,
    task_regex: str | None = None,
    task_names: Sequence[str] | None = None,
) -> list[RoboChallengeRawEpisode]:
    raw_root = Path(raw_root).expanduser()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"RoboChallenge raw root does not exist: {raw_root}")
    if frame_interval <= 0:
        raise ValueError("frame_interval must be positive")
    spec = get_robochallenge_raw_spec(embodiment=embodiment)

    task_dirs: list[Path]
    resolved_root = _resolve_robochallenge_task_dir(raw_root)
    if resolved_root is not None:
        task_dirs = [resolved_root]
    else:
        task_dirs = []
        for child in sorted(raw_root.iterdir()):
            if not child.is_dir():
                continue
            resolved = _resolve_robochallenge_task_dir(child)
            if resolved is not None:
                task_dirs.append(resolved)

    if task_regex:
        pattern = re.compile(task_regex)
        task_dirs = [task_dir for task_dir in task_dirs if pattern.search(task_dir.name)]
    if task_names:
        wanted_tasks = {str(task_name) for task_name in task_names}
        task_dirs = [task_dir for task_dir in task_dirs if task_dir.name in wanted_tasks]

    episodes: list[RoboChallengeRawEpisode] = []
    sample_cursor = 0

    for task_dir in task_dirs:
        task_info = _read_json(task_dir / "meta" / "task_info.json")
        if not _task_matches_embodiment(task_info, embodiment):
            continue

        prompt = _task_prompt(task_dir, task_info)
        for episode_dir in sorted((task_dir / "data").iterdir()):
            if not episode_dir.is_dir():
                continue

            states_dir = episode_dir / "states"
            videos_dir = episode_dir / "videos"
            left_states_path = states_dir / "left_states.jsonl"
            right_states_path = states_dir / "right_states.jsonl"
            video_paths = {
                key: videos_dir / filename
                for key, filename in spec.video_files.items()
            }

            required_paths = [left_states_path, right_states_path, *video_paths.values()]
            if any(not path.is_file() for path in required_paths):
                logging.warning("Skipping incomplete %s episode: %s", spec.robot_type, episode_dir)
                continue

            n_states = min(_count_jsonl_lines(left_states_path), _count_jsonl_lines(right_states_path))
            sample_count = _sample_count_from_states(n_states, frame_interval)
            if sample_count <= 0:
                continue

            episodes.append(
                RoboChallengeRawEpisode(
                    task_name=task_dir.name,
                    prompt=prompt,
                    episode_dir=episode_dir,
                    left_states_path=left_states_path,
                    right_states_path=right_states_path,
                    video_paths=video_paths,
                    n_states=n_states,
                    sample_start=sample_cursor,
                    sample_end=sample_cursor + sample_count,
                    frame_interval=frame_interval,
                )
            )
            sample_cursor += sample_count

    if not episodes:
        raise FileNotFoundError(
            f"No RoboChallenge {spec.robot_type} raw episodes found under {raw_root}"
            + (f" with task_regex={task_regex!r}" if task_regex else "")
        )
    if task_names:
        found_tasks = {record.task_name for record in episodes}
        missing_tasks = sorted(set(task_names) - found_tasks)
        if missing_tasks:
            message = f"Missing RoboChallenge {spec.robot_type} task(s): {', '.join(missing_tasks)}"
            if task_regex:
                logging.warning(message)
            else:
                raise FileNotFoundError(message)

    return episodes


def _state_vector_from_row(row: dict[str, Any]) -> np.ndarray:
    joints = np.asarray(row["joint_positions"], dtype=np.float32).reshape(-1)
    if joints.shape[0] < 6:
        raise ValueError(f"Expected at least 6 joint_positions values, got {joints.shape[0]}")
    gripper = np.asarray(row["gripper_width"], dtype=np.float32).reshape(-1)
    if gripper.shape[0] < 1:
        raise ValueError("Missing gripper_width value")
    return np.concatenate([joints[:6], gripper[:1]], axis=0).astype(np.float32, copy=False)


def _load_arm_state_vectors(path: Path, expected_count: int) -> np.ndarray:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= expected_count:
                break
            if line.strip():
                rows.append(_state_vector_from_row(json.loads(line)))
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} states in {path}, got {len(rows)}")
    return np.stack(rows, axis=0)


def _episode_cache_stem(record: RoboChallengeRawEpisode) -> str:
    key = str(record.episode_dir.resolve())
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"{record.task_name}_{record.episode_dir.name}_{digest}"


def build_robochallenge_w1_state_array(record: RoboChallengeRawEpisode) -> np.ndarray:
    left = _load_arm_state_vectors(record.left_states_path, record.n_states)
    right = _load_arm_state_vectors(record.right_states_path, record.n_states)
    return np.concatenate([left, right], axis=-1).astype(np.float32, copy=False)


def load_robochallenge_w1_state_array(
    record: RoboChallengeRawEpisode,
    cache_dir: str | Path | None,
) -> np.ndarray:
    if cache_dir is None:
        return build_robochallenge_w1_state_array(record)

    cache_dir = Path(cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{_episode_cache_stem(record)}.npy"

    if cache_path.is_file():
        return np.load(cache_path, mmap_mode="r")

    try:
        array = build_robochallenge_w1_state_array(record)
        with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".npy", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            np.save(tmp_path, array)
            os.replace(tmp_path, cache_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
    except Exception:
        if cache_path.is_file():
            return np.load(cache_path, mmap_mode="r")
        raise

    return np.load(cache_path, mmap_mode="r")


class RoboChallengeRawW1Dataset(Dataset):
    """Direct RoboChallenge dual-arm raw dataset for WSABase training."""

    def __init__(
        self,
        *,
        raw_root: str | Path,
        external_stats_path: str | Path,
        transforms: Sequence[DataTransformFn],
        chunk_size: int,
        image_delta_indices: Sequence[int] | None,
        image_transforms=None,
        embodiment: str = ROBOCHALLENGE_W1_ROBOT_TYPE,
        frame_interval: int = 1,
        task_regex: str | None = None,
        task_names: Sequence[str] | None = None,
        task_sampling_weights: dict[str, float] | None = None,
        task_sampling_mode: str = ROBOCHALLENGE_W1_TASK_SAMPLING_PER_TASK,
        task_sampling_gamma: float = 1.0,
        regular_task_total_weight: float | None = None,
        extra_task_total_weight: float | None = None,
        state_cache_dir: str | Path | None = None,
        state_cache_size: int = 32,
        validate_videos: bool = False,
    ) -> None:
        super().__init__()
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        self.spec = get_robochallenge_raw_spec(embodiment=embodiment, task_preset=None)
        self.camera_keys = self.spec.camera_keys
        self.raw_root = Path(raw_root).expanduser()
        self.chunk_size = int(chunk_size)
        self.image_delta_indices = tuple(image_delta_indices or (-15, 0, 15))
        self.image_transforms = image_transforms
        self.frame_interval = int(frame_interval)
        self.state_cache_dir = Path(state_cache_dir).expanduser() if state_cache_dir else None
        self.state_cache_size = int(max(state_cache_size, 1))
        self._state_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._video_cache: OrderedDict[str, Any] = OrderedDict()

        self.episodes = discover_robochallenge_w1_episodes(
            self.raw_root,
            embodiment=self.spec.robot_type,
            frame_interval=self.frame_interval,
            task_regex=task_regex,
            task_names=task_names,
        )
        self._episode_starts = [record.sample_start for record in self.episodes]
        self._base_num_frames = self.episodes[-1].sample_end
        self._episode_indices_by_id = {id(record): idx for idx, record in enumerate(self.episodes)}
        self._task_records, self._task_episode_starts, self._task_sample_counts = self._build_task_index()
        (
            self._weighted_task_names,
            self._weighted_task_lengths,
            self._weighted_task_cum_lengths,
        ) = self._build_weighted_task_plan(
            task_sampling_weights,
            task_sampling_mode=task_sampling_mode,
            task_sampling_gamma=task_sampling_gamma,
            regular_task_total_weight=regular_task_total_weight,
            extra_task_total_weight=extra_task_total_weight,
        )
        self._use_weighted_task_index = self._weighted_task_names is not None
        self.num_frames = (
            self._weighted_task_cum_lengths[-1] if self._use_weighted_task_index else self._base_num_frames
        )
        self.num_episodes = len(self.episodes)
        self.dataset_weights = None
        self.fps = 30
        self.repo_id = f"robochallenge_raw_{self.spec.key}"

        external_stats_path = Path(external_stats_path).expanduser()
        if not external_stats_path.is_file():
            raise FileNotFoundError(f"Missing RoboChallenge {self.spec.robot_type} raw stats: {external_stats_path}")
        stats = cast_stats_to_numpy(load_json(external_stats_path))

        image_height, image_width = self._probe_image_shape()
        self.meta = SimpleNamespace(
            robot_type=self.spec.robot_type,
            features=self._build_features(image_height=image_height, image_width=image_width),
            video_keys=list(self.camera_keys),
            image_keys=[],
            camera_keys=list(self.camera_keys),
            stats=stats,
            episodes={
                "dataset_from_index": [record.sample_start for record in self.episodes],
                "dataset_to_index": [record.sample_end for record in self.episodes],
            },
            total_frames=self.num_frames,
            total_episodes=self.num_episodes,
        )
        self.features = self.meta.features
        self.dataset_stats = stats

        if validate_videos:
            self._validate_first_video_frames()

        hydrated = list(transforms)
        hydrated = hydrate_normalize_transform(hydrated, self)
        hydrated = hydrate_compose_field_transform(hydrated, self)
        hydrated = hydrate_delta_action_transform(hydrated, self)
        hydrated = hydrate_remap_image_key_transform(hydrated, self)
        filter_image_features(self)
        self._transform = compose(hydrated)

    def _build_task_index(
        self,
    ) -> tuple[OrderedDict[str, list[RoboChallengeRawEpisode]], dict[str, list[int]], dict[str, int]]:
        task_records: OrderedDict[str, list[RoboChallengeRawEpisode]] = OrderedDict()
        for record in self.episodes:
            task_records.setdefault(record.task_name, []).append(record)

        task_episode_starts: dict[str, list[int]] = {}
        task_sample_counts: dict[str, int] = {}
        for task_name, records in task_records.items():
            cursor = 0
            starts = []
            for record in records:
                starts.append(cursor)
                cursor += record.sample_count
            task_episode_starts[task_name] = starts
            task_sample_counts[task_name] = cursor

        return task_records, task_episode_starts, task_sample_counts

    def _build_weighted_task_plan(
        self,
        task_sampling_weights: dict[str, float] | None,
        *,
        task_sampling_mode: str,
        task_sampling_gamma: float,
        regular_task_total_weight: float | None,
        extra_task_total_weight: float | None,
    ) -> tuple[list[str] | None, list[int] | None, list[int] | None]:
        task_sampling_mode = str(task_sampling_mode or ROBOCHALLENGE_W1_TASK_SAMPLING_NONE).strip().lower()
        if task_sampling_mode in {"", ROBOCHALLENGE_W1_TASK_SAMPLING_NONE}:
            return None, None, None
        if task_sampling_mode == ROBOCHALLENGE_W1_TASK_SAMPLING_GROUP_FRAMES_POW:
            spec = get_robochallenge_raw_spec(embodiment=getattr(self, "spec", ROBOCHALLENGE_RAW_W1_SPEC).robot_type)
            task_sampling_weights = _table30v2_group_frame_pow_task_weights(
                self._task_sample_counts,
                regular_tasks=spec.regular_tasks,
                extra_tasks=spec.extra_tasks,
                regular_total_weight=(
                    float(regular_task_total_weight)
                    if regular_task_total_weight is not None
                    else spec.regular_total_weight
                ),
                extra_total_weight=(
                    float(extra_task_total_weight)
                    if extra_task_total_weight is not None
                    else spec.extra_total_weight
                ),
                gamma=float(task_sampling_gamma),
            )
        elif task_sampling_mode == ROBOCHALLENGE_W1_TASK_SAMPLING_PER_TASK:
            if not task_sampling_weights:
                return None, None, None
        else:
            raise ValueError(f"Unknown RoboChallenge raw task_sampling_mode={task_sampling_mode!r}")

        present = [
            (task_name, float(task_sampling_weights.get(task_name, 0.0)))
            for task_name in self._task_records
            if float(task_sampling_weights.get(task_name, 0.0)) > 0.0
        ]
        if not present:
            raise ValueError(
                f"task_sampling_weights did not match any discovered RoboChallenge {self.spec.robot_type} task."
            )

        total_real_samples = sum(self._task_sample_counts[task_name] for task_name, _ in present)
        total_weight = sum(weight for _, weight in present)
        raw_lengths = [total_real_samples * weight / total_weight for _, weight in present]
        virtual_lengths = [max(1, int(round(length))) for length in raw_lengths]

        diff = total_real_samples - sum(virtual_lengths)
        if diff != 0:
            fractions = [length - int(length) for length in raw_lengths]
            order = sorted(
                range(len(virtual_lengths)),
                key=lambda idx: fractions[idx],
                reverse=diff > 0,
            )
            step = 1 if diff > 0 else -1
            remaining = abs(diff)
            cursor = 0
            while remaining > 0:
                idx = order[cursor % len(order)]
                if step > 0 or virtual_lengths[idx] > 1:
                    virtual_lengths[idx] += step
                    remaining -= 1
                cursor += 1

        task_names = [task_name for task_name, _ in present]
        cum_lengths = []
        cursor = 0
        for length in virtual_lengths:
            cursor += length
            cum_lengths.append(cursor)

        logging.info(
            "RoboChallenge %s task-weighted sampling enabled: mode=%s gamma=%s plan=%s",
            self.spec.robot_type,
            task_sampling_mode,
            task_sampling_gamma,
            {
                task_name: {
                    "weight": float(task_sampling_weights[task_name]),
                    "real_samples": self._task_sample_counts[task_name],
                    "virtual_samples": virtual_lengths[idx],
                }
                for idx, task_name in enumerate(task_names)
            },
        )

        return task_names, virtual_lengths, cum_lengths

    def _build_features(self, *, image_height: int, image_width: int) -> dict[str, dict[str, Any]]:
        features: dict[str, dict[str, Any]] = {
            OBS_STATE: {
                "dtype": "float32",
                "shape": [self.spec.state_dim],
                "names": ["state"],
            },
            ACTION: {
                "dtype": "float32",
                "shape": [self.spec.state_dim],
                "names": ["action"],
            },
        }
        for key in self.camera_keys:
            features[key] = {
                "dtype": "video",
                "shape": [3, int(image_height), int(image_width)],
                "names": ["channel", "height", "width"],
            }
        return features

    def _probe_image_shape(self) -> tuple[int, int]:
        first_video = self.episodes[0].video_paths[self.camera_keys[0]]
        cv2 = self._cv2()
        cap = self._open_temporary_capture(first_video)
        try:
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 448
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 448
        finally:
            cap.release()
        return height, width

    def _validate_first_video_frames(self) -> None:
        cv2 = self._cv2()
        for record in self.episodes[: min(len(self.episodes), 8)]:
            for path in record.video_paths.values():
                cap = self._open_temporary_capture(path)
                try:
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if frame_count and frame_count < record.n_states:
                        raise ValueError(
                            f"Video has fewer frames than states: {path} frames={frame_count}, states={record.n_states}"
                        )
                finally:
                    cap.release()

    @staticmethod
    def _cv2():
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("RoboChallenge raw training requires opencv-python (`cv2`).") from exc
        return cv2

    def _open_temporary_capture(self, path: Path):
        cv2 = self._cv2()
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")
        return cap

    def _open_capture(self, path: Path):
        cv2 = self._cv2()
        key = str(path)
        cap = self._video_cache.get(key)
        if cap is not None:
            self._video_cache.move_to_end(key)
            return cap

        cap = cv2.VideoCapture(key)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")
        self._video_cache[key] = cap
        while len(self._video_cache) > 12:
            _, old_cap = self._video_cache.popitem(last=False)
            old_cap.release()
        return cap

    def _state_array(self, episode_idx: int) -> np.ndarray:
        array = self._state_cache.get(episode_idx)
        if array is not None:
            self._state_cache.move_to_end(episode_idx)
            return array

        array = load_robochallenge_w1_state_array(self.episodes[episode_idx], self.state_cache_dir)
        self._state_cache[episode_idx] = array
        while len(self._state_cache) > self.state_cache_size:
            self._state_cache.popitem(last=False)
        return array

    def _locate_episode(self, idx: int) -> tuple[int, RoboChallengeRawEpisode, int]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")
        if self._use_weighted_task_index:
            return self._locate_weighted_task_episode(idx)

        episode_idx = bisect.bisect_right(self._episode_starts, idx) - 1
        record = self.episodes[episode_idx]
        return episode_idx, record, idx - record.sample_start

    def _locate_weighted_task_episode(self, idx: int) -> tuple[int, RoboChallengeRawEpisode, int]:
        task_pos = bisect.bisect_right(self._weighted_task_cum_lengths, idx)
        task_start = 0 if task_pos == 0 else self._weighted_task_cum_lengths[task_pos - 1]
        task_name = self._weighted_task_names[task_pos]
        task_real_count = self._task_sample_counts[task_name]
        task_virtual_count = self._weighted_task_lengths[task_pos]
        task_offset = idx - task_start
        task_local_idx = min(task_real_count - 1, int(task_offset * task_real_count / task_virtual_count))

        episode_starts = self._task_episode_starts[task_name]
        record_pos = bisect.bisect_right(episode_starts, task_local_idx) - 1
        record = self._task_records[task_name][record_pos]
        local_idx = task_local_idx - episode_starts[record_pos]
        return self._episode_indices_by_id[id(record)], record, local_idx

    def _raw_current_index(self, local_idx: int) -> int:
        return int(local_idx) * self.frame_interval

    def _raw_target_index(self, local_idx: int) -> int:
        return (int(local_idx) + 1) * self.frame_interval

    def _read_video_frame(self, path: Path, frame_idx: int) -> torch.Tensor:
        cv2 = self._cv2()
        cap = self._open_capture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            cap.release()
            self._video_cache.pop(str(path), None)
            cap = self._open_capture(path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to decode frame {frame_idx} from {path}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(frame).permute(2, 0, 1).contiguous().to(torch.float32) / 255.0

    def _read_camera_clip(self, record: RoboChallengeRawEpisode, camera_key: str, current_raw_idx: int) -> torch.Tensor:
        frames = []
        for delta in self.image_delta_indices:
            frame_idx = max(0, min(record.n_states - 1, current_raw_idx + int(delta)))
            frames.append(self._read_video_frame(record.video_paths[camera_key], frame_idx))
        return torch.stack(frames, dim=0)

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> dict[str, Any]:
        episode_idx, record, local_idx = self._locate_episode(idx)
        states = self._state_array(episode_idx)

        current_raw_idx = self._raw_current_index(local_idx)
        current_state = torch.from_numpy(np.asarray(states[current_raw_idx], dtype=np.float32).copy())

        action_rows = []
        for offset in range(self.chunk_size):
            action_local_idx = min(local_idx + offset, record.sample_count - 1)
            target_raw_idx = self._raw_target_index(action_local_idx)
            action_rows.append(torch.from_numpy(np.asarray(states[target_raw_idx], dtype=np.float32).copy()))
        action = torch.stack(action_rows, dim=0)

        item: dict[str, Any] = {
            OBS_STATE: current_state,
            ACTION: action,
            "task": record.prompt,
            "robot_type": self.meta.robot_type,
        }
        for camera_key in self.camera_keys:
            item[camera_key] = self._read_camera_clip(record, camera_key, current_raw_idx)

        if self.image_transforms is not None:
            for camera_key in self.camera_keys:
                if hasattr(self.image_transforms, "set_current_key"):
                    self.image_transforms.set_current_key(camera_key)
                item[camera_key] = self.image_transforms(item[camera_key])

        return self._transform(item)

    def __getstate__(self):
        state = self.__dict__.copy()
        for cap in state.get("_video_cache", {}).values():
            cap.release()
        state["_video_cache"] = OrderedDict()
        state["_state_cache"] = OrderedDict()
        return state

    def __del__(self):
        for cap in getattr(self, "_video_cache", {}).values():
            cap.release()


class RoboChallengeRawAlohaDataset(RoboChallengeRawW1Dataset):
    """ALOHA-named wrapper over the shared dual-arm RoboChallenge raw loader."""

    def __init__(self, *args, embodiment: str = ROBOCHALLENGE_ALOHA_ROBOT_TYPE, **kwargs) -> None:
        super().__init__(*args, embodiment=embodiment, **kwargs)
