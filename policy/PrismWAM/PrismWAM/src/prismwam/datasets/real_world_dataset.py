"""LeRobot-v3 real-robot RGB/depth/action samples for PrismWAM.

Each task is one LeRobot-v3 dataset folder named after the task. A task uses one
fixed camera, or two cameras composed side by side; either way the caller sees
the same aligned sample format. Depth is read from per-frame uint16 millimetre
PNGs under ``<task>/<depth_namespace>/``.
"""

from __future__ import annotations
import bisect
import hashlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence
import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from prismwam.rgb_history import RGBHistory
from .robotwin_instructions import select_instruction_candidate

# Hashed into the action-statistics signature; kept verbatim so that existing
# statistics files stay valid.
ACTION_STATS_SIGNATURE_VERSION = "real_world_rda_joint_cache_v1"
DEFAULT_REAL_WORLD_DEPTH_NAMESPACE = "depth_refined_realcal_v2"
REAL_WORLD_DEPTH_NAMESPACES = frozenset(
    {DEFAULT_REAL_WORLD_DEPTH_NAMESPACE, "depth_lingbot_smooth_v05"}
)
PROMPT_TEMPLATE = "A video recorded from a robot's point of view executing the following instruction: {task}"


@dataclass(frozen=True)
class RealWorldViewSpec:
    name: str
    rgb_key: str
    depth_key: str


@dataclass(frozen=True)
class RealWorldTaskSpec:
    """One recorded task; its LeRobot dataset folder is ``<root_dir>/<name>``."""

    name: str
    instruction: str
    layout: str
    views: tuple[RealWorldViewSpec, ...]
    action_dim: int


REAL_WORLD_TASK_SPECS: dict[str, RealWorldTaskSpec] = {
    "rgb": RealWorldTaskSpec(
        name="rgb",
        instruction="Arrange the cubes in one row: blue farthest, green middle, red nearest the robot.",
        layout="single",
        views=(
            RealWorldViewSpec(
                "door", "observation.images.door", "observation.depths.door"
            ),
        ),
        action_dim=7,
    ),
    "insert_test_tube": RealWorldTaskSpec(
        name="insert_test_tube",
        instruction="Place the blue-capped test tube upright in the rack's rightmost hole.",
        layout="single",
        views=(
            RealWorldViewSpec(
                "middle_cam",
                "observation.images.middle_cam",
                "observation.depths.middle_cam",
            ),
        ),
        action_dim=14,
    ),
    "reorder_cubes": RealWorldTaskSpec(
        name="reorder_cubes",
        instruction="Reorder the cubes.",
        layout="single",
        views=(
            RealWorldViewSpec(
                "door", "observation.images.door", "observation.depths.door"
            ),
        ),
        action_dim=7,
    ),
    "pickup_plate": RealWorldTaskSpec(
        name="pickup_plate",
        instruction="Pick up the plate.",
        layout="single",
        views=(
            RealWorldViewSpec(
                "left_cam_3",
                "observation.images.left_cam_3",
                "observation.depth.left_cam_3",
            ),
        ),
        action_dim=14,
    ),
    "grab_plate": RealWorldTaskSpec(
        name="grab_plate",
        instruction="Grab the thicker plate and put it onto the rack behind the thinner plate",
        layout="single",
        views=(
            RealWorldViewSpec(
                "door", "observation.images.door", "observation.depths.door"
            ),
        ),
        action_dim=7,
    ),
    "move_plate": RealWorldTaskSpec(
        name="move_plate",
        instruction="Move the plate from one rack to another",
        layout="single",
        views=(
            RealWorldViewSpec(
                "door", "observation.images.door", "observation.depths.door"
            ),
        ),
        action_dim=7,
    ),
    "usb": RealWorldTaskSpec(
        name="usb",
        instruction="Pick up the two USB cables and insert them into the USB port on the outlet.",
        layout="horizontal",
        views=(
            RealWorldViewSpec(
                "front", "observation.images.front", "observation.depths.front"
            ),
            RealWorldViewSpec(
                "front_2", "observation.images.front_2", "observation.depths.front_2"
            ),
        ),
        action_dim=7,
    ),
    "pnp": RealWorldTaskSpec(
        name="pnp",
        instruction="Pick up the corn, move it over the obstacle, and place it into the white container.",
        layout="single",
        views=(
            RealWorldViewSpec(
                "door", "observation.images.door", "observation.depths.door"
            ),
        ),
        action_dim=7,
    ),
    "football": RealWorldTaskSpec(
        name="football",
        instruction="Dribble through the lane and score a goal.",
        layout="single",
        views=(
            RealWorldViewSpec(
                "door", "observation.images.door", "observation.depths.door"
            ),
        ),
        action_dim=7,
    ),
    "microwave": RealWorldTaskSpec(
        name="microwave",
        instruction="Open the microwave and take the basket out.",
        layout="single",
        views=(
            RealWorldViewSpec(
                "left_cam_3",
                "observation.images.left_cam_3",
                "observation.depth.left_cam_3",
            ),
        ),
        action_dim=14,
    ),
}


@dataclass(frozen=True)
class RealWorldEpisode:
    task: str
    instruction: str
    layout: str
    views: tuple[RealWorldViewSpec, ...]
    episode_index: int
    length: int
    fps: float
    data_path: Path
    rgb_paths: tuple[Path, ...]
    rgb_start_indices: tuple[int, ...]
    depth_roots: tuple[Path, ...]
    num_samples: int
    sample_start_index: int = 0
    valid_start: int = 0
    valid_end: int | None = None
    instruction_candidates: tuple[str, ...] = ()


def _real_world_instruction_types(instruction_type: str) -> tuple[str, ...]:
    value = str(instruction_type).strip().lower()
    if value in {"all", "both"}:
        return ("seen", "unseen")
    if value not in {"seen", "unseen"}:
        raise ValueError(
            f"Unsupported instruction_type={instruction_type!r}; expected seen, unseen, or all."
        )
    return (value,)


def load_real_world_instruction_candidates(
    *,
    task_name: str,
    fallback_instruction: str,
    task_instruction_dir: str | Path | None,
    instruction_type: str = "seen",
) -> tuple[str, ...]:
    """Load one task's frozen deterministic instruction paraphrases."""
    kinds = _real_world_instruction_types(instruction_type)
    if task_instruction_dir is None:
        if kinds != ("seen",):
            raise ValueError(
                "unseen real-world instructions require task_instruction_dir."
            )
        fallback = " ".join(str(fallback_instruction).split())
        if not fallback:
            raise ValueError(f"Task {task_name!r} has an empty fallback instruction.")
        return (fallback,)
    directory = Path(task_instruction_dir).expanduser().resolve()
    path = directory / f"{task_name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Real-world task instruction file not found: {path}.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates: list[str] = []
    for kind in kinds:
        values = payload.get(kind)
        if not isinstance(values, list) or not values:
            raise ValueError(f"{path} has no non-empty {kind!r} instruction list.")
        for value in values:
            instruction = " ".join(str(value).split())
            if not instruction:
                raise ValueError(f"{path} contains an empty {kind!r} instruction.")
            candidates.append(instruction)
    if len(set(candidates)) != len(candidates):
        raise ValueError(f"{path} contains duplicate selected instructions.")
    return tuple(candidates)


def episode_belongs_to_split(
    episode_index: int,
    *,
    split: str,
    validation_modulus: int = 10,
    validation_remainder: int = 0,
) -> bool:
    """Return the deterministic episode-level train/validation assignment."""
    split = str(split).strip().lower()
    validation_modulus = int(validation_modulus)
    validation_remainder = int(validation_remainder)
    if split not in {"train", "val", "all"}:
        raise ValueError(f"split must be train, val, or all; got {split!r}.")
    if validation_modulus == 0 and validation_remainder == 0:
        return split != "val"
    if validation_modulus <= 1:
        raise ValueError(
            "validation_modulus must be zero (no validation) or greater than one."
        )
    if not 0 <= validation_remainder < validation_modulus:
        raise ValueError("validation_remainder must be in [0, validation_modulus).")
    if split == "all":
        return True
    is_validation = int(episode_index) % validation_modulus == validation_remainder
    return is_validation if split == "val" else not is_validation


def num_real_world_samples(
    *,
    length: int,
    sample_stride: int,
    window_span: int,
    action_start_offset: int,
    drop_incomplete_windows: bool,
) -> int:
    """Count anchors while excluding an all-padding final action sample."""
    length = int(length)
    sample_stride = int(sample_stride)
    window_span = int(window_span)
    action_start_offset = int(action_start_offset)
    if min(sample_stride, window_span) <= 0:
        raise ValueError("sample_stride and window_span must be positive.")
    if length <= 0:
        return 0
    action_span = max(length - action_start_offset, 0)
    action_samples = (action_span + sample_stride - 1) // sample_stride
    if not drop_incomplete_windows:
        return action_samples
    if length < window_span:
        return 0
    video_samples = (length - window_span) // sample_stride + 1
    return min(video_samples, action_samples)


def build_vector_window(
    values: np.ndarray,
    *,
    start: int,
    start_offset: int,
    horizon: int,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read, last-pad, and normalize one action or proprio window."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"values must be [T,D], got {values.shape}.")
    horizon = int(horizon)
    read_start = int(start) + int(start_offset)
    if horizon <= 0 or values.shape[0] <= 0:
        raise ValueError("horizon and source sequence length must be positive.")
    if tuple(mean.shape) != (values.shape[1],) or tuple(std.shape) != (
        values.shape[1],
    ):
        raise ValueError("normalizer tensors do not match the vector dimension.")
    if read_start < values.shape[0]:
        window = values[read_start : min(read_start + horizon, values.shape[0])]
    else:
        window = values[-1:]
    is_pad = torch.tensor(
        [read_start + offset >= values.shape[0] for offset in range(horizon)],
        dtype=torch.bool,
    )
    if window.shape[0] < horizon:
        window = np.concatenate(
            [window, np.repeat(window[-1:], horizon - window.shape[0], axis=0)], axis=0
        )
    tensor = torch.from_numpy(window[:horizon]).contiguous()
    return ((tensor - mean) / std, is_pad)


def compose_real_world_views(
    views: Sequence[np.ndarray],
    *,
    layout: str,
    size: tuple[int, int],
    interpolation: str = "linear",
) -> np.ndarray:
    """Compose one or two aligned views into a fixed ``(height, width)`` canvas."""
    if not views:
        raise ValueError("At least one view is required.")
    (height, width) = (int(size[0]), int(size[1]))
    if height <= 0 or width <= 0:
        raise ValueError("Canvas dimensions must be positive.")
    cv_interpolation = {
        "linear": cv2.INTER_LINEAR,
        "nearest": cv2.INTER_NEAREST,
        "area": cv2.INTER_AREA,
    }.get(str(interpolation).lower())
    if cv_interpolation is None:
        raise ValueError(f"Unsupported interpolation {interpolation!r}.")
    layout = str(layout).strip().lower()
    if layout == "single":
        if len(views) != 1:
            raise ValueError("single layout requires exactly one view.")
        return cv2.resize(views[0], (width, height), interpolation=cv_interpolation)
    if layout != "horizontal":
        raise ValueError(f"Unsupported real-world layout {layout!r}.")
    widths = [width // len(views)] * len(views)
    widths[-1] += width - sum(widths)
    resized = [
        cv2.resize(view, (view_width, height), interpolation=cv_interpolation)
        for (view, view_width) in zip(views, widths)
    ]
    return np.concatenate(resized, axis=1)


def _episode_rows(dataset_root: Path, columns: Sequence[str]) -> list[dict[str, Any]]:
    paths = sorted((dataset_root / "meta" / "episodes").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode metadata below {dataset_root}.")
    rows: list[dict[str, Any]] = []
    for path in paths:
        schema = set(pq.read_schema(path).names)
        missing = set(columns) - schema
        if missing:
            raise KeyError(f"{path} is missing columns {sorted(missing)}.")
        rows.extend(pq.read_table(path, columns=list(columns)).to_pylist())
    rows.sort(key=lambda row: int(row["episode_index"]))
    return rows


class RealWorldRGBDepthActionDataset(torch.utils.data.Dataset):
    """Aligned real-world RGB/depth/action samples for one or more tasks."""

    def __init__(
        self,
        root_dir: str,
        task_names: Sequence[str],
        split: str = "train",
        validation_modulus: int = 10,
        validation_remainder: int = 0,
        num_video_frames: int = 9,
        frame_stride: int = 4,
        sample_stride: int = 1,
        drop_incomplete_windows: bool = False,
        video_size: Sequence[int] = (384, 320),
        depth_namespace: str = DEFAULT_REAL_WORLD_DEPTH_NAMESPACE,
        depth_min: float = 0.0,
        depth_max: float = 1500.0,
        action_horizon: int = 32,
        action_start_offset: int = 1,
        proprio_start_offset: int = 0,
        action_dim: int = 7,
        normalize_actions: bool = True,
        action_std_floor: float = 1e-06,
        action_stats_cache_path: str | None = None,
        action_stats_cache_policy: str = "fixed",
        text_embedding_cache_dir: str | None = None,
        context_len: int = 128,
        text_encoder_id: str = "wan22ti2v5b",
        prompt_template: str = PROMPT_TEMPLATE,
        task_instruction_dir: str | None = None,
        instruction_type: str = "seen",
        instruction_selection: str = "episode_hash",
        max_episodes_per_task: int | None = None,
        require_refined_depth: bool = True,
        rgb_target_mode: str = "f8",
        rgb_history_num_frames: int = 1,
        rgb_history_frame_stride: int = 4,
        episode_frame_bounds_path: str | None = None,
        episode_filter_path: str | None = None,
        action_source: str = "action",
        view_names: Sequence[str] | None = None,
        rgb_path_overrides_path: str | None = None,
        preserve_root_alias: bool = False,
    ) -> None:
        root_path = Path(root_dir).expanduser()
        self.root_dir = (
            root_path.absolute() if preserve_root_alias else root_path.resolve()
        )
        self.task_names = tuple((str(name) for name in task_names))
        self.rgb_path_overrides = (
            json.loads(Path(rgb_path_overrides_path).read_text())
            if rgb_path_overrides_path
            else {}
        )
        if action_source not in {"action", "observation.state"}:
            raise ValueError(f"Unsupported action source: {action_source!r}")
        self.action_source = action_source
        self.view_names = None if view_names is None else tuple(view_names)
        if self.view_names is not None:
            if (
                len(self.task_names) != 1
                or not self.view_names
                or len(set(self.view_names)) != len(self.view_names)
            ):
                raise ValueError(
                    "View selection requires one task and unique nonempty view names."
                )
        unknown = set(self.task_names) - set(REAL_WORLD_TASK_SPECS)
        if unknown:
            raise ValueError(f"Unknown real-world tasks: {sorted(unknown)}.")
        self.episode_frame_bounds = None
        self.episode_frame_bounds_signature = None
        if episode_frame_bounds_path is not None:
            if len(self.task_names) != 1:
                raise ValueError("Episode frame bounds require exactly one task.")
            records = json.loads(Path(episode_frame_bounds_path).read_text())
            if not isinstance(records, list) or not records:
                raise ValueError("Episode frame bounds must be a non-empty list.")
            self.episode_frame_bounds = {}
            for record in records:
                fields = ("episode_index", "original_length", "start", "end_exclusive")
                if not isinstance(record, dict) or any(
                    (type(record.get(k)) is not int for k in fields)
                ):
                    raise ValueError(f"Invalid episode frame bounds: {record!r}")
                (index, length, start, end) = (record[k] for k in fields)
                if (
                    index < 0
                    or index in self.episode_frame_bounds
                    or (not 0 <= start < end <= length)
                ):
                    raise ValueError(
                        f"Invalid or duplicate episode frame bounds: {record!r}"
                    )
                self.episode_frame_bounds[index] = dict(
                    zip(fields, (index, length, start, end))
                )
            canonical = json.dumps(
                self.episode_frame_bounds, sort_keys=True, separators=(",", ":")
            )
            self.episode_frame_bounds_signature = hashlib.sha256(
                canonical.encode()
            ).hexdigest()
        self.split = str(split).strip().lower()
        self.validation_modulus = int(validation_modulus)
        self.validation_remainder = int(validation_remainder)
        self.num_video_frames = int(num_video_frames)
        self.frame_stride = int(frame_stride)
        self.sample_stride = int(sample_stride)
        self.drop_incomplete_windows = bool(
            drop_incomplete_windows or self.episode_frame_bounds is not None
        )
        self.video_size = (int(video_size[0]), int(video_size[1]))
        self.depth_namespace = str(depth_namespace).strip()
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.action_horizon = int(action_horizon)
        self.action_start_offset = int(action_start_offset)
        self.proprio_start_offset = int(proprio_start_offset)
        self.action_dim = int(action_dim)
        self.normalize_actions = bool(normalize_actions)
        self.action_std_floor = float(action_std_floor)
        self.action_stats_cache_path = (
            None
            if action_stats_cache_path is None
            else Path(action_stats_cache_path).expanduser().resolve()
        )
        self.action_stats_cache_policy = str(action_stats_cache_policy).lower()
        self.text_embedding_cache_dir = (
            None
            if text_embedding_cache_dir is None
            else Path(text_embedding_cache_dir).expanduser().resolve()
        )
        self.context_len = int(context_len)
        self.text_encoder_id = str(text_encoder_id)
        self.prompt_template = str(prompt_template)
        self.task_instruction_dir = (
            None
            if task_instruction_dir is None
            else Path(task_instruction_dir).expanduser().resolve()
        )
        if self.task_instruction_dir is not None and (
            not self.task_instruction_dir.is_dir()
        ):
            raise FileNotFoundError(
                f"Real-world task instruction directory not found: {self.task_instruction_dir}."
            )
        self.instruction_type = str(instruction_type).strip().lower()
        _real_world_instruction_types(self.instruction_type)
        self.instruction_selection = str(instruction_selection).strip().lower()
        if self.instruction_selection not in {
            "first",
            "hash",
            "episode_hash",
            "sample_hash",
            "window_hash",
        }:
            raise ValueError(
                "instruction_selection must be first, episode_hash, or sample_hash."
            )
        self.max_episodes_per_task = max_episodes_per_task
        self.require_refined_depth = bool(require_refined_depth)
        self.rgb_target_mode = str(rgb_target_mode)
        self.rgb_history_num_frames = int(rgb_history_num_frames)
        self.rgb_history_frame_stride = int(rgb_history_frame_stride)
        if self.rgb_history_num_frames < 1 or (self.rgb_history_num_frames - 1) % 4:
            raise ValueError("RGB history must have 4n+1 frames.")
        if self.rgb_history_frame_stride < 1:
            raise ValueError("RGB history stride must be positive.")
        self.rgb_history = RGBHistory.dense(
            self.rgb_history_num_frames, self.rgb_history_frame_stride
        )
        self.included_episode_indices = None
        if episode_filter_path is not None:
            if len(self.task_names) != 1:
                raise ValueError("An episode filter requires exactly one task.")
            selection = json.loads(Path(episode_filter_path).read_text())
            indices = selection["included_episode_indices"]
            if not indices or any((type(i) is not int or i < 0 for i in indices)):
                raise ValueError(
                    "Episode filter requires nonempty nonnegative integer indices."
                )
            if len(set(indices)) != len(indices):
                raise ValueError("Episode filter contains duplicate indices.")
            self.included_episode_indices = frozenset(indices)
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Real-world data root not found: {self.root_dir}.")
        if self.depth_namespace not in REAL_WORLD_DEPTH_NAMESPACES:
            raise ValueError(
                f"Unsupported real-world depth namespace {self.depth_namespace!r}; expected one of {sorted(REAL_WORLD_DEPTH_NAMESPACES)}."
            )
        if self.rgb_target_mode != "f8":
            raise ValueError("The real-world dataset supports rgb_target_mode=f8 only.")
        if self.num_video_frames <= 1 or self.num_video_frames % 4 != 1:
            raise ValueError(
                "num_video_frames must be greater than one and satisfy T % 4 == 1."
            )
        if min(self.frame_stride, self.sample_stride, self.action_horizon) <= 0:
            raise ValueError("frame/sample stride and action horizon must be positive.")
        task_action_dims = {
            REAL_WORLD_TASK_SPECS[name].action_dim for name in self.task_names
        }
        if len(task_action_dims) != 1:
            raise ValueError(
                f"Selected real-world tasks must share one action dimension; got {sorted(task_action_dims)} for {self.task_names}."
            )
        expected_action_dim = next(iter(task_action_dims))
        if self.action_dim != expected_action_dim:
            raise ValueError(
                f"Tasks {self.task_names} require joint{expected_action_dim}, got action_dim={self.action_dim}."
            )
        if self.depth_max <= self.depth_min:
            raise ValueError("depth_max must be greater than depth_min.")
        self.frame_offsets = tuple(
            (index * self.frame_stride for index in range(self.num_video_frames))
        )
        if min(self.action_start_offset, self.proprio_start_offset) < 0:
            raise ValueError("Action/proprio offsets must be non-negative.")
        self.window_span = max(
            self.frame_offsets[-1] + 1,
            self.action_start_offset + self.action_horizon,
            self.proprio_start_offset + self.action_horizon,
        )
        self.episodes: list[RealWorldEpisode] = []
        self.cumulative_sizes: list[int] = []
        self._build_index()
        self.action_mean = torch.zeros(self.action_dim, dtype=torch.float32)
        self.action_std = torch.ones(self.action_dim, dtype=torch.float32)
        if self.normalize_actions:
            (self.action_mean, self.action_std) = self._load_or_compute_action_stats()
        self._text_context_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def _build_index(self) -> None:
        total = 0
        for task_name in self.task_names:
            spec = REAL_WORLD_TASK_SPECS[task_name]
            instruction_candidates = load_real_world_instruction_candidates(
                task_name=spec.name,
                fallback_instruction=spec.instruction,
                task_instruction_dir=self.task_instruction_dir,
                instruction_type=self.instruction_type,
            )
            if self.view_names is not None:
                available = {view.name: view for view in spec.views}
                if not set(self.view_names) <= available.keys():
                    raise ValueError(
                        f"Unknown views for {task_name}: {self.view_names}"
                    )
                spec = replace(
                    spec,
                    views=tuple((available[name] for name in self.view_names)),
                    layout="single" if len(self.view_names) == 1 else "horizontal",
                )
            dataset_root = self.root_dir / spec.name
            info_path = dataset_root / "meta" / "info.json"
            if not info_path.is_file():
                raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}.")
            info = json.loads(info_path.read_text())
            fps = float(info["fps"])
            columns = ["episode_index", "length", "data/chunk_index", "data/file_index"]
            for view in spec.views:
                prefix = f"videos/{view.rgb_key}"
                columns.extend(
                    [
                        f"{prefix}/chunk_index",
                        f"{prefix}/file_index",
                        f"{prefix}/from_timestamp",
                    ]
                )
            if self.episode_frame_bounds is not None:
                columns.append("episode_success")
            rows = _episode_rows(dataset_root, columns)
            if self.episode_frame_bounds is not None:
                metadata = {int(row["episode_index"]): row for row in rows}
                if (
                    len(metadata) != len(rows)
                    or not self.episode_frame_bounds.keys() <= metadata.keys()
                ):
                    raise ValueError(
                        "Episode bounds do not match source episode identities."
                    )
                for index, bound in self.episode_frame_bounds.items():
                    if bound["original_length"] != int(metadata[index]["length"]):
                        raise ValueError(
                            f"Episode bounds source length mismatch: {index}"
                        )
                if self.episode_frame_bounds is not None:
                    if any(
                        (
                            row["episode_success"] not in {"success", "failure"}
                            for row in rows
                        )
                    ):
                        raise ValueError("Unknown episode_success label.")
                    successful = {
                        int(row["episode_index"])
                        for row in rows
                        if row["episode_success"] == "success"
                    }
                    if set(self.episode_frame_bounds) != successful:
                        raise ValueError(
                            "Bounds must cover exactly the successful episodes."
                        )
                rows = [
                    row
                    for row in rows
                    if int(row["episode_index"]) in self.episode_frame_bounds
                ]
            if self.included_episode_indices is not None:
                missing = self.included_episode_indices - {
                    int(row["episode_index"]) for row in rows
                }
                if missing:
                    raise ValueError(
                        f"Episode filter selects missing episodes: {sorted(missing)}."
                    )
                rows = [
                    row
                    for row in rows
                    if int(row["episode_index"]) in self.included_episode_indices
                ]
            if self.max_episodes_per_task is not None:
                rows = rows[: int(self.max_episodes_per_task)]
            for row in rows:
                episode_index = int(row["episode_index"])
                if not episode_belongs_to_split(
                    episode_index,
                    split=self.split,
                    validation_modulus=self.validation_modulus,
                    validation_remainder=self.validation_remainder,
                ):
                    continue
                length = int(row["length"])
                bound = (self.episode_frame_bounds or {}).get(episode_index)
                (valid_start, valid_end) = (
                    (bound["start"], bound["end_exclusive"]) if bound else (0, length)
                )
                num_samples = num_real_world_samples(
                    length=valid_end - valid_start,
                    sample_stride=self.sample_stride,
                    window_span=self.window_span,
                    action_start_offset=self.action_start_offset,
                    drop_incomplete_windows=self.drop_incomplete_windows,
                )
                if num_samples <= 0:
                    continue
                data_rel = info["data_path"].format(
                    chunk_index=int(row["data/chunk_index"]),
                    file_index=int(row["data/file_index"]),
                )
                data_path = dataset_root / data_rel
                sample_start_index = valid_start
                rgb_paths = []
                rgb_starts = []
                depth_roots = []
                for view in spec.views:
                    prefix = f"videos/{view.rgb_key}"
                    rgb_rel = info["video_path"].format(
                        video_key=view.rgb_key,
                        chunk_index=int(row[f"{prefix}/chunk_index"]),
                        file_index=int(row[f"{prefix}/file_index"]),
                    )
                    rgb_path = dataset_root / rgb_rel
                    replacement = self.rgb_path_overrides.get(str(rgb_path.resolve()))
                    rgb_paths.append(Path(replacement) if replacement else rgb_path)
                    rgb_starts.append(
                        int(round(float(row[f"{prefix}/from_timestamp"]) * fps))
                    )
                    depth_roots.append(
                        dataset_root
                        / self.depth_namespace
                        / view.depth_key
                        / f"episode_{episode_index:06d}"
                    )
                required = [data_path, *rgb_paths]
                if self.require_refined_depth:
                    required.extend(depth_roots)
                    required.extend(
                        (root / f"frame_{length - 1:06d}.png" for root in depth_roots)
                    )
                missing = [path for path in required if not path.exists()]
                if missing:
                    raise FileNotFoundError(
                        f"Episode {spec.name}/{episode_index} is incomplete: {missing[:4]}."
                    )
                self.episodes.append(
                    RealWorldEpisode(
                        task=spec.name,
                        instruction=spec.instruction,
                        layout=spec.layout,
                        views=spec.views,
                        episode_index=episode_index,
                        length=length,
                        fps=fps,
                        data_path=data_path,
                        rgb_paths=tuple(rgb_paths),
                        rgb_start_indices=tuple(rgb_starts),
                        depth_roots=tuple(depth_roots),
                        num_samples=num_samples,
                        sample_start_index=sample_start_index,
                        valid_start=valid_start,
                        valid_end=valid_end,
                        instruction_candidates=instruction_candidates,
                    )
                )
                total += num_samples
                self.cumulative_sizes.append(total)
        if not self.episodes:
            raise ValueError(
                f"No real-world episodes selected for tasks={self.task_names}, split={self.split}."
            )

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def _locate(self, idx: int) -> tuple[RealWorldEpisode, int]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} outside dataset length {len(self)}.")
        episode_position = bisect.bisect_right(self.cumulative_sizes, idx)
        previous = (
            0 if episode_position == 0 else self.cumulative_sizes[episode_position - 1]
        )
        local_index = idx - previous
        episode = self.episodes[episode_position]
        return (
            episode,
            episode.sample_start_index + int(local_index * self.sample_stride),
        )

    def task_index_groups(self) -> dict[str, list[int]]:
        groups: dict[str, list[int]] = {}
        previous = 0
        for episode, cumulative in zip(self.episodes, self.cumulative_sizes):
            groups.setdefault(episode.task, []).extend(range(previous, cumulative))
            previous = cumulative
        if previous != len(self):
            raise RuntimeError("Task index groups do not cover the dataset.")
        return groups

    def instruction_for_episode(self, episode: RealWorldEpisode, start: int = 0) -> str:
        candidates = episode.instruction_candidates or (episode.instruction,)
        return select_instruction_candidate(
            candidates,
            instruction_selection=self.instruction_selection,
            task_name=episode.task,
            phase=self.split,
            episode_index=episode.episode_index,
            start_index=int(start),
        )

    def prompt_for_episode(self, episode: RealWorldEpisode, start: int = 0) -> str:
        return self.prompt_template.format(
            task=self.instruction_for_episode(episode, start=start)
        )

    def prompts(self) -> list[str]:
        return sorted(
            {
                self.prompt_template.format(task=instruction)
                for episode in self.episodes
                for instruction in episode.instruction_candidates
                or (episode.instruction,)
            }
        )

    def load_episode_vectors(
        self, episode: RealWorldEpisode
    ) -> tuple[np.ndarray, np.ndarray]:
        table = pq.read_table(
            episode.data_path,
            columns=["episode_index", "frame_index", "action", "observation.state"],
        )
        rows = table.to_pydict()
        episode_indices = np.asarray(rows["episode_index"], dtype=np.int64)
        selected = np.flatnonzero(episode_indices == episode.episode_index)
        if selected.size != episode.length:
            raise ValueError(
                f"{episode.data_path} contains {selected.size} rows for episode {episode.episode_index}, expected {episode.length}."
            )
        frame_indices = np.asarray(rows["frame_index"], dtype=np.int64)[selected]
        order = np.argsort(frame_indices)
        if not np.array_equal(frame_indices[order], np.arange(episode.length)):
            raise ValueError(f"Non-contiguous frame indices in {episode.data_path}.")
        action = np.asarray(rows[self.action_source], dtype=np.float32)[selected][order]
        proprio = np.asarray(rows["observation.state"], dtype=np.float32)[selected][
            order
        ]
        expected = (episode.length, self.action_dim)
        if action.shape != expected or proprio.shape != expected:
            raise ValueError(
                f"Real-world vectors must be {expected}, got {action.shape}/{proprio.shape}."
            )
        return (action, proprio)

    def action_stats_signature(self) -> str:
        digest = hashlib.sha256()
        digest.update(ACTION_STATS_SIGNATURE_VERSION.encode("utf-8"))
        digest.update(str(self.root_dir).encode("utf-8"))
        digest.update(str(self.action_dim).encode("utf-8"))
        if self.action_source != "action":
            digest.update(self.action_source.encode("utf-8"))
        for task_name in self.task_names:
            digest.update(task_name.encode("utf-8"))
        digest.update(str(self.validation_modulus).encode("utf-8"))
        digest.update(str(self.validation_remainder).encode("utf-8"))
        if self.included_episode_indices is not None:
            digest.update(
                json.dumps(sorted(self.included_episode_indices)).encode("utf-8")
            )
        if self.episode_frame_bounds_signature is not None:
            digest.update(self.episode_frame_bounds_signature.encode())
        return digest.hexdigest()

    def _compute_action_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        count = 0
        total = np.zeros(self.action_dim, dtype=np.float64)
        total_sq = np.zeros(self.action_dim, dtype=np.float64)
        for episode in self.episodes:
            if not episode_belongs_to_split(
                episode.episode_index,
                split="train",
                validation_modulus=self.validation_modulus,
                validation_remainder=self.validation_remainder,
            ):
                continue
            (action, _) = self.load_episode_vectors(episode)
            action = action[episode.valid_start : episode.valid_end]
            action64 = action.astype(np.float64)
            total += action64.sum(axis=0)
            total_sq += np.square(action64).sum(axis=0)
            count += action.shape[0]
        if count <= 0:
            raise ValueError(
                "Cannot compute action statistics from an empty train split."
            )
        mean = total / count
        variance = np.maximum(total_sq / count - np.square(mean), 0.0)
        std = np.maximum(np.sqrt(variance), self.action_std_floor)
        return (
            torch.from_numpy(mean.astype(np.float32)),
            torch.from_numpy(std.astype(np.float32)),
        )

    def _load_or_compute_action_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.action_stats_cache_path
        if path is not None and path.is_file():
            payload = torch.load(path, map_location="cpu")
            (mean, std) = (payload.get("mean"), payload.get("std"))
            if payload.get("signature") != self.action_stats_signature():
                raise ValueError(f"Action stats signature mismatch: {path}.")
            if not isinstance(mean, torch.Tensor) or not isinstance(std, torch.Tensor):
                raise ValueError(f"Action stats lack mean/std tensors: {path}.")
            if tuple(mean.shape) != (self.action_dim,) or tuple(std.shape) != (
                self.action_dim,
            ):
                raise ValueError(f"Action stats shape mismatch: {path}.")
            if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all()):
                raise ValueError(f"Action stats contain non-finite values: {path}.")
            if not bool((std > 0).all()):
                raise ValueError(f"Action stats std must be positive: {path}.")
            return (mean.float(), std.float())
        if self.action_stats_cache_policy == "fixed":
            raise FileNotFoundError(f"Fixed action stats cache not found: {path}.")
        if self.action_stats_cache_policy != "compute":
            raise ValueError("action_stats_cache_policy must be fixed or compute.")
        (mean, std) = self._compute_action_stats()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
            torch.save(
                {
                    "signature": self.action_stats_signature(),
                    "mean": mean,
                    "std": std,
                    "action_dim": self.action_dim,
                    "tasks": list(self.task_names),
                    "split": "train",
                },
                temporary,
            )
            os.replace(temporary, path)
        return (mean, std)

    def _text_context(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.text_embedding_cache_dir is None:
            return None
        if prompt in self._text_context_cache:
            return self._text_context_cache[prompt]
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        path = (
            self.text_embedding_cache_dir
            / f"{digest}.t5_len{self.context_len}.{self.text_encoder_id}.pt"
        )
        if not path.is_file():
            raise FileNotFoundError(f"Missing real-world text embedding: {path}.")
        payload = torch.load(path, map_location="cpu")
        context = payload["context"].contiguous()
        mask = payload.get("mask", payload.get("context_mask"))
        if not isinstance(mask, torch.Tensor):
            raise ValueError(f"Text cache lacks a mask tensor: {path}.")
        value = (context, mask.bool().contiguous())
        self._text_context_cache[prompt] = value
        return value

    def _read_rgb_frame(
        self, episode: RealWorldEpisode, view_index: int, frame_index: int
    ) -> np.ndarray:
        capture = cv2.VideoCapture(str(episode.rgb_paths[view_index]))
        absolute = episode.rgb_start_indices[view_index] + int(frame_index)
        capture.set(cv2.CAP_PROP_POS_FRAMES, absolute)
        (ok, bgr) = capture.read()
        capture.release()
        if not ok or bgr is None:
            raise RuntimeError(
                f"Could not decode RGB frame {absolute} from {episode.rgb_paths[view_index]}."
            )
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _read_depth_frame(
        self, episode: RealWorldEpisode, view_index: int, frame_index: int
    ) -> np.ndarray:
        path = episode.depth_roots[view_index] / f"frame_{int(frame_index):06d}.png"
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None or depth.ndim != 2 or depth.dtype != np.uint16:
            raise RuntimeError(f"Invalid refined metric-depth frame: {path}.")
        return depth

    def _load_rgb_image(self, episode: RealWorldEpisode, start: int) -> torch.Tensor:
        """Decode and compose one RGB observation, normalized to [-1, 1]."""
        views = [
            self._read_rgb_frame(episode, view_index, int(start))
            for view_index in range(len(episode.views))
        ]
        composed = compose_real_world_views(
            views, layout=episode.layout, size=self.video_size, interpolation="area"
        )
        normalized = composed.astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(normalized).permute(2, 0, 1).contiguous()

    def rgb_sample_indices(self, episode: RealWorldEpisode, start: int) -> list[int]:
        history = self.rgb_history.indices(start, first=episode.sample_start_index)
        return history + [min(start + self.frame_offsets[-1], episode.length - 1)]

    def _load_rgb_video(self, episode: RealWorldEpisode, start: int) -> torch.Tensor:
        return torch.stack(
            [
                self._load_rgb_image(episode, index)
                for index in self.rgb_sample_indices(episode, start)
            ],
            dim=1,
        )

    def raw_sample(self, idx: int) -> dict[str, Any]:
        (episode, start) = self._locate(idx)
        (action_values, proprio_values) = self.load_episode_vectors(episode)
        depth_indices = [
            min(start + offset, episode.length - 1) for offset in self.frame_offsets
        ]
        depth_frames = []
        for frame_index in depth_indices:
            views = [
                self._read_depth_frame(episode, view_index, frame_index)
                for view_index in range(len(episode.views))
            ]
            depth_frames.append(
                compose_real_world_views(
                    views,
                    layout=episode.layout,
                    size=self.video_size,
                    interpolation="nearest",
                )
            )
        depth_array = np.stack(depth_frames).astype(np.float32)
        depth_array = np.clip(depth_array, self.depth_min, self.depth_max)
        depth_array = (depth_array - self.depth_min) / (
            self.depth_max - self.depth_min
        ) * 2.0 - 1.0
        (action, action_is_pad) = build_vector_window(
            action_values,
            start=start,
            start_offset=self.action_start_offset,
            horizon=self.action_horizon,
            mean=self.action_mean,
            std=self.action_std,
        )
        (proprio, proprio_is_pad) = build_vector_window(
            proprio_values,
            start=start,
            start_offset=self.proprio_start_offset,
            horizon=self.action_horizon,
            mean=self.action_mean,
            std=self.action_std,
        )
        return {
            "video": torch.from_numpy(depth_array)
            .unsqueeze(1)
            .permute(1, 0, 2, 3)
            .contiguous(),
            "rgb_video": self._load_rgb_video(episode, start),
            "action": action,
            "proprio": proprio,
            "image_is_pad": torch.tensor(
                [start + offset >= episode.length for offset in self.frame_offsets],
                dtype=torch.bool,
            ),
            "action_is_pad": action_is_pad,
            "proprio_is_pad": proprio_is_pad,
            "rgb_target_is_valid": torch.tensor(
                start + self.frame_offsets[-1] < episode.length, dtype=torch.bool
            ),
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        (episode, start) = self._locate(idx)
        sample = self.raw_sample(idx)
        prompt = self.prompt_for_episode(episode, start=start)
        context = self._text_context(prompt)
        if context is not None:
            (sample["context"], sample["context_mask"]) = context
        sample.update(
            prompt=prompt,
            idx=int(idx),
            task=episode.task,
            phase=self.split,
            episode_index=episode.episode_index,
            start_index=start,
            rgb_target_mode=self.rgb_target_mode,
        )
        return sample
