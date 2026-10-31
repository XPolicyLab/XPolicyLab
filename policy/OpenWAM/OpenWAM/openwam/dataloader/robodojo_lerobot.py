"""RoboDojo simulation episodes released as LeRobot v3.0, read into the same samples as the HDF5 reader.

XPolicyLab's official converter (``scripts/transform_lerobot_v30_format.py``)
keeps only the dual-X5 joint states, ``observation.state`` =
``[left arm joint1..6, left gripper, right arm joint1..6, right gripper]``,
and re-encodes the three cameras as video. The HDF5 reader's EEF20 comes from
the recorded ``state/*_ee_poses`` (``link6``, env-origin-relative position,
world wxyz), so this reader rebuilds those poses by forward kinematics of
``Assets/Robots/x5/X5A.urdf`` from the simulated arm roots, then runs the
HDF5 reader's own calibration, normalization, windowing and rendering.
Over round-1 RSI episodes the rebuilt poses match the recorded ones to about
3e-7 m and 1e-6 in rotation (float32 recording noise). Images are decoded
video frames, so they are close to, not equal to, the HDF5 JPEG frames.

``dataset_dir`` is one LeRobot v3 dataset or a list of them (e.g. one per
RSI round). Tasks come from the release's ``rsi_episodes.json`` (or the
``collection_quality.json`` of ``scripts/datagen/export.py``); a dataset with
neither is one task named after its directory. Samples are grouped by task in
sorted order like ``MultiTaskRoboDojoDataset``; within a task, episodes follow
the ``dataset_dir`` order, then ``episode_index``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from openwam.dataloader.robodojo import (
    DEPLOY_ACTION_MODE,
    MultiTaskRoboDojoDataset,
    RoboDojoDataset,
    _decode_instruction,
    _normalize_mode,
    calibrated_eef20,
)
from openwam.dataloader.robodojo_contract import (
    DUAL_X5_LEFT_BASE_POS,
    DUAL_X5_RIGHT_BASE_POS,
    ROBODOJO_EMBODIMENT,
    ROBODOJO_SIM_VARIANT,
    resolve_robodojo_calibration,
    validate_embodiment,
)
from openwam.dataloader.utils.lerobotv3 import load_episodes_parquet, parse_info_json
from openwam.dataloader.utils.poses import rot6d_to_quat_wxyz
from openwam.dataloader.utils.video_io import decode_video_frames

STATE_DIM = 14
STATE_NAMES = tuple(f"{arm}_joint_{index}" for arm in ("left", "right") for index in range(7))
_ARM_COLUMNS = {"left": slice(0, 6), "right": slice(7, 13)}
_GRIPPER_COLUMNS = {"left": slice(6, 7), "right": slice(13, 14)}
LEROBOT_CAMERAS = {
    "cam_head": "observation.images.cam_high",
    "cam_left_wrist": "observation.images.cam_left_wrist",
    "cam_right_wrist": "observation.images.cam_right_wrist",
}

# X5A.urdf joint1..joint6 (base_link -> link6): origin xyz, origin rpy, axis. The rpy values are the URDF's
# 3.1416, not pi; the simulator uses them as written.
X5_CHAIN = (
    ((0.0, 0.0, 0.0605), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.02, 0.0, 0.04), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ((-0.264, 0.0, 0.0), (3.1416, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ((0.245, 0.0, -0.056), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ((0.06775, 0.0005, -0.0865), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.02895, 0.0, 0.0865), (-3.1416, 0.0, 0.0), (1.0, 0.0, 0.0)),
)
# dual_x5.yml places both roots at quaternion (0.707, 0, 0, 0.707); Isaac keeps w and rescales the vector part to a
# unit quaternion, a yaw of 2 * acos(0.707) instead of pi / 2. The HDF5 reader's calibration still uses the
# normalized (0.707, 0, 0, 0.707), which is why the rebuilt poses must go through the simulated root, not it.
SIM_ROOT_QUAT_WXYZ = (0.707, 0.0, 0.0, math.sqrt(1.0 - 0.707**2))
SIM_ROOT_POS = {"left": DUAL_X5_LEFT_BASE_POS, "right": DUAL_X5_RIGHT_BASE_POS}


def _rotation(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rodrigues rotation matrices (..., 3, 3) about a fixed unit axis."""
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    angle = np.asarray(angle, dtype=np.float64)[..., None, None]
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def _rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return (
        _rotation(np.array([0.0, 0.0, 1.0]), yaw)
        @ _rotation(np.array([0.0, 1.0, 0.0]), pitch)
        @ _rotation(np.array([1.0, 0.0, 0.0]), roll)
    )


_CHAIN = tuple((np.asarray(xyz), _rpy(*rpy), np.asarray(axis)) for xyz, rpy, axis in X5_CHAIN)


def x5_link6_in_root(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``link6`` rotation (T, 3, 3) and position (T, 3) in the arm root frame for joint1..6 angles (T, 6)."""
    joints = np.asarray(joints, dtype=np.float64)
    if joints.ndim != 2 or joints.shape[1] != len(_CHAIN):
        raise ValueError(f"X5 joint angles must have shape (T, {len(_CHAIN)}), got {joints.shape}")
    rotation = np.broadcast_to(np.eye(3), (len(joints), 3, 3))
    position = np.zeros((len(joints), 3))
    for (offset, origin, axis), angle in zip(_CHAIN, joints.T):
        position = position + rotation @ offset
        rotation = rotation @ origin @ _rotation(axis, angle)
    return rotation, position


def _quat_to_matrix(wxyz: Sequence[float]) -> np.ndarray:
    w, x, y, z = wxyz
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


_SIM_ROOT_ROTATION = _quat_to_matrix(SIM_ROOT_QUAT_WXYZ)


def recorded_ee_poses(joints: np.ndarray, arm: str) -> np.ndarray:
    """Rebuild ``state/<arm>_ee_poses`` (T, 7): env-origin-relative xyz and world wxyz of ``link6``."""
    rotation, position = x5_link6_in_root(joints)
    world_rotation = _SIM_ROOT_ROTATION @ rotation
    world_position = position @ _SIM_ROOT_ROTATION.T + np.asarray(SIM_ROOT_POS[arm])
    rot6d = np.concatenate((world_rotation[..., 0], world_rotation[..., 1]), axis=-1)
    return np.concatenate((world_position, rot6d_to_quat_wxyz(rot6d)), axis=-1)


def lerobot_state_to_eef20(state: np.ndarray, calibration: Mapping[str, Any] | None = None) -> np.ndarray:
    """Calibrated EEF20 (T, 20) of converter ``observation.state`` rows (T, 14), as the HDF5 reader computes it."""
    state = np.asarray(state)
    if state.ndim != 2 or state.shape[1] != STATE_DIM:
        raise ValueError(f"RoboDojo LeRobot observation.state must have shape (T, {STATE_DIM}), got {state.shape}")
    return calibrated_eef20(
        recorded_ee_poses(state[:, _ARM_COLUMNS["left"]], "left"),
        state[:, _GRIPPER_COLUMNS["left"]],
        recorded_ee_poses(state[:, _ARM_COLUMNS["right"]], "right"),
        state[:, _GRIPPER_COLUMNS["right"]],
        resolve_robodojo_calibration(calibration),
    )


@dataclass(frozen=True)
class LeRobotEpisode:
    """One episode of a RoboDojo LeRobot v3 dataset: its state rows, prompt and video positions."""

    dataset_dir: str
    episode_index: int
    task: str
    instruction: str
    state: np.ndarray
    videos: Mapping[str, tuple[str, int]]  # RoboDojo camera -> (mp4 path, first frame of the episode)
    frame_size: Mapping[str, tuple[int, int]]  # RoboDojo camera -> (height, width)

    @property
    def name(self) -> str:
        return f"{self.dataset_dir}#episode_{self.episode_index:06d}"


def _feature_names(feature: Mapping[str, Any]) -> list[str]:
    names = feature.get("names")
    if isinstance(names, Mapping):
        names = next(iter(names.values()))
    if isinstance(names, list) and len(names) == 1 and isinstance(names[0], list):
        names = names[0]
    return [str(name) for name in names or []]


def _frame_size(feature: Mapping[str, Any], key: str) -> tuple[int, int]:
    shape, names = list(feature["shape"]), _feature_names(feature)
    if len(shape) != 3 or not {"height", "width"} <= set(names):
        raise ValueError(f"{key}: video feature needs a (channels, height, width) shape, got {shape} / {names}")
    return int(shape[names.index("height")]), int(shape[names.index("width")])


def _episode_tasks(dataset_dir: Path, count: int) -> list[str]:
    """Task of every episode from the release sidecars, else the directory name."""
    rsi = dataset_dir / "rsi_episodes.json"
    if rsi.is_file():
        tasks = {int(row["episode_index"]): str(row["task"]) for row in json.loads(rsi.read_text())}
        if sorted(tasks) != list(range(count)):
            raise ValueError(f"{rsi} lists episodes {sorted(tasks)[:5]}..., expected 0..{count - 1}")
        return [tasks[index] for index in range(count)]
    quality = dataset_dir / "collection_quality.json"
    if quality.is_file():
        rows = json.loads(quality.read_text())
        if len(rows) != count:
            raise ValueError(f"{quality} has {len(rows)} episodes, the dataset {count}")
        return [str(row["task"]) for row in rows]
    return [dataset_dir.name] * count


def _state_rows(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """observation.state (N, 14), episode_index and frame_index of every row, ordered by the global index."""
    paths = sorted((dataset_dir / "data").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no data parquet under {dataset_dir / 'data'}")
    columns = ["index", "episode_index", "frame_index", "observation.state"]
    table = pa.concat_tables([pq.read_table(path, columns=columns) for path in paths])
    order = np.argsort(table.column("index").to_numpy(), kind="stable")
    if not np.array_equal(table.column("index").to_numpy()[order], np.arange(len(order))):
        raise ValueError(f"{dataset_dir}: frame index is not 0..{len(order) - 1}")
    state = table.column("observation.state").combine_chunks()
    state = np.asarray(state.flatten(), dtype=np.float32).reshape(len(state), -1)[order]
    episode = table.column("episode_index").to_numpy()[order]
    frame = table.column("frame_index").to_numpy()[order]
    return state, episode, frame


def load_lerobot_episodes(dataset_dir: str | Path) -> list[LeRobotEpisode]:
    """Index and validate every episode of one RoboDojo LeRobot v3 dataset."""
    root = Path(dataset_dir)
    info = parse_info_json(root)
    features = info.get("features", {})
    state_feature = features.get("observation.state", {})
    if tuple(state_feature.get("shape", ())) != (STATE_DIM,) or tuple(_feature_names(state_feature)) != STATE_NAMES:
        raise ValueError(
            f"{root}: observation.state must be the dual-X5 converter layout {list(STATE_NAMES)}, "
            f"got shape {state_feature.get('shape')} names {_feature_names(state_feature)}"
        )
    missing = [key for key in LEROBOT_CAMERAS.values() if features.get(key, {}).get("dtype") != "video"]
    if missing:
        raise ValueError(f"{root}: missing video features {missing}")
    fps = float(info["fps"])
    sizes = {camera: _frame_size(features[key], key) for camera, key in LEROBOT_CAMERAS.items()}
    episodes = load_episodes_parquet(root)
    state, row_episode, row_frame = _state_rows(root)
    tasks = _episode_tasks(root, len(episodes))
    out = []
    for position, record in enumerate(episodes.to_dict("records")):
        index, length = int(record["episode_index"]), int(record["length"])
        source = f"{root}:episode {index}"
        if index != position:
            raise ValueError(f"{root}: episode indices are not 0..{len(episodes) - 1}")
        start, end = int(record["dataset_from_index"]), int(record["dataset_to_index"])
        if end - start != length or length < 2:
            raise ValueError(f"{source}: rows {start}..{end} for length {length}; episodes need two frames")
        if not (row_episode[start:end] == index).all() or not (row_frame[start:end] == np.arange(length)).all():
            raise ValueError(f"{source}: rows {start}..{end} are not its frames in order")
        rows = state[start:end]
        if not np.isfinite(rows).all():
            raise ValueError(f"{source}: observation.state must be finite")
        prompts = list(record["tasks"])
        if len(prompts) != 1:
            raise ValueError(f"{source}: expected one instruction, got {prompts}")
        videos = {}
        for camera, key in LEROBOT_CAMERAS.items():
            first = float(record[f"videos/{key}/from_timestamp"]) * fps
            last = float(record[f"videos/{key}/to_timestamp"]) * fps
            if abs(first - round(first)) > 1e-3 or abs(last - first - length) > 1e-3:
                raise ValueError(
                    f"{source}: {key} holds {last - first:.2f} frames from {first:.2f} for {length} rows "
                    "(frames dropped by the video encoder?)"
                )
            path = root / info["video_path"].format(
                video_key=key,
                chunk_index=int(record[f"videos/{key}/chunk_index"]),
                file_index=int(record[f"videos/{key}/file_index"]),
            )
            videos[camera] = (str(path), int(round(first)))
        out.append(
            LeRobotEpisode(
                dataset_dir=str(root),
                episode_index=index,
                task=tasks[index],
                instruction=_decode_instruction(prompts[0], source=source),
                state=rows,
                videos=videos,
                frame_size=sizes,
            )
        )
    return out


class RoboDojoLeRobotDataset(RoboDojoDataset):
    """One RoboDojo task served from LeRobot v3 episodes (``load_lerobot_episodes``)."""

    def __init__(self, episodes: Sequence[LeRobotEpisode], *, task_name: str, **kwargs: Any):
        variant = kwargs.get("variant", ROBODOJO_SIM_VARIANT)
        if variant != ROBODOJO_SIM_VARIANT:
            raise ValueError(
                f"RoboDojo LeRobot datasets are simulation releases; variant must be 'sim', got {variant!r}"
            )
        if _normalize_mode(kwargs.get("normalize_mode", "min-max")) and kwargs.get("normalization_stats_path") is None:
            raise ValueError("RoboDojo LeRobot datasets need an explicit normalization_stats_path")
        self._lerobot_episodes = list(episodes)
        root = self._lerobot_episodes[0].dataset_dir if self._lerobot_episodes else "."
        super().__init__(root, dataset_root=root, task_name=task_name, **kwargs)

    def _index_episodes(self, data_root, *, dataset_root, task_name) -> None:
        self.task_name = str(task_name)
        self.data_root = self.dataset_root = str(dataset_root)
        self._episode_files = [episode.name for episode in self._lerobot_episodes]
        self._episode_lengths = [len(episode.state) for episode in self._lerobot_episodes]
        self._instructions = [episode.instruction for episode in self._lerobot_episodes]

    def _read_window(self, episode_index, start, length, frame_indices) -> tuple[np.ndarray, list[Image.Image]]:
        episode = self._lerobot_episodes[episode_index]
        states = lerobot_state_to_eef20(episode.state[start : start + length], self.calibration)
        frames = {}
        for camera in self._view_cameras():
            path, offset = episode.videos[camera]
            height, width = episode.frame_size[camera]
            frames[camera] = decode_video_frames(path, [offset + index for index in frame_indices], height, width)
        return states, [
            self._render({camera: views[k] for camera, views in frames.items()}) for k in range(len(frame_indices))
        ]


class MultiTaskRoboDojoLeRobotDataset(MultiTaskRoboDojoDataset):
    """RoboDojo tasks from one or more LeRobot v3 datasets, with shared calibration and stats."""

    def __init__(
        self,
        dataset_dir: str | Path | Sequence[str | Path],
        *,
        calibration: Mapping[str, Any] | None = None,
        split: str = "train",
        embodiment: str = ROBODOJO_EMBODIMENT,
        variant: str = ROBODOJO_SIM_VARIANT,
        normalization_stats_path: str | Path | None = None,
        action_mode: str = DEPLOY_ACTION_MODE,
        robot: str | None = None,
        **dataset_kwargs: Any,
    ):
        if dataset_dir is None:
            raise ValueError("RoboDojo LeRobot dataset_dir is required")
        dirs = [dataset_dir] if isinstance(dataset_dir, (str, Path)) else list(dataset_dir)
        if not dirs:
            raise ValueError("RoboDojo LeRobot dataset_dir is an empty list")
        validate_embodiment(embodiment, variant=variant)
        by_task: dict[str, list[LeRobotEpisode]] = {}
        for directory in dirs:
            for episode in load_lerobot_episodes(str(directory)):
                by_task.setdefault(episode.task, []).append(episode)

        self.dataset_dir = ",".join(str(directory) for directory in dirs)
        self.tasks = sorted(by_task)
        self.split = split
        self.action_mode = action_mode
        self.embodiment = embodiment
        self.variant = variant
        self._sub_datasets: list[RoboDojoDataset] = []
        self._cumulative_lengths: list[int] = []
        cumulative = 0
        for task in self.tasks:
            dataset = RoboDojoLeRobotDataset(
                by_task[task],
                task_name=task,
                calibration=calibration,
                split=split,
                embodiment=embodiment,
                variant=variant,
                robot=robot,
                action_mode=action_mode,
                normalization_stats_path=normalization_stats_path,
                **dataset_kwargs,
            )
            self._sub_datasets.append(dataset)
            cumulative += len(dataset)
            self._cumulative_lengths.append(cumulative)
        if not self._sub_datasets or cumulative == 0:
            raise ValueError("RoboDojo LeRobot datasets produced an empty dataset")

        self._total_length = cumulative
        first = self._sub_datasets[0]
        self._action_dim_value = first.action_dim
        self.calibration = first.calibration
        self.calibration_fingerprint = first.calibration_fingerprint
        self.frame_contract = first.frame_contract
        self.frame_contract_fingerprint = first.frame_contract_fingerprint
        self.source_frame = first.source_frame
        self.contract_id = first.contract_id
        self._normalization_stats_shared = first.normalization_stats
        self.normalization_stats_path = first.normalization_stats_path


__all__ = [
    "LEROBOT_CAMERAS",
    "LeRobotEpisode",
    "MultiTaskRoboDojoLeRobotDataset",
    "RoboDojoLeRobotDataset",
    "SIM_ROOT_QUAT_WXYZ",
    "STATE_NAMES",
    "X5_CHAIN",
    "lerobot_state_to_eef20",
    "load_lerobot_episodes",
    "recorded_ee_poses",
    "x5_link6_in_root",
]
