"""Native HDF5 replay transport for AgileX evaluation."""

from __future__ import annotations

import base64
import binascii
from pathlib import Path
from typing import Any

import numpy as np

from evals.agilex.observation_adapter import (
    quaternion_pose_to_rot6d,
    resize_image,
    sample_video,
)

DEFAULT_VIDEO_COLUMN_ALIASES = (
    (
        "observations.images.cam_high",
        "observation.images.cam_high",
        "observations.image.cam_high",
        "observation.image.cam_high",
        "observation.camera.head",
        "observations.camera.head",
    ),
    (
        "observations.images.cam_left_wrist",
        "observation.images.cam_left_wrist",
        "observations.image.cam_left_wrist",
        "observation.image.cam_left_wrist",
        "observation.camera.left",
        "observations.camera.left",
    ),
    (
        "observations.images.cam_right_wrist",
        "observation.images.cam_right_wrist",
        "observations.image.cam_right_wrist",
        "observation.image.cam_right_wrist",
        "observation.camera.right",
        "observations.camera.right",
    ),
)
DEFAULT_VIDEO_COLUMNS = tuple(aliases[0] for aliases in DEFAULT_VIDEO_COLUMN_ALIASES)


def _iter_datasets(group: Any, prefix: str = ""):
    import h5py

    for key in group:
        item = group[key]
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, h5py.Group):
            yield from _iter_datasets(item, path)
        elif isinstance(item, h5py.Dataset):
            yield path, item


def _column(array: np.ndarray, length: int) -> list[Any]:
    if array.ndim == 0:
        return [np.asarray(array).item()] * length
    if array.shape[0] == length:
        return [array[index] for index in range(length)]
    if array.shape[0] == 1:
        value = array[0]
        return [
            value.item() if np.asarray(value).shape == () else np.asarray(value).copy()
            for _ in range(length)
        ]
    raise ValueError(f"Expected HDF5 time dimension {length} or 1, got {array.shape[0]}")


def hdf5_to_dataframe(path: str | Path):
    import h5py
    import pandas as pd

    with h5py.File(Path(path).as_posix(), "r") as stream:
        arrays = {
            name: np.asarray(dataset[()])
            for name, dataset in _iter_datasets(stream)
        }
    if not arrays:
        return pd.DataFrame()
    lengths = [int(array.shape[0]) for array in arrays.values() if array.ndim > 0]
    nontrivial = {length for length in lengths if length > 1}
    if len(nontrivial) > 1:
        raise ValueError(f"Inconsistent HDF5 time lengths: {sorted(nontrivial)}")
    length = max(lengths) if lengths else 1
    return pd.DataFrame(
        {name: _column(array, length) for name, array in arrays.items()}
    )


def _decode_frame(value: Any) -> np.ndarray:
    import cv2

    if isinstance(value, np.ndarray):
        if value.ndim == 3 and value.shape[-1] == 3:
            return value.astype(np.uint8, copy=False)
        if value.ndim == 1:
            value = value.tobytes()
        else:
            raise ValueError(f"Unsupported ndarray frame shape: {value.shape}")
    if isinstance(value, str):
        raw = base64.b64decode(value)
    elif isinstance(value, bytes):
        try:
            raw = base64.b64decode(value, validate=True)
        except (ValueError, binascii.Error):
            raw = value
    else:
        raise TypeError(f"Unsupported HDF5 frame type: {type(value)!r}")
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image bytes from HDF5")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)


def _eef6d_state(row: Any) -> dict[str, np.ndarray]:
    if "observation.eef6d" in row:
        state = np.asarray(row["observation.eef6d"], dtype=np.float32).reshape(-1)
        if state.shape[0] != 20:
            raise ValueError(f"Expected observation.eef6d dim 20, got {state.shape}")
        return {
            "state.left_pos": state[0:3].reshape(1, 3),
            "state.left_rot6d": state[3:9].reshape(1, 6),
            "state.left_gripper": state[9:10].reshape(1, 1),
            "state.right_pos": state[10:13].reshape(1, 3),
            "state.right_rot6d": state[13:19].reshape(1, 6),
            "state.right_gripper": state[19:20].reshape(1, 1),
        }
    end_pose = np.asarray(row["observations.end_pose"], dtype=np.float32).reshape(-1)
    qpos = np.asarray(row["observations.qpos"], dtype=np.float32).reshape(-1)
    if end_pose.shape[0] < 14 or qpos.shape[0] < 14:
        raise ValueError("Expected end_pose and qpos dimensions >= 14")
    left = quaternion_pose_to_rot6d(end_pose[:7])
    right = quaternion_pose_to_rot6d(end_pose[7:14])
    return {
        "state.left_pos": left[:3].reshape(1, 3).astype(np.float32),
        "state.left_rot6d": left[3:].reshape(1, 6).astype(np.float32),
        "state.left_gripper": qpos[6:7].reshape(1, 1).astype(np.float32),
        "state.right_pos": right[:3].reshape(1, 3).astype(np.float32),
        "state.right_rot6d": right[3:].reshape(1, 6).astype(np.float32),
        "state.right_gripper": qpos[13:14].reshape(1, 1).astype(np.float32),
    }


def _joint_state(row: Any) -> dict[str, np.ndarray]:
    qpos = np.asarray(row["observations.qpos"], dtype=np.float32).reshape(-1)
    if qpos.shape[0] < 14:
        raise ValueError(f"Expected qpos dim >= 14, got {qpos.shape}")
    return {
        "state.left_joint": qpos[:7].reshape(1, 7),
        "state.right_joint": qpos[7:14].reshape(1, 7),
    }


class NativeHDF5Robot:
    """Historical HDF5 schema behind the native robot adapter boundary."""

    def __init__(
        self,
        path: str | Path,
        *,
        video_columns: str | None = None,
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"HDF5 path does not exist: {self.path}")
        configured_columns = (
            tuple(item.strip() for item in video_columns.split(",") if item.strip())
            if video_columns
            else None
        )
        if configured_columns is not None and len(configured_columns) != 3:
            raise ValueError(f"Expected exactly 3 video columns, got {configured_columns}")
        dataframe = hdf5_to_dataframe(self.path)
        if dataframe.empty:
            raise ValueError(f"HDF5 file is empty: {self.path}")
        columns = configured_columns or tuple(
            next((name for name in aliases if name in dataframe.columns), aliases[0])
            for aliases in DEFAULT_VIDEO_COLUMN_ALIASES
        )
        missing = [name for name in columns if name not in dataframe.columns]
        if missing:
            raise KeyError(
                f"HDF5 is missing camera columns {missing}; available={sorted(dataframe.columns)}"
            )
        if "is_exec" in dataframe.columns:
            is_exec = dataframe["is_exec"].astype(bool).to_numpy()
        elif "meta.is_exec" in dataframe.columns:
            is_exec = dataframe["meta.is_exec"].astype(bool).to_numpy()
        else:
            is_exec = np.ones(len(dataframe), dtype=bool)
        self.context = dataframe.loc[~is_exec].reset_index(drop=True)
        self.execution = dataframe.loc[is_exec].reset_index(drop=True)
        if self.execution.empty:
            raise ValueError(f"HDF5 has no execution rows: {self.path}")
        self.videos = tuple(
            [_decode_frame(value) for value in self.execution[column].tolist()]
            for column in columns
        )
        self.context_videos = (
            tuple(
                [_decode_frame(value) for value in self.context[column].tolist()]
                for column in columns
            )
            if not self.context.empty
            else None
        )
        self.cursor = 0

    def reset_episode(self) -> None:
        self.cursor = 0

    def default_max_steps(self, rollout_steps: int) -> int:
        return max(1, (len(self.execution) - 1) // max(1, int(rollout_steps)))

    def read_context_video(
        self, num_frames: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        source = self.context_videos or self.videos
        return tuple(sample_video(frames, num_frames) for frames in source)  # type: ignore[return-value]

    def read_observation(
        self,
        use_history: bool = False,
        num_history_frames: int = 4,
        action_mode: str = "eef",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], int]:
        index = min(self.cursor, len(self.execution) - 1)
        self.cursor += 1
        if use_history:
            start = max(0, index - int(num_history_frames) + 1)
            indices = list(range(start, index + 1))
            while len(indices) < int(num_history_frames):
                indices.insert(0, indices[0])
        else:
            indices = [index]
        cameras = tuple(
            np.stack([resize_image(frames[item]) for item in indices], axis=0)
            for frames in self.videos
        )
        row = self.execution.iloc[index]
        state = _joint_state(row) if action_mode == "joint" else _eef6d_state(row)
        return cameras[0], cameras[1], cameras[2], state, index

    def send_end_pose_action(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def send_joint_state_action(self, *_args: Any, **_kwargs: Any) -> None:
        return None
