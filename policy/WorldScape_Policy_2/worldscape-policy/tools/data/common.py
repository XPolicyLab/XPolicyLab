"""Shared utilities for HDF5 / LeRobot dataset conversion."""

from __future__ import annotations

import base64
import binascii
import json
import logging
import math
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

log = logging.getLogger(__name__)

DEFAULT_AGILEX_STATE_MAP = OrderedDict(
    [
        ("left_pos", [0, 3]),
        ("left_rot6d", [3, 9]),
        ("left_gripper", [9, 10]),
        ("right_pos", [10, 13]),
        ("right_rot6d", [13, 19]),
        ("right_gripper", [19, 20]),
    ]
)

DEFAULT_HDF5_VIDEO_MAP = OrderedDict(
    [
        ("cam_high", "observation.camera.head"),
        ("cam_left_wrist", "observation.camera.left"),
        ("cam_right_wrist", "observation.camera.right"),
    ]
)

DEFAULT_LEROBOT_VIDEO_KEYS = OrderedDict(
    [
        ("cam_high", "observation.images.cam_high"),
        ("cam_left_wrist", "observation.images.cam_left_wrist"),
        ("cam_right_wrist", "observation.images.cam_right_wrist"),
    ]
)

JPEG_SOF = {
    0xC0,
    0xC1,
    0xC2,
    0xC3,
    0xC5,
    0xC6,
    0xC7,
    0xC9,
    0xCA,
    0xCB,
    0xCD,
    0xCE,
    0xCF,
}


def dot_key(key: str) -> str:
    return key.replace("/", ".")


def h5_key(key: str) -> str:
    return key.replace(".", "/")


def text_value(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    if isinstance(raw, np.ndarray):
        return text_value(raw.item()) if raw.shape == () else " ".join(text_value(x) for x in raw.tolist())
    if isinstance(raw, np.generic):
        return str(raw.item())
    return str(raw)


def parse_mapping(
    raw: str | None,
    default: OrderedDict[str, list[int]],
) -> OrderedDict[str, list[int]]:
    if raw is None:
        return OrderedDict((key, list(bounds)) for key, bounds in default.items())
    data = json.loads(raw, object_pairs_hook=OrderedDict)
    if not isinstance(data, dict):
        raise SystemExit("mapping must be a JSON object")
    out = OrderedDict()
    for key, bounds in data.items():
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise SystemExit(f"mapping for {key!r} must be [start, end]")
        start, end = int(bounds[0]), int(bounds[1])
        if start < 0 or end <= start:
            raise SystemExit(f"bad range for {key!r}: {bounds}")
        out[str(key)] = [start, end]
    return out


def parse_hdf5_video_map(raw: str | None) -> OrderedDict[str, str]:
    if raw is None:
        return OrderedDict(DEFAULT_HDF5_VIDEO_MAP)
    data = json.loads(raw, object_pairs_hook=OrderedDict)
    if not isinstance(data, dict):
        raise SystemExit("--video-keys must be a JSON object")
    return OrderedDict((str(name), dot_key(str(source))) for name, source in data.items())


def parse_lerobot_video_map(raw: str | None) -> OrderedDict[str, str]:
    if raw is None:
        return OrderedDict(DEFAULT_LEROBOT_VIDEO_KEYS)
    data = json.loads(raw, object_pairs_hook=OrderedDict)
    if not isinstance(data, dict):
        raise SystemExit("--video-keys must be a JSON object")
    return OrderedDict((str(name), dot_key(str(source))) for name, source in data.items())


def validate_embodiment(value: str) -> str:
    from worldscape_policy.embodiment import canonical_embodiment

    canonical = canonical_embodiment(value)
    if canonical not in {"agilex", "libero", "robotwin2"}:
        log.warning(
            "embodiment %r canonicalizes to %r; expected agilex, libero, or robotwin2",
            value,
            canonical,
        )
    return canonical


def validate_embodiment_tag(tag: str) -> str:
    """Deprecated alias for :func:`validate_embodiment`."""

    return validate_embodiment(tag)


def build_embodiment_json(embodiment: str) -> dict[str, str]:
    canonical = validate_embodiment(embodiment)
    return {
        "robot_type": canonical,
        "embodiment": canonical,
        "embodiment_tag": canonical,
    }


def scan_hdf5(
    root: Path,
    pattern: str,
    sources: Path | None,
    limit: int | None,
) -> list[Path]:
    if sources:
        paths = []
        for line in sources.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            path = Path(line)
            paths.append((path if path.is_absolute() else root / path).resolve())
    elif root.is_file():
        paths = [root.resolve()]
    else:
        paths = sorted(path.resolve() for path in root.glob(pattern) if path.is_file())
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"no HDF5 files found under {root} with pattern {pattern!r}")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing HDF5 files: " + ", ".join(str(path) for path in missing[:5]))
    return paths


def read_fps(handle: h5py.File, fallback: float) -> float:
    try:
        raw = handle.attrs.get("fps", fallback)
        return float(raw.item() if isinstance(raw, np.generic) else raw)
    except Exception:
        return float(fallback)


def episode_length(handle: h5py.File, preferred: str) -> int:
    key = h5_key(preferred)
    if key in handle and handle[key].ndim > 0:
        return int(handle[key].shape[0])
    lengths: list[int] = []

    def visit(_name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset) and obj.ndim > 0 and obj.shape[0] > 1:
            lengths.append(int(obj.shape[0]))

    handle.visititems(visit)
    if not lengths:
        raise ValueError("could not infer episode length")
    return max(set(lengths), key=lambda value: (lengths.count(value), value))


def bytes_payload(value: Any) -> bytes | None:
    if isinstance(value, bytes):
        raw = value
    elif isinstance(value, bytearray):
        raw = bytes(value)
    elif isinstance(value, (np.bytes_, np.void)):
        raw = bytes(value)
    elif isinstance(value, str):
        raw = value.encode("utf-8")
    elif isinstance(value, np.ndarray) and value.dtype == np.uint8:
        raw = value.tobytes()
    else:
        return None
    raw = b"".join(raw.split())
    if raw.startswith(b"\xff\xd8"):
        return raw
    try:
        decoded = base64.b64decode(raw, validate=False)
    except (binascii.Error, ValueError):
        return None
    return decoded if decoded.startswith(b"\xff\xd8") else None


def jpeg_size(data: bytes) -> tuple[int, int] | None:
    index, total = 2, len(data)
    if not data.startswith(b"\xff\xd8"):
        return None
    while index + 3 < total:
        if data[index] != 0xFF:
            index += 1
            continue
        while index < total and data[index] == 0xFF:
            index += 1
        if index >= total:
            break
        marker = data[index]
        index += 1
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            continue
        seg_len = int.from_bytes(data[index : index + 2], "big")
        if seg_len < 2 or index + seg_len > total:
            break
        if marker in JPEG_SOF and seg_len >= 7:
            height = int.from_bytes(data[index + 3 : index + 5], "big")
            width = int.from_bytes(data[index + 5 : index + 7], "big")
            return width, height
        index += seg_len
    return None


def video_shape(
    path: Path,
    source_key: str,
    fallback: tuple[int, int, int],
) -> tuple[int, int, int]:
    try:
        with h5py.File(path, "r") as handle:
            key = h5_key(source_key)
            if key not in handle or handle[key].shape[0] == 0:
                return fallback
            payload = bytes_payload(handle[key][0])
            size = jpeg_size(payload) if payload else None
            if not size:
                return fallback
            width, height = size
            return height, width, 3
    except Exception as exc:
        log.warning("failed to infer video shape for %s from %s: %s", source_key, path, exc)
        return fallback


def quat_to_rot6d(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    rotation = np.empty((len(q), 3, 3), dtype=np.float64)
    rotation[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rotation[:, 0, 1] = 2 * (x * y - z * w)
    rotation[:, 0, 2] = 2 * (x * z + y * w)
    rotation[:, 1, 0] = 2 * (x * y + z * w)
    rotation[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rotation[:, 1, 2] = 2 * (y * z - x * w)
    rotation[:, 2, 0] = 2 * (x * z - y * w)
    rotation[:, 2, 1] = 2 * (y * z + x * w)
    rotation[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rotation[:, :, :2].reshape(len(q), 6).astype(np.float32)


def pose9(pose7: np.ndarray) -> np.ndarray:
    return np.concatenate([pose7[:, :3], quat_to_rot6d(pose7[:, 3:7])], axis=1).astype(np.float32)


def agilex_eef_arrays(
    handle: h5py.File,
    *,
    end_pose_key: str = "observations.end_pose",
    qpos_key: str = "observations.qpos",
    arm_order: str = "left_first",
    align_to_first_frame: bool = False,
    state_column: str | None = None,
    action_column: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if state_column and dot_key(state_column) in handle:
        state = np.asarray(handle[h5_key(state_column)][:], dtype=np.float32)
        action_key = action_column or "action.eef6d"
        if h5_key(action_key) in handle:
            action = np.asarray(handle[h5_key(action_key)][:], dtype=np.float32)
        else:
            action = np.concatenate([state[1:], state[-1:]], axis=0) if len(state) else state.copy()
        return state, action

    end_key, qpos_h5 = h5_key(end_pose_key), h5_key(qpos_key)
    if end_key not in handle:
        raise KeyError(f"missing dataset {end_key}")
    if qpos_h5 not in handle:
        raise KeyError(f"missing dataset {qpos_h5}")
    end_pose = handle[end_key][:].astype(np.float32)
    qpos = handle[qpos_h5][:].astype(np.float32)
    if end_pose.ndim != 2 or end_pose.shape[1] != 14:
        raise ValueError(f"{end_pose_key} must have shape (T, 14), got {end_pose.shape}")
    if qpos.ndim != 2 or qpos.shape[1] != 14:
        raise ValueError(f"{qpos_key} must have shape (T, 14), got {qpos.shape}")
    if len(end_pose) != len(qpos):
        raise ValueError(f"length mismatch: end_pose={len(end_pose)} qpos={len(qpos)}")

    first_pose, second_pose = end_pose[:, :7], end_pose[:, 7:14]
    first_grip, second_grip = qpos[:, 6:7], qpos[:, 13:14]
    if arm_order == "left_first":
        left_pose, left_grip = first_pose, first_grip
        right_pose, right_grip = second_pose, second_grip
    else:
        right_pose, right_grip = first_pose, first_grip
        left_pose, left_grip = second_pose, second_grip

    state = np.concatenate(
        [pose9(left_pose), left_grip, pose9(right_pose), right_grip],
        axis=1,
    ).astype(np.float32)
    action = (
        np.concatenate([state[1:], state[-1:]], axis=0).astype(np.float32)
        if len(state)
        else state.copy()
    )
    if align_to_first_frame and len(state):
        reference = state[0:1]
        state = state - reference
        action = action - reference
    return state, action


def read_hdf5_camera_frames(handle: h5py.File, source_key: str) -> np.ndarray:
    from PIL import Image
    import io

    key = h5_key(source_key)
    if key not in handle:
        raise KeyError(f"missing camera dataset {source_key}")
    dataset = handle[key]
    if dataset.ndim == 4 and dataset.dtype == np.uint8:
        return np.asarray(dataset[:], dtype=np.uint8)
    frames: list[np.ndarray] = []
    for value in dataset:
        payload = bytes_payload(value)
        if payload is None:
            raise ValueError(f"unsupported encoded frame in {source_key}")
        frames.append(np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"), dtype=np.uint8))
    if not frames:
        raise ValueError(f"camera dataset {source_key} is empty")
    return np.stack(frames)


def encode_video(frames: np.ndarray, output_path: Path, fps: float) -> None:
    import av

    options = {
        "threads": "1",
        "thread_type": "slice",
        "preset": "ultrafast",
        "tune": "zerolatency",
        "crf": "23",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(output_path), mode="w")
    stream = container.add_stream("h264", rate=int(round(fps)), options=options)
    stream.width = int(frames.shape[2])
    stream.height = int(frames.shape[1])
    stream.pix_fmt = "yuv420p"
    video_frame = av.VideoFrame(width=stream.width, height=stream.height, format="rgb24")
    frame_array = video_frame.to_ndarray(format="rgb24")
    for frame in frames:
        frame_array[:] = frame
        for packet in stream.encode(video_frame):
            container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()


def array_stats(array: np.ndarray) -> dict[str, list[float]]:
    values = np.asarray(array, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def write_json(path: Path, obj: Any, force: bool) -> None:
    if path.exists() and not force:
        log.info("  %s exists, skip", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=4, ensure_ascii=False) + "\n")
    log.info("  wrote %s", path)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]], force: bool) -> None:
    if path.exists() and not force:
        log.info("  %s exists, skip", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    log.info("  wrote %s (%d rows)", path, len(rows))


def task_id_from_hdf5(handle: h5py.File, path: Path) -> str:
    for key in ("task_id", "task", "language", "caption"):
        value = text_value(handle.attrs.get(key)).strip()
        if value:
            return value
    for parent in path.parents:
        if parent.name and not parent.name.startswith("episode_"):
            return parent.name
    return "task"


def load_task_map(dataset_root: Path, path: Path | None) -> OrderedDict[str, str]:
    task_map_path = path or (dataset_root / "task_id_to_prompt.json")
    if not task_map_path.exists():
        return OrderedDict()
    data = json.loads(task_map_path.read_text(), object_pairs_hook=OrderedDict)
    if not isinstance(data, dict):
        raise ValueError(f"task map must be a JSON object: {task_map_path}")
    return OrderedDict((str(key), str(value)) for key, value in data.items())


def relative_path(path: Path, root: Path, absolute: bool) -> str:
    if absolute:
        return str(path)
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def build_modality(
    *,
    state_map: OrderedDict[str, list[int]],
    action_map: OrderedDict[str, list[int]],
    video_map: OrderedDict[str, str],
    state_column: str,
    action_column: str,
    task_key: str,
) -> dict[str, Any]:
    modality: dict[str, Any] = {"state": {}, "action": {}, "video": {}, "annotation": {}}
    for name, (start, end) in state_map.items():
        modality["state"][name] = {
            "original_key": state_column,
            "start": start,
            "end": end,
            "rotation_type": None,
            "absolute": True,
            "dtype": "float32",
            "range": None,
        }
    for name, (start, end) in action_map.items():
        modality["action"][name] = {
            "original_key": action_column,
            "start": start,
            "end": end,
            "rotation_type": None,
            "absolute": True,
            "dtype": "float32",
            "range": None,
        }
    for name, source in video_map.items():
        modality["video"][name] = {"original_key": source}
    modality["annotation"][task_key.replace("annotation.", "", 1)] = {"original_key": task_key}
    return modality


def build_hdf5_info(
    *,
    embodiment: str,
    dataset_root: Path,
    episodes: Sequence[Mapping[str, Any]],
    video_map: OrderedDict[str, str],
    first_h5: Path,
    task_count: int,
    state_map: OrderedDict[str, list[int]],
    action_map: OrderedDict[str, list[int]],
    state_column: str,
    action_column: str,
    task_key: str,
    fps: float,
    chunk_size: int,
    image_height: int,
    image_width: int,
) -> dict[str, Any]:
    state_dim = max(end for _start, end in state_map.values())
    action_dim = max(end for _start, end in action_map.values())
    features: dict[str, Any] = {
        state_column: {
            "dtype": "float32",
            "shape": [state_dim],
            "names": [list(state_map.keys())],
        },
        action_column: {
            "dtype": "float32",
            "shape": [action_dim],
            "names": [list(action_map.keys())],
        },
        task_key: {"dtype": "string", "shape": [1], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    fallback = (image_height, image_width, 3)
    for _name, source in video_map.items():
        height, width, _channels = video_shape(first_h5, source, fallback)
        features[source] = {
            "dtype": "image",
            "shape": [3, height, width],
            "names": ["channels", "height", "width"],
        }
    total_episodes = len(episodes)
    return {
        "codebase_version": "v2.1",
        "robot_type": embodiment,
        "data_format": "hdf5",
        "source_dataset_path": str(dataset_root),
        "hdf5_path_key": "path",
        "state_column": state_column,
        "action_column": action_column,
        "total_episodes": total_episodes,
        "total_frames": sum(int(episode["length"]) for episode in episodes),
        "total_tasks": task_count,
        "total_videos": 0,
        "total_chunks": int(math.ceil(total_episodes / chunk_size)),
        "chunks_size": chunk_size,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }


def build_lerobot_info(
    *,
    embodiment: str,
    episodes: Sequence[Mapping[str, Any]],
    video_map: OrderedDict[str, str],
    first_h5: Path,
    task_count: int,
    state_map: OrderedDict[str, list[int]],
    action_map: OrderedDict[str, list[int]],
    state_column: str,
    action_column: str,
    task_key: str,
    fps: float,
    chunk_size: int,
    image_height: int,
    image_width: int,
) -> dict[str, Any]:
    state_dim = max(end for _start, end in state_map.values())
    action_dim = max(end for _start, end in action_map.values())
    features: dict[str, Any] = {
        state_column: {
            "dtype": "float32",
            "shape": [state_dim],
            "names": [list(state_map.keys())],
        },
        action_column: {
            "dtype": "float32",
            "shape": [action_dim],
            "names": [list(action_map.keys())],
        },
        task_key: {"dtype": "string", "shape": [1], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "is_exec": {"dtype": "bool", "shape": [1], "names": None},
    }
    fallback = (image_height, image_width, 3)
    for _name, video_key in video_map.items():
        source = DEFAULT_HDF5_VIDEO_MAP.get(_name, "")
        if source:
            height, width, _channels = video_shape(first_h5, source, fallback)
        else:
            height, width = fallback[0], fallback[1]
        features[video_key] = {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channels"],
        }
    total_episodes = len(episodes)
    return {
        "codebase_version": "v2.1",
        "robot_type": embodiment,
        "data_format": "lerobot",
        "total_episodes": total_episodes,
        "total_frames": sum(int(episode["length"]) for episode in episodes),
        "total_tasks": task_count,
        "total_videos": total_episodes * len(video_map),
        "total_chunks": int(math.ceil(total_episodes / chunk_size)),
        "chunks_size": chunk_size,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }


def read_optional_bool_series(handle: h5py.File, length: int) -> np.ndarray | None:
    for key in ("is_exec", "meta/is_exec"):
        if key in handle:
            values = np.asarray(handle[key][:]).astype(bool).reshape(-1)
            if len(values) == length:
                return values
    return None


def read_optional_text_series(handle: h5py.File, key: str, length: int) -> np.ndarray | None:
    hdf5_key = h5_key(key)
    if hdf5_key not in handle:
        return None
    raw = handle[hdf5_key][:]
    values = np.asarray([text_value(item) for item in np.asarray(raw).reshape(-1)], dtype=object)
    if len(values) == 1:
        return np.full(length, values[0], dtype=object)
    if len(values) == length:
        return values
    return None
