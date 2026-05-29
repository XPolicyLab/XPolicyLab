#!/usr/bin/env python3
"""Convert XPolicyLab HDF5 episodes to DreamZero AgiBot LeRobot/GEAR data."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # pyright: ignore[reportMissingImports]
except ImportError:
    LeRobotDataset = None

try:
    from XPolicyLab.utils.load_file import load_hdf5
    from XPolicyLab.utils.process_data import (
        decode_image_bit,
        get_robot_action_dim_info,
        pack_robot_state,
    )
except ImportError:
    load_hdf5 = None
    decode_image_bit = None
    get_robot_action_dim_info = None
    pack_robot_state = None


ROOT_PATH = Path(__file__).resolve().parents[3]
POLICY_DIR = Path(__file__).resolve().parent

AGIBOT_STATE_DIM = 20
AGIBOT_ACTION_DIM = 22
CAMERA_ALIASES = {
    "cam_head": "top_head",
    "cam_left_wrist": "hand_left",
    "cam_right_wrist": "hand_right",
}
STATE_MAPPING = {
    "left_arm_joint_position": [0, 7],
    "right_arm_joint_position": [7, 14],
    "left_effector_position": [14, 15],
    "right_effector_position": [15, 16],
    "head_position": [16, 18],
    "waist_pitch": [18, 19],
    "waist_lift": [19, 20],
}
ACTION_MAPPING = {
    **STATE_MAPPING,
    "robot_velocity": [20, 22],
}
V3_CAMERA_MAPPING = {
    "top_head": "observation.images.cam_high",
    "hand_left": "observation.images.cam_left_wrist",
    "hand_right": "observation.images.cam_right_wrist",
}
RELATIVE_ACTION_KEYS = [
    "left_arm_joint_position",
    "right_arm_joint_position",
    "head_position",
    "waist_pitch",
    "waist_lift",
]


def _pad_or_trim(values: np.ndarray, dim: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape[-1] == dim:
        return values
    if values.shape[-1] > dim:
        return values[..., :dim]
    pad_width = [(0, 0)] * values.ndim
    pad_width[-1] = (0, dim - values.shape[-1])
    return np.pad(values, pad_width, mode="constant")


def _packed_to_agibot_vectors(packed: np.ndarray, robot_info: dict) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.float32)
    arm_dims = list(robot_info["arm_dim"])
    ee_dims = list(robot_info["ee_dim"])
    num_steps = packed.shape[0]

    out = np.zeros((num_steps, AGIBOT_STATE_DIM), dtype=np.float32)
    offset = 0
    for arm_idx, (arm_dim, ee_dim) in enumerate(zip(arm_dims, ee_dims)):
        arm = packed[:, offset : offset + arm_dim]
        offset += arm_dim
        ee = packed[:, offset : offset + ee_dim]
        offset += ee_dim

        if arm_idx == 0:
            out[:, 0:7] = _pad_or_trim(arm, 7)
            out[:, 14:15] = _pad_or_trim(ee, 1)
        elif arm_idx == 1:
            out[:, 7:14] = _pad_or_trim(arm, 7)
            out[:, 15:16] = _pad_or_trim(ee, 1)

    return out


def _agibot_action_from_packed(packed: np.ndarray, robot_info: dict) -> np.ndarray:
    state_like = _packed_to_agibot_vectors(packed, robot_info)
    action = np.zeros((state_like.shape[0], AGIBOT_ACTION_DIM), dtype=np.float32)
    action[:, :AGIBOT_STATE_DIM] = state_like
    return action


def _pad_v3_state(values: Any, dim: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    out = np.zeros((values.shape[0], dim), dtype=np.float32)
    copy_dim = min(values.shape[-1], 14)
    out[:, :copy_dim] = values[:, :copy_dim]
    return out


def _stats_for_padded_source(source_stats: dict[str, Any], dim: int) -> dict[str, list[float]]:
    result = {}
    for stat_name in ("mean", "std", "min", "max", "q01", "q99"):
        values = np.asarray(source_stats.get(stat_name, []), dtype=np.float32).reshape(-1)
        padded = np.zeros(dim, dtype=np.float32)
        copy_dim = min(len(values), 14, dim)
        if copy_dim:
            padded[:copy_dim] = values[:copy_dim]
        result[stat_name] = padded.tolist()
    return result


def _stats_from_samples(samples: np.ndarray, dim: int) -> dict[str, list[float]]:
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size == 0:
        samples = np.zeros((1, dim), dtype=np.float32)
    if samples.ndim == 1:
        samples = samples.reshape(-1, dim)
    return _stats(samples)


def _ensure_action_group(data: dict[str, Any]) -> dict[str, Any]:
    if "action" in data:
        return data
    data = dict(data)
    data["action"] = data.get("state", {})
    return data


def _pack(data: dict[str, Any], action_type: str, robot_info: dict, state_type: str) -> np.ndarray:
    try:
        return pack_robot_state(
            data,
            action_type,
            robot_info,
            source_type="dataset",
            state_type=state_type,
        ).astype(np.float32)
    except KeyError:
        if state_type != "action":
            raise
        return pack_robot_state(
            _ensure_action_group(data),
            action_type,
            robot_info,
            source_type="dataset",
            state_type="action",
        ).astype(np.float32)


def _decode_episode_images(data: dict[str, Any], num_steps: int) -> dict[str, np.ndarray]:
    images = {}
    vision = data.get("vision", {})
    for source_name, output_name in CAMERA_ALIASES.items():
        if source_name not in vision or "colors" not in vision[source_name]:
            images[output_name] = np.zeros((num_steps, 240, 320, 3), dtype=np.uint8)
            continue
        raw = decode_image_bit(vision[source_name]["colors"])
        processed = []
        for img in raw:
            img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            processed.append(img)
        images[output_name] = np.asarray(processed, dtype=np.uint8)
    return images


def _episode_instruction(data: dict[str, Any], fallback: str) -> str:
    instructions = data.get("instructions", data.get("instruction", None))
    if isinstance(instructions, bytes):
        return instructions.decode("utf-8")
    if isinstance(instructions, str):
        return instructions
    if isinstance(instructions, np.ndarray):
        instructions = instructions.tolist()
    if isinstance(instructions, (list, tuple)) and instructions:
        first = instructions[0]
        return first.decode("utf-8") if isinstance(first, bytes) else str(first)
    return fallback


def _features() -> dict[str, dict[str, Any]]:
    features = {
        "observation.state": {"dtype": "float32", "shape": (AGIBOT_STATE_DIM,)},
        "action": {"dtype": "float32", "shape": (AGIBOT_ACTION_DIM,)},
    }
    for camera_name in CAMERA_ALIASES.values():
        features[f"observation.images.{camera_name}"] = {
            "dtype": "video",
            "shape": (240, 320, 3),
            "names": ["height", "width", "channel"],
        }
    return features


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=4)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _stats(values: np.ndarray) -> dict[str, list[float]]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def _relative_stats(actions: list[np.ndarray], states: list[np.ndarray], horizon: int) -> dict[str, Any]:
    result = {}
    for key in RELATIVE_ACTION_KEYS:
        start, end = ACTION_MAPPING[key]
        samples = []
        for action, state in zip(actions, states):
            usable = len(action)
            for idx in range(usable):
                ref = state[idx, start:end]
                chunk = action[idx : min(idx + horizon, len(action)), start:end]
                samples.extend(chunk - ref)
        if samples:
            result[key] = _stats(np.asarray(samples, dtype=np.float32))
    return result


def _write_gear_metadata(
    dataset_path: Path,
    episode_lengths: list[int],
    tasks: list[str],
    states: list[np.ndarray],
    actions: list[np.ndarray],
    fps: int,
    action_horizon: int,
) -> None:
    modality = {
        "state": {
            key: {
                "original_key": "observation.state",
                "start": value[0],
                "end": value[1],
                "rotation_type": None,
                "absolute": True,
                "dtype": "float32",
                "range": None,
            }
            for key, value in STATE_MAPPING.items()
        },
        "action": {
            key: {
                "original_key": "action",
                "start": value[0],
                "end": value[1],
                "rotation_type": None,
                "absolute": True,
                "dtype": "float32",
                "range": None,
            }
            for key, value in ACTION_MAPPING.items()
        },
        "video": {
            camera: {"original_key": f"observation.images.{camera}"}
            for camera in CAMERA_ALIASES.values()
        },
        "annotation": {
            "language.action_text": {"original_key": "task_index"},
        },
    }
    _write_json(dataset_path / "meta" / "modality.json", modality)
    _write_json(dataset_path / "meta" / "embodiment.json", {"robot_type": "agibot", "embodiment_tag": "agibot"})

    all_states = np.concatenate(states, axis=0)
    all_actions = np.concatenate(actions, axis=0)
    _write_json(
        dataset_path / "meta" / "stats.json",
        {
            "observation.state": _stats(all_states),
            "action": _stats(all_actions),
        },
    )
    _write_json(dataset_path / "meta" / "relative_stats_dreamzero.json", _relative_stats(actions, states, action_horizon))

    task_rows = [{"task_index": idx, "task": task} for idx, task in enumerate(tasks)]
    _write_jsonl(dataset_path / "meta" / "tasks.jsonl", task_rows)
    _write_jsonl(
        dataset_path / "meta" / "episodes.jsonl",
        [
            {"episode_index": idx, "tasks": [tasks[idx]], "length": length}
            for idx, length in enumerate(episode_lengths)
        ],
    )

    info_path = dataset_path / "meta" / "info.json"
    if info_path.exists():
        with info_path.open("r", encoding="utf-8") as f:
            info = json.load(f)
    else:
        info = {}
    info.update(
        {
            "fps": fps,
            "total_episodes": len(episode_lengths),
            "total_frames": int(sum(episode_lengths)),
        }
    )
    _write_json(info_path, info)


def _read_v3_tasks(source_path: Path, episodes_df: pd.DataFrame) -> list[dict[str, Any]]:
    tasks_path = source_path / "meta" / "tasks.parquet"
    if tasks_path.exists():
        tasks_df = pd.read_parquet(tasks_path)
        rows = []
        for task_text, row in tasks_df.iterrows():
            rows.append({"task_index": int(row["task_index"]), "task": str(task_text)})
        rows.sort(key=lambda row: row["task_index"])
        return rows

    task_map: dict[int, str] = {}
    for _, row in episodes_df.iterrows():
        tasks = row.get("tasks", [])
        task_text = tasks[0] if isinstance(tasks, (list, tuple, np.ndarray)) and len(tasks) else str(tasks)
        task_index = int(row.get("task_index", len(task_map)))
        task_map.setdefault(task_index, str(task_text))
    return [{"task_index": idx, "task": task} for idx, task in sorted(task_map.items())]


def _read_v3_episodes(source_path: Path, expert_data_num: int) -> pd.DataFrame:
    episode_files = sorted((source_path / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"No LeRobot v3 episode metadata found under {source_path / 'meta' / 'episodes'}")
    episodes_df = pd.concat([pd.read_parquet(path) for path in episode_files], ignore_index=True)
    episodes_df = episodes_df.sort_values("episode_index").reset_index(drop=True)
    if expert_data_num > 0:
        episodes_df = episodes_df.iloc[:expert_data_num].copy()
    return episodes_df


def _as_task_list(value: Any) -> list[str]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]


def _write_v3_modality_metadata(dataset_path: Path) -> None:
    modality = {
        "state": {
            key: {
                "original_key": "observation.state",
                "start": value[0],
                "end": value[1],
                "rotation_type": None,
                "absolute": True,
                "dtype": "float32",
                "range": None,
            }
            for key, value in STATE_MAPPING.items()
        },
        "action": {
            key: {
                "original_key": "action",
                "start": value[0],
                "end": value[1],
                "rotation_type": None,
                "absolute": True,
                "dtype": "float32",
                "range": None,
            }
            for key, value in ACTION_MAPPING.items()
        },
        "video": {
            dreamzero_key: {"original_key": lerobot_key}
            for dreamzero_key, lerobot_key in V3_CAMERA_MAPPING.items()
        },
        "annotation": {
            "language.action_text": {"original_key": "task_index"},
        },
    }
    _write_json(dataset_path / "meta" / "modality.json", modality)
    _write_json(dataset_path / "meta" / "embodiment.json", {"robot_type": "agibot", "embodiment_tag": "agibot"})


def _feature_with_hwc_shape(feature: dict[str, Any]) -> dict[str, Any]:
    if feature.get("dtype") != "video":
        return feature
    names = list(feature.get("names") or [])
    shape = list(feature.get("shape") or [])
    if names == ["channels", "height", "width"] and len(shape) == 3:
        feature = dict(feature)
        feature["shape"] = [shape[1], shape[2], shape[0]]
        feature["names"] = ["height", "width", "channel"]
    return feature


def _write_v3_info_and_stats(
    dataset_path: Path,
    source_info: dict[str, Any],
    source_stats: dict[str, Any],
    total_episodes: int,
    total_frames: int,
    chunks_size: int,
    relative_samples: dict[str, list[np.ndarray]],
) -> None:
    features = dict(source_info.get("features", {}))
    features["observation.state"] = {
        "dtype": "float32",
        "shape": [AGIBOT_STATE_DIM],
        "names": None,
    }
    features["action"] = {
        "dtype": "float32",
        "shape": [AGIBOT_ACTION_DIM],
        "names": None,
    }
    for video_key in V3_CAMERA_MAPPING.values():
        if video_key in features:
            features[video_key] = _feature_with_hwc_shape(features[video_key])

    info = dict(source_info)
    info.update(
        {
            "codebase_version": "v2.1",
            "robot_type": "agibot",
            "total_episodes": int(total_episodes),
            "total_frames": int(total_frames),
            "chunks_size": int(chunks_size),
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4",
            "features": features,
        }
    )
    _write_json(dataset_path / "meta" / "info.json", info)

    stats = {
        "observation.state": _stats_for_padded_source(source_stats.get("observation.state", {}), AGIBOT_STATE_DIM),
        "action": _stats_for_padded_source(source_stats.get("action", {}), AGIBOT_ACTION_DIM),
    }
    _write_json(dataset_path / "meta" / "stats.json", stats)

    relative_stats = {}
    for key, samples in relative_samples.items():
        start, end = ACTION_MAPPING[key]
        values = np.concatenate(samples, axis=0) if samples else np.zeros((1, end - start), dtype=np.float32)
        relative_stats[key] = _stats_from_samples(values, end - start)
    _write_json(dataset_path / "meta" / "relative_stats_dreamzero.json", relative_stats)


def _replace_with_symlink(target: Path, link_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    link_path.symlink_to(target)


def convert_lerobot_v3(args: argparse.Namespace) -> None:
    source_path = Path(args.source_lerobot_path or DEFAULT_LEROBOT_V3_PATH).expanduser().resolve()
    if not (source_path / "meta" / "info.json").exists():
        raise FileNotFoundError(f"LeRobot v3 info.json not found: {source_path / 'meta' / 'info.json'}")

    repo_id = args.repo_id or f"{args.dataset_name}-{args.env_cfg_type}-{args.expert_data_num}-{args.action_type}"
    output_root = Path(args.output_dir).expanduser().resolve()
    dataset_path = output_root / repo_id
    if dataset_path.exists():
        shutil.rmtree(dataset_path)
    (dataset_path / "meta").mkdir(parents=True, exist_ok=True)

    source_info = json.loads((source_path / "meta" / "info.json").read_text(encoding="utf-8"))
    source_stats = json.loads((source_path / "meta" / "stats.json").read_text(encoding="utf-8"))
    chunks_size = int(source_info.get("chunks_size", 1000))
    episodes_df = _read_v3_episodes(source_path, int(args.expert_data_num))
    episode_ids = {int(ep_idx) for ep_idx in episodes_df["episode_index"].tolist()}
    tasks = _read_v3_tasks(source_path, episodes_df)

    _write_v3_modality_metadata(dataset_path)
    _write_jsonl(dataset_path / "meta" / "tasks.jsonl", tasks)
    _write_jsonl(
        dataset_path / "meta" / "episodes.jsonl",
        [
            {
                "episode_index": int(row["episode_index"]),
                "tasks": _as_task_list(row.get("tasks", [])),
                "length": int(row["length"]),
            }
            for _, row in episodes_df.iterrows()
        ],
    )

    video_offsets: dict[int, float] = {}
    total_frames = 0
    relative_samples: dict[str, list[np.ndarray]] = {key: [] for key in RELATIVE_ACTION_KEYS}

    for _, row in tqdm(episodes_df.iterrows(), total=len(episodes_df), desc="DreamZero link v3 videos"):
        episode_index = int(row["episode_index"])
        episode_chunk = episode_index // chunks_size
        primary_offset = None
        for video_key in V3_CAMERA_MAPPING.values():
            chunk_col = f"videos/{video_key}/chunk_index"
            file_col = f"videos/{video_key}/file_index"
            from_col = f"videos/{video_key}/from_timestamp"
            if chunk_col not in row or file_col not in row:
                continue
            video_chunk = int(row[chunk_col])
            video_file = int(row[file_col])
            source_video = source_path / "videos" / video_key / f"chunk-{video_chunk:03d}" / f"file-{video_file:03d}.mp4"
            target_video = dataset_path / "videos" / video_key / f"chunk-{episode_chunk:03d}" / f"episode_{episode_index:06d}.mp4"
            _replace_with_symlink(source_video, target_video)
            if primary_offset is None and from_col in row:
                primary_offset = float(row[from_col])
        video_offsets[episode_index] = primary_offset or 0.0

    data_files = sorted((source_path / "data").glob("chunk-*/*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No LeRobot v3 data parquet files found under {source_path / 'data'}")
    data_frames = []
    for parquet_file in tqdm(data_files, desc="DreamZero read v3 data"):
        df = pd.read_parquet(parquet_file)
        df = df[df["episode_index"].isin(episode_ids)]
        if not df.empty:
            data_frames.append(df)
    if not data_frames:
        raise FileNotFoundError(f"No selected episodes found in {source_path / 'data'}")

    all_data = pd.concat(data_frames, ignore_index=True)
    for episode_index, episode_df in tqdm(
        all_data.groupby("episode_index", sort=True),
        total=len(episode_ids),
        desc="DreamZero write episode parquet",
    ):
        episode_index = int(episode_index)
        episode_df = episode_df.sort_values("frame_index").reset_index(drop=True).copy()
        state = _pad_v3_state(np.stack(episode_df["observation.state"].to_numpy()), AGIBOT_STATE_DIM)
        action = _pad_v3_state(np.stack(episode_df["action"].to_numpy()), AGIBOT_ACTION_DIM)
        episode_df["observation.state"] = list(state)
        episode_df["action"] = list(action)

        offset = video_offsets.get(episode_index, 0.0)
        if offset:
            episode_df["timestamp"] = episode_df["timestamp"].astype(float) + offset

        episode_chunk = episode_index // chunks_size
        target_path = dataset_path / "data" / f"chunk-{episode_chunk:03d}" / f"episode_{episode_index:06d}.parquet"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        episode_df.to_parquet(target_path, index=False)
        total_frames += len(episode_df)

        for key in RELATIVE_ACTION_KEYS:
            start, end = ACTION_MAPPING[key]
            relative_samples[key].append(action[:, start:end] - state[:, start:end])

    _write_v3_info_and_stats(
        dataset_path=dataset_path,
        source_info=source_info,
        source_stats=source_stats,
        total_episodes=len(episodes_df),
        total_frames=total_frames,
        chunks_size=chunks_size,
        relative_samples=relative_samples,
    )
    print(f"[DreamZero process_data] Done. Normalized LeRobot v3 dataset saved to: {dataset_path}")


def convert(args: argparse.Namespace) -> None:
    if LeRobotDataset is None:
        raise ImportError("LeRobotDataset is required for legacy HDF5 conversion. Run install.sh first.")
    if load_hdf5 is None or get_robot_action_dim_info is None or pack_robot_state is None or decode_image_bit is None:
        raise ImportError("XPolicyLab is required for legacy HDF5 conversion. Run install.sh first.")

    repo_id = f"{args.dataset_name}-{args.task_name}-{args.env_cfg_type}-{args.expert_data_num}-{args.action_type}"
    output_root = Path(args.output_dir).resolve()
    dataset_path = output_root / repo_id
    source_dir = ROOT_PATH / "data" / args.dataset_name / args.task_name / args.env_cfg_type
    if not source_dir.exists():
        raise FileNotFoundError(f"XPolicyLab data directory not found: {source_dir}")

    if dataset_path.exists():
        shutil.rmtree(dataset_path)
    output_root.mkdir(parents=True, exist_ok=True)

    robot_info = get_robot_action_dim_info(args.env_cfg_type)
    episode_files = sorted((source_dir / "data").glob("episode_*.hdf5"))
    if args.expert_data_num > 0:
        episode_files = episode_files[: args.expert_data_num]
    if not episode_files:
        raise FileNotFoundError(f"No episode_*.hdf5 found under {source_dir / 'data'}")

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=args.fps,
        robot_type="agibot",
        features=_features(),
        root=dataset_path,
        use_videos=True,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
    )

    states_by_episode: list[np.ndarray] = []
    actions_by_episode: list[np.ndarray] = []
    episode_lengths: list[int] = []
    tasks: list[str] = []

    for task_idx, episode_file in enumerate(tqdm(episode_files, desc="DreamZero process_data")):
        data = load_hdf5(episode_file)
        state = _packed_to_agibot_vectors(_pack(data, args.action_type, robot_info, "state"), robot_info)
        action = _agibot_action_from_packed(_pack(data, args.action_type, robot_info, "action"), robot_info)
        num_steps = min(len(state), len(action))
        state = state[:num_steps]
        action = action[:num_steps]
        images = _decode_episode_images(data, num_steps)
        task_text = _episode_instruction(data, args.task_name)

        for frame_idx in range(num_steps):
            frame = {
                "observation.state": state[frame_idx],
                "action": action[frame_idx],
            }
            for camera_name, camera_images in images.items():
                frame[f"observation.images.{camera_name}"] = camera_images[frame_idx]
            dataset.add_frame(frame, task=task_text)
        dataset.save_episode()

        states_by_episode.append(state)
        actions_by_episode.append(action)
        episode_lengths.append(num_steps)
        tasks.append(task_text)

    _write_gear_metadata(
        dataset_path=dataset_path,
        episode_lengths=episode_lengths,
        tasks=tasks,
        states=states_by_episode,
        actions=actions_by_episode,
        fps=args.fps,
        action_horizon=args.action_horizon,
    )
    print(f"[DreamZero process_data] Done. Dataset saved to: {dataset_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy_args", nargs="*")
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--task_name", type=str)
    parser.add_argument("--env_cfg_type", type=str)
    parser.add_argument("--expert_data_num", type=int)
    parser.add_argument("--action_type", type=str, choices=["joint", "ee"])
    parser.add_argument("--source_lerobot_path", type=str, default=os.environ.get("LEROBOT_DATA_PATH"))
    parser.add_argument("--source_format", choices=["hdf5", "lerobot_v3"], default=None)
    parser.add_argument("--repo_id", type=str, default=os.environ.get("DREAMZERO_REPO_ID"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output_dir", type=str, default=str(POLICY_DIR / "data"))
    parser.add_argument("--action_horizon", type=int, default=24)
    parser.add_argument("--image_writer_processes", type=int, default=4)
    parser.add_argument("--image_writer_threads", type=int, default=4)
    args = parser.parse_args()

    if args.legacy_args:
        if len(args.legacy_args) == 5:
            args.dataset_name, args.task_name, args.env_cfg_type, expert_data_num, args.action_type = args.legacy_args
            args.expert_data_num = int(expert_data_num)
            args.source_format = args.source_format or "hdf5"
        elif len(args.legacy_args) == 4:
            args.dataset_name, args.env_cfg_type, expert_data_num, args.action_type = args.legacy_args
            args.expert_data_num = int(expert_data_num)
            args.source_format = args.source_format or "lerobot_v3"
        else:
            parser.error("Expected either 5 legacy HDF5 args or 4 multitask LeRobot v3 args.")

    required = ["dataset_name", "env_cfg_type", "expert_data_num", "action_type"]
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error(f"Missing required arguments: {', '.join(missing)}")

    source_format = args.source_format
    if source_format is None:
        source_path = Path(args.source_lerobot_path or DEFAULT_LEROBOT_V3_PATH)
        source_format = "lerobot_v3" if (source_path / "meta" / "info.json").exists() else "hdf5"

    if source_format == "lerobot_v3":
        convert_lerobot_v3(args)
    else:
        if not args.task_name:
            parser.error("task_name is required for HDF5 conversion.")
        convert(args)


if __name__ == "__main__":
    main()
