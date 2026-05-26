#!/usr/bin/env python3
"""Convert XPolicyLab HDF5 episodes to DreamZero AgiBot LeRobot/GEAR data."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

from XPolicyLab.utils.load_file import load_hdf5
from XPolicyLab.utils.process_data import (
    decode_image_bit,
    get_robot_action_dim_info,
    pack_robot_state,
)


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


def convert(args: argparse.Namespace) -> None:
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
    parser.add_argument("dataset_name", type=str)
    parser.add_argument("task_name", type=str)
    parser.add_argument("env_cfg_type", type=str)
    parser.add_argument("expert_data_num", type=int)
    parser.add_argument("action_type", type=str, choices=["joint", "ee"])
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output_dir", type=str, default=str(POLICY_DIR / "data"))
    parser.add_argument("--action_horizon", type=int, default=24)
    parser.add_argument("--image_writer_processes", type=int, default=4)
    parser.add_argument("--image_writer_threads", type=int, default=4)
    args = parser.parse_args()
    convert(args)


if __name__ == "__main__":
    main()
