#!/usr/bin/env python3
"""Convert XPolicyLab HDF5 episodes to GR00T LeRobot v2 format."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

POLICY_DIR = Path(__file__).resolve().parent
ROOT_DIR = POLICY_DIR.parents[2]
GR00T_DIR = POLICY_DIR / "Isaac-GR00T"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from XPolicyLab.utils.load_file import load_hdf5, load_yaml
from XPolicyLab.utils.process_data import (
    decode_image_bit,
    get_robot_action_dim_info,
    pack_robot_state,
)


IMAGE_SIZE = (320, 240)  # W, H
LANGUAGE_KEY = "annotation.human.task_description"


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _decode_text(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode_text(value.item())
        return [_decode_text(v) for v in value.tolist()]
    return value


def _load_instructions(data: dict, default_instruction: str) -> list[str]:
    raw = data.get("instructions", data.get("instruction", None))
    raw = _decode_text(raw)
    if raw is None:
        return [default_instruction]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(x) for x in parsed if str(x)]
        except json.JSONDecodeError:
            pass
        return [raw] if raw else [default_instruction]
    if isinstance(raw, (list, tuple)):
        items = [str(x) for x in raw if str(x)]
        return items or [default_instruction]
    return [str(raw)]


def _read_robot_type(env_cfg_type: str) -> str:
    cfg_path = ROOT_DIR / "env_cfg" / f"{env_cfg_type}.yml"
    env_cfg = load_yaml(str(cfg_path))
    return env_cfg["config"]["robot"]


def _state_modality_layout(action_type: str, dim_info: dict) -> list[tuple[str, int, str]]:
    arm_dims = dim_info["arm_dim"]
    ee_dims = dim_info["ee_dim"]
    if len(arm_dims) == 1:
        arm_name = "arm" if action_type == "joint" else "ee_pose"
        return [(arm_name, int(arm_dims[0]), "arm"), ("gripper", int(ee_dims[0]), "ee")]
    if len(arm_dims) == 2:
        arm_suffix = "arm" if action_type == "joint" else "ee_pose"
        return [
            (f"left_{arm_suffix}", int(arm_dims[0]), "arm"),
            ("left_gripper", int(ee_dims[0]), "ee"),
            (f"right_{arm_suffix}", int(arm_dims[1]), "arm"),
            ("right_gripper", int(ee_dims[1]), "ee"),
        ]
    raise ValueError(f"Only single-arm and dual-arm robots are supported, got {len(arm_dims)} arms")


def _camera_layout(dim_info: dict, vision: dict) -> list[tuple[str, str]]:
    cameras = []
    if "cam_head" in vision:
        cameras.append(("cam_head", "observation.images.cam_head"))
    if len(dim_info["arm_dim"]) == 1:
        if "cam_wrist" in vision:
            cameras.append(("cam_wrist", "observation.images.cam_wrist"))
        elif "cam_left_wrist" in vision:
            cameras.append(("cam_left_wrist", "observation.images.cam_wrist"))
    else:
        if "cam_left_wrist" in vision:
            cameras.append(("cam_left_wrist", "observation.images.cam_left_wrist"))
        if "cam_right_wrist" in vision:
            cameras.append(("cam_right_wrist", "observation.images.cam_right_wrist"))
    if not cameras:
        raise KeyError("No supported RGB camera found in episode vision data.")
    return cameras


def _normalize_single_arm_aliases(data: dict, action_type: str, dim_info: dict, source_type: str) -> dict:
    if action_type != "joint" or len(dim_info["arm_dim"]) != 1:
        return data
    suffix = "" if source_type == "obs" else "s"
    expected_key = f"joint_state{suffix}"
    alias_key = f"arm_joint_state{suffix}"
    for group_name in ("state", "action"):
        group = data.get(group_name)
        if isinstance(group, dict) and expected_key not in group and alias_key in group:
            group[expected_key] = group[alias_key]
    return data


def _decode_resize_bgr(raw) -> np.ndarray:
    image = decode_image_bit(raw)
    if image is None:
        raise ValueError("Failed to decode image bytes.")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 image, got {image.shape}")
    return cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA).astype(np.uint8)


def _write_video(path: Path, frames_bgr: list[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        IMAGE_SIZE,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {path}")
    try:
        for frame in frames_bgr:
            writer.write(frame)
    finally:
        writer.release()


def _build_modality_json(layout: list[tuple[str, int, str]], camera_layout: list[tuple[str, str]]) -> dict:
    state = {}
    offset = 0
    for key, dim, _kind in layout:
        state[key] = {"start": offset, "end": offset + dim}
        offset += dim

    video = {
        output_key.replace("observation.images.", ""): {"original_key": output_key}
        for _source_key, output_key in camera_layout
    }
    return {
        "state": state,
        "action": dict(state),
        "video": video,
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }


def _build_modality_config_py(
    path: Path,
    action_type: str,
    layout: list[tuple[str, int, str]],
    camera_layout: list[tuple[str, str]],
    action_horizon: int,
) -> None:
    video_keys = [output_key.replace("observation.images.", "") for _source_key, output_key in camera_layout]
    modality_keys = [key for key, _dim, _kind in layout]
    reps = []
    for _key, _dim, kind in layout:
        if kind == "ee":
            reps.append("ABSOLUTE")
        elif action_type == "ee":
            reps.append("ABSOLUTE")
        else:
            reps.append("RELATIVE")

    action_cfg_lines = []
    for rep in reps:
        action_cfg_lines.append(
            "            ActionConfig(\n"
            f"                rep=ActionRepresentation.{rep},\n"
            "                type=ActionType.NON_EEF,\n"
            "                format=ActionFormat.DEFAULT,\n"
            "            ),"
        )

    content = f'''"""GR00T modality config generated for XPolicyLab data."""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


xpolicylab_config = {{
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys={video_keys!r},
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys={modality_keys!r},
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, {action_horizon})),
        modality_keys={modality_keys!r},
        action_configs=[
{chr(10).join(action_cfg_lines)}
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["{LANGUAGE_KEY}"],
    ),
}}


register_modality_config(xpolicylab_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
'''
    path.write_text(content, encoding="utf-8")


def _write_info_json(
    path: Path,
    *,
    robot_type: str,
    fps: int,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    state_dim: int,
    camera_layout: list[tuple[str, str]],
) -> None:
    features = {
        "action": {"dtype": "float32", "shape": [state_dim], "names": [f"action_{i}" for i in range(state_dim)]},
        "observation.state": {
            "dtype": "float32",
            "shape": [state_dim],
            "names": [f"state_{i}" for i in range(state_dim)],
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "next.reward": {"dtype": "float32", "shape": [1], "names": None},
        "next.done": {"dtype": "bool", "shape": [1], "names": None},
    }
    for _source_key, video_key in camera_layout:
        features[video_key] = {
            "dtype": "video",
            "shape": [IMAGE_SIZE[1], IMAGE_SIZE[0], 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": IMAGE_SIZE[1],
                "video.width": IMAGE_SIZE[0],
                "video.codec": "mp4v",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": False,
            },
        }

    info = {
        "codebase_version": "v2.1",
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
        "total_chunks": 1,
        "total_videos": total_episodes * len(camera_layout),
    }
    path.write_text(json.dumps(info, indent=4, default=_json_default), encoding="utf-8")


def _episode_files(data_dir: Path, expert_data_num: int) -> list[Path]:
    files = sorted((data_dir / "data").glob("episode_*.hdf5"))
    if expert_data_num > 0:
        files = files[:expert_data_num]
    if not files:
        raise FileNotFoundError(f"No episode_*.hdf5 files found under {data_dir / 'data'}")
    return files


def convert_dataset(args: argparse.Namespace) -> Path:
    data_dir = ROOT_DIR / "data" / args.dataset_name / args.task_name / args.env_cfg_type
    if not data_dir.is_dir():
        alt = ROOT_DIR / "data" / args.task_name / args.env_cfg_type
        if alt.is_dir():
            data_dir = alt
        else:
            raise FileNotFoundError(f"Data directory not found: {data_dir} or {alt}")

    output_dir = Path(args.output_dir or POLICY_DIR / "data")
    dataset_id = args.dataset_id or (
        f"{args.dataset_name}-{args.task_name}-{args.env_cfg_type}-{args.expert_data_num}-{args.action_type}"
    )
    dataset_path = output_dir / dataset_id
    if dataset_path.exists():
        shutil.rmtree(dataset_path)

    (dataset_path / "meta").mkdir(parents=True)
    (dataset_path / "data" / "chunk-000").mkdir(parents=True)

    dim_info = get_robot_action_dim_info(args.env_cfg_type)
    robot_type = _read_robot_type(args.env_cfg_type)
    layout = _state_modality_layout(args.action_type, dim_info)
    state_dim = sum(dim for _key, dim, _kind in layout)

    task_to_index: dict[str, int] = {}
    episodes_meta = []
    total_frames = 0
    camera_layout = None

    files = _episode_files(data_dir, args.expert_data_num)
    for episode_index, episode_path in enumerate(tqdm(files, desc="Converting episodes", unit="episode")):
        data = load_hdf5(str(episode_path))
        data = _normalize_single_arm_aliases(data, args.action_type, dim_info, source_type="dataset")
        if camera_layout is None:
            camera_layout = _camera_layout(dim_info, data.get("vision", {}))

        state_all = pack_robot_state(
            data,
            args.action_type,
            dim_info,
            source_type="dataset",
            state_type="state",
        ).astype(np.float32)
        action_all = pack_robot_state(
            data,
            args.action_type,
            dim_info,
            source_type="dataset",
            state_type="action",
        ).astype(np.float32)
        if state_all.shape != action_all.shape:
            raise ValueError(f"State/action shape mismatch in {episode_path}: {state_all.shape} vs {action_all.shape}")
        if state_all.shape[-1] != state_dim:
            raise ValueError(f"Packed state dim mismatch: expected {state_dim}, got {state_all.shape[-1]}")

        instructions = _load_instructions(data, args.instruction or args.task_name.replace("_", " "))
        task_text = instructions[0]
        task_index = task_to_index.setdefault(task_text, len(task_to_index))

        rows = []
        frame_count = int(state_all.shape[0])
        for frame_index in range(frame_count):
            rows.append(
                {
                    "observation.state": state_all[frame_index].astype(np.float32),
                    "action": action_all[frame_index].astype(np.float32),
                    "timestamp": np.float32(frame_index / args.fps),
                    "frame_index": np.int64(frame_index),
                    "episode_index": np.int64(episode_index),
                    "index": np.int64(total_frames + frame_index),
                    "task_index": np.int64(task_index),
                    "next.reward": np.float32(0.0),
                    "next.done": bool(frame_index == frame_count - 1),
                }
            )

        parquet_path = dataset_path / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        pd.DataFrame(rows).to_parquet(parquet_path, index=False)

        for source_key, video_key in camera_layout:
            raw_frames = data["vision"][source_key]["colors"]
            frames = [_decode_resize_bgr(raw_frames[i]) for i in range(frame_count)]
            video_path = (
                dataset_path
                / "videos"
                / "chunk-000"
                / video_key
                / f"episode_{episode_index:06d}.mp4"
            )
            _write_video(video_path, frames, args.fps)

        episodes_meta.append({"episode_index": episode_index, "tasks": [task_text], "length": frame_count})
        total_frames += frame_count

    assert camera_layout is not None
    with (dataset_path / "meta" / "tasks.jsonl").open("w", encoding="utf-8") as f:
        for task, index in sorted(task_to_index.items(), key=lambda x: x[1]):
            f.write(json.dumps({"task_index": index, "task": task}, ensure_ascii=False) + "\n")
    with (dataset_path / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for item in episodes_meta:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    (dataset_path / "meta" / "modality.json").write_text(
        json.dumps(_build_modality_json(layout, camera_layout), indent=4),
        encoding="utf-8",
    )
    _write_info_json(
        dataset_path / "meta" / "info.json",
        robot_type=robot_type,
        fps=args.fps,
        total_episodes=len(episodes_meta),
        total_frames=total_frames,
        total_tasks=len(task_to_index),
        state_dim=state_dim,
        camera_layout=camera_layout,
    )
    _build_modality_config_py(
        dataset_path / "xpolicylab_gr00t_config.py",
        args.action_type,
        layout,
        camera_layout,
        args.action_horizon,
    )

    if not args.skip_stats:
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{GR00T_DIR}:{env.get('PYTHONPATH', '')}"
        subprocess.run(
            [
                "python",
                str(GR00T_DIR / "gr00t" / "data" / "stats.py"),
                "--dataset-path",
                str(dataset_path),
                "--embodiment-tag",
                "NEW_EMBODIMENT",
                "--modality-config-path",
                str(dataset_path / "xpolicylab_gr00t_config.py"),
            ],
            check=True,
            cwd=str(GR00T_DIR),
            env=env,
        )

    return dataset_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_name", type=str)
    parser.add_argument("task_name", type=str)
    parser.add_argument("env_cfg_type", type=str)
    parser.add_argument("expert_data_num", type=int)
    parser.add_argument("action_type", choices=["joint", "ee"])
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dataset-id", type=str, default=None)
    parser.add_argument("--skip-stats", action="store_true")
    args = parser.parse_args()

    dataset_path = convert_dataset(args)
    print(f"[GR00T process_data] Dataset saved to: {dataset_path}")
    print(f"[GR00T process_data] Modality config: {dataset_path / 'xpolicylab_gr00t_config.py'}")


if __name__ == "__main__":
    main()
