from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from XPolicyLab.utils.process_data import (
    decode_image_bit,
    get_robot_action_dim_info,
    pack_robot_state,
)


CAMERA_MAP = {
    "observation.images.cam_high": "cam_head",
    "observation.images.cam_left_wrist": "cam_left_wrist",
    "observation.images.cam_right_wrist": "cam_right_wrist",
}


def _decode_scalar(value):
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _read_instruction(h5_file: h5py.File, fallback: str) -> str:
    if "instruction" not in h5_file:
        return fallback
    return str(_decode_scalar(h5_file["instruction"][()]) or fallback)


def _decode_bgr_image(raw_bytes) -> np.ndarray:
    image = decode_image_bit(raw_bytes)
    if image is None:
        raise ValueError("Failed to decode compressed image bytes.")
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got {image.shape}.")
    if image.shape[-1] != 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] != 3:
        raise ValueError(f"Expected 3-channel image, got {image.shape}.")
    return image.astype(np.uint8, copy=False)


def _write_video_from_hdf5_camera(
    h5_file: h5py.File,
    camera_name: str,
    output_path: Path,
    length: int,
    fps: int,
    width: int = 640,
    height: int = 480,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path.as_posix(), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    try:
        colors = h5_file["vision"][camera_name]["colors"]
        for frame_index in range(length):
            image = _decode_bgr_image(colors[frame_index])
            if image.shape[:2] != (height, width):
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(image)
    finally:
        writer.release()


def _feature_image(height: int = 480, width: int = 640, fps: int = 30) -> dict:
    return {
        "dtype": "video",
        "shape": [3, height, width],
        "names": ["channels", "height", "width"],
        "info": {
            "video.height": height,
            "video.width": width,
            "video.codec": "mp4v",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": fps,
            "video.channels": 3,
            "has_audio": False,
        },
    }


def _feature_vector(names: list[str]) -> dict:
    return {
        "dtype": "float32",
        "shape": [len(names)],
        "names": [names],
    }


def _write_modality_json(dataset_dir: Path) -> None:
    modality = {
        "state": {
            "left_joints": {
                "start": 0,
                "end": 6,
                "absolute": True,
                "dtype": "float32",
                "original_key": "observation.state",
            },
            "left_gripper": {
                "start": 6,
                "end": 7,
                "absolute": True,
                "dtype": "float32",
                "original_key": "observation.state",
            },
            "right_joints": {
                "start": 7,
                "end": 13,
                "absolute": True,
                "dtype": "float32",
                "original_key": "observation.state",
            },
            "right_gripper": {
                "start": 13,
                "end": 14,
                "absolute": True,
                "dtype": "float32",
                "original_key": "observation.state",
            },
        },
        "action": {
            "left_joints": {
                "start": 0,
                "end": 6,
                "absolute": True,
                "dtype": "float32",
                "original_key": "action",
            },
            "left_gripper": {
                "start": 6,
                "end": 7,
                "absolute": True,
                "dtype": "float32",
                "original_key": "action",
            },
            "right_joints": {
                "start": 7,
                "end": 13,
                "absolute": True,
                "dtype": "float32",
                "original_key": "action",
            },
            "right_gripper": {
                "start": 13,
                "end": 14,
                "absolute": True,
                "dtype": "float32",
                "original_key": "action",
            },
        },
        "video": {
            "cam_high": {"original_key": "observation.images.cam_high"},
            "cam_left_wrist": {"original_key": "observation.images.cam_left_wrist"},
            "cam_right_wrist": {"original_key": "observation.images.cam_right_wrist"},
        },
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"},
        },
    }
    (dataset_dir / "meta").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "meta" / "modality.json").write_text(
        json.dumps(modality, indent=2),
        encoding="utf-8",
    )


def _write_info_json(
    dataset_dir: Path,
    total_episodes: int,
    total_frames: int,
    total_videos: int,
    fps: int,
) -> None:
    info = {
        "codebase_version": "v3.0",
        "robot_type": "unified_robot",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "fps": fps,
        "splits": {
            "train": f"0:{total_episodes}",
        },
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.state": _feature_vector(
                [
                    "left_joint_0",
                    "left_joint_1",
                    "left_joint_2",
                    "left_joint_3",
                    "left_joint_4",
                    "left_joint_5",
                    "left_joint_6",
                    "right_joint_0",
                    "right_joint_1",
                    "right_joint_2",
                    "right_joint_3",
                    "right_joint_4",
                    "right_joint_5",
                    "right_joint_6",
                ]
            ),
            "action": _feature_vector(
                [
                    "left_joint_0",
                    "left_joint_1",
                    "left_joint_2",
                    "left_joint_3",
                    "left_joint_4",
                    "left_joint_5",
                    "left_joint_6",
                    "right_joint_0",
                    "right_joint_1",
                    "right_joint_2",
                    "right_joint_3",
                    "right_joint_4",
                    "right_joint_5",
                    "right_joint_6",
                ]
            ),
            "observation.images.cam_high": _feature_image(fps=fps),
            "observation.images.cam_left_wrist": _feature_image(fps=fps),
            "observation.images.cam_right_wrist": _feature_image(fps=fps),
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (dataset_dir / "meta" / "info.json").write_text(
        json.dumps(info, indent=2),
        encoding="utf-8",
    )


def convert(args: argparse.Namespace) -> None:
    if args.env_cfg_type != "arx_x5" or args.action_type != "joint":
        raise NotImplementedError("First starVLA converter supports arx_x5 + joint only.")
    robot_action_dim_info = get_robot_action_dim_info(args.env_cfg_type)

    source_dir = (
        Path(args.root_dir).resolve()
        / "data"
        / args.dataset_name
        / args.ckpt_name
        / args.env_cfg_type
    )
    hdf5_dir = source_dir / "data"
    if not hdf5_dir.is_dir():
        raise FileNotFoundError(f"Missing XPolicy HDF5 data dir: {hdf5_dir}")

    output_root = Path(args.output_dir).resolve()
    dataset_dir = output_root / "arx_x5"
    if dataset_dir.exists() and not args.keep_existing:
        shutil.rmtree(dataset_dir)

    data_chunk_dir = dataset_dir / "data" / "chunk-000"
    episode_meta_dir = dataset_dir / "meta" / "episodes" / "chunk-000"
    data_chunk_dir.mkdir(parents=True, exist_ok=True)
    episode_meta_dir.mkdir(parents=True, exist_ok=True)

    episode_paths = sorted(hdf5_dir.glob("episode_*.hdf5"))[: int(args.expert_data_num)]
    if not episode_paths:
        raise FileNotFoundError(f"No episode_*.hdf5 files found in {hdf5_dir}")

    all_rows = []
    episode_rows = []
    task_instruction = args.ckpt_name.replace("_", " ")
    total_frames = 0
    total_videos = 0
    fps = 30

    episode_iter = tqdm(
        enumerate(episode_paths),
        total=len(episode_paths),
        desc="[starVLA] episodes",
        unit="episode",
    )
    for episode_index, episode_path in episode_iter:
        with h5py.File(episode_path, "r") as h5_file:
            instruction = _read_instruction(h5_file, task_instruction)
            task_instruction = instruction or task_instruction
            fps = int(np.asarray(h5_file["additional_info"]["frequency"]).item())

            dataset_like = {
                "state": h5_file["state"],
                "action": h5_file["action"],
            }
            state = pack_robot_state(
                dataset_like,
                args.action_type,
                robot_action_dim_info,
                source_type="dataset",
                state_type="state",
            ).astype(np.float32)
            action = pack_robot_state(
                dataset_like,
                args.action_type,
                robot_action_dim_info,
                source_type="dataset",
                state_type="action",
            ).astype(np.float32)
            length = min(len(state), len(action))
            episode_iter.set_postfix(file=episode_path.name, frames=length)

            video_meta = {}
            for lerobot_key, xpolicy_camera in CAMERA_MAP.items():
                video_path = (
                    dataset_dir
                    / "videos"
                    / lerobot_key
                    / "chunk-000"
                    / f"file-{episode_index:03d}.mp4"
                )
                _write_video_from_hdf5_camera(
                    h5_file,
                    xpolicy_camera,
                    video_path,
                    length,
                    fps,
                )
                video_meta[f"videos/{lerobot_key}/from_timestamp"] = 0.0
                video_meta[f"videos/{lerobot_key}/chunk_index"] = 0
                video_meta[f"videos/{lerobot_key}/file_index"] = episode_index
                total_videos += 1

            for frame_index in tqdm(
                range(length),
                desc=f"[starVLA] {episode_path.stem}",
                unit="frame",
                leave=False,
            ):
                global_frame_index = total_frames + frame_index
                row = {
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "timestamp": frame_index / float(fps),
                    "task_index": 0,
                    "index": global_frame_index,
                    "observation.state": state[frame_index].astype(np.float32),
                    "action": action[frame_index].astype(np.float32),
                }
                all_rows.append(row)

            episode_row = {
                "episode_index": episode_index,
                "length": length,
                "tasks": [task_instruction],
                "data/chunk_index": 0,
                "data/file_index": 0,
                "data/file_from_index": total_frames,
                "data/file_to_index": total_frames + length,
                "dataset_from_index": total_frames,
                "dataset_to_index": total_frames + length,
            }
            episode_row.update(video_meta)
            episode_rows.append(episode_row)
            total_frames += length

    pd.DataFrame(all_rows).to_parquet(data_chunk_dir / "file-000.parquet", index=False)
    pd.DataFrame(episode_rows).to_parquet(episode_meta_dir / "file-000.parquet", index=False)

    tasks = pd.DataFrame({"task_index": [0]}, index=[task_instruction])
    tasks.to_parquet(dataset_dir / "meta" / "tasks.parquet")

    _write_modality_json(dataset_dir)
    _write_info_json(
        dataset_dir,
        total_episodes=len(episode_rows),
        total_frames=total_frames,
        total_videos=total_videos,
        fps=fps,
    )

    print(f"[starVLA] wrote LeRobot v3 dataset: {dataset_dir}")
    print(f"[starVLA] episodes={len(episode_rows)}, frames={total_frames}, task={task_instruction!r}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--ckpt_name", required=True)
    parser.add_argument("--env_cfg_type", required=True)
    parser.add_argument("--expert_data_num", required=True, type=int)
    parser.add_argument("--action_type", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--keep_existing", action="store_true")
    return parser


if __name__ == "__main__":
    convert(build_argparser().parse_args())
