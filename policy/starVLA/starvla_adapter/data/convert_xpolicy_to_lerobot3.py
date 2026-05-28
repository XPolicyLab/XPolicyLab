from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


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


def _pack_starvla_joint_order(group: h5py.Group) -> np.ndarray:
    """Return [left_joints, right_joints, left_gripper, right_gripper]."""
    left_arm = np.asarray(group["left_arm_joint_states"], dtype=np.float32)
    right_arm = np.asarray(group["right_arm_joint_states"], dtype=np.float32)
    left_ee = np.asarray(group["left_ee_joint_states"], dtype=np.float32)
    right_ee = np.asarray(group["right_ee_joint_states"], dtype=np.float32)
    return np.concatenate([left_arm, right_arm, left_ee, right_ee], axis=-1)


def _write_image_bytes(raw_bytes, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(raw_bytes, np.bytes_):
        raw_bytes = raw_bytes.tobytes()
    elif isinstance(raw_bytes, memoryview):
        raw_bytes = raw_bytes.tobytes()
    elif isinstance(raw_bytes, bytearray):
        raw_bytes = bytes(raw_bytes)
    with output_path.open("wb") as fp:
        fp.write(raw_bytes)


def _feature_image(height: int = 480, width: int = 640, fps: int = 30) -> dict:
    return {
        "dtype": "image",
        "shape": [height, width, 3],
        "names": ["height", "width", "channel"],
        "info": {
            "video.fps": fps,
            "video.height": height,
            "video.width": width,
            "video.channels": 3,
        },
    }


def _feature_vector(dim: int) -> dict:
    return {
        "dtype": "float32",
        "shape": [dim],
        "names": [f"dim_{idx}" for idx in range(dim)],
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
            "right_joints": {
                "start": 6,
                "end": 12,
                "absolute": True,
                "dtype": "float32",
                "original_key": "observation.state",
            },
            "left_gripper": {
                "start": 12,
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
            "right_joints": {
                "start": 6,
                "end": 12,
                "absolute": True,
                "dtype": "float32",
                "original_key": "action",
            },
            "left_gripper": {
                "start": 12,
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


def _write_info_json(dataset_dir: Path, total_episodes: int, total_frames: int, fps: int) -> None:
    info = {
        "codebase_version": "v3.0",
        "robot_type": "arx_x5",
        "fps": fps,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": 0,
        "chunks_size": 1000,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "observation.state": _feature_vector(14),
            "action": _feature_vector(14),
            "observation.images.cam_high": _feature_image(fps=fps),
            "observation.images.cam_left_wrist": _feature_image(fps=fps),
            "observation.images.cam_right_wrist": _feature_image(fps=fps),
        },
    }
    (dataset_dir / "meta" / "info.json").write_text(
        json.dumps(info, indent=2),
        encoding="utf-8",
    )


def convert(args: argparse.Namespace) -> None:
    if args.env_cfg_type != "arx_x5" or args.action_type != "joint":
        raise NotImplementedError("First starVLA converter supports arx_x5 + joint only.")

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
    fps = 30

    for episode_index, episode_path in enumerate(episode_paths):
        with h5py.File(episode_path, "r") as h5_file:
            instruction = _read_instruction(h5_file, task_instruction)
            task_instruction = instruction or task_instruction
            fps = int(np.asarray(h5_file["additional_info"]["frequency"]).item())

            state = _pack_starvla_joint_order(h5_file["state"])
            action = _pack_starvla_joint_order(h5_file["action"])
            length = min(len(state), len(action))

            for frame_index in range(length):
                row = {
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "timestamp": frame_index / float(fps),
                    "task_index": 0,
                    "observation.state": state[frame_index].astype(np.float32),
                    "action": action[frame_index].astype(np.float32),
                }
                for lerobot_key, xpolicy_camera in CAMERA_MAP.items():
                    rel_path = (
                        Path("images")
                        / xpolicy_camera
                        / f"episode_{episode_index:06d}"
                        / f"frame_{frame_index:06d}.jpg"
                    )
                    _write_image_bytes(
                        h5_file["vision"][xpolicy_camera]["colors"][frame_index],
                        dataset_dir / rel_path,
                    )
                    row[lerobot_key] = {"path": rel_path.as_posix()}
                all_rows.append(row)

            episode_rows.append(
                {
                    "episode_index": episode_index,
                    "length": length,
                    "tasks": [task_instruction],
                    "data/chunk_index": 0,
                    "data/file_index": 0,
                    "data/file_from_index": total_frames,
                    "data/file_to_index": total_frames + length,
                }
            )
            total_frames += length

    pd.DataFrame(all_rows).to_parquet(data_chunk_dir / "file-000.parquet", index=False)
    pd.DataFrame(episode_rows).to_parquet(episode_meta_dir / "file-000.parquet", index=False)

    tasks = pd.DataFrame({"task_index": [0]}, index=[task_instruction])
    tasks.to_parquet(dataset_dir / "meta" / "tasks.parquet")

    _write_modality_json(dataset_dir)
    _write_info_json(dataset_dir, total_episodes=len(episode_rows), total_frames=total_frames, fps=fps)

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
