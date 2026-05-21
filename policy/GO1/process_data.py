#!/usr/bin/env python3
"""Convert XPolicyLab HDF5 episodes to LeRobot dataset format for GO1 training."""

import os
import argparse
import random
import sys
import copy
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import h5py
import numpy as np
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

from XPolicyLab.utils.load_file import load_yaml, load_json, load_hdf5
from XPolicyLab.utils.process_data import get_robot_action_dim_info, pack_robot_state, decode_image_bit

# ROOT_PATH points to demo_env/
ROOT_PATH = Path(__file__).parent.parent.parent.parent

# Camera name mapping: XPolicyLab HDF5 key -> LeRobot output key
CAMERA_ALIASES = {
    "cam_head": "cam_head",
    "cam_left_wrist": "cam_hand_left",
    "cam_right_wrist": "cam_hand_right",
}


def _quat_wxyz_to_rpy(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1.0, np.sign(sinp) * (np.pi / 2.0), np.arcsin(sinp))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.stack([roll, pitch, yaw], axis=-1).astype(np.float32)


def _pose7_to_pose6(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape[-1] != 7:
        return pose
    return np.concatenate([pose[..., :3], _quat_wxyz_to_rpy(pose[..., 3:7])], axis=-1).astype(np.float32)


def _prepare_ee_data_schema(data: dict) -> dict:
    data = copy.deepcopy(data)

    for group_name in ["state", "action"]:
        group = data.get(group_name, {})
        for arm in ["left", "right"]:
            pose_key = f"{arm}_ee_poses"
            if pose_key in group:
                group[pose_key] = _pose7_to_pose6(group[pose_key])

    state = data.get("state", {})
    action = data.setdefault("action", {})
    for arm in ["left", "right"]:
        action_pose_key = f"{arm}_ee_poses"
        state_pose_key = f"{arm}_ee_poses"
        if action_pose_key not in action and state_pose_key in state:
            action[action_pose_key] = state[state_pose_key]

        action_ee_key = f"{arm}_ee_joint_states"
        if action_ee_key not in action and action_ee_key in state:
            action[action_ee_key] = state[action_ee_key]

    return data
@dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = False
    tolerance_s: float = 0.0001
    image_writer_processes: int = 4
    image_writer_threads: int = 4
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig(
    image_writer_processes=max(1, min(8, (os.cpu_count() or 4) // 2)),
    image_writer_threads=max(2, min(8, os.cpu_count() or 4)),
)


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    fps: int,
    mode: Literal["video", "image"] = "image",
    *,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    robot_action_dim_info: dict = None,
    root: str = None,
) -> LeRobotDataset:
    MOTORS = [
        *[f"left_{i}" for i in range(robot_action_dim_info["arm_dim"][0])],
        *[f"left_ee_{i}" for i in range(robot_action_dim_info["ee_dim"][0])],
        *[f"right_{i}" for i in range(robot_action_dim_info["arm_dim"][1])],
        *[f"right_ee_{i}" for i in range(robot_action_dim_info["ee_dim"][1])],
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(MOTORS),),
            "names": MOTORS,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(MOTORS),),
            "names": [MOTORS],
        },
    }

    for camera_name in CAMERA_ALIASES.values():
        features[f"observation.images.{camera_name}"] = {
            "dtype": mode,
            "shape": (3, 240, 320),
            "names": ["height", "width", "channels"],
        }

    output_path = Path(root) / repo_id if root else HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        root=output_path,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def decode_image(img_bytes: bytes | np.bytes_ | np.ndarray) -> np.ndarray:
    if isinstance(img_bytes, (bytes, np.bytes_)):
        jpeg_bytes = bytes(img_bytes).rstrip(b"\0")
    elif isinstance(img_bytes, np.ndarray) and img_bytes.dtype.kind in ("S", "U"):
        jpeg_bytes = img_bytes.item().rstrip(b"\0")
    else:
        raise TypeError(f"Unsupported image payload type: {type(img_bytes)!r}")

    image = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode JPEG image from HDF5 payload.")
    return image


def _load_compressed_images(group: h5py.Group, key: str) -> np.ndarray:
    frames = [decode_image(frame) for frame in group[key]]
    return np.asarray(frames)


def load_data(ep_path: str | Path, action_type: str, robot_action_dim_info: dict) -> dict[str, Any]:
    """Load one HDF5 episode using the framework's pack_robot_state utility.

    Supports both 'joint' and 'ee' action_type, single-arm and dual-arm.
    """
    data = load_hdf5(ep_path)
    if action_type == "ee":
        data = _prepare_ee_data_schema(data)

    state = pack_robot_state(
        data, action_type, robot_action_dim_info,
        source_type="dataset", state_type="state",
    ).astype(np.float32)

    action = pack_robot_state(
        data, action_type, robot_action_dim_info,
        source_type="dataset", state_type="action",
    ).astype(np.float32)

    images = {}
    vision = data.get("vision", {})
    for source_name, output_name in CAMERA_ALIASES.items():
        if source_name in vision and "colors" in vision[source_name]:
            raw_imgs = decode_image_bit(vision[source_name]["colors"])  # (T, H, W, 3) BGR
            processed = []
            for img in raw_imgs:
                img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA)  # -> (240, 320, 3)
                # Training and inference both convert arrays to PIL/ImageNet-style RGB tensors.
                # Keeping BGR here would silently corrupt channel semantics.
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                processed.append(img)
            images[output_name] = np.asarray(processed)

    try:
        instructions = list(data["instructions"])
    except (KeyError, TypeError):
        instructions = None

    return {
        "images": images,
        "state": state,
        "action": action,
        "instructions": instructions,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert XPolicyLab HDF5 data to LeRobot format for GO1.")
    parser.add_argument("dataset_name", type=str, help="Dataset name (e.g., RoboDojo)")
    parser.add_argument("task_name", type=str, help="Task name (e.g., stack_bowls)")
    parser.add_argument("env_cfg_type", type=str, help="Environment config type (e.g., arx_x5)")
    parser.add_argument("expert_data_num", type=int, help="Number of expert episodes to convert")
    parser.add_argument("action_type", type=str, help="Action type (joint or ee)")
    parser.add_argument("--repo_id", type=str, default=None, help="LeRobot repo ID. Defaults to {dataset_name}-{task_name}-{env_cfg_type}")
    parser.add_argument("--mode", type=str, choices=["video", "image"], default="image", help="Storage mode")
    parser.add_argument("--instruction", type=str, default="Do your job.", help="Default instruction if not in data")
    parser.add_argument("--fps", type=int, default=30, help="Dataset FPS (GO1 default: 30)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for LeRobot dataset. Defaults to HF_LEROBOT_HOME.")
    args = parser.parse_args()

    if args.repo_id is None:
        args.repo_id = (
            f"{args.dataset_name}-{args.task_name}-{args.env_cfg_type}-"
            f"{args.expert_data_num}-{args.action_type}"
        )

    load_data_dir = os.path.join(ROOT_PATH, "data", args.dataset_name, args.task_name, args.env_cfg_type)
    if not os.path.isdir(load_data_dir):
        # Fallback: try without dataset_name level (some setups use data/{task_name}/{env_cfg_type})
        load_data_dir_alt = os.path.join(ROOT_PATH, "data", args.task_name, args.env_cfg_type)
        if os.path.isdir(load_data_dir_alt):
            load_data_dir = load_data_dir_alt
        else:
            raise FileNotFoundError(f"Data directory not found: {load_data_dir} or {load_data_dir_alt}")

    env_cfg = load_yaml(os.path.join(ROOT_PATH, "env_cfg", f"{args.env_cfg_type}.yml"))
    robot_type = env_cfg["config"]["robot"]
    robot_action_dim_info = load_json(
        os.path.join(ROOT_PATH, "env_cfg/robot", "_robot_info.json")
    )[robot_type]

    print(f"[GO1 process_data] Dataset: {args.dataset_name}, Task: {args.task_name}")
    print(f"[GO1 process_data] Robot: {robot_type}, Action dim info: {robot_action_dim_info}")
    print(f"[GO1 process_data] Output repo_id: {args.repo_id}, FPS: {args.fps}")
    print(
        "[GO1 process_data] Image writer config: "
        f"{DEFAULT_DATASET_CONFIG.image_writer_processes} processes / "
        f"{DEFAULT_DATASET_CONFIG.image_writer_threads} threads"
    )

    dataset = create_empty_dataset(
        repo_id=args.repo_id,
        robot_type=robot_type,
        fps=args.fps,
        mode=args.mode,
        dataset_config=DEFAULT_DATASET_CONFIG,
        robot_action_dim_info=robot_action_dim_info,
        root=args.output_dir,
    )

    episode_files = sorted(Path(load_data_dir).glob("data/episode_*.hdf5"))
    if args.expert_data_num > 0:
        episode_files = episode_files[: args.expert_data_num]

    print(f"[GO1 process_data] Found {len(episode_files)} episodes to process")

    for ep_file in tqdm(episode_files, desc="Processing episodes", unit="episode"):
        try:
            data = load_data(ep_file, args.action_type, robot_action_dim_info)
            num_frames = data["state"].shape[0]

            for i in range(num_frames):
                task_str = args.instruction if data["instructions"] is None else random.choice(data["instructions"])
                frame = {
                    "observation.state": data["state"][i],
                    "action": data["action"][i],
                }
                for camera_name, images in data["images"].items():
                    frame[f"observation.images.{camera_name}"] = images[i]

                dataset.add_frame(frame, task=task_str)

            dataset.save_episode()
            tqdm.write(f"Finished {ep_file.name} with {num_frames} frames")
        except Exception as e:
            tqdm.write(f"Error processing episode {ep_file}: {e}")

    output_root = Path(args.output_dir) if args.output_dir else HF_LEROBOT_HOME
    print(f"[GO1 process_data] Done! LeRobot dataset saved to: {output_root / args.repo_id}")


if __name__ == "__main__":
    main()
