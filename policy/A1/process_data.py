#!/usr/bin/env python3
"""Convert XPolicyLab HDF5 episodes to LeRobot format for A1 training."""

import argparse
import copy
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from tqdm import tqdm

try:
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
except ImportError:
    from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

from XPolicyLab.utils.load_file import load_hdf5, load_json, load_yaml
from XPolicyLab.utils.process_data import decode_image_bit, get_robot_action_dim_info, pack_robot_state


ROOT_PATH = Path(__file__).parent.parent.parent.parent

CAMERA_ALIASES = {
    "cam_head": "observation.images.cam_head",
    "cam_right_wrist": "observation.images.cam_right_wrist",
    "cam_left_wrist": "observation.images.cam_left_wrist",
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
    for group_name in ("state", "action"):
        group = data.get(group_name, {})
        for arm in ("left", "right"):
            pose_key = f"{arm}_ee_poses"
            if pose_key in group:
                group[pose_key] = _pose7_to_pose6(group[pose_key])
        if "ee_poses" in group:
            group["ee_poses"] = _pose7_to_pose6(group["ee_poses"])

    state = data.get("state", {})
    action = data.setdefault("action", {})
    for pose_key in ("ee_poses", "left_ee_poses", "right_ee_poses"):
        if pose_key not in action and pose_key in state:
            action[pose_key] = state[pose_key]
    for ee_key in ("ee_joint_states", "left_ee_joint_states", "right_ee_joint_states"):
        if ee_key not in action and ee_key in state:
            action[ee_key] = state[ee_key]
    return data


@dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = False
    tolerance_s: float = 0.0001
    image_writer_processes: int = 0
    image_writer_threads: int = 1
    video_backend: str | None = None


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    fps: int,
    mode: Literal["video", "image"] = "image",
    *,
    dataset_config: DatasetConfig = DatasetConfig(),
    robot_action_dim_info: dict = None,
    root: str = None,
) -> LeRobotDataset:
    names = []
    for arm_idx, (arm_dim, ee_dim) in enumerate(zip(robot_action_dim_info["arm_dim"], robot_action_dim_info["ee_dim"])):
        prefix = "" if len(robot_action_dim_info["arm_dim"]) == 1 else ("left_" if arm_idx == 0 else "right_")
        names.extend([f"{prefix}arm_{i}" for i in range(arm_dim)])
        names.extend([f"{prefix}ee_{i}" for i in range(ee_dim)])

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(names),),
            "names": names,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(names),),
            "names": names,
        },
    }

    for camera_name in CAMERA_ALIASES.values():
        features[camera_name] = {
            "dtype": mode,
            "shape": (3, 240, 320),
            "names": ["channels", "height", "width"],
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


def load_data(ep_path: str | Path, action_type: str, robot_action_dim_info: dict) -> dict[str, Any]:
    data = load_hdf5(ep_path)
    if action_type == "ee":
        data = _prepare_ee_data_schema(data)

    state = pack_robot_state(
        data,
        action_type,
        robot_action_dim_info,
        source_type="dataset",
        state_type="state",
    ).astype(np.float32)
    action = pack_robot_state(
        data,
        action_type,
        robot_action_dim_info,
        source_type="dataset",
        state_type="action",
    ).astype(np.float32)

    images = {}
    vision = data.get("vision", {})
    for source_name, output_name in CAMERA_ALIASES.items():
        if source_name not in vision or "colors" not in vision[source_name]:
            continue
        raw_imgs = decode_image_bit(vision[source_name]["colors"])
        processed = []
        for img in raw_imgs:
            img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA)
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
    parser = argparse.ArgumentParser(description="Convert XPolicyLab HDF5 data to LeRobot format for A1.")
    parser.add_argument("dataset_name", type=str)
    parser.add_argument("task_name", type=str)
    parser.add_argument("env_cfg_type", type=str)
    parser.add_argument("expert_data_num", type=int)
    parser.add_argument("action_type", type=str, choices=["joint", "ee"])
    parser.add_argument("--repo_id", type=str, default=None)
    parser.add_argument("--mode", type=str, choices=["video", "image"], default="image")
    parser.add_argument("--instruction", type=str, default="Do your job.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    repo_id = args.repo_id or f"{args.dataset_name}-{args.task_name}-{args.env_cfg_type}"
    load_data_dir = ROOT_PATH / "data" / args.dataset_name / args.task_name / args.env_cfg_type
    if not load_data_dir.is_dir():
        alt = ROOT_PATH / "data" / args.task_name / args.env_cfg_type
        if alt.is_dir():
            load_data_dir = alt
        else:
            raise FileNotFoundError(f"Data directory not found: {load_data_dir} or {alt}")

    ep_files = sorted((load_data_dir / "data").glob("episode_*.hdf5"))[: args.expert_data_num]
    if len(ep_files) < args.expert_data_num:
        raise ValueError(f"Expected {args.expert_data_num} episodes, found {len(ep_files)} in {load_data_dir / 'data'}")

    env_cfg = load_yaml(ROOT_PATH / "env_cfg" / f"{args.env_cfg_type}.yml")
    robot_type = env_cfg["config"]["robot"]
    robot_action_dim_info = load_json(ROOT_PATH / "env_cfg" / "robot" / "_robot_info.json")[robot_type]
    dataset = create_empty_dataset(
        repo_id=repo_id,
        robot_type=robot_type,
        fps=args.fps,
        mode=args.mode,
        robot_action_dim_info=robot_action_dim_info,
        root=args.output_dir,
    )

    for ep_file in tqdm(ep_files, desc="Converting episodes"):
        ep = load_data(ep_file, args.action_type, robot_action_dim_info)
        frames = ep["state"].shape[0]
        for i in range(frames):
            instruction = args.instruction
            if ep["instructions"]:
                instruction = ep["instructions"][min(i, len(ep["instructions"]) - 1)]
                if isinstance(instruction, bytes):
                    instruction = instruction.decode("utf-8")

            frame = {
                "observation.state": ep["state"][i],
                "action": ep["action"][i],
            }
            for camera_name, imgs in ep["images"].items():
                if i < len(imgs):
                    frame[camera_name] = imgs[i]
            dataset.add_frame(frame, task=str(instruction))
        dataset.save_episode()
        dataset.hf_dataset = dataset.create_hf_dataset()


if __name__ == "__main__":
    main()
