"""
Script to convert Aloha hdf5 data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal

import gc
import glob

import json
import h5py
from lerobot.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tqdm
import tyro
from PIL import Image

IMG_SIZE=256

CAMERAS = [
    "cam_high",
    "cam_left_wrist",
    "cam_right_wrist",
]

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 0
    image_writer_threads: int = 4
    video_backend: str | None = None
    fps: int = 30

DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = [
        "left_x",
        "left_y",
        "left_z",
        "left_rx",
        "left_ry",
        "left_rz",
        "left_w",
        "right_x",
        "right_y",
        "right_z",
        "right_rx",
        "right_ry",
        "right_rz",
        "right_w",
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
        "left_gripper",
        "right_gripper",
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    for cam in CAMERAS:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 256, 256),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )

def get_cameras(hdf5_files: list[Path]) -> list[str]:
    with h5py.File(hdf5_files[0], "r") as ep:
        # ignore depth channel, not currently handled
        return [key for key in ep["/observations/images"].keys() if "depth" not in key]  # noqa: SIM118

def has_velocity(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/qvel" in ep

def has_effort(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/effort" in ep


def load_episode_arrays(
    ep: h5py.File,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    state = ep["/observations/qpos"][:]
    action = ep["/action"][:]

    velocity = None
    if "/observations/qvel" in ep:
        velocity = ep["/observations/qvel"][:]

    effort = None
    if "/observations/effort" in ep:
        effort = ep["/observations/effort"][:]

    return state, action, velocity, effort

def load_image_frame(ep: h5py.File, camera: str, frame_idx: int) -> np.ndarray:
    image_ds = ep[f"/observations/images/{camera}"]
    if image_ds.ndim == 4:
        return np.asarray(image_ds[frame_idx])

    import cv2

    data = image_ds[frame_idx]

    if isinstance(data, np.ndarray):
        encoded = data if data.dtype == np.uint8 else np.frombuffer(data.tobytes(), dtype=np.uint8)
    elif isinstance(data, (bytes, bytearray, np.bytes_)):
        encoded = np.frombuffer(data, dtype=np.uint8)
    else:
        encoded = np.asarray(data, dtype=np.uint8)

    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    image = np.array(Image.fromarray(image).resize((IMG_SIZE, IMG_SIZE),resample=Image.BICUBIC))
    
    if image is None:
        raise ValueError(f"Failed to decode image for camera {camera} at frame {frame_idx}")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def add_frame_compat(dataset: LeRobotDataset, frame: dict, task: str) -> None:
    try:
        dataset.add_frame(frame, task=task)
    except TypeError:
        frame = {**frame, "task": task}
        dataset.add_frame(frame)


def save_episode_compat(dataset: LeRobotDataset, task: str) -> None:
    try:
        dataset.save_episode(task=task)
    except TypeError:
        dataset.save_episode()


def release_dataset_memory(dataset: LeRobotDataset) -> None:
    # Some LeRobot variants keep appending every saved episode to `hf_dataset`,
    # which makes RAM usage grow linearly until the process gets OOM-killed.
    if hasattr(dataset, "create_hf_dataset") and hasattr(dataset, "hf_dataset"):
        try:
            dataset.hf_dataset = dataset.create_hf_dataset()
        except Exception:
            pass


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(hdf5_files))

    cameras = get_cameras(hdf5_files)

    for ep_idx in tqdm.tqdm(episodes):
        try:
            ep_path = Path(hdf5_files[ep_idx])
            instruction_path = ep_path.parent.parent / "instructions.json"

            with open(instruction_path, "r", encoding="utf-8") as f:
                instruction_data = json.load(f)

            instructions = instruction_data.get("instructions", [])
            if isinstance(instructions, list) and instructions:
                first_instruction = str(instructions[0]).strip()
                if first_instruction:
                    instruction = first_instruction

            with h5py.File(ep_path, "r") as ep:
                state, action, velocity, effort = load_episode_arrays(ep)
                num_frames = state.shape[0]

                for i in range(num_frames):
                    frame = {
                        "observation.state": np.asarray(state[i]),
                        "action": np.asarray(action[i]),
                    }

                    for camera in cameras:
                        frame[f"observation.images.{camera}"] = load_image_frame(ep, camera, i)

                    if velocity is not None:
                        frame["observation.velocity"] = np.asarray(velocity[i])
                    if effort is not None:
                        frame["observation.effort"] = np.asarray(effort[i])

                    add_frame_compat(dataset, frame, instruction)

            save_episode_compat(dataset, instruction)
            release_dataset_memory(dataset)

            del state
            del action
            del velocity
            del effort
            gc.collect()

        except Exception as e:
            print(f"Error processing episode {ep_idx}: {e}")
            continue

    return dataset


def port_aloha(
    raw_dir: Path,
    repo_id: str,
    is_multi: bool = False,
    *,
    episodes: list[int] | None = None,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    if (HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    if is_multi:
        hdf5_files = [Path(p) for p in glob.glob(f"{raw_dir}/**/*.hdf5", recursive=True)]
    else:
        hdf5_files = sorted(raw_dir.glob("episode_*.hdf5"))
        print(hdf5_files)

    dataset = create_empty_dataset(
        repo_id,
        robot_type="mobile_aloha" if is_mobile else "aloha",
        mode=mode,
        has_effort=has_effort(hdf5_files),
        has_velocity=has_velocity(hdf5_files),
        dataset_config=dataset_config,
    )

    dataset = populate_dataset(
        dataset,
        hdf5_files,
        episodes=episodes,
    )
    if hasattr(dataset, "stop_image_writer"):
        dataset.stop_image_writer()


if __name__ == "__main__":
    tyro.cli(port_aloha)
