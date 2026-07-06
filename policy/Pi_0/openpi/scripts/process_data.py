import os
import numpy as np
import shutil
import argparse
import cv2
import h5py
import dataclasses
from pathlib import Path
from typing import Any, Literal
import random
from tqdm import tqdm

from XPolicyLab.utils.load_file import load_yaml, load_json

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

ROOT_PATH = Path(__file__).parent.parent.parent.parent.parent

CAMERA_ALIASES = {
    "cam_head": "cam_high",
    "cam_left_wrist": "cam_left_wrist",
    "cam_right_wrist": "cam_right_wrist",
}

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = False
    tolerance_s: float = 0.0001
    image_writer_processes: int = 0
    image_writer_threads: int = 1
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()

def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    fps: int,
    mode: Literal["video", "image"] = "image",
    *,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    robot_action_dim_info: dict = None,
) -> LeRobotDataset:
    
    MOTORS = [
        *[f"left_{i}" for i in range(robot_action_dim_info["arm_dim"][0])],
        *[f"left_ee_{i}" for i in range(robot_action_dim_info["ee_dim"][0])],
        *[f"right_{i}" for i in range(robot_action_dim_info["arm_dim"][1])],
        *[f"right_ee_{i}" for i in range(robot_action_dim_info["ee_dim"][1])]
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
            "shape": (3, 480, 640),
            "names": ["height", "width", "channels"],
        }

    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
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
    return image # return RGB data not BGR data

def _load_compressed_images(group: h5py.Group, key: str) -> np.ndarray:
    frames = [decode_image(frame) for frame in group[key]]
    return np.asarray(frames)

def _make_action_from_state(state: np.ndarray) -> np.ndarray:
    action = np.empty_like(state, dtype=np.float32)
    if len(state) == 1:
        action[0] = state[0]
        return action

    action[:-1] = state[1:]
    action[-1] = state[-1]
    return action


def _decode_instruction(instruction: Any) -> str:
    if isinstance(instruction, (bytes, np.bytes_)):
        return bytes(instruction).decode("utf-8")
    return str(instruction)


def load_data(ep_path) -> dict[str, Any]:
    with h5py.File(ep_path, "r") as ep:
        right_state = np.concatenate(
            [ep["state/right_arm_joint_states"][:], ep["state/right_ee_joint_states"][:][:, None]],
            axis=1,
        )
        left_state = np.concatenate(
            [ep["state/left_arm_joint_states"][:], ep["state/left_ee_joint_states"][:][:, None]],
            axis=1,
        )
        state = np.concatenate([left_state, right_state], axis=1).astype(np.float32)
        action = _make_action_from_state(state)

        images = {}
        for source_name, output_name in CAMERA_ALIASES.items():
            if source_name in ep["vision"]:
                images[output_name] = _load_compressed_images(ep["vision"][source_name], "colors")
        try:
            instructions = [_decode_instruction(instruction) for instruction in ep["instructions"][:]]
        except KeyError:
            instructions = None

    return {
        "images": images,
        "state": state,
        "action": action,
        "velocity": None,
        "effort": None,
        "timestamps": None,
        "instructions": instructions,
    }

def main():
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument("bench_name", type=str, help="Dataset bench name (e.g., RoboDojo)")
    parser.add_argument("ckpt_name", type=str, help="Run name; also selects raw task dir under data/<bench>/")
    parser.add_argument("env_cfg_type", type=str, help="Environment config type (e.g., arx_x5)")
    parser.add_argument("action_type", type=str, help="Action type for artifact naming (e.g., joint)")
    parser.add_argument(
        "expert_data_num",
        type=int,
        nargs="?",
        default=None,
        help="Optional number of episodes to process; defaults to all episodes.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["video", "image"],
        default="image",
        help="Whether to store images as videos or individual image files",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Do your job.",
        help="Default instruction when not present in HDF5",
    )
    parser.add_argument(
        "--raw-task-dirs",
        type=str,
        default=None,
        help="Comma-separated raw task dirs under data/<bench_name>/; defaults to ckpt_name.",
    )
    args = parser.parse_args()

    bench_name = args.bench_name
    ckpt_name = args.ckpt_name
    env_cfg_type = args.env_cfg_type
    action_type = args.action_type
    repo_id = f"{bench_name}-{ckpt_name}-{env_cfg_type}-{action_type}"
    mode = args.mode
    instruction = args.instruction
    raw_task_dirs = [
        task_dir.strip()
        for task_dir in (args.raw_task_dirs or ckpt_name).split(",")
        if task_dir.strip()
    ]

    env_cfg = load_yaml(os.path.join(ROOT_PATH, "./env_cfg", f"{env_cfg_type}.yml"))
    robot_type = env_cfg['config']['robot']

    robot_action_dim_info = robot_action_dim_info = load_json(os.path.join(ROOT_PATH, "env_cfg/robot", "_robot_info.json"))[robot_type]

    dataset = create_empty_dataset(
        repo_id=repo_id,
        robot_type=robot_type,
        fps=50, # pi default to 50
        mode=mode,
        dataset_config=DEFAULT_DATASET_CONFIG,
        robot_action_dim_info=robot_action_dim_info,
    )

    episode_jobs = []
    for raw_task_dir in raw_task_dirs:
        load_data_dir = ROOT_PATH / "data" / str(bench_name) / raw_task_dir / str(env_cfg_type)
        episode_files = sorted(load_data_dir.glob("data/episode_*.hdf5"))
        if not episode_files:
            episode_files = sorted(load_data_dir.glob("*.hdf5"))
        if args.expert_data_num is not None:
            episode_files = episode_files[: args.expert_data_num]
        episode_jobs.extend((raw_task_dir, ep_file) for ep_file in episode_files)

    if not episode_jobs:
        raise FileNotFoundError(
            "No HDF5 episodes found for raw task dirs "
            f"{raw_task_dirs} under data/{bench_name}/<task>/{env_cfg_type}."
        )

    for raw_task_dir, ep_file in tqdm(episode_jobs, desc="Processing episodes", unit="episode"):
        try:
            data = load_data(ep_file)
            num_frames = data["state"].shape[0]

            for i in range(num_frames):
                frame = {
                    "observation.state": data["state"][i],
                    "action": data["action"][i],
                    "task": instruction if data["instructions"] is None else random.choice(data["instructions"]),
                }
                for camera_name, images in data["images"].items():
                    frame[f"observation.images.{camera_name}"] = images[i]

                dataset.add_frame(frame)
            
            dataset.save_episode()
            dataset.hf_dataset = dataset.create_hf_dataset()
            tqdm.write(f"Finished {raw_task_dir}/{ep_file.name} with {num_frames} frames")
        except Exception as e:
            tqdm.write(f"Error processing episode {ep_file}: {e}")

if __name__ == "__main__":
    main()
