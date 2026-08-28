"""Convert RoboDojo HDF5 demonstrations into a KinRT LeRobot dataset."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import random
import shutil
from typing import Any, Literal

import h5py
import numpy as np
from tqdm import tqdm

from XPolicyLab.utils.load_file import load_yaml
from XPolicyLab.utils.process_data import decode_image_bit, get_robot_action_dim_info

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset


CAMERA_ALIASES = {
    "cam_head": "cam_high",
    "cam_left_wrist": "cam_left_wrist",
    "cam_right_wrist": "cam_right_wrist",
}


@dataclasses.dataclass(frozen=True)
class DatasetWriterConfig:
    use_videos: bool = False
    tolerance_s: float = 0.0001
    image_writer_processes: int = 0
    image_writer_threads: int = 1
    video_backend: str | None = None


def _motor_names(robot_action_dim_info: dict[str, list[int]]) -> list[str]:
    if len(robot_action_dim_info["arm_dim"]) != 2 or len(robot_action_dim_info["ee_dim"]) != 2:
        raise ValueError("The RoboDojo KinRT adapter requires a dual-arm robot.")
    return [
        *[f"left_{index}" for index in range(robot_action_dim_info["arm_dim"][0])],
        *[f"left_ee_{index}" for index in range(robot_action_dim_info["ee_dim"][0])],
        *[f"right_{index}" for index in range(robot_action_dim_info["arm_dim"][1])],
        *[f"right_ee_{index}" for index in range(robot_action_dim_info["ee_dim"][1])],
    ]


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    fps: int,
    image_shape: tuple[int, int, int],
    robot_action_dim_info: dict[str, list[int]],
    *,
    mode: Literal["video", "image"] = "image",
    writer_config: DatasetWriterConfig = DatasetWriterConfig(),
    overwrite: bool = False,
) -> LeRobotDataset:
    motors = _motor_names(robot_action_dim_info)
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": motors,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": motors,
        },
    }
    height, width, channels = image_shape
    for camera_name in CAMERA_ALIASES.values():
        features[f"observation.images.{camera_name}"] = {
            "dtype": mode,
            "shape": (channels, height, width),
            "names": ["channels", "height", "width"],
        }

    output_path = Path(HF_LEROBOT_HOME) / repo_id
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Dataset already exists: {output_path}. Pass --overwrite to replace it.")
        shutil.rmtree(output_path)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=writer_config.use_videos or mode == "video",
        tolerance_s=writer_config.tolerance_s,
        image_writer_processes=writer_config.image_writer_processes,
        image_writer_threads=writer_config.image_writer_threads,
        video_backend=writer_config.video_backend,
    )


def _ensure_column(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    return value[:, None] if value.ndim == 1 else value


def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    return str(value)


def _read_instructions(episode: h5py.File) -> list[str]:
    for key in ("instruction", "instructions"):
        if key not in episode:
            continue
        values = np.asarray(episode[key][()])
        if values.ndim == 0:
            return [_decode_text(values.item())]
        return [_decode_text(value) for value in values.reshape(-1).tolist()]
    return []


def _read_camera_frames(episode: h5py.File, source_name: str) -> np.ndarray:
    if source_name not in episode["vision"]:
        raise KeyError(f"Missing required camera {source_name!r}.")
    camera_group = episode["vision"][source_name]
    for key in ("colors", "color"):
        if key in camera_group:
            return np.asarray(decode_image_bit(camera_group[key][()]))
    raise KeyError(f"Camera {source_name!r} has no color stream.")


def load_episode(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as episode:
        left_state = np.concatenate(
            [
                np.asarray(episode["state/left_arm_joint_states"]),
                _ensure_column(np.asarray(episode["state/left_ee_joint_states"])),
            ],
            axis=1,
        )
        right_state = np.concatenate(
            [
                np.asarray(episode["state/right_arm_joint_states"]),
                _ensure_column(np.asarray(episode["state/right_ee_joint_states"])),
            ],
            axis=1,
        )
        state = np.concatenate([left_state, right_state], axis=1).astype(np.float32)
        action = np.concatenate([state[1:], state[-1:]], axis=0)
        images = {
            output_name: _read_camera_frames(episode, source_name)
            for source_name, output_name in CAMERA_ALIASES.items()
        }
        instructions = _read_instructions(episode)

    frame_count = state.shape[0]
    for camera_name, frames in images.items():
        if frames.shape[0] != frame_count:
            raise ValueError(
                f"Camera {camera_name!r} has {frames.shape[0]} frames, but state has {frame_count}."
            )
    return {"state": state, "action": action, "images": images, "instructions": instructions}


def _find_episode_files(
    benchmark_root: Path,
    bench_name: str,
    raw_task_dirs: list[str],
    env_cfg_type: str,
    source_dir: Path | None = None,
) -> list[Path]:
    if source_dir is not None:
        source_dir = source_dir.expanduser().resolve()
        task_files = sorted((source_dir / "data").glob("episode_*.hdf5"))
        if not task_files:
            task_files = sorted(source_dir.glob("episode_*.hdf5"))
        if not task_files:
            raise FileNotFoundError(f"No HDF5 episodes found under {source_dir}.")
        return task_files

    episode_files: list[Path] = []
    for raw_task_dir in raw_task_dirs:
        source_dir = benchmark_root / "data" / bench_name / raw_task_dir / env_cfg_type
        task_files = sorted((source_dir / "data").glob("episode_*.hdf5"))
        if not task_files:
            task_files = sorted(source_dir.glob("*.hdf5"))
        if not task_files:
            raise FileNotFoundError(f"No HDF5 episodes found under {source_dir}.")
        episode_files.extend(task_files)
    return episode_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bench_name")
    parser.add_argument("ckpt_name")
    parser.add_argument("env_cfg_type")
    parser.add_argument("action_type", choices=("joint",))
    parser.add_argument("expert_data_num", nargs="?", default=None)
    parser.add_argument("raw_task_dirs", nargs="?", default=None)
    parser.add_argument("--repo-id", default="RoboDojo-KinRT-arx_x5-joint")
    parser.add_argument("--benchmark-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Exact embodiment directory containing data/episode_*.hdf5 or episode_*.hdf5.",
    )
    parser.add_argument("--mode", choices=("video", "image"), default="image")
    parser.add_argument("--metadata-fps", type=int, default=50)
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--fallback-instruction", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.bench_name != "RoboDojo":
        raise ValueError(f"This converter supports bench_name='RoboDojo', got {args.bench_name!r}.")
    if args.env_cfg_type != "arx_x5":
        raise ValueError(f"This converter supports env_cfg_type='arx_x5', got {args.env_cfg_type!r}.")

    expert_data_num = None
    raw_task_dirs_arg = args.raw_task_dirs
    if args.expert_data_num is not None:
        try:
            expert_data_num = int(args.expert_data_num)
        except ValueError:
            if args.raw_task_dirs is not None:
                raise ValueError("raw_task_dirs was provided twice.") from None
            raw_task_dirs_arg = args.expert_data_num
    raw_task_dirs = [item.strip() for item in (raw_task_dirs_arg or args.ckpt_name).split(",") if item.strip()]
    episode_files = _find_episode_files(
        args.benchmark_root,
        args.bench_name,
        raw_task_dirs,
        args.env_cfg_type,
        source_dir=args.source_dir,
    )
    if expert_data_num is not None:
        episode_files = episode_files[:expert_data_num]
    if not episode_files:
        raise ValueError("No episodes were selected for conversion.")

    first_episode = load_episode(episode_files[0])
    first_image = first_episode["images"]["cam_high"]
    if first_image.ndim != 4 or first_image.shape[-1] not in (1, 3):
        raise ValueError(f"Expected HWC image frames, got {first_image.shape}.")

    env_cfg = load_yaml(str(args.benchmark_root / "env_cfg" / f"{args.env_cfg_type}.yml"))
    robot_type = env_cfg["config"]["robot"]
    robot_action_dim_info = get_robot_action_dim_info(args.env_cfg_type)
    dataset = create_empty_dataset(
        repo_id=args.repo_id,
        robot_type=robot_type,
        fps=args.metadata_fps,
        image_shape=tuple(first_image.shape[1:]),
        robot_action_dim_info=robot_action_dim_info,
        mode=args.mode,
        writer_config=DatasetWriterConfig(
            use_videos=args.mode == "video",
            image_writer_processes=args.image_writer_processes,
            image_writer_threads=args.image_writer_threads,
        ),
        overwrite=args.overwrite,
    )

    rng = random.Random(args.seed)
    for episode_path in tqdm(episode_files, desc="Converting RoboDojo episodes", unit="episode"):
        data = first_episode if episode_path == episode_files[0] else load_episode(episode_path)
        fallback_instruction = args.fallback_instruction or episode_path.parents[2].name.replace("_", " ")
        instructions = data["instructions"] or [fallback_instruction]
        for frame_index in range(data["state"].shape[0]):
            frame = {
                "observation.state": data["state"][frame_index],
                "action": data["action"][frame_index],
                "task": rng.choice(instructions),
            }
            for camera_name, images in data["images"].items():
                frame[f"observation.images.{camera_name}"] = images[frame_index]
            dataset.add_frame(frame)
        dataset.save_episode()
        dataset.hf_dataset = dataset.create_hf_dataset()

    output_path = Path(HF_LEROBOT_HOME) / args.repo_id
    conversion_metadata = {
        "source": "RoboDojo",
        "env_cfg_type": args.env_cfg_type,
        "action_type": args.action_type,
        "camera_mapping": CAMERA_ALIASES,
        "state_action_order": "left_arm,left_gripper,right_arm,right_gripper",
        "action_target": "next_state_with_terminal_repeat",
        "source_frequency_hz": 25,
        "lerobot_metadata_fps": args.metadata_fps,
        "image_writer_processes": args.image_writer_processes,
        "image_writer_threads": args.image_writer_threads,
        "episodes": len(episode_files),
        "raw_task_dirs": raw_task_dirs,
        "source_dir": str(args.source_dir.expanduser().resolve()) if args.source_dir is not None else None,
    }
    metadata_path = output_path / "meta" / "kinrt_robodojo_conversion.json"
    metadata_path.write_text(json.dumps(conversion_metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote LeRobot dataset to: {output_path}")
    print(f"Wrote conversion metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
