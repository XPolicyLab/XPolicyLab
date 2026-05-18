import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import h5py
import numpy as np


CAMERA_DATASETS = {
    "head_camera_rgb": "observation/head_camera/rgb",
    "left_camera_rgb": "observation/left_camera/rgb",
    "right_camera_rgb": "observation/right_camera/rgb",
}

CAMERA_VIDEOS = {
    "observation.images.cam_high": "head_camera_rgb",
    "observation.images.cam_left_wrist": "left_camera_rgb",
    "observation.images.cam_right_wrist": "right_camera_rgb",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert RobotWin dual-arm EE data into Spirit training format.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/vepfs-cnbje63de6fae220/xspark_shared/robotwin_data"),
        help="RobotWin root. The script will use raw/ under this path when present.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output directory for the converted Spirit-format dataset.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="aloha-agilex_clean_50",
        help="Task-local dataset variant to export, for example aloha-agilex_clean_50.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Optional comma-separated task names. Defaults to all task folders under the RobotWin raw root.",
    )
    parser.add_argument(
        "--max-episodes-per-task",
        type=int,
        default=None,
        help="Optional limit per task for smoke tests.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=50.0,
        help="Frame rate used to write timestamps into states.jsonl.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing output files in the target root before conversion.",
    )
    return parser.parse_args()


def humanize_task_name(task_name: str) -> str:
    return task_name.replace("_", " ").strip()


def resolve_raw_root(raw_root: Path) -> Path:
    raw_dir = raw_root / "raw"
    if raw_dir.is_dir():
        return raw_dir
    return raw_root


def discover_tasks(raw_root: Path, tasks_csv: str | None) -> List[str]:
    if tasks_csv:
        return [task.strip() for task in tasks_csv.split(",") if task.strip()]

    tasks = []
    for child in sorted(raw_root.iterdir()):
        if child.is_dir() and not child.name.startswith("."):
            tasks.append(child.name)
    return tasks


def episode_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)", path.stem)
    if match is None:
        return -1, path.name
    return int(match.group(1)), path.name


def load_prompt(instruction_file: Path, task_name: str) -> str:
    if not instruction_file.exists():
        return humanize_task_name(task_name)

    with open(instruction_file) as f:
        data = json.load(f)

    for key in ("unseen", "seen"):
        prompts = data.get(key, [])
        if prompts:
            return prompts[0]
    return humanize_task_name(task_name)


def decode_rgb_frame(encoded: bytes) -> np.ndarray:
    frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode RGB frame from RobotWin HDF5 payload.")
    return frame


def ensure_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output root already exists and is not empty: {output_root}. Use --overwrite to replace it."
        )

    if output_root.exists() and overwrite:
        for child in output_root.iterdir():
            if child.is_dir():
                for nested in sorted(child.rglob("*"), reverse=True):
                    if nested.is_file() or nested.is_symlink():
                        nested.unlink()
                    elif nested.is_dir():
                        nested.rmdir()
                child.rmdir()
            else:
                child.unlink()

    output_root.mkdir(parents=True, exist_ok=True)


def write_task_info(output_root: Path, fps: float) -> None:
    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    task_info = {
        "task_desc": {
            "task_name": "robotwin_dual_arm_multitask",
            "prompt": "Perform the instructed bimanual manipulation task.",
        },
        "robot_type": "aloha",
        "state_encoding": "dual_arm_ee",
        "fps": fps,
        "camera_videos": CAMERA_VIDEOS,
    }
    with open(meta_dir / "task_info.json", "w") as f:
        json.dump(task_info, f, indent=2)


def iter_episode_files(task_variant_dir: Path, max_episodes_per_task: int | None) -> Iterable[Path]:
    data_dir = task_variant_dir / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Expected RobotWin data directory at {data_dir}")

    episodes = sorted(data_dir.glob("episode*.hdf5"), key=episode_sort_key)
    if max_episodes_per_task is not None:
        episodes = episodes[:max_episodes_per_task]
    return episodes


def write_episode(
    hdf5_path: Path,
    task_name: str,
    prompt: str,
    output_episode_dir: Path,
    fps: float,
) -> None:
    states_dir = output_episode_dir / "states"
    meta_dir = output_episode_dir / "meta"
    videos_dir = output_episode_dir / "videos"
    states_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    with open(meta_dir / "episode_meta.json", "w") as f:
        json.dump(
            {
                "prompt": prompt,
                "source_task": task_name,
                "source_episode": hdf5_path.stem,
            },
            f,
            indent=2,
        )

    writers: Dict[str, cv2.VideoWriter] = {}

    try:
        with h5py.File(hdf5_path, "r") as data_file, open(states_dir / "states.jsonl", "w") as states_file:
            num_frames = int(data_file["endpose/left_endpose"].shape[0])
            for frame_idx in range(num_frames):
                state_record = {
                    "left_ee_positions": data_file["endpose/left_endpose"][frame_idx].astype(np.float64).tolist(),
                    "left_gripper_width": float(data_file["endpose/left_gripper"][frame_idx]),
                    "right_ee_positions": data_file["endpose/right_endpose"][frame_idx].astype(np.float64).tolist(),
                    "right_gripper_width": float(data_file["endpose/right_gripper"][frame_idx]),
                    "timestamp": float(frame_idx / fps),
                }
                states_file.write(json.dumps(state_record) + "\n")

                for video_name, dataset_name in CAMERA_DATASETS.items():
                    frame = decode_rgb_frame(data_file[dataset_name][frame_idx])
                    writer = writers.get(video_name)
                    if writer is None:
                        height, width = frame.shape[:2]
                        writer = cv2.VideoWriter(
                            str(videos_dir / f"{video_name}.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            (width, height),
                        )
                        if not writer.isOpened():
                            raise ValueError(f"Failed to open video writer for {video_name} at {videos_dir}")
                        writers[video_name] = writer
                    writer.write(frame)
    finally:
        for writer in writers.values():
            writer.release()


def convert_robotwin_dataset(
    raw_root: Path,
    output_root: Path,
    dataset_name: str,
    tasks_csv: str | None,
    max_episodes_per_task: int | None,
    fps: float,
    overwrite: bool,
) -> None:
    raw_root = resolve_raw_root(raw_root)
    ensure_output_root(output_root, overwrite)
    write_task_info(output_root, fps)

    episode_counter = 0
    for task_name in discover_tasks(raw_root, tasks_csv):
        task_variant_dir = raw_root / task_name / dataset_name
        if not task_variant_dir.is_dir():
            raise FileNotFoundError(f"Task variant directory not found: {task_variant_dir}")

        instruction_dir = task_variant_dir / "instructions"
        for episode_file in iter_episode_files(task_variant_dir, max_episodes_per_task):
            prompt = load_prompt(instruction_dir / f"{episode_file.stem}.json", task_name)
            output_episode_dir = output_root / "data" / f"episode_{episode_counter:06d}"
            write_episode(episode_file, task_name, prompt, output_episode_dir, fps)
            episode_counter += 1

    if episode_counter == 0:
        raise ValueError("No RobotWin episodes were converted. Check --tasks and --dataset-name.")


def main() -> None:
    args = parse_args()
    convert_robotwin_dataset(
        raw_root=args.raw_root,
        output_root=args.output_root,
        dataset_name=args.dataset_name,
        tasks_csv=args.tasks,
        max_episodes_per_task=args.max_episodes_per_task,
        fps=args.fps,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()