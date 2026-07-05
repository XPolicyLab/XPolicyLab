#!/usr/bin/env python3

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


XPOLICYLAB_ROOT = Path(__file__).resolve().parents[4]
if str(XPOLICYLAB_ROOT) not in sys.path:
    sys.path.insert(0, str(XPOLICYLAB_ROOT))

from XPolicyLab.utils.data_loader import load


DATA_ROOT = Path(os.environ.get("XPOLICYLAB_DATA_ROOT", XPOLICYLAB_ROOT / "data"))
TARGET_IMAGE_WIDTH = 640
TARGET_IMAGE_HEIGHT = 480

CAMERA_CANDIDATES = {
    "head_camera_rgb": [
        ("vision", "cam_head", "colors"),
    ],
    "left_camera_rgb": [
        ("vision", "cam_left_wrist", "colors"),
    ],
    "right_camera_rgb": [
        ("vision", "cam_right_wrist", "colors"),
    ],
}

CAMERA_VIDEOS = {
    "observation.images.cam_high": "head_camera_rgb",
    "observation.images.cam_left_wrist": "left_camera_rgb",
    "observation.images.cam_right_wrist": "right_camera_rgb",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert XPolicyLab trajectory HDF5 files into Spirit-v1.5 training format."
    )
    parser.add_argument(
        "patterns",
        nargs="+",
        help='Match expressions like "RoboDojo.*.arx_x5" or "RoboDojo.*.*".',
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output directory for the converted Spirit-format dataset.",
    )
    parser.add_argument(
        "--data-type",
        type=str,
        default="xspark",
        help="Dataset type passed into XPolicyLab.utils.data_loader.load.",
    )
    parser.add_argument(
        "--data-version",
        type=str,
        default="v1.0",
        help="Dataset version passed into XPolicyLab.utils.data_loader.load.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override fps written into task_info.json and timestamps. Defaults to additional_info/frequency or 30.",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="aloha",
        help="Robot type written into Spirit meta/task_info.json.",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default="xpolicylab_multitask",
        help="Task name written into meta/task_info.json.",
    )
    parser.add_argument(
        "--task-prompt",
        type=str,
        default="Perform the instructed bimanual manipulation task.",
        help="Default task prompt written into meta/task_info.json when per-episode prompt is unavailable.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing output files in the target root before conversion.",
    )
    parser.add_argument(
        "--max-episodes-per-target",
        type=int,
        default=None,
        help="Optional cap on the number of HDF5 episodes converted for each matched <dataset>/<task>/<env_cfg> target.",
    )
    return parser.parse_args()


def _split_pattern(pattern: str) -> Tuple[str, str, str]:
    parts = pattern.split(".")
    if len(parts) != 3:
        raise ValueError(f"Pattern should be <dataset>.<task>.<env_cfg>, got: {pattern}")
    return parts[0], parts[1], parts[2]


def _match_name(name: str, pattern: str) -> bool:
    if pattern == "*":
        return True
    import fnmatch

    return fnmatch.fnmatchcase(name, pattern)


def _discover_conversion_targets(patterns: List[str]) -> List[Tuple[str, str, str]]:
    targets: List[Tuple[str, str, str]] = []
    seen = set()
    if not DATA_ROOT.exists():
        return targets

    for pattern in patterns:
        dataset_pattern, task_pattern, env_pattern = _split_pattern(pattern)
        for dataset_dir in sorted(path for path in DATA_ROOT.iterdir() if path.is_dir()):
            if not _match_name(dataset_dir.name, dataset_pattern):
                continue
            for task_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
                if not _match_name(task_dir.name, task_pattern):
                    continue
                for env_dir in sorted(path for path in task_dir.iterdir() if path.is_dir()):
                    if not _match_name(env_dir.name, env_pattern):
                        continue
                    input_dir = env_dir / "data"
                    if not input_dir.is_dir():
                        continue
                    target = (dataset_dir.name, task_dir.name, env_dir.name)
                    if target not in seen:
                        seen.add(target)
                        targets.append(target)
    return targets


def _resolve_input_dir(bench_name: str, task_name: str, env_cfg_type: str) -> Path:
    return DATA_ROOT / bench_name / task_name / env_cfg_type / "data"


def _find_input_files(input_dir: Path) -> List[Path]:
    files = sorted(input_dir.rglob("*.hdf5"))
    files.extend(sorted(input_dir.rglob("*.h5")))
    unique_files: List[Path] = []
    seen = set()
    for file_path in files:
        resolved = str(file_path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique_files.append(file_path)
    return unique_files


def _collect_target_input_files(
    targets: List[Tuple[str, str, str]],
    max_episodes_per_target: Optional[int] = None,
) -> List[Tuple[str, str, str, Path, List[Path]]]:
    collected = []
    for bench_name, task_name, env_cfg_type in targets:
        input_dir = _resolve_input_dir(bench_name, task_name, env_cfg_type)
        input_files = _find_input_files(input_dir)
        if max_episodes_per_target is not None and max_episodes_per_target > 0:
            input_files = input_files[:max_episodes_per_target]
        collected.append((bench_name, task_name, env_cfg_type, input_dir, input_files))
    return collected


def _print_matched_targets(target_inputs: List[Tuple[str, str, str, Path, List[Path]]]) -> None:
    print("Matched targets:")
    for bench_name, task_name, env_cfg_type, input_dir, input_files in target_inputs:
        print(f"  - {bench_name}/{task_name}/{env_cfg_type}: {len(input_files)} files from {input_dir}")


def _get_nested(data, *keys, default=None):
    cur = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _ensure_utf8_strings(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, bytes):
        return [value.decode("utf-8")]
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _ensure_utf8_strings(value.item())
        return [str(x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else x) for x in value.tolist()]
    if isinstance(value, (list, tuple)):
        result: List[str] = []
        for item in value:
            result.extend(_ensure_utf8_strings(item))
        return result
    return [str(value)]


def _choose_instruction(data: dict, fallback: str) -> str:
    for key in ("instructions", "instruction"):
        candidates = _ensure_utf8_strings(_get_nested(data, key))
        candidates = [item for item in candidates if item]
        if candidates:
            return random.choice(candidates)
    return fallback


def _ensure_2d_float64(array, name: str) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{name} should be 2D, got shape {arr.shape}")
    return arr


def _frame_indices(num_frames: int, source_fps: float, target_fps: float) -> np.ndarray:
    """Uniformly resample frame indices when target_fps differs from source_fps."""
    if num_frames <= 1 or abs(source_fps - target_fps) < 1e-3:
        return np.arange(num_frames, dtype=int)
    duration = (num_frames - 1) / source_fps
    target_count = max(2, int(round(duration * target_fps)) + 1)
    target_count = min(target_count, num_frames)
    return np.linspace(0, num_frames - 1, target_count).astype(int)


def _extract_frequency(data: dict, fallback: float) -> float:
    value = _get_nested(data, "additional_info", "frequency")
    if value is None:
        return fallback
    arr = np.asarray(value)
    if arr.ndim == 0:
        return float(arr.item())
    if arr.size > 0:
        return float(arr.reshape(-1)[0])
    return fallback


def _reorder_wxyz_to_xyzw(poses: np.ndarray, name: str) -> np.ndarray:
    if poses.shape[1] != 7:
        raise ValueError(f"{name} last dim should be 7, got {poses.shape}")
    xyz = poses[:, :3]
    quat_wxyz = poses[:, 3:]
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    return np.concatenate([xyz, quat_xyzw], axis=1)


def _extract_gripper_width(array, name: str) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    raise ValueError(f"{name} should be shape (T,) or (T,1), got {arr.shape}")


def _extract_dual_arm_state(data: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left_pose = _get_nested(data, "state", "left_ee_poses")
    right_pose = _get_nested(data, "state", "right_ee_poses")
    left_gripper = _get_nested(data, "state", "left_ee_joint_states")
    right_gripper = _get_nested(data, "state", "right_ee_joint_states")

    if left_pose is None or right_pose is None:
        raise ValueError(
            "Spirit conversion requires state/left_ee_poses and state/right_ee_poses in XPolicyLab trajectory files."
        )
    if left_gripper is None or right_gripper is None:
        raise ValueError(
            "Spirit conversion requires state/left_ee_joint_states and state/right_ee_joint_states in XPolicyLab trajectory files."
        )

    left_pose_arr = _reorder_wxyz_to_xyzw(_ensure_2d_float64(left_pose, "state.left_ee_poses"), "state.left_ee_poses")
    right_pose_arr = _reorder_wxyz_to_xyzw(_ensure_2d_float64(right_pose, "state.right_ee_poses"), "state.right_ee_poses")
    left_gripper_arr = _extract_gripper_width(left_gripper, "state.left_ee_joint_states")
    right_gripper_arr = _extract_gripper_width(right_gripper, "state.right_ee_joint_states")

    num_frames = left_pose_arr.shape[0]
    if right_pose_arr.shape[0] != num_frames:
        raise ValueError("left/right ee pose horizon mismatch")
    if left_gripper_arr.shape[0] != num_frames or right_gripper_arr.shape[0] != num_frames:
        raise ValueError("ee pose and gripper horizon mismatch")

    return left_pose_arr, left_gripper_arr, right_pose_arr, right_gripper_arr


def _decode_one_image(frame) -> np.ndarray:
    if isinstance(frame, np.ndarray) and frame.ndim == 3:
        return frame.astype(np.uint8) if frame.dtype != np.uint8 else frame
    if isinstance(frame, np.ndarray) and frame.dtype == np.uint8 and frame.ndim == 1:
        img = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image from uint8 buffer")
        return img
    if isinstance(frame, (bytes, bytearray, np.bytes_)):
        img = cv2.imdecode(np.frombuffer(frame.rstrip(b"\0"), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image from byte buffer")
        return img
    if isinstance(frame, np.ndarray) and frame.dtype.kind in {"S", "U"}:
        raw = frame.item() if frame.ndim == 0 else frame.tobytes()
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        img = cv2.imdecode(np.frombuffer(raw.rstrip(b"\0"), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image from string buffer")
        return img
    raise ValueError(f"Unsupported image frame type: {type(frame)}")


def _resize_image(image: np.ndarray) -> np.ndarray:
    if image.shape[:2] == (TARGET_IMAGE_HEIGHT, TARGET_IMAGE_WIDTH):
        return image
    return cv2.resize(image, (TARGET_IMAGE_WIDTH, TARGET_IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)


def _decode_images_if_needed(images) -> np.ndarray:
    if isinstance(images, (bytes, bytearray, np.bytes_)):
        return np.stack([_resize_image(_decode_one_image(images))], axis=0).astype(np.uint8)

    arr = np.asarray(images)
    if arr.ndim == 4 and arr.dtype != object:
        arr = arr.astype(np.uint8) if arr.dtype != np.uint8 else arr
        if arr.shape[1:3] != (TARGET_IMAGE_HEIGHT, TARGET_IMAGE_WIDTH):
            arr = np.stack([_resize_image(frame) for frame in arr], axis=0)
        return arr

    if arr.ndim == 3 and arr.dtype != object:
        arr = arr.astype(np.uint8) if arr.dtype != np.uint8 else arr
        return _resize_image(arr)[None, ...]

    if arr.ndim == 0:
        frames = [arr.item()]
    elif isinstance(images, np.ndarray) and images.dtype == object:
        frames = images.tolist()
    else:
        frames = list(images)

    decoded = [_resize_image(_decode_one_image(frame)) for frame in frames]
    return np.stack(decoded, axis=0).astype(np.uint8)


def _find_camera_array(data: dict, camera_name: str) -> Optional[np.ndarray]:
    for keys in CAMERA_CANDIDATES[camera_name]:
        value = _get_nested(data, *keys)
        if value is not None:
            return _decode_images_if_needed(value)
    return None


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


def write_task_info(output_root: Path, task_name: str, task_prompt: str, robot_type: str, fps: float) -> None:
    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    task_info = {
        "task_desc": {
            "task_name": task_name,
            "prompt": task_prompt,
        },
        "robot_type": robot_type,
        "state_encoding": "dual_arm_ee",
        "fps": fps,
        "camera_videos": CAMERA_VIDEOS,
    }
    with open(meta_dir / "task_info.json", "w") as f:
        json.dump(task_info, f, indent=2)


def _create_black_frames(num_frames: int) -> np.ndarray:
    return np.zeros((num_frames, TARGET_IMAGE_HEIGHT, TARGET_IMAGE_WIDTH, 3), dtype=np.uint8)


def _write_video(video_path: Path, frames: np.ndarray, fps: float) -> None:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected frames shape [T,H,W,3], got {frames.shape}")
    height, width = frames.shape[1], frames.shape[2]
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise ValueError(f"Failed to open video writer for {video_path}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def write_episode(
    data: dict,
    source_meta: Dict[str, str],
    output_episode_dir: Path,
    default_prompt: str,
    source_fps: float,
    target_fps: float,
) -> None:
    states_dir = output_episode_dir / "states"
    meta_dir = output_episode_dir / "meta"
    videos_dir = output_episode_dir / "videos"
    states_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    prompt = _choose_instruction(data, default_prompt)
    with open(meta_dir / "episode_meta.json", "w") as f:
        json.dump(
            {
                "prompt": prompt,
                **source_meta,
            },
            f,
            indent=2,
        )

    left_pose, left_gripper, right_pose, right_gripper = _extract_dual_arm_state(data)
    source_num_frames = left_pose.shape[0]
    frame_indices = _frame_indices(source_num_frames, source_fps, target_fps)
    left_pose = left_pose[frame_indices]
    left_gripper = left_gripper[frame_indices]
    right_pose = right_pose[frame_indices]
    right_gripper = right_gripper[frame_indices]
    num_frames = left_pose.shape[0]

    with open(states_dir / "states.jsonl", "w") as states_file:
        for frame_idx in range(num_frames):
            state_record = {
                "left_ee_positions": left_pose[frame_idx].astype(np.float64).tolist(),
                "left_gripper_width": float(left_gripper[frame_idx]),
                "right_ee_positions": right_pose[frame_idx].astype(np.float64).tolist(),
                "right_gripper_width": float(right_gripper[frame_idx]),
                "timestamp": float(frame_idx / target_fps),
            }
            states_file.write(json.dumps(state_record) + "\n")

    images_by_name: Dict[str, np.ndarray] = {}
    for video_name in CAMERA_CANDIDATES:
        image_array = _find_camera_array(data, video_name)
        if image_array is not None:
            if image_array.shape[0] != source_num_frames:
                raise ValueError(
                    f"Video horizon mismatch for {video_name}: expected {source_num_frames}, got {image_array.shape[0]}"
                )
            images_by_name[video_name] = image_array[frame_indices]
        else:
            images_by_name[video_name] = _create_black_frames(num_frames)

    for video_name, image_array in images_by_name.items():
        _write_video(videos_dir / f"{video_name}.mp4", image_array, target_fps)


def convert_xpolicylab_dataset(
    patterns: List[str],
    output_root: Path,
    data_type: str,
    data_version: str,
    fps_override: Optional[float],
    task_name: str,
    task_prompt: str,
    robot_type: str,
    overwrite: bool,
    max_episodes_per_target: Optional[int],
) -> None:
    targets = _discover_conversion_targets(patterns)
    if not targets:
        raise FileNotFoundError(f"No matching dataset/task/env_cfg targets found for patterns: {patterns}")

    target_inputs = _collect_target_input_files(targets, max_episodes_per_target=max_episodes_per_target)
    _print_matched_targets(target_inputs)

    ensure_output_root(output_root, overwrite)

    total_files = 0
    total_success = 0
    failures: List[Tuple[str, str]] = []
    episode_counter = 0
    resolved_fps = fps_override or 30.0

    for bench_name, task_name_item, env_cfg_type, input_dir, input_files in target_inputs:
        if not input_files:
            failures.append((str(input_dir), "No .hdf5 or .h5 files found"))
            continue

        for input_path in input_files:
            total_files += 1
            try:
                data = load(str(input_path), data_type=data_type, data_version=data_version)
                source_fps = _extract_frequency(data, resolved_fps)
                target_fps = fps_override if fps_override is not None else source_fps
                if fps_override is None:
                    resolved_fps = source_fps
                output_episode_dir = output_root / "data" / f"episode_{episode_counter:06d}"
                write_episode(
                    data=data,
                    source_meta={
                        "source_dataset": bench_name,
                        "source_task": task_name_item,
                        "source_env_cfg": env_cfg_type,
                        "source_episode": input_path.stem,
                    },
                    output_episode_dir=output_episode_dir,
                    default_prompt=task_prompt,
                    source_fps=source_fps,
                    target_fps=target_fps,
                )
                episode_counter += 1
                total_success += 1
            except Exception as exc:
                failures.append((str(input_path), str(exc)))

    if total_success == 0:
        raise ValueError("No episodes were converted. Check the matched patterns and input file contents.")

    write_task_info(output_root, task_name, task_prompt, robot_type, fps_override or resolved_fps)

    print(f"Converted {total_success}/{total_files} files to {output_root}")
    if failures:
        print("Failed files:")
        for file_path, reason in failures:
            print(f"  - {file_path}: {reason}")


def main() -> None:
    args = parse_args()
    convert_xpolicylab_dataset(
        patterns=args.patterns,
        output_root=args.output_root,
        data_type=args.data_type,
        data_version=args.data_version,
        fps_override=args.fps,
        task_name=args.task_name,
        task_prompt=args.task_prompt,
        robot_type=args.robot_type,
        overwrite=args.overwrite,
        max_episodes_per_target=args.max_episodes_per_target,
    )


if __name__ == "__main__":
    main()