"""Convert RoboDojo sim_cloud HDF5 episodes to Dexbotic Dexdata jsonl format.

Reads xspark v1.0 HDF5 via XPolicyLab.utils.data_loader and writes per-episode
jsonl files plus three-view mp4 videos under a Dexdata-compatible directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

XPOLICYLAB_ROOT = Path(__file__).resolve().parents[3]
if str(XPOLICYLAB_ROOT) not in sys.path:
    sys.path.insert(0, str(XPOLICYLAB_ROOT))

from XPolicyLab.utils.data_loader import load

STATE_DIM = 32
GRIPPER_NON_DELTA_INDICES = (6, 20)

CAMERA_CANDIDATES = {
    "head": [
        ("vision", "cam_head", "colors"),
        ("vision", "cam_head", "color"),
        ("cam_head", "color"),
        ("cam_head", "colors"),
        ("cam_high", "color"),
        ("cam_high", "colors"),
    ],
    "left_wrist": [
        ("vision", "cam_left_wrist", "colors"),
        ("vision", "cam_left_wrist", "color"),
        ("cam_left_wrist", "color"),
        ("cam_left_wrist", "colors"),
    ],
    "right_wrist": [
        ("vision", "cam_right_wrist", "colors"),
        ("vision", "cam_right_wrist", "color"),
        ("cam_right_wrist", "color"),
        ("cam_right_wrist", "colors"),
    ],
}

DEFAULT_PROMPT = "Perform the instructed bimanual manipulation task."


@dataclass(frozen=True)
class EpisodeJob:
    episode_index: int
    input_path: str
    output_jsonl_path: str
    output_video_paths: dict[str, str]
    data_type: str
    data_version: str


def _get_nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _ensure_2d_float32(array: Any, name: str) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{name} should be 2D, got shape {arr.shape}")
    return arr


def _ensure_utf8_strings(value: Any) -> list[str]:
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
        result: list[str] = []
        for item in value:
            result.extend(_ensure_utf8_strings(item))
        return result
    return [str(value)]


def _find_instruction(data: dict[str, Any]) -> str:
    candidates = [
        _get_nested(data, "instruction"),
        _get_nested(data, "instructions"),
        _get_nested(data, "extra_episode_info", "instruction"),
        _get_nested(data, "extra_episode_info", "instructions"),
        _get_nested(data, "meta", "instructions"),
        _get_nested(data, "metadata", "instructions"),
    ]
    for candidate in candidates:
        strings = [s.strip() for s in _ensure_utf8_strings(candidate) if str(s).strip()]
        if strings:
            return strings[0]
    return DEFAULT_PROMPT


def _find_camera_array(data: dict[str, Any], camera_name: str) -> np.ndarray | None:
    for keys in CAMERA_CANDIDATES[camera_name]:
        value = _get_nested(data, *keys)
        if value is not None:
            arr = np.asarray(value)
            if arr.ndim != 4:
                raise ValueError(f"{camera_name} expected 4D images, got {arr.shape}")
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)
            return arr
    return None


def quat_wxyz_to_rotm(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = quat / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def rotm2aa(rotm: np.ndarray) -> np.ndarray:
    rotm = np.asarray(rotm, dtype=np.float32).reshape(3, 3)
    theta = np.arccos(np.clip((np.trace(rotm) - 1.0) / 2.0, -1.0, 1.0))
    if theta <= 1e-6:
        return np.zeros(3, dtype=np.float32)
    axis = np.array(
        [
            rotm[2, 1] - rotm[1, 2],
            rotm[0, 2] - rotm[2, 0],
            rotm[1, 0] - rotm[0, 1],
        ],
        dtype=np.float32,
    )
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        return np.zeros(3, dtype=np.float32)
    return axis / axis_norm * theta


def _extract_proprio_fields(data: dict[str, Any], side: str) -> dict[str, np.ndarray]:
    ee_poses = _ensure_2d_float32(_get_nested(data, "state", f"{side}_ee_poses"), f"state.{side}_ee_poses")
    if ee_poses.shape[1] != 7:
        raise ValueError(f"state.{side}_ee_poses expected 7 dims (xyz + wxyz), got {ee_poses.shape[1]}")

    arm_joint = _ensure_2d_float32(
        _get_nested(data, "state", f"{side}_arm_joint_states"),
        f"state.{side}_arm_joint_states",
    )
    gripper = _ensure_2d_float32(
        _get_nested(data, "state", f"{side}_ee_joint_states"),
        f"state.{side}_ee_joint_states",
    )

    ee_pos = ee_poses[:, :3]
    ee_rotm = np.stack([quat_wxyz_to_rotm(q).reshape(-1) for q in ee_poses[:, 3:7]], axis=0)
    return {
        "ee_pos": ee_pos.astype(np.float32),
        "ee_rotm": ee_rotm.astype(np.float32),
        "arm_joint": arm_joint.astype(np.float32),
        "gripper": gripper.astype(np.float32),
    }


def build_state_vector(proprio: dict[str, dict[str, np.ndarray]], frame_index: int) -> list[float]:
    """Pack bimanual proprio into a fixed 32D state vector for DM0."""
    state = np.zeros(STATE_DIM, dtype=np.float32)

    for side, base in (("left", 0), ("right", 14)):
        joint = proprio[side]["arm_joint"][frame_index].reshape(-1)[:6]
        gripper = proprio[side]["gripper"][frame_index].reshape(-1)[:1]
        ee_pos = proprio[side]["ee_pos"][frame_index].reshape(-1)[:3]
        ee_aa = rotm2aa(proprio[side]["ee_rotm"][frame_index].reshape(3, 3))

        state[base : base + 6] = joint
        state[base + 6] = gripper[0]
        state[base + 7 : base + 10] = ee_pos
        state[base + 10 : base + 13] = ee_aa

    return state.tolist()


def _write_video(frames: np.ndarray, output_path: Path, fps: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1:3]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")
    try:
        for frame in frames:
            bgr = frame
            if frame.shape[-1] == 3:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
    finally:
        writer.release()


def convert_one(job: EpisodeJob) -> tuple[int, str | None]:
    try:
        data = load(job.input_path, data_type=job.data_type, data_version=job.data_version)

        left_joint = _get_nested(data, "state", "left_arm_joint_states")
        if left_joint is None:
            raise ValueError("Cannot find state.left_arm_joint_states")

        num_frames = np.asarray(left_joint).shape[0]
        if num_frames < 2:
            raise ValueError(f"Episode too short: {num_frames} frames")

        proprio = {
            "left": _extract_proprio_fields(data, "left"),
            "right": _extract_proprio_fields(data, "right"),
        }

        images: dict[str, np.ndarray] = {}
        for camera_name in CAMERA_CANDIDATES:
            camera_frames = _find_camera_array(data, camera_name)
            if camera_frames is None:
                raise ValueError(f"Missing camera: {camera_name}")
            if camera_frames.shape[0] != num_frames:
                raise ValueError(
                    f"{camera_name} frame count mismatch: {camera_frames.shape[0]} vs {num_frames}"
                )
            images[camera_name] = camera_frames

        fps_value = _get_nested(data, "additional_info", "frequency", default=30)
        fps = float(np.asarray(fps_value).item() if isinstance(fps_value, np.ndarray) else fps_value)
        instruction = _find_instruction(data)

        video_paths = {name: Path(path) for name, path in job.output_video_paths.items()}
        for camera_name, frames in images.items():
            _write_video(frames, video_paths[camera_name], fps)

        head_name = video_paths["head"].name
        left_name = video_paths["left_wrist"].name
        right_name = video_paths["right_wrist"].name

        jsonl_path = Path(job.output_jsonl_path)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("w", encoding="utf-8") as file:
            for frame_index in range(num_frames):
                frame = {
                    "images_1": {
                        "type": "video",
                        "url": head_name,
                        "frame_idx": frame_index,
                    },
                    "images_2": {
                        "type": "video",
                        "url": left_name,
                        "frame_idx": frame_index,
                    },
                    "images_3": {
                        "type": "video",
                        "url": right_name,
                        "frame_idx": frame_index,
                    },
                    "state": build_state_vector(proprio, frame_index),
                    "prompt": instruction,
                    "is_robot": True,
                }
                file.write(json.dumps(frame, ensure_ascii=False) + "\n")

        return job.episode_index, None
    except Exception as exc:
        return job.episode_index, f"{job.input_path}: {exc}"


def find_input_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.rglob("*.hdf5"))
    files.extend(sorted(input_dir.rglob("*.h5")))
    unique_files: list[Path] = []
    seen: set[str] = set()
    for file_path in files:
        resolved = str(file_path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique_files.append(file_path)
    return unique_files


def build_jobs(
    input_files: list[Path],
    output_dir: Path,
    data_type: str,
    data_version: str,
    start_index: int = 0,
) -> list[EpisodeJob]:
    jobs: list[EpisodeJob] = []
    video_dir = output_dir / "video"
    for offset, input_path in enumerate(input_files):
        episode_index = start_index + offset
        stem = f"episode_{episode_index:06d}"
        jobs.append(
            EpisodeJob(
                episode_index=episode_index,
                input_path=str(input_path),
                output_jsonl_path=str(output_dir / f"{stem}.jsonl"),
                output_video_paths={
                    "head": str(video_dir / f"{stem}_head.mp4"),
                    "left_wrist": str(video_dir / f"{stem}_left_wrist.mp4"),
                    "right_wrist": str(video_dir / f"{stem}_right_wrist.mp4"),
                },
                data_type=data_type,
                data_version=data_version,
            )
        )
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert RoboDojo sim_cloud HDF5 to Dexdata jsonl.")
    parser.add_argument("input_dir", type=str, help="Input dataset directory")
    parser.add_argument("output_dir", type=str, help="Output Dexdata directory")
    parser.add_argument("--data_type", type=str, default="xspark", help="Dataset type")
    parser.add_argument("--data_version", type=str, default="v1.0", help="Dataset version")
    parser.add_argument("--num_workers", type=int, default=8, help="Parallel worker processes")
    parser.add_argument("--start_index", type=int, default=0, help="Starting episode index")
    parser.add_argument("--max_episodes", type=int, default=-1, help="Limit episodes, -1 for all")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = find_input_files(input_dir)
    if not input_files:
        raise FileNotFoundError(f"No .hdf5 or .h5 files found under {input_dir}")
    if args.max_episodes > 0:
        input_files = input_files[: args.max_episodes]

    jobs = build_jobs(
        input_files=input_files,
        output_dir=output_dir,
        data_type=args.data_type,
        data_version=args.data_version,
        start_index=args.start_index,
    )

    failures: list[tuple[int, str]] = []
    if args.num_workers <= 1:
        for job in tqdm(jobs, desc="Converting"):
            episode_index, error = convert_one(job)
            if error:
                failures.append((episode_index, error))
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(convert_one, job): job for job in jobs}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Converting"):
                episode_index, error = future.result()
                if error:
                    failures.append((episode_index, error))

    success_count = len(jobs) - len(failures)
    print(f"Converted {success_count}/{len(jobs)} episodes to {output_dir}")

    if failures:
        print("Failed episodes:")
        for episode_index, reason in sorted(failures):
            print(f"  - episode_{episode_index:06d}: {reason}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
