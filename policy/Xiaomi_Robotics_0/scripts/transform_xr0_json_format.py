"""Convert RoboDojo sim_cloud HDF5 episodes to Xiaomi-Robotics-0 XR-0 JSON format.

Reads joint + EE (xyz + wxyz quaternion) from raw xspark v1.0 HDF5 files and
writes per-episode JSON annotations plus three-view mp4 videos.

Data matching follows ``transform_aloha_hdf5_format.py`` / ``transform_lerobot_v30_format.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

XPOLICYLAB_ROOT = Path(__file__).resolve().parents[3]
if str(XPOLICYLAB_ROOT) not in sys.path:
    sys.path.insert(0, str(XPOLICYLAB_ROOT))

from XPolicyLab.utils.data_loader import load

ACTION_LENGTH = 30
ACTION_DIM = 32

CAMERA_CANDIDATES = {
    "ego": [
        ("vision", "cam_head", "colors"),
        ("vision", "cam_head", "color"),
        ("cam_head", "color"),
        ("cam_head", "colors"),
        ("cam_high", "color"),
        ("cam_high", "colors"),
    ],
    "wrist_left": [
        ("vision", "cam_left_wrist", "colors"),
        ("vision", "cam_left_wrist", "color"),
        ("cam_left_wrist", "color"),
        ("cam_left_wrist", "colors"),
    ],
    "wrist_right": [
        ("vision", "cam_right_wrist", "colors"),
        ("vision", "cam_right_wrist", "color"),
        ("cam_right_wrist", "color"),
        ("cam_right_wrist", "colors"),
    ],
}

XR0_PROMPT_TEMPLATE = (
    "The following observations are captured from multiple views.\n"
    "# Ego View\n<image>\n"
    "# Left-Wrist View\n<image>\n"
    "# Right-Wrist View\n<image>\n"
    "Generate robot actions for the task:\n{task}"
)


@dataclass(frozen=True)
class EpisodeJob:
    episode_index: int
    input_path: str
    output_json_path: str
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
    return "Perform the robot manipulation task."


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


def _concat_state_parts(parts: list[tuple[str, Any]], name: str) -> np.ndarray | None:
    valid_parts: list[np.ndarray] = []
    horizon: int | None = None
    for part_name, value in parts:
        if value is None:
            continue
        arr = _ensure_2d_float32(value, f"{name}.{part_name}")
        if horizon is None:
            horizon = arr.shape[0]
        elif arr.shape[0] != horizon:
            raise ValueError(f"{name}.{part_name} horizon mismatch: expected {horizon}, got {arr.shape[0]}")
        valid_parts.append(arr)
    if not valid_parts:
        return None
    return np.concatenate(valid_parts, axis=1)


def quat_wxyz_to_rotm(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert quaternion in wxyz order to a 3x3 rotation matrix."""
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


def rotm2aa_batch(rotms: np.ndarray) -> np.ndarray:
    """Rotation-matrix sequence to axis-angle, copied from XR-0 training utilities."""
    rotms = np.asarray(rotms, dtype=np.float32)
    theta = np.arccos(np.clip((np.einsum("nii->n", rotms) - 1.0) / 2.0, -1.0, 1.0))
    axis_angle = np.zeros((rotms.shape[0], 3), dtype=np.float32)
    near_zero = theta <= 1e-6
    near_pi = np.abs(theta - np.pi) <= 1e-6
    normal = ~(near_zero | near_pi)

    if np.any(normal):
        axis = np.stack(
            [
                rotms[:, 2, 1] - rotms[:, 1, 2],
                rotms[:, 0, 2] - rotms[:, 2, 0],
                rotms[:, 1, 0] - rotms[:, 0, 1],
            ],
            axis=1,
        )
        axis /= np.linalg.norm(axis, axis=1, keepdims=True) + 1e-12
        axis_angle[normal] = axis[normal] * theta[normal, None]

    if np.any(near_pi):
        for index in np.where(near_pi)[0]:
            rotm = rotms[index]
            rot00, rot11, rot22 = rotm[0, 0], rotm[1, 1], rotm[2, 2]
            if rot00 >= rot11 and rot00 >= rot22:
                axis = np.array([np.sqrt(max((rot00 + 1.0) / 2.0, 0.0)), 0.0, 0.0], dtype=np.float32)
                if axis[0] > 1e-8:
                    axis[1] = rotm[0, 1] / (2.0 * axis[0])
                    axis[2] = rotm[0, 2] / (2.0 * axis[0])
            elif rot11 >= rot22:
                axis = np.array([0.0, np.sqrt(max((rot11 + 1.0) / 2.0, 0.0)), 0.0], dtype=np.float32)
                if axis[1] > 1e-8:
                    axis[0] = rotm[0, 1] / (2.0 * axis[1])
                    axis[2] = rotm[1, 2] / (2.0 * axis[1])
            else:
                axis = np.array([0.0, 0.0, np.sqrt(max((rot22 + 1.0) / 2.0, 0.0))], dtype=np.float32)
                if axis[2] > 1e-8:
                    axis[0] = rotm[0, 2] / (2.0 * axis[2])
                    axis[1] = rotm[1, 2] / (2.0 * axis[2])
            norm = np.linalg.norm(axis)
            if norm < 1e-12:
                axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            else:
                axis = axis / norm
            axis_angle[index] = axis * theta[index]

    return axis_angle


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


def _build_action_targets(data: dict[str, Any], side: str, num_frames: int) -> dict[str, np.ndarray]:
    proprio = _extract_proprio_fields(data, side)

    action_joint = _ensure_2d_float32(
        _get_nested(data, "action", f"{side}_arm_joint_states"),
        f"action.{side}_arm_joint_states",
    )
    action_gripper = _ensure_2d_float32(
        _get_nested(data, "action", f"{side}_ee_joint_states"),
        f"action.{side}_ee_joint_states",
    )

    ee_pos = proprio["ee_pos"].copy()
    ee_rotm = proprio["ee_rotm"].copy()
    if num_frames > 1:
        ee_pos[:-1] = proprio["ee_pos"][1:]
        ee_rotm[:-1] = proprio["ee_rotm"][1:]

    return {
        "ee_pos": ee_pos,
        "ee_rotm": ee_rotm,
        "arm_joint": action_joint,
        "gripper": action_gripper,
    }


def _to_nested_list(array: np.ndarray) -> list[list[float]]:
    return np.asarray(array, dtype=np.float32).tolist()


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


def _build_episode_json(
    episode_index: int,
    instruction: str,
    num_frames: int,
    fps: float,
    proprio: dict[str, dict[str, np.ndarray]],
    action_targets: dict[str, dict[str, np.ndarray]],
    video_paths: dict[str, Path],
    output_root: Path,
) -> dict[str, Any]:
    def rel(path: Path) -> str:
        return str(path.relative_to(output_root))

    return {
        "trajectory_type": "success",
        "time": f"episode_{episode_index:06d}",
        "num_frames": int(num_frames),
        "instruction": {
            "general": [
                {
                    "images": [
                        "observations.ego",
                        "observations.wrist_left",
                        "observations.wrist_right",
                    ],
                    "conversations": [
                        {
                            "from": "human",
                            "value": XR0_PROMPT_TEMPLATE.format(task=instruction),
                        },
                        {"from": "gpt", "value": "<bot></bot>"},
                    ],
                }
            ]
        },
        "observations": {
            "ego": [{"path": rel(video_paths["ego"]), "start": 0, "end": num_frames, "fps": fps, "crop_bbox": None}],
            "wrist_left": [
                {"path": rel(video_paths["wrist_left"]), "start": 0, "end": num_frames, "fps": fps, "crop_bbox": None}
            ],
            "wrist_right": [
                {"path": rel(video_paths["wrist_right"]), "start": 0, "end": num_frames, "fps": fps, "crop_bbox": None}
            ],
        },
        "proprios": {
            "left_ee_pos": _to_nested_list(proprio["left"]["ee_pos"]),
            "left_ee_rotm": _to_nested_list(proprio["left"]["ee_rotm"]),
            "left_arm_joint": _to_nested_list(proprio["left"]["arm_joint"]),
            "left_gripper_pos": _to_nested_list(proprio["left"]["gripper"]),
            "right_ee_pos": _to_nested_list(proprio["right"]["ee_pos"]),
            "right_ee_rotm": _to_nested_list(proprio["right"]["ee_rotm"]),
            "right_arm_joint": _to_nested_list(proprio["right"]["arm_joint"]),
            "right_gripper_pos": _to_nested_list(proprio["right"]["gripper"]),
        },
        "actions": {
            "left_ee_pos": _to_nested_list(action_targets["left"]["ee_pos"]),
            "left_ee_rotm": _to_nested_list(action_targets["left"]["ee_rotm"]),
            "left_arm_joint": _to_nested_list(action_targets["left"]["arm_joint"]),
            "left_gripper_pos": _to_nested_list(action_targets["left"]["gripper"]),
            "right_ee_pos": _to_nested_list(action_targets["right"]["ee_pos"]),
            "right_ee_rotm": _to_nested_list(action_targets["right"]["ee_rotm"]),
            "right_arm_joint": _to_nested_list(action_targets["right"]["arm_joint"]),
            "right_gripper_pos": _to_nested_list(action_targets["right"]["gripper"]),
        },
    }


def _compose_relative_action(
    proprio: dict[str, dict[str, np.ndarray]],
    action_targets: dict[str, dict[str, np.ndarray]],
    frame_index: int,
    action_length: int,
) -> np.ndarray:
    action = np.zeros((action_length, ACTION_DIM), dtype=np.float32)
    num_frames = proprio["left"]["ee_pos"].shape[0]
    steps = min(action_length, num_frames - frame_index)

    for side, offset in (("left", 0), ("right", 14)):
        rotm = proprio[side]["ee_rotm"][frame_index].reshape(3, 3)
        pos = proprio[side]["ee_pos"][frame_index]
        target_pos = action_targets[side]["ee_pos"][frame_index : frame_index + steps]
        target_rotm = action_targets[side]["ee_rotm"][frame_index : frame_index + steps].reshape(-1, 3, 3)

        ee_pos_delta = (rotm.T @ (target_pos - pos[None, :]).T).T
        ee_aa_delta = rotm2aa_batch(np.stack([rotm.T @ tm for tm in target_rotm], axis=0))
        gripper_delta = action_targets[side]["gripper"][frame_index : frame_index + steps] - proprio[side]["gripper"][frame_index]
        joint_delta = action_targets[side]["arm_joint"][frame_index : frame_index + steps] - proprio[side]["arm_joint"][frame_index]

        if steps < action_length:
            pad = action_length - steps
            ee_pos_delta = np.concatenate([ee_pos_delta, np.repeat(ee_pos_delta[-1:], pad, axis=0)], axis=0)
            ee_aa_delta = np.concatenate([ee_aa_delta, np.repeat(ee_aa_delta[-1:], pad, axis=0)], axis=0)
            gripper_delta = np.concatenate([gripper_delta, np.repeat(gripper_delta[-1:], pad, axis=0)], axis=0)
            joint_delta = np.concatenate([joint_delta, np.repeat(joint_delta[-1:], pad, axis=0)], axis=0)

        action[:, offset + 0 : offset + 3] = ee_pos_delta
        action[:, offset + 3 : offset + 6] = ee_aa_delta
        action[:, offset + 6 : offset + 7] = gripper_delta
        action[:, offset + 7 : offset + 13] = joint_delta

    return action


def convert_one(job: EpisodeJob) -> tuple[int, np.ndarray | None, str | None]:
    try:
        data = load(job.input_path, data_type=job.data_type, data_version=job.data_version)

        left_state = _concat_state_parts(
            [
                ("left_arm_joint_states", _get_nested(data, "state", "left_arm_joint_states")),
                ("left_ee_joint_states", _get_nested(data, "state", "left_ee_joint_states")),
                ("right_arm_joint_states", _get_nested(data, "state", "right_arm_joint_states")),
                ("right_ee_joint_states", _get_nested(data, "state", "right_ee_joint_states")),
            ],
            "state",
        )
        if left_state is None:
            raise ValueError("Cannot find state joint data")

        num_frames = left_state.shape[0]
        if num_frames < ACTION_LENGTH:
            raise ValueError(f"Episode shorter than action_length={ACTION_LENGTH}: {num_frames} frames")

        proprio = {
            "left": _extract_proprio_fields(data, "left"),
            "right": _extract_proprio_fields(data, "right"),
        }
        action_targets = {
            "left": _build_action_targets(data, "left", num_frames),
            "right": _build_action_targets(data, "right", num_frames),
        }

        for side in ("left", "right"):
            for key, array in proprio[side].items():
                if array.shape[0] != num_frames:
                    raise ValueError(f"proprio.{side}.{key} horizon mismatch: {array.shape[0]} vs {num_frames}")
            for key, array in action_targets[side].items():
                if array.shape[0] != num_frames:
                    raise ValueError(f"action.{side}.{key} horizon mismatch: {array.shape[0]} vs {num_frames}")

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

        output_root = Path(job.output_json_path).parents[1]
        video_paths = {name: Path(path) for name, path in job.output_video_paths.items()}
        for camera_name, frames in images.items():
            _write_video(frames, video_paths[camera_name], fps)

        episode_json = _build_episode_json(
            episode_index=job.episode_index,
            instruction=_find_instruction(data),
            num_frames=num_frames,
            fps=fps,
            proprio=proprio,
            action_targets=action_targets,
            video_paths=video_paths,
            output_root=output_root,
        )

        json_path = Path(job.output_json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as file:
            json.dump(episode_json, file, ensure_ascii=False)

        sample_actions = []
        for frame_index in range(num_frames - ACTION_LENGTH + 1):
            sample_actions.append(
                _compose_relative_action(proprio, action_targets, frame_index, ACTION_LENGTH)
            )
        action_tensor = np.stack(sample_actions, axis=0) if sample_actions else None
        return job.episode_index, action_tensor, None
    except Exception as exc:
        return job.episode_index, None, f"{job.input_path}: {exc}"


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
    json_dir = output_dir / "json"
    video_dir = output_dir / "videos"
    for offset, input_path in enumerate(input_files):
        episode_index = start_index + offset
        stem = f"episode_{episode_index:06d}"
        jobs.append(
            EpisodeJob(
                episode_index=episode_index,
                input_path=str(input_path),
                output_json_path=str(json_dir / f"{stem}.json"),
                output_video_paths={
                    "ego": str(video_dir / f"{stem}_ego.mp4"),
                    "wrist_left": str(video_dir / f"{stem}_wrist_left.mp4"),
                    "wrist_right": str(video_dir / f"{stem}_wrist_right.mp4"),
                },
                data_type=data_type,
                data_version=data_version,
            )
        )
    return jobs


def compute_action_stats(action_tensor: np.ndarray) -> tuple[list[list[float]], list[list[float]]]:
    if action_tensor.size == 0:
        raise ValueError("No action samples collected for stats computation")

    mean = action_tensor.mean(axis=(0, 1))
    std = action_tensor.std(axis=(0, 1))
    mean_by_step = np.tile(mean[None, :], (ACTION_LENGTH, 1))
    std_by_step = np.tile(std[None, :], (ACTION_LENGTH, 1))
    return mean_by_step.tolist(), std_by_step.tolist()


def save_stats(output_dir: Path, mean: list[list[float]], std: list[list[float]]) -> Path:
    stats_path = output_dir / "action_stats.json"
    payload = {
        "action_length": ACTION_LENGTH,
        "action_dim": ACTION_DIM,
        "mean": mean,
        "std": std,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with stats_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    return stats_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert RoboDojo sim_cloud HDF5 to XR-0 JSON format.")
    parser.add_argument("input_dir", type=str, help="Input dataset directory, e.g. .../sim_cloud")
    parser.add_argument("output_dir", type=str, help="Output directory for json/ and videos/")
    parser.add_argument("--data_type", type=str, default="xspark", help="Dataset type")
    parser.add_argument("--data_version", type=str, default="v1.0", help="Dataset version")
    parser.add_argument("--num_workers", type=int, default=8, help="Parallel worker processes")
    parser.add_argument("--start_index", type=int, default=0, help="Starting episode index")
    parser.add_argument("--max_episodes", type=int, default=-1, help="Limit number of episodes, -1 for all")
    parser.add_argument("--compute_stats", action="store_true", help="Compute XR-0 mean/std after conversion")
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
    action_samples: list[np.ndarray] = []

    if args.num_workers <= 1:
        for job in tqdm(jobs, desc="Converting"):
            episode_index, action_tensor, error = convert_one(job)
            if error:
                failures.append((episode_index, error))
            elif args.compute_stats and action_tensor is not None:
                action_samples.append(action_tensor)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(convert_one, job): job for job in jobs}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Converting"):
                episode_index, action_tensor, error = future.result()
                if error:
                    failures.append((episode_index, error))
                elif args.compute_stats and action_tensor is not None:
                    action_samples.append(action_tensor)

    success_count = len(jobs) - len(failures)
    print(f"Converted {success_count}/{len(jobs)} episodes to {output_dir}")

    if args.compute_stats and action_samples:
        merged = np.concatenate(action_samples, axis=0)
        mean, std = compute_action_stats(merged)
        stats_path = save_stats(output_dir, mean, std)
        print(f"Saved action stats to {stats_path}")

    if failures:
        print("Failed episodes:")
        for episode_index, reason in sorted(failures):
            print(f"  - episode_{episode_index:06d}: {reason}")


if __name__ == "__main__":
    main()
