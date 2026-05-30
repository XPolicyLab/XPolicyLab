import argparse
import dataclasses
import fnmatch
import random
import sys
from pathlib import Path
from typing import Literal

import cv2
import h5py
import numpy as np
from tqdm import tqdm
import shutil
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from XPolicyLab.utils.data_loader import load
from XPolicyLab.utils.load_file import load_json, load_yaml


DEFAULT_DATASET_NAME = "RoboDojo"
DATA_ROOT = PROJECT_ROOT / "data"
ENV_CFG_ROOT = PROJECT_ROOT / "env_cfg"
ROBOT_INFO_PATH = ENV_CFG_ROOT / "robot" / "_robot_info.json"
TARGET_IMAGE_WIDTH = 640
TARGET_IMAGE_HEIGHT = 480



CAMERA_CANDIDATES = {
    "cam_high": [
        ("vision", "cam_head", "colors"),
    ],
    "cam_left_wrist": [
        ("vision", "cam_left_wrist", "colors"),
    ],
    "cam_right_wrist": [
        ("vision", "cam_right_wrist", "colors"),
    ],
    "cam_wrist": [
        ("vision", "cam_wrist", "colors"),
    ],
}

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    # Conservative defaults reduce RAM pressure during long conversions.
    image_writer_processes: int = 0
    image_writer_threads: int = 4
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def _load_env_metadata(env_cfg_type):
    env_cfg = load_yaml(str(ENV_CFG_ROOT / f"{env_cfg_type}.yml"))
    robot_name = env_cfg["config"]["robot"]
    robot_action_dim_info = load_json(str(ROBOT_INFO_PATH))[robot_name]
    fps = env_cfg.get("observation", {}).get("collect_freq", 50)
    return robot_name, robot_action_dim_info, fps


def _split_pattern(pattern):
    parts = pattern.split(".")
    if len(parts) != 3:
        raise ValueError(f"Pattern should be <dataset>.<task>.<env_cfg>, got: {pattern}")
    return parts[0], parts[1], parts[2]


def _match_name(name, pattern):
    return pattern == "*" or fnmatch.fnmatchcase(name, pattern)


def _discover_conversion_targets(patterns):
    targets = []
    seen = set()
    for pattern in patterns:
        dataset_pattern, task_pattern, env_pattern = _split_pattern(pattern)
        if not DATA_ROOT.exists():
            break

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


def _dims_from_robot_action_info(robot_action_dim_info):
    arm_dims = robot_action_dim_info.get("arm_dim", [])
    ee_dims = robot_action_dim_info.get("ee_dim", [])
    per_arm_dims = []
    for index, arm_dim in enumerate(arm_dims):
        ee_dim = ee_dims[index] if index < len(ee_dims) else 0
        per_arm_dims.append(arm_dim + ee_dim)
    return per_arm_dims


def _plan_target_metadata(targets):
    max_per_arm_dims = []
    metadata = {}
    max_fps = 0
    for dataset_name, task_name, env_cfg_type in targets:
        robot_name, robot_action_dim_info, fps = _load_env_metadata(env_cfg_type)
        per_arm_dims = _dims_from_robot_action_info(robot_action_dim_info)
        if len(max_per_arm_dims) < len(per_arm_dims):
            max_per_arm_dims.extend([0] * (len(per_arm_dims) - len(max_per_arm_dims)))
        for index, dim in enumerate(per_arm_dims):
            max_per_arm_dims[index] = max(max_per_arm_dims[index], dim)
        max_fps = max(max_fps, fps)
        metadata[(dataset_name, task_name, env_cfg_type)] = {
            "robot_name": robot_name,
            "robot_action_dim_info": robot_action_dim_info,
            "fps": fps,
            "per_arm_dims": per_arm_dims,
        }
    return metadata, max_per_arm_dims, max_fps


def _build_motor_names(robot_action_dim_info):
    arm_dims = robot_action_dim_info.get("arm_dim", [])
    ee_dims = robot_action_dim_info.get("ee_dim", [])
    if not arm_dims:
        raise ValueError("robot_action_dim_info.arm_dim is empty")

    if len(arm_dims) == 1:
        prefixes = ["arm"]
    elif len(arm_dims) == 2:
        prefixes = ["left", "right"]
    else:
        prefixes = [f"arm_{index}" for index in range(len(arm_dims))]

    motors = []
    for index, prefix in enumerate(prefixes):
        for joint_idx in range(arm_dims[index]):
            motors.append(f"{prefix}_arm_joint_{joint_idx}")

        ee_dim = ee_dims[index] if index < len(ee_dims) else 0
        if ee_dim == 1:
            motors.append(f"{prefix}_gripper")
        else:
            for ee_idx in range(ee_dim):
                motors.append(f"{prefix}_ee_joint_{ee_idx}")

    return motors


def _build_motor_names_from_dims(per_arm_dims):
    if not per_arm_dims:
        raise ValueError("per_arm_dims is empty")

    if len(per_arm_dims) == 1:
        prefixes = ["arm"]
    elif len(per_arm_dims) == 2:
        prefixes = ["left", "right"]
    else:
        prefixes = [f"arm_{index}" for index in range(len(per_arm_dims))]

    motors = []
    for index, total_dim in enumerate(per_arm_dims):
        prefix = prefixes[index]
        for joint_idx in range(total_dim):
            motors.append(f"{prefix}_joint_{joint_idx}")
    return motors


def _expected_state_dim(robot_action_dim_info):
    return sum(robot_action_dim_info.get("arm_dim", [])) + sum(robot_action_dim_info.get("ee_dim", []))


def _pad_state_to_target_dims(array, current_dims, target_dims, name):
    arr = _ensure_2d_float32(array, name)
    expected_dim = sum(current_dims)
    if arr.shape[1] != expected_dim:
        raise ValueError(f"{name} dim mismatch: expected {expected_dim}, got {arr.shape[1]}")

    padded_parts = []
    start = 0
    for arm_index, target_dim in enumerate(target_dims):
        current_dim = current_dims[arm_index] if arm_index < len(current_dims) else 0
        end = start + current_dim
        part = arr[:, start:end] if current_dim else np.zeros((arr.shape[0], 0), dtype=np.float32)
        start = end
        if target_dim < current_dim:
            raise ValueError(
                f"{name} target dim for arm {arm_index} is smaller than current dim: {target_dim} < {current_dim}"
            )
        if target_dim > current_dim:
            padding = np.zeros((arr.shape[0], target_dim - current_dim), dtype=np.float32)
            part = np.concatenate([part, padding], axis=1)
        padded_parts.append(part)

    if start != arr.shape[1]:
        raise ValueError(f"{name} dim mismatch after padding: consumed {start}, actual {arr.shape[1]}")
    return np.concatenate(padded_parts, axis=1) if padded_parts else arr


def _resolve_input_dir(dataset_name, task_name, env_cfg_type, input_dir=None):
    if input_dir is not None:
        return Path(input_dir)
    return DATA_ROOT / dataset_name / task_name / env_cfg_type / "data"


def _default_repo_id(dataset_name, task_name, env_cfg_type):
    return f"{dataset_name}_{task_name}_{env_cfg_type}".replace("/", "_").lower()

def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    motors: list[str],
    fps: int,
    mode: Literal["video", "image"] = "video",
    *,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    
) -> LeRobotDataset:
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


    for cam in CAMERA_CANDIDATES.keys():
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, TARGET_IMAGE_HEIGHT, TARGET_IMAGE_WIDTH),
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
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )

def _get_nested(data, *keys, default=None):
    cur = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _ensure_2d_float32(array, name):
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{name} should be 2D, got shape {arr.shape}")
    return arr


def _ensure_utf8_strings(value):
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
        result = []
        for item in value:
            result.extend(_ensure_utf8_strings(item))
        return result
    return [str(value)]


def _find_instructions(data):
    candidates = [
        _get_nested(data, "instructions"),
        _get_nested(data, "instruction"),
    ]
    for candidate in candidates:
        strings = [s for s in _ensure_utf8_strings(candidate) if s]
        if strings:
            return strings
    return []


def _choose_instruction(data):
    instructions = _find_instructions(data)
    if not instructions:
        return ""
    return random.choice(instructions)


def _find_camera_array(data, camera_name):
    for keys in CAMERA_CANDIDATES[camera_name]:
        value = _get_nested(data, *keys)
        if value is not None:
            return _decode_images_if_needed(value)
    return None


def _decode_one_image(frame):
    if isinstance(frame, np.ndarray) and frame.ndim == 3:
        if frame.dtype != np.uint8:
            return frame.astype(np.uint8)
        return frame

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


def _resize_image(image):
    if image.shape[:2] == (TARGET_IMAGE_HEIGHT, TARGET_IMAGE_WIDTH):
        return image
    return cv2.resize(image, (TARGET_IMAGE_WIDTH, TARGET_IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)


def _decode_images_if_needed(images):
    if isinstance(images, (bytes, bytearray, np.bytes_)):
        return np.stack([_resize_image(_decode_one_image(images))], axis=0).astype(np.uint8)

    arr = np.asarray(images)
    if arr.ndim == 4 and arr.dtype != object:
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        if arr.shape[1:3] != (TARGET_IMAGE_HEIGHT, TARGET_IMAGE_WIDTH):
            arr = np.stack([_resize_image(frame) for frame in arr], axis=0)
        return arr

    if arr.ndim == 3 and arr.dtype != object:
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return _resize_image(arr)[None, ...]

    if arr.ndim == 0:
        frames = [arr.item()]
    elif isinstance(images, np.ndarray) and images.dtype == object:
        frames = images.tolist()
    else:
        frames = list(images)

    decoded = [_resize_image(_decode_one_image(frame)) for frame in frames]
    return np.stack(decoded, axis=0).astype(np.uint8)


def _empty_image_sequence(num_frames):
    return np.zeros((num_frames, TARGET_IMAGE_HEIGHT, TARGET_IMAGE_WIDTH, 3), dtype=np.uint8)


def _concat_state_parts(parts, name):
    valid_parts = []
    horizon = None
    for part_name, value in parts:
        if value is None:
            continue
        arr = _ensure_2d_float32(value, f"{name}.{part_name}")
        if horizon is None:
            horizon = arr.shape[0]
        elif arr.shape[0] != horizon:
            raise ValueError(
                f"{name}.{part_name} horizon mismatch: expected {horizon}, got {arr.shape[0]}"
            )
        valid_parts.append(arr)

    if not valid_parts:
        return None
    return np.concatenate(valid_parts, axis=1)


def _extract_qpos(data):
    state_qpos = _concat_state_parts(
        [
            ("left_arm_joint_states", _get_nested(data, "state", "left_arm_joint_states")),
            ("left_ee_joint_states", _get_nested(data, "state", "left_ee_joint_states")),
            ("right_arm_joint_states", _get_nested(data, "state", "right_arm_joint_states")),
            ("right_ee_joint_states", _get_nested(data, "state", "right_ee_joint_states")),
        ],
        "state",
    )
    if state_qpos is not None:
        return state_qpos

    raise ValueError("Cannot find qpos data")


def _extract_action(data):
    action = _concat_state_parts(
        [
            ("left_arm_joint_states", _get_nested(data, "action", "left_arm_joint_states")),
            ("left_ee_joint_states", _get_nested(data, "action", "left_ee_joint_states")),
            ("right_arm_joint_states", _get_nested(data, "action", "right_arm_joint_states")),
            ("right_ee_joint_states", _get_nested(data, "action", "right_ee_joint_states")),
        ],
        "action",
    )
    if action is not None:
        return action

    raise ValueError("Cannot find action data")

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


def convert_one(input_path, dataset, data_type, data_version, current_dims, target_dims):
    data = load(str(input_path), data_type=data_type, data_version=data_version)

    state = _extract_qpos(data)
    action = _extract_action(data)
    if state.shape != action.shape:
        raise ValueError(f"state/action shape mismatch: {state.shape} vs {action.shape}")
    state = _pad_state_to_target_dims(state, current_dims, target_dims, "state")
    action = _pad_state_to_target_dims(action, current_dims, target_dims, "action")
    instruction = _choose_instruction(data)
    if not instruction:
        raise ValueError("No instruction found in data")
    
    images = {}
    for camera_name in CAMERA_CANDIDATES:
        image_array = _find_camera_array(data, camera_name)
        if image_array is not None:
            images[camera_name] = image_array
    
    num_frames = state.shape[0]
    for camera_name in CAMERA_CANDIDATES:
        if camera_name not in images:
            images[camera_name] = _empty_image_sequence(num_frames)

    for i in range(num_frames):
        frame = {
            "observation.state": np.asarray(state[i]),
            "action": np.asarray(action[i]),
        }

        for image_name, image_array in images.items():
            frame[f"observation.images.{image_name}"] = image_array[i]

        add_frame_compat(dataset, frame, instruction)

    save_episode_compat(dataset, instruction)
    release_dataset_memory(dataset)
    



def find_input_files(input_dir):
    input_dir = Path(input_dir)
    files = sorted(input_dir.rglob("*.hdf5"))
    files.extend(sorted(input_dir.rglob("*.h5")))
    unique_files = []
    seen = set()
    for file_path in files:
        resolved = str(file_path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique_files.append(file_path)
    return unique_files


def _collect_target_input_files(targets):
    collected = []
    for dataset_name, task_name, env_cfg_type in targets:
        input_dir = _resolve_input_dir(dataset_name, task_name, env_cfg_type)
        input_files = find_input_files(input_dir)
        collected.append((dataset_name, task_name, env_cfg_type, input_dir, input_files))
    return collected


def _print_matched_targets(target_inputs):
    print("Matched targets:")
    for dataset_name, task_name, env_cfg_type, input_dir, input_files in target_inputs:
        print(f"  - {dataset_name}/{task_name}/{env_cfg_type}: {len(input_files)} files from {input_dir}")


def main():
    parser = argparse.ArgumentParser(description="Transform RoboDojo dataset to LeRobot v2.1 format.")
    parser.add_argument(
        "patterns",
        nargs="+",
        help='Match expressions like "RoboDojo.*.arx_x5" or "RoboDojo.*.*"',
    )
    parser.add_argument("--repo_id", type=str, default=None, help="LeRobot repo_id. Defaults to unified_<dataset> patterns summary")
    parser.add_argument("--data_type", type=str, default=DEFAULT_DATASET_NAME, help="Dataset type, e.g. RoboDojo")
    parser.add_argument("--data_version", type=str, default="v1.0", help="Dataset version, e.g. v1.0")
    parser.add_argument("--max_episode", type=int, default=200, help="Path to environment config files")
    args = parser.parse_args()

    targets = _discover_conversion_targets(args.patterns)
    if not targets:
        raise FileNotFoundError(f"No matching dataset/task/env_cfg targets found for patterns: {args.patterns}")

    metadata_by_target, target_dims, max_fps = _plan_target_metadata(targets)
    target_inputs = _collect_target_input_files(targets)
    _print_matched_targets(target_inputs)
    repo_id = args.repo_id or f"unified_{'_'.join(pattern.replace('*', 'all').replace('.', '_') for pattern in args.patterns)}".lower()
    motors = _build_motor_names_from_dims(target_dims)
    
    dataset = create_empty_dataset(
                repo_id,
                robot_type="unified_robot",
                motors=motors,
                fps=max_fps or 50,
                mode="video",
                dataset_config=DEFAULT_DATASET_CONFIG,
        )

    failures = []
    total_files = 0
    total_success = 0
    for dataset_name, task_name, env_cfg_type, input_dir, input_files in target_inputs:
        task_success = 0

        if not input_files:
            failures.append((str(input_dir), "No .hdf5 or .h5 files found"))
            continue

        current_dims = metadata_by_target[(dataset_name, task_name, env_cfg_type)]["per_arm_dims"]
        for input_path in tqdm(input_files, desc=f"Converting {dataset_name}/{task_name}/{env_cfg_type}"):
            total_files += 1

            if task_success >= args.max_episode:
                print(
                    f"Reached max_episode={args.max_episode} "
                    f"for {dataset_name}/{task_name}/{env_cfg_type}"
                )
                break
            try:
                convert_one(input_path, dataset, args.data_type, args.data_version, current_dims, target_dims)
                task_success += 1
                total_success += 1
            except Exception as exc:
                print(exc)
                failures.append((str(input_path), str(exc)))

    print(f"Unified target dims per arm: {target_dims}")
    print(f"Converted {total_success}/{total_files} files to {HF_LEROBOT_HOME / repo_id}")
    if failures:
        print("Failed files:")
        for file_path, reason in failures:
            print(f"  - {file_path}: {reason}")


if __name__ == "__main__":
    main()
