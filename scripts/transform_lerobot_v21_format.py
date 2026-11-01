import argparse
import dataclasses
import fnmatch
import glob
import random
import sys
from pathlib import Path
from typing import Literal

import av
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import shutil
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from lerobot.datasets import lerobot_dataset as lerobot_dataset_mod
from lerobot.datasets.video_utils import encode_video_frames as encode_video_frames_orig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from XPolicyLab.utils.data_loader import load
from XPolicyLab.utils.load_file import load_json, load_yaml
from XPolicyLab.utils.process_data import decode_image_bit
from XPolicyLab.utils import wuji58


DEFAULT_DATASET_NAME = "RoboDojo"
DATA_ROOT = PROJECT_ROOT / "data"
ENV_CFG_ROOT = PROJECT_ROOT / "env_cfg"
ROBOT_INFO_PATH = ENV_CFG_ROOT / "robot" / "_robot_info.json"
DEFAULT_IMAGE_WIDTH = 640
DEFAULT_IMAGE_HEIGHT = 480
# Common source resolutions: 480x640 (RoboDojo) and 240x320 (RoboTwin).
SUPPORTED_IMAGE_SIZES = ((480, 640), (240, 320))



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
}

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    # Conservative defaults reduce RAM pressure during long conversions.
    image_writer_processes: int = 0
    image_writer_threads: int = 4
    # Decoder. None keeps LeRobot's default decoder. It does not select the encoder.
    video_backend: str | None = None
    # Encoder. LeRobot 0.3.3 create() does not take vcodec; this script
    # replaces encode_video_frames in-process from this field.
    # libsvtav1: AV1, CRF 30, g=2, yuv420p.
    # h264_nvenc: NVIDIA H.264, rc=constqp, qp=18, yuv420p. Not video_backend.
    vcodec: Literal["libsvtav1", "h264_nvenc"] = "libsvtav1"


DEFAULT_DATASET_CONFIG = DatasetConfig()


def _encode_video_frames_nvenc(imgs_dir, video_path, fps, overwrite=False) -> None:
    # LeRobot 0.3.3 rejects any vcodec outside h264/hevc/libsvtav1, so NVENC is
    # encoded here. The inputs are the PNG frames LeRobot itself wrote, not
    # XPolicyLab trajectories, so they are read with PIL like LeRobot does.
    video_path = Path(video_path)
    imgs_dir = Path(imgs_dir)
    video_path.parent.mkdir(parents=True, exist_ok=overwrite)

    template = "frame_" + ("[0-9]" * 6) + ".png"
    input_list = sorted(
        glob.glob(str(imgs_dir / template)),
        key=lambda x: int(x.split("_")[-1].split(".")[0]),
    )
    if not input_list:
        raise FileNotFoundError(f"No images found in {imgs_dir}.")
    with Image.open(input_list[0]) as dummy_image:
        width, height = dummy_image.size

    # No GOP option: NVENC refuses to open with a keyframe interval below 4, so
    # the g=2 used for AV1 cannot be matched. LeRobot 0.4.4 likewise omits it for
    # hardware encoders (_get_codec_options in datasets/video_utils.py).
    options = {"rc": "constqp", "qp": "18"}
    with av.open(str(video_path), "w") as output:
        output_stream = output.add_stream("h264_nvenc", fps, options=options)
        output_stream.pix_fmt = "yuv420p"
        output_stream.width = width
        output_stream.height = height
        for input_data in input_list:
            with Image.open(input_data) as input_image:
                input_frame = av.VideoFrame.from_image(input_image.convert("RGB"))
            packet = output_stream.encode(input_frame)
            if packet:
                output.mux(packet)
        packet = output_stream.encode()
        if packet:
            output.mux(packet)

    if not video_path.exists():
        raise OSError(f"Video encoding did not work. File not found: {video_path}.")


def configure_video_encoding(dataset_config: DatasetConfig) -> None:
    if dataset_config.vcodec == "libsvtav1":

        def encode_video_frames(imgs_dir, video_path, fps, overwrite=False, **_kwargs):
            return encode_video_frames_orig(
                imgs_dir,
                video_path,
                fps,
                vcodec="libsvtav1",
                pix_fmt="yuv420p",
                g=2,
                crf=30,
                overwrite=overwrite,
            )

    elif dataset_config.vcodec == "h264_nvenc":

        def encode_video_frames(imgs_dir, video_path, fps, overwrite=False, **_kwargs):
            return _encode_video_frames_nvenc(
                imgs_dir, video_path, fps, overwrite=overwrite
            )

    else:
        raise ValueError(
            f"Unsupported vcodec {dataset_config.vcodec!r}; "
            "use 'libsvtav1' or 'h264_nvenc'"
        )

    # Fail loudly rather than write the wrong codec if a future LeRobot stops
    # calling encode_video_frames as a module global.
    current = getattr(lerobot_dataset_mod, "encode_video_frames", None)
    if current is not encode_video_frames_orig and not getattr(current, "_vcodec_patched", False):
        raise RuntimeError(
            "LeRobot no longer calls encode_video_frames as a module global; "
            "the vcodec selection would be a no-op"
        )

    encode_video_frames._vcodec_patched = True
    lerobot_dataset_mod.encode_video_frames = encode_video_frames


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
    for bench_name, task_name, env_cfg_type in targets:
        robot_name, robot_action_dim_info, fps = _load_env_metadata(env_cfg_type)
        per_arm_dims = _dims_from_robot_action_info(robot_action_dim_info)
        if len(max_per_arm_dims) < len(per_arm_dims):
            max_per_arm_dims.extend([0] * (len(per_arm_dims) - len(max_per_arm_dims)))
        for index, dim in enumerate(per_arm_dims):
            max_per_arm_dims[index] = max(max_per_arm_dims[index], dim)
        max_fps = max(max_fps, fps)
        metadata[(bench_name, task_name, env_cfg_type)] = {
            "robot_name": robot_name,
            "robot_action_dim_info": robot_action_dim_info,
            "fps": fps,
            "per_arm_dims": per_arm_dims,
        }
    return metadata, max_per_arm_dims, max_fps


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


def _resolve_input_dir(bench_name, task_name, env_cfg_type, input_dir=None):
    if input_dir is not None:
        return Path(input_dir)
    return DATA_ROOT / bench_name / task_name / env_cfg_type / "data"


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    motors: list[str],
    fps: int,
    mode: Literal["video", "image"] = "video",
    *,
    image_height: int = DEFAULT_IMAGE_HEIGHT,
    image_width: int = DEFAULT_IMAGE_WIDTH,
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
            "shape": (3, image_height, image_width),
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


def _find_camera_array(data, camera_name, image_height, image_width):
    for keys in CAMERA_CANDIDATES[camera_name]:
        value = _get_nested(data, *keys)
        if value is not None:
            return _decode_images_if_needed(value, image_height, image_width)
    return None


def _resize_image(image, image_height, image_width):
    if image.shape[:2] == (image_height, image_width):
        return image
    return cv2.resize(image, (image_width, image_height), interpolation=cv2.INTER_LINEAR)


def _decode_images_if_needed(images, image_height, image_width):
    frames = np.asarray(decode_image_bit(images))
    if frames.ndim == 3:
        frames = frames[None, ...]
    if frames.ndim != 4:
        raise ValueError(f"Expected decoded frames with shape [T,H,W,3], got {frames.shape}")
    if frames.dtype != np.uint8:
        frames = frames.astype(np.uint8)
    if frames.shape[1:3] != (image_height, image_width):
        frames = np.stack(
            [_resize_image(frame, image_height, image_width) for frame in frames],
            axis=0,
        )
    return frames


def _empty_image_sequence(num_frames, image_height, image_width):
    return np.zeros((num_frames, image_height, image_width, 3), dtype=np.uint8)


def _probe_raw_image_hw(target_inputs, data_type, data_version):
    """Return (H, W) from the first readable camera frame, without resizing."""
    for _, _, _, _, input_files in target_inputs:
        for input_path in input_files:
            data = load(str(input_path), data_type=data_type, data_version=data_version)
            for camera_name in CAMERA_CANDIDATES:
                for keys in CAMERA_CANDIDATES[camera_name]:
                    value = _get_nested(data, *keys)
                    if value is None:
                        continue
                    frames = np.asarray(decode_image_bit(value))
                    if frames.ndim == 3:
                        return int(frames.shape[0]), int(frames.shape[1])
                    if frames.ndim == 4 and frames.shape[0] > 0:
                        return int(frames.shape[1]), int(frames.shape[2])
    return DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH


def _parse_resolution(text: str) -> tuple[int, int]:
    """Parse '240x320' / '240,320' into (height, width)."""
    normalized = text.lower().replace(",", "x").replace("*", "x")
    parts = [p.strip() for p in normalized.split("x") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"resolution must look like HxW (e.g. 240x320), got: {text}"
        )
    height, width = int(parts[0]), int(parts[1])
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError(f"invalid resolution: {text}")
    return height, width


def _resolve_image_hw(args, target_inputs) -> tuple[int, int]:
    if args.resolution is not None:
        image_hw = _parse_resolution(args.resolution)
    elif args.image_height is not None or args.image_width is not None:
        if args.image_height is None or args.image_width is None:
            raise ValueError("Pass both --image_height and --image_width, or use --resolution")
        image_hw = (args.image_height, args.image_width)
    else:
        image_hw = _probe_raw_image_hw(
            target_inputs,
            args.data_type,
            args.data_version,
        )

    if image_hw not in SUPPORTED_IMAGE_SIZES:
        supported = ", ".join(f"{h}x{w}" for h, w in SUPPORTED_IMAGE_SIZES)
        print(
            f"[warn] Using image size {image_hw[0]}x{image_hw[1]}; "
            f"commonly tested sizes are: {supported}"
        )
    return image_hw


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


def convert_one(
    input_path,
    dataset,
    data_type,
    data_version,
    current_dims,
    target_dims,
    image_height,
    image_width,
    representation="joint",
    wrist_frame=None,
):
    data = load(str(input_path), data_type=data_type, data_version=data_version)

    if representation == "wuji58":
        state, action = wuji58.trajectory58_from_data(data, wrist_frame)
        state = state.astype(np.float32)
        action = action.astype(np.float32)
    else:
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
        image_array = _find_camera_array(data, camera_name, image_height, image_width)
        if image_array is not None:
            images[camera_name] = image_array
    
    num_frames = state.shape[0]
    for camera_name in CAMERA_CANDIDATES:
        if camera_name not in images:
            images[camera_name] = _empty_image_sequence(num_frames, image_height, image_width)

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
    for bench_name, task_name, env_cfg_type in targets:
        input_dir = _resolve_input_dir(bench_name, task_name, env_cfg_type)
        input_files = find_input_files(input_dir)
        collected.append((bench_name, task_name, env_cfg_type, input_dir, input_files))
    return collected


def _print_matched_targets(target_inputs):
    print("Matched targets:")
    for bench_name, task_name, env_cfg_type, input_dir, input_files in target_inputs:
        print(f"  - {bench_name}/{task_name}/{env_cfg_type}: {len(input_files)} files from {input_dir}")


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
    parser.add_argument("--max_episode", type=int, default=200, help="Max episodes per task/env")
    parser.add_argument(
        "--representation",
        choices=("joint", "wuji58"),
        default="joint",
        help="State/action layout: 'joint' (default) or 'wuji58' "
        "(tianji_wuji hybrid: 20 hand joints + 9-D wrist pose per side; see utils/wuji58.py).",
    )
    parser.add_argument(
        "--wrist_frame",
        choices=wuji58.WRIST_FRAMES,
        default="te2",
        help="wuji58 only: frame the wrist poses are expressed in (default te2).",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default=None,
        help="Target image size as HxW, e.g. 240x320 or 480x640. "
        "Default: auto-detect from the first source frame.",
    )
    parser.add_argument(
        "--image_height",
        type=int,
        default=None,
        help="Override target image height (use with --image_width).",
    )
    parser.add_argument(
        "--image_width",
        type=int,
        default=None,
        help="Override target image width (use with --image_height).",
    )
    args = parser.parse_args()
    configure_video_encoding(DEFAULT_DATASET_CONFIG)

    targets = _discover_conversion_targets(args.patterns)
    if not targets:
        raise FileNotFoundError(f"No matching dataset/task/env_cfg targets found for patterns: {args.patterns}")

    metadata_by_target, target_dims, max_fps = _plan_target_metadata(targets)
    wrist_frame = None
    if args.representation == "wuji58":
        wrist_frame = wuji58.WristFrame.from_name(args.wrist_frame)
        for (bench_name, task_name, env_cfg_type), meta in metadata_by_target.items():
            if not wuji58.is_wuji58(meta["robot_action_dim_info"], "ee"):
                raise ValueError(
                    f"--representation wuji58 needs ee_dim == [20, 20] on every target, but "
                    f"{bench_name}/{task_name}/{env_cfg_type} resolves to robot "
                    f"{meta['robot_name']!r} with {meta['robot_action_dim_info']}"
                )
    target_inputs = _collect_target_input_files(targets)
    _print_matched_targets(target_inputs)
    image_height, image_width = _resolve_image_hw(args, target_inputs)
    print(f"Image size: {image_height}x{image_width}")
    repo_id = args.repo_id or f"unified_{'_'.join(pattern.replace('*', 'all').replace('.', '_') for pattern in args.patterns)}".lower()
    if args.representation == "wuji58":
        motors = list(wuji58.MOTOR_NAMES)
        robot_type = "tianji_wuji"
    else:
        motors = _build_motor_names_from_dims(target_dims)
        robot_type = "unified_robot"

    dataset = create_empty_dataset(
                repo_id,
                robot_type=robot_type,
                motors=motors,
                fps=max_fps or 50,
                mode="video",
                image_height=image_height,
                image_width=image_width,
                dataset_config=DEFAULT_DATASET_CONFIG,
        )

    failures = []
    total_files = 0
    total_success = 0
    for bench_name, task_name, env_cfg_type, input_dir, input_files in target_inputs:
        task_success = 0

        if not input_files:
            failures.append((str(input_dir), "No .hdf5 or .h5 files found"))
            continue

        current_dims = metadata_by_target[(bench_name, task_name, env_cfg_type)]["per_arm_dims"]
        for input_path in tqdm(input_files, desc=f"Converting {bench_name}/{task_name}/{env_cfg_type}"):
            total_files += 1

            if task_success >= args.max_episode:
                print(
                    f"Reached max_episode={args.max_episode} "
                    f"for {bench_name}/{task_name}/{env_cfg_type}"
                )
                break
            try:
                convert_one(
                    input_path,
                    dataset,
                    args.data_type,
                    args.data_version,
                    current_dims,
                    target_dims,
                    image_height,
                    image_width,
                    representation=args.representation,
                    wrist_frame=wrist_frame,
                )
                task_success += 1
                total_success += 1
            except Exception as exc:
                print(exc)
                failures.append((str(input_path), str(exc)))

    if args.representation == "wuji58":
        info_path = Path(HF_LEROBOT_HOME) / repo_id / "meta" / "info.json"
        if info_path.is_file():
            import json
            info = load_json(str(info_path))
            info["repr_mode"] = wuji58.REPR_MODE
            info["robot"] = "tianji_wuji"
            info["wrist_frame"] = args.wrist_frame
            info_path.write_text(json.dumps(info, indent=4) + "\n")
        print(f"Representation: wuji58 ({wuji58.REPR_MODE}, wrist_frame={args.wrist_frame})")
    else:
        print(f"Unified target dims per arm: {target_dims}")
    print(f"Converted {total_success}/{total_files} files to {HF_LEROBOT_HOME / repo_id}")
    if failures:
        print("Failed files:")
        for file_path, reason in failures:
            print(f"  - {file_path}: {reason}")


if __name__ == "__main__":
    main()
