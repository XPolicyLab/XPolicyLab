from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from tqdm import tqdm

from XPolicyLab.utils.load_file import load_hdf5
from XPolicyLab.utils.process_data import decode_image_bit, get_robot_action_dim_info, pack_robot_state


POLICY_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = POLICY_DIR.parents[2]
OFFLINE_DIR = POLICY_DIR / "RISE" / "policy_and_value" / "policy_offline_and_value"
MINI_LEROBOT_SRC = OFFLINE_DIR / "mini_lerobot" / "src"

if str(MINI_LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(MINI_LEROBOT_SRC))

from mini_lerobot.builder import LeRobotDatasetBuilder


FEATURES = {
    "observation.images.top_head": {
        "dtype": "video",
        "shape": [240, 320, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 30.0,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.images.hand_left": {
        "dtype": "video",
        "shape": [240, 320, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 30.0,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.images.hand_right": {
        "dtype": "video",
        "shape": [240, 320, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 30.0,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.state": {"dtype": "float32", "shape": (14,)},
    "action": {"dtype": "float32", "shape": (14,)},
    "action_advantage": {"dtype": "float32", "shape": (1,)},
}

CAMERA_MAP = {
    "observation.images.top_head": "cam_head",
    "observation.images.hand_left": "cam_left_wrist",
    "observation.images.hand_right": "cam_right_wrist",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bench_name")
    parser.add_argument("ckpt_name")
    parser.add_argument("env_cfg_type")
    parser.add_argument("expert_data_num", type=int)
    parser.add_argument("action_type")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--prompt", default=None)
    args = parser.parse_args()

    if args.action_type != "joint":
        raise ValueError("RISE data conversion expects action_type=joint.")

    data_dir = args.data_dir or REPO_ROOT / "data" / args.bench_name / args.ckpt_name / args.env_cfg_type
    if not data_dir.exists():
        raise FileNotFoundError(f"Cannot find XPolicyLab data directory: {data_dir.relative_to(REPO_ROOT)}")

    robot_action_dim_info = get_robot_action_dim_info(args.env_cfg_type)
    action_dim = sum(robot_action_dim_info["arm_dim"]) + sum(robot_action_dim_info["ee_dim"])
    if action_dim != 14:
        raise ValueError(f"RISE release configs expect 14-D joint actions, got {action_dim} for {args.env_cfg_type}.")

    output_root = POLICY_DIR / "data" / (
        f"{args.bench_name}-{args.ckpt_name}-{args.env_cfg_type}-{args.action_type}-lerobot"
    )
    if output_root.exists():
        shutil.rmtree(output_root)

    episode_paths = [
        data_dir / "data" / f"episode_{episode_idx:07d}.hdf5"
        for episode_idx in range(args.expert_data_num)
    ]
    missing = [path for path in episode_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing episode file: {missing[0].relative_to(REPO_ROOT)}")

    builder = LeRobotDatasetBuilder(
        repo_id=output_root.name,
        fps=30,
        features=FEATURES,
        robot_type="dual_x5",
        root=output_root,
    )

    bar = tqdm(episode_paths, desc=f"convert {output_root.name}", unit="ep", dynamic_ncols=True)
    for episode_index, episode_path in enumerate(bar):
        builder.add_episode(produce_episode, episode_path, args.prompt, robot_action_dim_info, args.action_type)
        bar.set_postfix(ep=episode_index + 1, total=len(episode_paths))
    bar.close()

    builder.flush()
    tqdm.write(f"[RISE] Converted dataset: {output_root.relative_to(POLICY_DIR)} ({len(episode_paths)} episodes)")


def produce_episode(video_map, episode_path, prompt, robot_action_dim_info, action_type):
    data = load_hdf5(episode_path)
    state = pack_robot_state(data, action_type, robot_action_dim_info, source_type="dataset", state_type="state").astype(
        np.float32
    )
    action = pack_robot_state(data, action_type, robot_action_dim_info, source_type="dataset", state_type="action").astype(
        np.float32
    )
    tasks = [_extract_prompt(data, prompt)] * state.shape[0]

    for video_key, camera_name in CAMERA_MAP.items():
        frames = [_decode_standard_rgb(frame_bits) for frame_bits in data["vision"][camera_name]["colors"]]
        encode_video_frames(np.asarray(frames, dtype=np.uint8), video_map[video_key], fps=30)

    action_advantage = np.ones((state.shape[0], 1), dtype=np.float32)

    return {
        "observation.state": state,
        "action": action,
        "action_advantage": action_advantage,
    }, tasks


def _decode_standard_rgb(frame_bits):
    image = decode_image_bit(frame_bits)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 image, got {image.shape}.")

    image = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
    if image.shape != (240, 320, 3):
        raise ValueError(f"Expected standardized image shape (240, 320, 3), got {image.shape}.")
    return image


def encode_video_frames(images, dst, fps):
    try:
        import av
    except ImportError as exc:
        raise ImportError("RISE data conversion requires PyAV. Run the RISE policy install script first.") from exc

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(dst), "w") as output:
        stream = output.add_stream("libsvtav1", fps, options={"crf": "30", "g": "2"})
        stream.pix_fmt = "yuv420p"
        stream.width = int(images.shape[2])
        stream.height = int(images.shape[1])
        for image in images:
            frame = av.VideoFrame.from_ndarray(image, format="rgb24", channel_last=True)
            packet = stream.encode(frame)
            if packet:
                output.mux(packet)
        packet = stream.encode()
        if packet:
            output.mux(packet)


def _extract_prompt(data, override):
    if override:
        return override
    instruction = data.get("instructions", data.get("instruction", ""))
    if isinstance(instruction, bytes):
        instruction = instruction.decode("utf-8", errors="replace")
    if isinstance(instruction, str):
        try:
            parsed = json.loads(instruction)
            if isinstance(parsed, list) and parsed:
                return str(parsed[0])
        except json.JSONDecodeError:
            return instruction
    if isinstance(instruction, (list, tuple)) and instruction:
        return str(instruction[0])
    return ""


if __name__ == "__main__":
    main()
