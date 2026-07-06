"""Convert XPolicyLab episodes into the HDF5 layout used by TinyVLA."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np


POLICY_DIR = Path(__file__).resolve().parent
REPO_ROOT = POLICY_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from XPolicyLab.utils.process_data import (  # noqa: E402
    decode_image_bit,
    get_robot_action_dim_info,
    pack_robot_state,
)


# Order matters: index 0/1/2 map to TinyVLA's image / image_r / image_top inputs.
CAMERA_KEYS = ("cam_left_wrist", "cam_right_wrist", "cam_head")
TARGET_SIZE = (640, 480)  # (W, H), matching TinyVLA's native 480x640 HDF5 examples.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("bench_name")
    parser.add_argument("ckpt_name")
    parser.add_argument("env_cfg_type")
    parser.add_argument("action_type", choices=["joint", "ee"])
    parser.add_argument("expert_data_num", type=int, nargs="?", default=None,
                        help="Optional per-task episode cap; defaults to all episodes.")
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--tasks",
        default=None,
        help="Optional comma-separated task directory list; defaults to all tasks under source root.",
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--compression",
        choices=["lzf", "gzip", "none"],
        default="lzf",
        help="HDF5 dataset compression for decoded image arrays.",
    )
    return parser.parse_args()


def read_state_action(
    root: h5py.File,
    action_type: str,
    robot_action_dim_info: dict,
) -> tuple[np.ndarray, np.ndarray]:
    state_root = root["state"] if "state" in root else root["states"]
    action_root = root["action"] if "action" in root else root["actions"]
    state_data = {k: state_root[k][()] for k in state_root.keys()}
    action_data = {k: action_root[k][()] for k in action_root.keys()}
    for key, value in tuple(state_data.items()):
        if not key.endswith("s"):
            state_data.setdefault(f"{key}s", value)
    for key, value in tuple(action_data.items()):
        if not key.endswith("s"):
            action_data.setdefault(f"{key}s", value)

    data = {
        "state": state_data,
        "action": action_data,
    }
    state = pack_robot_state(
        data,
        action_type=action_type,
        robot_action_dim_info=robot_action_dim_info,
        source_type="dataset",
        state_type="state",
    ).astype(np.float32)
    action = pack_robot_state(
        data,
        action_type=action_type,
        robot_action_dim_info=robot_action_dim_info,
        source_type="dataset",
        state_type="action",
    ).astype(np.float32)
    return state, action


def read_instructions(root: h5py.File) -> list[str]:
    value = root["instructions"][()] if "instructions" in root else root["instruction"][()]
    value = decode_text(value)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return [value]
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(decode_text(item)) for item in value]
    return [str(value)]


def decode_text(value):
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, (bytes, bytearray, np.bytes_)):
        return value.decode("utf-8")
    return value


def decode_frame(raw) -> np.ndarray:
    if isinstance(raw, (bytes, bytearray, np.bytes_)):
        image = decode_image_bit(raw)
    else:
        raw = np.asarray(raw)
        if raw.ndim == 1:
            image = decode_image_bit(raw)
        else:
            image = raw

    image = np.asarray(image, dtype=np.uint8)
    if image.shape[:2] != (TARGET_SIZE[1], TARGET_SIZE[0]):
        image = cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(image)


def write_camera_images(
    src_root: h5py.File,
    dst_images: h5py.Group,
    camera_key: str,
    expected_len: int,
    compression: str,
) -> None:
    camera_root = src_root["vision"][camera_key]
    dataset = camera_root["colors"] if "colors" in camera_root else camera_root["color"]

    create_kwargs = {}
    if compression != "none":
        create_kwargs["compression"] = compression
        create_kwargs["shuffle"] = True

    out = dst_images.create_dataset(
        camera_key,
        shape=(expected_len, TARGET_SIZE[1], TARGET_SIZE[0], 3),
        dtype=np.uint8,
        chunks=(1, TARGET_SIZE[1], TARGET_SIZE[0], 3),
        **create_kwargs,
    )
    for frame_idx in range(expected_len):
        out[frame_idx] = decode_frame(dataset[frame_idx])


def write_tinyvla_episode(
    src_path: Path,
    dst_path: Path,
    task_name: str,
    action_type: str,
    robot_action_dim_info: dict,
    compression: str,
) -> int:
    with h5py.File(src_path, "r") as src:
        state, action = read_state_action(src, action_type, robot_action_dim_info)
        instructions = read_instructions(src)

        with h5py.File(dst_path, "w") as dst:
            # TinyVLA's official loader uses this attr to avoid the real-robot action lag.
            dst.attrs["sim"] = True
            dst.attrs["compress"] = False
            dst.attrs["xpolicylab_source"] = str(src_path)
            dst.attrs["xpolicylab_task_name"] = task_name
            dst.attrs["image_width"] = TARGET_SIZE[0]
            dst.attrs["image_height"] = TARGET_SIZE[1]
            dst.attrs["hdf5_image_compression"] = compression

            dst.create_dataset("action", data=action, dtype=np.float32)

            observations = dst.create_group("observations")
            observations.create_dataset("qpos", data=state, dtype=np.float32)
            observations.create_dataset("qvel", data=np.zeros_like(state), dtype=np.float32)

            images = observations.create_group("images")
            for camera_key in CAMERA_KEYS:
                write_camera_images(src, images, camera_key, state.shape[0], compression)

            string_dtype = h5py.string_dtype(encoding="utf-8")
            dst.create_dataset(
                "language_raw",
                data=np.asarray(instructions, dtype=object),
                dtype=string_dtype,
            )
            dst.create_dataset(
                "instructions",
                data=json.dumps(instructions, ensure_ascii=False),
                dtype=string_dtype,
            )

    return state.shape[0]


def convert_one_episode(job: tuple[int, str, Path, Path, str, dict, str]) -> tuple[int, str, int]:
    merged_idx, task_name, src_path, out_dir, action_type, robot_action_dim_info, compression = job
    dst_path = out_dir / f"episode_{merged_idx:07d}.hdf5"
    frames = write_tinyvla_episode(
        src_path,
        dst_path,
        task_name,
        action_type,
        robot_action_dim_info,
        compression,
    )
    return merged_idx, task_name, frames


def main() -> None:
    args = parse_args()
    ckpt_setting = (
        f"{args.bench_name}-{args.ckpt_name}-{args.env_cfg_type}"
        f"-{args.action_type}"
    )
    root = args.source_root if args.source_root is not None else (REPO_ROOT / "data" / args.bench_name)
    out_dir = args.output_dir if args.output_dir is not None else (POLICY_DIR / "data" / ckpt_setting)

    out_dir.mkdir(parents=True, exist_ok=False)

    robot_action_dim_info = get_robot_action_dim_info(args.env_cfg_type)
    task_filter = None
    if args.tasks:
        task_filter = {task.strip() for task in args.tasks.split(",") if task.strip()}
        if not task_filter:
            raise ValueError("--tasks was provided but no valid task names were parsed.")

    episodes: list[tuple[str, Path]] = []
    for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        task_name = task_dir.name
        if task_filter is not None and task_name not in task_filter:
            continue
        data_dir = task_dir / args.env_cfg_type / "data"
        episode_paths = sorted(data_dir.glob("episode_*.hdf5"))
        if args.expert_data_num is not None:
            episode_paths = episode_paths[: args.expert_data_num]
        for episode_path in episode_paths:
            episodes.append((task_name, episode_path))
        print(
            f"[TinyVLA process_data] task='{task_name}': "
            f"queued {len(episode_paths)} episodes"
        )
    if task_filter is not None:
        found_tasks = {task_name for task_name, _ in episodes}
        missing_tasks = sorted(task_filter - found_tasks)
        if missing_tasks:
            raise FileNotFoundError(
                "Requested task(s) not found with episodes under source root: "
                + ", ".join(missing_tasks)
            )
    workers = args.workers
    jobs = [
        (
            merged_idx,
            task_name,
            src_path,
            out_dir,
            args.action_type,
            robot_action_dim_info,
            args.compression,
        )
        for merged_idx, (task_name, src_path) in enumerate(episodes)
    ]

    total_frames = 0
    print(
        f"[TinyVLA process_data] converting {len(jobs)} episodes "
        f"with workers={workers}, compression={args.compression}"
    )
    if workers == 1:
        for job in jobs:
            merged_idx, task_name, frames = convert_one_episode(job)
            total_frames += frames
            print(
                f"[TinyVLA process_data] wrote episode_{merged_idx:07d}.hdf5: "
                f"{frames} frames from {task_name}"
            )
    else:
        completed = 0
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(convert_one_episode, job) for job in jobs]
            for future in as_completed(futures):
                merged_idx, task_name, frames = future.result()
                completed += 1
                total_frames += frames
                print(
                    f"[TinyVLA process_data] [{completed}/{len(jobs)}] "
                    f"wrote episode_{merged_idx:07d}.hdf5: {frames} frames from {task_name}"
                )

    print(
        f"[TinyVLA process_data] total: {len(episodes)} episodes, "
        f"{total_frames} frames -> {out_dir}"
    )


if __name__ == "__main__":
    main()
