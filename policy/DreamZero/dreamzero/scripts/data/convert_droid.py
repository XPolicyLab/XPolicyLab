"""
Convert DROID 1.0.1 (RLDS/TFDS format) to LeRobot format with idle frame filtering.

This script takes the raw DROID dataset in RLDS format, applies idle frame filtering
using a pre-computed JSON file of non-idle frame ranges, filters out failed episodes
and episodes without language annotations, and outputs the dataset in LeRobot v2.0 format.

The idle filtering is based on Physical Intelligence's approach (see openpi):
  https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/README_train.md

The pre-computed idle filter ranges can be downloaded from:
  gsutil cp gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json <path>

Usage:
  python scripts/data/convert_droid.py <raw_dir> <output_dir> \\
      --keep-ranges-path <path/to/keep_ranges.json> \\
      [--fps 15] [--first-n N] [-n 16] [--filter-failed]

Example:
  # Download DROID 1.0.1 raw dataset
  gsutil -m cp -r gs://gresearch/robotics/droid/1.0.1 ./data/droid/1.0.1

  # Download idle filter ranges from openpi
  gsutil cp gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json ./data/keep_ranges.json

  # Run conversion
  python scripts/data/convert_droid.py ./data/droid/1.0.1 ./data/droid_lerobot \\
      --keep-ranges-path ./data/keep_ranges.json --filter-failed

Original dataset structure (RLDS):
  - 3 camera views: exterior_image_1_left, exterior_image_2_left, wrist_image_left
  - State: cartesian_position (6), gripper_position (1), joint_position (7)
  - Action: cartesian_position (6), cartesian_velocity (6), gripper_position (1),
            gripper_velocity (1), joint_position (7), joint_velocity (7)
  - Language instructions (up to 3 per episode)

Credits:
  - Original conversion script by Loic Magne (NVIDIA)
  - Idle filtering by Scott Reed (NVIDIA), based on Physical Intelligence's approach
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path

import av
import numpy as np
import polars as pl
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
import tqdm

# Limit thread counts to avoid oversubscription in multiprocessing
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MPI_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

tf.get_logger().setLevel("WARN")
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.set_soft_device_placement(True)


def tf_to_torch(data):
    return torch.from_numpy(data.numpy())


def tf_img_convert(img):
    if img.dtype == tf.string:
        img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
    elif img.dtype != tf.uint8:
        raise ValueError(f"Unsupported image dtype: found with dtype {img.dtype}")
    return img.numpy()


def _broadcast_metadata_rlds(i: tf.Tensor, traj: dict) -> dict:
    steps = traj.pop("steps")
    traj_len = tf.shape(tf.nest.flatten(steps)[0])[0]
    metadata = tf.nest.map_structure(lambda x: tf.repeat(x, traj_len), traj)
    traj = {**steps, "traj_metadata": metadata}
    traj["_len"] = tf.repeat(traj_len, traj_len)
    traj["_traj_index"] = tf.repeat(i, traj_len)
    traj["_frame_index"] = tf.range(traj_len)
    return traj


def concat_state_or_action(modality_dict, keys, compute_concat_info=False):
    arrays = []
    if compute_concat_info:
        concat_info = {}
        start_index = 0
    for key in keys:
        array = tf_to_torch(modality_dict[key])
        arrays.append(array)
        if compute_concat_info:
            D = array.shape[1]
            data_dtype = array.numpy().dtype
            if np.issubdtype(data_dtype, bool):
                data_dtype = "int64"
                data_range = [0, 1]
            else:
                data_dtype = data_dtype.name
                data_range = None
            concat_info[key] = {
                "start": start_index,
                "end": start_index + D,
            }
            if data_dtype != "float64":
                concat_info[key]["dtype"] = data_dtype
            if data_range is not None:
                concat_info[key]["range"] = data_range
            start_index += D
    concatenated = torch.cat(arrays, dim=1)
    ret_dict = {}
    if compute_concat_info:
        ret_dict["concat_info"] = concat_info
    ret_dict["concatenated"] = concatenated
    return ret_dict


def encode_video(frames: np.ndarray, output_path: Path, fps: int) -> None:
    """Encode a sequence of frames to a video file using PyAV."""
    options = {
        "threads": "1",
        "thread_type": "slice",
        "preset": "ultrafast",
        "tune": "zerolatency",
        "crf": "23",
    }

    container = av.open(str(output_path), mode="w")
    stream = container.add_stream("h264", rate=fps, options=options)
    stream.width = frames.shape[2]
    stream.height = frames.shape[1]
    stream.pix_fmt = "yuv420p"

    video_frame = av.VideoFrame(width=stream.width, height=stream.height, format="rgb24")
    frame_array = video_frame.to_ndarray(format="rgb24")

    for frame in frames:
        frame_array[:] = frame
        packet = stream.encode(video_frame)
        container.mux(packet)

    packet = stream.encode(None)
    container.mux(packet)
    container.close()


def process_tfrecord(
    ith_shard,
    raw_dir,
    output_path,
    fps,
    all_tasks,
    state_keys,
    action_keys,
    lang_keys,
    image_keys,
    start_episode_idx,
    kept_registry,
    keep_ranges_path,
):
    config = tfds.ReadConfig(
        try_autocache=False,
        num_parallel_calls_for_decode=1,
        num_parallel_calls_for_interleave_files=1,
        interleave_cycle_length=1,
        shuffle_reshuffle_each_iteration=False,
    )

    ds_builder = tfds.builder_from_directory(str(raw_dir))
    dataset = ds_builder.as_dataset(
        split=f"train[{ith_shard}shard]",
        decoders={"steps": tfds.decode.SkipDecoding()},
        read_config=config,
    )

    dataset = dataset.enumerate().map(_broadcast_metadata_rlds)
    all_keep_ranges = json.load(open(keep_ranges_path, "r"))

    episodes_data = []
    for local_idx, episode in enumerate(dataset):

        # Add keep frame info to episode.
        file_path = (
            episode["traj_metadata"]["episode_metadata"]["file_path"][0].numpy().decode("utf-8")
        )
        recording_folderpath = (
            episode["traj_metadata"]["episode_metadata"]["recording_folderpath"][0]
            .numpy()
            .decode("utf-8")
        )
        idle_key = f"{recording_folderpath}--{file_path}"
        keep_ranges = all_keep_ranges[idle_key]

        global_episode_idx = start_episode_idx + local_idx

        # check if the episode has been filtered
        if global_episode_idx not in kept_registry:
            continue

        episode_idx = kept_registry[global_episode_idx]
        episode_data = process_sample(
            episode_idx,
            episode,
            output_path,
            fps,
            all_tasks,
            state_keys,
            action_keys,
            lang_keys,
            image_keys,
            keep_ranges,
        )
        episodes_data.append(episode_data)
    return episodes_data


def process_sample(
    ep_idx,
    episode,
    output_path,
    fps,
    all_tasks,
    state_keys,
    action_keys,
    lang_keys,
    image_keys,
    keep_ranges,
):
    chunk_idx = ep_idx // 1000

    # Create chunk directory
    (output_path / f"data/chunk-{chunk_idx:03d}").mkdir(parents=True, exist_ok=True)
    for img_key in image_keys:
        (output_path / f"videos/chunk-{chunk_idx:03d}/observation.images.{img_key}").mkdir(
            parents=True, exist_ok=True
        )

    # Use concat_state_or_action for state and action
    state_dict = concat_state_or_action(episode["observation"], state_keys)
    action_dict = concat_state_or_action(episode["action_dict"], action_keys)

    # Count number of non-idle frames.
    num_frames = len(episode["observation"][state_keys[0]])
    actual_num_frames = 0
    for start_ix, end_ix in keep_ranges:
        actual_num_frames += end_ix - start_ix

    # Build episode data dictionary
    episode_dict = {
        "observation.state": state_dict["concatenated"].numpy(),
        "action": action_dict["concatenated"].numpy(),
        "next.reward": tf_to_torch(episode["reward"]).numpy(),
        "next.done": tf_to_torch(episode["is_last"]).numpy(),
        "is_terminal": tf_to_torch(episode["is_terminal"]).numpy(),
        "is_first": tf_to_torch(episode["is_first"]).numpy(),
        "discount": tf_to_torch(episode["discount"]).numpy(),
        "timestamp": np.arange(actual_num_frames) / fps,
        "episode_index": np.full(actual_num_frames, ep_idx),
        "frame_index": np.arange(actual_num_frames),
    }

    # Initialize all annotation columns with default value
    for lang_key in lang_keys:
        episode_dict[f"annotation.language.{lang_key}"] = np.full(
            num_frames, all_tasks["not provided"], dtype=np.int64
        )

    # Add language instruction indices to parquet
    episode_tasks = []
    for lang_key in lang_keys:
        if lang_key in episode:
            task = episode[lang_key][0].numpy().decode("utf-8")
            if task and len(task) > 1:
                episode_tasks.append(task)
                task_idx = all_tasks[task]
                episode_dict[f"annotation.language.{lang_key}"] = np.full(
                    num_frames, task_idx, dtype=np.int64
                )

    # Set task_index to match the first language instruction annotation
    episode_dict["task_index"] = episode_dict[f"annotation.language.{lang_keys[0]}"].copy()

    # Filter idle frames from episode_dict.
    for key in episode_dict:
        if key in ["timestamp", "episode_index", "frame_index"]:
            continue
        tensor_parts = []
        for start_ix, end_ix in keep_ranges:
            tensor_parts.append(episode_dict[key][start_ix:end_ix])
        episode_dict[key] = np.concatenate(tensor_parts, axis=0)

    # Filter idle frames from observation images.
    for img_key in image_keys:
        video_parts = []
        all_frames = np.stack(
            [tf_img_convert(episode["observation"][img_key][i]) for i in range(num_frames)]
        )
        for start_ix, end_ix in keep_ranges:
            video_parts.append(all_frames[start_ix:end_ix])
        new_video = np.concatenate(video_parts, axis=0)
        assert new_video.shape[0] == actual_num_frames
        episode["observation"][img_key] = new_video

    # Save to parquet using polars
    df = pl.DataFrame(episode_dict)
    parquet_path = output_path / f"data/chunk-{chunk_idx:03d}/episode_{ep_idx:06d}.parquet"
    df.write_parquet(parquet_path)

    # Process videos for each image key
    for img_key in image_keys:
        frames = episode["observation"][img_key]
        video_path = (
            output_path
            / f"videos/chunk-{chunk_idx:03d}/observation.images.{img_key}/episode_{ep_idx:06d}.mp4"
        )
        encode_video(frames, video_path, fps)

    episode_data = {
        "episode_index": ep_idx,
        "tasks": episode_tasks,
        "length": actual_num_frames,
        "success": bool(np.any(tf_to_torch(episode["reward"]).numpy() != 0)),
    }
    return episode_data


def convert_droid_dataset(
    raw_dir: str,
    output_dir: str,
    keep_ranges_path: str,
    fps: int = 15,
    first_n: int | None = None,
    max_workers: int = 16,
    filter_failed: bool = False,
):
    """
    Convert DROID 1.0.1 RLDS dataset to LeRobot format with idle filtering.

    Args:
        raw_dir: Path to raw DROID RLDS dataset (e.g., ./data/droid/1.0.1)
        output_dir: Path to output directory for LeRobot dataset
        keep_ranges_path: Path to JSON file containing idle filter ranges.
            Download from: gsutil cp gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json <path>
        fps: Frames per second for output videos
        first_n: Only process the first N tfrecord shards (for debugging)
        max_workers: Max workers for multiprocessing
        filter_failed: Whether to filter out failed episodes (all zero rewards)
    """
    output_path = Path(output_dir)

    # Validate keep_ranges_path exists
    if not os.path.exists(keep_ranges_path):
        raise FileNotFoundError(
            f"Keep ranges file not found: {keep_ranges_path}\n"
            "Download it with: gsutil cp gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json <path>"
        )

    # Load dataset
    config = tfds.ReadConfig(
        try_autocache=False,
        num_parallel_calls_for_decode=1,
        num_parallel_calls_for_interleave_files=1,
        interleave_cycle_length=1,
        shuffle_reshuffle_each_iteration=False,
    )
    ds_builder = tfds.builder_from_directory(str(raw_dir))
    split_str = f"train[:{first_n}shard]" if first_n is not None else "train"
    dataset = ds_builder.as_dataset(
        split=split_str,
        decoders={"steps": tfds.decode.SkipDecoding()},
        read_config=config,
    )
    dataset_info = ds_builder.info
    dataset = dataset.enumerate().map(_broadcast_metadata_rlds)

    # Extract keys
    image_keys = []
    state_keys = [
        "cartesian_position",
        "gripper_position",
        "joint_position",
    ]
    action_keys = [
        "cartesian_position",
        "cartesian_velocity",
        "gripper_position",
        "gripper_velocity",
        "joint_position",
        "joint_velocity",
    ]
    lang_keys = [
        "language_instruction",
        "language_instruction_2",
        "language_instruction_3",
    ]

    observation_info = dataset_info.features["steps"]["observation"]
    for key in observation_info:
        if len(observation_info[key].shape) == 3:
            if observation_info[key].dtype == tf.uint8:
                image_keys.append(key)
        else:
            assert key in state_keys, f"{key=}, {state_keys=}"

    print(f"Found image keys: {image_keys}")
    print(f"Using state keys: {state_keys}")
    print(f"Using action keys: {action_keys}")

    (output_path / "meta").mkdir(parents=True, exist_ok=True)

    # Get concat info for modality.json from first episode
    first_episode = next(iter(dataset))
    state_info = concat_state_or_action(
        first_episode["observation"], state_keys, compute_concat_info=True
    )
    action_info = concat_state_or_action(
        first_episode["action_dict"], action_keys, compute_concat_info=True
    )

    # Generate modality.json
    modality_config = {
        "state": state_info["concat_info"],
        "action": action_info["concat_info"],
        "video": {k: {"original_key": f"observation.images.{k}"} for k in image_keys},
        "annotation": {f"language.{lang_key}": {} for lang_key in lang_keys},
    }

    with open(output_path / "meta/modality.json", "w") as f:
        json.dump(modality_config, f, indent=4)

    # Get file instructions from TFDS
    ds_builder = tfds.builder_from_directory(str(raw_dir))
    file_instructions = ds_builder.info.splits["train"].file_instructions
    if first_n is not None:
        file_instructions = file_instructions[:first_n]

    # First pass: collect unique tasks and determine which episodes to keep
    all_tasks = {}  # task string -> task index
    task_counter = 0
    print(f"First pass: collecting unique tasks from {len(dataset)} episodes")
    # Add a default "not provided" task
    all_tasks["not provided"] = task_counter
    task_counter += 1

    # kept_registry maps global episode index -> filtered episode index
    kept_registry = {}
    kept_count = 0
    all_keep_ranges = json.load(open(keep_ranges_path, "r"))

    for i, episode in enumerate(tqdm.tqdm(dataset)):
        # filter out failed episodes
        filtered = False
        if filter_failed:
            if not np.any(tf_to_torch(episode["reward"]).numpy() != 0):
                filtered = True

        # Check language annotations
        has_lang = False
        for lang_key in lang_keys:
            if lang_key in episode:
                task = episode[lang_key][0].numpy().decode("utf-8")
                if task and (len(task) > 1) and task not in all_tasks:
                    has_lang = True
                    all_tasks[task] = task_counter
                    task_counter += 1
        if not has_lang:
            # Do not include episodes missing language annotations
            filtered = True

        # Filter out episodes that are only idle
        file_path = (
            episode["traj_metadata"]["episode_metadata"]["file_path"][0].numpy().decode("utf-8")
        )
        recording_folderpath = (
            episode["traj_metadata"]["episode_metadata"]["recording_folderpath"][0]
            .numpy()
            .decode("utf-8")
        )
        idle_key = f"{recording_folderpath}--{file_path}"
        keep_ranges = all_keep_ranges[idle_key]
        if len(keep_ranges) == 0:
            filtered = True

        if not filtered:
            kept_registry[i] = kept_count
            kept_count += 1

    print(f"Kept {len(kept_registry)}/{len(dataset)} episodes")

    # Write tasks.jsonl
    with open(output_path / "meta/tasks.jsonl", "w") as f:
        for task, task_idx in all_tasks.items():
            f.write(json.dumps({"task_index": task_idx, "task": task}) + "\n")

    if max_workers > 1:
        # Calculate process args with cumulative indices
        cumsum = 0
        process_args = []
        for i, instruction in enumerate(file_instructions):
            args = (
                i,
                raw_dir,
                output_path,
                fps,
                all_tasks,
                state_keys,
                action_keys,
                lang_keys,
                image_keys,
                cumsum,
                kept_registry,
                keep_ranges_path,
            )
            process_args.append(args)
            cumsum += instruction.examples_in_shard

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(mp_context=ctx, max_workers=max_workers) as executor:
            futures = [executor.submit(process_tfrecord, *args) for args in process_args]
            episodes_data = []
            for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
                episodes_data.extend(future.result())
    else:
        episodes_data = []
        cumsum = 0
        for i, instruction in enumerate(file_instructions):
            episodes_data.extend(
                process_tfrecord(
                    i,
                    raw_dir,
                    output_path,
                    fps,
                    all_tasks,
                    state_keys,
                    action_keys,
                    lang_keys,
                    image_keys,
                    cumsum,
                    kept_registry,
                    keep_ranges_path,
                )
            )
            cumsum += instruction.examples_in_shard

    # Order episodes by episode index
    episodes_data = sorted(episodes_data, key=lambda x: x["episode_index"])

    # Generate episodes.jsonl
    with open(output_path / "meta/episodes.jsonl", "w") as f:
        for episode in episodes_data:
            f.write(json.dumps(episode) + "\n")

    # Generate info.json
    ds_length = len(episodes_data)
    num_chunks = (ds_length // 1000) + (1 if ds_length % 1000 else 0)
    info = {
        "codebase_version": "v2.0",
        "robot_type": "droid",
        "total_episodes": ds_length,
        "total_frames": sum(ep["length"] for ep in episodes_data),
        "total_tasks": len(all_tasks),
        "total_videos": len(image_keys),
        "total_chunks": num_chunks,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": "0:100"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            # Video features
            **{
                f"observation.images.{k}": {
                    "dtype": "video",
                    "shape": list(tf_img_convert(first_episode["observation"][k][0]).shape),
                    "names": ["height", "width", "channel"],
                    "video_info": {
                        "video.fps": fps,
                        "video.codec": "h264",
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": False,
                        "has_audio": False,
                    },
                }
                for k in image_keys
            },
            # State feature
            "observation.state": {
                "dtype": "float64",
                "shape": [state_info["concatenated"].shape[1]],
                "names": state_keys,
            },
            # Action feature
            "action": {
                "dtype": "float64",
                "shape": [action_info["concatenated"].shape[1]],
                "names": action_keys,
            },
            # Single value features
            "timestamp": {"dtype": "float64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "next.reward": {"dtype": "float64", "shape": [1]},
            "next.done": {"dtype": "bool", "shape": [1]},
            "is_terminal": {"dtype": "bool", "shape": [1]},
            "is_first": {"dtype": "bool", "shape": [1]},
            "discount": {"dtype": "float64", "shape": [1]},
            # Language annotation features
            **{f"annotation.language.{k}": {"dtype": "int64", "shape": [1]} for k in lang_keys},
        },
    }

    with open(output_path / "meta/info.json", "w") as f:
        json.dump(info, f, indent=4)

    # Sanity check: chunk directories should contain exactly 1000 episodes (except last)
    for i in range(num_chunks):
        chunk_path = output_path / f"data/chunk-{i:03d}"
        episodes = list(chunk_path.glob("episode_*.parquet"))
        assert (
            len(episodes) == 1000 if i != num_chunks - 1 else len(episodes) <= 1000
        ), f"chunk-{i:03d} contains {len(episodes)} episodes"

        for img_key in image_keys:
            img_path = output_path / f"videos/chunk-{i:03d}/observation.images.{img_key}"
            episodes = list(img_path.glob("episode_*.mp4"))
            assert (
                len(episodes) == 1000 if i != num_chunks - 1 else len(episodes) <= 1000
            ), f"{img_path} contains {len(episodes)} episodes"

    print("Sanity check passed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert DROID 1.0.1 (RLDS) to LeRobot format with idle filtering."
    )
    parser.add_argument("raw_dir", help="Path to raw DROID RLDS dataset (e.g., ./data/droid/1.0.1)")
    parser.add_argument("output_dir", help="Path to output directory for LeRobot dataset")
    parser.add_argument(
        "--keep-ranges-path",
        required=True,
        help="Path to idle filter JSON file. Download with: "
        "gsutil cp gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json <path>",
    )
    parser.add_argument("--fps", type=int, default=15, help="Frames per second for videos")
    parser.add_argument(
        "--first-n", type=int, help="Only convert first N tfrecord shards (for debugging)"
    )
    parser.add_argument("-n", type=int, default=16, help="Max workers for multiprocessing")
    parser.add_argument(
        "--filter-failed",
        action="store_true",
        help="Whether to filter out failed episodes (i.e., episodes with all zero rewards)",
    )
    args = parser.parse_args()

    convert_droid_dataset(
        args.raw_dir,
        args.output_dir,
        args.keep_ranges_path,
        args.fps,
        args.first_n,
        args.n,
        args.filter_failed,
    )