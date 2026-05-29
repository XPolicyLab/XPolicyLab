import argparse
from copy import deepcopy
from functools import partial
import gc
import json
import logging
from math import ceil
import os
from pathlib import Path
import shutil
from typing import Callable

import einops
import h5py
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import (
    STATS_PATH,
    check_timestamps_sync,
    get_episode_data_index,
    serialize_dict,
    write_json,
)
import numpy as np
import torch
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


def generate_modality_json(output_dir: str) -> None:
    """Generate modality.json file defining field mappings for the dataset."""
    modality_config = {
        "state": {
            "left_arm_joint_position": {
                "original_key": "observation.state",
                "start": 0,
                "end": 7,
                "rotation_type": None,
                "absolute": True,
                "dtype": "float64",
                "range": None
            },
            "right_arm_joint_position": {
                "original_key": "observation.state",
                "start": 7,
                "end": 14,
                "rotation_type": None,
                "absolute": True,
                "dtype": "float64",
                "range": None
            },
            "left_effector_position": {
                "original_key": "observation.state",
                "start": 14,
                "end": 15,
                "rotation_type": None,
                "absolute": True,
                "dtype": "float64",
                "range": None
            },
            "right_effector_position": {
                "original_key": "observation.state",
                "start": 15,
                "end": 16,
                "rotation_type": None,
                "absolute": True,
                "dtype": "float64",
                "range": None
            },
            "head_position": {
                "original_key": "observation.state",
                "start": 16,
                "end": 18,
                "rotation_type": None,
                "absolute": True,
                "dtype": "float64",
                "range": None
            },
            "waist_pitch": {
                "original_key": "observation.state",
                "start": 18,
                "end": 19,
                "rotation_type": None,
                "absolute": True,
                "dtype": "float64",
                "range": None
            },
            "waist_lift": {
                "original_key": "observation.state",
                "start": 19,
                "end": 20,
                "rotation_type": None,
                "absolute": True,
                "dtype": "float64",
                "range": None
            },
        },
        "action": {
            "left_arm_joint_position": {
                "original_key": "action",
                "start": 0,
                "end": 7,
                "rotation_type": None,
                "absolute": True,
                "dtype": "float64",
                "range": None
            },
            "right_arm_joint_position": {
                "original_key": "action",
                "start": 7,
                "end": 14,
                "rotation_type": None,
                "absolute": True,
                "dtype": "float64",
                "range": None
            },
            "left_effector_position": {
                "original_key": "action",
                "start": 14,
                "end": 15,
                "rotation_type": None,
                "absolute": True,
                "dtype": "float64",
                "range": None
            },
            "right_effector_position": {
                "original_key": "action",
                "start": 15,
                "end": 16,
                "rotation_type": None,
                "absolute": True,
                "dtype": "float64",
                "range": None
            },
            "head_position": {
                "original_key": "action",
                "start": 16,
                "end": 18,
                "rotation_type": None,
                "absolute": True,
                "dtype": "float64",
                "range": None
            },
            "waist_pitch": {
                "original_key": "action",
                "start": 18,
                "end": 19,
                "rotation_type": None,
                "absolute": True,
                "dtype": "float64",
                "range": None
            },
            "waist_lift": {
                "original_key": "action",
                "start": 19,
                "end": 20,
                "rotation_type": None,
                "absolute": True,
                "dtype": "float64",
                "range": None
            },
            "robot_velocity": {
                "original_key": "action",
                "start": 20,
                "end": 22,
                "rotation_type": None,
                "absolute": True,
                "dtype": "float64",
                "range": None
            },
        },
        "video": {
            "top_head": {
                "original_key": "observation.images.top_head"
            },
            "hand_left": {
                "original_key": "observation.images.hand_left"
            },
            "hand_right": {
                "original_key": "observation.images.hand_right"
            },
        },
        "annotation": {
            "language.action_text": {
                "original_key": "task_index"
            },
            "agibot.sub_task": {
                "original_key": "annotation.agibot.sub_task"
            },
            "frame_type": {
                "original_key": "annotation.frame_type"
            },
        },
    }

    modality_path = os.path.join(output_dir, "modality.json")
    with open(modality_path, "w") as f:
        json.dump(modality_config, f, indent=4)
    print(f"Generated modality.json at {modality_path}")


HEAD_COLOR = "head_color.mp4"
HAND_LEFT_COLOR = "hand_left_color.mp4"
HAND_RIGHT_COLOR = "hand_right_color.mp4"

FEATURES = {
    "observation.images.top_head": {
        "dtype": "video",
        "shape": [480, 640, 3],
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
        "shape": [480, 640, 3],
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
        "shape": [480, 640, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 30.0,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.state": {
        "dtype": "float32",
        "shape": [20],
    },
    "action": {
        "dtype": "float32",
        "shape": [22],
    },
    "annotation.language.action_text": {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    },
    "annotation.agibot.tasks": {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    },
    "episode_index": {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    },
    "frame_index": {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    },
    "index": {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    },
    "task_index": {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    },
}


def get_stats_einops_patterns(dataset, num_workers=0):
    """These einops patterns will be used to aggregate batches and compute statistics.

    Note: We assume the images are in channel first format
    """

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=2,
        shuffle=False,
    )
    batch = next(iter(dataloader))

    stats_patterns = {}

    for key in dataset.features:
        # sanity check that tensors are not float64
        assert batch[key].dtype != torch.float64

        # if isinstance(feats_type, (VideoFrame, Image)):
        if key in dataset.meta.camera_keys:
            # sanity check that images are channel first
            _, c, h, w = batch[key].shape
            assert c < h and c < w, f"expect channel first images, but instead {batch[key].shape}"
            assert (
                batch[key].dtype == torch.float32
            ), f"expect torch.float32, but instead {batch[key].dtype=}"
            # assert batch[key].max() <= 1, f"expect pixels lower than 1, but instead {batch[key].max()=}"
            # assert batch[key].min() >= 0, f"expect pixels greater than 1, but instead {batch[key].min()=}"
            stats_patterns[key] = "b c h w -> c 1 1"
        elif batch[key].ndim == 2:
            stats_patterns[key] = "b c -> c "
        elif batch[key].ndim == 1:
            stats_patterns[key] = "b -> 1"
        else:
            raise ValueError(f"{key}, {batch[key].shape}")

    return stats_patterns


def compute_stats(dataset, batch_size=8, num_workers=4, max_num_samples=None):
    """Compute mean/std and min/max statistics of all data keys in a LeRobotDataset."""
    if max_num_samples is None:
        max_num_samples = len(dataset)

    # for more info on why we need to set the same number of workers, see `load_from_videos`
    stats_patterns = get_stats_einops_patterns(dataset, num_workers)

    # mean and std will be computed incrementally while max and min will track the running value.
    mean, std, max, min = {}, {}, {}, {}
    for key in stats_patterns:
        mean[key] = torch.tensor(0.0).float()
        std[key] = torch.tensor(0.0).float()
        max[key] = torch.tensor(-float("inf")).float()
        min[key] = torch.tensor(float("inf")).float()

    def create_seeded_dataloader(dataset, batch_size, seed):
        generator = torch.Generator()
        generator.manual_seed(seed)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            generator=generator,
        )
        return dataloader

    # Note: Due to be refactored soon. The point of storing `first_batch` is to make sure we don't get
    # surprises when rerunning the sampler.
    first_batch = None
    running_item_count = 0  # for online mean computation
    dataloader = create_seeded_dataloader(dataset, batch_size, seed=1337)
    for i, batch in enumerate(
        tqdm(
            dataloader,
            total=ceil(max_num_samples / batch_size),
            desc="Compute mean, min, max",
        )
    ):
        this_batch_size = len(batch["index"])
        running_item_count += this_batch_size
        if first_batch is None:
            first_batch = deepcopy(batch)
        for key, pattern in stats_patterns.items():
            batch[key] = batch[key].float()
            # Numerically stable update step for mean computation.
            batch_mean = einops.reduce(batch[key], pattern, "mean")
            # Hint: to update the mean we need x̄ₙ = (Nₙ₋₁x̄ₙ₋₁ + Bₙxₙ) / Nₙ, where the subscript represents
            # the update step, N is the running item count, B is this batch size, x̄ is the running mean,
            # and x is the current batch mean. Some rearrangement is then required to avoid risking
            # numerical overflow. Another hint: Nₙ₋₁ = Nₙ - Bₙ. Rearrangement yields
            # x̄ₙ = x̄ₙ₋₁ + Bₙ * (xₙ - x̄ₙ₋₁) / Nₙ
            mean[key] = mean[key] + this_batch_size * (batch_mean - mean[key]) / running_item_count
            max[key] = torch.maximum(max[key], einops.reduce(batch[key], pattern, "max"))
            min[key] = torch.minimum(min[key], einops.reduce(batch[key], pattern, "min"))

        if i == ceil(max_num_samples / batch_size) - 1:
            break

    first_batch_ = None
    running_item_count = 0  # for online std computation
    dataloader = create_seeded_dataloader(dataset, batch_size, seed=1337)
    for i, batch in enumerate(
        tqdm(dataloader, total=ceil(max_num_samples / batch_size), desc="Compute std")
    ):
        this_batch_size = len(batch["index"])
        running_item_count += this_batch_size
        # Sanity check to make sure the batches are still in the same order as before.
        if first_batch_ is None:
            first_batch_ = deepcopy(batch)
            for key in stats_patterns:
                assert torch.equal(first_batch_[key], first_batch[key])
        for key, pattern in stats_patterns.items():
            batch[key] = batch[key].float()
            # Numerically stable update step for mean computation (where the mean is over squared
            # residuals).See notes in the mean computation loop above.
            batch_std = einops.reduce((batch[key] - mean[key]) ** 2, pattern, "mean")
            std[key] = std[key] + this_batch_size * (batch_std - std[key]) / running_item_count

        if i == ceil(max_num_samples / batch_size) - 1:
            break

    for key in stats_patterns:
        std[key] = torch.sqrt(std[key])

    stats = {}
    for key in stats_patterns:
        stats[key] = {
            "mean": mean[key],
            "std": std[key],
            "max": max[key],
            "min": min[key],
        }
    return stats


class AgiBotDataset(LeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        download_videos: bool = True,
        local_files_only: bool = False,
        video_backend: str | None = None,
    ):
        super().__init__(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            download_videos=download_videos,
            local_files_only=local_files_only,
            video_backend=video_backend,
        )

    def save_episode(
        self, task: str, episode_data: dict | None = None, videos: dict | None = None
    ) -> None:
        """
        We rewrite this method to copy mp4 videos to the target position
        """
        if not episode_data:
            episode_buffer = self.episode_buffer

        episode_length = episode_buffer.pop("size")
        episode_index = episode_buffer["episode_index"]
        if episode_index != self.meta.total_episodes:
            # TODO(aliberts): Add option to use existing episode_index
            raise NotImplementedError(
                "You might have manually provided the episode_buffer with an episode_index that doesn't "
                "match the total number of episodes in the dataset. This is not supported for now."
            )

        if episode_length == 0:
            raise ValueError(
                "You must add one or several frames with `add_frame` before calling `add_episode`."
            )

        # Use our custom task indexing instead of LeRobot's built-in mechanism
        task_index = getattr(self, "_custom_task_to_index", {}).get(task, 0)

        # Remove the 'task' key if it exists (it's passed as a parameter, not needed in buffer)
        episode_buffer.pop("task", None)

        if not set(episode_buffer.keys()) == set(self.features):
            raise ValueError()

        for key, ft in self.features.items():
            if key == "index":
                episode_buffer[key] = np.arange(
                    self.meta.total_frames, self.meta.total_frames + episode_length
                )
            elif key == "episode_index":
                episode_buffer[key] = np.full((episode_length,), episode_index)
            elif key == "task_index":
                episode_buffer[key] = np.full((episode_length,), task_index)
            elif ft["dtype"] in ["image", "video"]:
                continue
            elif ft["dtype"] == "string":
                pass
            elif len(ft["shape"]) == 1 and ft["shape"][0] == 1:
                episode_buffer[key] = np.array(episode_buffer[key], dtype=ft["dtype"])
            elif len(ft["shape"]) == 1 and ft["shape"][0] > 1:
                episode_buffer[key] = np.stack(episode_buffer[key])
            else:
                raise ValueError(key)

        self._wait_image_writer()
        self._save_episode_table(episode_buffer, episode_index)

        # Copy videos first before calling meta.save_episode which might try to read them
        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            episode_buffer[key] = video_path
            video_path.parent.mkdir(parents=True, exist_ok=True)
            # Copy video files to target location
            shutil.copyfile(str(videos[key]), str(video_path))

        try:
            # FIX: Call meta.save_episode with correct parameters
            # Note: We pass an empty task list to prevent duplicate entries in tasks.jsonl
            # since we create our own tasks.jsonl file with proper indexing
            self.meta.save_episode(episode_index, episode_length, [], {})
        except AttributeError as e:
            if "'NoneType' object has no attribute 'items'" in str(e):
                # Handle the episode stats issue - skip episode stats for now
                print(
                    f"Warning: Episode stats computation failed, proceeding without stats "
                    f"for episode {episode_index}"
                )
                # Just skip the problematic save_episode call - the core data is already saved
                pass
            else:
                raise
        if not episode_data:  # Reset the buffer
            self.episode_buffer = self.create_episode_buffer()
        self.consolidated = False

    def consolidate(self, run_compute_stats: bool = True, keep_image_files: bool = False) -> None:
        self.hf_dataset = self.load_hf_dataset()
        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)
        check_timestamps_sync(self.hf_dataset, self.episode_data_index, self.fps, self.tolerance_s)
        if len(self.meta.video_keys) > 0:
            self.meta.write_video_info()

        if not keep_image_files:
            img_dir = self.root / "images"
            if img_dir.is_dir():
                shutil.rmtree(self.root / "images")
        video_files = list(self.root.rglob("*.mp4"))
        assert len(video_files) == self.num_episodes * len(self.meta.video_keys)

        parquet_files = list(self.root.rglob("*.parquet"))
        assert len(parquet_files) == self.num_episodes

        if run_compute_stats:
            self.stop_image_writer()
            self.meta.stats = compute_stats(self)
            serialized_stats = serialize_dict(self.meta.stats)
            write_json(serialized_stats, self.root / STATS_PATH)
            self.consolidated = True
        else:
            logging.warning(
                "Skipping computation of the dataset statistics, dataset is not fully consolidated."
            )

    def add_frame(self, frame: dict) -> None:
        """
        This function only adds the frame to the episode_buffer. Apart from images — which are written in a
        temporary directory — nothing is written to disk. To save those frames, the 'save_episode()' method
        then needs to be called.
        """
        # TODO(aliberts, rcadene): Add sanity check for the input, check it's numpy or torch,
        # check the dtype and shape matches, etc.

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        frame_index = self.episode_buffer["size"]
        timestamp = frame.pop("timestamp") if "timestamp" in frame else frame_index / self.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)

        for key in frame:
            if key not in self.features:
                raise ValueError(key)
            item = frame[key].numpy() if isinstance(frame[key], torch.Tensor) else frame[key]
            self.episode_buffer[key].append(item)

        self.episode_buffer["size"] += 1


def detect_dataset_format(src_path: str) -> str:
    """Detect whether the dataset follows old or new format structure"""
    src_path = Path(src_path)

    # Check for old format indicators
    if (src_path / "task_info").exists() and (src_path / "proprio_stats").exists():
        return "old"

    # Check for new format indicators
    # Look for pattern: job_id/robot_id/episode_id with aligned_joints.h5
    # The test_data/2810125 directory contains job_id directories
    subdirs = [d for d in src_path.iterdir() if d.is_dir()]
    if subdirs:
        # Check if we have nested structure with aligned_joints.h5
        for job_dir in subdirs:  # These are job_id directories like 3335477
            if not job_dir.is_dir():
                continue
            for (
                robot_dir
            ) in job_dir.iterdir():  # These are robot_id directories like A2D0015AB00061
                if not robot_dir.is_dir():
                    continue
                for (
                    episode_dir
                ) in robot_dir.iterdir():  # These are episode_id directories like 12052353
                    if episode_dir.is_dir() and (episode_dir / "aligned_joints.h5").exists():
                        return "new"

    return "unknown"


def load_local_dataset_old_format(episode_id: int, src_path: str, task_id: int) -> list | None:
    """Load local dataset from old format and return a dict with observations and actions"""

    # --- Load task info for this specific call ---
    task_json_path = Path(src_path) / f"task_info/task_{task_id}.json"
    task_info_list = None
    if task_json_path.exists():
        try:
            with open(task_json_path, "r") as f:
                task_info_list = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Failed to decode JSON {task_json_path} for episode {episode_id}")
            task_info_list = []  # Treat as empty if decode fails
    else:
        print(f"Warning: Task info JSON not found at {task_json_path} for episode {episode_id}")
        task_info_list = []  # Treat as empty if not found

    # --- Find action_config for this episode_id ---
    episode_action_config = None
    if isinstance(task_info_list, list):  # Check if loading succeeded and it's a list
        for item in task_info_list:
            # Ensure episode_id exists and compare as int
            if "episode_id" in item and int(item["episode_id"]) == episode_id:
                episode_action_config = item.get("label_info", {}).get("action_config")
                break

    default_action_text = "N/A"

    ob_dir = Path(src_path) / f"observations/{task_id}/{episode_id}"
    proprio_dir = Path(src_path) / f"proprio_stats/{task_id}/{episode_id}"

    with h5py.File(proprio_dir / "proprio_stats.h5") as f:
        state_joint = np.array(f["state/joint/position"])
        state_effector = np.clip((np.array(f["state/effector/position"]) - 35.0) / (120.0 - 35.0), 0.0, 1.0)
        state_head = np.array(f["state/head/position"])
        state_waist = np.array(f["state/waist/position"])
        action_joint = np.array(f["action/joint/position"])
        action_effector = np.clip((np.array(f["action/effector/position"]) - 35.0) / (120.0 - 35.0), 0.0, 1.0)
        action_head = np.array(f["action/head/position"])
        action_waist = np.array(f["action/waist/position"])
        action_velocity = np.array(f["action/robot/velocity"])

    # State (20 DOF): joint(14) + effector(2) + head(2) + waist(2)
    states_value = np.hstack(
        [
            state_joint,
            state_effector,
            state_head,
            state_waist,
        ]
    ).astype(np.float32)
    assert (
        action_joint.shape[0] == action_effector.shape[0]
    ), f"shape of action_joint:{action_joint.shape};shape of action_effector:{action_effector.shape}"
    # Action (22 DOF): joint(14) + effector(2) + head(2) + waist(2) + velocity(2)
    action_value = np.hstack(
        [
            action_joint,
            action_effector,
            action_head,
            action_waist,
            action_velocity,
        ]
    ).astype(np.float32)

    num_frames = len(states_value)

    # --- Create frame -> action_text mapping ---
    frame_action_texts = [default_action_text] * num_frames
    if episode_action_config:  # Only proceed if config was found
        for action in episode_action_config:
            start = action.get("start_frame")
            end = action.get("end_frame")
            text = action.get("action_text", default_action_text)

            if start is None or end is None:
                continue

            clamped_start = max(0, start)
            clamped_end = min(num_frames, end)
            for i in range(clamped_start, clamped_end):
                frame_action_texts[i] = text

    frames = [
        {
            "observation.state": states_value[i],
            "action": action_value[i],
            "annotation.language.action_text": [frame_action_texts[i]],  # Add action_text here
        }
        for i in range(num_frames)
    ]

    v_path = ob_dir / "videos"
    videos = {
        "observation.images.top_head": v_path / HEAD_COLOR,
        "observation.images.hand_left": v_path / HAND_LEFT_COLOR,
        "observation.images.hand_right": v_path / HAND_RIGHT_COLOR,
    }
    return frames, videos


def load_local_dataset_new_format(episode_path: str) -> list | None:
    """Load local dataset from new format and return a dict with observations and actions"""

    episode_dir = Path(episode_path)

    # Load data info JSON
    data_info_path = episode_dir / "data_info.json"
    episode_action_config = None
    default_action_text = "N/A"

    if data_info_path.exists():
        try:
            with open(data_info_path, "r") as f:
                data_info = json.load(f)
                episode_action_config = data_info.get("label_info", {}).get("action_config")
        except json.JSONDecodeError:
            print(f"Warning: Failed to decode JSON {data_info_path}")

    # Load aligned joints data
    joints_path = episode_dir / "aligned_joints.h5"
    if not joints_path.exists():
        print(f"Warning: aligned_joints.h5 not found at {joints_path}")
        return None

    with h5py.File(joints_path) as f:
        # Extract state data - using same mapping as old format for compatibility
        state_joint = np.array(f["state/joint/position"])
        state_head = np.array(f["state/head/position"])
        state_waist = np.array(f["state/waist/position"])

        # For new format, use separate left/right effector position data
        # Normalize from raw [35, 120] to [0, 1]
        state_left_effector = np.array(f["state/left_effector/position"])  # Shape: (N, 1)
        state_right_effector = np.array(f["state/right_effector/position"])  # Shape: (N, 1)
        state_effector = np.clip(
            np.column_stack([state_left_effector.flatten(), state_right_effector.flatten()]) - 35.0,
            0.0, 85.0
        ) / 85.0  # Shape: (N, 2), range [0, 1]

        # Extract action data
        action_joint = np.array(f["action/joint/position"])
        action_head = np.array(f["action/head/position"])
        action_waist = np.array(f["action/waist/position"])

        # For new format, use separate left/right effector position data for actions
        # Normalize from raw [35, 120] to [0, 1]
        action_left_effector = np.array(f["action/left_effector/position"])  # Shape: (N, 1)
        action_right_effector = np.array(f["action/right_effector/position"])  # Shape: (N, 1)
        action_effector = np.clip(
            np.column_stack([action_left_effector.flatten(), action_right_effector.flatten()]) - 35.0,
            0.0, 85.0
        ) / 85.0  # Shape: (N, 2), range [0, 1]

        # Get robot velocity (N,) or (N, 2)
        action_velocity_raw = np.array(f["action/robot/velocity"])
        if action_velocity_raw.ndim == 1:
            # Scalar velocity: pad second component with zeros (linear_x only)
            action_velocity = np.column_stack(
                [action_velocity_raw, np.zeros_like(action_velocity_raw)]
            )
        else:
            action_velocity = action_velocity_raw[:, :2]

    # State (20 DOF): joint(14) + effector(2) + head(2) + waist(2)
    states_value = np.hstack(
        [
            state_joint,
            state_effector,
            state_head,
            state_waist,
        ]
    ).astype(np.float32)
    # Action (22 DOF): joint(14) + effector(2) + head(2) + waist(2) + velocity(2)
    action_value = np.hstack(
        [
            action_joint,
            action_effector,
            action_head,
            action_waist,
            action_velocity,
        ]
    ).astype(np.float32)

    num_frames = len(states_value)

    # --- Create frame -> action_text mapping ---
    frame_action_texts = [default_action_text] * num_frames
    if episode_action_config:  # Only proceed if config was found
        for action in episode_action_config:
            start = action.get("start_frame")
            end = action.get("end_frame")
            # Use English action text if available, otherwise use Chinese
            text = action.get("english_action_text") or action.get(
                "action_text", default_action_text
            )

            if start is None or end is None:
                continue

            clamped_start = max(0, start)
            clamped_end = min(num_frames, end)
            for i in range(clamped_start, clamped_end):
                frame_action_texts[i] = text

    frames = [
        {
            "observation.state": states_value[i],
            "action": action_value[i],
            "annotation.language.action_text": [frame_action_texts[i]],
        }
        for i in range(num_frames)
    ]

    # Videos are at episode level in new format
    videos = {
        "observation.images.top_head": episode_dir / HEAD_COLOR,
        "observation.images.hand_left": episode_dir / HAND_LEFT_COLOR,
        "observation.images.hand_right": episode_dir / HAND_RIGHT_COLOR,
    }
    return frames, videos


def load_local_dataset(
    episode_id: int,
    src_path: str,
    task_id: int = None,
    episode_path: str = None,
    format_type: str = "old",
) -> list | None:
    """Load local dataset and return a dict with observations and actions

    Args:
        episode_id: Episode ID (used for old format)
        src_path: Source path (used for old format)
        task_id: Task ID (used for old format)
        episode_path: Full path to episode directory (used for new format)
        format_type: "old" or "new" format
    """
    if format_type == "old":
        return load_local_dataset_old_format(episode_id, src_path, task_id)
    elif format_type == "new":
        return load_local_dataset_new_format(episode_path)
    else:
        raise ValueError(f"Unknown format type: {format_type}")


def get_task_instruction_old_format(task_json_path: str) -> str:
    """Get task language instruction from old format"""
    with open(task_json_path, "r") as f:
        task_info = json.load(f)
    task_name = task_info[0]["task_name"]
    task_init_scene = task_info[0]["init_scene_text"]
    task_instruction = f"{task_name}.{task_init_scene}"
    print(f"Get Task Instruction <{task_instruction}>")
    return task_instruction


def get_task_instruction_new_format(episode_paths: list) -> str:
    """Get task language instruction from new format - use first episode's data_info.json"""
    if not episode_paths:
        return "Unknown Task"

    first_episode_path = Path(episode_paths[0])
    data_info_path = first_episode_path / "data_info.json"

    if data_info_path.exists():
        try:
            with open(data_info_path, "r") as f:
                data_info = json.load(f)
            # Use English task name if available, otherwise use Chinese
            task_name = data_info.get("english_task_name") or data_info.get(
                "task_name", "Unknown Task"
            )
            task_instruction = task_name
            print(
                f"Get Task Instruction <{task_instruction}> "
                f"(english_task_name: {data_info.get('english_task_name')}, "
                f"task_name: {data_info.get('task_name')})"
            )
            return task_instruction
        except json.JSONDecodeError:
            print(f"Warning: Failed to decode JSON {data_info_path}")

    return "Unknown Task"


def get_task_instruction(
    task_json_path: str = None, episode_paths: list = None, format_type: str = "old"
) -> str:
    """Get task language instruction"""
    if format_type == "old":
        return get_task_instruction_old_format(task_json_path)
    elif format_type == "new":
        return get_task_instruction_new_format(episode_paths)
    else:
        raise ValueError(f"Unknown format type: {format_type}")


def load_new_format_episode(episode_path):
    """Helper function for multiprocessing - load new format episode"""
    return load_local_dataset(
        episode_id=0, src_path="", episode_path=episode_path, format_type="new"
    )


def create_tasks_jsonl(tgt_path: str, repo_id: str, task_name: str, all_action_texts: set) -> dict:
    """Create tasks.jsonl file with unique task names and action texts."""
    meta_path = os.path.join(tgt_path, repo_id, "meta")
    os.makedirs(meta_path, exist_ok=True)

    tasks_jsonl_path = os.path.join(meta_path, "tasks.jsonl")

    # Create a list of unique tasks combining task name and action texts
    # Remove task_name from action texts if it exists to avoid duplicates
    unique_action_texts = all_action_texts - {task_name}

    tasks = [task_name]  # First entry is the main task
    tasks.extend(sorted(unique_action_texts))  # Then all unique action texts (excluding task name)

    # Check if file already exists and has correct content
    should_write = True
    if os.path.exists(tasks_jsonl_path):
        try:
            with open(tasks_jsonl_path, "r") as f:
                existing_content = f.read().strip()

            # Generate expected content
            expected_lines = []
            for i, task in enumerate(tasks):
                task_entry = {"task_index": i, "task": task}
                expected_lines.append(json.dumps(task_entry))
            expected_content = "\n".join(expected_lines)

            # If content matches, don't rewrite
            if existing_content == expected_content:
                should_write = False
        except Exception as e:
            print(f"Warning: Failed to read tasks.jsonl: {e}")
            # If there's any issue reading, we'll rewrite
            pass

    if should_write:
        # Write tasks.jsonl (overwrite to ensure clean content)
        with open(tasks_jsonl_path, "w") as f:
            for i, task in enumerate(tasks):
                task_entry = {"task_index": i, "task": task}
                f.write(json.dumps(task_entry) + "\n")

        print(f"Created tasks.jsonl with {len(tasks)} entries at {tasks_jsonl_path}")
    else:
        print(f"tasks.jsonl already exists with correct content at {tasks_jsonl_path}")

    # Create mapping for lookups
    task_to_index = {task: i for i, task in enumerate(tasks)}

    return task_to_index


def main(
    src_path: str,
    tgt_path: str,
    task_id: int = None,
    repo_id: str = None,
    task_info_json: str = None,
    debug: bool = False,
    chunk_size: int = 10,
):
    # Detect dataset format
    format_type = detect_dataset_format(src_path)
    print(f"Detected dataset format: {format_type}")

    if format_type == "unknown":
        raise ValueError(f"Unable to detect dataset format for path: {src_path}")

    # Collect all unique action texts first
    all_action_texts = set()

    # Initialize dataset
    if not repo_id:
        if format_type == "old":
            repo_id = f"agibotworld/task_{task_id}"
        else:  # new format
            # Use the top-level directory name as task_id
            task_id = Path(src_path).name
            repo_id = f"agibotworld/task_{task_id}"

    dataset = AgiBotDataset.create(
        repo_id=repo_id,
        root=f"{tgt_path}/{repo_id}",
        fps=30,
        robot_type="a2d",
        features=FEATURES,
    )

    if format_type == "old":
        # Old format processing
        task_name = get_task_instruction(task_json_path=task_info_json, format_type="old")

        all_subdir = sorted(
            [f.as_posix() for f in Path(src_path).glob(f"observations/{task_id}/*") if f.is_dir()]
        )

        if debug:
            all_subdir = all_subdir[:2]

        # Get all episode id
        all_subdir_eids = [int(Path(path).name) for path in all_subdir]
        all_subdir_episode_desc = [task_name] * len(all_subdir_eids)

        # First pass: collect all unique action texts
        print("Collecting unique action texts...")
        for episode_id in tqdm(all_subdir_eids, desc="Scanning for action texts"):
            frames_data, _ = load_local_dataset(
                episode_id, src_path=src_path, task_id=task_id, format_type="old"
            )
            if frames_data:
                for frame in frames_data:
                    action_text = frame["annotation.language.action_text"][0]
                    all_action_texts.add(action_text)

        # Create tasks.jsonl with all unique texts
        task_to_index = create_tasks_jsonl(tgt_path, repo_id, task_name, all_action_texts)

        # Pass the task mapping to the dataset for custom indexing
        dataset._custom_task_to_index = task_to_index

        # Process in chunks to reduce memory usage
        for chunk_start in tqdm(
            range(0, len(all_subdir_eids), chunk_size), desc="Processing chunks"
        ):
            chunk_end = min(chunk_start + chunk_size, len(all_subdir_eids))
            chunk_eids = all_subdir_eids[chunk_start:chunk_end]
            chunk_descs = all_subdir_episode_desc[chunk_start:chunk_end]

            # Process only this chunk
            if debug:
                raw_datasets_chunk = [
                    load_local_dataset(
                        subdir, src_path=src_path, task_id=task_id, format_type="old"
                    )
                    for subdir in tqdm(chunk_eids, desc="Loading chunk data")
                ]
            else:
                raw_datasets_chunk = process_map(
                    partial(
                        load_local_dataset, src_path=src_path, task_id=task_id, format_type="old"
                    ),
                    chunk_eids,
                    max_workers=os.cpu_count() // 2,
                    desc=f"Loading chunk {chunk_start//chunk_size + 1}/"
                    f"{(len(all_subdir_eids) + chunk_size - 1)//chunk_size}",
                )

            # Filter out None results
            valid_datasets = [
                (ds, desc) for ds, desc in zip(raw_datasets_chunk, chunk_descs) if ds is not None
            ]

            # Process each dataset in the chunk
            for raw_dataset, episode_desc in tqdm(
                valid_datasets, desc="Processing episodes in chunk"
            ):
                for raw_dataset_sub in tqdm(raw_dataset[0], desc="Processing frames", leave=False):
                    # Convert string annotation to int index
                    action_text = raw_dataset_sub["annotation.language.action_text"][0]
                    raw_dataset_sub["annotation.language.action_text"] = [
                        task_to_index[action_text]
                    ]
                    raw_dataset_sub["annotation.agibot.tasks"] = [task_to_index[episode_desc]]
                    dataset.add_frame(raw_dataset_sub)
                dataset.save_episode(task=episode_desc, videos=raw_dataset[1])

            # Clear memory after each chunk
            raw_datasets_chunk = None
            valid_datasets = None
            gc.collect()

    else:  # new format
        # Find all episode directories
        all_episode_paths = []
        src_path = Path(src_path)

        # Walk through job_id/robot_id/episode_id structure (test_data/2810125 contains job_id directories)
        for job_dir in src_path.iterdir():  # These are job_id directories like 3335477
            if not job_dir.is_dir():
                continue
            for (
                robot_dir
            ) in job_dir.iterdir():  # These are robot_id directories like A2D0015AB00061
                if not robot_dir.is_dir():
                    continue
                for (
                    episode_dir
                ) in robot_dir.iterdir():  # These are episode_id directories like 12052353
                    if episode_dir.is_dir() and (episode_dir / "aligned_joints.h5").exists():
                        all_episode_paths.append(str(episode_dir))

        all_episode_paths = sorted(all_episode_paths)

        if debug:
            all_episode_paths = all_episode_paths[:2]

        # Get task name from first episode
        task_name = get_task_instruction(episode_paths=all_episode_paths, format_type="new")
        all_episode_descs = [task_name] * len(all_episode_paths)

        # First pass: collect all unique action texts
        print("Collecting unique action texts...")
        for episode_path in tqdm(all_episode_paths, desc="Scanning for action texts"):
            frames_data, _ = load_local_dataset(
                episode_id=0, src_path="", episode_path=episode_path, format_type="new"
            )
            if frames_data:
                for frame in frames_data:
                    action_text = frame["annotation.language.action_text"][0]
                    all_action_texts.add(action_text)

        # Create tasks.jsonl with all unique texts
        task_to_index = create_tasks_jsonl(tgt_path, repo_id, task_name, all_action_texts)

        # Pass the task mapping to the dataset for custom indexing
        dataset._custom_task_to_index = task_to_index

        # Process in chunks to reduce memory usage
        for chunk_start in tqdm(
            range(0, len(all_episode_paths), chunk_size), desc="Processing chunks"
        ):
            chunk_end = min(chunk_start + chunk_size, len(all_episode_paths))
            chunk_paths = all_episode_paths[chunk_start:chunk_end]
            chunk_descs = all_episode_descs[chunk_start:chunk_end]

            # Process only this chunk
            if debug:
                raw_datasets_chunk = [
                    load_local_dataset(
                        episode_id=0, src_path="", episode_path=episode_path, format_type="new"
                    )
                    for episode_path in tqdm(chunk_paths, desc="Loading chunk data")
                ]
            else:
                raw_datasets_chunk = process_map(
                    load_new_format_episode,
                    chunk_paths,
                    max_workers=os.cpu_count() // 2,
                    desc=f"Loading chunk {chunk_start//chunk_size + 1}/"
                    f"{(len(all_episode_paths) + chunk_size - 1)//chunk_size}",
                )

            # Filter out None results
            valid_datasets = [
                (ds, desc) for ds, desc in zip(raw_datasets_chunk, chunk_descs) if ds is not None
            ]

            # Process each dataset in the chunk
            for raw_dataset, episode_desc in tqdm(
                valid_datasets, desc="Processing episodes in chunk"
            ):
                for raw_dataset_sub in tqdm(raw_dataset[0], desc="Processing frames", leave=False):
                    # Convert string annotation to int index
                    action_text = raw_dataset_sub["annotation.language.action_text"][0]
                    raw_dataset_sub["annotation.language.action_text"] = [
                        task_to_index[action_text]
                    ]
                    raw_dataset_sub["annotation.agibot.tasks"] = [task_to_index[episode_desc]]
                    dataset.add_frame(raw_dataset_sub)
                dataset.save_episode(task=episode_desc, videos=raw_dataset[1])

            # Clear memory after each chunk
            raw_datasets_chunk = None
            valid_datasets = None
            gc.collect()

    # Only consolidate at the end
    try:
        dataset.consolidate(run_compute_stats=False)
    except Exception as e:
        print(f"Warning: Consolidation failed with error: {e}")
        print("Dataset conversion completed but may not be fully consolidated.")

    # Generate modality.json file in the dataset meta directory
    meta_path = os.path.join(tgt_path, repo_id, "meta")
    os.makedirs(meta_path, exist_ok=True)
    generate_modality_json(meta_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert AgiBot dataset to LeRobot format. Supports both old and new format datasets."
    )
    parser.add_argument(
        "--src_path", type=str, required=True, help="Path to source dataset directory"
    )
    parser.add_argument(
        "--task_id",
        type=str,
        required=False,
        help="Task ID (required for old format, optional for new format)",
    )
    parser.add_argument(
        "--tgt_path", type=str, required=True, help="Path to target output directory"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=False,
        help="Repository ID for the dataset (auto-generated if not provided)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Process only first 2 episodes for debugging"
    )
    parser.add_argument(
        "--chunk_size", type=int, default=10, help="Number of episodes to process at once"
    )
    args = parser.parse_args()

    # Detect format first
    format_type = detect_dataset_format(args.src_path)
    print(f"Auto-detected format: {format_type}")

    # Validate arguments based on format
    if format_type == "old":
        if not args.task_id:
            parser.error("--task_id is required for old format datasets")

        task_id = int(args.task_id)
        json_file = f"{args.src_path}/task_info/task_{args.task_id}.json"

        if not Path(json_file).exists():
            parser.error(f"Cannot find task info file: {json_file}")

        main(
            src_path=args.src_path,
            tgt_path=args.tgt_path,
            task_id=task_id,
            repo_id=args.repo_id,
            task_info_json=json_file,
            debug=args.debug,
            chunk_size=args.chunk_size,
        )

    elif format_type == "new":
        main(
            src_path=args.src_path,
            tgt_path=args.tgt_path,
            task_id=args.task_id,
            repo_id=args.repo_id,
            task_info_json=None,
            debug=args.debug,
            chunk_size=args.chunk_size,
        )

    else:
        parser.error(
            f"Unknown dataset format. Please check the directory structure at: {args.src_path}"
        )
		
