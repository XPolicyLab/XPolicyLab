import os
import random
import traceback
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

from XPolicyLab.utils.process_data import (
    decode_image_bit,
    get_robot_action_dim_info,
    pack_robot_state,
)


class XPolicyLabDataset:
    """
    Direct reader for XPolicyLab standard HDF5 trajectories.

    It returns the same sample fields as RobotwinAgilexDataset, but avoids
    materializing a converted H-RDT copy under policy/H_RDT/data.
    """

    DATASET_NAME = "xpolicylab"

    def __init__(
        self,
        mode="single_task",
        data_root=None,
        raw_bench_name="RoboDojo",
        task_name=None,
        env_cfg_type=None,
        action_type="joint",
        max_episodes=None,
        config=None,
        stat_path=None,
        upsample_rate=3,
        val=False,
    ):
        self.mode = mode
        self.data_root = Path(data_root).expanduser() if data_root else None
        self.raw_bench_name = raw_bench_name
        self.task_name = task_name
        self.env_cfg_type = env_cfg_type
        self.action_type = action_type
        self.max_episodes = max_episodes
        self.stat_path = Path(stat_path).expanduser() if stat_path else Path(__file__).resolve().parent / "stats.json"
        self.upsample_rate = upsample_rate
        self.val = val

        if self.mode not in ("single_task", "multi_task"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'single_task' or 'multi_task'.")
        if config is None:
            raise ValueError("XPolicyLabDataset requires config.")
        if self.env_cfg_type is None:
            raise ValueError("XPolicyLabDataset requires env_cfg_type.")
        if self.action_type != "joint":
            raise ValueError("XPolicyLabDataset currently supports only action_type='joint'.")

        self.chunk_size = config["common"]["action_chunk_size"]
        self.img_history_size = config["common"]["img_history_size"]
        self.num_cameras = config["common"]["num_cameras"]
        if self.num_cameras != 3:
            raise ValueError(f"Unsupported num_cameras={self.num_cameras}, only 3 cameras supported.")

        self.cameras = ["cam_head", "cam_right_wrist", "cam_left_wrist"]
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.action_q01, self.action_q99 = self._load_action_stats()
        self.action_scale = self.action_q99 - self.action_q01
        self.action_scale = np.where(self.action_scale < 1e-6, 1.0, self.action_scale)

        self.episode_files = []
        self.task_to_episodes = {}
        self.task_weights = {}
        self.total_episodes = 0
        self._initialize_dataset()

    def _load_action_stats(self):
        if not self.stat_path.exists():
            raise FileNotFoundError(
                f"XPolicyLab q01/q99 stats file not found: {self.stat_path}. "
                "Run datasets/xpolicylab/calc_stat.py or run_xpolicylab_pipeline.sh first."
            )

        with self.stat_path.open("r", encoding="utf-8") as fp:
            stats = json.load(fp)

        dataset_stats = stats.get(self.DATASET_NAME)
        if dataset_stats is None:
            raise KeyError(f"Missing '{self.DATASET_NAME}' section in stats file: {self.stat_path}")

        if "q01" not in dataset_stats or "q99" not in dataset_stats:
            raise KeyError(f"Stats file must contain q01 and q99 arrays: {self.stat_path}")

        action_q01 = np.asarray(dataset_stats["q01"], dtype=np.float32)
        action_q99 = np.asarray(dataset_stats["q99"], dtype=np.float32)
        expected_dim = sum(self.robot_action_dim_info["arm_dim"]) + sum(self.robot_action_dim_info["ee_dim"])
        if action_q01.shape[0] != expected_dim or action_q99.shape[0] != expected_dim:
            raise ValueError(
                f"Stats action dim mismatch: expected {expected_dim}, "
                f"got q01={action_q01.shape[0]}, q99={action_q99.shape[0]}."
            )

        print(f"Loaded XPolicyLab q01/q99 stats from {self.stat_path}")
        return action_q01, action_q99

    def _normalize_action(self, action):
        clipped = np.clip(action, self.action_q01, self.action_q99)
        return ((clipped - self.action_q01) / self.action_scale * 2.0 - 1.0).astype(np.float32)

    def _candidate_task_data_dirs(self, task_name):
        root = self.data_root
        if root is None:
            return []

        return [
            root / self.raw_bench_name / task_name / self.env_cfg_type / "data",
            root / task_name / self.env_cfg_type / "data",
            root / task_name / "data" / task_name / self.env_cfg_type / "data",
            root / task_name / "data",
            root,
        ]

    def _resolve_task_data_dir(self, task_name):
        for candidate in self._candidate_task_data_dirs(task_name):
            if candidate.is_dir() and self._find_hdf5_files(candidate):
                return candidate

        tried = "\n".join(str(path) for path in self._candidate_task_data_dirs(task_name))
        raise FileNotFoundError(f"No XPolicyLab HDF5 data found for task={task_name}. Tried:\n{tried}")

    def _find_hdf5_files(self, data_dir):
        files = sorted(Path(data_dir).glob("episode_*.hdf5"))
        if not files:
            files = sorted(Path(data_dir).glob("*.hdf5"))
        if not files:
            files = sorted(Path(data_dir).glob("*.h5"))
        return [str(path) for path in files]

    def _scan_single_task(self):
        if not self.task_name:
            raise ValueError("single_task mode requires task_name.")

        data_dir = self._resolve_task_data_dir(self.task_name)
        files = self._find_hdf5_files(data_dir)
        if self.max_episodes is not None:
            files = files[: self.max_episodes]

        print(f"XPolicyLab single task {self.task_name}: found {len(files)} files from {data_dir}")
        return files

    def _scan_multi_task(self):
        if self.data_root is None:
            raise ValueError("multi_task mode requires data_root.")

        search_root = self.data_root / self.raw_bench_name
        if not search_root.exists():
            search_root = self.data_root

        task_names = sorted(
            path.name
            for path in search_root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        )

        task_to_episodes = {}
        for task_name in task_names:
            try:
                data_dir = self._resolve_task_data_dir(task_name)
            except FileNotFoundError:
                continue
            files = self._find_hdf5_files(data_dir)
            if self.max_episodes is not None:
                files = files[: self.max_episodes]
            if files:
                random.shuffle(files)
                task_to_episodes[task_name] = files
                print(f"XPolicyLab multi task {task_name}: found {len(files)} files")

        return task_to_episodes

    def _initialize_dataset(self):
        if self.mode == "single_task":
            self.episode_files = self._scan_single_task()
            self.total_episodes = len(self.episode_files)
        else:
            self.task_to_episodes = self._scan_multi_task()
            self.total_episodes = sum(len(files) for files in self.task_to_episodes.values())
            task_count = len(self.task_to_episodes)
            if task_count:
                self.task_weights = {
                    task_name: 1.0 / task_count for task_name in self.task_to_episodes
                }

        if self.total_episodes == 0:
            raise ValueError("Error: No XPolicyLab HDF5 files found, please check data path.")

        print(f"XPolicyLab dataset initialized. Total {self.total_episodes} episodes")

    def __len__(self):
        return self.total_episodes * 200

    def _load_action_data(self, h5_file):
        action_group = h5_file["action"]
        action = {key: action_group[key][:] for key in action_group.keys()}
        return pack_robot_state(
            {"action": action},
            self.action_type,
            self.robot_action_dim_info,
            source_type="dataset",
            state_type="action",
        ).astype(np.float32)

    def _decode_xpolicy_image(self, image_bits):
        image = decode_image_bit(image_bits)
        if image is None:
            raise ValueError("Failed to decode XPolicyLab image bytes.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[:2] != (480, 640):
            image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
        return image

    def _parse_camera_history(self, dataset, idx):
        start_i = max(idx - (self.img_history_size - 1) * self.upsample_rate, 0)
        frame_indices = list(range(start_i, idx + 1, self.upsample_rate))

        frames = []
        for frame_idx in frame_indices:
            if frame_idx < len(dataset):
                frames.append(self._decode_xpolicy_image(dataset[frame_idx]))

        if not frames:
            frames.append(self._decode_xpolicy_image(dataset[0]))
        if len(frames) < self.img_history_size:
            frames = [frames[0]] * (self.img_history_size - len(frames)) + frames

        return np.asarray(frames[-self.img_history_size :], dtype=np.uint8)

    def _task_name_from_path(self, hdf5_file_path):
        if self.mode == "single_task":
            return self.task_name

        path = Path(hdf5_file_path)
        parts = path.parts
        if self.env_cfg_type in parts:
            env_index = parts.index(self.env_cfg_type)
            if env_index > 0:
                return parts[env_index - 1]
        if self.raw_bench_name in parts:
            index = parts.index(self.raw_bench_name)
            if index + 1 < len(parts):
                return parts[index + 1]
        return path.parents[2].name

    def load_language_embedding(self, hdf5_file_path):
        task_name = self._task_name_from_path(hdf5_file_path)
        current_dir = Path(__file__).resolve().parent
        embedding_path = current_dir / "lang_embeddings" / f"{task_name}.pt"

        if not embedding_path.exists():
            print(f"Warning: Task embedding file not found: {embedding_path}")
            return None

        embedding_data = torch.load(str(embedding_path), map_location="cpu")
        embeddings = embedding_data.get("embeddings") if isinstance(embedding_data, dict) else embedding_data
        if embeddings is None:
            print(f"Warning: No embeddings found in {embedding_path}")
            return None
        if embeddings.dim() == 3:
            embeddings = embeddings.squeeze(0)
        return embeddings

    def extract_episode_item(self, hdf5_file):
        try:
            with h5py.File(hdf5_file, "r", swmr=True, libver="latest") as f:
                action_data = self._load_action_data(f)
                max_index = len(action_data) - 2
                if max_index <= 0:
                    return None

                index = random.randint(0, max_index)
                action_current = self._normalize_action(action_data[index])
                action_end = min(index + self.chunk_size * self.upsample_rate, max_index + 1)
                action_chunk = self._normalize_action(
                    action_data[index + 1 : action_end + 1 : self.upsample_rate]
                )

                if action_chunk.shape[0] < self.chunk_size:
                    last_part = np.repeat(
                        action_chunk[-1:],
                        self.chunk_size - action_chunk.shape[0],
                        axis=0,
                    )
                    action_chunk = np.concatenate([action_chunk, last_part], axis=0)

                camera_paths = {
                    "cam_head": "vision/cam_head/colors",
                    "cam_left_wrist": "vision/cam_left_wrist/colors",
                    "cam_right_wrist": "vision/cam_right_wrist/colors",
                }

                current_images = []
                for camera_name in self.cameras:
                    camera_path = camera_paths[camera_name]
                    if camera_path not in f:
                        print(f"Warning: Camera {camera_path} not found in {hdf5_file}")
                        return None
                    current_images.append(self._parse_camera_history(f[camera_path], index))

                language_embedding = self.load_language_embedding(hdf5_file)
                if language_embedding is None:
                    print(f"Warning: Failed to load language embedding for {hdf5_file}")
                    return None

                current_images_mask = [
                    np.array([True] * self.img_history_size, dtype=bool)
                    for _ in range(self.num_cameras)
                ]

                return {
                    "current_images": np.asarray(current_images, dtype=np.uint8),
                    "current_images_mask": current_images_mask,
                    "actions": action_chunk,
                    "states": np.expand_dims(action_current, axis=0),
                    "state_indicator": np.ones_like(action_current),
                    "action_norm": np.ones_like(action_chunk),
                    "instruction": language_embedding,
                    "dataset_name": self.DATASET_NAME,
                }
        except Exception as exc:
            print(f"Error processing {hdf5_file}: {exc}")
            traceback.print_exc()
            return None

    def get_item(self, index=None):
        if self.mode == "single_task":
            if not self.episode_files:
                self._initialize_dataset()
            episode_file = random.choice(self.episode_files)
            task_name = self.task_name
        else:
            if not self.task_to_episodes:
                self._initialize_dataset()
            task_name = random.choices(
                list(self.task_weights.keys()),
                weights=list(self.task_weights.values()),
                k=1,
            )[0]
            episode_file = random.choice(self.task_to_episodes[task_name])

        for _ in range(3):
            item = self.extract_episode_item(episode_file)
            if item is not None:
                return item

            if self.mode == "single_task":
                episode_file = random.choice(self.episode_files)
            else:
                episode_file = random.choice(self.task_to_episodes[task_name])

        print("Warning: Failed to extract XPolicyLab sample, returning None")
        return None
