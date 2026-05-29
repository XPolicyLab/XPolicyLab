"""
RoboChallenge dataset reader.

Features:
- read mode: returns the same dict structure as RoboMINDDatasetReader (question/action/proprio/images/metadata)
- stats mode: computes min/max/mean/std/q01/q99 for action/state, grouped by embodiment
- supports single/dual-arm embodiments and optional filtering by task or embodiment
- configurable control source: joint positions (default) or end-effector pose
"""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import cv2
import numpy as np
import time
from tqdm import tqdm
from a1.data.dataset import Dataset
from a1.data.vla.utils import NormalizationType

def load_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def quaternion_to_euler(x: float, y: float, z: float, w: float) -> Tuple[float, float, float]:
    """Convert quaternion to roll/pitch/yaw (radians)."""
    # roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    # pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)

    # yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    return roll, pitch, yaw


class RoboChallengeDatasetReader(Dataset):
    EMBODIMENT_CONFIGS = {
        "ARX5": {
            "video_names": ["global_realsense_rgb.mp4", "arm_realsense_rgb.mp4", "right_realsense_rgb.mp4"],
            "is_dual_arm": False,
            'control_type': 'joint',
        },
        "UR5": {
            "video_names": ["global_realsense_rgb.mp4", "handeye_realsense_rgb.mp4"],
            "is_dual_arm": False,
            'control_type': 'joint',
        },
        "FRANKA": {
            "video_names": ["main_realsense_rgb.mp4", "handeye_realsense_rgb.mp4", "side_realsense_rgb.mp4"],
            "is_dual_arm": False,
            'control_type': 'ee',
        },
        "ALOHA": {
            "video_names": ["cam_high_rgb.mp4", "cam_wrist_left_rgb.mp4", "cam_wrist_right_rgb.mp4"],
            "is_dual_arm": True,
            'control_type': 'joint',
        },
    }

    CONTROL_CHOICES = ("joint", "ee")

    def __init__(
        self,
        dataset_path: str,
        control_type: str = None,
        resolution: Optional[Tuple[int, int]] = None,
        chunk_size: int = 50,
        pad_action_and_proprio=True,
        fixed_action_dim: int = 32,
        frame_interval: int = 1,
        env_names: Optional[List[str]] = None,
        embodiment: Optional[str] = None,
        skip_images: bool = False,
        state_cache_size: int = 64,
        verbose: bool = False,
        normalization_type: Optional[NormalizationType] = None,
        norm_stats_path: Optional[str] = None,
    ):
        """
        Args:
            dataset_path: root path of RoboChallenge data (containing task folders or a single task)
            control_type: 'joint' (default) or 'ee' to choose action/state source
            resolution: (width, height) for resizing frames; None keeps original video resolution
            chunk_size: number of future actions to return in read mode
            frame_interval: stride when building the read-mode frame index
            env_names: list of task names to include; None = all tasks found under dataset_path
            embodiments: list of embodiments to include (ARX5/UR5/FRANKA/ALOHA); None = all
            skip_images: if True, do not decode videos (faster for pure control-only use)
            state_cache_size: number of episodes to keep in memory (states cache), set small to reduce RAM
            verbose: print progress during indexing/stats
        """
        if control_type is not None and control_type not in self.CONTROL_CHOICES:
            raise ValueError(f"control_type must be one of {self.CONTROL_CHOICES}")
        self.dataset_path = Path(dataset_path)
        self.control_type = control_type if control_type is not None else self.EMBODIMENT_CONFIGS[embodiment]["control_type"]
        self.resolution = tuple[int, ...](resolution) if resolution else None
        self.chunk_size = chunk_size
        self.fixed_action_dim = fixed_action_dim
        self.pad_action_and_proprio = pad_action_and_proprio
        self.frame_interval = max(1, frame_interval)
        self.skip_images = skip_images
        self.state_cache_size = max(0, state_cache_size)
        self.verbose = verbose

        self.target_embodiment = embodiment.upper()

        # 归一化配置：从指定 stats JSON 中读取，格式与 compute_stats() / lerobot 保持一致
        self.normalization_type: Optional[NormalizationType] = normalization_type
        self._norm_stats: Optional[Dict[str, Dict[str, List[float]]]] = None

        if self.normalization_type is not None and norm_stats_path:
            try:
                stats_path = Path(norm_stats_path)
                with stats_path.open("r", encoding="utf-8") as f:
                    stats_json = json.load(f)
                # 兼容 compute_stats() 的输出结构
                if "embodiments" in stats_json:
                    emb_stats = stats_json["embodiments"][self.target_embodiment]["stats"]
                else:
                    emb_stats = stats_json
                # 只保留 action/state 两个 key，结构与 lerobot 的 metadata[key] 对齐
                self._norm_stats = {
                    "actions": emb_stats["actions"],
                    "state": emb_stats["state"],
                }
            except Exception as e:
                logging.warning(f"Failed to load norm stats from {norm_stats_path}: {e}")

        self._state_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}  # key -> (states_14, timestamps)
        self._state_cache_order: List[str] = []
        self._video_cache: Dict[Path, cv2.VideoCapture] = {}
        self._video_cache_order: List[Path] = []
        self._max_video_cache = 0
        self._episodes: List[Dict] = []
        self._index: List[Tuple[int, int]] = []  # (episode_id, frame_idx)
        self._frame_ranges: List[Tuple[int, int]] = []  # (start_frame, end_frame)
        self._tasks = self._discover_tasks(env_names)
        self._build_index()

    # --- helpers for discovering dataset structure ---
    def _normalize_embodiment(self, robot_id: str) -> Optional[str]:
        rid = robot_id.lower()
        if "arx5" in rid:
            return "ARX5"
        if "ur5" in rid:
            return "UR5"
        if "franka" in rid:
            return "FRANKA"
        if "aloha" in rid:
            return "ALOHA"
        return None

    def _load_task_info(self, task_dir: Path) -> Optional[Dict]:
        info_path = task_dir / "meta" / "task_info.json"
        if not info_path.exists():
            return None
        with info_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        robot_id = data.get("robot_id", "")
        embodiment = self._normalize_embodiment(robot_id)
        if embodiment is None:
            return None

        video_info = data.get("video_info", {})
        res = None
        if "width" in video_info and "height" in video_info:
            res = (int(video_info["width"]), int(video_info["height"]))

        return {
            "task_name": data.get("task_desc", {}).get("task_name", task_dir.name),
            "prompt": data.get("task_desc", {}).get("prompt", task_dir.name),
            "fps": float(video_info.get("fps", 30.0)),
            "embodiment": embodiment,
            "robot_id": robot_id,
            "default_resolution": res,
        }

    def _discover_tasks(self, env_names: Optional[List[str]]) -> List[Dict]:
        # Determine whether dataset_path points to a task folder or the root containing tasks.
        task_dirs: List[Path]
        if (self.dataset_path / "meta" / "task_info.json").exists():
            task_dirs = [self.dataset_path]
        else:
            task_dirs = [
                p for p in self.dataset_path.iterdir() if p.is_dir() and (p / "meta" / "task_info.json").exists()
            ]

        if self.verbose:
            print(f"Discovered {len(task_dirs)} task folders before filtering")
        if env_names:
            env_set = set(env_names)
            task_dirs = [p for p in task_dirs if p.name in env_set]

        tasks = []
        for task_dir in sorted(task_dirs, key=lambda p: p.name):
            task_meta = self._load_task_info(task_dir)
            if task_meta is None:
                if self.verbose:
                    print(f"Skip task (missing/invalid meta): {task_dir}")
                continue
            if task_meta["embodiment"] != self.target_embodiment:
                if self.verbose:
                    print(f"Skip task (embodiment filtered): {task_dir} -> {task_meta['embodiment']}")
                continue
            task_meta["path"] = task_dir
            tasks.append(task_meta)

        if not tasks:
            raise ValueError("No valid tasks found under dataset_path with the given filters.")
        return tasks

    # --- core data loading ---
    def _build_index(self) -> None:
        episode_counter = 0
        for task in self._tasks:
            data_dir = task["path"] / "data"
            if not data_dir.exists():
                continue
            if self.verbose:
                print(f"Indexing task {task['task_name']} ({task['embodiment']}) at {data_dir}")
            episode_dirs = [d for d in sorted(data_dir.iterdir()) if d.is_dir()]
            iterator = tqdm(episode_dirs, desc=f"Index {task['task_name']}", disable=not self.verbose)
            for episode_dir in iterator:
                if not episode_dir.is_dir():
                    continue
                meta_dir = episode_dir / "meta"
                states_dir = episode_dir / "states"
                videos_dir = episode_dir / "videos"
                if not states_dir.exists() or not videos_dir.exists():
                    continue

                single_state = states_dir / "states.jsonl"
                left_state = states_dir / "left_states.jsonl"
                right_state = states_dir / "right_states.jsonl"
                meta_info = meta_dir / "episode_meta.json"
                state_paths: List[Path]
                if single_state.exists():
                    state_paths = [single_state]
                elif left_state.exists() and right_state.exists():
                    state_paths = [left_state, right_state]
                else:
                    continue

                episode_info = {
                    "task_name": task["task_name"],
                    "prompt": task["prompt"],
                    "embodiment": task["embodiment"],
                    "fps": task["fps"],
                    "default_resolution": task["default_resolution"],
                    "episode_dir": episode_dir,
                    "videos_dir": videos_dir,
                    "state_paths": state_paths,
                    "episode_index": episode_counter,
                }
                try:
                    with open(meta_info, "r") as f:
                        meta_data = json.load(f)
                        num_frames = meta_data["frames"] -1
                    episode_info["num_frames"] = num_frames
                except Exception as e:
                    print(f"Error loading meta_info: {e}")
                    continue
                self._episodes.append(episode_info)
                start_frame = len(self._index)
                for frame_idx in range(0, num_frames, self.frame_interval):
                    self._index.append((len(self._episodes) - 1, frame_idx))
                end_frame = len(self._index)
                self._frame_ranges.append((start_frame, end_frame))
                episode_counter += 1
    
    def _cache_key(self, episode_info: Dict) -> str:
        return "|".join(str(p) for p in episode_info["state_paths"])
    
    def _prune_state_cache(self):
        while self.state_cache_size and len(self._state_cache_order) > self.state_cache_size:
            old_key = self._state_cache_order.pop(0)
            self._state_cache.pop(old_key, None)
            
    def _get_states_sequence(self, episode_info: Dict, use_cache: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        key = self._cache_key(episode_info)
        if use_cache and key in self._state_cache:
            # move to end for simple LRU
            if key in self._state_cache_order:
                self._state_cache_order.remove(key)
                self._state_cache_order.append(key)
            return self._state_cache[key]

        if len(episode_info["state_paths"]) == 1:
            states_14, timestamps = self._load_single_arm_states(
                episode_info["state_paths"][0], episode_info["embodiment"], episode_info["fps"]
            )
        else:
            states_14, timestamps = self._load_dual_arm_states(
                episode_info["state_paths"][0],
                episode_info["state_paths"][1],
                episode_info["embodiment"],
                episode_info["fps"],
            )

        if use_cache and self.state_cache_size > 0:
            self._state_cache[key] = (states_14, timestamps)
            self._state_cache_order.append(key)
            self._prune_state_cache()
        return states_14, timestamps

    def _normalize_array(self, data: np.ndarray, key: str) -> np.ndarray:
        """按照 lerobot 的 normalize_action_and_proprio 风格进行归一化。"""
        if self.normalization_type is None or self._norm_stats is None:
            return data

        meta = self._norm_stats.get(key)
        if meta is None:
            return data

        if self.normalization_type == NormalizationType.NORMAL:
            # mask 直接用全 1，与 lerobot 当前实现一致
            mean = np.array(meta["mean"], dtype=np.float32)
            std = np.array(meta["std"], dtype=np.float32)
            mask = np.ones_like(mean, dtype=bool)
            return np.where(mask, (data - mean) / (std + 1e-8), data)

        if self.normalization_type in (NormalizationType.BOUNDS, NormalizationType.BOUNDS_Q99):
            if self.normalization_type == NormalizationType.BOUNDS:
                low = np.array(meta["min"], dtype=np.float32)
                high = np.array(meta["max"], dtype=np.float32)
            else:
                low = np.array(meta["q01"], dtype=np.float32)
                high = np.array(meta["q99"], dtype=np.float32)

            mask = np.ones_like(high, dtype=bool)
            normalized = np.where(
                mask,
                np.clip(2.0 * (data - low) / (high - low + 1e-8) - 1.0, -1.0, 1.0),
                data,
            )

            # 与 lerobot 一致：未使用的维度（min == max）直接置 0
            if "min" in meta and "max" in meta:
                zeros_mask = np.array(meta["min"], dtype=np.float32) == np.array(meta["max"], dtype=np.float32)
                normalized = np.where(zeros_mask, 0.0, normalized)
            return normalized

        raise ValueError(f"Unknown Normalization Type {self.normalization_type}")

    # def _ensure_len(self, values: List[float], target_len: int) -> List[float]:
    #     vals = list(values)
    #     if len(vals) >= target_len:
    #         return vals[:target_len]
    #     return vals + [0.0] * (target_len - len(vals))

    def _select_gripper(self, entry: Dict, embodiment: str) -> float:
        if "gripper_width" in entry:
            gw = entry["gripper_width"]
            if isinstance(gw, (list, tuple, np.ndarray)):
                assert len(gw) >0
                grip = float(gw[0])
            else:
                grip = float(gw)
        elif "gripper" in entry:
            g = entry["gripper"]
            if isinstance(g, (list, tuple)):
                assert len(g) > 0
                grip = float(np.mean(g))
            else:
                grip = float(g)
        else:
            raise ValueError(f"Unknown gripper type: {entry}")
        if embodiment == "UR5":
            grip /= 255
        elif embodiment == "ALOHA":
            grip *= 10
        elif embodiment == "ARX5":
            grip *= 10
        elif embodiment == "FRANKA":
            grip *= 10
        return grip

    def _select_joint_vector(self, entry: Dict, embodiment: str) -> Optional[List[float]]:
        joint = entry.get("joint_positions")
        assert joint is not None
        grip = self._select_gripper(entry, embodiment)
        vec6 = joint
        return vec6 + [grip]

    def _select_ee_vector(self, entry: Dict, embodiment: str) -> Optional[List[float]]:
        grip = self._select_gripper(entry, embodiment)

        if embodiment == "ARX5":
            pose = entry.get("end_effector_pose")
            if pose is None:
                return None
            vec6 = pose
            return vec6 + [grip]

        if embodiment in ("UR5", "FRANKA"):
            pose = entry.get("ee_positions") or entry.get("end_effector_pose")
            if pose is None:
                return None
            pose_list = list(pose)
            if len(pose_list) >= 7:
                pos = pose_list[:3]
                quat = pose_list[3:7]
                euler = quaternion_to_euler(*quat)
                vec6 = pos + list(euler)
            else:
                vec6 = pose_list
            
            return vec6 + [grip]

        if embodiment == "ALOHA":
            pose_rpy = entry.get("ee_pose_rpy")
            if pose_rpy is not None:
                vec6 = pose_rpy
                return vec6 + [grip]
            pose_quat = entry.get("ee_pose_quaternion")
            if pose_quat is not None and len(pose_quat) >= 7:
                pos = pose_quat[:3]
                quat = pose_quat[3:7]
                euler = quaternion_to_euler(*quat)
                vec6 = pos + list(euler)
                return vec6 + [grip]
        return None

    def _extract_arm_vector(self, entry: Dict, embodiment: str) -> Optional[List[float]]:
        if self.control_type == "joint":
            vec = self._select_joint_vector(entry, embodiment)
            return vec
        elif self.control_type == "ee":
            vec = self._select_ee_vector(entry, embodiment)
            return vec
        else:
            raise ValueError(f"Unknown control type: {self.control_type}")

    def _load_single_arm_states(self, state_path: Path, embodiment: str, fps: float) -> Tuple[np.ndarray, np.ndarray]:
        entries = load_jsonl(state_path)
        pose_list = []
        timestamps = []
        for idx, entry in enumerate(entries):
            vec = self._extract_arm_vector(entry, embodiment)
            if vec is None:
                continue
            pose_list.append(vec)
            ts = entry.get("timestamp", None)
            timestamps.append(float(ts) if ts is not None else idx / fps)

        if not pose_list:
            raise ValueError(f"No valid frames in {state_path}")
            # return np.empty((0, 14), dtype=np.float32), np.empty((0,), dtype=np.float64)
 
        left = np.asarray(pose_list, dtype=np.float32)
        # zeros = np.zeros((left.shape[0], 7), dtype=np.float32)
        # states_14 = np.concatenate([left, zeros], axis=1)  # [left(real), right(zeros)]
        ts_arr = np.asarray(timestamps, dtype=np.float64)
        return left, ts_arr

    def _load_dual_arm_states(
        self, left_state_path: Path, right_state_path: Path, embodiment: str, fps: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        left_entries = load_jsonl(left_state_path)
        right_entries = load_jsonl(right_state_path)
        total = min(len(left_entries), len(right_entries))
        pose_list = []
        timestamps = []
        for idx in range(total):
            left_vec = self._extract_arm_vector(left_entries[idx], embodiment)
            right_vec = self._extract_arm_vector(right_entries[idx], embodiment)
            if left_vec is None or right_vec is None:
                continue
            pose_list.append(left_vec + right_vec)  # order: left then right
            l_ts = left_entries[idx].get("timestamp")
            r_ts = right_entries[idx].get("timestamp")
            if l_ts is not None and r_ts is not None:
                ts = min(float(l_ts), float(r_ts))
            elif l_ts is not None:
                ts = float(l_ts)
            elif r_ts is not None:
                ts = float(r_ts)
            else:
                ts = idx / fps
            timestamps.append(ts)

        if not pose_list:
            raise ValueError(f"No valid frames in {left_state_path} and {right_state_path}")
            # return np.empty((0, 14), dtype=np.float32), np.empty((0,), dtype=np.float64)

        states_14 = np.asarray(pose_list, dtype=np.float32)
        ts_arr = np.asarray(timestamps, dtype=np.float64)
        return states_14, ts_arr

    # --- frame/image helpers ---
    def _get_video_list(self, videos_dir: Path, embodiment: str) -> List[Path]:
        cfg = self.EMBODIMENT_CONFIGS.get(embodiment, {})
        preferred = cfg.get("video_names", [])
        existing = []
        for name in preferred:
            candidate = videos_dir / name
            if candidate.exists():
                existing.append(candidate)
        if existing:
            return existing
        # fallback: all mp4 files sorted
        return sorted(videos_dir.glob("*.mp4"))

    def _get_video_capture(self, video_path: Path) -> Optional[cv2.VideoCapture]:
        if video_path in self._video_cache:
            return self._video_cache[video_path]
        cap = cv2.VideoCapture(str(video_path))

        return cap

    def _grab_frame(self, video_path: Path, frame_idx: int, target_resolution: Optional[Tuple[int, int]]) -> Optional[np.ndarray]:
        cap = self._get_video_capture(video_path)
        if cap is None or not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"Failed to read frame: {video_path}, frame_idx: {frame_idx}")
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if target_resolution:
            frame = cv2.resize(frame, target_resolution, interpolation=cv2.INTER_AREA)
        return frame

    def _load_images(self, episode_info: Dict, frame_idx: int) -> List[np.ndarray]:
        if self.skip_images:
            return []
        target_res = self.resolution or episode_info.get("default_resolution")
        videos = self._get_video_list(episode_info["videos_dir"], episode_info["embodiment"])
        assert len(videos) > 0
        images = []
        for video_path in videos:
            frame = self._grab_frame(video_path, frame_idx, target_res)
            if frame is None:
                if target_res is None:
                    target_res = (640, 480)
                frame = np.zeros((target_res[1], target_res[0], 3), dtype=np.uint8)
            images.append(frame)
        return images

    # --- public interface ---
    def __len__(self) -> int:
        return len(self._index)

    def get(self, item, rng=None):
        return self[item]

    def __getitem__(self, idx: int) -> Dict:
        if idx < 0 or idx >= len(self._index):
            raise IndexError(f"Index {idx} out of range [0, {len(self._index)})")
        episode_id, frame_idx = self._index[idx]
        episode_info = self._episodes[episode_id]
        return self.read_sample(episode_info, frame_idx, dataset_idx=idx)

    def read_sample(self, episode_info: Dict, frame_idx: int, dataset_idx: int) -> Dict:
        states_14, timestamps = self._get_states_sequence(episode_info)
        if states_14.shape[0] <= 1:
            raise ValueError(f"No valid frames in {episode_info['episode_dir']}")

        action_seq = states_14[1:]
        state_seq = states_14[:-1]
        max_frame_idx = state_seq.shape[0] - 1
        frame_idx = min(frame_idx, max_frame_idx)

        frame_state = state_seq[frame_idx : frame_idx + 1]
        frame_action = action_seq[frame_idx : frame_idx + self.chunk_size]
        if frame_action.shape[0] < self.chunk_size and frame_action.shape[0] > 0:
            last_action = frame_action[-1:]
            padding = np.repeat(last_action, self.chunk_size - frame_action.shape[0], axis=0)
            frame_action = np.concatenate([frame_action, padding], axis=0)
        elif frame_action.shape[0] == 0:
            frame_action = np.zeros((self.chunk_size, states_14.shape[1]), dtype=np.float32)

        # 使用与 lerobot 相同风格的归一化；未配置时退回到原始裁剪逻辑
        frame_state[frame_state > 4] = 4
        frame_action[frame_action > 4] = 4
        if self.normalization_type is not None and self._norm_stats is not None:
            frame_state = self._normalize_array(frame_state, "state")
            frame_action = self._normalize_array(frame_action, "actions")
            

        images = self._load_images(episode_info, frame_idx)

        ts = timestamps[frame_idx] if frame_idx < len(timestamps) else frame_idx / episode_info["fps"]
        question = episode_info["prompt"]

        if self.pad_action_and_proprio:
            if frame_state.shape[-1] < self.fixed_action_dim:
                frame_state = np.pad(frame_state, ((0, 0), (0, self.fixed_action_dim-frame_state.shape[-1])), mode='constant')
            if frame_action.shape[-1] < self.fixed_action_dim:
                frame_action = np.pad(frame_action, ((0, 0), (0, self.fixed_action_dim-frame_action.shape[-1])), mode='constant')
        
        return {
            "question": question,
            "timestep": ts,
            "answer": "Action",
            "style": "action",
            "action": frame_action,
            "action_pad_mask": np.zeros_like(frame_action, dtype=bool),
            "proprio": frame_state,
            "images": images,
            "metadata": {
                "timestamp": ts,
                "frame_index": frame_idx,
                "episode_index": episode_info["episode_index"],
                "index": dataset_idx,
                "task_index": 0,
                "task": question,
                "file_path": str(episode_info["state_paths"][0]),
                "num_frames": episode_info.get("num_frames", state_seq.shape[0]),
                "embodiment": episode_info["embodiment"],
            },
        }

    def compute_stats(self) -> Dict:
        per_emb_actions: Dict[str, List[np.ndarray]] = {}
        per_emb_states: Dict[str, List[np.ndarray]] = {}
        per_emb_envs: Dict[str, set] = {}

        if self.verbose:
            print(f"Computing stats over {len(self._episodes)} episodes...")
        for episode_info in tqdm(self._episodes, desc="Computing stats"):
            states_14, _ = self._get_states_sequence(episode_info, use_cache=False)
            states_14[states_14 > 4] = 4
            if states_14.shape[0] <= 1:
                continue
            action_seq = states_14[1:]
            state_seq = states_14[:-1]
            emb = episode_info["embodiment"]
            per_emb_actions.setdefault(emb, []).append(action_seq)
            per_emb_states.setdefault(emb, []).append(state_seq)
            per_emb_envs.setdefault(emb, set()).add(episode_info["task_name"])

        if not per_emb_actions:
            raise ValueError("No valid data found for computing stats.")

        result = {"embodiments": {}, "overall_stats": None}
        all_actions = []
        all_states = []

        for emb, action_list in per_emb_actions.items():
            state_list = per_emb_states[emb]
            actions = np.concatenate(action_list, axis=0)
            states = np.concatenate(state_list, axis=0)
            stats = {
                "actions": {
                    "max": np.max(actions, axis=0).tolist(),
                    "min": np.min(actions, axis=0).tolist(),
                    "mean": np.mean(actions, axis=0).tolist(),
                    "std": np.std(actions, axis=0).tolist(),
                    "q01": np.percentile(actions, 1, axis=0).tolist(),
                    "q99": np.percentile(actions, 99, axis=0).tolist(),
                },
                "state": {
                    "max": np.max(states, axis=0).tolist(),
                    "min": np.min(states, axis=0).tolist(),
                    "mean": np.mean(states, axis=0).tolist(),
                    "std": np.std(states, axis=0).tolist(),
                    "q01": np.percentile(states, 1, axis=0).tolist(),
                    "q99": np.percentile(states, 99, axis=0).tolist(),
                },
            }
            result["embodiments"][emb] = {
                "environments": sorted(list(per_emb_envs[emb])),
                "stats": stats,
            }
            all_actions.append(actions)
            all_states.append(states)

        if all_actions:
            combined_actions = np.concatenate(all_actions, axis=0)
            combined_states = np.concatenate(all_states, axis=0)
            result["overall_stats"] = {
                "actions": {
                    "max": np.max(combined_actions, axis=0).tolist(),
                    "min": np.min(combined_actions, axis=0).tolist(),
                    "mean": np.mean(combined_actions, axis=0).tolist(),
                    "std": np.std(combined_actions, axis=0).tolist(),
                    "q01": np.percentile(combined_actions, 1, axis=0).tolist(),
                    "q99": np.percentile(combined_actions, 99, axis=0).tolist(),
                },
                "state": {
                    "max": np.max(combined_states, axis=0).tolist(),
                    "min": np.min(combined_states, axis=0).tolist(),
                    "mean": np.mean(combined_states, axis=0).tolist(),
                    "std": np.std(combined_states, axis=0).tolist(),
                    "q01": np.percentile(combined_states, 1, axis=0).tolist(),
                    "q99": np.percentile(combined_states, 99, axis=0).tolist(),
                },
            }
        return result

def export_trajectory_video(ds,
                                    start_index: int,
                                    end_index: int,
                                    out_path: str = 'trajectory.mp4',
                                    fps: int = 30,
                                    max_action_dims: int = 6,
                                    frame_interval: int = 1):
    """从已加载的 LeRobot 数据集对象导出某段轨迹到 mp4。

    参数：
    - ds: 已加载的 LeRobot 数据集对象（支持 __getitem__ 返回包含图像与动作的数据字典）
    - start_index, end_index: 帧索引区间 [start_index, end_index)（左闭右开）
    - out_path: 输出 mp4 路径
    - fps: 视频帧率
    - max_action_dims: 动作维度可视化的最多维数
    - frame_interval: 帧间隔
    """
    import cv2
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互后端，加速渲染
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from math import ceil

    assert end_index > start_index, "end_index 需大于 start_index"
    num_frames = end_index - start_index

    # 辅助函数：提取图像
    def extract_images(item):
        imgs = []
        if 'images' in item and isinstance(item['images'], (list, tuple)) and len(item['images']) > 0:
            for im in item['images']:
                if im is None:
                    continue
                arr = im.cpu().numpy().squeeze() if hasattr(im, 'cpu') else np.asarray(im)
                if arr.ndim == 3 and arr.shape[0] in (1, 3):  # CHW -> HWC
                    arr = (np.transpose(arr, (1, 2, 0)) * 255).astype(np.uint8)
                if arr.ndim == 3 and arr.shape[2] in (1, 3):
                    imgs.append(arr if arr.shape[2] == 3 else np.repeat(arr, 3, axis=2))
        else:
            for k, v in item.items():
                if k in ('action', 'actions', 'state', 'observation', 'metadata', 'timestep', 'timestamp', 'frame_index', 'episode_index', 'index', 'task', 'task_index'):
                    continue
                if v is None:
                    continue
                arr = v.cpu().numpy().squeeze() if hasattr(v, 'cpu') else np.asarray(v)
                if arr is None:
                    continue
                if arr.ndim == 3 and arr.shape[0] in (1, 3):
                    arr = (np.transpose(arr, (1, 2, 0)) * 255).astype(np.uint8)
                if arr.ndim == 3 and arr.shape[2] in (1, 3):
                    imgs.append(arr if arr.shape[2] == 3 else np.repeat(arr, 3, axis=2))
        return imgs

    # 辅助函数：提取动作向量
    def extract_action_vec(item):
        act = item.get('action', None)
        if act is None:
            act = item.get('actions', None)
        if act is None:
            return None
        act_np = act.cpu().numpy() if hasattr(act, 'cpu') else np.asarray(act)
        if act_np.ndim == 2:
            return act_np[0]
        if act_np.ndim == 1:
            return act_np
        return None

    # 辅助函数：拼接图像
    def concat_images(imgs):
        if len(imgs) == 0:
            return None
        if len(imgs) == 1:
            return imgs[0]
        min_h = min(img.shape[0] for img in imgs)
        resized = []
        for img in imgs:
            if img.shape[0] != min_h:
                scale = min_h / img.shape[0]
                resized.append(cv2.resize(img, (int(img.shape[1] * scale), min_h)))
            else:
                resized.append(img)
        return np.concatenate(resized, axis=1)

    logging.info(f'开始预加载 {num_frames} 帧数据...')
    
    # ========== 一次性预加载所有数据 ==========
    all_images = []
    all_actions = []
    for offset in tqdm(range(0, num_frames, frame_interval), desc="预加载数据"):
        item = ds[start_index + offset]
        # 提取并拼接图像
        imgs = extract_images(item)
        concat_img = concat_images(imgs)
        all_images.append(concat_img)
        # 提取动作
        vec = extract_action_vec(item)
        all_actions.append(vec)
    
    if all(img is None for img in all_images):
        raise ValueError("未从数据集中解析到图像通道。请确认数据项包含 'images' 或图像键。")
    logging.info('预加载数据完成')

    # 确定动作维度
    first_act_vec = next((v for v in all_actions if v is not None), None)
    action_dims = 0 if first_act_vec is None else min(max_action_dims, int(first_act_vec.shape[0]))
    
    # 构建动作数组
    actions_all = None
    if action_dims > 0:
        series = []
        for vec in all_actions:
            if vec is None:
                vec = np.full((action_dims,), np.nan, dtype=float)
            else:
                vec = np.asarray(vec).astype(float)
                if vec.shape[0] < action_dims:
                    pad = np.full((action_dims - vec.shape[0],), np.nan, dtype=float)
                    vec = np.concatenate([vec, pad], axis=0)
                elif vec.shape[0] > action_dims:
                    vec = vec[:action_dims]
            series.append(vec)
        actions_all = np.stack(series, axis=0)
    logging.info('构建动作数组完成')

    # ========== 创建画布（只创建一次） ==========
    draw_num_frames = len(all_images)
    if action_dims > 0:
        action_cols = 4
        action_rows = int(ceil(action_dims / action_cols))
    else:
        action_cols = 4
        action_rows = 0

    total_rows = 1 + action_rows
    height_ratios = [2] + [1] * action_rows
    fig = plt.figure(figsize=(12, 6 + 2.0 * action_rows))
    gs = gridspec.GridSpec(total_rows, action_cols, height_ratios=height_ratios)
    ax_img = fig.add_subplot(gs[0, :])
    ax_action_list = []
    if action_rows > 0:
        for d in range(action_dims):
            r = 1 + d // action_cols
            c = d % action_cols
            ax_action_list.append(fig.add_subplot(gs[r, c]))

    # 初始化图像显示（使用 set_data 更新，避免每帧 clear）
    first_valid_img = next((img for img in all_images if img is not None), None)
    if first_valid_img is None:
        raise ValueError("没有有效图像")
    img_artist = ax_img.imshow(first_valid_img)
    ax_img.axis('off')

    # 初始化动作曲线和当前点
    point_artists = []
    if action_dims > 0 and actions_all is not None:
        for d in range(action_dims):
            ax = ax_action_list[d]
            ax.plot(actions_all[:, d], color='C0', linewidth=1.0)
            ax.set_xlim(0, max(1, draw_num_frames - 1))
            finite_mask = np.isfinite(actions_all[:, d])
            if finite_mask.any():
                ymin = float(np.nanmin(actions_all[:, d]))
                ymax = float(np.nanmax(actions_all[:, d]))
            else:
                ymin, ymax = -1.0, 1.0
            if ymin == ymax:
                ymin -= 1.0
                ymax += 1.0
            ax.set_ylim(ymin, ymax)
            ax.set_title(f'action dim {d}')
            # 预创建点对象（初始不可见）
            point, = ax.plot([], [], marker='o', markersize=7, color='C3', linewidth=0)
            point_artists.append(point)

    fig.tight_layout()
    fig.canvas.draw()
    logging.info('创建画布完成')

    # 获取画布尺寸并创建视频写入器
    w, h = fig.canvas.get_width_height()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    logging.info('开始写入视频帧...')

    # ========== 渲染循环（使用 set_data 更新，避免重绘） ==========
    
    for offset in tqdm(range(0, draw_num_frames, frame_interval), desc="渲染视频"):
        concat_img = all_images[offset]
        if concat_img is None:
            continue
        
        # 更新图像（使用 set_data 而非 clear + imshow）
        # 如果图像尺寸变化，需要重新设置
        if concat_img.shape[:2] != (img_artist.get_array().shape[0], img_artist.get_array().shape[1]):
            ax_img.clear()
            img_artist = ax_img.imshow(concat_img)
            ax_img.axis('off')
        else:
            img_artist.set_data(concat_img)
        
        # 更新当前帧的点（使用 set_data 而非 remove + plot）
        if action_dims > 0 and actions_all is not None:
            for d in range(action_dims):
                if np.isfinite(actions_all[offset, d]):
                    point_artists[d].set_data([offset], [actions_all[offset, d]])
                else:
                    point_artists[d].set_data([], [])

        # 只重绘变化的部分
        ax_img.draw_artist(img_artist)
        for pa in point_artists:
            pa.axes.draw_artist(pa)
        
        # 抓帧
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = buf.reshape((h, w, 4))
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        writer.write(frame_bgr)

    logging.info('视频写入完成')
    writer.release()
    plt.close(fig)

def _export_single_trajectory(args):
    """单个轨迹视频导出的包装函数，用于并行处理。"""
    (dataset_path, control_type, resolution, chunk_size, frame_interval,
     env_names, embodiment, skip_images, state_cache_size, verbose,
     episode_idx, out_path, fps, max_action_dims, video_frame_interval) = args
    
    try:
        # 每个进程创建独立的 reader 实例
        reader = RoboChallengeDatasetReader(
            dataset_path=dataset_path,
            control_type=control_type,
            resolution=resolution,
            chunk_size=chunk_size,
            frame_interval=frame_interval,
            env_names=env_names,
            embodiment=embodiment,
            skip_images=skip_images,
            pad_action_and_proprio=False,
            verbose=False,
            state_cache_size=state_cache_size,
        )
        
        if episode_idx >= len(reader._frame_ranges):
            print(f"Episode {episode_idx} out of range (total: {len(reader._frame_ranges)})")
            return None
        
        start_index, end_index = reader._frame_ranges[episode_idx]
        print(f"[Episode {episode_idx}] start_index: {start_index}, end_index: {end_index}, output: {out_path}")
        
        export_trajectory_video(
            reader,
            start_index=start_index,
            end_index=end_index,
            out_path=out_path,
            fps=fps,
            max_action_dims=max_action_dims,
            frame_interval=video_frame_interval
        )
        return out_path
    except Exception as e:
        print(f"[Episode {episode_idx}] Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def visualize_dataset(reader, frame_interval=2, episode_indices=None, num_workers=4, 
                      output_dir='outputs', fps=5, max_action_dims=14):
    """并行生成多组轨迹视频。
    
    Args:
        reader: RoboChallengeDatasetReader 实例
        frame_interval: 视频帧间隔
        episode_indices: 要生成视频的 episode 索引列表，None 表示使用默认值 [423]
        num_workers: 并行进程数
        output_dir: 输出目录
        fps: 视频帧率
        max_action_dims: 最大动作维度可视化数
    """
    import multiprocessing as mp
    from pathlib import Path
    
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 默认 episode 索引
    if episode_indices is None:
        episode_indices = [423]
    
    # 构建参数列表
    tasks = []
    for ep_idx in episode_indices:
        out_path = f'{output_dir}/{reader._tasks[0]["task_name"]}_ep{ep_idx}.mp4'
        args = (
            str(reader.dataset_path),
            reader.control_type,
            reader.resolution,
            reader.chunk_size,
            reader.frame_interval,
            [reader._tasks[0]["task_name"]],
            reader.target_embodiment,
            reader.skip_images,
            reader.state_cache_size,
            reader.verbose,
            ep_idx,
            out_path,
            fps,
            max_action_dims,
            frame_interval,
        )
        tasks.append(args)
    
    print(f"准备并行生成 {len(tasks)} 个视频，使用 {num_workers} 个进程...")
    start_time = time.time()
    
    if num_workers <= 1 or len(tasks) == 1:
        # 单进程模式
        results = [_export_single_trajectory(task) for task in tasks]
    else:
        # 多进程并行
        # 使用 spawn 方法以避免 fork 在某些情况下的问题
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=min(num_workers, len(tasks))) as pool:
            results = pool.map(_export_single_trajectory, tasks)
    
    end_time = time.time()
    
    # 统计结果
    success_count = sum(1 for r in results if r is not None)
    print(f"\n完成！成功生成 {success_count}/{len(tasks)} 个视频")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    
    return results


def visualize_dataset_legacy(reader, frame_interval=2):
    """旧版单视频生成函数（保留兼容性）。"""
    start_index, end_index = reader._frame_ranges[423]
    print(f"start_index: {start_index}, end_index: {end_index}")
    start_time = time.time()
    export_trajectory_video(reader,
        start_index=start_index,
        end_index=end_index,
        out_path='outputs/robochallenge_trajectory.mp4',
        fps=5,
        max_action_dims=14,
        frame_interval=frame_interval)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

def main():
    """
    Example 1 (单个视频):
        export PYTHONPATH=$PWD
        export DATA_DIR=$PWD/data
        python olmo/data/vla/rc_reader.py \
            --mode visualize \
            --dataset_path data/RoboChallenge \
            --env_name hang_toothbrush_cup \
            --embodiment UR5

    Example 2 (并行生成多个连续视频):
        export PYTHONPATH=$PWD
        export DATA_DIR=$PWD/data
        python olmo/data/vla/rc_reader.py \
            --mode visualize \
            --dataset_path data/RoboChallenge \
            --env_name turn_on_faucet \
            --embodiment ALOHA \
            --episode_indices $(seq 0 7) \
            --num_workers 8 \
            --output_dir outputs/videos
    
    Example 3 (随机选择多个视频):
        export PYTHONPATH=$PWD
        export DATA_DIR=$PWD/data
        python olmo/data/vla/rc_reader.py \
            --mode visualize \
            --dataset_path data/RoboChallenge \
            --env_name turn_on_faucet \
            --embodiment ALOHA \
            --random_episodes 8 \
            --num_workers 8 \
            --output_dir outputs/videos
    """
    parser = argparse.ArgumentParser(description="RoboChallenge Dataset Reader")
    parser.add_argument("--mode", choices=["read", "stats", "visualize"], required=True, help="read: sample data; stats: compute statistics; visualize: export trajectory video")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to RoboChallenge dataset root (task folder or root of tasks)")
    parser.add_argument(
        "--embodiment",
        type=str,
        choices=list(RoboChallengeDatasetReader.EMBODIMENT_CONFIGS.keys()),
        help="Embodiment to include",
        required=True
    )
    parser.add_argument("--env_name", type=str, nargs="+", help="Task name(s) to include")
    parser.add_argument("--control_type", type=str, choices=list(RoboChallengeDatasetReader.CONTROL_CHOICES)+[None,''], default=None)
    parser.add_argument("--resolution", type=int, nargs=2, default=None, help="Optional resize (width height), default uses video resolution")
    parser.add_argument("--chunk_size", type=int, default=8, help="Number of future actions to return")
    parser.add_argument("--frame_interval", type=int, default=1, help="Stride when sampling frames for read mode")
    parser.add_argument("--output_path", type=str, default="./robochallenge_stats.json", help="Where to save stats (stats mode)")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to print in read mode")
    parser.add_argument("--skip_images", action="store_true", help="Do not decode images (faster if only controls are needed)")
    parser.add_argument("--verbose", action="store_true", help="Print progress information")
    parser.add_argument("--state_cache_size", type=int, default=64, help="Number of episodes to keep cached for states (0 to disable)")
    parser.add_argument("--only_save_overall_stats", action="store_true", help="Only save overall stats (stats mode)")
    # 新增并行视频生成参数
    parser.add_argument("--episode_indices", type=int, nargs="+", default=None, 
                        help="Episode indices to visualize (visualize mode). Default: [423]")
    parser.add_argument("--random_episodes", type=int, default=None,
                        help="Randomly select N episodes to visualize (visualize mode). Overrides --episode_indices if set.")
    parser.add_argument("--num_workers", type=int, default=4, 
                        help="Number of parallel workers for video generation (visualize mode)")
    parser.add_argument("--output_dir", type=str, default="outputs", 
                        help="Output directory for generated videos (visualize mode)")
    parser.add_argument("--video_fps", type=int, default=5, 
                        help="FPS for output video (visualize mode)")
    parser.add_argument("--video_frame_interval", type=int, default=2, 
                        help="Frame interval for video rendering (visualize mode)")
    parser.add_argument("--max_action_dims", type=int, default=14, 
                        help="Max action dimensions to visualize (visualize mode)")
    args = parser.parse_args()

    reader = RoboChallengeDatasetReader(
        dataset_path=args.dataset_path,
        control_type=args.control_type,
        resolution=tuple(args.resolution) if args.resolution else None,
        chunk_size=args.chunk_size,
        frame_interval=args.frame_interval,
        env_names=args.env_name,
        embodiment=args.embodiment,
        skip_images=args.skip_images,
        pad_action_and_proprio=False if args.mode == "visualize" else True,
        verbose=args.verbose,
        state_cache_size=args.state_cache_size,
    )

    if args.mode == "visualize":
        # 确定 episode indices
        episode_indices = args.episode_indices
        if args.random_episodes is not None:
            total_episodes = len(reader._episodes)
            if total_episodes == 0:
                print("No episodes found in dataset.")
                return
            n = min(args.random_episodes, total_episodes)
            episode_indices = random.sample(range(total_episodes), n)
            print(f"随机选择了 {n} 个 episode: {episode_indices}")
        
        visualize_dataset(
            reader,
            frame_interval=args.video_frame_interval,
            episode_indices=episode_indices,
            num_workers=args.num_workers,
            output_dir=args.output_dir,
            fps=args.video_fps,
            max_action_dims=args.max_action_dims,
        )
        return

    if args.mode == "read":
        print(f"Total frames: {len(reader)}")
        if len(reader) == 0:
            print("No data found.")
            return
        sample_indices = random.sample(range(len(reader)), min(args.num_samples, len(reader)))
        print(f"Sampling {len(sample_indices)} items...")
        for idx in tqdm(sample_indices, desc="Reading samples"):
            data = reader[idx]
            print(f"\nSample {idx}:")
            print(f"  Question: {data['question']}")
            print(f"  Frame index: {data['metadata']['frame_index']}")
            print(f"  Episode index: {data['metadata']['episode_index']}")
            print(f"  Embodiment: {data['metadata'].get('embodiment')}")
            print(f"  Images: {len(data['images'])}")
            if data["images"]:
                print(f"  Image shape: {data['images'][0].shape}")
            print(f"  Action shape: {data['action'].shape}")
            print(f"  State shape: {data['proprio'].shape}")

    elif args.mode == "stats":
        stats = reader.compute_stats()
        if args.only_save_overall_stats:
            stats = stats["overall_stats"]
        with open(args.output_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Stats saved to {args.output_path}")
        if not args.only_save_overall_stats:
            for emb, payload in stats["embodiments"].items():
                print(f"\n=== {emb} ===")
                print(f"Tasks: {', '.join(payload['environments'])}")
                print(f"Action dim: {len(payload['stats']['action']['mean'])}")
                print(f"State dim: {len(payload['stats']['state']['mean'])}")


if __name__ == "__main__":
    main()
