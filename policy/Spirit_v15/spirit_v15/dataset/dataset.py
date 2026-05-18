# ==============================================================================
# Attribution
# ------------------------------------------------------------------------------
# Released by Spirit AI Team.
# ==============================================================================

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torchvision.transforms import Resize

from .transforms import process_images, ColorJitter


@dataclass
class DataConfig:
    data_root: str = ""
    action_horizon: int = 60
    state_history: int = 1
    chunk_size: int = 60


class RoboChallengeDataset(torch.utils.data.Dataset):
    """RoboChallenge Dataset.

    Supports the original single-arm Spirit finetune format and a dual-arm EE format.

    directory structure:
        {data_root}/
        ├── task_desc.json
        ├── meta/task_info.json
        └── data/
            └── episode_{XXXXXX}/
                ├── meta/episode_meta.json
                ├── states/states.jsonl
                └── videos/
                    ├── handeye_realsense_rgb.mp4
                    ├── main_realsense_rgb.mp4
                    └── side_realsense_rgb.mp4

    state encode:
        single arm raw: ee_positions[7] = [x, y, z, qx, qy, qz, qw] + gripper_width[1]
        dual arm raw: left/right ee_positions[7] + left/right gripper_width[1]
        internal: [left xyz, left rotvec, left gripper, right xyz, right rotvec, right gripper]
        single-arm data is zero-padded on the right arm branch.

    action encode(delta):
        delta_xyz[3] = action_xyz - state_xyz
        delta_rot[3] = R_action * R_state^{-1}  (rotation manifold)
        gripper[1] = action_gripper  (absolute)
        → zero-pad to 14D
        → pad to action_horizon with action_is_pad=True

    Memory Optimization while training:
        Store state data as compact NumPy arrays instead of Python dicts for memory optimization.
        _state_data: [N_total, D] float64  (state payload + timestamp)
        _ep_offsets: [num_episodes] int64   (offset of each episode in _state_data)
        _ep_lengths: [num_episodes] int64   (frames per episode)
    """

    _SINGLE_EE_SLICE = slice(0, 7)
    _SINGLE_GRIPPER_IDX = 7
    _SINGLE_TS_IDX = 8

    _LEFT_EE_SLICE = slice(0, 7)
    _LEFT_GRIPPER_IDX = 7
    _RIGHT_EE_SLICE = slice(8, 15)
    _RIGHT_GRIPPER_IDX = 15
    _DUAL_TS_IDX = 16

    _DEFAULT_CAMERA_VIDEOS = {
        "Franka": {
            "observation.images.cam_high": "main_realsense_rgb",
            "observation.images.cam_left_wrist": "handeye_realsense_rgb",
            "observation.images.cam_right_wrist": "side_realsense_rgb",
        },
        "aloha": {
            "observation.images.cam_high": "head_camera_rgb",
            "observation.images.cam_left_wrist": "left_camera_rgb",
            "observation.images.cam_right_wrist": "right_camera_rgb",
        },
    }

    def __init__(self, config):
        self.data_root = Path(config.data_root)
        self.action_horizon = config.action_horizon
        self.chunk_size = config.chunk_size
        self.state_history = config.state_history

        task_info = self._load_task_info()
        self.task_name = task_info["task_name"]
        self.task_prompt = task_info["task_prompt"]
        self.robot_type = task_info["robot_type"]
        self.state_encoding = task_info["state_encoding"]
        self.fps = float(task_info["fps"])
        self.camera_videos = task_info["camera_videos"]

        self._state_data, self._ep_offsets, self._ep_lengths = self._load_all_states()
        self._episode_prompts = self._load_episode_prompts()
        self.index: List[Tuple[int, int]] = self._build_index()

        self._resize = Resize((240, 320), antialias=True)
        self._jitter = ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.1, p=0.5)

    def _get_state_row(self, episode_idx: int, frame_idx: int) -> np.ndarray:
        offset = self._ep_offsets[episode_idx] + frame_idx
        return self._state_data[offset]

    def _get_ep_length(self, episode_idx: int) -> int:
        return int(self._ep_lengths[episode_idx])

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            {
                "observation.images.cam_high": Tensor[3, H, W],
                "observation.images.cam_left_wrist": Tensor[3, H, W],
                "observation.images.cam_right_wrist": Tensor[3, H, W],
                "observation.state": Tensor[1, 14],
                "action": Tensor[60, 14],
                "action_mask": Tensor[60, 14],  # bool
                "task": str,
                "robot_type": str,
            }
        """
        episode_idx, frame_idx = self.index[idx]

        images = self._load_images(episode_idx, frame_idx)
        images = process_images(images, self._resize, self._jitter, augment=True)
        state, _ = self._encode_state(episode_idx, frame_idx)
        actions, action_mask, _ = self._encode_actions(
            episode_idx, frame_idx
        )

        return {
            **images,
            "observation.state": state,
            "action": actions,
            "action_mask": action_mask,
            "task": self._episode_prompts[episode_idx],
            "robot_type": self.robot_type,
        }

    def get_lowdim_item(self, idx: int) -> dict:
        episode_idx, frame_idx = self.index[idx]
        state, _ = self._encode_state(episode_idx, frame_idx)
        actions, action_mask, _ = self._encode_actions(
            episode_idx, frame_idx
        )
        return {
            "observation.state": state,
            "action": actions,
            "action_mask": action_mask,
        }

    def _load_task_info(self) -> Dict[str, Any]:
        task_file = self.data_root / "meta" / "task_info.json"
        if not task_file.exists():
            raise FileNotFoundError(
                f"Task info file not found: {task_file}\n"
                f"Expected directory structure: {self.data_root}/meta/task_info.json"
            )
        with open(task_file) as f:
            data = json.load(f)

        try:
            task_name = data["task_desc"]["task_name"]
            task_prompt = data["task_desc"]["prompt"]
        except KeyError as e:
            raise KeyError(
                f"Missing required field in {task_file}: {e}\n"
                f"Expected structure: {{'task_desc': {{'task_name': ..., 'prompt': ...}}}}"
            )
        robot_type = data.get("robot_type", "Franka")
        state_encoding = data.get("state_encoding", "single_arm_ee")
        fps = data.get("fps", 30.0)
        camera_videos = data.get("camera_videos")
        if camera_videos is None:
            camera_videos = self._DEFAULT_CAMERA_VIDEOS.get(robot_type, self._DEFAULT_CAMERA_VIDEOS["Franka"])

        return {
            "task_name": task_name,
            "task_prompt": task_prompt,
            "robot_type": robot_type,
            "state_encoding": state_encoding,
            "fps": fps,
            "camera_videos": camera_videos,
        }

    @staticmethod
    def _as_scalar_or_singleton_list(value: Any) -> List[float]:
        if isinstance(value, list):
            return value
        return [float(value)]

    def _build_state_row(self, raw_state: Dict[str, Any]) -> List[float]:
        if self.state_encoding == "dual_arm_ee":
            return (
                raw_state["left_ee_positions"]
                + self._as_scalar_or_singleton_list(raw_state["left_gripper_width"])
                + raw_state["right_ee_positions"]
                + self._as_scalar_or_singleton_list(raw_state["right_gripper_width"])
                + [raw_state["timestamp"]]
            )

        return (
            raw_state["ee_positions"]
            + self._as_scalar_or_singleton_list(raw_state["gripper_width"])
            + [raw_state["timestamp"]]
        )

    def _load_all_states(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        data_dir = self.data_root / "data"
        episodes = sorted(data_dir.glob("episode_*"))

        all_rows = []
        ep_lengths = []
        for ep_dir in episodes:
            states_file = ep_dir / "states" / "states.jsonl"
            ep_rows = []
            with open(states_file) as f:
                for line in f:
                    d = json.loads(line)
                    row = self._build_state_row(d)
                    ep_rows.append(row)
            all_rows.extend(ep_rows)
            ep_lengths.append(len(ep_rows))

        state_data = np.array(all_rows, dtype=np.float64)
        ep_lengths = np.array(ep_lengths, dtype=np.int64)
        ep_offsets = np.zeros(len(ep_lengths), dtype=np.int64)
        ep_offsets[1:] = np.cumsum(ep_lengths[:-1])

        return state_data, ep_offsets, ep_lengths

    def _load_episode_prompts(self) -> List[str]:
        prompts: List[str] = []
        data_dir = self.data_root / "data"
        for ep_dir in sorted(data_dir.glob("episode_*")):
            prompt = self.task_prompt
            ep_meta_file = ep_dir / "meta" / "episode_meta.json"
            if ep_meta_file.exists():
                with open(ep_meta_file) as f:
                    ep_meta = json.load(f)
                prompt = ep_meta.get("prompt", prompt)
            prompts.append(prompt)
        return prompts

    def _build_index(self) -> List[Tuple[int, int]]:
        index = []
        for ep_idx in range(len(self._ep_lengths)):
            ep_len = int(self._ep_lengths[ep_idx])
            for frame_idx in range(ep_len - 2):
                index.append((ep_idx, frame_idx))
        return index

    def _load_images(self, episode_idx: int, frame_idx: int) -> dict:
        ep_dir = self.data_root / "data" / f"episode_{episode_idx:06d}"
        video_dir = ep_dir / "videos"

        images = {}
        for key, video_name in self.camera_videos.items():
            video_path = video_dir / f"{video_name}.mp4"
            frame = self._decode_video_frame(video_path, frame_idx)
            images[key] = frame

        return images

    def _decode_video_frame(self, video_path: Path, frame_idx: int) -> torch.Tensor:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Frame {frame_idx} 读取失败 {video_path}")

        # cv2: HWC BGR → RGB CHW
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).contiguous()
        return frame_tensor.float() / 255.0

    def _encode_state(
        self, episode_idx: int, frame_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        state_idx = max(0, frame_idx - 1)
        row = self._get_state_row(episode_idx, state_idx)
        if self.state_encoding == "dual_arm_ee":
            left_state = self._encode_arm_state(row[self._LEFT_EE_SLICE], row[self._LEFT_GRIPPER_IDX])
            right_state = self._encode_arm_state(row[self._RIGHT_EE_SLICE], row[self._RIGHT_GRIPPER_IDX])
            state_14d = np.concatenate([left_state, right_state])
            state_mask = np.ones(14, dtype=bool)
        else:
            state_7d = self._encode_arm_state(row[self._SINGLE_EE_SLICE], row[self._SINGLE_GRIPPER_IDX])
            state_14d = np.pad(state_7d, (0, 7), mode="constant")
            state_mask = np.zeros(14, dtype=bool)
            state_mask[:7] = True

        return (
            torch.from_numpy(state_14d).float().unsqueeze(0),
            torch.from_numpy(state_mask).unsqueeze(0),
        )

    @staticmethod
    def _encode_arm_state(ee_pos: np.ndarray, gripper_value: float) -> np.ndarray:
        xyz = ee_pos[:3]
        quat = ee_pos[3:]
        rotvec = Rotation.from_quat(quat).as_rotvec()
        gripper = np.array([gripper_value])
        return np.concatenate([xyz, rotvec, gripper])

    @staticmethod
    def _encode_arm_action(curr_pose: np.ndarray, curr_gripper: float, target_pose: np.ndarray, target_gripper: float) -> np.ndarray:
        curr_xyz = curr_pose[:3]
        curr_rot = Rotation.from_quat(curr_pose[3:])
        target_xyz = target_pose[:3]
        target_rot = Rotation.from_quat(target_pose[3:])

        delta_xyz = target_xyz - curr_xyz
        delta_rotvec = (target_rot * curr_rot.inv()).as_rotvec()
        return np.concatenate([delta_xyz, delta_rotvec, np.array([target_gripper])])

    def _encode_actions(
        self, episode_idx: int, frame_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ref_idx = max(0, frame_idx - 1)
        ep_len = self._get_ep_length(episode_idx)

        curr_row = self._get_state_row(episode_idx, ref_idx)

        actions_list = []
        action_is_pad_list = []
        last_valid_action = None
        num_steps = min(self.chunk_size, self.action_horizon)
        for i in range(num_steps):
            target_idx = frame_idx + i

            if target_idx < ep_len:
                target_row = self._get_state_row(episode_idx, target_idx)
            else:
                target_row = self._get_state_row(episode_idx, ep_len - 1)

            if self.state_encoding == "dual_arm_ee":
                left_action = self._encode_arm_action(
                    curr_row[self._LEFT_EE_SLICE],
                    curr_row[self._LEFT_GRIPPER_IDX],
                    target_row[self._LEFT_EE_SLICE],
                    target_row[self._LEFT_GRIPPER_IDX],
                )
                right_action = self._encode_arm_action(
                    curr_row[self._RIGHT_EE_SLICE],
                    curr_row[self._RIGHT_GRIPPER_IDX],
                    target_row[self._RIGHT_EE_SLICE],
                    target_row[self._RIGHT_GRIPPER_IDX],
                )
                action_14d = np.concatenate([left_action, right_action])
            else:
                action_7d = self._encode_arm_action(
                    curr_row[self._SINGLE_EE_SLICE],
                    curr_row[self._SINGLE_GRIPPER_IDX],
                    target_row[self._SINGLE_EE_SLICE],
                    target_row[self._SINGLE_GRIPPER_IDX],
                )
                action_14d = np.pad(action_7d, (0, 7), mode="constant")

            if target_idx < ep_len:
                last_valid_action = action_14d
                actions_list.append(action_14d)
                action_is_pad_list.append(False)
            else:
                if last_valid_action is None:
                    raise ValueError(
                        f"No valid future action found for episode {episode_idx}, frame {frame_idx}"
                    )
                actions_list.append(last_valid_action.copy())
                action_is_pad_list.append(True)

        num_valid = len(actions_list)
        actions = np.zeros((self.action_horizon, 14), dtype=np.float32)
        actions[:num_valid] = np.array(actions_list)

        action_mask = np.zeros((self.action_horizon, 14), dtype=bool)
        if self.state_encoding == "dual_arm_ee":
            action_mask[:num_valid, :] = True
        else:
            action_mask[:num_valid, :7] = True

        action_is_pad = np.ones(self.action_horizon, dtype=bool)
        action_is_pad[:num_valid] = np.array(action_is_pad_list, dtype=bool)

        return (
            torch.from_numpy(actions),
            torch.from_numpy(action_mask),
            torch.from_numpy(action_is_pad),
        )

    @staticmethod
    def collate_fn(batch: List[dict]) -> dict:
        result = {}
        for key in batch[0]:
            values = [b[key] for b in batch]
            if isinstance(values[0], torch.Tensor):
                result[key] = torch.stack(values)
            else:
                result[key] = values
        return result

    @staticmethod
    def collate_lowdim_fn(batch: List[dict]) -> dict:
        return {
            "observation.state": torch.stack([b["observation.state"] for b in batch]),
            "action": torch.stack([b["action"] for b in batch]),
            "action_mask": torch.stack([b["action_mask"] for b in batch]),
        }
