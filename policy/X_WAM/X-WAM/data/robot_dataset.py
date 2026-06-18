import json
import time
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import torchvision.transforms.functional as TF
from decord import VideoReader

from data.augmentation import VideoAugmentation


class _ClipIndex(Sequence[tuple[str, int]]):
    """Compact clip index that maps flat sample ids to (episode_key, start_frame)."""

    def __init__(self, metadata: dict[str, int], required_span: int):
        self.episode_keys: list[str] = []
        cumulative_counts: list[int] = []

        total_clips = 0
        for episode_key, length in metadata.items():
            clip_count = int(length) - int(required_span)
            if clip_count <= 0:
                continue
            total_clips += clip_count
            self.episode_keys.append(episode_key)
            cumulative_counts.append(total_clips)

        self.total_clips = total_clips
        self.cumulative_counts = np.asarray(cumulative_counts, dtype=np.int64)

    def __len__(self) -> int:
        return self.total_clips

    def __getitem__(self, idx: int | slice) -> tuple[str, int] | list[tuple[str, int]]:
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self.total_clips)
            return [self[i] for i in range(start, stop, step)]

        idx = int(idx)
        if idx < 0:
            idx += self.total_clips
        if idx < 0 or idx >= self.total_clips:
            raise IndexError(f"Index {idx} is out of range for clip index of length {self.total_clips}.")

        episode_pos = int(np.searchsorted(self.cumulative_counts, idx, side="right"))
        previous_cumulative = 0 if episode_pos == 0 else int(self.cumulative_counts[episode_pos - 1])
        start_frame = idx - previous_cumulative
        return self.episode_keys[episode_pos], start_frame


class RobotDataset(Dataset):
    """Dataset for single pretrain dataset root with JSON + RGB/Depth videos."""

    _PROPRIO_COMPONENTS: tuple[tuple[str, int], ...] = (
        ("left_ee_pos", 3),
        ("left_ee_rotm", 4),
        ("left_gripper_pos", 1),
        ("right_ee_pos", 3),
        ("right_ee_rotm", 4),
        ("right_gripper_pos", 1),
    )
    _PROPRIO_DIM: int = sum(d for _, d in _PROPRIO_COMPONENTS)

    _ACTION_COMPONENTS: tuple[tuple[str, int], ...] = (
        ("left_ee_pos", 3),
        ("left_ee_rotm", 3),
        ("left_gripper_pos", 1),
        ("right_ee_pos", 3),
        ("right_ee_rotm", 3),
        ("right_gripper_pos", 1),
    )
    _ACTION_DIM: int = sum(d for _, d in _ACTION_COMPONENTS)

    _ARMS: tuple[str, str] = ("left", "right")

    def __init__(
        self,
        dataset_path: str,
        sequence_length: int,
        frame_skip: int,
        video_size: tuple[int, int],
        action_skip: int = 1,
        augment: bool = True,
        crop_ratio: float = 0.95,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.05,
        inverse_gripper: bool = False,
        normalize_depths_per_view: bool = True,
        shuffle_view_order: bool = True,
        statistics: dict[str, dict[str, list]] | None = None,
    ):
        if sequence_length <= 1:
            raise ValueError(f"sequence_length must be > 1, got {sequence_length}.")
        if frame_skip <= 0:
            raise ValueError(f"frame_skip must be > 0, got {frame_skip}.")
        if action_skip <= 0:
            raise ValueError(f"action_skip must be > 0, got {action_skip}.")
        if frame_skip % action_skip != 0:
            raise ValueError(
                f"frame_skip must be divisible by action_skip to keep a fixed number of actions per frame: "
                f"got frame_skip={frame_skip}, action_skip={action_skip}."
            )

        self.dataset_root = Path(dataset_path)
        self._configure_paths()
        self.sequence_length = sequence_length
        self.frame_skip = frame_skip
        self.action_skip = action_skip
        self.action_num = self.frame_skip // self.action_skip
        self.action_span = (self.sequence_length - 1) * self.frame_skip
        self.video_size = self._validate_video_size(video_size)
        self.augment = augment
        self.inverse_gripper = bool(inverse_gripper)
        self.normalize_depths_per_view = bool(normalize_depths_per_view)
        self.augmentation = VideoAugmentation(
            crop_ratio=crop_ratio,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )
        self.shuffle_view_order = shuffle_view_order
        self.proprio_dim = self._PROPRIO_DIM
        self.action_dim = self._ACTION_DIM
        self.proprio_component_slices = self._build_component_slices(self._PROPRIO_COMPONENTS)
        self.action_component_slices = self._build_component_slices(self._ACTION_COMPONENTS)

        self._validate_dataset_dirs()
        self.quantile_stats = self._parse_quantile_stats(statistics)
        self.metadata_path = self.dataset_root / "metadata.json"
        if self.metadata_path.exists():
            self.metadata = self._load_metadata(self.metadata_path)
        else:
            self.metadata = self._build_metadata()
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2, sort_keys=True)

        self.data_list: Sequence[tuple[str, int]] = self._build_data_list()
        if len(self.data_list) == 0:
            raise ValueError(
                "No valid clips found. "
                f"Please check sequence_length={self.sequence_length}, frame_skip={self.frame_skip}, and dataset content."
            )

        print(f"Loaded {len(self.metadata)} episodes, built {len(self.data_list)} valid clips.")

    def _configure_paths(self) -> None:
        """Set data/video/depth roots. Subclasses may override for alternate layouts."""
        self.data_root = self.dataset_root / "data"
        self.video_root = self.dataset_root / "video"
        self.depth_root = self.dataset_root / "depth"

    def _iter_episode_json_paths(self) -> list[Path]:
        """All episode JSON paths used to build metadata (relative discovery under data_root)."""
        return sorted(self.data_root.glob("chunk-*/episode_*.json"))

    def _episode_key_from_json_path(self, json_path: Path) -> str:
        """Stable episode id string from a JSON path under this dataset layout."""
        rel = json_path.relative_to(self.data_root)
        return str(rel.with_suffix(""))

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        episode_key, start_frame = self.data_list[idx]
        episode = self._load_episode_json(episode_key)

        frame_ids = np.arange(
            start_frame,
            start_frame + self.sequence_length * self.frame_skip,
            self.frame_skip,
            dtype=np.int64,
        )

        use_raw = self._has_raw_actions(episode)
        if use_raw:
            action_ids = self._build_raw_action_ids(start_frame)
        else:
            action_ids = self._build_action_ids(start_frame)

        video, depths, fps, camera_type_mask = self._load_multiview_video(episode, frame_ids)
        if self.normalize_depths_per_view:
            depths = self._normalize_depths_per_view(depths)
        proprios, proprio_mask = self._build_proprio_tensor(episode, frame_ids, episode_key)
        if use_raw:
            actions, action_mask = self._build_raw_action_tensor(episode, action_ids, episode_key)
        else:
            actions, action_mask = self._build_delta_action_tensor(episode, action_ids, episode_key)
        prompt = self._sample_instruction(episode, episode_key)

        data = {
            "video": video,
            "depths": depths,
            "fps": fps,
            "proprios": proprios,
            "proprio_mask": proprio_mask,
            "actions": actions,
            "action_mask": action_mask,
            "camera_type_mask": camera_type_mask,
            "prompt": prompt,
            "episode_key": episode_key,
        }
        if self.augment:
            data = self.augmentation(data)
        return data

    def _validate_dataset_dirs(self) -> None:
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.dataset_root}")
        if not self.data_root.exists():
            raise FileNotFoundError(f"Missing data directory: {self.data_root}")
        if not self.video_root.exists():
            raise FileNotFoundError(f"Missing video directory: {self.video_root}")
        # if not self.depth_root.exists():
        #     raise FileNotFoundError(f"Missing depth directory: {self.depth_root}")

    def _load_metadata(self, metadata_path: Path) -> dict[str, int]:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        if not isinstance(metadata, dict):
            raise ValueError(f"Invalid metadata format in {metadata_path}, expected dict.")
        cleaned: dict[str, int] = {}
        for key, value in metadata.items():
            cleaned[str(key)] = int(value)
        return cleaned

    @staticmethod
    def _parse_quantile_stats(statistics: dict[str, dict[str, list]] | None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        if statistics is None:
            raise ValueError("statistics must be provided (dict with 'q01' and 'q99' keys).")

        q01_block = statistics.get("q01")
        q99_block = statistics.get("q99")
        if not isinstance(q01_block, dict) or not isinstance(q99_block, dict):
            raise ValueError("statistics must contain 'q01' and 'q99' dicts.")

        stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for key in q01_block:
            if key not in q99_block:
                raise ValueError(f"Key {key} found in q01 but not q99.")
            q01 = np.asarray(q01_block[key], dtype=np.float32).reshape(-1)
            q99 = np.asarray(q99_block[key], dtype=np.float32).reshape(-1)
            if q01.shape != q99.shape:
                raise ValueError(f"Shape mismatch for {key}: q01={q01.shape}, q99={q99.shape}.")
            stats[key] = (q01, q99)
        return stats

    @staticmethod
    def _build_component_slices(
        components: tuple[tuple[str, int], ...],
    ) -> dict[str, slice]:
        slices: dict[str, slice] = {}
        offset = 0
        for name, dim in components:
            slices[name] = slice(offset, offset + dim)
            offset += dim
        return slices

    def _build_metadata(self) -> dict[str, int]:
        json_paths = self._iter_episode_json_paths()
        if len(json_paths) == 0:
            raise ValueError(f"No episode json found under: {self.data_root}")

        metadata: dict[str, int] = {}
        with ThreadPoolExecutor(max_workers=32) as executor:
            iterator = executor.map(self._load_single_info, json_paths)
            for episode_key, num_frames in tqdm(iterator, total=len(json_paths), desc="Building metadata"):
                metadata[episode_key] = num_frames
        return metadata

    def _load_single_info(self, json_path: Path) -> tuple[str, int]:
        num_frames = self._extract_num_frames(json_path)
        episode_key = self._episode_key_from_json_path(json_path)
        return episode_key, num_frames

    @staticmethod
    def _extract_num_frames(json_path: Path) -> int:
        with open(json_path, "r", encoding="utf-8") as f:
            head = f.read(8192)
            match = re.search(r'"num_frames"\s*:\s*(\d+)', head)
            if match is not None:
                return int(match.group(1))

            # Fallback if num_frames not present in first bytes.
            f.seek(0)
            payload = json.load(f)
            if "num_frames" not in payload:
                raise ValueError(f"Missing num_frames in {json_path}")
            return int(payload["num_frames"])

    def _build_data_list(self) -> Sequence[tuple[str, int]]:
        required_span = (self.sequence_length - 1) * self.frame_skip
        return _ClipIndex(metadata=self.metadata, required_span=required_span)

    @staticmethod
    def _has_raw_actions(episode: dict[str, Any]) -> bool:
        actions_dict = episode.get("actions")
        return isinstance(actions_dict, dict) and "raw_actions" in actions_dict

    def _build_action_ids(self, start_frame: int) -> np.ndarray:
        # For each consecutive block of action_skip actions, keep the last one.
        return np.arange(
            start_frame + self.action_skip - 1,
            start_frame + self.action_span,
            self.action_skip,
            dtype=np.int64,
        )

    def _build_raw_action_ids(self, start_frame: int) -> np.ndarray:
        """Build action indices with step=1 (action_skip is bypassed for raw actions)."""
        return np.arange(
            start_frame,
            start_frame + self.action_span,
            dtype=np.int64,
        )

    def _episode_json_path(self, episode_key: str) -> Path:
        return self.data_root / f"{episode_key}.json"

    def _load_episode_json(self, episode_key: str) -> dict[str, Any]:
        json_path = self._episode_json_path(episode_key)
        if not json_path.exists():
            raise FileNotFoundError(f"Episode json not found: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"Episode payload is not a dict: {json_path}")
        return payload

    def _load_multiview_video(
        self,
        episode: dict[str, Any],
        frame_ids: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
        observations = episode.get("observations")
        if not isinstance(observations, dict) or len(observations) == 0:
            raise ValueError("Episode observations are missing or empty.")

        video_views: list[torch.Tensor] = []
        depth_views: list[torch.Tensor] = []
        fps_values: list[float] = []
        camera_types: list[int] = []

        for view_name in sorted(observations.keys()):
            obs = observations[view_name]
            if not isinstance(obs, dict):
                raise ValueError(f"Observation entry is invalid for view {view_name}.")

            if "rgb_path" not in obs or "depth_path" not in obs:
                raise ValueError(f"Missing rgb_path/depth_path in view {view_name}.")

            rgb_path = self._resolve_media_path(obs["rgb_path"])
            depth_path = self._resolve_media_path(obs["depth_path"])

            obs_start = int(obs.get("start", 0))
            obs_end = int(obs.get("end", obs_start + int(frame_ids[-1]) + 1))
            abs_frame_ids = frame_ids + obs_start
            if int(abs_frame_ids[-1]) >= obs_end:
                raise ValueError(
                    f"Requested frame index {int(abs_frame_ids[-1])} exceeds observation end {obs_end} for view {view_name}."
                )

            video_views.append(
                self._read_video_frames(
                    rgb_path,
                    abs_frame_ids,
                    video_size=self.video_size,
                    interpolation=TF.InterpolationMode.BILINEAR,
                    antialias=False,
                )
            )
            depth_views.append(
                self._read_video_frames(
                    depth_path,
                    abs_frame_ids,
                    video_size=self.video_size,
                    interpolation=TF.InterpolationMode.NEAREST_EXACT,
                    antialias=False,
                )
            )

            if "fps" not in obs:
                raise ValueError(f"Missing fps in view {view_name}.")
            fps_values.append(float(obs["fps"]) / self.frame_skip)

            cam_type = obs.get("type")
            if cam_type == "static":
                camera_types.append(0)
            elif cam_type == "dynamic":
                camera_types.append(1)
            else:
                raise ValueError(f"Unknown camera type {cam_type!r} for view {view_name}.")

        fps = fps_values[0]
        for value in fps_values[1:]:
            if abs(value - fps) > 1e-6:
                raise ValueError(f"Inconsistent fps across views: {fps_values}")

        video = torch.stack(video_views, dim=0)  # [V, T, C, H, W]
        depths = torch.stack(depth_views, dim=0)  # [V, T, C, H, W]
        camera_type_mask = torch.tensor(camera_types, dtype=torch.long)
        if self.shuffle_view_order:
            video, depths, camera_type_mask = self._shuffle_view_order(video, depths, camera_type_mask)
        return video, depths, fps, camera_type_mask

    @staticmethod
    def _shuffle_view_order(
        video: torch.Tensor,
        depths: torch.Tensor,
        camera_type_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_views = video.shape[0]
        if num_views <= 1:
            return video, depths, camera_type_mask

        view_perm = torch.randperm(num_views)
        return video[view_perm], depths[view_perm], camera_type_mask[view_perm]

    def _resolve_media_path(self, path_str: Any) -> Path:
        if not isinstance(path_str, str):
            raise ValueError(f"Media path must be string, got {type(path_str)}")
        path = Path(path_str)
        if not path.is_absolute():
            path = self.dataset_root / path
        if not path.exists():
            raise FileNotFoundError(f"Media file not found: {path}")
        return path

    @staticmethod
    def _validate_video_size(video_size: tuple[int, int] | None) -> tuple[int, int] | None:
        if video_size is None:
            return None

        height = int(video_size[0])
        width = int(video_size[1])
        if height <= 0 or width <= 0:
            raise ValueError(f"video_size values must be positive, got {(height, width)}.")
        return (height, width)

    @staticmethod
    def _read_video_frames(
        video_path: Path,
        frame_ids: np.ndarray,
        video_size: tuple[int, int] | None,
        interpolation: TF.InterpolationMode,
        antialias: bool,
    ) -> torch.Tensor:
        reader = VideoReader(str(video_path))
        try:
            array = reader.get_batch(frame_ids.tolist()).asnumpy()
        except Exception as exc:
            raise ValueError(f"Failed to decode frames from {video_path}: {exc}") from exc

        if array.ndim != 4:
            raise ValueError(f"Unexpected decoded batch shape {array.shape} from {video_path}")

        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        elif array.shape[-1] > 3:
            array = array[:, :, :, :3]

        if array.shape[-1] != 3:
            raise ValueError(f"Unexpected channel count {array.shape[-1]} from {video_path}")

        array = array.astype(np.float32)  # [T, H, W, C]
        array = array / 127.5 - 1.0
        array = np.transpose(array, (0, 3, 1, 2))  # [T, C, H, W]
        tensor = torch.from_numpy(array)
        if video_size is not None:
            tensor = TF.resize(
                tensor,
                size=list(video_size),
                interpolation=interpolation,
                antialias=antialias,
            )
        return tensor

    @staticmethod
    def _normalize_depths_per_view(depths: torch.Tensor) -> torch.Tensor:
        if depths.ndim != 5:
            raise ValueError(f"Expected depths with shape [V, T, C, H, W], got {tuple(depths.shape)}")

        # Compute one min/max per view over all frames to keep temporal scaling consistent per view.
        depth_min = depths.amin(dim=(1, 2, 3, 4), keepdim=True)
        depth_max = depths.amax(dim=(1, 2, 3, 4), keepdim=True)
        depth_range = depth_max - depth_min
        valid = depth_range > 1e-6
        safe_range = torch.where(valid, depth_range, torch.ones_like(depth_range))
        normalized = 2.0 * (depths - depth_min) / safe_range - 1.0
        return torch.where(valid, normalized, torch.zeros_like(normalized))

    @staticmethod
    def _to_rotmat(raw: Any) -> np.ndarray | None:
        """Convert raw rotation data to rotation matrices [T, 3, 3]."""
        if raw is None:
            return None
        arr = np.asarray(raw, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] == 3:
            return Rotation.from_rotvec(arr).as_matrix()
        if arr.ndim == 2 and arr.shape[1] == 9:
            return arr.reshape(-1, 3, 3)
        if arr.ndim == 3 and arr.shape[1:] == (3, 3):
            return arr
        return None

    @staticmethod
    def _rotm_to_canonical_quat_wxyz(rotm: np.ndarray) -> np.ndarray:
        """Convert rotation matrices [T, 3, 3] to canonical quaternion wxyz [T, 4] with positive w."""
        quat_xyzw = Rotation.from_matrix(rotm).as_quat().astype(np.float32)
        quat_wxyz = quat_xyzw[..., [3, 0, 1, 2]]
        neg_w = quat_wxyz[:, 0] < 0
        quat_wxyz[neg_w] *= -1
        return quat_wxyz

    def _quantile_normalize_np(self, data: np.ndarray, quantile_key: str) -> np.ndarray:
        q01, q99 = self.quantile_stats[quantile_key]
        denom = np.maximum(q99 - q01, 1e-6)
        return (2.0 * (data - q01) / denom - 1.0).astype(np.float32)

    def _build_proprio_tensor(
        self,
        episode: dict[str, Any],
        frame_ids: np.ndarray,
        episode_key: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        proprios_dict = episode.get("proprios")
        if not isinstance(proprios_dict, dict):
            proprios_dict = {}

        length = len(frame_ids)
        chunks: list[np.ndarray] = []
        mask_chunks: list[np.ndarray] = []

        for arm in self._ARMS:
            raw_pos = proprios_dict.get(f"{arm}_ee_pos")
            if raw_pos is not None:
                pos = np.asarray(raw_pos, dtype=np.float32)
                if pos.ndim == 1:
                    pos = pos.reshape(-1, 3)
                qk = f"proprio_{arm}_ee_xyz"
                if qk not in self.quantile_stats:
                    raise ValueError(f"Missing quantile key {qk} for {arm}_ee_pos in episode {episode_key}.")
                chunks.append(self._quantile_normalize_np(pos[frame_ids], qk))
                mask_chunks.append(np.ones((length, 3), dtype=np.float32))
            else:
                chunks.append(np.zeros((length, 3), dtype=np.float32))
                mask_chunks.append(np.zeros((length, 3), dtype=np.float32))

            raw_rot = proprios_dict.get(f"{arm}_ee_rotm")
            if raw_rot is not None:
                rotm = self._to_rotmat(raw_rot)
                if rotm is None:
                    raise ValueError(f"Invalid rotation shape for {arm}_ee_rotm in episode {episode_key}.")
                chunks.append(self._rotm_to_canonical_quat_wxyz(rotm[frame_ids]))
                mask_chunks.append(np.ones((length, 4), dtype=np.float32))
            else:
                chunks.append(np.zeros((length, 4), dtype=np.float32))
                mask_chunks.append(np.zeros((length, 4), dtype=np.float32))

            raw_grip = proprios_dict.get(f"{arm}_gripper_pos")
            if raw_grip is not None:
                if "gripper_pos" not in self.quantile_stats:
                    raise ValueError(
                        f"Missing quantile key gripper_pos for {arm}_gripper_pos in episode {episode_key}."
                    )
                grip = np.asarray(raw_grip, dtype=np.float32).reshape(-1, 1)
                grip = self._quantile_normalize_np(grip[frame_ids], "gripper_pos")
                if self.inverse_gripper:
                    grip = -grip
                chunks.append(grip)
                mask_chunks.append(np.ones((length, 1), dtype=np.float32))
            else:
                chunks.append(np.zeros((length, 1), dtype=np.float32))
                mask_chunks.append(np.zeros((length, 1), dtype=np.float32))

        values = np.concatenate(chunks, axis=1)
        mask = np.concatenate(mask_chunks, axis=1)
        values[mask == 0] = 0.0
        return torch.from_numpy(values), torch.from_numpy(mask)

    _RAW_ACTION_DIM_PER_ARM: int = 7  # xyz(3) + axisangle(3) + gripper(1)

    def _build_raw_action_tensor(
        self,
        episode: dict[str, Any],
        action_ids: np.ndarray,
        episode_key: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build action tensor from pre-computed raw_actions [xyz, axisangle, gripper]."""
        raw_actions = np.asarray(episode["actions"]["raw_actions"], dtype=np.float32)
        num_arms_avail = raw_actions.shape[1] // self._RAW_ACTION_DIM_PER_ARM

        length = len(action_ids)
        chunks: list[np.ndarray] = []
        mask_chunks: list[np.ndarray] = []

        for arm_idx, arm in enumerate(self._ARMS):
            if arm_idx < num_arms_avail:
                off = arm_idx * self._RAW_ACTION_DIM_PER_ARM
                arm_data = raw_actions[action_ids, off : off + self._RAW_ACTION_DIM_PER_ARM]

                xyz = arm_data[:, :3]
                qk_xyz = f"action_{arm}_ee_xyz"
                if qk_xyz not in self.quantile_stats:
                    raise ValueError(f"Missing quantile key {qk_xyz} for raw action xyz in episode {episode_key}.")
                chunks.append(self._quantile_normalize_np(xyz, qk_xyz))
                mask_chunks.append(np.ones((length, 3), dtype=np.float32))

                aa = arm_data[:, 3:6]
                qk_aa = f"action_{arm}_ee_axisangle"
                if qk_aa not in self.quantile_stats:
                    raise ValueError(f"Missing quantile key {qk_aa} for raw action axisangle in episode {episode_key}.")
                chunks.append(self._quantile_normalize_np(aa, qk_aa))
                mask_chunks.append(np.ones((length, 3), dtype=np.float32))

                grip = -arm_data[:, 6:7]  # note: -1 means open for robocasa gripper
                chunks.append(grip)
                mask_chunks.append(np.ones((length, 1), dtype=np.float32))
            else:
                chunks.append(np.zeros((length, 3), dtype=np.float32))
                mask_chunks.append(np.zeros((length, 3), dtype=np.float32))
                chunks.append(np.zeros((length, 3), dtype=np.float32))
                mask_chunks.append(np.zeros((length, 3), dtype=np.float32))
                chunks.append(np.zeros((length, 1), dtype=np.float32))
                mask_chunks.append(np.zeros((length, 1), dtype=np.float32))

        values = np.concatenate(chunks, axis=1)
        mask = np.concatenate(mask_chunks, axis=1)
        values[mask == 0] = 0.0
        return torch.from_numpy(values), torch.from_numpy(mask)

    def _build_delta_action_tensor(
        self,
        episode: dict[str, Any],
        action_ids: np.ndarray,
        episode_key: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_offset = self.action_skip - 1
        proprio_ids = action_ids - action_offset

        proprios_dict = episode.get("proprios")
        actions_dict = episode.get("actions")
        if not isinstance(proprios_dict, dict):
            proprios_dict = {}
        if not isinstance(actions_dict, dict):
            actions_dict = {}

        length = len(action_ids)
        chunks: list[np.ndarray] = []
        mask_chunks: list[np.ndarray] = []

        for arm in self._ARMS:
            raw_p_pos = proprios_dict.get(f"{arm}_ee_pos")
            raw_a_pos = actions_dict.get(f"{arm}_ee_pos")
            if raw_p_pos is not None and raw_a_pos is not None:
                p_pos = np.asarray(raw_p_pos, dtype=np.float32)
                a_pos = np.asarray(raw_a_pos, dtype=np.float32)
                if p_pos.ndim == 1:
                    p_pos = p_pos.reshape(-1, 3)
                if a_pos.ndim == 1:
                    a_pos = a_pos.reshape(-1, 3)
                delta_pos = a_pos[action_ids] - p_pos[proprio_ids]
                qk = f"action_{arm}_ee_xyz"
                if qk not in self.quantile_stats:
                    raise ValueError(f"Missing quantile key {qk} for delta pos in episode {episode_key}.")
                chunks.append(self._quantile_normalize_np(delta_pos, qk))
                mask_chunks.append(np.ones((length, 3), dtype=np.float32))
            else:
                chunks.append(np.zeros((length, 3), dtype=np.float32))
                mask_chunks.append(np.zeros((length, 3), dtype=np.float32))

            raw_p_rot = proprios_dict.get(f"{arm}_ee_rotm")
            raw_a_rot = actions_dict.get(f"{arm}_ee_rotm")
            if raw_p_rot is not None and raw_a_rot is not None:
                p_rotm = self._to_rotmat(raw_p_rot)
                a_rotm = self._to_rotmat(raw_a_rot)
                if p_rotm is None or a_rotm is None:
                    raise ValueError(f"Invalid rotation shape for delta rot ({arm}) in episode {episode_key}.")
                delta_R = np.einsum(
                    "tij,tkj->tik",
                    a_rotm[action_ids],
                    p_rotm[proprio_ids],
                )
                delta_aa = Rotation.from_matrix(delta_R).as_rotvec().astype(np.float32)
                qk = f"action_{arm}_ee_axisangle"
                if qk not in self.quantile_stats:
                    raise ValueError(f"Missing quantile key {qk} for delta rot in episode {episode_key}.")
                chunks.append(self._quantile_normalize_np(delta_aa, qk))
                mask_chunks.append(np.ones((length, 3), dtype=np.float32))
            else:
                chunks.append(np.zeros((length, 3), dtype=np.float32))
                mask_chunks.append(np.zeros((length, 3), dtype=np.float32))

            raw_p_grip = proprios_dict.get(f"{arm}_gripper_pos")
            if raw_p_grip is not None:
                if "gripper_action" not in self.quantile_stats:
                    raise ValueError(
                        f"Missing quantile key gripper_action for delta {arm}_gripper_pos in episode {episode_key}."
                    )
                p_grip = np.asarray(raw_p_grip, dtype=np.float32).reshape(-1, 1)
                future_ids = proprio_ids + self.action_skip
                if int(future_ids.max()) >= p_grip.shape[0]:
                    raise ValueError(
                        f"proprio {arm}_gripper_pos length {p_grip.shape[0]} insufficient for "
                        f"proprio[t+action_skip]-proprio[t] (need index < {int(future_ids.max()) + 1}) "
                        f"in episode {episode_key}."
                    )
                delta_grip = p_grip[future_ids] - p_grip[proprio_ids]
                grip_action = self._quantile_normalize_np(delta_grip, "gripper_action")
                if self.inverse_gripper:
                    grip_action = -grip_action
                chunks.append(grip_action)
                mask_chunks.append(np.ones((length, 1), dtype=np.float32))
            else:
                chunks.append(np.zeros((length, 1), dtype=np.float32))
                mask_chunks.append(np.zeros((length, 1), dtype=np.float32))

        values = np.concatenate(chunks, axis=1)
        mask = np.concatenate(mask_chunks, axis=1)
        values[mask == 0] = 0.0
        return torch.from_numpy(values), torch.from_numpy(mask)

    @staticmethod
    def _sample_instruction(episode: dict[str, Any], episode_key: str) -> str:
        instructions = episode.get("instructions")
        if not isinstance(instructions, list) or len(instructions) == 0:
            raise ValueError(f"Missing or empty instructions in episode {episode_key}.")

        valid = [item.strip() for item in instructions if isinstance(item, str) and item.strip()]
        if len(valid) == 0:
            raise ValueError(f"No valid instruction strings in episode {episode_key}.")
        return valid[torch.randint(0, len(valid), (1,)).item()]
