# Copyright 2025 MemoryMatters Team. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Yuran Wang / Peking University] in [2025].
"""
RMBench LeRobot Dataset Loader

This module provides a selective loading mechanism for LeRobot datasets, allowing
users to load only specific features instead of all features. This significantly
reduces I/O operations and memory usage, especially for datasets with many
video/image features.

Key Features:
- Selective loading of parquet features (reduces disk I/O)
- Selective loading of video features (reduces video decoding overhead)
- Automatic filtering of meta.video_keys and meta.camera_keys to prevent
  unnecessary video decoding in LeRobotDataset.__getitem__
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from typing import List, Optional
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hf_transform_to_torch
from torchvision.transforms import v2
from typing import Tuple, Union
import torch
import numpy as np
from PIL import Image

import source.utils.pil_tools as pil_tools


class LeRobot_Selective_Dataset(LeRobotDataset):
    """
    LeRobotDataset subclass that supports selective loading of specific features.
    
    This class extends LeRobotDataset to allow loading only a subset of features
    from the dataset, which can significantly improve loading speed and reduce
    memory usage. The key optimization is filtering meta.video_keys and
    meta.camera_keys to prevent LeRobotDataset.__getitem__ from decoding
    unnecessary videos.
    
    Example:
        >>> dataset = SelectiveLeRobotDataset(
        ...     repo_id="path/to/dataset",
        ...     features_to_load=[
        ...         "observation.images.cam_high_rgb",
        ...         "observation.state",
        ...         "action",
        ...         "gripper_mode_action",
        ...     ]
        ... )
    """
    
    def __init__(
        self,
        repo_id: str,
        features_to_load: Optional[List[str]] = None,
        image_scale: Optional[Tuple[int, int]] = (224, 224),
        fps: int = 30,
        action_horizon: int = 16,
        video_backend: str = "pyav",
        **kwargs
    ):
        """
        Initialize SelectiveLeRobotDataset.
        
        Args:
            repo_id: Dataset path or repository ID.
            features_to_load: List of feature keys to load. If None, loads all features.
                Must include all required features (e.g., episode_index, timestamp, etc.).
                Required features are automatically added if missing.
            **kwargs: Additional arguments passed to LeRobotDataset.
        
        Note:
            The _filter_meta_keys() method is called automatically in load_hf_dataset()
            after _loaded_features is set, so it should not be called in __init__.
        """
        self.features_to_load = features_to_load
        self._loaded_features: Optional[dict] = None  # Set in load_hf_dataset()
        
        # 先初始化父类以获取 meta（包含 fps 信息）
        # 但我们需要在 __init__ 之前设置 delta_timestamps
        # 所以先创建一个临时对象来获取 fps，或者使用传入的 fps
        # 如果 fps 未指定，我们会在 super().__init__ 后从 meta 获取并更新
        self._action_horizon = action_horizon
        self._fps_param = fps
        
        # 计算 delta_timestamps
        delta_seconds = [i / fps for i in range(action_horizon)]
        self.delta_timestamps = {"action": delta_seconds}
        
        # setup image transforms
        if image_scale is not None:
            self.image_transforms = v2.Compose([
                v2.Resize(image_scale),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0),
            ])
        else:
            self.image_transforms = None
        
        super().__init__(repo_id, delta_timestamps=self.delta_timestamps, image_transforms=self.image_transforms, video_backend=video_backend, **kwargs)
        
    @property
    def loaded_features(self) -> dict[str, dict]:
        """
        Return the actually loaded features (after selective loading).
        
        Note:
            - meta.features still contains all features (read from info.json)
            - loaded_features only contains features that were actually loaded
        
        Returns:
            Dictionary of actually loaded features with their metadata.
        """
        if self._loaded_features is not None:
            return self._loaded_features
        
        # Fallback: return all features if not yet loaded
        return self.meta.features
    
    def _filter_meta_keys(self) -> None:
        """
        Filter meta.video_keys and meta.camera_keys to only include loaded features.
        
        This is a critical optimization: LeRobotDataset.__getitem__ decodes videos
        based on meta.video_keys. If info.json contains many video features but
        we only selected a subset, without this filtering, all videos would still
        be decoded, causing significant performance degradation.
        
        Method:
            Temporarily modifies meta.info["features"] to only include loaded
            features. Since video_keys and camera_keys are properties computed
            from meta.features, they will automatically return filtered results.
        
        Note:
            This method should only be called after _loaded_features is set
            (i.e., in load_hf_dataset()).
        """
        if self._loaded_features is None:
            return
        
        # Save original features if not already saved
        if not hasattr(self.meta, '_original_features'):
            self.meta._original_features = self.meta.info["features"].copy()
        
        # Temporarily modify meta.info["features"] to only include loaded features
        # This makes meta.video_keys and meta.camera_keys properties return filtered results
        self.meta.info["features"] = self._loaded_features.copy()
        
        # Calculate filtered keys for logging
        original_video_keys = [
            key for key, ft in self.meta._original_features.items()
            if ft.get("dtype") == "video"
        ]
        loaded_video_keys = self.meta.video_keys
        
        original_camera_keys = [
            key for key, ft in self.meta._original_features.items()
            if ft.get("dtype") in ["video", "image"]
        ]
        loaded_camera_keys = self.meta.camera_keys
        
        # --- debug information ---
        # print(
        #     f"Filtered video_keys: {len(loaded_video_keys)} "
        #     f"(from {len(original_video_keys)} total)"
        # )
        # print(
        #     f"Filtered camera_keys: {len(loaded_camera_keys)} "
        #     f"(from {len(original_camera_keys)} total)"
        # )
    
    def load_hf_dataset(self):
        """
        Override load_hf_dataset to selectively load only specified features.
        
        This method:
        1. Separates parquet features from video features
        2. Uses select_columns() for parquet features only
        3. Records video features in _loaded_features even though they're not in parquet
        4. Filters meta.video_keys to prevent unnecessary video decoding
        
        Returns:
            HuggingFace Dataset with only selected features loaded.
        
        Raises:
            ValueError: If critical required features are missing from the dataset.
        """
        # Load dataset from parquet files
        if self.episodes is None:
            path = str(self.root / "data")
            hf_dataset = load_dataset("parquet", data_dir=path, split="train")
        else:
            files = [
                str(self.root / self.meta.get_data_file_path(ep_idx))
                for ep_idx in self.episodes
            ]
            hf_dataset = load_dataset("parquet", data_files=files, split="train")
        
        # Apply selective loading if features_to_load is specified
        if self.features_to_load is not None:
            # Ensure required features are included
            # These are needed by LeRobotDataset.__getitem__
            required_features = [
                "episode_index",  # Required: for episode boundary detection
                "timestamp",      # Required: for timestamp synchronization
                "frame_index",    # May be needed: frame index
                "index",          # May be needed: sample index
                "task_index",     # Required: used in LeRobotDataset.__getitem__ (line 729)
            ]
            all_requested_features = list(
                set(self.features_to_load + required_features)
            )
            
            # Separate parquet features from video features
            # Video features are not in parquet files; they are loaded from video files
            available_features = set(hf_dataset.column_names)
            parquet_features_to_select = [
                f for f in all_requested_features if f in available_features
            ]
            video_features = [
                f for f in all_requested_features if f not in available_features
            ]
            
            # Apply select_columns only to parquet features
            if parquet_features_to_select:
                hf_dataset = hf_dataset.select_columns(parquet_features_to_select)
            
            # Build _loaded_features dictionary
            # Note: Video features are not in parquet, but we need to record them
            # in _loaded_features so _filter_meta_keys() can correctly filter
            # meta.video_keys
            all_features = self.meta.features
            self._loaded_features = {}
            
            # Add parquet features
            for key in parquet_features_to_select:
                if key in all_features:
                    self._loaded_features[key] = all_features[key]
            
            # Add video features (even though they're not in parquet)
            for key in video_features:
                if key in all_features:
                    self._loaded_features[key] = all_features[key]
            
            # --- debug information ---
            # print(
            #     f"Selective loading: Loading {len(parquet_features_to_select)} "
            #     f"parquet features + {len(video_features)} video features"
            # )
            # print(f"  Parquet features: {sorted(parquet_features_to_select)}")
            # if video_features:
            #     print(f"  Video features: {sorted(video_features)}")
            
            # Validate critical required features
            critical_required = ["episode_index", "timestamp", "task_index"]
            missing_critical = set(critical_required) - set(parquet_features_to_select)
            if missing_critical:
                raise ValueError(
                    f"Critical required features {missing_critical} are not "
                    f"available in the dataset. "
                    f"Available features: {sorted(available_features)}"
                )
            
            # --- debug information ---
            # print(
            #     f"Loaded features count: {len(self._loaded_features)} "
            #     f"(out of {len(all_features)} total features)"
            # )
            
            # Filter meta.video_keys and meta.camera_keys (critical optimization)
            # Must be called after _loaded_features is set
            self._filter_meta_keys()
        
        # Apply transform
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

class LeRobot_Dataset(Dataset):
    def __init__(
        self,
        repo_id: str,
        features_to_load: Optional[List[str]] = None,
        image_scale: Optional[Tuple[int, int]] = (224, 224),
        fps: int = 30,
        action_horizon: int = 16,
        video_backend: str = "pyav",
        **kwargs
    ):
        self.dataset = LeRobot_Selective_Dataset(
            repo_id=repo_id,
            features_to_load=features_to_load,
            image_scale=image_scale,
            fps=fps,
            action_horizon=action_horizon,
            video_backend=video_backend,
            **kwargs
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        Normalize the image and state.
        """
        # get sample from dataset
        sample = self.dataset[idx]
        # get stats from dataset
        stats = self.dataset.meta.stats
        state_std = torch.tensor(stats["observation.state"]["std"]) if not isinstance(stats["observation.state"]["std"], torch.Tensor) else stats["observation.state"]["std"]
        state_mean = torch.tensor(stats["observation.state"]["mean"]) if not isinstance(stats["observation.state"]["mean"], torch.Tensor) else stats["observation.state"]["mean"]
        action_std = torch.tensor(stats["action"]["std"]) if not isinstance(stats["action"]["std"], torch.Tensor) else stats["action"]["std"]
        action_mean = torch.tensor(stats["action"]["mean"]) if not isinstance(stats["action"]["mean"], torch.Tensor) else stats["action"]["mean"]
        
        # print (f"state_mean: {state_mean}")
        # print (f"state_std: {state_std}")
        # print (f"action_mean: {action_mean}")
        # print (f"action_std: {action_std}")
        
        # get image from sample
        cam_high_rgb = sample["observation.image.head_camera"]
        
        if isinstance(cam_high_rgb, torch.Tensor):
            cam_high_rgb = cam_high_rgb.numpy()
        
        # Convert image format (C, H, W) -> (H, W, C)
        if cam_high_rgb.ndim == 3 and cam_high_rgb.shape[0] == 3:
            cam_high_rgb = np.transpose(cam_high_rgb, (1, 2, 0))
        
        # Convert to uint8 and create PIL Image
        if cam_high_rgb.dtype != np.uint8:
            if cam_high_rgb.max() <= 2.0:
                cam_high_rgb = (cam_high_rgb * 255).astype(np.uint8)
            else:
                cam_high_rgb = cam_high_rgb.astype(np.uint8)

        # get PIL image
        image = Image.fromarray(cam_high_rgb, mode='RGB')
        
        # get final image in numpy array
        images = [image]
        
        # normalize state: 前14位归一化，后2位保留原值
        state_full = sample["observation.state"]  # shape: (16,)
        if not isinstance(state_full, torch.Tensor):
            state_full = torch.tensor(state_full)
        safe_state_std = np.where(state_std[ : 14] < 1e-6, 1.0, state_std[ : 14])
        state_normalized = (state_full[:14] - state_mean[:14]) / safe_state_std # shape: (14,)
        state_original = state_full[14:]  # shape: (2,)
        state = torch.cat([state_normalized, state_original]).reshape(1, -1)  # shape: (1, 16)
        
        # normalize action: 前14位归一化，后2位保留原值
        # sample["action"] shape: (action_horizon, action_dim) = (16, 16)
        # action_mean/action_std shape: (action_dim,) = (16,)
        action_full = sample["action"]  # shape: (16, 16)
        if not isinstance(action_full, torch.Tensor):
            action_full = torch.tensor(action_full)
        safe_action_std = np.where(action_std[ : 14] < 1e-6, 1.0, action_std[ : 14])
        action_normalized = (action_full[:, :14] - action_mean[:14]) / safe_action_std  # shape: (16, 14)
        action_original = action_full[:, 14:]  # shape: (16, 2)
        action = torch.cat([action_normalized, action_original], dim=1)  # shape: (16, 16)
        
        # get subtask from sample
        subtask = sample["subtask"]
        
        # episode_id from sample
        episode_id = sample["episode_id"]
        
        # episode_pos from sample
        frame_index = sample["frame_index"]
        
        # global_idx from sample
        global_idx = sample["index"]
        
        # subtask_end from sample
        subtask_end = sample["subtask_end"]
        if subtask_end == True:
            subtask_end = 1
        else:
            subtask_end = 0
        
        return {
            "image": images,  # List[PIL.Image]
            "lang": subtask,  # str
            "action": action,  # torch.Tensor (action_horizon, action_dim) = (16, 16), 全部16维quantile归一化到[-1, 1]
            "state": state,  # torch.Tensor (1, 16), 全部16维quantile归一化到[-1, 1]
            "episode_id": int(episode_id) if isinstance(episode_id, torch.Tensor) else episode_id,  # int
            "episode_pos": frame_index,  # offset within episode
            "global_idx": global_idx,  # global sample index
            "subtask_end": int(subtask_end) if isinstance(subtask_end, torch.Tensor) else subtask_end,  # int
        }


if __name__ == "__main__":
    print ("dataset test\n")
    
    dataset = LeRobot_Dataset(
        repo_id="your_repo_id",
        features_to_load=[
            "observation.image.head_camera",
            "observation.state",
            "action",
            "subtask",
            "subtask_end",
            "episode_id"
        ]
    )
    sample = dataset[0]
    
    print (f"instruction: {sample['lang']}")
    
    out = pil_tools.save_plot_and_numpy(sample['image'][0], "images/train_images/_debug_obs_image.png")
    print (f"RGB stats: {out['stats']}")
    print (f"RGB stats: {out['saved_stats']}")