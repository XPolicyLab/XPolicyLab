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
import json
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
                v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0),
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
        norm_stats_path: Optional[str] = None,
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
        
        self.norm_stats_path = norm_stats_path
        if self.norm_stats_path is not None:
            self.state_min = torch.tensor(json.load(open(self.norm_stats_path, "r"))["state_min"]).float()
            self.state_max = torch.tensor(json.load(open(self.norm_stats_path, "r"))["state_max"]).float()
            self.action_min = torch.tensor(json.load(open(self.norm_stats_path, "r"))["action_min"]).float()
            self.action_max = torch.tensor(json.load(open(self.norm_stats_path, "r"))["action_max"]).float()
            
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
        
        if self.norm_stats_path is not None:
            state_min = self.state_min
            state_max = self.state_max
            action_min = self.action_min
            action_max = self.action_max
        else:
            # Extract state min/max with proper type conversion
            state_min_raw = stats["observation.state"]["min"]
            state_max_raw = stats["observation.state"]["max"]
            state_min = torch.tensor(state_min_raw) if not isinstance(state_min_raw, torch.Tensor) else state_min_raw
            state_max = torch.tensor(state_max_raw) if not isinstance(state_max_raw, torch.Tensor) else state_max_raw
            
            # Extract action min/max with proper type conversion (BUG FIX: was using "std" instead of "min")
            action_min_raw = stats["action"]["min"]
            action_max_raw = stats["action"]["max"]
            action_min = torch.tensor(action_min_raw) if not isinstance(action_min_raw, torch.Tensor) else action_min_raw
            action_max = torch.tensor(action_max_raw) if not isinstance(action_max_raw, torch.Tensor) else action_max_raw
        
            # Ensure min/max are float tensors for division
            state_min = state_min.float()
            state_max = state_max.float()
            action_min = action_min.float()
            action_max = action_max.float()

		# Print norm_stats
        # torch.set_printoptions(sci_mode=False, precision=8)
        # print (f"state_min: {state_min}")
        # print (f"state_max: {state_max}")
        # print (f"action_min: {action_min}")
        # print (f"action_max: {action_max}")
        
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
        
        # normalize state: 只对前14位进行归一化，但跳过索引6和索引13，最后两位（索引14、15）不归一化
        # 需要归一化的索引: 0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12 (共12位)
        # 不归一化的索引: 6, 13, 14, 15 (共4位)
        state_full = sample["observation.state"]  # shape: (16,)
        if not isinstance(state_full, torch.Tensor):
            state_full = torch.tensor(state_full)
        state_full = state_full.float()
        
        # Create mask for dimensions to normalize: [0,1,2,3,4,5,7,8,9,10,11,12] = True, others = False
        state_normalize_mask = torch.ones(16, dtype=torch.bool)
        state_normalize_mask[6] = False   # 索引6不归一化
        state_normalize_mask[13] = False  # 索引13不归一化
        state_normalize_mask[14] = False  # 索引14不归一化
        state_normalize_mask[15] = False  # 索引15不归一化
        
        # Initialize normalized state with original values
        state_normalized = state_full.clone()
        
        # Normalize only the masked dimensions to [-1, 1] range
        if state_normalize_mask.any():
            state_range = state_max - state_min
            state_range = torch.where(state_range < 1e-8, torch.ones_like(state_range), state_range)
            # Normalize to [0, 1] first, then scale to [-1, 1]
            normalized_01 = (state_full[state_normalize_mask] - state_min[state_normalize_mask]) / state_range[state_normalize_mask]
            state_normalized[state_normalize_mask] = 2.0 * normalized_01 - 1.0
            
            # Clip normalized dimensions to [-1, 1] range
            state_normalized[state_normalize_mask] = torch.clamp(state_normalized[state_normalize_mask], -1.0, 1.0)
            
            # Check for NaN or Inf values in normalized dimensions
            if torch.isnan(state_normalized[state_normalize_mask]).any() or torch.isinf(state_normalized[state_normalize_mask]).any():
                raise ValueError(f"State normalization produced NaN/Inf values. Check state_min/max stats.")
        
            # shape: (1, 16)
            state_normalized = state_normalized.reshape(1, -1)
            
        # normalize action: 只对前14位进行归一化，但跳过索引6和索引13，最后两位（索引14、15）不归一化
        # 对每个action的16维分别应用相同的归一化规则
        action_full = sample["action"]  # shape: (16, 16)
        if not isinstance(action_full, torch.Tensor):
            action_full = torch.tensor(action_full)
        action_full = action_full.float()
        
        # Create mask for dimensions to normalize (same as state)
        action_normalize_mask = torch.ones(16, dtype=torch.bool)
        action_normalize_mask[6] = False   # 索引6不归一化
        action_normalize_mask[13] = False  # 索引13不归一化
        action_normalize_mask[14] = False  # 索引14不归一化
        action_normalize_mask[15] = False  # 索引15不归一化
        
        # Initialize normalized action with original values
        action_normalized = action_full.clone()
        
        # Normalize only the masked dimensions for each action timestep to [-1, 1] range
        if action_normalize_mask.any():
            action_range = action_max - action_min
            action_range = torch.where(action_range < 1e-8, torch.ones_like(action_range), action_range)
            # Apply normalization to masked dimensions across all timesteps
            # Normalize to [0, 1] first, then scale to [-1, 1]
            normalized_01 = (action_full[:, action_normalize_mask] - action_min[action_normalize_mask]) / action_range[action_normalize_mask]
            action_normalized[:, action_normalize_mask] = 2.0 * normalized_01 - 1.0
            
            # Clip normalized dimensions to [-1, 1] range
            action_normalized[:, action_normalize_mask] = torch.clamp(action_normalized[:, action_normalize_mask], -1.0, 1.0)
            
            # Check for NaN or Inf values in normalized dimensions
            if torch.isnan(action_normalized[:, action_normalize_mask]).any() or torch.isinf(action_normalized[:, action_normalize_mask]).any():
                raise ValueError(f"Action normalization produced NaN/Inf values. Check action_min/max stats.")

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
            "action": action_normalized,  # torch.Tensor (action_horizon, action_dim) = (16, 16), 全部16维quantile归一化到[-1, 1]
            "state": state_normalized,  # torch.Tensor (1, 16), 全部16维quantile归一化到[-1, 1]
            "episode_id": int(episode_id) if isinstance(episode_id, torch.Tensor) else episode_id,  # int
            "episode_pos": frame_index,  # offset within episode
            "global_idx": global_idx,  # global sample index
            "subtask_end": int(subtask_end) if isinstance(subtask_end, torch.Tensor) else subtask_end,  # int
        }


def analyze_dataset_distribution(dataset: LeRobot_Dataset, save_dir: str = "distribution_plots"):
    """
    Analyze and plot the distribution of state and action dimensions.
    Also plots distribution after filtering outliers (keeping q01-q99).
    """
    import matplotlib.pyplot as plt
    import os
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Analyzing dataset with {len(dataset)} samples...")
    
    # Collect data (sample if too large)
    num_samples = min(1000, len(dataset))
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    all_states = []
    all_actions = []
    
    for idx in indices:
        sample = dataset[int(idx)]
        all_states.append(sample["state"]) # (1, 16)
        all_actions.append(sample["action"]) # (16, 16)
        
    # Process states
    states = torch.cat(all_states, dim=0) # (N, 16)
    
    # Process actions
    # Flatten time dimension: (N, 16, 16) -> (N*16, 16)
    actions = torch.stack(all_actions, dim=0).reshape(-1, 16)
    
    def plot_distribution(data_tensor, name, filename_suffix=""):
        num_dims = data_tensor.shape[1]
        rows = int(np.ceil(np.sqrt(num_dims)))
        cols = int(np.ceil(num_dims / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        axes = axes.flatten()
        
        for d in range(num_dims):
            data = data_tensor[:, d].numpy()
            
            # If filtered version requested
            if "filtered" in filename_suffix:
                q01 = np.percentile(data, 1)
                q99 = np.percentile(data, 99)
                data = data[(data >= q01) & (data <= q99)]
                title_suffix = " (Q01-Q99)"
            else:
                title_suffix = ""

            ax = axes[d]
            if len(data) > 0:
                ax.hist(data, bins=50, density=True, alpha=0.7, color='blue' if 'state' in name else 'green')
                mean = np.mean(data)
                std = np.std(data)
                ax.text(0.05, 0.95, f"Mean: {mean:.2f}\nStd: {std:.2f}", transform=ax.transAxes, verticalalignment='top', fontsize=8)
            
            ax.set_title(f"{name} Dim {d}{title_suffix}")
            
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{name}_distribution{filename_suffix}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved {name} distribution to {save_path}")

    # Plot original distributions
    plot_distribution(states, "state")
    plot_distribution(actions, "action")
    
    # Plot filtered distributions
    plot_distribution(states, "state", "_filtered")
    plot_distribution(actions, "action", "_filtered")


def _format_list_fixed_10(values) -> str:
    """Format a 1D iterable as JSON list with floats fixed to 10 decimals (no scientific notation)."""
    flat = np.asarray(values).reshape(-1).tolist()
    return "[" + ", ".join(f"{float(v):.10f}" for v in flat) + "]"


def save_norm_stats(dataset: LeRobot_Dataset, task_name: str, lang: str) -> str:
    """Persist norm stats in a brace-wrapped JSON object with fixed 10-decimal floats."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    target_dir = os.path.join(project_root, "assets", task_name)
    os.makedirs(target_dir, exist_ok=True)
    norm_path = os.path.join(target_dir, "norm_stats.json")
    lang_path = os.path.join(target_dir, "global_instruction.txt")

    stats = dataset.dataset.meta.stats
    state_min = stats["observation.state"]["min"]
    state_max = stats["observation.state"]["max"]
    action_min = stats["action"]["min"]
    action_max = stats["action"]["max"]
    
    with open(lang_path, "w", encoding="utf-8") as f:
        f.write(lang)

    with open(norm_path, "w", encoding="utf-8") as f:
        f.write("{\n")
        f.write(f"  \"state_min\": {_format_list_fixed_10(state_min)},\n")
        f.write(f"  \"state_max\": {_format_list_fixed_10(state_max)},\n")
        f.write(f"  \"action_min\": {_format_list_fixed_10(action_min)},\n")
        f.write(f"  \"action_max\": {_format_list_fixed_10(action_max)}\n")
        f.write("}\n")

    return norm_path


if __name__ == "__main__":
    task_name = "your_task_name"
    repo_id = "your_repo_id"

    dataset = LeRobot_Dataset(
        repo_id=repo_id,
        features_to_load=[
            "observation.image.head_camera",
            "observation.state",
            "action",
            "subtask",
            "subtask_end",
            "episode_id"
        ],
    )
    sample = dataset[0]
    
    print (f"instruction: {sample['lang']}")
    print (f"action: {sample['action'].shape}")
    print (f"state: {sample['state']}")
    
    norm_stats_path = save_norm_stats(dataset, task_name, sample['lang'])
    print(f"Saved norm stats to {norm_stats_path}")
    
    # Run analysis
    # analyze_dataset_distribution(dataset)