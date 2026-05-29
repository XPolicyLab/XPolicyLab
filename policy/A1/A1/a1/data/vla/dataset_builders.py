"""VLA Dataset Builders - Factory pattern implementation for building various VLA datasets.

This module provides a unified interface for building different types of VLA datasets
(RLDS, LeRobot, Droid, RoboChallenge, AgiBot) using the factory pattern.

Each builder parses its own config from raw dict, allowing dataset-specific flexibility.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, cast
from pathlib import Path

from a1.config import TrainConfig
from a1.data.dataset import IterableDatasetWrapper
from a1.data.vla.utils import NormalizationType

log = logging.getLogger(__name__)


class DatasetBuilder(ABC):
    """Abstract base class for dataset builders."""
    
    @abstractmethod
    def build(self, raw_config: dict, train_config: TrainConfig, device: str) -> Tuple[Any, float]:
        """Build dataset from raw config dict.
        
        Args:
            raw_config: Raw dictionary configuration from YAML
            train_config: Global training configuration
            device: Device to use
            
        Returns:
            Tuple of (dataset, weight)
        """
        pass
    
    def validate_path(self, path: str) -> bool:
        """Check if path exists and is a directory."""
        return path and os.path.isdir(path)


class RLDSBuilder(DatasetBuilder):
    """Builder for RLDS datasets."""
    
    def build(self, raw_config: dict, train_config: TrainConfig, device: str) -> Tuple[Any, float]:
        from a1.data.vla.rlds_datasets import RLDSDataset, RLDSBatchTransform
        from a1.data import build_mm_preprocessor
        from a1.data.vla.utils import NormalizationType
        
        # Parse config fields from raw dict
        path = raw_config.get("path", "")
        name = raw_config.get("name")
        weight = raw_config.get("weight", 1.0)
        image_aug = raw_config.get("image_augmentation", False)
        
        # Parse normalization_type
        norm_type_str = raw_config.get("normalization_type") or raw_config.get("action_proprio_normalization_type")
        norm_type = NormalizationType.BOUNDS  # default
        if norm_type_str:
            try:
                norm_type = NormalizationType(norm_type_str)
            except ValueError:
                log.warning(f"Invalid normalization_type: {norm_type_str}, using default")
        
        # Parse RLDS pipeline controls from raw_config (with defaults)
        shuffle_buffer_size = raw_config.get("shuffle_buffer_size", 100000)
        traj_transform_threads = raw_config.get("traj_transform_threads", 8)
        traj_read_threads = raw_config.get("traj_read_threads", 8)
        
        # Create preprocessor
        preprocessor = build_mm_preprocessor(
            train_config.model,
            shuffle_messages=train_config.data.shuffle,
            is_training=True,
            require_image_features=True
        )
        
        batch_transform = RLDSBatchTransform(
            fixed_action_dim=train_config.model.fixed_action_dim,
            use_wrist_image=train_config.data.use_wrist_image,
            use_proprio=train_config.data.use_proprio,
        )
        
        dataset = RLDSDataset(
            data_root_dir=path,
            data_mix=name,
            normalization_type=norm_type,
            num_actions_chunk=train_config.model.num_actions_chunk,
            batch_transform=batch_transform,
            resize_resolution=train_config.model.vision_backbone.image_default_input_size,
            shuffle_buffer_size=shuffle_buffer_size,
            traj_transform_threads=traj_transform_threads,
            traj_read_threads=traj_read_threads,
            train=True,
            image_aug=image_aug,
        )
        
        # Wrap with IterableDatasetWrapper
        dataset = IterableDatasetWrapper(dataset, preprocessor, train_config.data.seed)
        
        return dataset, weight
    
    def validate_mixture(self, raw_config: dict) -> bool:
        """Validate that at least one dataset in the mixture exists."""
        from a1.data.vla.rlds.oxe.mixtures import OXE_NAMED_MIXTURES
        
        name = raw_config.get("name")
        path = raw_config.get("path", "")
        
        if not name:
            log.warning(f"RLDS config missing 'name' field")
            return False
        
        mixture = OXE_NAMED_MIXTURES.get(name, [])
        if not mixture:
            log.warning(f"Unknown mixture name: {name}")
            return False
            
        any_present = any(os.path.isdir(os.path.join(path, d_name)) 
                         for d_name, _ in mixture)
        if not any_present:
            log.warning(f"Mixture '{name}' not found under: {path}")
            log.warning(f"Expect one of: {[d for d, _ in mixture]}")
        return any_present


class LeRobotBuilder(DatasetBuilder):
    """Builder for LeRobot datasets."""
    
    def build(self, raw_config: dict, train_config: TrainConfig, device: str) -> Tuple[Any, float]:
        from a1.data.vla.lerobot_datasets import LeRobotDatasetWrapper, LeRobotDatasetWrapperAgiBotWorld
        from a1.data import build_mm_preprocessor
        from a1.data.vla.utils import NormalizationType
        
        # Parse config fields from raw dict
        path = raw_config.get("path", "")
        weight = raw_config.get("weight", 1.0)
        num_episodes = raw_config.get("num_episodes")
        image_aug = raw_config.get("image_augmentation", False)
        
        # Parse normalization_type (default None, meaning no normalization)
        norm_type_str = raw_config.get("normalization_type") or raw_config.get("action_proprio_normalization_type")
        norm_type = None
        if norm_type_str:
            try:
                norm_type = NormalizationType(norm_type_str)
            except ValueError:
                log.warning(f"Invalid normalization_type: {norm_type_str}, using None (no normalization)")
        
        # Create preprocessor
        preprocessor = build_mm_preprocessor(
            train_config.model,
            shuffle_messages=train_config.data.shuffle,
            is_training=True,
            require_image_features=True
        )
        
        # Determine which wrapper to use
        if "AgiBotWorld-Alpha" in path:
            dataset_wrapper = LeRobotDatasetWrapperAgiBotWorld
        else:
            dataset_wrapper = LeRobotDatasetWrapper
        
        dataset = dataset_wrapper(
            path,
            normalization_type=norm_type,
            use_proprio=train_config.data.use_proprio,
            fixed_action_dim=train_config.model.fixed_action_dim,
            use_wrist_image=train_config.data.use_wrist_image,
            chunk_size=train_config.model.num_actions_chunk,
            num_episodes=num_episodes,
            image_aug=image_aug,
        )
        
        # Wrap with IterableDatasetWrapper
        dataset = IterableDatasetWrapper(dataset, preprocessor, train_config.data.seed)
        
        return dataset, weight


class DroidBuilder(DatasetBuilder):
    """Builder for Droid datasets (LeRobot format)."""
    
    def build(self, raw_config: dict, train_config: TrainConfig, device: str) -> Tuple[Any, float]:
        from a1.data.vla.lerobot_datasets import LeRobotDatasetWrapperDroid
        from a1.data import build_mm_preprocessor
        from a1.data.vla.utils import NormalizationType
        
        # Parse config fields from raw dict
        path = raw_config.get("path", "")
        weight = raw_config.get("weight", 1.0)
        num_episodes = raw_config.get("num_episodes")
        image_aug = raw_config.get("image_augmentation", False)
        
        # Parse normalization_type (default None, meaning no normalization)
        norm_type_str = raw_config.get("normalization_type") or raw_config.get("action_proprio_normalization_type")
        norm_type = None
        if norm_type_str:
            try:
                norm_type = NormalizationType(norm_type_str)
            except ValueError:
                log.warning(f"Invalid normalization_type: {norm_type_str}, using None (no normalization)")
        
        # Create preprocessor
        preprocessor = build_mm_preprocessor(
            train_config.model,
            shuffle_messages=train_config.data.shuffle,
            is_training=True,
            require_image_features=True
        )
        
        dataset = LeRobotDatasetWrapperDroid(
            path,
            normalization_type=norm_type,
            use_proprio=train_config.data.use_proprio,
            fixed_action_dim=train_config.model.fixed_action_dim,
            use_wrist_image=train_config.data.use_wrist_image,
            chunk_size=train_config.model.num_actions_chunk,
            num_episodes=num_episodes,
            image_aug=image_aug,
        )
        
        # Wrap with IterableDatasetWrapper
        dataset = IterableDatasetWrapper(dataset, preprocessor, train_config.data.seed)
        
        return dataset, weight


class RoboChallengeBuilder(DatasetBuilder):
    """Builder for RoboChallenge datasets."""
    
    def build(self, raw_config: dict, train_config: TrainConfig, device: str) -> Tuple[Any, float]:
        from a1.data.vla.rc_reader import RoboChallengeDatasetReader
        from a1.data import build_mm_preprocessor
        from a1.data.vla.utils import NormalizationType
        
        # Parse config fields from raw dict
        path = raw_config.get("path", "")
        weight = raw_config.get("weight", 1.0)
        embodiment = raw_config.get("embodiment")
        task_name = raw_config.get("task_name")
        norm_stats_path = raw_config.get("norm_stats_path")
        
        # Required fields
        if not embodiment:
            raise ValueError(f"RoboChallenge dataset requires 'embodiment' field: {raw_config}")
        if not task_name:
            raise ValueError(f"RoboChallenge dataset requires 'task_name' field: {raw_config}")
        
        # Parse normalization_type
        norm_type_str = raw_config.get("normalization_type") or raw_config.get("action_proprio_normalization_type")
        norm_type = None
        if norm_type_str:
            try:
                norm_type = NormalizationType(norm_type_str)
            except ValueError:
                log.warning(f"Invalid normalization_type: {norm_type_str}, using default")
        
        # Create preprocessor
        preprocessor = build_mm_preprocessor(
            train_config.model,
            shuffle_messages=train_config.data.shuffle,
            is_training=True,
            require_image_features=True
        )
        
        dataset = RoboChallengeDatasetReader(
            dataset_path=path,
            embodiment=embodiment,
            env_names=[task_name],
            fixed_action_dim=train_config.model.fixed_action_dim,
            chunk_size=train_config.model.num_actions_chunk,
            normalization_type=norm_type,
            norm_stats_path=norm_stats_path,
        )
        
        # Wrap with IterableDatasetWrapper
        dataset = IterableDatasetWrapper(dataset, preprocessor, train_config.data.seed)
        
        return dataset, weight


class AgiBotBuilder(DatasetBuilder):
    """Builder for AgiBotWorld-Alpha datasets."""
    
    def build(self, raw_config: dict, train_config: TrainConfig, device: str) -> Tuple[Any, float]:
        from a1.data.vla.agibot_dataset import AgiBotWorldAlphaDataset
        from a1.data import build_mm_preprocessor
        from a1.data.vla.utils import NormalizationType
        
        # Parse config fields from raw dict
        path = raw_config.get("path", "")
        weight = raw_config.get("weight", 1.0)
        
        # Parse normalization_type
        norm_type_str = raw_config.get("normalization_type") or raw_config.get("action_proprio_normalization_type")
        norm_type = None  # default
        if norm_type_str:
            try:
                norm_type = NormalizationType(norm_type_str)
            except ValueError:
                raise ValueError(f"Invalid normalization_type: {norm_type_str}")
        
        # Create preprocessor
        preprocessor = build_mm_preprocessor(
            train_config.model,
            shuffle_messages=train_config.data.shuffle,
            is_training=True,
            require_image_features=True
        )
        
        dataset = AgiBotWorldAlphaDataset(
            root_dir=path,
            normalization_type=norm_type,
            use_proprio=train_config.data.use_proprio,
            use_wrist_image=train_config.data.use_wrist_image,
        )
        
        # Wrap with IterableDatasetWrapper
        dataset = IterableDatasetWrapper(dataset, preprocessor, train_config.data.seed)
        
        return dataset, weight


class RoboMINDBuilder(DatasetBuilder):
    """Builder for RoboMIND datasets."""
    
    def build(self, raw_config: dict, train_config: TrainConfig, device: str) -> Tuple[Any, float]:
        from a1.data.vla.robomind_datasets import RoboMINDDatasetReader
        from a1.data import build_mm_preprocessor
        from a1.data.vla.utils import NormalizationType
        
        # Parse config fields from raw dict
        path = raw_config.get("path", "")
        weight = raw_config.get("weight", 1.0)
        embodiment = raw_config.get("embodiment")
        
        # Required field
        if not embodiment:
            raise ValueError(f"RoboMIND dataset requires 'embodiment' field: {raw_config}")
        
        # Parse normalization_type (default BOUNDS_Q99)
        norm_type_str = raw_config.get("normalization_type") or raw_config.get("action_proprio_normalization_type")
        norm_type = None  # default
        if norm_type_str:
            try:
                norm_type = NormalizationType(norm_type_str)
            except ValueError:
                log.warning(f"Invalid normalization_type: {norm_type_str}, using BOUNDS_Q99")
        
        # Create preprocessor
        preprocessor = build_mm_preprocessor(
            train_config.model,
            shuffle_messages=train_config.data.shuffle,
            is_training=True,
            require_image_features=True
        )
        
        dataset = RoboMINDDatasetReader(
            dataset_path=path,
            embodiment=embodiment,
            normalization_type=norm_type,
            fixed_action_dim=train_config.model.fixed_action_dim,
            chunk_size=train_config.model.num_actions_chunk,
        )
        
        # Wrap with IterableDatasetWrapper
        dataset = IterableDatasetWrapper(dataset, preprocessor, train_config.data.seed)
        
        return dataset, weight


class RoboCOINBuilder(DatasetBuilder):
    """Builder for RoboCOIN datasets.
    
    RoboCOIN is organized with multiple sub-datasets under a root directory.
    Each sub-directory is a standard LeRobot-format dataset.
    """
    
    def build(self, raw_config: dict, train_config: TrainConfig, device: str) -> Tuple[Any, float]:
        from a1.data.vla.robocoin_reader import RoboCoinDatasetWrapper
        from a1.data import build_mm_preprocessor
        from a1.data.vla.utils import NormalizationType
        
        # Parse config fields from raw dict
        path = raw_config.get("path", "")
        weight = raw_config.get("weight", 1.0)
        num_episodes = raw_config.get("num_episodes")
        
        # Parse normalization_type (default NORMAL)
        norm_type_str = raw_config.get("normalization_type") or raw_config.get("action_proprio_normalization_type")
        norm_type = None  # default for RoboCOIN
        if norm_type_str:
            try:
                norm_type = NormalizationType(norm_type_str)
            except ValueError:
                log.warning(f"Invalid normalization_type: {norm_type_str}, using NORMAL")
        
        # Create preprocessor
        preprocessor = build_mm_preprocessor(
            train_config.model,
            shuffle_messages=train_config.data.shuffle,
            is_training=True,
            require_image_features=True
        )
        
        dataset = RoboCoinDatasetWrapper(
            dataset_path=path,
            chunk_size=train_config.model.num_actions_chunk,
            fixed_action_dim=train_config.model.fixed_action_dim,
            normalization_type=norm_type,
            use_proprio=train_config.data.use_proprio,
            use_wrist_image=train_config.data.use_wrist_image,
            num_episodes=num_episodes,
        )
        
        # Wrap with IterableDatasetWrapper
        dataset = IterableDatasetWrapper(dataset, preprocessor, train_config.data.seed)
        
        return dataset, weight


class DatasetBuilderFactory:
    """Factory for managing and creating dataset builders."""
    
    _builders: Dict[str, DatasetBuilder] = {
        "rlds": RLDSBuilder(),
        "lerobot": LeRobotBuilder(),
        "droid": DroidBuilder(),
        "robochallenge": RoboChallengeBuilder(),
        "agibot": AgiBotBuilder(),
        "robomind": RoboMINDBuilder(),
        "robocoin": RoboCOINBuilder(),
    }
    
    @classmethod
    def register(cls, name: str, builder: DatasetBuilder) -> None:
        """Register a new dataset builder."""
        cls._builders[name] = builder
        log.info(f"Registered dataset builder: {name}")
    
    @classmethod
    def get(cls, name: str) -> Optional[DatasetBuilder]:
        """Get a dataset builder by name."""
        return cls._builders.get(name)
    
    @classmethod
    def build_from_config(
        cls,
        name: str,
        raw_config: dict,
        train_config: TrainConfig,
        device: str = "cpu"
    ) -> Optional[Tuple[Any, float]]:
        """Build dataset from raw dict config.
        
        Args:
            name: Dataset type name (rlds, lerobot, etc.)
            raw_config: Dict configuration from YAML
            train_config: Global training configuration
            device: Device to use
            
        Returns:
            Tuple of (dataset, weight) or None if validation fails
        """
        builder = cls.get(name)
        if builder is None:
            raise ValueError(f"Unknown dataset type: {name}")
        
        # Validate path if present in config
        path = raw_config.get("path", "")
        if path and not builder.validate_path(path):
            raise ValueError(f"Path not found: {path}")
        
        # Special validation for RLDS mixtures
        if name == "rlds":
            rlds_builder = cast(RLDSBuilder, builder)
            if not rlds_builder.validate_mixture(raw_config):
                raise ValueError(f"Invalid mixture: {raw_config}")
        
        dataset, weight = builder.build(raw_config, train_config, device)
        log.info(f"Built {name} dataset from {path or 'N/A'} (weight={weight})")
        return dataset, weight
