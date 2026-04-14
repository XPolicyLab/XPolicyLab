# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import torch
import os
import json
from collections import defaultdict
from pathlib import Path
import random
from PIL import Image
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any
import bisect
from scipy.spatial.transform import Rotation as R

from BeingH.utils.video_utils import get_frames_by_timestamps
from BeingH.utils.conversation import get_conv_template
from BeingH.utils.constants import EmbodimentTag, EMBODIMENT_TAG_MAPPING, INSTRUCTION_TEMPLATE, MULTI_DB_INSTRUCT_TEMPLATE
from BeingH.utils.schema import (
    DatasetMetadata, DatasetStatistics, DatasetModalities,
    StateActionMetadata, VideoMetadata,
)
from ..preprocess import build_vit_transform_base
from ..parquet_utils import calculate_dataset_statistics
from configs.data_config import DATA_CONFIG_MAP, BaseDataConfig

LE_ROBOT_MODALITY_FILENAME = "meta/modality.json"
LE_ROBOT_EPISODE_FILENAME = "meta/episodes.jsonl"
LE_ROBOT_TASKS_FILENAME = "meta/tasks.jsonl"
LE_ROBOT_INFO_FILENAME = "meta/info.json"
LE_ROBOT_STATS_FILENAME = "meta/stats.json"
LE_ROBOT_DATA_FILENAME = "data/*/*.parquet"


def merge_statistics(
    per_dataset_stats: List[Dict[str, Any]],
    per_dataset_configs: List[BaseDataConfig], # Accept DataConfig list
    dataset_weights: List[float],
) -> Dict[str, Any]:
    """Merge statistics from multiple datasets, generating a structured global statistics organized by final modality keys."""
    if not per_dataset_stats:
        return {}

    # Normalize weights
    weights = np.array(dataset_weights)
    normalized_weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(weights)) / len(weights)

    # 1. Collect all modality keys defined across all datasets
    all_modality_keys = set()
    for config in per_dataset_configs:
        all_modality_keys.update(config.define_modalities().keys())

    merged_stats = {"state": {}, "action": {}}

    # 2. Iterate through each final modality key for merging
    for key in all_modality_keys:
        if not (key.startswith('state.') or key.startswith('action.')):
            continue

        modality_type, modality_name = key.split('.', 1)

        # Initialize accumulators
        num_dims = -1
        for config in per_dataset_configs:
            if key in config.define_modalities():
                defn = config.define_modalities()[key]
                num_dims = defn.end - defn.start
                break
        if num_dims == -1: continue # Should not happen in theory

        weighted_means = np.zeros(num_dims)
        weighted_squares = np.zeros(num_dims)
        min_list, max_list, q01_list, q99_list = [], [], [], []

        # Iterate through each dataset to extract statistics for this modality
        for i, (stats, config) in enumerate(zip(per_dataset_stats, per_dataset_configs)):
            modality_defs = config.define_modalities()

            if key not in modality_defs:
                continue # Current dataset does not have this modality

            defn = modality_defs[key]
            source_col, start, end = defn.source_column, defn.start, defn.end

            if source_col not in stats:
                print(f"Warning: Source column '{source_col}' for modality '{key}' not in stats for dataset {i}. Skipping.")
                continue

            # 4. Slice the statistics for the current modality from the original column statistics
            source_stats = stats[source_col]
            w_i = normalized_weights[i]
            
            means = np.array(source_stats["mean"][start:end])
            stds = np.array(source_stats["std"][start:end])

            # breakpoint()
            
            if len(means) != num_dims:
                print(f"Warning: Dimension mismatch for modality '{key}' in dataset {i}. Expected {num_dims}, got {len(means)}. Skipping.")
                continue

            weighted_means += w_i * means
            weighted_squares += w_i * (stds**2 + means**2)
            
            min_list.append(source_stats["min"][start:end])
            max_list.append(source_stats["max"][start:end])
            if "q01" in source_stats:
                q01_list.append(source_stats["q01"][start:end])
            if "q99" in source_stats:
                q99_list.append(source_stats["q99"][start:end])

        if not min_list: continue

        # Calculate merged statistics
        overall_variance = weighted_squares - weighted_means**2
        overall_variance[overall_variance < 0] = 0

        merged_stats[modality_type][modality_name] = {
            "min": np.min(np.array(min_list), axis=0).tolist(),
            "max": np.max(np.array(max_list), axis=0).tolist(),
            "mean": weighted_means.tolist(),
            "std": np.sqrt(overall_variance).tolist(),
            "q01": np.min(np.array(q01_list), axis=0).tolist() if len(q01_list)>0 else q01_list,
            "q99": np.max(np.array(q99_list), axis=0).tolist() if len(q99_list)>0 else q99_list,
        }
    return merged_stats
    
    
class LeRobotIterableDataset(torch.utils.data.IterableDataset):
    """Base dataset class for LeRobot that supports sharding."""

    def __init__(
        self,
        dataset_name: str,
        data_config_names,
        dataset_path_list: List[str],
        embodiment_tags: List[str],
        vit_transform_args,
        num_used_episodes_per_dataset: List[int] = None,
        num_used_episodes_per_task: List[int] = None,
        num_used_frames_per_dataset: List[int] = None,
        frame_step_size: List[int] = None,
        is_train=True,
        video_backend: str = "decord",
        video_backend_kwargs: dict | None = None,
        logger = None,
        # Tokenizer and text processing
        tokenizer=None, 
        template_name=None,
        prompt_template="long",
        # Visual related
        force_image_size=448,
        num_image_tokens=0,
        max_view_num=-1,
        use_fixed_view=False,
        is_relative=False,
        is_abstract_action=False,
        vit_dropout_prob=0,
        state_dropout_prob=0,
        # Action related
        sampling_strategy = "step",
        gen_action_type = "action_token",
        unified_state_dim=200,
        unified_action_dim=200,
        history_num=1,
        action_chunk_length=16,
        override_stats_path: Optional[str] = None,
        stats_level: str = 'auto',  # 'auto', 'task', or 'embodiment' for hierarchical stats
        local_rank=0, world_size=1, num_workers=8,
        seed=42,
        **kwargs
    ):

        self.is_train = is_train
        self.logger = logger
        self.world_size = world_size
        self.local_rank = local_rank
        self.num_workers = num_workers

        self.initial_seed = seed
        self.chunk_size = 1000
        self.history_num = history_num
        self.num_used_episodes_per_task = num_used_episodes_per_task or [-1] * len(dataset_path_list)
        self.num_used_episodes_per_dataset = num_used_episodes_per_dataset or [-1] * len(dataset_path_list)
        self.sampling_strategy = sampling_strategy

        self.dataset_name = dataset_name
        # VISUAL
        self.force_image_size = force_image_size
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs if video_backend_kwargs is not None else {}
        self.num_image_tokens = num_image_tokens
        self.vit_dropout_prob = vit_dropout_prob
        self.state_dropout_prob = state_dropout_prob
    
        _, self.vit_transform = build_vit_transform_base(is_train=self.is_train, force_image_size=force_image_size, **vit_transform_args)
        # TOKENIZER AND TEXT
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.prompt_template = prompt_template
        self.instruction_template = INSTRUCTION_TEMPLATE
        self.multi_db_instruction_template = MULTI_DB_INSTRUCT_TEMPLATE

        conv = get_conv_template(self.template_name)
        self.system_prompt = conv.system_message

        # ACTION
        self.gen_action_type = gen_action_type
        self.unified_state_dim = unified_state_dim
        self.unified_action_dim = unified_action_dim
        self.history_num = history_num
        self.action_chunk_length = action_chunk_length
        self.max_view_num = max_view_num
        self.is_relative = is_relative
        self.is_abstract_action = is_abstract_action
        
        self.data_configs = {}
        self.num_used_frames_per_dataset = {}
        self.frame_step_size = {}
        self.sub_dataset_names = []
        self.dataset_name_to_path = {}
        self.modality_meta = {"def": {}, "sampling_indices": {}}

        for i, dataset_path in enumerate(dataset_path_list):
            """stanford_xxx, dorid,...
            for each data_dir, including data, meta, videos
            """
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")
            sub_dataset_name = dataset_path.split("/")[-1]
            self.dataset_name_to_path[sub_dataset_name] = dataset_path
            embodiment_tag = EmbodimentTag(embodiment_tags[i])
            self.data_config_names = data_config_names

            self.data_configs[sub_dataset_name] = DATA_CONFIG_MAP[data_config_names[i]](
                embodiment_tag=embodiment_tag,
                use_fixed_view=use_fixed_view,
                max_view_num=max_view_num,
                obs_indices=[0],
                action_indices=list(range(action_chunk_length)),
            )
            self.modality_meta['def'][sub_dataset_name] = self.data_configs[sub_dataset_name].define_modalities()
            self.modality_meta['sampling_indices'][sub_dataset_name] = self.data_configs[sub_dataset_name].get_sampling_indices()

            self.sub_dataset_names.append(sub_dataset_name)
            self.num_used_frames_per_dataset[sub_dataset_name] = num_used_frames_per_dataset[i] or -1
            self.frame_step_size[sub_dataset_name] = frame_step_size[i] or 1

        self.frame_steping = not all(i==1 for i in frame_step_size)
        self.stats_level = stats_level  # Store for hierarchical metadata

        self.logger.info(f"Preparing baseline statistics for dataset group '{dataset_name}' (stats_level={stats_level})...")
        structured_statistics = self._prepare_statistics(dataset_path_list, stats_level=stats_level)

        if override_stats_path and os.path.exists(override_stats_path):
            self.logger.info(f"Loading OVERRIDE statistics from: {override_stats_path}")
            with open(override_stats_path, 'r') as f:
                full_metadata = json.load(f)
            
            # Extract 'statistics' dictionary from possibly nested JSON
            override_stats = None
            if len(full_metadata) == 1 and 'statistics' in next(iter(full_metadata.values())):
                dataset_group_key = next(iter(full_metadata))
                override_stats = full_metadata[dataset_group_key]['statistics']
                self.logger.info(f"Extracted override statistics for group '{dataset_group_key}'.")
            elif 'state' in full_metadata and 'action' in full_metadata:
                override_stats = full_metadata
                self.logger.info("Loaded override statistics directly from file root.")
            else:
                self.logger.warning(f"Could not find a valid 'statistics' object in {override_stats_path}. Skipping override.")

            if override_stats:
                # Merge state statistics: update structured_statistics with override_stats
                if 'state' in override_stats:
                    num_overridden = len(override_stats['state'])
                    structured_statistics.setdefault('state', {}).update(override_stats['state'])
                    self.logger.info(f"Merged 'state' statistics. Overrode/updated {num_overridden} keys.")

                # Merge action statistics
                if 'action' in override_stats:
                    num_overridden = len(override_stats['action'])
                    structured_statistics.setdefault('action', {}).update(override_stats['action'])
                    self.logger.info(f"Merged 'action' statistics. Overrode/updated {num_overridden} keys.")

        self.info_metas = self._load_info_metas(dataset_path_list)

        self.video_path_template = None
        self.chunk_size = None

        if self.info_metas:
            first_info = next(iter(self.info_metas.values()))
            self.video_path_template = first_info.get("video_path")
            self.chunk_size = first_info.get("chunks_size")

        if not self.video_path_template or not self.chunk_size:
            raise ValueError("Could not find 'video_path' template or 'chunks_size' in any info.json files.")
        
        self.dataset_metadatas = {}
        self.transforms = {}
        self.dataset_fps = {}
        for dataset_name, data_config in self.data_configs.items():
            specific_metadata, fps = self._create_dataset_specific_metadata(
                data_config=data_config,
                global_statistics=structured_statistics
            )
            self.dataset_fps[dataset_name] = fps
            self.dataset_metadatas[dataset_name] = specific_metadata
            transform_pipeline = data_config.get_transforms()
            
            transform_pipeline.set_metadata(specific_metadata)
            self.transforms[dataset_name] = transform_pipeline
        
        self.logger.info(f"Filtering episodes for sampling...")
        selected_parquet_paths_map = self._filter_dataset_episodes(dataset_path_list)
        
        self.logger.info(f"Preparing sample index from filtered data...")
        self.all_steps = self._prepare_sample_index(selected_parquet_paths_map)
        
        self.task_info = self._load_task_info(dataset_path_list)
        self.episode_info = self._load_episode_info(dataset_path_list)

        self.rng = random.Random(self.initial_seed)
        self.set_epoch(seed=self.initial_seed)

        self.logger.info(f"VLA dataset group '{self.dataset_name}' initialized with {self.__len__()/1_000_000:.3f}M total sample units.")

        
    def __len__(self):
        return len(self.all_steps)

    def _load_episode_info(self, dataset_path_list: List[str]) -> dict:
        """
        Load detailed metadata for each episode and preprocess action_config to support fast lookup.
        """
        episode_info = {}
        for dataset_path in dataset_path_list:
            sub_dataset_name = Path(dataset_path).name
            episode_info[sub_dataset_name] = {}
            episodes_path = Path(dataset_path) / "meta/episodes.jsonl"
            if episodes_path.exists():
                with open(episodes_path) as f:
                    for line in f:
                        episode_data = json.loads(line)
                        episode_idx = episode_data['episode_index']

                        # ========== Core optimization: preprocess action_config ==========
                        if 'action_config' in episode_data and episode_data['action_config']:
                            # 1. Ensure action_config is sorted by start_frame, prerequisite for binary search
                            sorted_actions = sorted(
                                episode_data['action_config'],
                                key=lambda x: x.get('start_frame', 0)
                            )

                            # 2. Create two separate lists for binary search
                            starts = [action.get('start_frame', -1) for action in sorted_actions]
                            # Store end_frame and action_text
                            actions_meta = [
                                (action.get('end_frame', -1), action.get('action_text'))
                                for action in sorted_actions
                            ]

                            # 3. Replace original action_config with preprocessed structure
                            episode_data['processed_actions'] = {
                                'starts': starts,
                                'actions_meta': actions_meta
                            }
                            # Can delete original to save memory
                            del episode_data['action_config']

                        episode_info[sub_dataset_name][episode_idx] = episode_data
        return episode_info

    def set_epoch(self, seed, reset_sample=False):
        """
        Set epoch for data sharding across DDP ranks.

        CRITICAL: This function MUST use the same seed across all ranks to ensure
        all GPUs see the same global shuffle before sharding. Different seeds will
        cause data overlap or missing samples across ranks.

        Args:
            seed: Random seed for shuffling (must be same across all ranks)
            reset_sample: Whether to regenerate sample index (for frame stepping)
        """
        # 1. Reset sample index if needed (frame stepping mode)
        if reset_sample and self.frame_steping:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info is not None else 0

            if self.local_rank == 0 and worker_id == 0:
                self.logger.info(f"Resetting sample index at epoch with seed {seed}")
                self.all_steps = self._prepare_sample_index(self.selected_parquet_paths_map)

                # Ensure consistent sample count across epochs
                if len(self.all_steps) < self.all_steps_len:
                    self.all_steps.extend(self.all_steps[:self.all_steps_len - len(self.all_steps)])
                elif len(self.all_steps) > self.all_steps_len:
                    self.all_steps = self.all_steps[len(self.all_steps) - self.all_steps_len:]
                assert len(self.all_steps) == self.all_steps_len

        # 2. Global shuffle (MUST be identical across all ranks)
        # Use numpy's Generator API for better reproducibility
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(self))

        # 3. DDP sharding using array_split (no data loss, handles remainder)
        # array_split ensures all samples are included even when len(self) % world_size != 0
        rank_indices_splits = np.array_split(indices, self.world_size)
        self.episode_idxs_per_rank = rank_indices_splits[self.local_rank]

        # Store for compatibility with existing code
        self.total_episode_idxs = indices
        self.num_files_per_rank = len(self.episode_idxs_per_rank)

    def _prepare_statistics(self, dataset_path_list: List[str], stats_level: str = 'auto') -> Dict:
        """
        Calculate or load statistics for the *full set* of each dataset, then merge.

        Args:
            dataset_path_list: List of dataset paths
            stats_level: 'auto' (task->embodiment fallback), 'task', or 'embodiment'
        """
        all_raw_stats = []
        weights = []
        configs_in_order = []
        self.stats_sources = {}  # Track where stats came from for hierarchical metadata

        for i, dataset_path in enumerate(dataset_path_list):
            sub_dataset_name = self.sub_dataset_names[i]

            _dataset_path = Path(dataset_path)
            stats_path = _dataset_path / "meta" / "stats.json"
            stats_source = None

            if stats_path.exists():
                with open(stats_path, 'r') as f:
                    stats = json.load(f)
                stats_source = str(stats_path)
            else:
                self.logger.info(f"Warning: Statistics file not found at {stats_path}. Calculating from ALL parquet files...")
                # **Important**: glob all files from the entire dataset here
                all_parquet_files = list(_dataset_path.glob("data/**/*.parquet"))
                if not all_parquet_files:
                    self.logger.info(f"No parquet files found in {dataset_path}, skipping stats calculation.")
                    continue
                stats = calculate_dataset_statistics(all_parquet_files)
                with open(stats_path, 'w') as f:
                    json.dump(stats, f, indent=4)
                stats_source = str(stats_path)

            all_raw_stats.append(stats)
            self.stats_sources[sub_dataset_name] = stats_source

            weight = 1
            info_path = _dataset_path / "meta" / "info.json"
            if info_path.exists():
                with open(info_path, 'r') as f:
                    info_data = json.load(f)
                if 'total_frames' in info_data and info_data['total_frames'] > 0:
                    weight = info_data['total_frames']
                    self.logger.info(f"Using weight {weight/1_000_000:.1f}K for {sub_dataset_name} based on 'total_frames'.")
                else:
                    self.logger.info(f"Warning: 'total_frames' not found or is zero in {info_path}. Using default weight 1.")
            weights.append(weight)

            sub_dataset_name_for_config = dataset_path.split("/")[-1]
            configs_in_order.append(self.data_configs[sub_dataset_name_for_config])

            if not all_raw_stats: return {}

        self.logger.info("Merging statistics into a structured format across all datasets...")
        return merge_statistics(all_raw_stats, configs_in_order, weights)
    
    def _load_info_metas(self, dataset_path_list: List[str]) -> Dict[str, Dict]:
        info_metas = {}
        for data_dir_str in dataset_path_list:
            info_path = Path(data_dir_str) / "meta/info.json"
            if info_path.exists():
                with open(info_path, 'r') as f:
                    info_metas[data_dir_str] = json.load(f)
            else:
                self.logger.info(f"Warning: meta/info.json not found in {data_dir_str}")
        return info_metas

    def _create_dataset_specific_metadata(self, data_config, global_statistics):
        statistics_obj = DatasetStatistics.model_validate(global_statistics)
        specific_modality_defs = data_config.define_modalities()

        state_modality_meta = {}
        action_modality_meta = {}
        video_modality_meta = {}
        fps = 30
        
        for key, defn in specific_modality_defs.items():
            modality_type, modality_name = key.split('.', 1)
            
            if modality_type == 'state' or modality_type == 'action':
                dim = defn.end - defn.start

                meta_obj = StateActionMetadata(
                    absolute=defn.absolute,
                    rotation_type=defn.rotation_type,
                    shape=(dim,),
                    continuous=defn.continuous
                )
                
                if modality_type == 'state':
                    state_modality_meta[modality_name] = meta_obj
                else:
                    action_modality_meta[modality_name] = meta_obj
            
            elif modality_type == 'video':
                video_meta_dict, fps = self._get_merged_video_meta(defn.source_column)
                if video_meta_dict:
                    video_modality_meta[modality_name] = VideoMetadata.model_validate(video_meta_dict)
                else:
                    self.logger.warning(f"Could not find video metadata for source column '{defn.source_column}'")

        modalities_obj = DatasetModalities(
            video=video_modality_meta,
            state=state_modality_meta,
            action=action_modality_meta
        )

        final_metadata = DatasetMetadata(
            statistics=statistics_obj,
            modalities=modalities_obj,
            embodiment_tag=data_config.embodiment_tag.value,
        )
        
        return final_metadata, fps

    def _create_mock_metadata(self, structured_statistics):
        """
        Slice and reorganize flat statistics based on raw Parquet columns into the nested structure
        expected by the Pydantic Schema, according to DataConfig definitions.
        Also extract video information from dataset metadata to populate modality metadata.
        """
        # 1. Directly validate structured statistics
        statistics_obj = DatasetStatistics.model_validate(structured_statistics)

        # 2. Merge modality definitions from all data sources
        all_modality_defs = {}
        for config in self.data_configs.values():
            all_modality_defs.update(config.define_modalities())

        # 3. Build structured modality metadata (similar to before, but data source is merged all_modality_defs)
        state_modality_meta = {}
        action_modality_meta = {}
        video_modality_meta = {}

        for key, defn in all_modality_defs.items():
            modality_type, modality_name = key.split('.', 1)

            if modality_type == 'state' or modality_type == 'action':
                dim = defn.end - defn.start

                # --- Key modification: get all information from DataConfig's ModalityDef ---
                meta_obj = StateActionMetadata(
                    absolute=defn.absolute,
                    rotation_type=defn.rotation_type, # Get directly from definition
                    shape=(dim,),
                    continuous=defn.continuous       # Get directly from definition
                )

                if modality_type == 'state':
                    state_modality_meta[modality_name] = meta_obj
                else:
                    action_modality_meta[modality_name] = meta_obj

            elif modality_type == 'video':
                # --- Dynamically get video metadata from loaded info.json ---
                video_meta_dict, fps = self._get_merged_video_meta(defn.source_column)
                video_modality_meta[modality_name] = VideoMetadata.model_validate(video_meta_dict)

        modalities_obj = DatasetModalities(
            video=video_modality_meta,
            state=state_modality_meta,
            action=action_modality_meta
        )

        # --- Part C: Assemble the final DatasetMetadata object ---
        final_metadata = DatasetMetadata(
            statistics=statistics_obj,
            modalities=modalities_obj,
            embodiment_tag="new_embodiment", # Or any placeholder you prefer
        )

        return final_metadata
    
    def _get_merged_video_meta(self, source_column: str) -> Optional[Dict]:
        fps = 30
        for info in self.info_metas.values():
            #print(info)
            if 'features' in info and source_column in info['features']:
                le_video_meta = info['features'][source_column]

                # --- Direct, precise parsing logic ---

                # 1. Get height and width from 'shape' and 'names'
                names_list = le_video_meta["names"]
                shape_list = le_video_meta["shape"]
                height = shape_list[names_list.index("height")]
                width = shape_list[names_list.index("width")]

                # 2. Get channels and fps from 'info' dictionary
                info_dict = le_video_meta["info"] if "info" in le_video_meta else le_video_meta["video_info"]
                channels = info_dict["video.channels"] if "video.channels" in info_dict else shape_list[names_list.index("channel")]
                fps = info_dict["video.fps"]

                return {"resolution": (width, height), "channels": channels, "fps": fps}, fps

        # If no matching source_column found after iterating all info.json files, return None
        return None, fps

    def _filter_dataset_episodes(self, dataset_path_list):
        """
        Filter episodes based on num_used_data and num_used_data_per_task,
        and return a list of filtered (Parquet file path, episode length) tuples for each dataset directory.
        """
        selected_paths_map = {}
        for i, dataset_path in enumerate(dataset_path_list):
            _dataset_path = Path(dataset_path)
            num_episodes_to_use = self.num_used_episodes_per_dataset[i]
            num_per_task = self.num_used_episodes_per_task[i]

            sub_dataset_name = self.sub_dataset_names[i]
            selected_paths_map[sub_dataset_name] = []

            # Load metadata
            episodes_path = _dataset_path / "meta/episodes.jsonl"
            info_path = _dataset_path / "meta/info.json"
            if not (episodes_path.exists() and info_path.exists()):
                self.logger.warning(f"Warning: Metadata not found in {dataset_path}, skipping filtering.")
                continue

            with open(episodes_path, "r") as f:
                all_episodes = [json.loads(line) for line in f]
            with open(info_path, "r") as f:
                info_meta = json.load(f)
                data_path_pattern = info_meta["data_path"]
                chunk_size = info_meta["chunks_size"]

            # Execute filtering logic
            filtered_episodes = []
            if num_episodes_to_use > 0:

                # Ensure there are enough episodes available for sampling
                if len(all_episodes) > num_episodes_to_use:
                    self.logger.info(f"Randomly sampling {num_episodes_to_use/1000:.1f}K episodes for {dataset_path}.")
                    # Use a fixed seed to ensure the same subset is selected each run, ensuring reproducibility
                    # You can set self.initial_seed in __init__
                    rng = random.Random(self.initial_seed)
                    filtered_episodes = rng.sample(all_episodes, num_episodes_to_use)
                else:
                    # If requested amount is greater than or equal to total, use all
                    self.logger.info(f"Using all {num_episodes_to_use/1000:.1f}K episodes for {dataset_path}.")
                    filtered_episodes = all_episodes
                # ----------------------------------------------------

            elif num_per_task > 0:
                # For per-task sampling, it's usually already some form of randomization (since it iterates through the entire list)
                # But if you want sampling within each task to be more random, you can shuffle all_episodes first
                self.logger.info(f"Filtering top {num_per_task/1000:.1f}K episodes per task for {dataset_path}.")

                rng = random.Random(self.initial_seed)
                rng.shuffle(all_episodes)

                task_counts = defaultdict(int)
                for episode in all_episodes:
                    task_key = tuple(episode.get("tasks", []))
                    if task_counts[task_key] < num_per_task:
                        filtered_episodes.append(episode)
                        task_counts[task_key] += 1
            else: # -1 means use all
                filtered_episodes = all_episodes

            # Build (parquet path, length) tuples based on filtered episodes
            for ep in filtered_episodes:
                ep_idx = ep["episode_index"]
                ep_len = ep.get("length", 0) 
                if ep_len == 0:
                    continue
                    
                chunk_idx = ep_idx // chunk_size
                parquet_path = _dataset_path / data_path_pattern.format(
                    episode_chunk=chunk_idx, episode_index=ep_idx
                )
                if parquet_path.exists():
                    selected_paths_map[sub_dataset_name].append((parquet_path, ep_len))
        
        return selected_paths_map

    def _prepare_sample_index(self, selected_paths_map: Dict[str, List[Tuple[Path, int]]]) -> List:
        """
        [Final optimization] Efficiently prepare sampling index based on pre-provided (path, length) metadata.
        """
        index = []
        # Keys of selected_paths_map are sub_dataset_name, values are lists of (Path, length) tuples
        for sub_dataset_name, episode_info_list in selected_paths_map.items():

            if self.sampling_strategy == 'trajectory':
                # Adjust to handle tuple lists
                sub_dataset_index = [(sub_dataset_name, str(path)) for path, length in episode_info_list]
                index.extend(sub_dataset_index)
                self.logger.info(f"Indexed {len(sub_dataset_index)} trajectories for {self.dataset_name}/{sub_dataset_name}")
                continue

            # --- Efficient implementation for 'step' mode ---
            if not episode_info_list:
                continue

            self.logger.info(f"Building manifest from pre-loaded metadata for {len(episode_info_list)} episodes in {sub_dataset_name}...")
            manifest = [{'path': str(path), 'rows': length} for path, length in episode_info_list]
            total_rows = sum(item['rows'] for item in manifest)

            if total_rows == 0:
                continue

            # 2. Determine the final number of frames to sample
            num_frames_to_sample = self.num_used_frames_per_dataset.get(sub_dataset_name, -1)

            if num_frames_to_sample == -1 or num_frames_to_sample >= total_rows:
                self.logger.info(f"Sampling {total_rows/(1_000_000*self.frame_step_size[sub_dataset_name]):.3f}M steps, Using all {total_rows/1_000_000:.3f}M steps with frame_step_size {self.frame_step_size[sub_dataset_name]} from {sub_dataset_name}.")

                sub_dataset_index = []
                for entry in manifest:
                    #for step in range(entry['rows']):
                    #   sub_dataset_index.append((sub_dataset_name, entry['path'], step))

                    start_frame_idx = random.randint(0, self.frame_step_size[sub_dataset_name]-1)
                    for step in range(start_frame_idx, entry['rows'], self.frame_step_size[sub_dataset_name]):
                        sub_dataset_index.append((sub_dataset_name, entry['path'], step))
            else:
                # 3. Efficient weighted sampling
                self.logger.info(f"Performing weighted sampling for {num_frames_to_sample/1_000_000:.3f}M steps from {sub_dataset_name} (total: {total_rows/1_000_000:.3f}M)...")
                
                file_paths = [entry['path'] for entry in manifest]
                weights = np.array([entry['rows'] for entry in manifest], dtype=np.float64)
                probabilities = weights / weights.sum()
                
                sampled_file_indices = np.random.choice(
                    len(file_paths), 
                    size=num_frames_to_sample, 
                    p=probabilities,
                    replace=True
                )
                
                sub_dataset_index = []
                unique_indices, counts = np.unique(sampled_file_indices, return_counts=True)

                for file_idx, num_samples in zip(unique_indices, counts):
                    selected_path = manifest[file_idx]['path']
                    num_rows_in_file = manifest[file_idx]['rows']
                    sampled_steps = np.random.randint(0, num_rows_in_file, size=num_samples)
                    
                    for step in sampled_steps:
                        sub_dataset_index.append((sub_dataset_name, selected_path, int(step)))
                        
                self.logger.info(f"Sampled {len(sub_dataset_index)/1_000_000:.3f}M steps for pretraining for {self.dataset_name}/{sub_dataset_name}")
            index.extend(sub_dataset_index)
            
        if not index:
            self.logger.warning("Warning: No samples were indexed. The dataset might be empty or paths are incorrect.")
        
        return index
    
    def _extract_and_sample_step(self, 
                                df: pd.DataFrame, 
                                base_index: int, 
                                file_path: str,
                                modality_defs: Dict, 
                                sampling_indices: Dict,
                                fps: int = 30) -> Dict[str, np.ndarray]:
        data = {}
        # Preprocess language data
        lang_source_columns = list(set([
            defn.source_column for key, defn in modality_defs.items() if key.startswith('language.')
        ]))
        if lang_source_columns:
            df[lang_source_columns] = df[lang_source_columns].ffill() # Forward fill method, fill missing values with previous valid values

        # Get trajectory length
        trajectory_length = len(df)

        for key, defn in modality_defs.items():
            # Calculate original, unclipped target indices
            original_delta = np.array(sampling_indices.get(key, [0]))

            modality, _ = key.split('.', 1)

            if self.is_abstract_action and modality == 'action':
                delta = np.arange(fps * 2)
            else:
                delta = original_delta

            indices = base_index + delta

            if modality == 'video':
                # Calculate DataFrame row indices to sample (same as state/action)
                padded_indices = np.clip(indices, 0, trajectory_length - 1)

                timestamps_to_fetch = df['timestamp'].iloc[padded_indices].to_numpy()
                # Ensure timestamps is a 1D array
                if timestamps_to_fetch.ndim > 1:
                    timestamps_to_fetch = timestamps_to_fetch.squeeze()

                episode_index = int(Path(file_path).stem.split('_')[-1])

                chunk_index = episode_index // self.chunk_size
                video_key = defn.source_column
                dataset_base_path = Path(file_path).parent.parent.parent
                video_relative_path = self.video_path_template.format(
                    episode_chunk=chunk_index,
                    episode_index=episode_index,
                    video_key=video_key
                )
                video_path = dataset_base_path / video_relative_path

                # Call get_frames_by_timestamps, passing the timestamps we extracted
                frames = get_frames_by_timestamps(
                    video_path=str(video_path),
                    timestamps=timestamps_to_fetch, # Use timestamps instead of frame indices
                    video_backend=self.video_backend,
                    video_backend_kwargs=self.video_backend_kwargs
                )

                data[key] = frames

            # --- Language processing ---
            elif modality == 'language':
                padded_indices = np.clip(indices, 0, trajectory_length - 1)
                data[key] = df[defn.source_column].iloc[padded_indices].values

                # breakpoint()

            # --- Unified processing for state and action ---
            else:
                # Extract complete column data from DataFrame
                raw_col = np.stack(df[defn.source_column].values)
                sliced_col = raw_col[:, defn.start:defn.end]

                # Calculate clipped indices for sampling (for first/last padding)
                padded_indices = np.clip(indices, 0, trajectory_length - 1)
                padded_indices = padded_indices.astype(int)

                # Sample using clipped indices.
                #    This operation implements the "first_last" padding strategy by default.
                #    For all indices both within and outside boundaries, a valid data point is obtained.
                # print(sliced_col)
                # print(padded_indices)
                sampled_data = sliced_col[padded_indices]

                if not defn.absolute:
                    # Calculate which indices are truly out of bounds
                    is_out_of_bounds = (indices < 0) | (indices >= trajectory_length)
                    # If there are out-of-bounds indices and this modality is relative (absolute=False),
                    # fill those positions with 0.
                    if np.any(is_out_of_bounds):
                        sampled_data[is_out_of_bounds] = 0

                if self.is_abstract_action and modality == 'action':
                    target_len = len(original_delta)
                    if target_len > 0:
                        resample_indices = np.linspace(0, len(sampled_data) - 1, target_len).astype(int)
                        data[key] = sampled_data[resample_indices]
                    # else:
                    #     data[key] = np.empty((0, sampled_data.shape[1]))
                else:
                    data[key] = sampled_data
        
        if self.is_relative:
            
            prefixes = ['', 'left_', 'right_']
            
            for prefix in prefixes:
                state_pos_key = f'state.{prefix}eef_position'
                state_rot_key = f'state.{prefix}eef_rotation' # axis-angle
                action_pos_key = f'action.{prefix}eef_position'
                action_rot_key = f'action.{prefix}eef_rotation' # axis-angle

                if all(k in data for k in [state_pos_key, state_rot_key, action_pos_key, action_rot_key]):

                    ref_pos = data[state_pos_key][-1] # Shape: (3,)
                    ref_rot_vec = data[state_rot_key][-1] # Shape: (3,)

                    # Build reference frame rotation matrix (State Rotation Matrix)
                    # Using scipy's Rotation
                    ref_rot_obj = R.from_rotvec(ref_rot_vec)
                    ref_rot_inv = ref_rot_obj.inv() # Inverse, for transforming world coordinates back to local

                    # 2. Get target data (Action)
                    act_pos = data[action_pos_key] # Shape: (N, 3)
                    act_rot_vec = data[action_rot_key] # Shape: (N, 3)

                    # 3. Calculate relative position
                    # Formula: P_local = R_state_inv * (P_action_world - P_state_world)
                    delta_pos_world = act_pos - ref_pos # Broadcast subtraction
                    # apply method can batch rotate (N, 3) vectors
                    rel_pos = ref_rot_inv.apply(delta_pos_world)

                    # 4. Calculate relative rotation
                    act_rot_obj = R.from_rotvec(act_rot_vec)

                    # Calculate relative rotation matrix
                    rel_rot_obj = ref_rot_inv * act_rot_obj

                    # Convert back to axis-angle
                    rel_rot_vec = rel_rot_obj.as_rotvec()

                    # 5. Update Data
                    data[action_pos_key] = rel_pos.astype(np.float32)
                    data[action_rot_key] = rel_rot_vec.astype(np.float32)

        return data

    def _pad_to_max_dim(self, tensor: torch.Tensor, max_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if tensor.numel() == 0:
            return torch.empty(0, max_dim), torch.empty(0, max_dim, dtype=torch.bool)
            
        time_steps, current_dim = tensor.shape
        pad_size = max_dim - current_dim
        if pad_size < 0:
            raise ValueError(f"Tensor dim {current_dim} of {self.dataset_name} exceeds max_dim {max_dim}")
        
        padded_tensor = F.pad(tensor, (0, pad_size), 'constant', 0)
        mask = torch.ones(time_steps, max_dim, dtype=torch.bool)
        if pad_size > 0:
            mask[:, -pad_size:] = False
        return padded_tensor, mask
    
    def _load_task_info(self, dataset_path_list: List[str]) -> dict:
        task_info = {}
        for dataset_path in dataset_path_list:
            sub_dataset_name = Path(dataset_path).name
            task_info[sub_dataset_name] = {}
            tasks_path = Path(dataset_path) / "meta/tasks.jsonl"
            if tasks_path.exists():
                with open(tasks_path) as f:
                    for line in f:
                        task = json.loads(line)
                        task_info[sub_dataset_name][task['task_index']] = task['task']
        return task_info
    
    def fill_instruction_template(self, 
                                  modality_feat_meta, 
                                  view_list, 
                                  embodiment_tag, 
                                  task_description, 
                                  action_chunk_length
        ):
        if self.prompt_template=="short":
            return task_description
        elif self.prompt_template=="long":
            return self.instruction_template.format(task_description=task_description, k=action_chunk_length)
        elif self.prompt_template=="detail":
            state_metas, state_dims = [], []
            action_metas, action_dims = [], []
            for k, d in modality_feat_meta.items():
                if k.startswith("state"):
                    state_metas.append(d[0])
                    state_dims.append(d[1])
                elif k.startswith("action"):
                    action_metas.append(d[0])
                    action_dims.append(d[1])

            arm_type, eef_type = embodiment_tag.split("_")[-2], embodiment_tag.split("_")[-1]
            
            state_dim, state_desc = sum(state_dims), ", ".join(state_metas)
            action_dim, action_desc = sum(action_dims), ", ".join(action_metas)

            detail_instruction = self.multi_db_instruction_template.format(
                view_list=", ".join(view_list),
                arm_type=arm_type, eef_type=eef_type,
                max_state_dim=self.max_state_dim, max_action_dim=self.max_action_dim,
                state_dim=state_dim, state_desc=state_desc,
                action_dim=action_dim, action_desc=action_desc,
                task_description=task_description, 
                k=action_chunk_length
                )
     
            return detail_instruction
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        epoch_count = 0

        while True:
            # ============================================================
            # CRITICAL: Separate sharding seed from operation seed
            # ============================================================

            # A. Sharding Seed (MUST be same across all ranks/workers)
            # Used for set_epoch to ensure all GPUs see the same global shuffle
            sharding_seed = self.initial_seed + epoch_count

            # Execute epoch sharding (updates self.episode_idxs_per_rank)
            reset_sample = (epoch_count > 0)  # Reset sample index after first epoch
            self.set_epoch(sharding_seed, reset_sample)

            # B. Operation Seed (unique per rank + worker)
            # Used for data augmentation, dropout, and base_index sampling
            # MUST differ across workers to avoid identical random behaviors
            # CRITICAL: Ensure seed stays within numpy's valid range [0, 2^32 - 1]
            MAX_NUMPY_SEED = 2**32 - 1
            op_seed = (sharding_seed * 1000) + (self.local_rank * 100) + worker_id
            op_seed = op_seed % MAX_NUMPY_SEED  # Modulo to ensure valid range

            # Seed all three random sources for full reproducibility
            self.rng.seed(op_seed)      # Python random.Random (for self.rng.randint)
            random.seed(op_seed)        # Global random (for dropout in line 1099)
            np.random.seed(op_seed)     # NumPy random (for sampling operations)

            epoch_count += 1

            # ============================================================
            # Worker Splitting (use array_split to avoid data loss)
            # ============================================================

            # Split current rank's data among workers using array_split
            # This ensures all samples are included even when division has remainder
            worker_splits = np.array_split(self.episode_idxs_per_rank, num_workers)
            episode_ids_per_worker = worker_splits[worker_id]

            if len(episode_ids_per_worker) == 0:
                continue

            for episode_id in episode_ids_per_worker: 
                
                step_index = self.all_steps[episode_id]
                if len(step_index)==3:
                    sub_dataset_name, parquet_file_path, base_index = step_index
                else:
                    sub_dataset_name, parquet_file_path = step_index
                    temp_df = pd.read_parquet(parquet_file_path)
                    if len(temp_df) <= 1:
                        continue # Skip trajectories that are too short
                    base_index = self.rng.randint(0, len(temp_df) - 1)

                df = pd.read_parquet(parquet_file_path)

                # print(self.dataset_metadatas[sub_dataset_name])
                # breakpoint()

                fps = self.dataset_fps[sub_dataset_name]
      
                raw_data = self._extract_and_sample_step(
                    df, base_index, parquet_file_path,
                    modality_defs=self.data_configs[sub_dataset_name].define_modalities(), # Pass current config
                    sampling_indices=self.data_configs[sub_dataset_name].get_sampling_indices(),
                    fps=fps
                )

                language_data = {}
                numerical_data = {}
                video_data = []
                for k, v in raw_data.items():
                    if k.startswith('language.'):
                        language_data[k] = v
                    elif k.startswith('video.'):
                        video_data.append(v)
                    else:
                        numerical_data[k] = v
            
                transformed_data = self.transforms[sub_dataset_name](numerical_data)

                current_config = self.data_configs[sub_dataset_name]
                mapping = current_config.UNIFIED_MAPPING

                # Use self.history_num and self.action_chunk_length as expected step lengths
                T_state = self.history_num
                T_action = self.action_chunk_length
                
                state_data = torch.zeros(T_state, self.unified_state_dim, dtype=torch.float32)
                
                action_data = torch.zeros(T_action, self.unified_action_dim, dtype=torch.float32)
                action_mask = torch.zeros(T_action, self.unified_action_dim, dtype=torch.bool)
                
                drop_state_cond = (self.state_dropout_prob > 1e-9) and (random.random() < self.state_dropout_prob)     # random view num
                for key, (start, end) in mapping.items():
                    if key in transformed_data:
                        source_tensor = transformed_data[key]
                        
                        if key.startswith('state.') and not drop_state_cond:
                                state_data[:, start:end] = source_tensor
                        elif key.startswith('action.'):
                            action_data[:, start:end] = source_tensor
                            action_mask[:, start:end] = True
            

                instruction_key = next(iter(language_data.keys()), None)
                if instruction_key:
                    instruction_index_or_text = language_data[instruction_key][0]
                    if isinstance(instruction_index_or_text, (int, np.integer)):
                        # sub_dataset_name = Path(parquet_file_path).parts[-4]
                        task_description = self.task_info.get(sub_dataset_name, {}).get(instruction_index_or_text, "Unknown Task")
                    else:
                        task_description = str(instruction_index_or_text)
                else:
                    task_description = "Default Task"

                if 'agibot' in sub_dataset_name:
                    episode_index = int(Path(parquet_file_path).stem.split('_')[-1])
                    current_episode_data = self.episode_info.get(sub_dataset_name, {}).get(episode_index)

                    if current_episode_data and 'processed_actions' in current_episode_data:
                        processed = current_episode_data['processed_actions']
                        starts = processed['starts']
                        actions_meta = processed['actions_meta']

                        idx = bisect.bisect_right(starts, base_index)
                        
                        if idx > 0:
                            candidate_idx = idx - 1
                            end_frame, sub_action_text = actions_meta[candidate_idx]
                            if base_index <= end_frame and sub_action_text:
                                task_description += f" | Current Sub Task: {sub_action_text}"
                    

                embodiment_tag = self.data_configs[sub_dataset_name].embodiment_tag
                tag_string = embodiment_tag
                default_id = EMBODIMENT_TAG_MAPPING[EmbodimentTag.NEW_EMBODIMENT.value]
                embodiment_id = EMBODIMENT_TAG_MAPPING.get(tag_string, default_id)

                packet = {
                    'sequence_plan': [], 'text_ids_list': [], 'image_tensor_list': [],
                    'state_tensor_list': [], 'action_tensor_list': [], 'num_tokens': 0,
                    'embodiment_id': embodiment_id, 
                    'action_mask': action_mask, 
                }

                system_prompt = f"system\n{self.system_prompt}"
                text_ids = self.tokenizer.encode(system_prompt)
                packet['text_ids_list'].append(text_ids)
                packet['sequence_plan'].append({'type': 'text', 'has_loss': 0, 'enable_cfg': 0,
                                                'special_token_loss': 0, 'special_token_label': None,
                                                'is_bos': True, 'is_eos': True})
                packet['num_tokens'] += len(text_ids) + 2 +1 # bos & eos\n

                # ASSISTANT: add assistant\n
                text_ids = self.tokenizer.encode(f"user\n")
                packet['text_ids_list'].append(text_ids)
                packet['sequence_plan'].append({'type': 'text', 'has_loss': 0, 'enable_cfg': 0,
                                                'special_token_loss': 0, 'special_token_label': None,
                                                'is_bos': True, 'is_eos': False})
                packet['num_tokens'] += len(text_ids) + 1 # bos

                # add vision
                num_views = len(video_data)
                if self.vit_dropout_prob > 1e-9 and num_views > 1:
                    drop_decisions = [random.random() < self.vit_dropout_prob for _ in range(num_views)]
                    
                    if all(drop_decisions):
                        keep_index = random.randint(0, num_views - 1)
                        drop_decisions[keep_index] = False
                else:
                    drop_decisions = [False] * num_views

                for v_idx, view_group in enumerate(video_data):
                    drop_this_view = drop_decisions[v_idx]

                    for frame in view_group:           
                        image_tensor = self.vit_transform(Image.fromarray(frame)).unsqueeze(0)
                        #image_tensor.save("old.jpg")
           
                        packet['sequence_plan'].append({'type': 'vit_image', 'has_loss': 0, 'enable_cfg': 0, 
                                            'special_token_loss': 0, 'special_token_label': None,
                                            'num_image_tokens': self.num_image_tokens,
                                            'drop_vit_cond': drop_this_view,
                                            'is_bos': False, 'is_eos': False})
                        packet['num_tokens'] += self.num_image_tokens + 2 # vision_start & vision_end
              
                        packet['image_tensor_list'].append(image_tensor)
           
                # add state
                packet['state_tensor_list'].append(state_data)
                packet['sequence_plan'].append({'type': 'state', 'has_loss': 0, 'enable_cfg': 0,
                                                'special_token_loss': 0, 'special_token_label': None,
                                                'is_bos': False, 'is_eos': False})
                packet['num_tokens'] += state_data.shape[0] + 2 # state_start & state_end
                
                # add text
                modality_feat_meta = self.data_configs[sub_dataset_name].get_feature_meta()
                
                instruction = self.fill_instruction_template(modality_feat_meta=modality_feat_meta,
                                                             view_list=[k.split(".")[-1] for k in raw_data.keys() if k.startswith("video")],
                                                             embodiment_tag=embodiment_tag.value,
                                                             task_description=task_description, 
                                                             action_chunk_length=self.action_chunk_length)
                text_ids = self.tokenizer.encode(instruction)
                packet['text_ids_list'].append(text_ids)
                packet['sequence_plan'].append({'type': 'text', 'has_loss': 0, 'enable_cfg': 0,
                                                'special_token_loss': 0, 'special_token_label': None,
                                                'is_bos': False, 'is_eos': True})
                packet['num_tokens'] += len(text_ids) + 1 +1 # eos\n

                # assistant
                text_ids = self.tokenizer.encode(f"assistant\n")
                packet['text_ids_list'].append(text_ids)
                packet['sequence_plan'].append({'type': 'text', 'has_loss': 0, 'enable_cfg': 0,
                                                'special_token_loss': 0, 'special_token_label': None,
                                                'is_bos': True, 'is_eos': False})
                packet['num_tokens'] += len(text_ids) + 1 # bos

                # add action, action_tensor shape: (T_action, D_action_padded)
                packet['action_tensor_list'].append(action_data)
                packet['sequence_plan'].append({'type': 'action', 'has_loss': 1, 'enable_cfg': 0, 
                                                'special_token_loss': 0, 'special_token_label': None,
                                                'is_bos': False, 'is_eos': False})
                packet['num_tokens'] += action_data.shape[0]

                packet['text_ids_list'].append([])
                packet['sequence_plan'].append({'type': 'text', 'has_loss': 0, 'enable_cfg': 0, 
                                                'special_token_loss': 0, 'special_token_label': None,
                                                'is_bos': False, 'is_eos': True, 'is_end': True})
                packet['num_tokens'] += 1
                #breakpoint()
                yield packet



