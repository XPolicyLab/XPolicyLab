# Copyright 2025 MemoryMatters Team. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Yuran Wang / Peking University] in [2025].
"""
Random Episode Iterable Dataset Loader

This module provides an IterableDataset that randomly selects episodes per rank
and sequentially reads frames within each episode. This ensures that:
- Each rank is assigned a specific set of episodes (episode % world_size == rank)
- Episodes are randomly shuffled per rank
- Frames are read sequentially within each episode
- When an episode is exhausted, randomly selects the next episode
- Supports infinite iteration (cycles through episodes)

Key Features:
- Episode-level sharding: Episodes are never split across ranks
- Random episode selection: Episodes are randomly shuffled per rank
- Sequential frame reading: Frames within each episode are read in order
- Batch compatibility: Works with MemoryBank's episode transition handling

Usage:
    >>> from source.dataloader.random_episode_dataloader import RandomEpisodeIterableDataset
    >>> from source.dataloader.dataset_min_max import LeRobot_Dataset
    >>> from torch.utils.data import DataLoader
    >>> 
    >>> base_dataset = LeRobot_Dataset(...)
    >>> iterable_dataset = RandomEpisodeIterableDataset(
    ...     base_dataset=base_dataset,
    ...     rank=0,
    ...     world_size=2,
    ...     shuffle=True,
    ...     seed=42
    ... )
    >>> dataloader = DataLoader(iterable_dataset, batch_size=16, num_workers=4)  # Supports multiple workers
"""
import os
import sys
import json
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from typing import Dict, List, Optional
import torch
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info
from termcolor import cprint
from source.utils import BOLD, RESET_BOLD


class RandomEpisodeIterableDataset(IterableDataset):
    """
    IterableDataset that randomly selects episodes per rank and sequentially reads frames.
    
    Key Features:
    - Each rank is assigned a specific set of episodes (episode % world_size == rank)
    - Episodes are randomly shuffled per rank
    - Frames are read sequentially within each episode
    - When an episode is exhausted, randomly selects the next episode
    - Supports infinite iteration (cycles through episodes)
    - Multi-worker support: Episodes are split among workers, each worker has independent RNG
    
    Usage:
        >>> base_dataset = LeRobot_Dataset(...)
        >>> iterable_dataset = RandomEpisodeIterableDataset(
        ...     base_dataset=base_dataset,
        ...     rank=0,
        ...     world_size=2,
        ...     shuffle=True,
        ...     seed=42
        ... )
        >>> dataloader = DataLoader(iterable_dataset, batch_size=16, num_workers=4)  # Supports multiple workers
    """
    
    def __init__(
        self,
        base_dataset,
        rank: int,
        world_size: int,
        shuffle: bool = True,
        seed: Optional[int] = None,
        infinite: bool = True,
    ):
        """
        Initialize RandomEpisodeIterableDataset.
        
        Args:
            base_dataset: LeRobot_Dataset instance (or any dataset with episode_id in samples)
            rank: Current process rank (0-indexed)
            world_size: Total number of processes/ranks
            shuffle: Whether to shuffle episode order (default: True)
            seed: Random seed for shuffling (if shuffle=True)
            infinite: Whether to cycle through episodes infinitely (default: True)
        """
        self.base_dataset = base_dataset
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.seed = seed
        self.infinite = infinite
        
        # Build episode to frame indices mapping
        self._build_episode_mapping()
        
        # Assign episodes to this rank
        self._assign_rank_episodes()
    
    def _build_episode_mapping(self):
        """
        Build mapping from episode_id to list of frame indices.
        First tries to read from episodes.jsonl (fast), falls back to scanning dataset (slow).
        """
        # Try to read from episodes.jsonl first (fast method)
        episodes_jsonl_path = self._get_episodes_jsonl_path()
        
        if episodes_jsonl_path and episodes_jsonl_path.exists():
            if self.rank == 0:
                cprint(f"[RandomEpisodeDataset] Reading episode mapping from {episodes_jsonl_path}...", "cyan")
            self._build_episode_mapping_from_jsonl(episodes_jsonl_path)
        else:
            # Fallback: scan dataset (slow method)
            if self.rank == 0:
                cprint(f"[RandomEpisodeDataset] Building episode mapping by scanning {len(self.base_dataset)} samples...", "cyan")
            self._build_episode_mapping_from_scan()
    
    def _get_episodes_jsonl_path(self) -> Optional[Path]:
        """
        Get path to episodes.jsonl file from base_dataset.
        
        Returns:
            Path to episodes.jsonl if base_dataset has root attribute, None otherwise.
        """
        # Try to get root path from base_dataset
        if hasattr(self.base_dataset, 'dataset') and hasattr(self.base_dataset.dataset, 'root'):
            # LeRobot_Dataset wraps LeRobot_Selective_Dataset
            root = self.base_dataset.dataset.root
        elif hasattr(self.base_dataset, 'root'):
            # Direct LeRobotDataset
            root = self.base_dataset.root
        else:
            return None
        
        # Construct path to episodes.jsonl
        episodes_jsonl_path = Path(root) / "meta" / "episodes.jsonl"
        return episodes_jsonl_path
    
    def _build_episode_mapping_from_jsonl(self, episodes_jsonl_path: Path):
        """
        Build episode mapping from episodes.jsonl file (fast method).
        
        Args:
            episodes_jsonl_path: Path to episodes.jsonl file
        """
        # Dictionary: episode_id -> list of sample indices
        self.episode_to_indices: Dict[int, List[int]] = {}
        
        # Read episodes.jsonl
        episode_lengths = {}
        with open(episodes_jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    episode_data = json.loads(line)
                    episode_index = episode_data.get("episode_index")
                    episode_length = episode_data.get("length")
                    if episode_index is not None and episode_length is not None:
                        episode_lengths[episode_index] = episode_length
        
        # Build mapping: assume frames are stored sequentially by episode
        current_idx = 0
        for episode_id in sorted(episode_lengths.keys()):
            length = episode_lengths[episode_id]
            # Create list of indices for this episode
            self.episode_to_indices[episode_id] = list(range(current_idx, current_idx + length))
            current_idx += length
        
        if self.rank == 0:
            total_episodes = len(self.episode_to_indices)
            total_frames = sum(len(indices) for indices in self.episode_to_indices.values())
            cprint(
                f"[RandomEpisodeDataset] Built mapping from JSONL: {BOLD}{total_episodes}{RESET_BOLD} episodes, "
                f"{BOLD}{total_frames}{RESET_BOLD} total frames",
                "green"
            )
    
    def _build_episode_mapping_from_scan(self):
        """
        Build episode mapping by scanning the entire dataset (slow method, fallback).
        """
        # Dictionary: episode_id -> list of sample indices
        self.episode_to_indices: Dict[int, List[int]] = {}
        
        # Scan dataset to build episode mapping
        for idx in range(len(self.base_dataset)):
            try:
                sample = self.base_dataset[idx]
                episode_id = sample.get("episode_id")
                
                # Handle different episode_id types
                if isinstance(episode_id, torch.Tensor):
                    episode_id = int(episode_id.item())
                elif isinstance(episode_id, np.generic):
                    episode_id = int(episode_id)
                else:
                    episode_id = int(episode_id)
                
                if episode_id not in self.episode_to_indices:
                    self.episode_to_indices[episode_id] = []
                self.episode_to_indices[episode_id].append(idx)
            except Exception as e:
                if self.rank == 0:
                    cprint(f"[RandomEpisodeDataset] Warning: Failed to process sample {idx}: {e}", "yellow")
                continue
        
        # Sort indices within each episode to ensure sequential reading
        for episode_id in self.episode_to_indices:
            self.episode_to_indices[episode_id].sort()
        
        if self.rank == 0:
            total_episodes = len(self.episode_to_indices)
            total_frames = sum(len(indices) for indices in self.episode_to_indices.values())
            cprint(
                f"[RandomEpisodeDataset] Built mapping from scan: {BOLD}{total_episodes}{RESET_BOLD} episodes, "
                f"{BOLD}{total_frames}{RESET_BOLD} total frames",
                "green"
            )
    
    def _assign_rank_episodes(self):
        """
        Assign episodes to this rank based on episode_id % world_size == rank.
        """
        all_episode_ids = sorted(self.episode_to_indices.keys())
        self.rank_episodes = [eid for eid in all_episode_ids if eid % self.world_size == self.rank]
        
        if self.rank == 0:
            cprint(
                f"[RandomEpisodeDataset] Rank {self.rank}: assigned {BOLD}{len(self.rank_episodes)}{RESET_BOLD} "
                f"episodes out of {BOLD}{len(all_episode_ids)}{RESET_BOLD} total",
                "cyan"
            )
    
    def __len__(self):
        """
        Return total number of frames assigned to this rank.
        
        Note: For IterableDataset with multiple workers, PyTorch calls __len__() 
        before worker processes are created, so we cannot use get_worker_info() here.
        
        For infinite datasets, returns a large number to avoid warnings about 
        length mismatch when DataLoader fetches more samples than reported length.
        """
        # Calculate total frames for this rank
        total = sum(len(self.episode_to_indices[eid]) for eid in self.rank_episodes)
        
        # For infinite datasets, return a large number to avoid warnings
        # This is necessary because:
        # 1. Each worker will cycle through episodes infinitely
        # 2. DataLoader will fetch many more samples than the base length
        # 3. Returning a large number prevents the warning while still allowing DataLoader to work
        if self.infinite:
            import sys
            # Return a large number that represents "effectively infinite"
            # Use a reasonable multiplier to avoid the warning
            # The actual number fetched will be much larger, but this prevents the warning
            return min(total * 10000, sys.maxsize // 2)  # Cap at reasonable value
        
        return total
    
    def __iter__(self):
        """
        Iterate through episodes randomly, reading frames sequentially within each episode.
        Supports multiple workers by splitting episodes among workers.
        
        Shuffle strategy:
        - Each worker is assigned a fixed set of episodes (only assigned once)
        - Within each cycle, the worker shuffles its fixed episodes
        - This ensures no episode overlap between workers while maintaining randomness
        
        Key fix: Fixed episode assignment prevents multiple workers from processing
        the same episode simultaneously, which would cause frame order confusion in batches.
        """
        worker_info = get_worker_info()
        
        # Create random number generator for worker-level shuffling
        worker_rng = np.random.RandomState(self.seed if self.seed is not None else None)
        
        # Fixed episode assignment (only done once, not per cycle)
        # This prevents episode overlap between workers when they enter different cycles
        if worker_info is not None:
            # Multi-worker mode: assign fixed episodes to this worker
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            
            num_episodes = len(self.rank_episodes)
            episodes_per_worker = num_episodes // num_workers
            remainder = num_episodes % num_workers
            
            # Calculate start and end indices for this worker
            start_idx = worker_id * episodes_per_worker + min(worker_id, remainder)
            end_idx = start_idx + episodes_per_worker + (1 if worker_id < remainder else 0)
            
            fixed_worker_episodes = self.rank_episodes[start_idx:end_idx]
            
            if self.rank == 0 and worker_id == 0:
                cprint(
                    f"[RandomEpisodeDataset] Worker {worker_id}: fixed assignment of {BOLD}{len(fixed_worker_episodes)}{RESET_BOLD} "
                    f"episodes out of {BOLD}{num_episodes}{RESET_BOLD} total episodes for rank {self.rank}",
                    "cyan"
                )
        else:
            # Single-worker mode: use all rank_episodes
            fixed_worker_episodes = self.rank_episodes
        
        def _gen():
            cycle_count = 0
            while True:
                # Each cycle: shuffle only the fixed episodes assigned to this worker
                # This maintains randomness while preventing episode overlap
                worker_episodes = fixed_worker_episodes.copy()
                if self.shuffle:
                    worker_rng.shuffle(worker_episodes)
                
                # Get current worker_id for logging
                current_worker_id = worker_info.id if worker_info is not None else 0
                
                # Log episode order (if first cycle)
                if self.rank == 0 and current_worker_id == 0 and cycle_count == 0:
                    cprint(
                        f"[RandomEpisodeDataset] Worker {current_worker_id}: Starting iteration cycle {cycle_count}, "
                        f"episode order: {worker_episodes[:5]}..." if len(worker_episodes) > 5 else f"episode order: {worker_episodes}",
                        "cyan"
                    )
                
                # Iterate through worker's fixed episodes (shuffled within this cycle)
                for episode_id in worker_episodes:
                    # Get frame indices for this episode (already sorted)
                    frame_indices = self.episode_to_indices[episode_id]
                    
                    # Sequentially yield frames from this episode
                    for idx in frame_indices:
                        try:
                            sample = self.base_dataset[idx]
                            # Filter out samples with lang == "null"
                            if sample.get("lang") != "null":
                                yield sample
                        except Exception as e:
                            if self.rank == 0 and current_worker_id == 0:
                                cprint(f"[RandomEpisodeDataset] Warning: Failed to load sample {idx}: {e}", "yellow")
                            continue
                
                cycle_count += 1
                
                # If not infinite, stop after one cycle
                if not self.infinite:
                    break
                
                # For infinite mode, continue to next cycle
                if self.rank == 0 and current_worker_id == 0:
                    cprint(
                        f"[RandomEpisodeDataset] Worker {current_worker_id}: Completed cycle {cycle_count}, starting next cycle...",
                        "cyan"
                    )
        
        return _gen()

