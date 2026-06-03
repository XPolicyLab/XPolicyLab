"""
Sequential Episode DataLoader implementation.

This DataLoader initializes batches by selecting random episodes and then
returning samples sequentially from each episode starting from the first step.
When reset, it selects new random episodes and starts from the first step again.
"""

from collections.abc import Iterator
import logging
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import random
from typing import Literal, Optional, List, Dict, Any, Callable, Tuple

import jax
import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoaderBase
from torch.utils.data import Sampler

import openpi_value.training.config as _config
import openpi_value.models.model as _model


class SequentialEpisodeSampler(Sampler):
    """
    Sampler that samples batches where each element in the batch has its own independent episode sequence.
    
    Each element in the batch maintains its own episode state and can be reset independently.
    
    Args:
        dataset: The dataset to sample from
        batch_size: Number of elements (episodes) in each batch
        shuffle_episodes: Whether to shuffle the episodes when resetting
        seed: Random seed for reproducibility
    """
    def __init__(
        self,
        dataset,
        batch_size: int,
        shuffle_episodes: bool = True,
        seed: int = 0
    ):
        self.dataset = dataset._dataset
        
        # * self.dataset is correct
        # * CustomMultiLeRobotDataset
        
        self.batch_size = batch_size
        self.shuffle_episodes = shuffle_episodes
        self.seed = seed
        self.rng = random.Random(seed)
        
        # Get all unique episode indices in the dataset
        # self.episode_indices = self._get_unique_episode_indices()
        # self.num_episodes = len(self.episode_indices)
        
        self.episode_to_frames: Dict[int, List[int]] = {}
        # 1. Handle Multi/Concat Datasets
        if hasattr(self.dataset, '_datasets'):
            current_frame_offset = 0
            global_ep_counter = 0
            
            for sub_ds in self.dataset._datasets:
                # Extract frames for this sub-dataset (returns {local_ep_id: [local_frames]})
                sub_mapping = self._extract_frames_from_single_ds(sub_ds)
                
                # Remap to global IDs and global Frame indices
                for local_ep_id, local_frames in sub_mapping.items():
                    # Create a new unique ID for the sampler to use
                    unique_ep_id = global_ep_counter
                    
                    # Shift frame indices by the accumulated offset
                    global_frames = [f + current_frame_offset for f in local_frames]
                    
                    if len(global_frames) > 0:
                        self.episode_to_frames[unique_ep_id] = global_frames
                        global_ep_counter += 1
                
                # Increment offset for the next dataset
                current_frame_offset += len(sub_ds)
                
        # 2. Handle Single Dataset
        else:
            # Standard extraction
            sub_mapping = self._extract_frames_from_single_ds(self.dataset)
            # Just verify length > 0
            self.episode_to_frames = {k: v for k, v in sub_mapping.items() if len(v) > 0}
        
        
        # Create a mapping from episode index to list of frame indices in that episode
        # self.episode_to_frames = self._create_episode_to_frames_mapping()
        
        # Filter out any episodes that don't have frames
        # self.episode_to_frames = {k: v for k, v in self.episode_to_frames.items() if len(v) > 0}
        # self.episode_indices = [ep_idx for ep_idx in self.episode_indices if ep_idx in self.episode_to_frames]
        
        self.episode_indices = sorted(list(self.episode_to_frames.keys()))
        self.num_episodes = len(self.episode_indices)
        self.step_chunk_size = 1
        
        # Check if we have enough episodes
        if self.num_episodes < 1:
            raise ValueError(f"Not enough episodes in dataset. Need at least 1, "
                             f"but only found {self.num_episodes}")
        
        # Initialize state for each element in the batch
        # Each element has its own episode and step counter
        self.element_states = []
        for i in range(batch_size):
            # Create a separate RNG for each element to ensure independent sampling
            element_rng = random.Random(seed + i)
            self.element_states.append({
                'current_episode': None,
                'current_step': 0,
                'rng': element_rng
            })
        
        # Reset all elements to select initial episodes
        self.reset()
        
        logger.info(f"SequentialEpisodeSampler initialized with {self.num_episodes} episodes")
        logger.info(f"Batch size: {self.batch_size}, shuffle_episodes: {self.shuffle_episodes}")
        
        
    def _extract_frames_from_single_ds(self, dataset) -> Dict[int, List[int]]:
        """
        Helper to extract {local_ep_id: [local_frame_indices]} from a single dataset.
        Does NOT handle global offsets.
        """
        mapping = {}
        
        has_skipping = (
            # hasattr(dataset, 'preceding_skipping_ratio') and 
            # dataset.preceding_skipping_ratio > 0. and 
            hasattr(dataset, 'index_map') and 
            dataset.index_map is not None
        )

        # Optimized path for LeRobot/Custom datasets
        if hasattr(dataset, 'episode_data_index') and hasattr(dataset, 'episodes'):
            # We must reconstruct the VIRTUAL indices.
            # The dataset's index_map is built sequentially based on the order of episodes.
            # So we maintain a running counter of virtual indices.
            virtual_idx_counter = 0
            
            # dataset.episodes is the list of episode_indices used to init the dataset.
            # These correspond exactly to the order in episode_data_index['from'] / ['to']
            num_episodes = len(dataset.episode_data_index['from'])
            
            for arr_idx in range(num_episodes):
                # 1. Get raw info
                start = dataset.episode_data_index['from'][arr_idx].item()
                end = dataset.episode_data_index['to'][arr_idx].item()
                raw_len = end - start
                
                # 2. Calculate valid length (matching Dataset logic EXACTLY)
                if has_skipping:
                    # Get start skip ratio
                    ratio_start = getattr(dataset, 'preceding_skipping_ratio', 0.0)
                    skip_n_start = int(raw_len * ratio_start)
                    
                    # Get end skip ratio
                    ratio_end = getattr(dataset, 'trailing_skipping_ratio', 0.0)
                    skip_n_end = int(raw_len * ratio_end)

                    # Replicate Safety Logic from Dataset to ensure counts align
                    max_allowed_skip_end = raw_len - skip_n_start - 1
                
                    if max_allowed_skip_end < 0:
                        # If start skip is already consuming the whole episode
                        skip_n_start = raw_len - 1
                        skip_n_end = 0
                    else:
                        # Cap end skip if it overlaps with start skip
                        skip_n_end = min(skip_n_end, max_allowed_skip_end)

                    valid_len = raw_len - skip_n_start - skip_n_end
                else:
                    valid_len = raw_len

                # 3. Retrieve the actual Episode ID (the key for the dict)
                # dataset.episodes is the list [ep_idx_1, ep_idx_5, ...] 
                # arr_idx is the index in that list.
                ep_idx = dataset.episodes[arr_idx]
                
                # 4. Generate VIRTUAL indices [current_counter, current_counter + len]
                # When passed to __getitem__, dataset will map these -> index_map -> Raw
                mapping[ep_idx] = list(range(virtual_idx_counter, virtual_idx_counter + valid_len))
                
                # 5. Advance the virtual counter
                virtual_idx_counter += valid_len
        
        # # Optimized path for LeRobot/Custom datasets
        # if hasattr(dataset, 'episode_data_index') and hasattr(dataset, 'ep_idx_to_arr_idx'):
        #     for ep_idx, arr_idx in dataset.ep_idx_to_arr_idx.items():
        #         start = dataset.episode_data_index['from'][arr_idx].item()
        #         end = dataset.episode_data_index['to'][arr_idx].item()
        #         mapping[ep_idx] = list(range(start, end))
                
        # Fallback path (Safety net)
        else:
            # Only use this if absolute necessary, it is slow
            # You might want to skip this or implement specific logic for your other dataset types
            # pass 
            raise NotImplementedError("Dataset type not supported in multi-dataset extraction.")
            
        return mapping



    # def _create_episode_to_frames_mapping(self) -> Dict[int, List[int]]:
    #     """Create a mapping from episode index to list of frame indices in that episode"""
    #     episode_to_frames = {}
        
    #     if hasattr(self.dataset, 'episode_data_index'):
    #         # For CustomLeRobotDataset which has episode_data_index
    #         for ep_idx in self.episode_indices:
    #             if ep_idx in self.dataset.ep_idx_to_arr_idx:
    #                 arr_idx = self.dataset.ep_idx_to_arr_idx[ep_idx]
    #                 start = self.dataset.episode_data_index['from'][arr_idx].item()
    #                 end = self.dataset.episode_data_index['to'][arr_idx].item()
    #                 episode_to_frames[ep_idx] = list(range(start, end))
    #     else:
    #         # Fallback for other dataset types
    #         for i in range(len(self.dataset)):
    #             item = self.dataset[i]
    #             ep_idx = item['episode_index'].item()
    #             if ep_idx not in episode_to_frames:
    #                 episode_to_frames[ep_idx] = []
    #             episode_to_frames[ep_idx].append(i)
        
    #     return episode_to_frames



    def reset(self, reset_episode_state: Optional[List[Tuple[int, int]]] = None, element_indices: Optional[List[int]] = None) -> None:
        """
        Reset the sampler: select new random episodes for elements and start from the first step.
        
        Args:
            element_indices: If specified, only reset these particular elements. 
                           If None, reset all elements.
        """
        # Calculate episode lengths if not already done
        if not hasattr(self, 'episode_lengths'):
            self.episode_lengths: Dict[int, int] = {ep: len(frames) for ep, frames in self.episode_to_frames.items()}
        
        # Determine which elements to reset
        elements_to_reset = element_indices if element_indices is not None else range(self.batch_size)
        
        for idx in elements_to_reset:
            if idx < 0 or idx >= self.batch_size:
                raise ValueError(f"Invalid element index: {idx}. Must be between 0 and {self.batch_size - 1}")
            
            element_state = self.element_states[idx]
            if reset_episode_state is not None:
                element_state['current_episode'] = reset_episode_state[idx][0]
                element_state['current_step'] = reset_episode_state[idx][1]
            else:
                # Select a random episode for this element
                if self.shuffle_episodes:
                    # Select a random episode
                    selected_ep = element_state['rng'].choice(self.episode_indices)
                else:
                    # Select in order, cycling if necessary
                    current_ep = element_state['current_episode'] if element_state['current_episode'] is not None else 0
                    current_idx = self.episode_indices.index(current_ep) if current_ep in self.episode_indices else 0
                    selected_ep = self.episode_indices[(current_idx + 1) % self.num_episodes]
                
                element_state['current_episode'] = selected_ep
                element_state['current_step'] = element_state['rng'].randint(0, self.episode_lengths[element_state['current_episode']] - 1) 
            
                logger.debug(f"Element {idx} reset. Selected episode: {selected_ep}, starting at step 0")

    def __iter__(self):
        """
        Iterate over the dataset, yielding batches where each element in the batch
        independently progresses through its own episode sequence.
        """
        # Calculate episode lengths if not already done
        if not hasattr(self, 'episode_lengths'):
            self.episode_lengths: Dict[int, int] = {ep: len(frames) for ep, frames in self.episode_to_frames.items()}
        
        while True:
            batch_indices = []
            
            for i in range(self.batch_size):
                element_state = self.element_states[i]
                
                # Check if we've reached the end of this element's current episode
                if element_state['current_step'] >= self.episode_lengths[element_state['current_episode']]:
                    # This episode is exhausted, replace it with a new one
                    new_ep_idx = self._get_new_episode(element_state['current_episode'], element_state['rng'])
                    element_state['current_episode'] = new_ep_idx
                    element_state['current_step'] = element_state['rng'].randint(0, self.episode_lengths[element_state['current_episode']] - 1) 
                
                # Get the frame index for this element's current step
                frame_idx = self.episode_to_frames[element_state['current_episode']][element_state['current_step']]
                batch_indices.append(frame_idx)
                
                # Move to next step for this element
                element_state['current_step'] += self.step_chunk_size
            
            # We should always have exactly batch_size indices now
            assert len(batch_indices) == self.batch_size, \
                f"Batch size mismatch: expected {self.batch_size}, got {len(batch_indices)}"
            
            yield batch_indices

    def _get_new_episode(self, exclude_ep_idx: int, rng: Optional[random.Random] = None) -> int:
        """Get a new episode index that is not the excluded one"""
        # Use the provided RNG or fall back to the main RNG
        rng = rng or self.rng
        
        # Create a list of available episodes excluding the exhausted one
        available_episodes = [ep for ep in self.episode_indices if ep != exclude_ep_idx]
        
        # If all episodes are excluded (shouldn't happen with proper batch_size),
        # just return the excluded one
        if not available_episodes:
            return exclude_ep_idx
        
        # Select a random episode from available ones
        return rng.choice(available_episodes)

    def __len__(self):

        return len(self.dataset)


class SequentialEpisodeDataLoader:
    """
    DataLoader that initializes batches by selecting episodes and then returning
    samples sequentially from each episode starting from the first step.
    
    Args:
        dataset: The dataset to load data from
        batch_size: Number of episodes to include in each batch
        shuffle_episodes: Whether to shuffle the episodes when resetting
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory in DataLoader
        seed: Random seed for reproducibility
        framework: The framework to use ("jax" or "pytorch")
        sharding: JAX sharding configuration (only for JAX framework)
    """
    def __init__(
        self,
        dataset,
        batch_size: int,
        shuffle_episodes: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: int = 0,
        framework: Literal["jax", "pytorch"] = "pytorch",
        sharding: Optional[jax.sharding.Sharding] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle_episodes = shuffle_episodes
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.framework = framework
        self.sharding = sharding
        
        # Create the sequential episode sampler
        self.sampler = SequentialEpisodeSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle_episodes=shuffle_episodes,
            seed=seed
        )
        
        # Create the underlying PyTorch DataLoader with our custom sampler
        self.torch_loader = TorchDataLoaderBase(
            dataset=dataset,
            batch_sampler=self.sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self._collate_fn,
            worker_init_fn=self._worker_init_fn,
            
            
            # ! Experimental, seems to speed up data in 2-gpus scenario.
            # prefetch_factor=4,
        )
        
        # Set up sharding for JAX if needed
        if self.sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX
            self.sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

    def _collate_fn(self, items):
        """Collate function to handle dictionary items"""
        # Make sure to convert to numpy arrays before stacking since some elements
        # may be JAX arrays
        return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)

    def _worker_init_fn(self, worker_id: int) -> None:
        """Worker initialization function"""
        # Set random seed for reproducibility
        worker_seed = self.seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        
        # Configure JAX in worker processes
        if self.framework == "jax":
            import os
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    def reset(self, reset_episode_state: Optional[List[Tuple[int, int]]] = None, element_indices: Optional[List[int]] = None) -> None:
        """
        Reset the data loader: select new random episodes for elements and start from the first step.
        
        Args:
            element_indices: If specified, only reset these particular elements. 
                           If None, reset all elements.
        """
        self.sampler.reset(reset_episode_state, element_indices)
        
        if element_indices is None:
            logger.info("DataLoader reset: all elements have new episodes")
        else:
            logger.info(f"DataLoader reset: elements {element_indices} have new episodes")

    def __iter__(self):
        """Iterate over batches"""
        self._iterator = iter(self.torch_loader)
        return self

    def __next__(self):
        """Get the next batch"""
        if self._iterator is None:
            self._iterator = iter(self.torch_loader)

        batch = next(self._iterator)

        if self.framework == "jax" and self.sharding is not None:
            # Convert to sharded arrays for JAX
            return jax.tree.map(lambda x: jax.make_array_from_process_local_data(self.sharding, x), batch)
        else:
            # Return as torch tensors for PyTorch
            return jax.tree.map(torch.as_tensor, batch)

    def __len__(self):
        """Number of batches"""
        return len(self.sampler)


class SequentialEpisodeDataLoaderImpl:
    """Implementation of the DataLoader interface using SequentialEpisodeDataLoader"""
    
    def __init__(self, data_config: _config.DataConfig, data_loader: SequentialEpisodeDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def reset(self) -> None:
        """Reset the data loader"""
        self._data_loader.reset()

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]


def create_sequential_episode_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: Optional[jax.sharding.Sharding] = None,
    skip_norm_stats: bool = False,
    shuffle_episodes: bool = True,
    num_workers: int = 0,
    seed: int = 0,
    framework: Literal["jax", "pytorch"] = "jax",
    config: Optional[Any] = None,
) -> SequentialEpisodeDataLoaderImpl:
    """
    Create a sequential episode-based data loader for training.
    
    Args:
        data_config: The data configuration
        model_config: The model configuration
        action_horizon: The action horizon
        batch_size: Number of episodes per batch
        sharding: JAX sharding configuration
        skip_norm_stats: Whether to skip data normalization
        shuffle_episodes: Whether to shuffle episodes when resetting
        num_workers: Number of worker processes
        seed: Random seed
        framework: Framework to use ("jax" or "pytorch")
        config: Additional configuration
    """
    # Import locally to avoid circular imports
    from openpi_value.training.data_loader import create_torch_dataset, transform_dataset
    
    dataset = create_torch_dataset(
        data_config=data_config,
        action_horizon=action_horizon,
        model_config=model_config,
        config=config
    )
    
    # Apply transformations
    dataset = transform_dataset(
        dataset=dataset,
        data_config=data_config,
        skip_norm_stats=skip_norm_stats
    )
    
    # Handle distributed training
    local_batch_size = batch_size
    if framework == "pytorch" and torch.distributed.is_initialized():
        local_batch_size = batch_size // torch.distributed.get_world_size()
    elif framework == "jax":
        local_batch_size = batch_size // jax.process_count()
    
    logging.info(f"Creating SequentialEpisodeDataLoader with batch_size: {local_batch_size}")
    
    # Create the sequential episode-based data loader
    data_loader = SequentialEpisodeDataLoader(
        dataset=dataset,
        batch_size=local_batch_size,
        shuffle_episodes=shuffle_episodes,
        num_workers=num_workers,
        pin_memory=True if framework == "pytorch" else False,
        seed=seed,
        framework=framework,
        sharding=sharding,
    )
    
    return SequentialEpisodeDataLoaderImpl(data_config, data_loader)


def modify_create_data_loader_for_sequential_episodes():
    """
    Modify the original create_data_loader function to support sequential episode-based batching.
    This function should be called after importing the original create_data_loader.
    """
    import openpi_value.training.data_loader as original_dataloader
    
    def modified_create_torch_data_loader(
        data_config: _config.DataConfig,
        model_config: _model.BaseModelConfig,
        action_horizon: int,
        batch_size: int,
        *,
        sharding: Optional[jax.sharding.Sharding] = None,
        skip_norm_stats: bool = False,
        shuffle: bool = False,
        num_batches: Optional[int] = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
        config: Optional[Any] = None,
    ):
        """Modified create_torch_data_loader that supports sequential episode-based batching"""
        # Use sequential episode-based data loader
        logging.info("Using sequential episode-based data loader")
        return create_sequential_episode_data_loader(
            data_config=data_config,
            model_config=model_config,
            action_horizon=action_horizon,
            batch_size=batch_size,
            sharding=sharding,
            skip_norm_stats=skip_norm_stats,
            shuffle_episodes=shuffle,
            num_workers=num_workers,
            seed=seed,
            framework=framework,
            config=config,
        )
    
    # Replace the original function
    original_dataloader.create_torch_data_loader = modified_create_torch_data_loader
    
    logging.info("Modified create_torch_data_loader to support sequential episode-based batching")


def create_dataloader_with_sequential_episode(config: _config.TrainConfig, framework = "pytorch", shuffle=True):
    """
    Example function to create a data loader with sequential episode-based batching.
    
    Args:
        config: Training configuration
    """
    # First, modify the original create_data_loader to support sequential episode-based batching
    modify_create_data_loader_for_sequential_episodes()
    
    # Then create the data loader as usual
    from openpi_value.training.data_loader import create_data_loader
    
    # Set the flag to use sequential episode-based batching
    
    data_loader = create_data_loader(config, framework=framework, shuffle=shuffle)
    
    return data_loader
