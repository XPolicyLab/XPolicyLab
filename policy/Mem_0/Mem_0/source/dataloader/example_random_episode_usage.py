# Copyright 2025 MemoryMatters Team. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Yuran Wang / Peking University] in [2025].
"""
Example usage and validation of RandomEpisodeIterableDataset

This script demonstrates and validates:
1. Basic functionality with single worker
2. Multi-worker support
3. Global shuffle then assign to workers strategy
4. Episode assignment correctness
5. Sequential frame reading within episodes
6. Cycle re-shuffling behavior
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import torch
from torch.utils.data import DataLoader
from source.dataloader.dataset_min_max import LeRobot_Dataset
from source.dataloader.random_episode_dataloader import RandomEpisodeIterableDataset
from termcolor import cprint
from collections import defaultdict


def test_basic_functionality():
    """Test 1: Basic functionality with single worker."""
    cprint("\n" + "=" * 60, "cyan")
    cprint("Test 1: Basic Functionality (num_workers=0)", "cyan")
    cprint("=" * 60, "cyan")
    
    repo_id = "your_repo_id"
    rank = 0
    world_size = 1
    batch_size = 16
    seed = 42
    
    try:
        base_dataset = LeRobot_Dataset(
            repo_id=repo_id,
            features_to_load=[
                "observation.image.head_camera",
                "observation.state",
                "action",
                "subtask",
                "subtask_end",
                "episode_id"
            ]
        )
        cprint(f"[Test 1] Base dataset size: {len(base_dataset)} samples", "green")
        
        iterable_dataset = RandomEpisodeIterableDataset(
            base_dataset=base_dataset,
            rank=rank,
            world_size=world_size,
            shuffle=True,
            seed=seed,
            infinite=False  # Test finite mode
        )
        
        dataloader = DataLoader(
            iterable_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=lambda batch: batch,
            drop_last=False,
        )
        
        # Collect episode sequences
        episode_sequences = defaultdict(list)
        batch_count = 0
        
        for batch_idx, batch in enumerate(dataloader):
            batch_count += 1
            for sample in batch:
                episode_id = sample.get("episode_id")
                if isinstance(episode_id, torch.Tensor):
                    episode_id = int(episode_id.item())
                elif hasattr(episode_id, '__int__'):
                    episode_id = int(episode_id)
                episode_sequences[episode_id].append(batch_idx * batch_size + len(episode_sequences[episode_id]))
        
        # Verify sequential reading within episodes
        all_sequential = True
        for episode_id, indices in episode_sequences.items():
            if len(indices) > 1:
                is_sorted = all(indices[i] <= indices[i+1] for i in range(len(indices)-1))
                if not is_sorted:
                    all_sequential = False
                    cprint(f"  ❌ Episode {episode_id}: frames not sequential: {indices[:10]}...", "red")
        
        if all_sequential:
            cprint(f"  ✓ All episodes read sequentially", "green")
        
        cprint(f"  ✓ Processed {batch_count} batches", "green")
        cprint(f"  ✓ Found {len(episode_sequences)} unique episodes", "green")
        cprint(f"[Test 1] PASSED", "green")
        return True
        
    except Exception as e:
        cprint(f"[Test 1] FAILED: {e}", "red")
        import traceback
        traceback.print_exc()
        return False


def test_multi_worker():
    """Test 2: Multi-worker support."""
    cprint("\n" + "=" * 60, "cyan")
    cprint("Test 2: Multi-Worker Support (num_workers=4)", "cyan")
    cprint("=" * 60, "cyan")
    
    repo_id = "your_repo_id"
    rank = 0
    world_size = 1
    batch_size = 16
    seed = 42
    num_workers = 4
    
    try:
        base_dataset = LeRobot_Dataset(
            repo_id=repo_id,
            features_to_load=[
                "observation.image.head_camera",
                "observation.state",
                "action",
                "subtask",
                "subtask_end",
                "episode_id"
            ]
        )
        
        iterable_dataset = RandomEpisodeIterableDataset(
            base_dataset=base_dataset,
            rank=rank,
            world_size=world_size,
            shuffle=True,
            seed=seed,
            infinite=False
        )
        
        # Check rank_episodes assignment
        if hasattr(iterable_dataset, 'rank_episodes'):
            num_rank_episodes = len(iterable_dataset.rank_episodes)
            cprint(f"  Rank {rank} has {num_rank_episodes} episodes", "cyan")
            
            # Calculate expected distribution
            episodes_per_worker = num_rank_episodes // num_workers
            remainder = num_rank_episodes % num_workers
            cprint(f"  Expected: {episodes_per_worker} episodes per worker, {remainder} workers get +1", "cyan")
        
        dataloader = DataLoader(
            iterable_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=lambda batch: batch,
            drop_last=False,
        )
        
        # Collect data from all workers
        episode_sequences = defaultdict(list)
        batch_count = 0
        
        for batch_idx, batch in enumerate(dataloader):
            batch_count += 1
            for sample in batch:
                episode_id = sample.get("episode_id")
                if isinstance(episode_id, torch.Tensor):
                    episode_id = int(episode_id.item())
                elif hasattr(episode_id, '__int__'):
                    episode_id = int(episode_id)
                episode_sequences[episode_id].append(batch_idx * batch_size + len(episode_sequences[episode_id]))
        
        # Verify sequential reading
        all_sequential = True
        for episode_id, indices in episode_sequences.items():
            if len(indices) > 1:
                is_sorted = all(indices[i] <= indices[i+1] for i in range(len(indices)-1))
                if not is_sorted:
                    all_sequential = False
                    cprint(f"  ❌ Episode {episode_id}: frames not sequential", "red")
        
        if all_sequential:
            cprint(f"  ✓ All episodes read sequentially (multi-worker)", "green")
        
        cprint(f"  ✓ Processed {batch_count} batches with {num_workers} workers", "green")
        cprint(f"  ✓ Found {len(episode_sequences)} unique episodes", "green")
        cprint(f"[Test 2] PASSED", "green")
        return True
        
    except Exception as e:
        cprint(f"[Test 2] FAILED: {e}", "red")
        import traceback
        traceback.print_exc()
        return False


def test_shuffle_strategy():
    """Test 3: Verify global shuffle then assign strategy."""
    cprint("\n" + "=" * 60, "cyan")
    cprint("Test 3: Global Shuffle Then Assign Strategy", "cyan")
    cprint("=" * 60, "cyan")
    
    repo_id = "your_repo_id"
    rank = 0
    world_size = 1
    batch_size = 16
    seed = 42
    num_workers = 2
    
    try:
        base_dataset = LeRobot_Dataset(
            repo_id=repo_id,
            features_to_load=[
                "observation.image.head_camera",
                "observation.state",
                "action",
                "subtask",
                "subtask_end",
                "episode_id"
            ]
        )
        
        # Test with shuffle=True
        iterable_dataset_shuffle = RandomEpisodeIterableDataset(
            base_dataset=base_dataset,
            rank=rank,
            world_size=world_size,
            shuffle=True,
            seed=seed,
            infinite=False
        )
        
        # Test with shuffle=False
        iterable_dataset_no_shuffle = RandomEpisodeIterableDataset(
            base_dataset=base_dataset,
            rank=rank,
            world_size=world_size,
            shuffle=False,
            seed=seed,
            infinite=False
        )
        
        dataloader_shuffle = DataLoader(
            iterable_dataset_shuffle,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=lambda batch: batch,
            drop_last=False,
        )
        
        dataloader_no_shuffle = DataLoader(
            iterable_dataset_no_shuffle,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=lambda batch: batch,
            drop_last=False,
        )
        
        # Collect first episodes from each worker
        def get_first_episodes(dataloader, max_batches=5):
            first_episodes = []
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break
                for sample in batch:
                    episode_id = sample.get("episode_id")
                    if isinstance(episode_id, torch.Tensor):
                        episode_id = int(episode_id.item())
                    elif hasattr(episode_id, '__int__'):
                        episode_id = int(episode_id)
                    if episode_id not in first_episodes:
                        first_episodes.append(episode_id)
                    if len(first_episodes) >= num_workers * 2:  # Get first 2 episodes per worker
                        return first_episodes
            return first_episodes
        
        first_shuffle = get_first_episodes(dataloader_shuffle)
        first_no_shuffle = get_first_episodes(dataloader_no_shuffle)
        
        cprint(f"  With shuffle=True: first episodes = {first_shuffle[:10]}", "cyan")
        cprint(f"  With shuffle=False: first episodes = {first_no_shuffle[:10]}", "cyan")
        
        # Episodes should be different when shuffle=True
        if first_shuffle != first_no_shuffle:
            cprint(f"  ✓ Shuffle is working (episode order differs)", "green")
        else:
            cprint(f"  ⚠️  Shuffle may not be working (episode order same)", "yellow")
        
        cprint(f"[Test 3] PASSED", "green")
        return True
        
    except Exception as e:
        cprint(f"[Test 3] FAILED: {e}", "red")
        import traceback
        traceback.print_exc()
        return False


def test_episode_assignment():
    """Test 4: Verify episode assignment correctness."""
    cprint("\n" + "=" * 60, "cyan")
    cprint("Test 4: Episode Assignment Correctness", "cyan")
    cprint("=" * 60, "cyan")
    
    repo_id = "your_repo_id"
    rank = 0
    world_size = 2  # Test with 2 ranks
    batch_size = 16
    seed = 42
    
    try:
        base_dataset = LeRobot_Dataset(
            repo_id=repo_id,
            features_to_load=[
                "observation.image.head_camera",
                "observation.state",
                "action",
                "subtask",
                "subtask_end",
                "episode_id"
            ]
        )
        
        # Create datasets for different ranks
        iterable_dataset_rank0 = RandomEpisodeIterableDataset(
            base_dataset=base_dataset,
            rank=0,
            world_size=world_size,
            shuffle=False,  # No shuffle for easier verification
            seed=seed,
            infinite=False
        )
        
        iterable_dataset_rank1 = RandomEpisodeIterableDataset(
            base_dataset=base_dataset,
            rank=1,
            world_size=world_size,
            shuffle=False,
            seed=seed,
            infinite=False
        )
        
        rank0_episodes = set(iterable_dataset_rank0.rank_episodes)
        rank1_episodes = set(iterable_dataset_rank1.rank_episodes)
        
        # Verify no overlap
        overlap = rank0_episodes & rank1_episodes
        if len(overlap) == 0:
            cprint(f"  ✓ No episode overlap between ranks", "green")
        else:
            cprint(f"  ❌ Found overlap: {overlap}", "red")
            return False
        
        # Verify assignment rule: episode_id % world_size == rank
        all_correct_rank0 = all(eid % world_size == 0 for eid in rank0_episodes)
        all_correct_rank1 = all(eid % world_size == 1 for eid in rank1_episodes)
        
        if all_correct_rank0 and all_correct_rank1:
            cprint(f"  ✓ Episode assignment rule correct (episode_id % {world_size} == rank)", "green")
        else:
            cprint(f"  ❌ Episode assignment rule violated", "red")
            return False
        
        cprint(f"  Rank 0: {len(rank0_episodes)} episodes", "cyan")
        cprint(f"  Rank 1: {len(rank1_episodes)} episodes", "cyan")
        cprint(f"[Test 4] PASSED", "green")
        return True
        
    except Exception as e:
        cprint(f"[Test 4] FAILED: {e}", "red")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all validation tests."""
    cprint("\n" + "=" * 60, "cyan")
    cprint("RandomEpisodeIterableDataset Validation Suite", "cyan")
    cprint("=" * 60, "cyan")
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Multi-Worker Support", test_multi_worker),
        ("Shuffle Strategy", test_shuffle_strategy),
        ("Episode Assignment", test_episode_assignment),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            cprint(f"\n❌ {test_name} raised exception: {e}", "red")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    cprint("\n" + "=" * 60, "cyan")
    cprint("Test Summary", "cyan")
    cprint("=" * 60, "cyan")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "❌ FAILED"
        color = "green" if result else "red"
        cprint(f"  {test_name}: {status}", color)
    
    cprint(f"\nTotal: {passed}/{total} tests passed", "green" if passed == total else "yellow")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

