#!/usr/bin/env python3
"""Compute statistics for action vectors across all datasets."""

import argparse
import json
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from tqdm import tqdm


class OnlineStatistics:
    """Online statistics calculator using Welford's algorithm."""
    
    def __init__(self, dim: int = 14):
        self.dim = dim
        self.count = 0
        self.min = np.full(dim, np.inf)
        self.max = np.full(dim, -np.inf)
        self.mean = np.zeros(dim)
        self.M2 = np.zeros(dim)  # Sum of squares of differences from mean
    
    def update(self, batch: np.ndarray):
        """Update statistics with a new batch of action vectors."""
        if len(batch.shape) != 2 or batch.shape[1] != self.dim:
            raise ValueError(f"Expected batch shape (N, {self.dim}), got {batch.shape}")
        
        batch_size = batch.shape[0]
        
        # Update min and max
        self.min = np.minimum(self.min, np.min(batch, axis=0))
        self.max = np.maximum(self.max, np.max(batch, axis=0))
        
        # Update mean and variance using Welford's algorithm
        for i in range(batch_size):
            self.count += 1
            delta = batch[i] - self.mean
            self.mean += delta / self.count
            delta2 = batch[i] - self.mean
            self.M2 += delta * delta2
    
    def get_statistics(self) -> dict:
        """Get final statistics."""
        if self.count == 0:
            return None
        
        var = self.M2 / self.count if self.count > 1 else np.zeros(self.dim)
        std = np.sqrt(var)
        
        return {
            'min': self.min,
            'max': self.max,
            'mean': self.mean,
            'std': std,
            'var': var,
        }




def print_statistics(stats: dict):
    """Print statistics as vectors."""
    print("\n" + "=" * 80)
    print("Action Statistics (14-dimensional vector)")
    print("=" * 80)
    
    def format_vector(v):
        return "[" + ", ".join(f"{x:.6f}" for x in v) + "]"
    
    print(f"min = {format_vector(stats['min'])}")
    print(f"max = {format_vector(stats['max'])}")
    print(f"mean = {format_vector(stats['mean'])}")
    print(f"std = {format_vector(stats['std'])}")
    print(f"var = {format_vector(stats['var'])}")
    print("=" * 80)


def save_to_config(stats: dict, dataset_names: list, config_path: Path):
    """Save normalization statistics to JSON config file."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing config if it exists
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Update config for each dataset
    for bench_name in dataset_names:
        config[bench_name] = {
            "min": stats['min'].tolist(),
            "max": stats['max'].tolist(),
        }
    
    # Save updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nNormalization statistics saved to: {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute action statistics from parquet files")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/cpfs01/user/yangjiazhi/lijinwei/RISE/dataset",
        help="Base directory containing datasets"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        help="Specific datasets to process (default: all datasets)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for processing (default: 1000)"
    )
    parser.add_argument(
        "--save-config",
        type=str,
        default=None,
        help="Path to save normalization config (JSON). If not specified, only prints statistics."
    )
    
    args = parser.parse_args()
    
    dataset_base = Path(args.dataset_dir)
    
    if not dataset_base.exists():
        print(f"Error: dataset directory not found: {dataset_base}")
        sys.exit(1)
    
    if args.datasets:
        dataset_dirs = [dataset_base / d for d in args.datasets]
    else:
        dataset_dirs = [d for d in dataset_base.iterdir() if d.is_dir()]
    
    if len(dataset_dirs) == 0:
        print("No datasets found")
        sys.exit(0)
    
    print(f"Found {len(dataset_dirs)} dataset(s)")
    
    # Process each dataset separately if saving config
    if args.save_config:
        config_path = Path(args.save_config)
        processed_datasets = []
        
        for dataset_dir in tqdm(dataset_dirs, desc="Processing datasets"):
            bench_name = dataset_dir.name
            dataset_stats = OnlineStatistics(dim=14)
            dataset_vectors = 0
            
            parquet_files = list(dataset_dir.rglob("data/chunk-*/episode_*.parquet"))
            
            if len(parquet_files) == 0:
                continue
            
            for parquet_file in tqdm(parquet_files, desc=f"  {bench_name}", leave=False):
                try:
                    df = pd.read_parquet(parquet_file)
                    if 'action' not in df.columns:
                        continue
                    
                    actions = np.stack(df['action'].values)
                    dataset_vectors += len(actions)
                    
                    for i in range(0, len(actions), args.batch_size):
                        batch = actions[i:i + args.batch_size]
                        dataset_stats.update(batch)
                except Exception as e:
                    print(f"Error reading {parquet_file}: {e}", file=sys.stderr)
                    continue
            
            if dataset_stats.count > 0:
                stats = dataset_stats.get_statistics()
                print(f"\nDataset: {bench_name} - {dataset_vectors} vectors")
                print_statistics(stats)
                processed_datasets.append(bench_name)
                save_to_config(stats, [bench_name], config_path)
        
        if len(processed_datasets) == 0:
            print("No action data found in any dataset")
            sys.exit(0)
    else:
        # Process all datasets together (original behavior)
        global_stats = OnlineStatistics(dim=14)
        total_vectors = 0
        
        for dataset_dir in tqdm(dataset_dirs, desc="Processing datasets"):
            bench_name = dataset_dir.name
            parquet_files = list(dataset_dir.rglob("data/chunk-*/episode_*.parquet"))
            
            if len(parquet_files) == 0:
                continue
            
            for parquet_file in tqdm(parquet_files, desc=f"  {bench_name}", leave=False):
                try:
                    df = pd.read_parquet(parquet_file)
                    if 'action' not in df.columns:
                        continue
                    
                    actions = np.stack(df['action'].values)
                    total_vectors += len(actions)
                    
                    for i in range(0, len(actions), args.batch_size):
                        batch = actions[i:i + args.batch_size]
                        global_stats.update(batch)
                except Exception as e:
                    print(f"Error reading {parquet_file}: {e}", file=sys.stderr)
                    continue
        
        if global_stats.count == 0:
            print("No action data found in any dataset")
            sys.exit(0)
        
        print(f"\nTotal action vectors processed: {total_vectors}")
        
        stats = global_stats.get_statistics()
        print_statistics(stats)


if __name__ == "__main__":
    main()
