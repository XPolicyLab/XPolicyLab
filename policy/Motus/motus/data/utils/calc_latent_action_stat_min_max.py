#!/usr/bin/env python3
import os
import json
import numpy as np
import torch
from tqdm import tqdm
from glob import glob
import multiprocessing as mp
from functools import partial
import time
import argparse
from pathlib import Path


def process_single_file(file_path):
    """
    Process a single latent_action .pt file to extract statistics

    Args:
        file_path: Path to latent_action .pt file

    Returns:
        tuple: (file_path, success, mean_vals, std_vals, error_msg, outlier_info)
    """
    try:
        # Load latent_action data
        data = torch.load(file_path, map_location='cpu')

        # Handle different formats
        if isinstance(data, torch.Tensor):
            latent_action = data
        elif isinstance(data, dict) and 'latent_action' in data:
            latent_action = data['latent_action']
        else:
            raise ValueError(f"Unsupported format for {file_path}")

        # Convert to numpy array if it's a tensor
        if isinstance(latent_action, torch.Tensor):
            # Ensure BF16/FP16 tensors are cast to float32 before numpy conversion
            la = latent_action.detach().cpu()
            if la.dtype in (torch.bfloat16, torch.float16):
                la = la.to(torch.float32)
            action_data = la.numpy()
        else:
            action_data = np.array(latent_action)

        # Ensure 2D: [T, D] where T is number of frames, D is latent dimension (2048)
        if action_data.ndim == 1:
            action_data = action_data.reshape(1, -1)

        # Check for outliers (absolute value > 100, common threshold for latent spaces)
        outlier_info = {}
        if np.any(np.abs(action_data) > 100):
            outlier_dims = np.where(np.any(np.abs(action_data) > 100, axis=0))[0].tolist()
            max_outlier_val = np.max(np.abs(action_data))
            outlier_info = {
                'dims': outlier_dims,
                'max_abs_value': float(max_outlier_val)
            }

        # Calculate mean and std for this file across all frames
        file_mean = np.mean(action_data, axis=0)
        file_std = np.std(action_data, axis=0)

        return (file_path, True, file_mean, file_std, None, outlier_info)

    except Exception as e:
        return (file_path, False, None, None, str(e), {})


def _has_triplet_subdirs(directory: Path) -> bool:
    """Check if directory contains the three required subdirs"""
    return all((directory / sub).exists() for sub in ["videos", "umt5_wan", "latent_action"])


def _find_leaf_dataset_dirs(root: Path):
    """Recursively find directories that contain videos/umt5_wan/latent_action subfolders."""
    results = []
    try:
        if _has_triplet_subdirs(root):
            results.append(root)
            return results
        for dirpath, dirnames, _ in os.walk(root):
            current = Path(dirpath)
            if _has_triplet_subdirs(current):
                results.append(current)
    except Exception as e:
        print(f"Warning: Failed scanning {root}: {e}")
    return results


def process_task_files(task_info, fail_fast: bool = False):
    """
    Process all latent_action .pt files in a single task directory
    
    Args:
        task_info: tuple of (task_name, latent_action_dir)
        fail_fast: if True, raise immediately on first error to abort the whole run
        
    Returns:
        dict: Results for this task
    """
    task_name, latent_action_dir = task_info
    
    # Find all .pt files in latent_action directory
    latent_files = glob(os.path.join(latent_action_dir, "*.pt"))
    
    results = {
        'task_name': task_name,
        'file_count': 0,
        'success_count': 0,
        'error_files': [],
        'outlier_files': [],
        'global_mean': None,
        'global_std': None,
        'global_var_sum': None,
        'global_count': 0
    }
    
    print(f"Processing task '{task_name}' with {len(latent_files)} files...")
    print(f"  Latent action directory: {latent_action_dir}")

    # Process each file in this task
    for file_path in tqdm(latent_files, desc=f"Task {task_name}"):
        file_path, success, file_mean, file_std, error_msg, outlier_info = process_single_file(file_path)

        results['file_count'] += 1

        if success:
            results['success_count'] += 1

            # Get the number of frames in this file for weighted averaging
            data = torch.load(file_path, map_location='cpu')
            if isinstance(data, torch.Tensor):
                latent_action = data
            elif isinstance(data, dict) and 'latent_action' in data:
                latent_action = data['latent_action']
            else:
                latent_action = data

            if isinstance(latent_action, torch.Tensor):
                la = latent_action.detach().cpu()
                if la.dtype in (torch.bfloat16, torch.float16):
                    la = la.to(torch.float32)
                action_data = la.numpy()
            else:
                action_data = np.array(latent_action)

            if action_data.ndim == 1:
                action_data = action_data.reshape(1, -1)

            frame_count = action_data.shape[0]

            # Update global statistics for this task using Welford's online algorithm
            if results['global_count'] == 0:
                results['global_mean'] = file_mean.copy()
                results['global_var_sum'] = (file_std ** 2) * frame_count
                results['global_count'] = frame_count
            else:
                # Update mean and variance using parallel algorithm
                old_count = results['global_count']
                new_count = old_count + frame_count
                old_mean = results['global_mean']
                new_mean = (old_mean * old_count + file_mean * frame_count) / new_count

                # Update variance sum
                mean_diff = file_mean - old_mean
                results['global_var_sum'] = results['global_var_sum'] + (file_std ** 2) * frame_count + (mean_diff ** 2) * old_count * frame_count / new_count
                results['global_mean'] = new_mean
                results['global_count'] = new_count

            # Record outlier files
            if outlier_info:
                results['outlier_files'].append((file_path, outlier_info))
        else:
            results['error_files'].append((file_path, error_msg))
            if fail_fast:
                # Abort immediately on first error in this task
                raise RuntimeError(f"Fail-fast: error processing {file_path}: {error_msg}")
    
    return results


def collect_latent_action_stats_multiprocess(root_dirs, output_path, outlier_path, num_processes=16, fail_fast=False):
    """
    Collect latent action statistics from .pt files using multiprocessing
    
    Args:
        root_dirs: List of root directories
        output_path: Output JSON file path
        outlier_path: Output text file path for outlier files
        num_processes: Number of processes to use
        fail_fast: If True, abort on the first error encountered
    """
    print(f"Starting multiprocess statistics calculation with {num_processes} processes...")
    print(f"Root directories: {root_dirs}")
    print(f"Fail-fast mode: {fail_fast}")
    
    # Find all leaf dataset directories
    all_leaf_dirs = []
    for root_dir in root_dirs:
        root_path = Path(root_dir)
        if root_path.exists():
            leaf_dirs = _find_leaf_dataset_dirs(root_path)
            all_leaf_dirs.extend(leaf_dirs)
            print(f"Found {len(leaf_dirs)} leaf directories under {root_dir}")
        else:
            print(f"Warning: Root directory does not exist: {root_dir}")
    
    if not all_leaf_dirs:
        raise RuntimeError("No valid leaf dataset directories found")
    
    print(f"\nTotal leaf directories found: {len(all_leaf_dirs)}")
    
    # Prepare task info for multiprocessing
    task_infos = []
    for leaf_dir in all_leaf_dirs:
        latent_action_dir = leaf_dir / "latent_action"
        if latent_action_dir.exists():
            task_name = str(leaf_dir.relative_to(Path(root_dirs[0]).parent))
            task_infos.append((task_name, str(latent_action_dir)))
    
    print(f"Processing {len(task_infos)} tasks:")
    for task_name, _ in task_infos[:5]:  # Show first 5
        print(f"  - {task_name}")
    if len(task_infos) > 5:
        print(f"  ... and {len(task_infos) - 5} more")
    
    # Record start time
    start_time = time.time()
    
    # Process tasks in parallel
    try:
        with mp.Pool(processes=num_processes) as pool:
            task_results = pool.map(partial(process_task_files, fail_fast=fail_fast), task_infos)
    except Exception as e:
        print("\nERROR: Aborting due to failure in worker process.")
        print(f"Reason: {e}")
        raise
    
    # Aggregate results from all tasks
    total_file_count = 0
    total_success_count = 0
    all_error_files = []
    all_outlier_files = []
    global_mean = None
    global_var_sum = None
    global_count = 0

    for result in task_results:
        total_file_count += result['file_count']
        total_success_count += result['success_count']
        all_error_files.extend(result['error_files'])
        all_outlier_files.extend(result['outlier_files'])

        # Update global statistics across all tasks using parallel algorithm
        if result['global_mean'] is not None:
            if global_count == 0:
                global_mean = result['global_mean']
                global_var_sum = result['global_var_sum']
                global_count = result['global_count']
            else:
                # Combine statistics from this task with global statistics
                old_count = global_count
                new_count = old_count + result['global_count']
                old_mean = global_mean
                new_mean = (old_mean * old_count + result['global_mean'] * result['global_count']) / new_count

                # Update variance sum
                mean_diff = result['global_mean'] - old_mean
                global_var_sum = global_var_sum + result['global_var_sum'] + (mean_diff ** 2) * old_count * result['global_count'] / new_count
                global_mean = new_mean
                global_count = new_count
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Calculate global standard deviation from variance sum
    global_std = None
    if global_count > 0 and global_var_sum is not None:
        global_std = np.sqrt(global_var_sum / global_count)

    # Generate statistics dictionary
    stat_dict = {
        "latent_action": {
            "mean": global_mean.tolist() if global_mean is not None else [],
            "std": global_std.tolist() if global_std is not None else [],
            "file_count": total_success_count,
            "total_files_scanned": total_file_count,
            "latent_dim": len(global_mean) if global_mean is not None else 0,
            "total_frames": global_count,
            "processing_time_seconds": elapsed_time,
            "num_processes_used": num_processes,
            "num_tasks": len(task_infos),
            "num_outlier_files": len(all_outlier_files)
        }
    }
    
    # Save statistics results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stat_dict, f, indent=4, ensure_ascii=False)
    
    # Save outlier files list
    os.makedirs(os.path.dirname(outlier_path), exist_ok=True)
    with open(outlier_path, 'w', encoding='utf-8') as f:
        f.write(f"Outlier files (absolute value > 100) - Total: {len(all_outlier_files)}\n")
        f.write("=" * 80 + "\n\n")
        for file_path, outlier_info in all_outlier_files:
            f.write(f"File: {file_path}\n")
            f.write(f"  Outlier dimensions: {outlier_info['dims']}\n")
            f.write(f"  Max absolute value: {outlier_info['max_abs_value']:.4f}\n\n")
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print(f"MULTIPROCESS STATISTICS CALCULATION COMPLETED")
    print(f"{'='*80}")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"Processes used: {num_processes}")
    print(f"Average time per process: {elapsed_time/num_processes:.2f} seconds")
    print(f"Files per second: {total_file_count/elapsed_time:.2f}")
    print(f"\nResults:")
    print(f"- Total files scanned: {total_file_count}")
    print(f"- Successfully processed: {total_success_count}")
    print(f"- Failed files: {len(all_error_files)}")
    print(f"- Files with outliers (|value| > 100): {len(all_outlier_files)}")
    print(f"- Latent dimensions: {len(global_mean) if global_mean is not None else 'N/A'}")
    
    if global_mean is not None:
        print(f"\nGlobal statistics across all dimensions:")
        print(f"- Mean value across all dims: {np.mean(global_mean):.4f}")
        print(f"- Std value across all dims: {np.mean(global_std):.4f}")
        print(f"- Min mean across dims: {np.min(global_mean):.4f}")
        print(f"- Max mean across dims: {np.max(global_mean):.4f}")
        print(f"- Min std across dims: {np.min(global_std):.4f}")
        print(f"- Max std across dims: {np.max(global_std):.4f}")
        print(f"- Total frames processed: {global_count}")

        # Show distribution of mean/std values
        print(f"\nValue distribution:")
        print(f"- Dims with |mean| > 1: {np.sum(np.abs(global_mean) > 1)}")
        print(f"- Dims with std < 0.1: {np.sum(global_std < 0.1)}")
        print(f"- Dims with std > 10: {np.sum(global_std > 10)}")
        print(f"- Dims with |mean| > 5: {np.sum(np.abs(global_mean) > 5)}")
    
    # Show task-wise breakdown
    print(f"\nTask-wise breakdown (first 10):")
    for i, result in enumerate(task_results[:10]):
        print(f"- {result['task_name']}: {result['success_count']}/{result['file_count']} files")
    if len(task_results) > 10:
        print(f"... and {len(task_results) - 10} more tasks")
    
    # Show some error examples
    if all_error_files:
        print(f"\nError examples (showing first 5 out of {len(all_error_files)}):")
        for i, (path, err) in enumerate(all_error_files[:5]):
            print(f"- {path}: {err}")
        if len(all_error_files) > 5:
            print(f"... and {len(all_error_files) - 5} more errors")


def main():
    """Main function to run the multiprocess statistics calculation"""
    parser = argparse.ArgumentParser(description='Calculate latent action statistics')
    parser.add_argument('--root_dirs', type=str, nargs='+',
                       default=[
                           "/share/dataset/preprocess/pretrain",
                           "/share/dataset/preprocess/egodexresized_human_1",
                           "/share/dataset/preprocess/0710_aloha_3",
                           "/share/dataset/preprocess/0902_aloha_3",
                           "/share/dataset/preprocess/0820_aloha_3",
                           "/share/dataset/preprocess/0910retry_aloha_3",
                           "/share/dataset/preprocess/robotwin2_copy/clean"
                       ],
                       help='Root directories of dataset')
    parser.add_argument('--output_path', type=str,
                       default="/share/home/bhz/test/latent_action_world_model/lawm/data/utils/latent_action_stat_mean_std.json",
                       help='Output JSON file path for mean/std statistics')
    parser.add_argument('--outlier_path', type=str,
                       default="/share/home/bhz/test/latent_action_world_model/lawm/data/utils/latent_action_outlier_files_mean_std.txt",
                       help='Output text file path for outlier files')
    parser.add_argument('--num_processes', type=int, default=16,
                       help='Number of processes to use')
    parser.add_argument('--fail_fast', action='store_true', default=True,
                       help='Abort immediately on first error encountered')
    
    args = parser.parse_args()
    
    print(f"Configuration:")
    print(f"- Root directories: {len(args.root_dirs)} directories")
    for root in args.root_dirs:
        print(f"    {root}")
    print(f"- Output file: {args.output_path}")
    print(f"- Outlier file: {args.outlier_path}")
    print(f"- Number of processes: {args.num_processes}")
    print(f"- Fail-fast: {args.fail_fast}")
    
    # Run the calculation
    collect_latent_action_stats_multiprocess(
        root_dirs=args.root_dirs,
        output_path=args.output_path,
        outlier_path=args.outlier_path,
        num_processes=args.num_processes,
        fail_fast=args.fail_fast
    )
    
    print(f"\nResults saved to:")
    print(f"- Statistics: {args.output_path}")
    print(f"- Outliers: {args.outlier_path}")


if __name__ == "__main__":
    # Ensure proper multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()
