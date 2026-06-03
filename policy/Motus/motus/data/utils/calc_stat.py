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

def process_single_file(file_path):
    """
    Process a single qpos.pt file to extract action statistics
    
    Args:
        file_path: Path to qpos.pt file
        
    Returns:
        tuple: (file_path, success, min_vals, max_vals, error_msg, outlier_dims)
    """
    try:
        # Load qpos data
        qpos_data = torch.load(file_path, map_location='cpu')
        
        # Convert to numpy array if it's a tensor
        if isinstance(qpos_data, torch.Tensor):
            action_data = qpos_data.numpy()
        else:
            action_data = np.array(qpos_data)
        
        # Check for outliers (absolute value > 4)
        outlier_dims = []
        if np.any(np.abs(action_data) > 3):
            outlier_dims = np.where(np.any(np.abs(action_data) > 3, axis=0))[0].tolist()
        
        # Calculate min/max for this file
        file_min = np.min(action_data, axis=0)
        file_max = np.max(action_data, axis=0)
        
        return (file_path, True, file_min, file_max, None, outlier_dims)
                
    except Exception as e:
        return (file_path, False, None, None, str(e), [])

def process_task_files(task_info):
    """
    Process all qpos.pt files in a single task directory
    
    Args:
        task_info: tuple of (task_name, task_dir, dataset_type)
        
    Returns:
        dict: Results for this task
    """
    task_name, task_dir, dataset_type = task_info
    
    # Find all qpos.pt files in this task directory
    if dataset_type == "robotwin":
        # RobotWin structure: task_dir/qpos/*.pt
        qpos_dir = os.path.join(task_dir, "qpos")
        if os.path.exists(qpos_dir):
            qpos_files = glob(os.path.join(qpos_dir, "*.pt"))
        else:
            qpos_files = []
    elif dataset_type in ["ac_one", "aloha_agilex_2"]:
        # AC-One / Aloha-Agilex2 structure: nested task directories; recursively search for any 'qpos' folder
        # Apply pruning to avoid heavy descent into irrelevant dirs (videos, instructions, backups, hidden)
        qpos_files = []
        if os.path.exists(task_dir):
            for current_root, dirnames, filenames in os.walk(task_dir, topdown=True):
                # Prune directories to reduce IO
                pruned = []
                for d in list(dirnames):
                    if d == "videos" or d == "instructions" or d == "umt5_wan":
                        continue
                    if d.endswith("_bak") or d.startswith("."):
                        continue
                    pruned.append(d)
                dirnames[:] = pruned

                # If this level contains a qpos directory, collect and do not descend into it
                if "qpos" in dirnames:
                    qpos_dir = os.path.join(current_root, "qpos")
                    found = glob(os.path.join(qpos_dir, "*.pt"))
                    found_numeric = [f for f in found if os.path.basename(f).replace('.pt', '').isdigit()]
                    if found_numeric:
                        qpos_files.extend(found_numeric)
                        print(f"    Found {len(found_numeric)} qpos files in {qpos_dir}")
                    # Do not walk into qpos directory
                    dirnames.remove("qpos")
    else:
        # Aloha structure: task_dir/*_qpos.pt (fallback for other aloha datasets)
        qpos_files = glob(os.path.join(task_dir, "*_qpos.pt"))
    
    results = {
        'task_name': task_name,
        'file_count': 0,
        'success_count': 0,
        'error_files': [],
        'outlier_files': [],
        'global_min': None,
        'global_max': None
    }
    
    print(f"Processing task '{task_name}' with {len(qpos_files)} files...")
    print(f"  Task directory: {task_dir}")
    print(f"  Looking for files in subdirectories of {task_dir}")

    # Debug: list some directories
    if os.path.exists(task_dir):
        for item in os.listdir(task_dir)[:3]:  # Show first 3 items
            item_path = os.path.join(task_dir, item)
            if os.path.isdir(item_path):
                print(f"    Subdir: {item}")
                subdir_items = os.listdir(item_path)[:3]
                print(f"      Contains: {subdir_items}")

    # Process each file in this task
    for file_path in tqdm(qpos_files, desc=f"Task {task_name}"):
        file_path, success, file_min, file_max, error_msg, outlier_dims = process_single_file(file_path)
        
        results['file_count'] += 1
        
        if success:
            results['success_count'] += 1
            
            # Update global min/max for this task
            if results['global_min'] is None:
                results['global_min'] = file_min
                results['global_max'] = file_max
            else:
                results['global_min'] = np.minimum(results['global_min'], file_min)
                results['global_max'] = np.maximum(results['global_max'], file_max)
            
            # Record outlier files
            if outlier_dims:
                results['outlier_files'].append((file_path, outlier_dims))
        else:
            results['error_files'].append((file_path, error_msg))
    
    return results

def collect_action_stats_multiprocess(root_dir, output_path, outlier_path, num_processes=16, dataset_type="aloha"):
    """
    Collect action statistics from qpos.pt files using multiprocessing
    
    Args:
        root_dir: Root directory of qpos.pt files
        output_path: Output JSON file path
        outlier_path: Output text file path for outlier files
        num_processes: Number of processes to use
        dataset_type: "aloha", "robotwin", or "ac_one"
    """
    print(f"Starting multiprocess statistics calculation for {dataset_type} with {num_processes} processes...")
    
    if dataset_type == "robotwin":
        # RobotWin has clean/ and randomized/ subdirs - only use clean
        all_task_dirs = []
        for split in ["clean", "randomized"]:  # Only process clean split
            split_dir = os.path.join(root_dir, split)
            if os.path.exists(split_dir):
                task_dirs = [d for d in os.listdir(split_dir) 
                           if os.path.isdir(os.path.join(split_dir, d))]
                for task in task_dirs:
                    all_task_dirs.append((f"{split}_{task}", os.path.join(split_dir, task)))
        task_dirs = all_task_dirs
    elif dataset_type in ["ac_one", "aloha_agilex_2"]:
        # AC-One structure: root_dir/task_category/
        task_dirs = [(d, os.path.join(root_dir, d)) for d in os.listdir(root_dir) 
                     if os.path.isdir(os.path.join(root_dir, d))]
    else:
        # Aloha structure
        task_dirs = [(d, os.path.join(root_dir, d)) for d in os.listdir(root_dir) 
                     if os.path.isdir(os.path.join(root_dir, d))]
    
    print(f"Found {len(task_dirs)} task directories")
    for task_name, task_dir in task_dirs:
        print(f"  - {task_name}: {task_dir}")

    # Prepare task info for multiprocessing
    task_infos = [(task, task_dir, dataset_type) for task, task_dir in task_dirs]
    
    # Record start time
    start_time = time.time()
    
    # Process tasks in parallel
    with mp.Pool(processes=num_processes) as pool:
        task_results = pool.map(process_task_files, task_infos)
    
    # Aggregate results from all tasks
    total_file_count = 0
    total_success_count = 0
    all_error_files = []
    all_outlier_files = []
    global_min = None
    global_max = None
    
    for result in task_results:
        total_file_count += result['file_count']
        total_success_count += result['success_count']
        all_error_files.extend(result['error_files'])
        all_outlier_files.extend(result['outlier_files'])
        
        # Update global min/max across all tasks
        if result['global_min'] is not None:
            if global_min is None:
                global_min = result['global_min']
                global_max = result['global_max']
            else:
                global_min = np.minimum(global_min, result['global_min'])
                global_max = np.maximum(global_max, result['global_max'])
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Generate statistics dictionary
    if dataset_type == "robotwin":
        dataset_key = "robotwin2"
    elif dataset_type == "ac_one":
        dataset_key = "ac_one"
    elif dataset_type == "aloha_agilex_2":
        dataset_key = "aloha_agilex_2"
    else:
        dataset_key = "aloha_agilex"
        
    stat_dict = {
        dataset_key: {
            "min": global_min.tolist() if global_min is not None else [],
            "max": global_max.tolist() if global_max is not None else [],
            "file_count": total_success_count,
            "total_files_scanned": total_file_count,
            "action_dim": len(global_min) if global_min is not None else 0,
            "processing_time_seconds": elapsed_time,
            "num_processes_used": num_processes
        }
    }
    
    # Load existing statistics if file exists and append new data
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_stats = json.load(f)
            # Merge with existing stats
            existing_stats.update(stat_dict)
            stat_dict = existing_stats
        except Exception as e:
            print(f"Warning: Could not load existing stats from {output_path}: {e}")
    
    # Save statistics results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stat_dict, f, indent=4, ensure_ascii=False)
    
    # Save outlier files list
    with open(outlier_path, 'w', encoding='utf-8') as f:
        f.write(f"Outlier files (absolute value > 4) - Total: {len(all_outlier_files)}\n")
        f.write("=" * 80 + "\n")
        for file_path, dims in all_outlier_files:
            f.write(f"{file_path} - Outlier dimensions: {dims}\n")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"MULTIPROCESS STATISTICS CALCULATION COMPLETED")
    print(f"{'='*60}")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"Processes used: {num_processes}")
    print(f"Average time per process: {elapsed_time/num_processes:.2f} seconds")
    print(f"Files per second: {total_file_count/elapsed_time:.2f}")
    print(f"\nResults:")
    print(f"- Total files scanned: {total_file_count}")
    print(f"- Successfully processed: {total_success_count}")
    print(f"- Failed files: {len(all_error_files)}")
    print(f"- Files with outliers: {len(all_outlier_files)}")
    print(f"- Action dimensions: {len(global_min) if global_min is not None else 'N/A'}")
    print(f"- Min values: {global_min}")
    print(f"- Max values: {global_max}")
    
    # Show task-wise breakdown
    print(f"\nTask-wise breakdown:")
    for result in task_results:
        print(f"- {result['task_name']}: {result['success_count']}/{result['file_count']} files")
    
    # Show some error examples
    if all_error_files:
        print(f"\nError examples (showing first 10 out of {len(all_error_files)}):")
        for i, (path, err) in enumerate(all_error_files[:10]):
            print(f"- {path}: {err}")
        if len(all_error_files) > 10:
            print(f"... and {len(all_error_files) - 10} more errors")

def main():
    """Main function to run the multiprocess statistics calculation"""
    parser = argparse.ArgumentParser(description='Calculate dataset statistics')
    parser.add_argument('--root_dir', type=str, 
                       default="/share/dataset/preprocess/aloha_agilex_2",
                       help='Root directory of qpos.pt files')
    parser.add_argument('--output_path', type=str, 
                       default="/share/home/bhz/test/latent_action_world_model/lawm/data/utils/stat.json",
                       help='Output JSON file path for statistics')
    parser.add_argument('--outlier_path', type=str, 
                       default="/share/home/bhz/test/latent_action_world_model/lawm/data/aloha_agilex_2_outlier_files.txt",
                       help='Output text file path for outlier files')
    parser.add_argument('--num_processes', type=int, default=16,
                       help='Number of processes to use')
    parser.add_argument('--dataset_type', type=str, default="aloha_agilex_2", choices=["aloha", "robotwin", "ac_one", "aloha_agilex_2"],
                       help='Dataset type: aloha, robotwin, or ac_one')
    
    args = parser.parse_args()
    
    print(f"Configuration:")
    print(f"- Root directory: {args.root_dir}")
    print(f"- Output file: {args.output_path}")
    print(f"- Outlier file: {args.outlier_path}")
    print(f"- Number of processes: {args.num_processes}")
    print(f"- Dataset type: {args.dataset_type}")
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.outlier_path), exist_ok=True)
    
    # Run the calculation
    collect_action_stats_multiprocess(
        root_dir=args.root_dir,
        output_path=args.output_path,
        outlier_path=args.outlier_path,
        num_processes=args.num_processes,
        dataset_type=args.dataset_type
    )
    
    print(f"\nResults saved to:")
    print(f"- Statistics: {args.output_path}")
    print(f"- Outliers: {args.outlier_path}")

if __name__ == "__main__":
    # Ensure proper multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()