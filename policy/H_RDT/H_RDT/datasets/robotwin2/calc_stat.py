#!/usr/bin/env python3
import os
import h5py
import json
import numpy as np
from tqdm import tqdm
from glob import glob
import multiprocessing as mp
from functools import partial
import time
import argparse

def process_single_file(file_path):
    """
    Process a single HDF5 file to extract action statistics
    
    Args:
        file_path: Path to HDF5 file
        
    Returns:
        tuple: (file_path, success, min_vals, max_vals, error_msg, outlier_dims)
    """
    try:
        with h5py.File(file_path, 'r') as f:
            if "joint_action/vector" in f:
                # Load action data
                action_data = f["joint_action/vector"][:]
                
                # Check for outliers (absolute value > 4)
                outlier_dims = []
                if np.any(np.abs(action_data) > 4):
                    outlier_dims = np.where(np.any(np.abs(action_data) > 4, axis=0))[0].tolist()
                
                # Calculate min/max for this file
                file_min = np.min(action_data, axis=0)
                file_max = np.max(action_data, axis=0)
                
                return (file_path, True, file_min, file_max, None, outlier_dims)
            else:
                return (file_path, False, None, None, "Missing joint_action/vector data", [])
                
    except Exception as e:
        return (file_path, False, None, None, str(e), [])

def process_task_files(task_info):
    """
    Process all HDF5 files in a single task directory
    
    Args:
        task_info: tuple of (task_name, task_dir)
        
    Returns:
        dict: Results for this task
    """
    task_name, task_dir = task_info
    
    # Find all HDF5 files in this task directory
    hdf5_files = glob(os.path.join(task_dir, "**/*.hdf5"), recursive=True)
    
    results = {
        'task_name': task_name,
        'file_count': 0,
        'success_count': 0,
        'error_files': [],
        'outlier_files': [],
        'global_min': None,
        'global_max': None
    }
    
    print(f"Processing task '{task_name}' with {len(hdf5_files)} files...")
    
    # Process each file in this task
    for file_path in tqdm(hdf5_files, desc=f"Task {task_name}"):
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

def collect_action_stats_multiprocess(root_dir, output_path, outlier_path, num_processes=16):
    """
    Collect action statistics from all RoboTwin HDF5 files using multiprocessing
    
    Args:
        root_dir: Root directory of HDF5 files
        output_path: Output JSON file path
        outlier_path: Output text file path for outlier files
        num_processes: Number of processes to use
    """
    print(f"Starting multiprocess statistics calculation with {num_processes} processes...")
    
    # Get all task directories (exclude seed directory)
    task_dirs = [d for d in os.listdir(root_dir) 
                 if os.path.isdir(os.path.join(root_dir, d)) and d != "seed"]
    
    print(f"Found {len(task_dirs)} task directories")
    
    # Prepare task info for multiprocessing
    task_infos = [(task, os.path.join(root_dir, task)) for task in task_dirs]
    
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
    stat_dict = {
        "robotwin_dual_arm": {
            "min": global_min.tolist() if global_min is not None else [],
            "max": global_max.tolist() if global_max is not None else [],
            "file_count": total_success_count,
            "total_files_scanned": total_file_count,
            "action_dim": len(global_min) if global_min is not None else 0,
            "processing_time_seconds": elapsed_time,
            "num_processes_used": num_processes
        }
    }
    
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
    parser = argparse.ArgumentParser(description='Calculate RobotWin2 dataset statistics')
    parser.add_argument('--root_dir', type=str, 
                       default="/share/hongzhe/datasets/robotwin2/dataset/aloha-agilex",
                       help='Root directory of HDF5 files')
    parser.add_argument('--output_path', type=str, 
                       default="/share/hongzhe/h_rdt/datasets/robotwin2/stats.json",
                       help='Output JSON file path for statistics')
    parser.add_argument('--outlier_path', type=str, 
                       default="/share/hongzhe/h_rdt/datasets/robotwin2/outlier_files.txt",
                       help='Output text file path for outlier files')
    parser.add_argument('--num_processes', type=int, default=64,
                       help='Number of processes to use')
    
    args = parser.parse_args()
    
    print(f"Configuration:")
    print(f"- Root directory: {args.root_dir}")
    print(f"- Output file: {args.output_path}")
    print(f"- Outlier file: {args.outlier_path}")
    print(f"- Number of processes: {args.num_processes}")
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.outlier_path), exist_ok=True)
    
    # Run the calculation
    collect_action_stats_multiprocess(
        root_dir=args.root_dir,
        output_path=args.output_path,
        outlier_path=args.outlier_path,
        num_processes=args.num_processes
    )
    
    print(f"\nResults saved to:")
    print(f"- Statistics: {args.output_path}")
    print(f"- Outliers: {args.outlier_path}")

if __name__ == "__main__":
    # Ensure proper multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main() 