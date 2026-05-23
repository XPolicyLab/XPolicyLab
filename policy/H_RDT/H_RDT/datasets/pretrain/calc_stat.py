import os
import h5py
import json
import numpy as np
from tqdm import tqdm
from glob import glob
from pathlib import Path
import argparse

def collect_egodex_action_stats(root_dir, output_path, large_values_log="large_values.txt"):
    """
    Collect statistics for pre-computed 48-dimensional action data in EgoDex dataset
    Directly read actions_48d data from HDF5 files instead of real-time computation
    
    Args:
        root_dir: EgoDex dataset root directory
        output_path: Output JSON file path
        large_values_log: Log file for recording outliers
    """
    # Initialize statistics containers
    global_min = None
    global_max = None
    file_count = 0
    error_files = []
    
    # Record files with absolute values exceeding threshold
    large_values_files = []
    threshold = 5.0  # Set threshold to 5

    # Find all HDF5 files
    root_path = Path(root_dir)
    hdf5_files = []
    
    # Traverse all part directories
    for part in ['part1', 'part2', 'part3', 'part4', 'part5', 'extra', 'test']:
        part_dir = root_path / part
        if part_dir.exists():
            for task_dir in part_dir.iterdir():
                if task_dir.is_dir():
                    task_hdf5_files = list(task_dir.glob('*.hdf5'))
                    hdf5_files.extend(task_hdf5_files)
    
    print(f"Found {len(hdf5_files)} HDF5 files in EgoDex dataset")

    # Process each file
    for file_path in tqdm(hdf5_files, desc="Processing EgoDex files"):
        try:
            with h5py.File(file_path, 'r') as f:
                # Directly read pre-computed 48-dimensional action data
                if "actions_48d" in f:
                    action_data = f['actions_48d'][:]
                    
                    # Check if data shape is correct
                    if action_data.shape[1] != 48:
                        error_files.append((str(file_path), f"Wrong action dimension: {action_data.shape[1]}, expected 48"))
                        continue
                    
                    # Check if there are dimensions with absolute values exceeding threshold
                    if np.any(np.abs(action_data) > threshold):
                        large_values_files.append(str(file_path))
                        
                        # Get specific frame and dimension information
                        abs_data = np.abs(action_data)
                        frames, dims = np.where(abs_data > threshold)
                        max_val = abs_data.max()
                        print(f"Large values found in {file_path}: max={max_val:.3f}")
                    
                    # Initialize or update global extremes
                    if global_min is None:
                        global_min = np.min(action_data, axis=0)
                        global_max = np.max(action_data, axis=0)
                    else:
                        global_min = np.minimum(global_min, np.min(action_data, axis=0))
                        global_max = np.maximum(global_max, np.max(action_data, axis=0))
                    
                    file_count += 1
                else:
                    error_files.append((str(file_path), "Missing actions_48d data (run precompute_48d_actions.py first)"))
        except Exception as e:
            error_files.append((str(file_path), str(e)))

    # Generate statistics results
    stat_dict = {
        "egodex": {
            "min": global_min.tolist() if global_min is not None else [],
            "max": global_max.tolist() if global_max is not None else [],
        }
    }

    # Save results
    with open(output_path, 'w') as f:
        json.dump(stat_dict, f, indent=4)
    
    # Save list of files with outliers
    with open(large_values_log, 'w') as f:
        f.write(f"Found {len(large_values_files)} files containing 48-dimensional action data with absolute values > {threshold}:\n\n")
        for file_path in large_values_files:
            f.write(f"{file_path}\n")

    # Print statistics information
    print(f"\nEgoDx 48-dimensional action statistics completed! Successfully processed {file_count} files")
    print(f"Action dimensions: {len(global_min) if global_min is not None else 'N/A'}")
    if global_min is not None:
        print(f"Min values (first 10): {global_min[:10]}")
        print(f"Max values (first 10): {global_max[:10]}")
        print(f"Overall range: [{global_min.min():.6f}, {global_max.max():.6f}]")
    print(f"Found {len(large_values_files)} files containing data with absolute values > {threshold}, saved to {large_values_log}")
    
    if error_files:
        print(f"Failed to process files ({len(error_files)}):")
        for i, (path, err) in enumerate(error_files):
            if i < 10:  # Only show first 10 errors
                print(f"- {path}: {err}")
            else:
                print(f"...and {len(error_files)-10} more errors")
                break

def collect_action_stats(root_dir, output_path, large_values_log="large_values.txt"):
    """
    Calculate the extreme values of joint action data from all RobotWin HDF5 files
    Args:
        root_dir: Root directory of HDF5 files (recursively searched)
        output_path: Output JSON file path
        large_values_log: Log file for recording data files with absolute values > 10
    """
    # Initialize statistics containers
    global_min = None
    global_max = None
    file_count = 0
    error_files = []
    
    # Record files with absolute values > 10
    large_values_files = []

    # Recursively find all hdf5 files
    hdf5_files = glob(os.path.join(root_dir, "**/*.hdf5"), recursive=True)
    print(f"Found {len(hdf5_files)} HDF5 files")

    # Process each file
    for file_path in tqdm(hdf5_files, desc="Processing files"):
        try:
            with h5py.File(file_path, 'r') as f:
                if "joint_states" in f:
                    action_data = f['joint_states']['positions'][:]
                    
                    # Check if there are dimensions with absolute values > 10
                    if np.any(np.abs(action_data) > 5):
                        large_values_files.append(file_path)
                        
                        # Get specific frame and dimension information (optional)
                        abs_data = np.abs(action_data)
                        frames, dims = np.where(abs_data > 5)
                        for frame, dim in zip(frames, dims):
                            # Can record more detailed information like frame and dimension indices
                            pass
                    
                    # Initialize or update global extremes
                    if global_min is None:
                        global_min = np.min(action_data, axis=0)
                        global_max = np.max(action_data, axis=0)
                    else:
                        global_min = np.minimum(global_min, np.min(action_data, axis=0))
                        global_max = np.maximum(global_max, np.max(action_data, axis=0))
                    
                    file_count += 1
                else:
                    error_files.append((file_path, "Missing joint_states/positions data"))
        except Exception as e:
            error_files.append((file_path, str(e)))

    # Generate statistics results
    stat_dict = {
        "cvpr_real": {
            "min": global_min.tolist() if global_min is not None else [],
            "max": global_max.tolist() if global_max is not None else [],
        }
    }

    # Save results
    with open(output_path, 'w') as f:
        json.dump(stat_dict, f, indent=4)
    
    # Save list of files with absolute values > 10
    with open(large_values_log, 'w') as f:
        f.write(f"Found {len(large_values_files)} files containing joint position data with absolute values > 10:\n\n")
        for file_path in large_values_files:
            f.write(f"{file_path}\n")

    # Print statistics information
    print(f"\nStatistics completed! Successfully processed {file_count} files")
    print(f"Action dimensions: {len(global_min) if global_min is not None else 'N/A'}")
    print(f"Min values: {global_min}")
    print(f"Max values: {global_max}")
    print(f"Found {len(large_values_files)} files with absolute values > 10, saved to {large_values_log}")
    
    if error_files:
        print(f"Failed files ({len(error_files)}):")
        for i, (path, err) in enumerate(error_files):
            if i < 10:  # Only show first 10 errors
                print(f"- {path}: {err}")
            else:
                print(f"...and {len(error_files)-10} more errors")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate EgoDex dataset statistics')
    parser.add_argument('--data_root', type=str,
                       default=os.environ.get('EGODEX_DATA_ROOT', '/share/hongzhe/datasets/egodex'),
                       help='EgoDex dataset root directory')
    parser.add_argument('--output_path', type=str,
                       help='Output JSON file path for statistics')
    parser.add_argument('--large_values_log', type=str,
                       help='Output text file path for files with large values')
    
    args = parser.parse_args()
    
    # Set default paths if not provided
    if args.output_path is None:
        project_root = os.environ.get('HRDT_PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
        output_dir = os.environ.get('HRDT_OUTPUT_DIR', os.path.join(project_root, 'datasets/pretrain'))
        args.output_path = os.path.join(output_dir, "egodex_stat.json")
    
    if args.large_values_log is None:
        output_dir = os.path.dirname(args.output_path)
        args.large_values_log = os.path.join(output_dir, "egodex_large_values.txt")
    
    print(f"Configuration:")
    print(f"- Data root: {args.data_root}")
    print(f"- Output file: {args.output_path}")
    print(f"- Large values log: {args.large_values_log}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Usage example for EgoDex dataset
    collect_egodex_action_stats(
        root_dir=args.data_root,  # EgoDx dataset root directory
        output_path=args.output_path,  # Output file path
        large_values_log=args.large_values_log  # Outlier log file
    )
