#!/usr/bin/env python3
"""
Pre-compute 48-dimensional action data and store in HDF5 files
Add 'actions_48d' key to all EgoDex dataset HDF5 files, containing pre-computed 48-dimensional action data
"""

import os
import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path
import multiprocessing as mp
from multiprocessing import Process, Queue
import traceback
import time


def construct_48d_action_from_hdf5(transforms_group, frame_idx):
    """
    Construct 48-dimensional hand action representation from HDF5 file
    
    Args:
        transforms_group: transforms group in HDF5
        frame_idx: frame index
        
    Returns:
        action_vector: 48-dimensional action vector
    """
    action_vector = []
    
    # Fingertip joint name mapping
    fingertip_joints = {
        'left': ['leftThumbTip', 'leftIndexFingerTip', 'leftMiddleFingerTip', 
                'leftRingFingerTip', 'leftLittleFingerTip'],
        'right': ['rightThumbTip', 'rightIndexFingerTip', 'rightMiddleFingerTip',
                 'rightRingFingerTip', 'rightLittleFingerTip']
    }
    
    for hand_side in ['left', 'right']:
        hand_key = f"{hand_side}Hand"
        
        # Get hand 4x4 transformation matrix
        hand_transform = transforms_group[hand_key][frame_idx]
            
        # 1. Wrist 3D position
        hand_position = hand_transform[:3, 3]
        action_vector.extend(hand_position)
            
        # 2. Wrist 6D orientation (first two columns of rotation matrix)
        rotation_matrix = hand_transform[:3, :3]
        rotation_6d = np.concatenate([rotation_matrix[:, 0], rotation_matrix[:, 1]])
        action_vector.extend(rotation_6d)
            
        # 3. 3D positions of 5 fingertips
        for fingertip in fingertip_joints[hand_side]:
            fingertip_transform = transforms_group[fingertip][frame_idx]
            fingertip_pos = fingertip_transform[:3, 3]
            action_vector.extend(fingertip_pos)
    
    return np.array(action_vector, dtype=np.float32)


def process_single_file(hdf5_path, force_overwrite=False):
    """
    Process single HDF5 file, add 48-dimensional action data
    
    Args:
        hdf5_path: HDF5 file path
        force_overwrite: Whether to force overwrite existing actions_48d data
        
    Returns:
        (success, error_message, total_frames)
    """
    try:
        # First check if file already has actions_48d
        with h5py.File(hdf5_path, 'r') as f:
            if 'actions_48d' in f and not force_overwrite:
                # Check data integrity
                if 'transforms' in f:
                    transforms_group = f['transforms']
                    expected_frames = list(transforms_group.values())[0].shape[0]
                    actual_frames = f['actions_48d'].shape[0]
                    if actual_frames == expected_frames:
                        return True, "Already exists and complete", actual_frames
                    else:
                        print(f"Warning: {hdf5_path} has incomplete actions_48d ({actual_frames}/{expected_frames}), will regenerate")
                else:
                    return False, "Missing transforms data", 0
        
        # Read data and compute 48-dimensional actions
        with h5py.File(hdf5_path, 'r') as f:
            if "transforms" not in f:
                return False, "Missing transforms data", 0
                
            transforms_group = f['transforms']
            total_frames = list(transforms_group.values())[0].shape[0]
            
            # Construct 48-dimensional action data for all frames
            actions_48d = []
            for frame_idx in range(total_frames):
                try:
                    action_vector = construct_48d_action_from_hdf5(transforms_group, frame_idx)
                    actions_48d.append(action_vector)
                except Exception as e:
                    print(f"Error processing frame {frame_idx} in {hdf5_path}: {e}")
            
            if not actions_48d:
                return False, "No valid frames processed", 0
            
            actions_array = np.array(actions_48d)
        
        # Write 48-dimensional action data to file
        with h5py.File(hdf5_path, 'a') as f:
            # Delete old actions_48d data (if exists)
            if 'actions_48d' in f:
                del f['actions_48d']
            
            # Create new actions_48d dataset
            f.create_dataset(
                'actions_48d', 
                data=actions_array,
                compression='gzip',
                compression_opts=9
            )
            
            # Add metadata
            f['actions_48d'].attrs['description'] = '48-dimensional hand action representation'
            f['actions_48d'].attrs['format'] = 'left_hand(24d) + right_hand(24d)'
            f['actions_48d'].attrs['hand_format'] = 'position(3d) + rotation_6d(6d) + fingertips(15d)'
            f['actions_48d'].attrs['created_by'] = 'precompute_48d_actions.py'
            f['actions_48d'].attrs['created_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        return True, "Success", total_frames
        
    except Exception as e:
        return False, str(e), 0


def worker_process(process_id, file_list, progress_queue, force_overwrite):
    """Worker process function"""
    try:
        print(f"Process {process_id} starting to process {len(file_list)} files")
        
        success_count = 0
        error_count = 0
        skip_count = 0
        total_frames_processed = 0
        
        for file_path in file_list:
            success, message, frames = process_single_file(file_path, force_overwrite)
            
            if success:
                if "Already exists" in message:
                    skip_count += 1
                else:
                    success_count += 1
                total_frames_processed += frames
                progress_queue.put(('processed', process_id, frames))
            else:
                error_count += 1
                print(f"Process {process_id}: Error processing {file_path}: {message}")
                progress_queue.put(('error', process_id, file_path, message))
        
        print(f"Process {process_id} completed: successful {success_count}, skipped {skip_count}, error {error_count}, total frames {total_frames_processed}")
        progress_queue.put(('done', process_id, success_count, skip_count, error_count, total_frames_processed))
        
    except Exception as e:
        print(f"Process {process_id} encountered serious error: {str(e)}")
        traceback.print_exc()
        progress_queue.put(('process_error', process_id, str(e)))


def progress_monitor(total_files, progress_queue, num_processes):
    """Progress monitoring function"""
    processed_files = 0
    error_files = 0
    finished_processes = 0
    total_frames = 0
    
    pbar = tqdm(total=total_files, desc="Processing files")
    
    while finished_processes < num_processes:
        try:
            msg = progress_queue.get(timeout=1)
            
            if msg[0] == 'processed':
                processed_files += 1
                total_frames += msg[2]  # frames count
                pbar.update(1)
            elif msg[0] == 'error':
                error_files += 1
                pbar.update(1)
            elif msg[0] == 'done':
                finished_processes += 1
                process_id, success, skip, error, frames = msg[1], msg[2], msg[3], msg[4], msg[5]
                print(f"\nProcess {process_id} completed: successful {success}, skipped {skip}, error {error}, total frames {frames}")
            elif msg[0] == 'process_error':
                finished_processes += 1
                process_id, error_msg = msg[1], msg[2]
                print(f"\nProcess {process_id} error: {error_msg}")
                
        except:
            # Timeout, continue waiting
            continue
    
    pbar.close()
    return processed_files, error_files, total_frames


def collect_all_hdf5_files(root_dir):
    """Collect all HDF5 files"""
    hdf5_files = []
    root_path = Path(root_dir)
    
    # Traverse all part directories
    for part in ['part1', 'part2', 'part3', 'part4', 'part5', 'extra', 'test']:
        part_dir = root_path / part
        if part_dir.exists():
            for task_dir in part_dir.iterdir():
                if task_dir.is_dir():
                    task_hdf5_files = list(task_dir.glob('*.hdf5'))
                    hdf5_files.extend(task_hdf5_files)
    
    return hdf5_files


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Pre-compute 48-dimensional action data for EgoDex dataset')
    parser.add_argument('--data_root', type=str, default='/share/hongzhe/datasets/egodex',
                       help='EgoDex dataset root directory')
    parser.add_argument('--num_processes', type=int, 
                       default=int(os.environ.get('NUM_PROCESSES', 64)),
                       help='Number of parallel processes')
    parser.add_argument('--force_overwrite', default=True, action='store_true',
                       help='Force overwrite existing actions_48d data')
    parser.add_argument('--test_mode', action='store_true',
                       help='Test mode, only process a few files')
    
    args = parser.parse_args()
    
    print("🚀 Starting to pre-compute 48-dimensional action data...")
    print(f"Data root directory: {args.data_root}")
    print(f"Number of parallel processes: {args.num_processes}")
    print(f"Force overwrite: {args.force_overwrite}")
    print("=" * 60)
    
    # Collect all HDF5 files
    print("📂 Collecting HDF5 files...")
    all_files = collect_all_hdf5_files(args.data_root)
    
    if args.test_mode:
        # Test mode, only process first 10 files
        all_files = all_files[:10]
        print(f"🧪 Test mode: processing first {len(all_files)} files")
    
    if not all_files:
        print("❌ No HDF5 files found!")
        return
    
    print(f"📊 Found {len(all_files)} HDF5 files")
    
    # Distribute files to processes
    files_per_process = len(all_files) // args.num_processes
    file_lists = []
    
    for i in range(args.num_processes):
        start_idx = i * files_per_process
        if i == args.num_processes - 1:  # Last process handles remaining files
            end_idx = len(all_files)
        else:
            end_idx = start_idx + files_per_process
        
        file_lists.append(all_files[start_idx:end_idx])
    
    print(f"🔧 Will use {args.num_processes} processes for parallel processing")
    for i, file_list in enumerate(file_lists):
        print(f"  Process {i}: {len(file_list)} files")
    
    # Create progress queue
    progress_queue = Queue()
    
    # Start progress monitoring process
    monitor_process = Process(
        target=progress_monitor, 
        args=(len(all_files), progress_queue, args.num_processes)
    )
    monitor_process.start()
    
    # Start worker processes
    processes = []
    for i in range(args.num_processes):
        process = Process(
            target=worker_process,
            args=(i, file_lists[i], progress_queue, args.force_overwrite)
        )
        process.start()
        processes.append(process)
        time.sleep(0.1)  # Slight delay for startup
    
    # Wait for all processes to complete
    for process in processes:
        process.join()
    
    # Wait for progress monitoring to complete
    monitor_process.join()
    
    print("\n" + "=" * 60)
    print("🎉 48-dimensional action data pre-computation completed!")
    
    # Verify some files
    print("\n🔍 Verifying processing results...")
    sample_files = all_files[:5]  # Verify first 5 files
    for file_path in sample_files:
        try:
            with h5py.File(file_path, 'r') as f:
                if 'actions_48d' in f:
                    actions_shape = f['actions_48d'].shape
                    transforms_shape = list(f['transforms'].values())[0].shape[0] if 'transforms' in f else 0
                    print(f"✅ {file_path.name}: actions_48d {actions_shape}, transforms frames {transforms_shape}")
                else:
                    print(f"❌ {file_path.name}: Missing actions_48d")
        except Exception as e:
            print(f"❌ {file_path.name}: Verification failed - {e}")
    
    print("\n💡 Usage:")
    print("In EgoDexDataset class, you can directly use f['actions_48d'] to read pre-computed 48-dimensional action data")
    print("This will significantly speed up data loading during training!")


if __name__ == "__main__":
    # Set multi-process start method
    mp.set_start_method('spawn', force=True)
    main() 