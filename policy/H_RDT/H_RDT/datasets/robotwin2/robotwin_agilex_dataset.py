import os
import random
import h5py
import numpy as np
import cv2
import json
import pandas as pd
import time
import glob
from typing import List, Dict, Optional
import sys
import warnings
import traceback
import torch

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*multichannel.*"
)

# Import imgaug libraries
import imgaug as ia
import imgaug.augmenters as iaa

if not hasattr(np, 'bool'):
    np.bool = bool

class RobotwinAgilexDataset:
    """
    Dataset for loading RobotWin Agilex robot data with joint actions.
    Supports both single-task and multi-task modes with balanced sampling.
    """
    def __init__(
        self, 
        mode="multi_task",  # "single_task" or "multi_task"
        single_task_root_dir=None,  # For single task mode
        multi_task_root_dir=None,  # For multi task mode
        task_name=None,  # Required for single_task mode
        hdf5_folder=None,  # Required for single_task mode, e.g., "demo_clean/data"
        max_episodes=None,  # Maximum number of episodes to load for single task
        config=None,
        stat_path=None,
        upsample_rate=3,
        val=False,
        image_corrupt_severity=5
    ):
        """
        Initialize RobotWin dataset with single/multi-task support.
        
        Args:
            mode: "single_task" or "multi_task"
            single_task_root_dir: Root directory for single task mode
            multi_task_root_dir: Root directory for multi task mode
            task_name: Task name for single task mode (e.g., "beat_block_hammer")
            hdf5_folder: HDF5 folder path for single task mode (e.g., "demo_clean/data")
            max_episodes: Maximum episodes to load for single task (None = all)
            config: Configuration dictionary
            stat_path: Path to normalization statistics file
            upsample_rate: Temporal data upsampling rate
            val: Whether this is validation set
            image_corrupt_severity: Image corruption severity level
        """
        self.DATASET_NAME = "robotwin_agilex"
        self.mode = mode
        self.single_task_root_dir = single_task_root_dir
        self.multi_task_root_dir = multi_task_root_dir
        self.task_name = task_name
        self.hdf5_folder = hdf5_folder
        self.max_episodes = max_episodes
        self.upsample_rate = upsample_rate
        self.val = val
        self.image_corrupt_severity = image_corrupt_severity
        
        # Validate mode-specific parameters
        if mode == "single_task":
            if not task_name or not hdf5_folder:
                raise ValueError("single_task mode requires task_name and hdf5_folder parameters")
        elif mode == "multi_task":
            pass  # No additional validation needed
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'single_task' or 'multi_task'")
        
        # Set basic parameters
        self.chunk_size = config['common']['action_chunk_size']
        self.state_dim = config['common']['action_dim']
        self.img_history_size = config['common']['img_history_size']
        
        # Set camera parameters for multi-view (3 cameras only)
        self.num_cameras = config['common']['num_cameras']
        if self.num_cameras == 3:
            self.cameras = ["cam_high", "cam_right_wrist", "cam_left_wrist"]
        else:
            raise ValueError(f"Unsupported num_cameras={self.num_cameras}, only 3 cameras supported.")

        self.camera_mapping = {
            "cam_high": "head_camera", 
            "cam_left_wrist": "left_camera",
            "cam_right_wrist": "right_camera"
        }
        
        # Set default stat_path if not provided (relative to this file)
        if stat_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            stat_path = os.path.join(current_dir, 'stats.json')
        
        # Load statistics for normalization
        with open(stat_path, 'r') as file:
            stat = json.load(file)
        self.action_min = np.array(stat['robotwin_agilex']['min'])
        self.action_max = np.array(stat['robotwin_agilex']['max'])
        
        # Initialize data structures
        if mode == "single_task":
            self.episode_files = []  # List of episode files for single task
        else:
            self.task_to_episodes = {}  # Task name -> episode files mapping for multi task
            self.task_weights = {}      # Task sampling weights
        
        self.total_episodes = 0     # Total number of episodes
        
        # Initialize dataset
        self._initialize_dataset()
    
    def _scan_single_task_folder(self):
        """Scan single task folder to get all HDF5 files"""
        task_dir = os.path.join(self.single_task_root_dir, self.task_name, self.hdf5_folder)
        if not os.path.exists(task_dir):
            print(f"Warning: Task folder {task_dir} does not exist")
            return []
        
        # Find all HDF5 files in the folder
        hdf5_files = []
        for f in os.listdir(task_dir):
            if f.endswith(".hdf5"):
                hdf5_path = os.path.join(task_dir, f)
                hdf5_files.append(hdf5_path)
        
        # Sort files to ensure consistent ordering
        hdf5_files.sort()
        
        # Limit number of episodes if specified
        if self.max_episodes is not None:
            hdf5_files = hdf5_files[:self.max_episodes]
        
        print(f"Single task {self.task_name}: Found {len(hdf5_files)} HDF5 files")
        return hdf5_files
    
    def _scan_multi_task_folders(self):
        """Scan multi-task folders to get all HDF5 files"""
        if not os.path.exists(self.multi_task_root_dir):
            print(f"Warning: Multi-task root directory {self.multi_task_root_dir} does not exist")
            return {}
        
        # Get all task folders
        task_folders = [d for d in os.listdir(self.multi_task_root_dir) 
                       if os.path.isdir(os.path.join(self.multi_task_root_dir, d))]
        
        task_to_episodes = {}
        
        for task_folder in task_folders:
            task_dir = os.path.join(self.multi_task_root_dir, task_folder)
            hdf5_files = []
            
            # Recursively find all HDF5 files in task folder
            for root, dirs, files in os.walk(task_dir):
                for file in files:
                    if file.endswith(".hdf5"):
                        hdf5_path = os.path.join(root, file)
                        hdf5_files.append(hdf5_path)
            
            # Shuffle files for randomness
            random.shuffle(hdf5_files)
            task_to_episodes[task_folder] = hdf5_files
            print(f"Multi-task {task_folder}: Found {len(hdf5_files)} HDF5 files")
        
        return task_to_episodes
    
    def _initialize_dataset(self):
        """Initialize dataset by rescanning folders and updating sampling weights"""
        print("Initializing dataset...")
        
        if self.mode == "single_task":
            # Single task mode: scan single task folder
            self.episode_files = self._scan_single_task_folder()
            self.total_episodes = len(self.episode_files)
            
            if self.total_episodes == 0:
                raise ValueError("Error: No HDF5 files found, please check data path")
            
            print(f"Single task dataset initialized. Total {self.total_episodes} episodes")
        
        else:
            # Multi task mode: scan all task folders
            self.task_to_episodes = self._scan_multi_task_folders()
            
            # Calculate total episodes
            all_task_count = sum(len(episodes) for episodes in self.task_to_episodes.values())
            
            if all_task_count == 0:
                raise ValueError("Error: No HDF5 files found, please check data path")
            
            # Calculate sampling weights - equal weight for all tasks (1:1:1:1:1:1 sampling)
            task_count = len(self.task_to_episodes)
            for task_name in self.task_to_episodes.keys():
                self.task_weights[task_name] = 1.0 / task_count
            
            self.total_episodes = all_task_count
            
            print(f"Multi-task dataset initialized. Total {all_task_count} episodes across {task_count} tasks")
            print(f"Task weights: {self.task_weights}")
    
    def __len__(self):
        """Return approximate dataset length"""
        return self.total_episodes * 200  # Assume 200 samples per HDF5 file
    
    def get_dataset_name(self):
        """Return dataset name"""
        return self.DATASET_NAME

    def parse_img_data(self, dataset, idx):
        """
        Process single camera image data from individual camera paths
        
        Args:
            dataset: HDF5 dataset containing single camera images
            idx: Current frame index
            
        Returns:
            Processed image sequence with shape [history_size, H, W, 3]
        """
        start_i = max(idx - self.img_history_size * self.upsample_rate + 1, 0)
        num_frames = (idx - start_i) // self.upsample_rate + 1

        # Use standard resolution for individual camera images
        frames = np.zeros((num_frames, 240, 320, 3), dtype=np.uint8)  # Standard size: 320x240
        
        try:
            for i, frame_idx in enumerate(range(start_i, idx + 1, self.upsample_rate)):
                if frame_idx < len(dataset):
                    img_data = dataset[frame_idx]
                    
                    decoded_img = self.decode_image_with_opencv(img_data)
                    if decoded_img is None:
                        raise Exception(f"[DEBUG] decode error")

                    if decoded_img is not None:
                        frames[i] = decoded_img

        except Exception as e:
            print(f"[DEBUG] decode_image_with_opencv error: {e}")

        if num_frames < self.img_history_size:
            pad_frames = np.repeat(frames[:1], self.img_history_size - num_frames, axis=0)
            frames = np.concatenate([pad_frames, frames])
        
        return frames

    def decode_image_with_opencv(self, img_data):
        """
        Decode image data using OpenCV, maintaining RGB format
        
        Args:
            img_data (bytes): Binary image data
        
        Returns:
            np.ndarray: Decoded image array, shape=(240, 320, 3), RGB format
        """
        try:
            # Decode using OpenCV
            nparr = np.frombuffer(img_data, np.uint8)
            bgr_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Ensure decoding was successful
            if bgr_img is None:
                raise Exception("OpenCV decoding failed")
                
            # OpenCV uses BGR format by default, convert to RGB
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            
            # Resize to expected standard size if needed (320x240)
            if rgb_img.shape[:2] != (240, 320):
                rgb_img = cv2.resize(rgb_img, (320, 240))
                
            return rgb_img
            
        except Exception as e:
            print(f"[DEBUG] Image decoding failed: {e}")
            return None

    def load_language_embedding(self, hdf5_file_path):
        """
        Load pre-encoded language instruction embedding from centralized lang_embeddings folder
        
        Args:
            hdf5_file_path (str): Path to HDF5 episode file
            
        Returns:
            torch.Tensor: Task instruction embedding, or None if not found
        """
        try:
            # Extract task name from HDF5 file path
            # Example: HDF5 file: /path/to/dataset/adjust_bottle/demo_clean/data/episode0.hdf5
            # Embedding file: /path/to/H_RDT/datasets/robotwin2/lang_embeddings/adjust_bottle.pt
            # -> task_name = "adjust_bottle"
            
            task_name = None
            
            if self.mode == "single_task":
                # For single task mode, use the provided task_name
                task_name = self.task_name
            else:
                # For multi-task mode, extract task name from file path
                if self.multi_task_root_dir and self.multi_task_root_dir in hdf5_file_path:
                    # Get relative path from multi_task_root_dir
                    relative_path = os.path.relpath(hdf5_file_path, self.multi_task_root_dir)
                    # Task name is the first directory in relative path
                    task_name = relative_path.split(os.sep)[0]
            
            if task_name is None:
                print(f"Warning: Could not extract task name from path {hdf5_file_path}")
                return None
            
            # Build embedding file path using relative path to current script location
            current_dir = os.path.dirname(os.path.abspath(__file__))
            embedding_path = os.path.join(current_dir, 'lang_embeddings', f"{task_name}.pt")
            
            if not os.path.exists(embedding_path):
                print(f"Warning: Task embedding file not found: {embedding_path}")
                return None
            
            # Load embedding data
            embedding_data = torch.load(embedding_path, map_location='cpu')
            
            # Extract embeddings tensor
            embeddings = embedding_data.get('embeddings', None)
            
            if embeddings is None:
                print(f"Warning: No embeddings found in {embedding_path}")
                return None
            
            # Remove batch dimension if present (convert from 3D to 2D)
            if embeddings.dim() == 3:
                embeddings = embeddings.squeeze(0)
            
            # print(f"[DEBUG] Loaded task instruction for '{task_name}' from {embedding_path}")
            return embeddings
            
        except Exception as e:
            print(f"Error loading language embedding from {hdf5_file_path}: {e}")
            return None

    def extract_episode_item(self, hdf5_file):
        """
        Extract a single sample from HDF5 file
        
        Args:
            hdf5_file: HDF5 file path
            
        Returns:
            Dictionary containing extracted data, or None if extraction fails
        """
        try:
            with h5py.File(hdf5_file, 'r', swmr=True, libver='latest') as f:
                # Load joint action data from new HDF5 structure
                try:
                    left_arm = f["joint_action/left_arm"][:]
                    left_gripper = f["joint_action/left_gripper"][:]
                    right_arm = f["joint_action/right_arm"][:]
                    right_gripper = f["joint_action/right_gripper"][:]
                    
                    # Handle dimension mismatch
                    # If left arm is 2D but left gripper is 1D, expand gripper dimensions
                    if len(left_arm.shape) == 2 and len(left_gripper.shape) == 1:
                        left_gripper = left_gripper.reshape(-1, 1)
                    
                    # If right arm is 2D but right gripper is 1D, expand gripper dimensions
                    if len(right_arm.shape) == 2 and len(right_gripper.shape) == 1:
                        right_gripper = right_gripper.reshape(-1, 1)
                    
                    # Concatenate all parts into complete action vector
                    action_data = np.concatenate([left_arm, left_gripper, right_arm, right_gripper], axis=1)
                    
                except Exception as e:
                    print(f"Error loading joint action data: {e}")
                    return None
                
                # Adjust data indexing
                max_index = len(action_data) - 2
                index = random.randint(0, max_index)
                
                # Current state (using joint action data)
                action_current = action_data[index]
                
                # Future action sequence
                action_end = min(index + self.chunk_size * self.upsample_rate, max_index + 1)
                action_chunk = action_data[index+1:action_end+1:self.upsample_rate]
                
                # If action sequence is insufficient, repeat last frame for padding
                if action_chunk.shape[0] < self.chunk_size:
                    last_part = np.repeat(action_chunk[-1:], self.chunk_size - action_chunk.shape[0], axis=0)
                    action_chunk = np.concatenate([action_chunk, last_part], axis=0)
                
                # Get multi-view camera data
                try:
                    current_images = []
                    
                    # Define camera paths for 3-view setup (no front camera)
                    camera_paths = {
                        "cam_high": "observation/head_camera/rgb",
                        "cam_left_wrist": "observation/left_camera/rgb",
                        "cam_right_wrist": "observation/right_camera/rgb"
                    }
                    
                    # Load images from each configured camera
                    for cam_idx, cam_name in enumerate(self.cameras):
                        cam_path = camera_paths.get(cam_name)
                        if cam_path and cam_path in f:
                            camera_data = f[cam_path]
                            img_frames = self.parse_img_data(camera_data, index)
                            current_images.append(img_frames)
                        else:
                            print(f"Warning: Camera {cam_name} not found in {hdf5_file}")
                            return None
                    
                    # Ensure we have the correct number of cameras
                    if len(current_images) != self.num_cameras:
                        print(f"Error: Expected {self.num_cameras} cameras, but got {len(current_images)}")
                        return None
                    
                    # Convert to numpy array with shape [num_cameras, history_size, H, W, 3]
                    img_frames_np = np.array(current_images)
                    
                    # Create image masks for each camera
                    mask_length = self.img_history_size
                    current_images_mask = [
                        np.array([True]*mask_length, dtype=bool) for _ in range(self.num_cameras)
                    ]
                    
                except Exception as e:
                    print(f"Error accessing camera data in {hdf5_file}: {e}")
                    traceback.print_exc()
                    return None
                
                # Load pre-encoded language instruction
                language_embedding = self.load_language_embedding(hdf5_file)
                if language_embedding is None:
                    print(f"Warning: Failed to load language embedding for {hdf5_file}")
                    return None
                
                state_indicator = np.ones_like(action_current)
                action_norm = np.ones_like(action_chunk)
                
                # Create result dictionary
                result = {
                    "current_images": img_frames_np,  # Current frame images (possibly augmented)
                    "current_images_mask": current_images_mask,  # Image masks
                    "actions": action_chunk,  # Action sequence
                    "states": np.expand_dims(action_current, axis=0),  # States
                    "state_indicator": state_indicator,
                    "action_norm": action_norm,
                    "instruction": language_embedding,  # Pre-encoded language instruction
                    "dataset_name": self.DATASET_NAME,  # Dataset name
                }
                
                return result

        except Exception as e:
            print(f"Error processing {hdf5_file}: {e}")
            traceback.print_exc()
            return None

    def get_item(self, index=None):
        """
        Get data item, randomly select one if index is None
        
        Args:
            index: Optional, specified index. If None, randomly select
            
        Returns:
            Processed data dictionary, or None if failed
        """
        if self.mode == "single_task":
            # Single task mode: randomly select from episode files
            if not self.episode_files:
                self._initialize_dataset()
            
            if not self.episode_files:
                print("Error: No available episodes")
                return None
            
            # Randomly select an HDF5 file
            episode_file = random.choice(self.episode_files)
            
        else:
            # Multi task mode: balanced task sampling
            if not self.task_to_episodes:
                self._initialize_dataset()
            
            # Randomly select a task based on task weights
            task_name = random.choices(
                list(self.task_weights.keys()),
                weights=list(self.task_weights.values()),
                k=1
            )[0]
            
            # Randomly select a sample from the selected task
            task_episodes = self.task_to_episodes.get(task_name, [])
            if not task_episodes:
                print(f"Warning: Task {task_name} has no available samples")
                # Select from other tasks
                alternative_tasks = [t for t in self.task_to_episodes.keys() if t != task_name and self.task_to_episodes.get(t, [])]
                if not alternative_tasks:
                    print("Error: No available samples")
                    return None
                task_name = random.choice(alternative_tasks)
                task_episodes = self.task_to_episodes.get(task_name, [])
            
            # Randomly select an HDF5 file
            episode_file = random.choice(task_episodes)
        
        # Try to extract sample data
        for _ in range(3):  # Maximum 3 attempts
            item = self.extract_episode_item(episode_file)
            if item is not None:
                return item
            # If current sample extraction fails, randomly select another
            if self.mode == "single_task":
                episode_file = random.choice(self.episode_files)
            else:
                task_episodes = self.task_to_episodes.get(task_name, [])
                if task_episodes:
                    episode_file = random.choice(task_episodes)
        
        print(f"Warning: Failed to extract sample, returning None")
        return None

if __name__ == "__main__":
    # Test code
    import argparse
    from omegaconf import OmegaConf
    
    parser = argparse.ArgumentParser(description='Test RobotwinAgilexDataset')
    parser.add_argument('--config', type=str, default='/share/hongzhe/VLA/round2/dino_siglip/configs/hrdt.yaml', help='Config file path')
    parser.add_argument('--mode', type=str, default='single_task', choices=['single_task', 'multi_task'], help='Dataset mode')
    parser.add_argument('--task_name', type=str, default='beat_block_hammer', help='Task name for single task mode')
    parser.add_argument('--hdf5_folder', type=str, default='demo_clean/data', help='HDF5 folder for single task mode')
    parser.add_argument('--max_episodes', type=int, default=None, help='Maximum episodes for single task mode')
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Create dataset instance
    print("Creating dataset instance...")
    if args.mode == "single_task":
        ds = RobotwinAgilexDataset(
            mode="single_task",
            task_name=args.task_name,
            hdf5_folder=args.hdf5_folder,
            max_episodes=args.max_episodes,
            config=config, 
            val=False
        )
    else:
        ds = RobotwinAgilexDataset(
            mode="multi_task",
            config=config, 
            val=False
        )
    
    # Test data loading
    print("\nTesting data loading...")
    success_count = 0
    test_times = 10
    
    for i in range(test_times):
        print(f"\nAttempt {i+1}/{test_times}...")
        item = ds.get_item()
        
        if item is not None:
            success_count += 1
            print("Successfully loaded data")
            print(f"Instruction embedding shape: {item['instruction'].shape}")
            print(f"Multi-view images shape: {item['current_images'].shape}")
            print(f"Number of cameras: {len(item['current_images_mask'])}")
            print(f"Actions shape: {item['actions'].shape}")
            print(f"States shape: {item['states'].shape}")
    
    print(f"\nResult: Successfully loaded {success_count}/{test_times} samples") 