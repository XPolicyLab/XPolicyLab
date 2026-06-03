# AC-One Dataset Loader for Motus
# Supports AC-One robot data with video and action data from multiple tasks

import os
import random
import json
import numpy as np
import cv2
import torch
import torch.utils.data as data
from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path
from PIL import Image
import glob

# VLM processing imports
from utils.vlm_utils import preprocess_vlm_messages
from transformers import AutoProcessor
import warnings

# Import image processing utilities
from data.utils.image_utils import (
    resize_with_padding, tensor_to_pil, apply_image_augmentation,
    load_video_frames, get_video_frame_count, load_first_frame
)

# Import normalization functions
from data.utils.norm import normalize_actions, denormalize_actions, load_normalization_stats

warnings.filterwarnings("ignore", category=FutureWarning, message=".*multichannel.*")

logger = logging.getLogger(__name__)

class ACOneDataset(data.Dataset):
    """
    Dataset for AC-One data with task-level organization.
    Updated for three-modal UniDiffuser (WAN + Action Expert + VLM).
    
    Data structure:
    /share/dataset/preprocess/ac_one/
    ├── task_category_1/  # e.g., fold_towel
    │   ├── task_variant_1.json  # e.g., fold_blue_and_white_striped_towel_neatly_using_both_hands.json
    │   ├── task_variant_1/      # e.g., fold_blue_and_white_striped_towel_neatly_using_both_hands/
    │   │   ├── videos/
    │   │   │   ├── 0.mp4
    │   │   │   ├── 1.mp4
    │   │   │   └── ...
    │   │   ├── qpos/
    │   │   │   ├── 0.pt
    │   │   │   ├── 1.pt
    │   │   │   └── ...
    │   │   └── instructions/
    │   │       ├── task_variant_1.txt
    │   │       └── task_variant_1.pt
    │   └── ...
    └── task_category_2/
    """
    
    def __init__(
        self,
        dataset_dir: str = "/share/dataset/preprocess/ac_one",
        
        # Sampling parameters
        global_downsample_rate: int = 1,  # Global downsampling (e.g., 30Hz -> 10Hz)
        video_action_freq_ratio: int = 5,  # Video:Action frequency ratio  
        num_video_frames: int = 8,  # Number of video frames to predict
        video_size: Tuple[int, int] = (736, 640),  # (height, width)
        
        # Task selection
        task_mode: str = "multi",             # "single" or "multi" 
        task_name: Optional[str | List[str]] = None,  # For single: single task name; For multi: task list or None (all tasks)
        
        # Episode limits
        max_episodes: int = 10000,
        val_episodes: int = 100,
        val: bool = False,
        
        # Data augmentation
        image_aug: bool = False,
        
        # VLM processing
        vlm_checkpoint_path: Optional[str] = None,
        **kwargs
    ):
        super().__init__()
        
        self.dataset_dir = Path(dataset_dir)
        self.global_downsample_rate = global_downsample_rate
        self.video_action_freq_ratio = video_action_freq_ratio
        self.num_video_frames = num_video_frames
        self.video_size = video_size
        self.task_mode = task_mode
        self.task_name = task_name
        self.action_chunk_size = self.num_video_frames * self.video_action_freq_ratio
        
        # Validate task mode configuration
        if task_mode == "single" and task_name is None:
            raise ValueError("task_name must be specified when task_mode is 'single'")
        if task_mode not in ["single", "multi"]:
            raise ValueError("task_mode must be either 'single' or 'multi'")
        
        # Normalize task_name to list for consistent handling
        if task_name is not None:
            self.task_list = [task_name] if isinstance(task_name, str) else list(task_name)
        else:
            self.task_list = None
            
        self.max_episodes = max_episodes
        self.val = val
        self.image_aug = image_aug and not val  # No augmentation for validation
        
        # VLM processor
        self.vlm_processor = None
        if vlm_checkpoint_path:
            try:
                self.vlm_processor = AutoProcessor.from_pretrained(vlm_checkpoint_path)
                logger.info(f"Loaded VLM processor from {vlm_checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to load VLM processor: {e}")
        
        # Load dataset episodes
        self.episodes = self._load_episodes()
        
        # Load normalization statistics
        current_dir = Path(__file__).parent.parent  # Go up to data directory
        stat_path = current_dir / "utils" / "stat.json"
        self.action_min, self.action_max = load_normalization_stats(str(stat_path), 'ac_one')
        
        # Filter episodes if needed (same episodes for both training and validation)
        if self.max_episodes is not None and self.max_episodes > 0:
            self.episodes = self.episodes[:min(self.max_episodes, len(self.episodes))]
        
        logger.info(f"AC-One Dataset initialized with {len(self.episodes)} episodes")
        logger.info(f"Task mode: {self.task_mode}")
        if self.task_mode == "single":
            logger.info(f"Single task: {self.task_name}")
        elif self.task_mode == "multi":
            if self.task_list:
                logger.info(f"Multi-task mode with specified tasks: {', '.join(self.task_list)}")
            else:
                logger.info(f"Multi-task mode: loading all available tasks")
        logger.info(f"Video size: {self.video_size}, Frames: {self.num_video_frames}")
        logger.info(f"Validation mode: {self.val}")
        
    def _is_task_folder(self, path: str) -> bool:
        """Check if a path is a valid task folder (contains videos/qpos/instructions)."""
        return all(os.path.exists(os.path.join(path, d)) for d in ["videos", "qpos", "instructions"])
    
    def _collect_task_paths(self) -> List[str]:
        """Collect all task paths based on task mode configuration.
        
        Returns:
            List of task relative paths (from dataset_dir)
        """
        def _find_task_folders(root_dir: str) -> List[str]:
            """Recursively find all directories under root_dir that contain the
            required subfolders: videos, qpos, instructions. Return paths
            relative to self.dataset_dir.
            """
            results: List[str] = []
            for current_root, dirnames, _ in os.walk(root_dir):
                # Skip backup-like directories
                base = os.path.basename(current_root)
                if base.endswith('_bak'):
                    continue

                # Check if current_root itself is a valid task folder
                if self._is_task_folder(current_root):
                    rel = os.path.relpath(current_root, self.dataset_dir)
                    results.append(rel)
                    # Do not descend further inside a valid task folder to avoid
                    # collecting nested duplicates
                    dirnames[:] = []
                    continue
            return results
        if self.task_mode == "single":
            # Single task: return the specified task path
            task_path = os.path.join(self.dataset_dir, self.task_name)
            if not os.path.exists(task_path):
                logger.error(f"Task '{self.task_name}' not found in {self.dataset_dir}")
                return []
            logger.info(f"Single task mode: loading '{self.task_name}'")
            return [self.task_name]
        
        # Multi task mode
        task_paths = []
        if self.task_list:
            # Load specified tasks (each can be a category or a concrete task folder)
            for task in self.task_list:
                task_full_path = os.path.join(self.dataset_dir, task)
                if not os.path.exists(task_full_path):
                    logger.warning(f"Task '{task}' not found, skipping")
                    continue

                if self._is_task_folder(task_full_path):
                    task_paths.append(task)
                else:
                    # Recursively search under this specified entry
                    found = _find_task_folders(task_full_path)
                    task_paths.extend(found)
            
            if not task_paths:
                logger.error(f"None of the specified tasks found")
                return []
            logger.info(f"Multi task mode: loading {len(task_paths)} tasks")
        else:
            # Recursively search the whole dataset directory
            found = _find_task_folders(str(self.dataset_dir))
            task_paths = found
            logger.info(f"Multi task mode: recursively found {len(task_paths)} task folders under dataset_dir")
        
        return task_paths
    
    def _load_task_episodes(self, task_relative_path: str) -> List[Dict[str, Any]]:
        """Load all episodes from a single task folder.
        
        Args:
            task_relative_path: Relative path from dataset_dir to the task folder
            
        Returns:
            List of episode dictionaries
        """
        episodes = []
        task_full_path = os.path.join(self.dataset_dir, task_relative_path)
        
        # Validate task folder structure
        if not self._is_task_folder(task_full_path):
            logger.warning(f"Invalid task folder structure: {task_relative_path}")
            return episodes
        
        # Get directory paths
        videos_dir = os.path.join(task_full_path, "videos")
        qpos_dir = os.path.join(task_full_path, "qpos")
        instructions_dir = os.path.join(task_full_path, "instructions")
        actual_task_name = os.path.basename(task_relative_path)

        # Discover language resources (embedding .pt and text .txt)
        # 1) Prefer instructions/*.pt; if none, fallback to umt5_wan/*.pt
        emb_base_dir: Optional[str] = None
        pt_files: List[str] = []
        for candidate_dir in [instructions_dir, os.path.join(task_full_path, "umt5_wan")]:
            if os.path.isdir(candidate_dir):
                found_pts = sorted([f for f in os.listdir(candidate_dir) if f.endswith('.pt')])
                if found_pts:
                    emb_base_dir = candidate_dir
                    pt_files = found_pts
                    break

        # Collect txt files only under instructions directory
        txt_files: List[str] = sorted([f for f in os.listdir(instructions_dir) if f.endswith('.txt')]) if os.path.isdir(instructions_dir) else []
        
        # Load all video episodes
        video_files = sorted([f for f in os.listdir(videos_dir) if f.endswith('.mp4')])
        
        # Determine mapping mode
        use_global_instruction: bool = (len(pt_files) == 1 and len(txt_files) == 1 and emb_base_dir is not None)
        global_lang_pt_path: Optional[str] = None
        global_txt_path: Optional[str] = None
        if use_global_instruction:
            global_lang_pt_path = os.path.join(emb_base_dir, pt_files[0])
            global_txt_path = os.path.join(instructions_dir, txt_files[0])

        for video_file in video_files:
            episode_id = os.path.splitext(video_file)[0]

            # Build common file paths
            video_path = os.path.join(videos_dir, video_file)
            qpos_path = os.path.join(qpos_dir, f"{episode_id}.pt")

            # Validate qpos exists
            if not os.path.exists(qpos_path):
                logger.warning(f"Missing qpos file: {qpos_path}")
                continue

            # Resolve language embedding and text path according to rules
            lang_path: Optional[str] = None
            text_path: Optional[str] = None

            if use_global_instruction:
                lang_path = global_lang_pt_path
                text_path = global_txt_path
            else:
                # Multi-file mode: align by episode_id (e.g., 1.mp4 -> 1.pt and 1.txt)
                if emb_base_dir is not None:
                    candidate_lang = os.path.join(emb_base_dir, f"{episode_id}.pt")
                    if os.path.exists(candidate_lang):
                        lang_path = candidate_lang

                candidate_txt = os.path.join(instructions_dir, f"{episode_id}.txt")
                if os.path.exists(candidate_txt):
                    text_path = candidate_txt

                # Backward compatibility fallback: actual_task_name.{pt,txt}
                if lang_path is None:
                    fallback_lang = os.path.join(instructions_dir, f"{actual_task_name}.pt")
                    if os.path.exists(fallback_lang):
                        lang_path = fallback_lang
                if text_path is None:
                    fallback_txt = os.path.join(instructions_dir, f"{actual_task_name}.txt")
                    if os.path.exists(fallback_txt):
                        text_path = fallback_txt

            # Validate language files
            if lang_path is None or not os.path.exists(lang_path):
                logger.warning(f"Missing language embedding for episode {episode_id} under {task_relative_path}")
                continue
            if text_path is None or not os.path.exists(text_path):
                logger.warning(f"Missing text instruction for episode {episode_id} under {task_relative_path}")
                continue

            episodes.append({
                'task_name': task_relative_path,
                'episode_name': f"episode_{episode_id}",
                'video_path': video_path,
                'qpos_path': qpos_path,
                'lang_path': lang_path,
                'text_path': text_path,
            })
        
        return episodes
    
    def _load_episodes(self) -> List[Dict[str, Any]]:
        """Load all episodes from the dataset directory."""
        # Collect all task paths
        task_paths = self._collect_task_paths()
        if not task_paths:
            return []
        
        # Load episodes from each task
        all_episodes = []
        for task_path in task_paths:
            episodes = self._load_task_episodes(task_path)
            all_episodes.extend(episodes)
        
        logger.info(f"Loaded {len(all_episodes)} episodes from {len(task_paths)} tasks")
        return all_episodes
    
    def _load_text_instruction(self, task_name: str, episode_name: str, text_path: Optional[str] = None) -> str:
        """Load text instruction for VLM processing."""
        try:
            # Preferred: use provided text_path if given
            if text_path is not None and os.path.exists(text_path):
                with open(text_path, 'r', encoding='utf-8') as f:
                    return f.read().strip()

            # Backward compatibility: dataset_dir/task_name/instructions/{actual_task_name}.txt
            actual_task_name = os.path.basename(task_name)
            instruction_path = self.dataset_dir / task_name / "instructions" / f"{actual_task_name}.txt"
            if instruction_path.exists():
                with open(instruction_path, 'r', encoding='utf-8') as f:
                    text_instruction = f.read().strip()
                return text_instruction

            raise FileNotFoundError(f"Text instruction file not found for task {task_name}")
                
        except Exception as e:
            logger.error(f"Error loading text instruction for {task_name}/{episode_name}: {e}")
            raise
    
    def __len__(self):
        """Return approximate dataset length."""
        return len(self.episodes) * 10000  # Assume 1000 samples per episode
    
    def __getitem__(self, idx):
        """
        Get a training sample.
        
        Args:
            idx: Sample index (not used, random sampling)
            
        Returns:
            Dictionary containing training data
        """
        # Random sampling like robotwin - ignore idx parameter
        if not self.episodes:
            return None
        episode = random.choice(self.episodes)
        
        try:
            # Load action data (qpos)
            action_data = torch.load(episode['qpos_path'], map_location='cpu').float()
            
            # Process language instruction
            # Load text instruction for VLM processing (use episode-specific text_path when available)
            language_instruction = self._load_text_instruction(episode['task_name'], episode['episode_name'], episode.get('text_path'))
            
            # Get video frame count efficiently without loading all frames
            total_frames = get_video_frame_count(episode['video_path'])
            
            if total_frames < 2:
                return None
            
            # Calculate sampling indices
            condition_frame_idx, video_indices, action_indices = self._calculate_sampling_indices(total_frames)
            
            # Load condition frame and video frames separately
            first_frame, original_frame = load_first_frame(episode['video_path'], condition_frame_idx, self.video_size)
            video_frames_sampled = load_video_frames(episode['video_path'], video_indices, self.video_size)
            initial_state, action_sequence = self._load_robot_data(action_data, action_indices, condition_frame_idx)
            language_embedding, instruction_idx = self._load_language_embedding(episode['lang_path'])
            
            # Complete VLM processing in dataset (following robotwin's approach)
            vlm_tokens = None
            if self.vlm_processor:
                first_frame_pil = tensor_to_pil(original_frame)
                vlm_tokens = preprocess_vlm_messages(language_instruction, first_frame_pil, self.vlm_processor)
            
            # Normalize actions and initial state (single normalization)
            normalized_actions = normalize_actions(action_sequence, self.action_min, self.action_max)
            normalized_initial_state = normalize_actions(initial_state.unsqueeze(0), self.action_min, self.action_max).squeeze(0)
            
            return {
                'first_frame': first_frame,             # [C, H, W] - condition frame
                'video_frames': video_frames_sampled,   # [num_video_frames, C, H, W] - target frames  
                'initial_state': normalized_initial_state,         # [action_dim] - normalized initial state
                'action_sequence': normalized_actions,     # [action_chunk_size, action_dim] - normalized actions
                'language_embedding': language_embedding, # [seq_len, dim] - for WAN
                'vlm_inputs': vlm_tokens,               # Complete VLM inputs ready for model
            }
            
        except Exception as e:
            logger.error(f"Error loading episode {idx} ({episode['episode_name']}): {e}")
            return None
    
    def _calculate_sampling_indices(self, total_frames: int) -> Tuple[int, List[int], List[int]]:
        """
        Calculate sampling indices for video and actions (following robotwin's logic).
        
        Args:
            total_frames: Total number of frames in the episode
            
        Returns:
            - condition_frame_idx: Index of condition frame (corresponds to initial state)
            - video_indices: List of video frame indices to predict
            - action_indices: List of action frame indices to predict
        """
        # Calculate physical span of one chunk
        physical_chunk_size = self.action_chunk_size * self.global_downsample_rate
        
        # Sample condition frame directly in physical space
        # Ensure the last action doesn't exceed total_frames - 1
        max_condition_idx = total_frames - physical_chunk_size - 1
        
        if max_condition_idx < 0:
            condition_frame_idx = 0
        else:
            condition_frame_idx = random.randint(0, max_condition_idx)
        
        # Action indices: from condition_frame_idx+1 onwards, with downsampling
        action_indices = []
        for i in range(self.action_chunk_size):
            # Each action is separated by global_downsample_rate frames
            action_idx = condition_frame_idx + (i + 1) * self.global_downsample_rate
            action_indices.append(min(action_idx, total_frames - 1))
        
        # Video indices: sample at frequency ratio intervals from action indices
        video_indices = []
        for i in range(self.num_video_frames):
            action_step = (i + 1) * self.video_action_freq_ratio - 1
            if action_step < len(action_indices):
                video_indices.append(action_indices[action_step])
            else:
                video_indices.append(action_indices[-1])
        
        return condition_frame_idx, video_indices, action_indices
    
    def _load_robot_data(self, action_data: torch.Tensor, action_indices: List[int], initial_state_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load robot position data (following robotwin's approach).
        
        Args:
            action_data: Full action data tensor [T, action_dim]
            action_indices: List of action frame indices
            initial_state_idx: Index for initial state (should match condition frame)
            
        Returns:
            - initial_state: Robot state at condition frame [state_dim]
            - action_sequence: Actions at specified indices [len(action_indices), action_dim]
        """
        # Get initial state at the condition frame index
        if initial_state_idx >= len(action_data):
            initial_state_idx = len(action_data) - 1
        initial_state = action_data[initial_state_idx].float()
        
        # Get action sequence at specified indices
        actions = []
        for idx in action_indices:
            if idx >= len(action_data):
                raise IndexError(f"Action index {idx} out of bounds for action data length {len(action_data)}")
            else:
                action = action_data[idx]
            actions.append(action)
        
        action_sequence = torch.stack(actions).float()
        
        return initial_state, action_sequence
    
    def _load_language_embedding(self, lang_path: str) -> tuple[torch.Tensor, int]:
        """Load pre-encoded language embedding and return the selected index."""
        try:
            embedding_data = torch.load(lang_path, map_location='cpu')
            
            if isinstance(embedding_data, list):
                selected_idx = random.randint(0, len(embedding_data) - 1)
                embeddings = embedding_data[selected_idx]  # [seq_len, dim]
            else:
                # If it's a single tensor
                embeddings = embedding_data
                selected_idx = 0
            
            # Remove batch dimension if present
            if embeddings.dim() == 3:
                embeddings = embeddings.squeeze(0)
            
            return embeddings, selected_idx
            
        except Exception as e:
            logger.error(f"Error loading language embedding from {lang_path}: {e}")
            raise