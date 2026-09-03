#!/usr/bin/env python3
"""
Image processing utilities for dataset loading.
Common functions shared across different datasets (AC-One, RobotWin, ALOHA, etc.)
"""

import numpy as np
import cv2
import torch
from PIL import Image
from typing import Tuple, List
import random


def resize_with_padding(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image with aspect ratio preservation and padding to target size.
    
    This function ensures no image distortion by:
    1. Calculating the minimum scale ratio to fit the image within target size
    2. Resizing the image with this ratio to preserve aspect ratio
    3. Padding with black borders to reach exact target size
    4. Centering the resized image within the padded frame
    
    Args:
        frame: Input image [H, W, C]
        target_size: Target size (height, width)
        
    Returns:
        Processed image [target_height, target_width, C]
        
    Example:
        >>> frame = np.random.randint(0, 255, (720, 640, 3), dtype=np.uint8)
        >>> resized = resize_with_padding(frame, (384, 320))
        >>> print(resized.shape)  # (384, 320, 3)
    """
    target_height, target_width = target_size
    original_height, original_width = frame.shape[:2]
    
    # Calculate scaling ratio, use the smaller ratio to ensure image fits completely
    scale_height = target_height / original_height
    scale_width = target_width / original_width
    scale = min(scale_height, scale_width)
    
    # Calculate new dimensions after scaling
    new_height = int(original_height * scale)
    new_width = int(original_width * scale)
    
    # Resize with aspect ratio preservation
    resized_frame = cv2.resize(frame, (new_width, new_height))
    
    # Create black background with target size
    padded_frame = np.zeros((target_height, target_width, frame.shape[2]), dtype=frame.dtype)
    
    # Calculate center placement position
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    
    # Place resized image at center
    padded_frame[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_frame
    
    return padded_frame


def load_video_frames(video_path: str, frame_indices: List[int], target_size: Tuple[int, int] = None) -> torch.Tensor:
    """
    Load specific video frames efficiently with optional resizing.
    
    Args:
        video_path: Path to video file
        frame_indices: List of frame indices to load
        target_size: Optional target size (height, width) for resizing with padding
        
    Returns:
        Video tensor [T, C, H, W] in range [0, 1]
    """
    cap = cv2.VideoCapture(video_path)
    
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        
        for frame_idx in frame_indices:
            # Ensure frame index is valid
            if frame_idx >= total_frames:
                raise ValueError(f"Frame index {frame_idx} out of bounds for video {video_path} (total frames: {total_frames})")
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                raise ValueError(f"Failed to read frame {frame_idx} from {video_path}")
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize if target size is specified
            if target_size is not None:
                if frame.shape[:2] != target_size:
                    frame = resize_with_padding(frame, target_size)
            
            frames.append(frame)
        
        # Convert to tensor [T, C, H, W] and normalize to [0, 1]
        video_tensor = torch.from_numpy(np.array(frames)).permute(0, 3, 1, 2).float() / 255.0
        
        return video_tensor
        
    finally:
        cap.release()


def get_video_frame_count(video_path: str) -> int:
    """Get total frame count of video efficiently without loading frames."""
    cap = cv2.VideoCapture(video_path)
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return total_frames
    finally:
        cap.release()


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert tensor [C, H, W] to PIL Image.
    
    Args:
        tensor: Input tensor in format [C, H, W]
        
    Returns:
        PIL Image in RGB mode
    """
    # Convert from [C, H, W] to [H, W, C] and to numpy
    if tensor.shape[0] == 3:  # RGB
        image_np = tensor.permute(1, 2, 0).numpy()
        # Convert from [0, 1] to [0, 255] if needed
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
        return Image.fromarray(image_np, mode='RGB')
    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")


def apply_image_augmentation(frame: np.ndarray, 
                           brightness_prob: float = 0.5,
                           brightness_range: Tuple[float, float] = (0.8, 1.2),
                           flip_prob: float = 0.3) -> np.ndarray:
    """
    Apply common image augmentations to a frame.
    
    Args:
        frame: Input image [H, W, C]
        brightness_prob: Probability of applying brightness adjustment
        brightness_range: Range of brightness factors (min, max)
        flip_prob: Probability of applying horizontal flip
        
    Returns:
        Augmented image [H, W, C]
    """
    # Random brightness adjustment
    if random.random() < brightness_prob:
        brightness_factor = random.uniform(*brightness_range)
        frame = np.clip(frame * brightness_factor, 0, 255)
    
    # Random horizontal flip
    if random.random() < flip_prob:
        frame = np.fliplr(frame)
    
    return frame