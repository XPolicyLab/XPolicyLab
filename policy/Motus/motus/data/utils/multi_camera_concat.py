#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Camera View Concatenation Utility

Simple utility for concatenating three camera views:
- Head camera: Keep original size
- Left/Right wrist cameras: Resize to half and stack vertically

"""
import cv2
import numpy as np
from typing import Optional, Tuple


def resize_and_concatenate_frames(
    self, 
    head_img: np.ndarray, 
    left_img: np.ndarray, 
    right_img: np.ndarray
) -> Optional[np.ndarray]:
    """
    Concatenate three camera views in T-shape layout:
    - Top: Head camera (keep original size, e.g., 480x640)
    - Bottom left: Left wrist camera (resize to half, e.g., 240x320)
    - Bottom right: Right wrist camera (resize to half, e.g., 240x320)
    Final output: 720x640 (height x width)
        
    Args:
        head_img: Head camera image (keep original size)
        left_img: Left wrist camera image (resize to half size)  
        right_img: Right wrist camera image (resize to half size)
            
    Returns:
        Concatenated image with T-shape layout
    """
    try:
        # Get original dimensions
        orig_h, orig_w = head_img.shape[:2]
            
        # Resize wrist cameras to half size
        half_h, half_w = orig_h // 2, orig_w // 2
        left_resized = cv2.resize(left_img, (half_w, half_h))
        right_resized = cv2.resize(right_img, (half_w, half_h))
            
        # Concatenate left and right wrist cameras horizontally for bottom row
        bottom_row = np.hstack([left_resized, right_resized])
            
        # Create final T-shape layout:
        # Top row: head camera (orig_h x orig_w)
        # Bottom row: combined wrist cameras (half_h x orig_w)
        combined = np.vstack([head_img, bottom_row])
            
        return combined
    except Exception as e:
        return None


def get_concatenated_dimensions(original_shape: Tuple[int, int]) -> Tuple[int, int]:
    """
    Calculate output dimensions for concatenated frame.
    
    Args:
        original_shape: (height, width) of original images
        
    Returns:
        (height, width) of concatenated result
    """
    h, w = original_shape
    # Final: (3w/2) × h
    return h, int(w * 1.5)


# Example usage
if __name__ == "__main__":
    # Create dummy test images
    h, w = 240, 320
    
    head_img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    left_img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)  
    right_img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    
    # Test concatenation
    result = resize_and_concatenate_frames(head_img, left_img, right_img)
    
    if result is not None:
        print(f"Original shape: {head_img.shape}")
        print(f"Concatenated shape: {result.shape}")
        print(f"Expected shape: {get_concatenated_dimensions((h, w))}")
        
        # Save test result (optional)
        # cv2.imwrite("test_concatenated.jpg", result)
    else:
        print("Concatenation failed")
