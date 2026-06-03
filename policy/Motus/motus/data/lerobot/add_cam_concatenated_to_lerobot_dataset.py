#!/usr/bin/env python3
"""
Add cam_concatenated video to a LeRobot dataset by stitching three camera views.

What it does
------------
- Checks if `observation.images.cam_concatenated` videos already exist
- If not, loads three camera videos (cam_high, cam_left_wrist, cam_right_wrist)
- Stitches them together following the logic in lerobot_dataset.py (lines 460-482)
- Saves the concatenated videos in the correct format
- Updates meta/info.json to include the cam_concatenated feature
- Computes video statistics and updates metadata

Usage example:
    python -m data.lerobot.add_cam_concatenated_to_lerobot_dataset \
      --repo_id fold_grey_t_shirt_neatly_using_hands \
      --root /share/home/lht/.cache/huggingface/lerobot/test_multi/fold_grey_t_shirt_neatly_using_hands \
      --overwrite false
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm

# Import LeRobot utilities
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.video_utils import decode_video_frames, encode_video_frames, get_video_info
from lerobot.datasets.compute_stats import estimate_num_samples, sample_indices


def _load_jsonlines(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _write_jsonlines_atomic(path: Path, rows: List[Dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _resolve_dataset_root(repo_id: str, root: Optional[str]) -> Path:
    """Resolve dataset root directory."""
    meta = LeRobotDatasetMetadata(repo_id=repo_id, root=root)
    return Path(meta.root)


def _get_chunk_number(episode_index: int, chunks_size: int = 1000) -> int:
    """Get chunk number for an episode index."""
    return episode_index // chunks_size


def _stitch_frames(
    cam_high: np.ndarray,  # [H, W, 3] uint8
    cam_left: np.ndarray,  # [H, W, 3] uint8
    cam_right: np.ndarray,  # [H, W, 3] uint8
) -> np.ndarray:
    """
    Stitch three camera views into a concatenated view.
    
    Logic (inverse of split):
    - Split: frame[:split_h, :] = high, frame[split_h:, :split_w] = left, frame[split_h:, split_w:] = right
    - Stitch: top = high, bottom-left = left, bottom-right = right
    
    Args:
        cam_high: High camera frame [H_high, W, 3] uint8
        cam_left: Left wrist camera frame [H_left, W_left, 3] uint8
        cam_right: Right wrist camera frame [H_right, W_right, 3] uint8
        
    Returns:
        Concatenated frame [H_total, W, 3] uint8
    """
    h_high, w_high = cam_high.shape[:2]
    h_left, w_left = cam_left.shape[:2]
    h_right, w_right = cam_right.shape[:2]
    
    # Use high camera width as target width
    target_w = w_high
    
    # According to original split logic: split_h = (H//3)*2
    # If total height is H, then:
    #   - top (high camera) should be 2/3 of total height = split_h
    #   - bottom (left+right) should be 1/3 of total height = H - split_h
    # Since top is 2/3 and bottom is 1/3, bottom_h = top_h / 2
    top_h = h_high
    bottom_h = top_h // 2  # Bottom region is half the height of top region
    
    # Split width: left and right cameras each take half the width
    split_w = target_w // 2
    right_w = target_w - split_w
    
    # Resize left and right cameras to fit bottom region (half width each)
    cam_left_resized = cv2.resize(cam_left, (split_w, bottom_h))
    cam_right_resized = cv2.resize(cam_right, (right_w, bottom_h))
    
    # Create output frame
    total_h = top_h + bottom_h
    out_frame = np.zeros((total_h, target_w, 3), dtype=np.uint8)
    
    # Place high camera on top
    out_frame[:top_h, :target_w] = cam_high
    
    # Place left camera on bottom-left
    out_frame[top_h:top_h + bottom_h, :split_w] = cam_left_resized
    
    # Place right camera on bottom-right
    out_frame[top_h:top_h + bottom_h, split_w:] = cam_right_resized
    
    return out_frame


def _process_episode(
    dataset_root: Path,
    episode_index: int,
    chunk_num: int,
    fps: float,
    overwrite: bool = False,
    episode_length: Optional[int] = None,
) -> bool:
    """
    Process a single episode: load three videos, stitch, and save concatenated video.
    
    Returns:
        True if video was created/updated, False if skipped
    """
    # Paths for input videos
    chunk_dir = dataset_root / "videos" / f"chunk-{chunk_num:03d}"
    
    cam_high_path = chunk_dir / "observation.images.cam_high" / f"episode_{episode_index:06d}.mp4"
    cam_left_path = chunk_dir / "observation.images.cam_left_wrist" / f"episode_{episode_index:06d}.mp4"
    cam_right_path = chunk_dir / "observation.images.cam_right_wrist" / f"episode_{episode_index:06d}.mp4"
    
    # Output path for concatenated video
    cam_concat_dir = chunk_dir / "observation.images.cam_concatenated"
    cam_concat_dir.mkdir(parents=True, exist_ok=True)
    cam_concat_path = cam_concat_dir / f"episode_{episode_index:06d}.mp4"
    
    # Skip if already exists and not overwriting
    if cam_concat_path.exists() and not overwrite:
        return False
    
    # Check if all three input videos exist
    if not cam_high_path.exists():
        print(f"Warning: {cam_high_path} does not exist, skipping episode {episode_index}")
        return False
    if not cam_left_path.exists():
        print(f"Warning: {cam_left_path} does not exist, skipping episode {episode_index}")
        return False
    if not cam_right_path.exists():
        print(f"Warning: {cam_right_path} does not exist, skipping episode {episode_index}")
        return False
    
    # Get frame count - prefer episode_length from metadata, fallback to video metadata
    num_frames = None
    if episode_length is not None and episode_length > 0:
        num_frames = episode_length
    else:
        # Try to get frame count from video metadata using av library
        try:
            import av
            
            def get_frame_count_from_video(video_path: Path) -> Optional[int]:
                """Get frame count from video using av library."""
                try:
                    with av.open(str(video_path)) as container:
                        stream = container.streams.video[0]
                        if stream.frames and stream.frames > 0:
                            return int(stream.frames)
                        # Fallback: calculate from duration
                        if stream.duration is not None and stream.time_base is not None:
                            duration_seconds = float(stream.duration * stream.time_base)
                            fps_estimate = float(stream.average_rate) if stream.average_rate else fps
                            estimated = int(duration_seconds * fps_estimate)
                            if estimated > 0:
                                return estimated
                except Exception:
                    pass
                return None
            
            high_frames = get_frame_count_from_video(cam_high_path)
            left_frames = get_frame_count_from_video(cam_left_path)
            right_frames = get_frame_count_from_video(cam_right_path)
            
            if high_frames and left_frames and right_frames:
                num_frames = min(high_frames, left_frames, right_frames)
        except Exception as e:
            print(f"Warning: Could not get frame count from video metadata for episode {episode_index}: {e}")
    
    if num_frames is None or num_frames == 0:
        print(f"Warning: Could not determine frame count for episode {episode_index}, skipping")
        return False
    
    # Create timestamps for all frames
    timestamps = [i / fps for i in range(num_frames)]
    
    # Decode frames from all three videos
    high_frames = decode_video_frames(cam_high_path, timestamps, tolerance_s=1.0 / fps, backend="pyav")
    left_frames = decode_video_frames(cam_left_path, timestamps, tolerance_s=1.0 / fps, backend="pyav")
    right_frames = decode_video_frames(cam_right_path, timestamps, tolerance_s=1.0 / fps, backend="pyav")
    
    # Convert to numpy uint8 [T, H, W, 3]
    high_frames_np = (high_frames.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8).clip(0, 255)
    left_frames_np = (left_frames.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8).clip(0, 255)
    right_frames_np = (right_frames.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8).clip(0, 255)
    
    # Stitch frames and save directly to avoid storing all in memory
    # This reduces memory usage by processing frames in batches
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Process and save frames (stitch + save immediately)
        for i in range(num_frames):
            stitched = _stitch_frames(high_frames_np[i], left_frames_np[i], right_frames_np[i])
            # Save immediately to reduce memory usage
            frame_pil = Image.fromarray(stitched, mode="RGB")
            frame_pil.save(tmp_path / f"frame_{i:06d}.png")
        
        # Encode video using LeRobot's encoder (after all frames are saved)
        # Use lower CRF for higher quality: 18-23 is high quality (default is 30)
        encode_video_frames(
            imgs_dir=tmp_path,
            video_path=cam_concat_path,
            fps=int(fps),
            vcodec="h264",  # Much faster than libsvtav1 (10-50x speedup)
            pix_fmt="yuv420p",
            crf=18,  # High quality (lower = better quality, larger file size)
            overwrite=True,
        )
    
    # Update episodes_stats.jsonl with statistics
    try:
        _update_episodes_stats(dataset_root, episode_index, cam_concat_path, fps, num_frames)
    except Exception as e:
        print(f"Warning: Failed to update stats for episode {episode_index}: {e}")
    
    return True


def _compute_video_stats(video_path: Path, fps: float, num_frames: int) -> Dict[str, Any]:
    """
    Compute statistics (min, max, mean, std) for a video.
    
    Returns stats in the format expected by episodes_stats.jsonl.
    The format should match compute_episode_stats: (C, 1, 1) for image/video features.
    Format: {"min": [[[R]], [[G]], [[B]]], "max": ..., "mean": ..., "std": ..., "count": [N]}
    """
    # Sample frames to reduce computation
    sampled_indices = sample_indices(num_frames)
    timestamps = [i / fps for i in sampled_indices]
    
    # Decode sampled frames
    frames = decode_video_frames(video_path, timestamps, tolerance_s=1.0 / fps, backend="pyav")
    # frames is [T, C, H, W] float in [0, 1] (channel_first format)
    
    # Convert to numpy, keeping channel_first format [T, C, H, W]
    frames_np = frames.cpu().numpy()  # [T, C, H, W]
    
    # Compute stats over (time, height, width) axes, keeping channel dim
    # Use axis=(0, 2, 3) to reduce over time, height, width, keeping channel
    # keepdims=True gives [1, C, 1, 1], then squeeze first dim to get [C, 1, 1]
    min_arr = np.squeeze(np.min(frames_np, axis=(0, 2, 3), keepdims=True), axis=0)  # [C, 1, 1]
    max_arr = np.squeeze(np.max(frames_np, axis=(0, 2, 3), keepdims=True), axis=0)  # [C, 1, 1]
    mean_arr = np.squeeze(np.mean(frames_np, axis=(0, 2, 3), keepdims=True), axis=0)  # [C, 1, 1]
    std_arr = np.squeeze(np.std(frames_np, axis=(0, 2, 3), keepdims=True), axis=0)  # [C, 1, 1]
    
    # Convert to nested list format: [[[R]], [[G]], [[B]]]
    # Shape [C, 1, 1] -> [[[R]], [[G]], [[B]]]
    def to_nested_list(arr):
        """Convert [C, 1, 1] array to [[[R]], [[G]], [[B]]] format."""
        return [[[float(arr[c, 0, 0])]] for c in range(arr.shape[0])]
    
    stats = {
        "min": to_nested_list(min_arr),
        "max": to_nested_list(max_arr),
        "mean": to_nested_list(mean_arr),
        "std": to_nested_list(std_arr),
        "count": [len(sampled_indices)],  # [N] format
    }
    
    return stats


def _update_meta_info(
    dataset_root: Path,
    concatenated_shape: Tuple[int, int, int],  # (C, H, W)
    fps: float,
) -> None:
    """Update meta/info.json to include cam_concatenated feature."""
    info_path = dataset_root / "meta" / "info.json"
    
    with open(info_path, "r") as f:
        info = json.load(f)
    
    # Add or update cam_concatenated feature
    features = info.get("features", {})
    concat_key = "observation.images.cam_concatenated"
    
    c, h, w = concatenated_shape
    features[concat_key] = {
        "dtype": "video",
        "shape": [float(h), w, c],  # Format: [height, width, channel] to match other video features
        "names": ["height", "width", "channel"],
        "info": {
            "video.height": h,
            "video.width": w,
            "video.codec": "h264",  # Changed from av1 to h264 for faster encoding
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": fps,
            "video.channels": c,
            "has_audio": False,
        },
    }
    
    # Update info.json
    info["features"] = features
    
    # Write back atomically
    tmp_path = info_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    tmp_path.replace(info_path)


def _update_episodes_stats(
    dataset_root: Path,
    episode_index: int,
    video_path: Path,
    fps: float,
    episode_length: int,
) -> None:
    """Update episodes_stats.jsonl with cam_concatenated statistics for a single episode."""
    stats_path = dataset_root / "meta" / "episodes_stats.jsonl"
    
    # Compute stats for cam_concatenated video
    concat_stats = _compute_video_stats(video_path, fps, episode_length)
    
    # Use file lock to ensure atomic updates in multi-process environment
    lock_path = stats_path.with_suffix(stats_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(lock_path, "w") as lock_file:
        try:
            # Acquire exclusive lock (blocking)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            
            # Load existing stats
            stats_list = _load_jsonlines(stats_path) if stats_path.exists() else []
            
            # Find or create episode stats entry
            episode_stats_entry = None
            for entry in stats_list:
                if entry.get("episode_index") == episode_index:
                    episode_stats_entry = entry
                    break
            
            if episode_stats_entry is None:
                # Create new entry if not found
                episode_stats_entry = {"episode_index": episode_index, "stats": {}}
                stats_list.append(episode_stats_entry)
                # Sort by episode_index
                stats_list.sort(key=lambda x: x.get("episode_index", 0))
            
            # Update stats entry
            if "stats" not in episode_stats_entry:
                episode_stats_entry["stats"] = {}
            episode_stats_entry["stats"]["observation.images.cam_concatenated"] = concat_stats
            
            # Write back atomically
            _write_jsonlines_atomic(stats_path, stats_list)
        finally:
            # Release lock
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _process_episode_wrapper(
    args_tuple: Tuple[int, str, str, int, float, bool, Optional[int]]
) -> Tuple[int, bool, Optional[Tuple[int, int, int]]]:
    """
    Wrapper function for processing a single episode in a worker process.
    
    Returns:
        (episode_index, success, concatenated_shape)
    """
    (
        episode_index,
        dataset_root_str,
        repo_id,
        chunk_num,
        fps,
        overwrite,
        episode_length,
    ) = args_tuple
    
    dataset_root = Path(dataset_root_str)
    try:
        success = _process_episode(
            dataset_root, episode_index, chunk_num, fps, overwrite, episode_length
        )
        
        # Get shape from created video if successful
        concatenated_shape = None
        if success:
            chunk_dir = dataset_root / "videos" / f"chunk-{chunk_num:03d}"
            cam_concat_path = chunk_dir / "observation.images.cam_concatenated" / f"episode_{episode_index:06d}.mp4"
            if cam_concat_path.exists():
                try:
                    concat_info = get_video_info(cam_concat_path)
                    concatenated_shape = (3, concat_info["height"], concat_info["width"])
                except Exception:
                    pass
        
        return (episode_index, success, concatenated_shape)
    except Exception as e:
        print(f"Error processing episode {episode_index}: {e}")
        return (episode_index, False, None)


def main():
    parser = argparse.ArgumentParser(
        description="Add cam_concatenated videos to a LeRobot dataset by stitching three camera views"
    )
    parser.add_argument("--repo_id", type=str, required=True, help="LeRobot dataset repo_id")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Local dataset root. If omitted, LeRobot default cache is used.",
    )
    parser.add_argument(
        "--overwrite",
        type=lambda x: x.lower() in ["true", "1", "yes"],
        default=False,
        help="Overwrite existing concatenated videos",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=0,
        help="Process at most N episodes (0 = all)",
    )
    parser.add_argument(
        "--chunks_size",
        type=int,
        default=1000,
        help="Episodes per chunk (default: 1000)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=25,
        help="Number of parallel worker processes (default: 4, 0 = sequential)",
    )
    args = parser.parse_args()

    dataset_root = _resolve_dataset_root(args.repo_id, args.root)
    print(f"Processing dataset: {dataset_root}")
    
    # Load metadata
    meta = LeRobotDatasetMetadata(repo_id=args.repo_id, root=str(dataset_root))
    total_episodes = meta.total_episodes
    fps = meta.fps
    
    # Load episodes.jsonl to get episode lengths
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    episodes_data = {}
    if episodes_path.exists():
        episodes_list = _load_jsonlines(episodes_path)
        episodes_data = {int(ep["episode_index"]): ep for ep in episodes_list}
    
    # Check if cam_concatenated already exists in features
    info_path = dataset_root / "meta" / "info.json"
    with open(info_path, "r") as f:
        info = json.load(f)
    
    features = info.get("features", {})
    has_concat = "observation.images.cam_concatenated" in features
    
    # Process episodes
    episodes_to_process = list(range(total_episodes))
    if args.max_episodes > 0:
        episodes_to_process = episodes_to_process[: args.max_episodes]
    
    processed = 0
    skipped = 0
    
    # Get shape from first episode (we'll use it to update meta)
    concatenated_shape = None
    
    # Prepare task arguments
    tasks = []
    for ep_idx in episodes_to_process:
        chunk_num = _get_chunk_number(ep_idx, args.chunks_size)
        
        # Check if concatenated video already exists
        chunk_dir = dataset_root / "videos" / f"chunk-{chunk_num:03d}"
        cam_concat_path = chunk_dir / "observation.images.cam_concatenated" / f"episode_{ep_idx:06d}.mp4"
        
        if cam_concat_path.exists() and not args.overwrite:
            skipped += 1
            # Get shape from existing video if not set
            if concatenated_shape is None:
                try:
                    concat_info = get_video_info(cam_concat_path)
                    concatenated_shape = (3, concat_info["height"], concat_info["width"])
                except Exception:
                    pass
            continue
        
        # Get episode length from metadata
        episode_length = None
        if ep_idx in episodes_data:
            episode_length = episodes_data[ep_idx].get("length", None)
        
        tasks.append(
            (
                ep_idx,
                str(dataset_root),
                args.repo_id,
                chunk_num,
                fps,
                args.overwrite,
                episode_length,
            )
        )
    
    # Process episodes (parallel or sequential)
    if args.num_workers > 0 and len(tasks) > 0:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            # Submit all tasks
            future_to_ep = {
                executor.submit(_process_episode_wrapper, task): task[0]
                for task in tasks
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(tasks), desc="Processing episodes") as pbar:
                for future in as_completed(future_to_ep):
                    ep_idx = future_to_ep[future]
                    try:
                        _, success, shape = future.result()
                        if success:
                            processed += 1
                            # Get shape from newly created video
                            if concatenated_shape is None and shape is not None:
                                concatenated_shape = shape
                        else:
                            skipped += 1
                    except Exception as e:
                        print(f"Error processing episode {ep_idx}: {e}")
                        skipped += 1
                    finally:
                        pbar.update(1)
    else:
        # Sequential processing (original logic)
        for task in tqdm(tasks, desc="Processing episodes"):
            ep_idx, dataset_root_str, _, chunk_num, fps, overwrite, episode_length = task
            try:
                success = _process_episode(
                    Path(dataset_root_str), ep_idx, chunk_num, fps, overwrite, episode_length
                )
                if success:
                    processed += 1
                    # Get shape from newly created video
                    if concatenated_shape is None:
                        chunk_dir = dataset_root / "videos" / f"chunk-{chunk_num:03d}"
                        cam_concat_path = chunk_dir / "observation.images.cam_concatenated" / f"episode_{ep_idx:06d}.mp4"
                        try:
                            concat_info = get_video_info(cam_concat_path)
                            concatenated_shape = (3, concat_info["height"], concat_info["width"])
                        except Exception:
                            pass
                else:
                    skipped += 1
            except Exception as e:
                print(f"Error processing episode {ep_idx}: {e}")
                skipped += 1
    
    print(f"\nDone. Processed: {processed}, Skipped: {skipped}")
    
    # If shape is still None, try to get it from any existing video as a last resort
    if concatenated_shape is None:
        print("Attempting to get shape from any existing concatenated video...")
        for ep_idx in episodes_to_process:
            chunk_num = _get_chunk_number(ep_idx, args.chunks_size)
            chunk_dir = dataset_root / "videos" / f"chunk-{chunk_num:03d}"
            cam_concat_path = chunk_dir / "observation.images.cam_concatenated" / f"episode_{ep_idx:06d}.mp4"
            if cam_concat_path.exists():
                try:
                    concat_info = get_video_info(cam_concat_path)
                    concatenated_shape = (3, concat_info["height"], concat_info["width"])
                    print(f"Successfully got shape from episode {ep_idx}: {concatenated_shape}")
                    break
                except Exception as e:
                    print(f"Warning: Could not get shape from {cam_concat_path}: {e}")
                    continue
    
    # If still None, infer from cam_high shape in meta/info.json
    if concatenated_shape is None:
        print("Attempting to infer shape from cam_high in meta/info.json...")
        try:
            cam_high_key = "observation.images.cam_high"
            if cam_high_key in features:
                cam_high_shape = features[cam_high_key].get("shape", [])
                if len(cam_high_shape) >= 3:
                    # shape format: [height, width, channel]
                    high_h, high_w, high_c = int(cam_high_shape[0]), int(cam_high_shape[1]), int(cam_high_shape[2])
                    # According to stitch logic: top_h = high_h, bottom_h = top_h // 2
                    # total_h = top_h + bottom_h = high_h + high_h // 2
                    total_h = high_h + (high_h // 2)
                    concatenated_shape = (high_c, total_h, high_w)
                    print(f"Inferred concatenated shape from cam_high [{high_h}, {high_w}, {high_c}]: {concatenated_shape}")
                else:
                    print(f"Warning: cam_high shape format is incorrect: {cam_high_shape}")
            else:
                print(f"Warning: {cam_high_key} not found in features")
        except Exception as e:
            print(f"Warning: Could not infer shape from meta/info.json: {e}")
    
    # Update meta/info.json with cam_concatenated feature
    if concatenated_shape is not None:
        if not has_concat:
            _update_meta_info(dataset_root, concatenated_shape, fps)
            print(f"Added observation.images.cam_concatenated feature to meta/info.json")
        else:
            # Update existing entry to ensure format is correct
            _update_meta_info(dataset_root, concatenated_shape, fps)
            print(f"Updated observation.images.cam_concatenated feature in meta/info.json")
    else:
        print(f"Warning: Could not determine concatenated video shape, skipping meta update")


if __name__ == "__main__":
    main()

