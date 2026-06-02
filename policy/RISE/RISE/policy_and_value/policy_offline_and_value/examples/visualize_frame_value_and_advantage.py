#!/usr/bin/env python3
"""
Visualize per-episode frame_value and action_advantage overlaid on multi-camera videos.

Dataset layout (per episode):
- data/chunk-XXX/episode_000000.parquet              # contains frame_value, action_advantage
- videos/chunk-XXX/observation.images.hand_left/episode_000000.mp4
- videos/chunk-XXX/observation.images.hand_right/episode_000000.mp4
- videos/chunk-XXX/observation.images.top_head/episode_000000.mp4

Output: one MP4 per episode with a left-most plot panel (Value & Advantage over time)
followed by the available camera views for that frame.
"""

# python examples/visualize_frame_value_and_advantage.py --dataset_root

from __future__ import annotations

import argparse
import glob
import io
import os
from typing import Dict, List, Sequence

import cv2
import matplotlib
import numpy as np
import pyarrow.parquet as pq
import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_parquet_metrics(parquet_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return frame_value and action_advantage as float32 arrays."""
    table = pq.read_table(parquet_path)
    cols = table.column_names

    if "frame_value" not in cols or "action_advantage" not in cols:
        missing = [c for c in ("frame_value", "action_advantage") if c not in cols]
        raise ValueError(f"{parquet_path} missing columns: {missing}")

    frame_value = table["frame_value"].to_numpy(zero_copy_only=False).astype(np.float32)
    action_advantage = (
        table["action_advantage"].to_numpy(zero_copy_only=False).astype(np.float32)
    )
    
    
    # * clip action_advantage to [-1, 1]
    action_advantage = np.clip(action_advantage, -1.0, 1.0)
    return frame_value, action_advantage


def load_video_frames(video_path: str) -> tuple[List[np.ndarray], float]:
    """Load all frames (RGB) from a video. Returns frames and fps."""
    if not os.path.exists(video_path):
        return [], 30.0

    cap = cv2.VideoCapture(video_path)
    # fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    fps = 30
    
    frames: List[np.ndarray] = []

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    cap.release()
    return frames, fps


def make_overlay_frame(
    idx: int,
    frame_value: np.ndarray,
    action_advantage: np.ndarray,
    cam_frames: Sequence[np.ndarray],
    y_range: tuple[float, float],
    fig_w: float,
    fig_h: float,
    dpi: int,
) -> np.ndarray:
    """Render a single overlay frame as BGR image."""
    fig, axes = plt.subplots(1, 1 + len(cam_frames), figsize=(fig_w, fig_h), dpi=dpi)

    ax_plot = axes[0]
    x = np.arange(idx + 1)
    ax_plot.plot(x, frame_value[: idx + 1], linewidth=2, color="tab:blue", label="Value")
    ax_plot.plot(
        x, action_advantage[: idx + 1], linewidth=2, color="tab:orange", label="Advantage"
    )
    ax_plot.set_xlim(0, len(frame_value))
    ax_plot.set_ylim(*y_range)
    ax_plot.set_xlabel("Frame")
    ax_plot.set_ylabel("Predicted Value / Advantage")
    ax_plot.set_title("Value & Advantage Over Time")
    ax_plot.grid(True)
    ax_plot.legend(loc="upper left", fontsize="small")

    titles = ["Base Frame", "Wrist Left", "Wrist Right"]
    titles = titles[: len(cam_frames)]
    for ax_img, img, title in zip(axes[1:], cam_frames, titles):
        ax_img.imshow(img)
        ax_img.set_title(title)
        ax_img.axis("off")

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    png_bytes = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    plt.close(fig)

    frame_bgr = cv2.imdecode(png_bytes, cv2.IMREAD_COLOR)
    return frame_bgr


def write_episode_video(
    episode_id: int,
    frame_value: np.ndarray,
    action_advantage: np.ndarray,
    cam_views: Dict[str, List[np.ndarray]],
    fps: float,
    output_path: str,
    fig_w: float = 12.0,
    fig_h: float = 4.0,
    dpi: int = 100,
) -> None:
    """Generate overlay video for one episode."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if len(frame_value) == 0:
        return

    # Align lengths across metrics and cameras
    n_frames = len(frame_value)
    for frames in cam_views.values():
        n_frames = min(n_frames, len(frames))

    frame_value = frame_value[:n_frames]
    action_advantage = action_advantage[:n_frames]
    cam_names = sorted(cam_views.keys())

    y_min = min(frame_value.min(), action_advantage.min())
    y_max = max(frame_value.max(), action_advantage.max())
    if y_max - y_min < 1e-6:
        y_range = (-1.0, 1.0)
    else:
        pad = 0.1 * (y_max - y_min)
        y_range = (y_min - pad, y_max + pad)

    writer = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    for idx in tqdm.tqdm(range(n_frames), desc=f"Episode {episode_id:03d}", leave=False):
        views = [cam_views[name][idx] for name in cam_names]
        frame_bgr = make_overlay_frame(
            idx, frame_value, action_advantage, views, y_range, fig_w, fig_h, dpi
        )

        if writer is None:
            h, w = frame_bgr.shape[:2]
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        writer.write(frame_bgr)

    if writer is not None:
        writer.release()


def find_episode_parquets(data_root: str, chunk: str) -> List[str]:
    pattern = os.path.join(data_root, "data", chunk, "episode_*.parquet")
    out = sorted(glob.glob(pattern))
    
    return out


def build_camera_paths(videos_root: str, chunk: str, episode_stem: str) -> Dict[str, str]:
    base = os.path.join(videos_root, "videos", chunk)
    cams = {
        "hand_left": os.path.join(base, "observation.images.hand_left", f"{episode_stem}.mp4"),
        "hand_right": os.path.join(base, "observation.images.hand_right", f"{episode_stem}.mp4"),
        "top_head": os.path.join(base, "observation.images.top_head", f"{episode_stem}.mp4"),
    }
    return {k: v for k, v in cams.items() if os.path.exists(v)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay frame_value and action_advantage on episode videos."
    )
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Root of dataset (contains data/, videos/).",
    )
    parser.add_argument(
        "--chunk",
        default="chunk-000",
        help="Chunk name under data/ and videos/ (default: chunk-000).",
    )
    parser.add_argument(
        "--output_dir",
        default="./visualizations_labeled",
        help="Where to write overlay videos.",
    )
    parser.add_argument(
        "--fig_w", type=float, default=12.0, help="Figure width in inches (per frame)."
    )
    parser.add_argument(
        "--fig_h", type=float, default=4.0, help="Figure height in inches (per frame)."
    )
    parser.add_argument("--dpi", type=int, default=100, help="Figure DPI.")
    args = parser.parse_args()

    parquet_files = find_episode_parquets(args.dataset_root, args.chunk)
    if not parquet_files:
        raise FileNotFoundError("No parquet files found. Check --dataset_root/--chunk.")

    os.makedirs(args.output_dir, exist_ok=True)

    for parquet_path in tqdm.tqdm(parquet_files, desc="Episodes"):
        episode_stem = os.path.splitext(os.path.basename(parquet_path))[0]
        try:
            frame_value, action_advantage = load_parquet_metrics(parquet_path)
        except Exception as exc:  # noqa: BLE001
            tqdm.tqdm.write(f"Skipping {episode_stem}: {exc}")
            continue

        cam_paths = build_camera_paths(args.dataset_root, args.chunk, episode_stem)
        cam_frames: Dict[str, List[np.ndarray]] = {}
        fps_candidates = []

        for cam, path in cam_paths.items():
            frames, fps = load_video_frames(path)
            if frames:
                cam_frames[cam] = frames
                fps_candidates.append(fps)

        if not cam_frames:
            tqdm.tqdm.write(f"Skipping {episode_stem}: no camera videos found.")
            continue

        fps = fps_candidates[0] if fps_candidates else 30.0
        
        output_dir = os.path.join(args.output_dir, args.dataset_root.split('/')[-1])
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{episode_stem}_vis.mp4")

        write_episode_video(
            episode_id=int(episode_stem.split("_")[-1]),
            frame_value=frame_value,
            action_advantage=action_advantage,
            cam_views=cam_frames,
            fps=fps,
            output_path=output_path,
            fig_w=args.fig_w,
            fig_h=args.fig_h,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
