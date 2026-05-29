"""
Overlay sub-task instructions from a Mem_0 language_annotation.json onto the
preview MP4s next to the dataset, for sanity-checking VLM segmentation.

Reads:  <Mem_0>/language_annotations/<dataset>/<task>/<env_cfg>/language_annotation.json
        data/<dataset>/<task>/<env_cfg>/preview_video/episode_<i:07d>_<cam>.mp4
Writes: <Mem_0>/language_annotations/<dataset>/<task>/<env_cfg>/preview_video_annotated/episode_<i:07d>_<cam>_annotated.mp4

Annotation format (as produced by segment_language_annotation.py):
    {"episode_<i>": [[subtask_text, duration_in_frames], ...]}
Frame f belongs to segment k where sum(durations[0..k-1]) <= f < sum(durations[0..k]).

Example:
    python visualize_annotation.py RoboDojo swap_T arx_x5
    python visualize_annotation.py RoboDojo swap_T arx_x5 --episodes 0 --cameras cam_head,cam_third_view
"""

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ADAPTER_DIR = os.path.dirname(os.path.abspath(__file__))
UPSTREAM_DIR = os.path.dirname(ADAPTER_DIR)
ROOT_DIR = os.path.abspath(os.path.join(UPSTREAM_DIR, "..", "..", "..", ".."))
ANNOTATIONS_ROOT = os.path.join(UPSTREAM_DIR, "language_annotations")


def _frame_to_segment(frame_idx: int, segments):
    """Return (segment_index, segment_text) for frame_idx given [(text, duration), ...]."""
    cur = 0
    for i, (text, dur) in enumerate(segments):
        if frame_idx < cur + int(dur):
            return i, text
        cur += int(dur)
    return len(segments) - 1, segments[-1][0]  # past end -> last segment


def _segment_color(idx: int, total: int):
    """Distinct-ish color per segment (HSV cycle), returned as BGR tuple."""
    hue = int((idx / max(total, 1)) * 180) % 180
    bgr = cv2.cvtColor(np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _draw_overlay(frame, primary_text: str, status_line: str, progress: float, accent_bgr):
    """Draw a bottom info-bar with the sub-task text + a top-right status + a progress bar."""
    h, w = frame.shape[:2]
    pad = 8
    line_h = 22
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.55
    wrap_chars = max(36, w // 14)
    lines = textwrap.wrap(primary_text.strip() or "(empty)", width=wrap_chars) or [""]

    bar_h = pad * 2 + line_h * len(lines) + 6  # +6 for progress strip
    y0 = h - bar_h

    # Translucent backdrop
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    # Accent stripe on the left to make the segment color obvious
    cv2.rectangle(frame, (0, y0), (4, h), accent_bgr, -1)

    y = y0 + pad + line_h - 6
    for line in lines:
        cv2.putText(frame, line, (12, y), font, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (12, y), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        y += line_h

    # Top-right status badge
    (sw, sh), _ = cv2.getTextSize(status_line, font, font_scale, 1)
    cv2.rectangle(frame, (w - sw - 14, 4), (w - 4, 14 + sh), (0, 0, 0), -1)
    cv2.putText(frame, status_line, (w - sw - 9, 12 + sh), font, font_scale, accent_bgr, 1, cv2.LINE_AA)

    # Progress strip at the very bottom
    bar_y = h - 4
    cv2.rectangle(frame, (0, bar_y), (w, h), (40, 40, 40), -1)
    cv2.rectangle(frame, (0, bar_y), (int(w * max(0.0, min(1.0, progress))), h), accent_bgr, -1)
    return frame


def _visualize(video_path: Path, segments, out_path: Path,
               scale: float = 1.0, max_width: int = 0):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_ann = sum(int(d) for _, d in segments)
    if n_ann != n_video:
        tqdm.write(f"  [warn] {video_path.name}: video has {n_video} frames, annotation covers {n_ann}")

    eff_scale = float(scale) if scale and scale > 0 else 1.0
    if max_width and src_w * eff_scale > max_width:
        eff_scale = max_width / float(src_w)
    out_w = max(2, int(round(src_w * eff_scale)) // 2 * 2)
    out_h = max(2, int(round(src_h * eff_scale)) // 2 * 2)
    do_resize = (out_w, out_h) != (src_w, src_h)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open writer for {out_path}")

    n_total = max(n_video, 1)
    pbar = tqdm(total=n_video, desc=f"  {video_path.name}", leave=False, unit="f", dynamic_ncols=True)
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if do_resize:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        seg_idx, text = _frame_to_segment(frame_idx, segments)
        accent = _segment_color(seg_idx, len(segments))
        status = f"[{seg_idx + 1}/{len(segments)}]  frame {frame_idx}/{n_video - 1}"
        progress = (frame_idx + 1) / n_total
        _draw_overlay(frame, text, status, progress, accent)
        writer.write(frame)
        frame_idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    writer.release()
    return frame_idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay sub-task instructions onto preview MP4s.")
    parser.add_argument("dataset_name")
    parser.add_argument("task_name")
    parser.add_argument("env_cfg_type")
    parser.add_argument("--annotation", default=None,
                        help="path to language_annotation.json (default: "
                             "<Mem_0>/language_annotations/<dataset>/<task>/<env_cfg>/language_annotation.json)")
    parser.add_argument("--episodes", default=None,
                        help="comma-separated episode indices (default: all in annotation)")
    parser.add_argument("--cameras", default="cam_head",
                        help="comma-separated camera names (default: cam_head). "
                             "Use 'all' to overlay every preview MP4 found per episode.")
    parser.add_argument("--out_dir", default=None,
                        help="output dir (default: "
                             "<Mem_0>/language_annotations/<dataset>/<task>/<env_cfg>/preview_video_annotated)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="uniform downscale factor for the output video (e.g. 0.5 = half size). "
                             "Use together with or instead of --max_width.")
    parser.add_argument("--max_width", type=int, default=0,
                        help="cap output width in pixels while preserving aspect ratio (0 = no cap). "
                             "If both --scale and --max_width are given, the stricter one wins.")
    args = parser.parse_args()

    data_dir = Path(ROOT_DIR) / "final_data" / args.dataset_name / args.task_name / args.env_cfg_type
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Source data dir not found: {data_dir}")
    ann_root = Path(ANNOTATIONS_ROOT) / args.dataset_name / args.task_name / args.env_cfg_type

    ann_path = Path(args.annotation) if args.annotation else ann_root / "language_annotation.json"
    if not ann_path.is_file():
        raise FileNotFoundError(
            f"Annotation not found: {ann_path}\n"
            "Generate it first with: bash segment_data.sh <dataset> <task> <env_cfg> <expert_data_num>"
        )
    annotations = json.loads(ann_path.read_text(encoding="utf-8"))

    preview_dir = data_dir / "preview_video"
    if not preview_dir.is_dir():
        raise FileNotFoundError(f"preview_video dir not found: {preview_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else ann_root / "preview_video_annotated"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.episodes:
        ep_indices = [int(e) for e in args.episodes.split(",") if e.strip()]
    else:
        ep_indices = sorted(int(k.split("_")[-1]) for k in annotations
                            if k.startswith("episode_") and k.split("_")[-1].isdigit())

    cameras_arg = args.cameras.strip().lower()
    if cameras_arg == "all":
        cameras = None
    else:
        cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]

    written = 0
    for idx in tqdm(ep_indices, desc=f"viz {args.task_name}", unit="ep", dynamic_ncols=True):
        key = f"episode_{idx}"
        if key not in annotations:
            tqdm.write(f"[viz] skip {key}: no annotation")
            continue
        segments = annotations[key]
        if cameras is None:
            ep_videos = sorted(preview_dir.glob(f"episode_{idx:07d}_*.mp4"))
            cams_for_ep = [v.stem.split(f"episode_{idx:07d}_", 1)[1] for v in ep_videos]
        else:
            cams_for_ep = cameras
        for cam in cams_for_ep:
            video_path = preview_dir / f"episode_{idx:07d}_{cam}.mp4"
            if not video_path.is_file():
                tqdm.write(f"[viz] skip {video_path.name}: not found")
                continue
            out_path = out_dir / f"episode_{idx:07d}_{cam}_annotated.mp4"
            tqdm.write(f"[viz] {video_path.name} -> {out_path.name}  "
                       f"({len(segments)} segments, durations={[int(d) for _, d in segments]})")
            _visualize(video_path, segments, out_path,
                       scale=args.scale, max_width=args.max_width)
            written += 1

    if written == 0:
        raise RuntimeError("No videos visualized; check --episodes / --cameras / preview_video/.")
    tqdm.write(f"[viz] done -> {out_dir}  ({written} video(s))")


if __name__ == "__main__":
    main()
