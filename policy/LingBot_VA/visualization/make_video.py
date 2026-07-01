#!/usr/bin/env python3
"""Concatenate per-frame triple-view obs into a video: left=cam_left_wrist,
middle=cam_high, right=cam_right_wrist; frames ordered by file index then
in-file frame order."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def natural_key(p: Path) -> int:
    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else 0


def load_frames(obs_dir: Path) -> list[np.ndarray]:
    files = sorted(obs_dir.glob("obs_data_*.pt"), key=natural_key)
    if not files:
        raise SystemExit(f"no obs_data_*.pt under {obs_dir}")

    frames: list[np.ndarray] = []
    for f in files:
        data = torch.load(f, map_location="cpu", weights_only=False)
        for entry in data:
            left = np.asarray(entry["observation.images.cam_left_wrist"])
            mid = np.asarray(entry["observation.images.cam_high"])
            right = np.asarray(entry["observation.images.cam_right_wrist"])
            row = np.concatenate([left, mid, right], axis=1)
            frames.append(row)
        print(f"  {f.name}: {len(data)} frames", flush=True)
    return frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("obs_dir", type=Path, help="dir with obs_data_*.pt")
    ap.add_argument("-o", "--out", type=Path, default=None, help="output mp4 path")
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()

    out = args.out or args.obs_dir.parent / f"{args.obs_dir.name}.mp4"
    print(f"[make_video] src={args.obs_dir}")
    print(f"[make_video] out={out} fps={args.fps}")

    frames = load_frames(args.obs_dir)
    print(f"[make_video] total frames={len(frames)}, "
          f"frame shape={frames[0].shape}")

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, args.fps, (w, h))
    if not writer.isOpened():
        raise SystemExit(f"cv2.VideoWriter failed to open {out}")
    for fr in frames:
        writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"[make_video] done -> {out}")


if __name__ == "__main__":
    main()
