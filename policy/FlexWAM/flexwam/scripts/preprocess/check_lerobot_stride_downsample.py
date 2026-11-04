#!/usr/bin/env python3
"""Verify stride-downsampled LeRobot: parquet rows == video frames == expected len from source."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from downsample_lerobot_fps import expected_stride_length  # noqa: E402


def _video_frame_count(path: Path) -> int:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "csv=p=0",
            str(path),
        ],
        text=True,
    ).strip()
    return int(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True, help="Original 25 Hz (or pre-downsample) dataset")
    p.add_argument("--dst", type=Path, required=True, help="Stride-downsampled dataset")
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument(
        "--video-key",
        type=str,
        default="observation.images.cam_high",
        help="LeRobot video folder name under videos/chunk-000/",
    )
    args = p.parse_args()

    src_eps = [
        json.loads(line)
        for line in (args.src / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dst_eps = {
        json.loads(line)["episode_index"]: json.loads(line)
        for line in (args.dst / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    n_check = len(src_eps) if args.max_episodes is None else min(len(src_eps), args.max_episodes)

    mismatches: list[str] = []
    for ep in src_eps[:n_check]:
        idx = int(ep["episode_index"])
        src_len = int(ep["length"])
        exp = expected_stride_length(src_len, args.stride)
        if idx not in dst_eps:
            mismatches.append(f"ep {idx}: missing in dst episodes.jsonl")
            continue
        dst_len = int(dst_eps[idx]["length"])
        chunk = f"chunk-{idx // 1000:03d}"
        ep_name = f"episode_{idx:06d}"
        pq = args.dst / "data" / chunk / f"{ep_name}.parquet"
        vid = args.dst / "videos" / chunk / args.video_key / f"{ep_name}.mp4"

        pq_len = len(pd.read_parquet(pq))
        vid_len = _video_frame_count(vid) if vid.is_file() else -1

        if dst_len != exp:
            mismatches.append(f"ep {idx}: dst meta length {dst_len} != expected {exp} from src {src_len}")
        if pq_len != exp:
            mismatches.append(f"ep {idx}: parquet rows {pq_len} != expected {exp}")
        if vid_len != exp:
            mismatches.append(f"ep {idx}: video frames {vid_len} != expected {exp}")

    if mismatches:
        print(f"FAILED: {len(mismatches)} issue(s) in {n_check} episodes")
        for line in mismatches[:50]:
            print(line)
        if len(mismatches) > 50:
            print(f"... and {len(mismatches) - 50} more")
        raise SystemExit(1)

    print(
        f"OK: {n_check} episodes — dst length, parquet, and {args.video_key} "
        f"all match expected_stride_length(src_len, stride={args.stride})"
    )


if __name__ == "__main__":
    main()
