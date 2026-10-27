#!/usr/bin/env python3
# Copyright (C) 2026 the Rex-M1-preview authors. Licensed under the Apache License, Version 2.0.
"""Precompute the head-camera thumbnails that JsonDataset reads as history frames.

For every converted episode JSON, decode its ego (head-camera) video once and store uint8
thumbnails of shape (num_kept, H, W, 3):

    <data_root>/thumbs/<chunk>/<episode>.ego<RES>s<STRIDE>.npy     # frames 0, STRIDE, 2*STRIDE, ...

Frames are resized with mibot.utils.io.resize_image (factor 32, max_pixels = RES*RES), so a stored
thumbnail is exactly what the processor receives with do_resize=False. RES and STRIDE must match
history_res and thumbs_stride in the data config.

Usage (from rex_m1/, so the JSON video paths resolve):
  python scripts/precompute_history_thumbs.py --data-root data/RoboDojo-all-arx_x5-ee \\
      [--res 400] [--stride 5] [--workers 8]
Existing outputs with the expected frame count are skipped.
"""
import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mibot.utils.io import resize_image  # noqa: E402


def process_episode(json_path, data_root, res, stride):
    from decord import VideoReader
    from PIL import Image

    with open(json_path) as f:
        traj = json.load(f)
    ego = traj["observations"]["ego"][0]
    n = int(traj["num_frames"])
    rel = os.path.relpath(json_path, os.path.join(data_root, "data"))
    out = os.path.join(data_root, "thumbs", rel.replace(".json", f".ego{res}s{stride}.npy"))
    idx = list(range(0, n, stride))
    if os.path.exists(out):
        try:
            if np.load(out, mmap_mode="r").shape[0] == len(idx):
                return "skip"
        except Exception:
            pass
    start = int(ego.get("start", 0))
    frames = VideoReader(ego["path"], num_threads=2).get_batch([start + i for i in idx]).asnumpy()
    bbox = ego.get("crop_bbox")
    if bbox is not None:
        top, left, height, width = (int(v) for v in bbox)
        frames = frames[:, top : top + height, left : left + width]
    thumbs = np.stack(
        [np.asarray(resize_image(Image.fromarray(fr), factor=32, max_pixels=res * res)) for fr in frames]
    ).astype(np.uint8)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp.npy"
    np.save(tmp, thumbs)
    os.replace(tmp, out)
    return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--res", type=int, default=400)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    jobs = sorted(glob.glob(os.path.join(a.data_root, "data", "chunk-*", "*.json")))
    if not jobs:
        sys.exit(f"no episode JSON under {a.data_root}/data")
    print(f"{len(jobs)} episodes, res={a.res} stride={a.stride} workers={a.workers}", flush=True)
    t0, counts = time.time(), {"done": 0, "skip": 0, "fail": 0}
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(process_episode, j, a.data_root, a.res, a.stride): j for j in jobs}
        for fut in as_completed(futs):
            try:
                counts[fut.result()] += 1
            except Exception as e:
                counts["fail"] += 1
                print("FAIL", futs[fut], repr(e)[:200], flush=True)
            if sum(counts.values()) % 100 == 0:
                print(f"  {sum(counts.values())}/{len(jobs)} {counts} {time.time() - t0:.0f}s", flush=True)
    print(f"finished: {counts} in {time.time() - t0:.0f}s", flush=True)
    sys.exit(1 if counts["fail"] else 0)


if __name__ == "__main__":
    main()
