#!/usr/bin/env python3
"""Downsample a LeRobot v2.1 dataset by integer stride (e.g. 25 Hz -> 12.5 Hz with stride=2).

Video frames use the same index rule as parquet: keep 0, stride, 2*stride, ...
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


def expected_stride_length(num_frames: int, stride: int) -> int:
    if num_frames <= 0:
        return 0
    return int(np.arange(0, num_frames, stride).size)


def _subsample_parquet(df: pd.DataFrame, stride: int) -> pd.DataFrame:
    t = len(df)
    if t == 0:
        return df
    state_idx = np.arange(0, t, stride, dtype=np.int64)
    action_src = np.minimum(state_idx + 1, t - 1)

    out = pd.DataFrame()
    out["observation.state"] = df["observation.state"].iloc[state_idx].to_numpy()
    out["action"] = df["action"].iloc[action_src].to_numpy()
    if "observation.ee_pose" in df.columns:
        out["observation.ee_pose"] = df["observation.ee_pose"].iloc[state_idx].to_numpy()
    if "action.ee_pose" in df.columns:
        out["action.ee_pose"] = df["action.ee_pose"].iloc[action_src].to_numpy()

    n = len(state_idx)
    out["frame_index"] = np.arange(n, dtype=np.int64)
    out["timestamp"] = np.zeros(n, dtype=np.float64)
    out["episode_index"] = df["episode_index"].iloc[0]
    out["task_index"] = df["task_index"].iloc[0]
    return out


def _downsample_video(
    src: Path,
    dst: Path,
    stride: int,
    target_fps: float,
    crf: int,
    cpu_used: int,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf = f"select='not(mod(n\\,{stride}))',setpts=N/{target_fps}/TB"
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vf",
        vf,
        "-r",
        str(target_fps),
        "-c:v",
        "libaom-av1",
        "-cpu-used",
        str(cpu_used),
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def _episode_stats(df: pd.DataFrame, episode_index: int) -> dict:
    stats: dict = {}
    for col in df.columns:
        if col in ("index",):
            continue
        arr = np.stack(df[col].to_numpy())
        if arr.ndim == 1:
            arr = arr[:, None]
        stats[col] = {
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "count": [int(arr.shape[0])],
        }
    return {"episode_index": int(episode_index), "stats": stats}


def _process_episode(
    episode_index: int,
    src_root: Path,
    dst_root: Path,
    stride: int,
    src_fps: float,
    video_keys: list[str],
    crf: int,
    cpu_used: int,
) -> tuple[int, int, dict]:
    chunk = f"chunk-{episode_index // 1000:03d}"
    ep_name = f"episode_{episode_index:06d}"
    src_parquet = src_root / "data" / chunk / f"{ep_name}.parquet"
    dst_parquet = dst_root / "data" / chunk / f"{ep_name}.parquet"
    dst_parquet.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(src_parquet)
    out = _subsample_parquet(df, stride)
    dst_fps = src_fps / stride
    out["timestamp"] = out["frame_index"].to_numpy(dtype=np.float64) / dst_fps
    out.to_parquet(dst_parquet, index=False)

    for vk in video_keys:
        rel = Path("videos") / chunk / vk / f"{ep_name}.mp4"
        _downsample_video(src_root / rel, dst_root / rel, stride, dst_fps, crf, cpu_used)

    ep_stat = _episode_stats(out, episode_index)
    return episode_index, len(out), ep_stat


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--crf", type=int, default=35)
    p.add_argument("--cpu-used", type=int, default=8)
    p.add_argument("--max-episodes", type=int, default=None)
    args = p.parse_args()

    src: Path = args.src
    dst: Path = args.dst
    if dst.exists():
        raise SystemExit(f"Refusing to overwrite existing dst: {dst}")

    info = json.loads((src / "meta" / "info.json").read_text())
    src_fps = float(info["fps"])
    dst_fps = src_fps / args.stride

    video_keys = [k for k in info.get("features", {}) if k.startswith("observation.images.")]
    n_ep = int(info["total_episodes"])
    if args.max_episodes is not None:
        n_ep = min(n_ep, args.max_episodes)

    dst.mkdir(parents=True)
    (dst / "meta").mkdir()
    (dst / "data").mkdir()
    (dst / "videos").mkdir()

    for name in ("tasks.jsonl",):
        sp = src / "meta" / name
        if sp.exists():
            shutil.copy2(sp, dst / "meta" / name)

    episodes_meta = []
    with open(src / "meta" / "episodes.jsonl", encoding="utf-8") as f:
        for line in f:
            episodes_meta.append(json.loads(line))
    episodes_meta = episodes_meta[:n_ep]

    new_episodes: list[dict] = []
    new_stats: list[dict] = []
    total_frames = 0

    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futures = {
            ex.submit(
                _process_episode,
                i,
                src,
                dst,
                args.stride,
                src_fps,
                video_keys,
                args.crf,
                args.cpu_used,
            ): i
            for i in range(n_ep)
        }
        done = 0
        for fut in as_completed(futures):
            ep_idx, length, ep_stat = fut.result()
            done += 1
            total_frames += length
            old = episodes_meta[ep_idx]
            new_episodes.append(
                {
                    "episode_index": ep_idx,
                    "tasks": old["tasks"],
                    "length": length,
                }
            )
            new_stats.append(ep_stat)
            if done % 20 == 0 or done == n_ep:
                print(f"progress {done}/{n_ep} episodes, last ep {ep_idx} len {length}")

    new_episodes.sort(key=lambda x: x["episode_index"])
    new_stats.sort(key=lambda x: x["episode_index"])

    with open(dst / "meta" / "episodes.jsonl", "w", encoding="utf-8") as f:
        for ep in new_episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    with open(dst / "meta" / "episodes_stats.jsonl", "w", encoding="utf-8") as f:
        for st in new_stats:
            f.write(json.dumps(st) + "\n")

    global_index = 0
    chunk_dir = dst / "data" / "chunk-000"
    for ep in new_episodes:
        ep_path = chunk_dir / f"episode_{ep['episode_index']:06d}.parquet"
        df = pd.read_parquet(ep_path)
        n = len(df)
        df["index"] = np.arange(global_index, global_index + n, dtype=np.int64)
        df.to_parquet(ep_path, index=False)
        global_index += n

    out_info = dict(info)
    out_info["fps"] = dst_fps
    out_info["total_frames"] = total_frames
    (dst / "meta" / "info.json").write_text(json.dumps(out_info, indent=4) + "\n", encoding="utf-8")
    print(f"done: {dst} fps={dst_fps} episodes={n_ep} frames={total_frames}")


if __name__ == "__main__":
    main()
