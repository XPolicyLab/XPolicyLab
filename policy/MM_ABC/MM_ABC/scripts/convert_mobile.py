#!/usr/bin/env python
"""Convert XPolicyLab HDF5 episodes into per-task LeRobot v3.0 profiles.

Images use the shared RGB decoder and the same resize as runtime inference.
Completed episodes are tracked in .done/; reruns rebuild metadata and configs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO.parents[3]))

from mmabc.embodiments import mobile  # noqa: E402
from mmabc.paths import home_path, relative_to_home  # noqa: E402

DEFAULT_SRC = home_path("data/raw")
DEFAULT_OUT = home_path("data/mobile_mmabc")
VIDEO_KEY = "observation.images.{cam}"
FILES_PER_CHUNK = 1000


def _chunk_file(source_id: int) -> tuple[int, int]:
    return source_id // FILES_PER_CHUNK, source_id % FILES_PER_CHUNK


def _paths(profile: Path, source_id: int) -> dict[str, Path]:
    c, f = _chunk_file(source_id)
    out = {
        "data": profile / "data" / f"chunk-{c:03d}" / f"file-{f:03d}.parquet",
        "done": profile / ".done" / f"episode_{source_id:07d}.json",
    }
    for cam in mobile.CAMERAS.values():
        out[cam] = profile / "videos" / VIDEO_KEY.format(cam=cam) / f"chunk-{c:03d}" / f"file-{f:03d}.mp4"
    return out


def _decode_str(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode_str(value.item())
    return str(value)


def _partial_stats(x: np.ndarray) -> dict:
    return {
        "count": int(x.shape[0]),
        "sum": x.sum(0).tolist(),
        "sumsq": (x * x).sum(0).tolist(),
        "min": x.min(0).tolist(),
        "max": x.max(0).tolist(),
    }


def _encode_camera(colors, dest: Path, size: int, crf: int, gop: int) -> int:
    import av
    from XPolicyLab.utils.process_data import decode_image_bit

    from mmabc.data.video import resize_for_model

    tmp = dest.with_suffix(".mp4.tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(tmp), mode="w", format="mp4")
    stream = container.add_stream("libx264", rate=int(mobile.FPS))
    stream.width = size
    stream.height = size
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": str(crf), "g": str(gop), "bf": "0", "preset": "veryfast", "threads": "2"}
    n = colors.shape[0]
    block = 64
    written = 0
    for lo in range(0, n, block):
        rows = colors[lo : min(lo + block, n)]
        for raw in rows:
            rgb = decode_image_bit(raw)
            frame = av.VideoFrame.from_ndarray(resize_for_model(rgb, size), format="rgb24")
            frame.pts = written
            frame.time_base = Fraction(1, int(mobile.FPS))
            for packet in stream.encode(frame):
                container.mux(packet)
            written += 1
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    os.replace(tmp, dest)
    return written


def convert_episode(job: dict) -> dict:
    import h5py
    import pyarrow as pa
    import pyarrow.parquet as pq

    src = Path(job["src"])
    profile = Path(job["profile"])
    sid = int(job["source_id"])
    paths = _paths(profile, sid)
    t0 = time.time()
    with h5py.File(src, "r") as f:
        state = mobile.pack({k: f["state"][k][:] for k in f["state"]})
        action = mobile.pack({k: f["action"][k][:] for k in f["action"]})
        n = state.shape[0]
        if action.shape[0] != n or n < 2:
            raise ValueError(f"{src}: state/action length mismatch {state.shape} {action.shape}")
        if not (np.isfinite(state).all() and np.isfinite(action).all()):
            raise ValueError(f"{src}: non-finite low-dim values")
        instruction = _decode_str(f["instruction"][()]).strip()
        for cam in mobile.CAMERAS.values():
            colors = f[f"vision/{cam}/colors"]
            if colors.shape[0] != n:
                raise ValueError(f"{src}: {cam} has {colors.shape[0]} frames, low-dim has {n}")
            written = _encode_camera(colors, paths[cam], job["size"], job["crf"], job["gop"])
            if written != n:
                raise ValueError(f"{src}: wrote {written} frames for {cam}, expected {n}")

    st32, ac32 = state.astype(np.float32), action.astype(np.float32)
    table = pa.table(
        {
            "observation.state": pa.FixedSizeListArray.from_arrays(pa.array(st32.reshape(-1)), mobile.MODEL_DIM),
            "action": pa.FixedSizeListArray.from_arrays(pa.array(ac32.reshape(-1)), mobile.MODEL_DIM),
            "timestamp": pa.array(np.arange(n, dtype=np.float32) / np.float32(mobile.FPS)),
            "frame_index": pa.array(np.arange(n, dtype=np.int64)),
            "source_episode_index": pa.array(np.full(n, sid, dtype=np.int64)),
        }
    )
    paths["data"].parent.mkdir(parents=True, exist_ok=True)
    tmp = paths["data"].with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, row_group_size=256)
    os.replace(tmp, paths["data"])

    record = {
        "source_id": sid,
        "source_path": str(src),
        "source_mtime": src.stat().st_mtime,
        "length": int(n),
        "instruction": instruction,
        "stats": {"action": _partial_stats(action), "state": _partial_stats(state)},
        "seconds": round(time.time() - t0, 1),
    }
    paths["done"].parent.mkdir(parents=True, exist_ok=True)
    tmp = paths["done"].with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record))
    os.replace(tmp, paths["done"])
    return {"task": job["task"], "source_id": sid, "length": n, "seconds": record["seconds"]}


# metadata
def _merge_stats(parts: list[dict]) -> dict:
    count = sum(p["count"] for p in parts)
    s = np.sum([p["sum"] for p in parts], axis=0)
    ss = np.sum([p["sumsq"] for p in parts], axis=0)
    mean = s / count
    var = np.maximum(ss / count - mean**2, 0.0) * count / max(count - 1, 1)
    return {
        "count": [count] * len(mean),
        "mean": mean.tolist(),
        "std": np.sqrt(var).tolist(),
        "min": np.min([p["min"] for p in parts], axis=0).tolist(),
        "max": np.max([p["max"] for p in parts], axis=0).tolist(),
    }


def write_profile_meta(profile: Path, task: str, size: int) -> dict | None:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    records = [json.loads(p.read_text()) for p in sorted((profile / ".done").glob("episode_*.json"))]
    if not records:
        return None
    records.sort(key=lambda r: r["source_id"])
    instructions = sorted({r["instruction"] for r in records})
    task_index = {text: i for i, text in enumerate(instructions)}

    rows, cursor = [], 0
    for ep, r in enumerate(records):
        c, f = _chunk_file(r["source_id"])
        row = {
            "episode_index": ep,
            "tasks": [r["instruction"]],
            "length": r["length"],
            "dataset_from_index": cursor,
            "dataset_to_index": cursor + r["length"],
            "data/chunk_index": c,
            "data/file_index": f,
            "source_episode_index": r["source_id"],
        }
        for cam in mobile.CAMERAS.values():
            vk = VIDEO_KEY.format(cam=cam)
            row[f"videos/{vk}/chunk_index"] = c
            row[f"videos/{vk}/file_index"] = f
            row[f"videos/{vk}/from_timestamp"] = 0.0
            row[f"videos/{vk}/to_timestamp"] = r["length"] / mobile.FPS
        rows.append(row)
        cursor += r["length"]

    meta = profile / "meta"
    (meta / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False),
                   meta / "episodes" / "chunk-000" / "file-000.parquet")
    pq.write_table(pa.table({"task_index": list(task_index.values()), "task": list(task_index)}),
                   meta / "tasks.parquet")

    video_feature = {
        "dtype": "video",
        "shape": [size, size, 3],
        "names": ["height", "width", "channels"],
        "info": {"video.height": size, "video.width": size, "video.codec": "h264",
                 "video.pix_fmt": "yuv420p", "video.fps": int(mobile.FPS), "video.channels": 3},
    }
    info = {
        "codebase_version": "v3.0",
        "robot_type": mobile.ROBOT_TYPE,
        "task": task,
        "total_episodes": len(records),
        "total_frames": cursor,
        "total_tasks": len(instructions),
        "fps": int(mobile.FPS),
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            **{VIDEO_KEY.format(cam=cam): video_feature for cam in mobile.CAMERAS.values()},
            "observation.state": {"dtype": "float32", "shape": [mobile.MODEL_DIM]},
            "action": {"dtype": "float32", "shape": [mobile.MODEL_DIM]},
        },
        "layout": mobile.contract(),
    }
    (meta / "info.json").write_text(json.dumps(info, indent=1))
    (meta / "modality.json").write_text(json.dumps(
        {"video": {cam: {"original_key": VIDEO_KEY.format(cam=cam)} for cam in mobile.CAMERAS.values()}},
        indent=1,
    ))
    (meta / "embodiment.json").write_text(json.dumps(embodiment_json(), indent=1))
    stats = {
        "action": _merge_stats([r["stats"]["action"] for r in records]),
        "state": _merge_stats([r["stats"]["state"] for r in records]),
        "_partials": {"action": [r["stats"]["action"] for r in records],
                      "state": [r["stats"]["state"] for r in records]},
    }
    (meta / "stats.json").write_text(json.dumps(stats))
    return {"task": task, "episodes": len(records), "frames": cursor, "instructions": instructions,
            "stats": stats}


def embodiment_json() -> dict:
    return {
        "embodiment_tag": mobile.EMBODIMENT_TAG,
        "robot_type": mobile.ROBOT_TYPE,
        "record_frequency": mobile.FPS,
        "canonical": {
            "embodiment_tag": mobile.EMBODIMENT_TAG,
            "total_dim": mobile.MODEL_DIM,
            "action_valid_dims": list(range(mobile.MODEL_DIM)),
            "state_valid_dims": list(range(mobile.MODEL_DIM)),
        },
    }


def write_configs(out: Path, summaries: list[dict], weight_power: float) -> None:
    """Norm stats (exact, over every converted frame), per-task profiles, mixture."""
    from mmabc.canonical.normalize import NormStats

    parts_a = [p for s in summaries for p in s["stats"]["_partials"]["action"]]
    parts_s = [p for s in summaries for p in s["stats"]["_partials"]["state"]]
    stats = NormStats(
        embodiment_tag=mobile.EMBODIMENT_TAG,
        action={k: np.asarray(v) for k, v in _merge_stats(parts_a).items()},
        state={k: np.asarray(v) for k, v in _merge_stats(parts_s).items()},
    )
    stats.save(REPO / "configs" / "norm_stats" / f"{mobile.EMBODIMENT_TAG}.json")

    emb_dir = REPO / "configs" / "embodiments"
    emb_dir.mkdir(parents=True, exist_ok=True)
    views = "\n".join(f"  {slot}: {cam}" for slot, cam in mobile.CAMERAS.items())
    entries = []
    for s in sorted(summaries, key=lambda s: s["task"]):
        name = f"mobile_{s['task']}"
        (emb_dir / f"{name}.yaml").write_text(
            f"# Generated by scripts/convert_mobile.py ({s['episodes']} episodes, {s['frames']} frames).\n"
            f"name: {name}\n"
            f"path: {relative_to_home(out / s['task'])}\n"
            f"embodiment_tag: {mobile.EMBODIMENT_TAG}\n"
            f"robot_type: {mobile.ROBOT_TYPE}\n"
            f"fps: {mobile.FPS}\n"
            f"action_type: joint\n"
            f"available_action_types: [joint]\n"
            f"switch_prob: 0.0\n"
            f"reference_frame: base\n"
            f"hours: {s['frames'] / mobile.FPS / 3600:.3f}\n"
            f"views:\n{views}\n"
        )
        entries.append((name, s["frames"]))

    weights = np.asarray([f for _, f in entries], dtype=np.float64) ** weight_power
    weights = weights / weights.sum()
    lines = [
        "# Generated by scripts/convert_mobile.py: all converted Mobile tasks.",
        f"# weight ~ frames^{weight_power} (1.0 = uniform over frames, 0.0 = uniform over tasks).",
        "canonical: configs/canonical/canonical_mobile.yaml",
        "datasets:",
    ]
    lines += [f"  - {{config: configs/embodiments/{n}.yaml, weight: {w:.5f}}}" for (n, _), w in zip(entries, weights)]
    (REPO / "configs" / "mixture" / "mobile_all.yaml").write_text("\n".join(lines) + "\n")
    print(f"wrote norm stats, {len(entries)} profile configs and configs/mixture/mobile_all.yaml")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--episode-subdir", default="data",
                    help="directory below each task containing episode_*.hdf5")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--gop", type=int, default=8)
    ap.add_argument("--min-age", type=float, default=120.0,
                    help="skip source files modified less than this many seconds ago")
    ap.add_argument("--max-episodes", type=int, default=None, help="per task, for smoke tests")
    ap.add_argument("--weight-power", type=float, default=0.5)
    ap.add_argument("--configs-only", action="store_true")
    args = ap.parse_args()

    tasks = args.tasks or sorted(p.name for p in args.src.iterdir() if (p / args.episode_subdir).is_dir())
    args.out.mkdir(parents=True, exist_ok=True)

    jobs, skipped_fresh = [], 0
    now = time.time()
    for task in tasks:
        profile = args.out / task
        files = sorted((args.src / task / args.episode_subdir).glob("episode_*.hdf5"))
        if args.max_episodes:
            files = files[: args.max_episodes]
        for src in files:
            sid = int(src.stem.split("_")[-1])
            done = _paths(profile, sid)["done"]
            if done.exists():
                continue
            if now - src.stat().st_mtime < args.min_age:
                skipped_fresh += 1
                continue
            jobs.append({"task": task, "src": str(src), "profile": str(profile), "source_id": sid,
                         "size": args.image_size, "crf": args.crf, "gop": args.gop})

    if not args.configs_only and jobs:
        print(f"converting {len(jobs)} episodes with {args.workers} workers "
              f"({skipped_fresh} still downloading, skipped)", flush=True)
        t0, done_frames, failed = time.time(), 0, []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(convert_episode, j): j for j in jobs}
            for i, fut in enumerate(as_completed(futures), 1):
                job = futures[fut]
                try:
                    r = fut.result()
                    done_frames += r["length"]
                except Exception:
                    failed.append(job["src"])
                    print(f"FAILED {job['src']}\n{traceback.format_exc()}", flush=True)
                    continue
                if i % 20 == 0 or i == len(jobs):
                    el = time.time() - t0
                    print(f"  [{i}/{len(jobs)}] {done_frames} frames in {el:.0f}s "
                          f"({done_frames / max(el, 1e-6):.0f} frames/s)", flush=True)
        if failed:
            print(f"{len(failed)} episodes failed (will be retried on the next run)")

    summaries = []
    for task in tasks:
        s = write_profile_meta(args.out / task, task, args.image_size)
        if s:
            summaries.append(s)
            print(f"{task:32s} episodes={s['episodes']:4d} frames={s['frames']:7d} "
                  f"instructions={len(s['instructions'])}")
    if summaries:
        write_configs(args.out, summaries, args.weight_power)
    print(f"total: {sum(s['episodes'] for s in summaries)} episodes, "
          f"{sum(s['frames'] for s in summaries)} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
