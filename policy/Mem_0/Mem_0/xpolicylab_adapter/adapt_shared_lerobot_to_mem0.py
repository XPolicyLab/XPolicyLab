"""
Fast-path adapter: copy a pre-built LeRobot v2.1 dataset (e.g.
xspark_shared/lerobot/RoboDojo_sim_v21_video_abot) into Mem_0 training format
WITHOUT re-encoding video.

The source dataset is never modified. Videos are symlinked
(observation.images.cam_high -> observation.image.head_camera). Parquet rows are
rewritten in-place at the destination:

- observation.state / action: 14-dim [LA(6),LGrip,RA(6),RGrip] ->
  16-dim Mem_0 layout [LA(6),pad,RA(6),pad,LGrip,RGrip]
- global_task: tasks.jsonl instruction for the episode's task_index
- subtask: same as global_task for M1; per-segment text for Mn (language_annotation)
- subtask_end / episode_id: RMBench-compatible flags

Usage:
    python adapt_shared_lerobot_to_mem0.py \\
        --source /path/to/RoboDojo_sim_v21_video_abot \\
        --dest   Mem_0/lerobot_datasets/RoboDojo-arx_5-100-joint \\
        [--annotation_root xpolicylab_adapter/language_annotation] \\
        [--task_config xpolicylab_adapter/task_config.json] \\
        [--hdf5_root data/RoboDojo_first100] \\
        [--workers 16]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

ADAPTER_DIR = Path(__file__).resolve().parent
UPSTREAM_DIR = ADAPTER_DIR.parent
ROOT_DIR = UPSTREAM_DIR.parents[3]
SUBTASK_END_WINDOW = 8
EPISODES_PER_TASK = 100

STATE_NAMES = [
    "left_joint_0", "left_joint_1", "left_joint_2", "left_joint_3", "left_joint_4",
    "left_joint_5", "left_joint_6", "right_joint_0", "right_joint_1", "right_joint_2",
    "right_joint_3", "right_joint_4", "right_joint_5", "right_joint_6",
    "left_gripper", "right_gripper",
]

MEM0_FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (16,), "names": STATE_NAMES},
    "action": {"dtype": "float32", "shape": (16,), "names": STATE_NAMES},
    "observation.image.head_camera": {
        "dtype": "video",
        "shape": (3, 480, 640),
        "names": ["channels", "height", "width"],
    },
    "subtask": {"dtype": "string", "shape": (1,), "names": ["subtask_annotation"]},
    "global_task": {"dtype": "string", "shape": (1,), "names": ["global_task_annotation"]},
    "subtask_end": {"dtype": "int32", "shape": (1,), "names": ["subtask_end_flag"]},
    "episode_id": {"dtype": "int32", "shape": (1,), "names": ["episode_id"]},
    "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
    "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
    "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
    "index": {"dtype": "int64", "shape": (1,), "names": None},
    "task_index": {"dtype": "int64", "shape": (1,), "names": None},
}


def _packed14_to_model16(arr14: np.ndarray) -> np.ndarray:
    out = np.zeros(16, dtype=np.float32)
    out[0:6] = arr14[0:6]
    out[7:13] = arr14[7:13]
    out[14] = arr14[6]
    out[15] = arr14[13]
    return out


def _segment_boundaries(episode_annotation, episode_length: int):
    boundaries, cur = [], 0
    for text, duration in episode_annotation:
        start = cur
        end = min(cur + int(duration) - 1, episode_length - 1)
        boundaries.append((start, end, text))
        cur = end + 1
        if cur >= episode_length:
            break
    if boundaries:
        s, _e, t = boundaries[-1]
        boundaries[-1] = (s, episode_length - 1, t)
    return boundaries


def _feature_stats(arr: np.ndarray) -> dict:
    return {
        "min": arr.min(axis=0),
        "max": arr.max(axis=0),
        "mean": arr.mean(axis=0),
        "std": arr.std(axis=0),
        "count": np.array([arr.shape[0]]),
    }


def build_task_index_map(
    tasks_jsonl: Path,
    task_config: Path,
    hdf5_root: Path,
    env_cfg_type: str,
) -> Dict[int, Dict[str, str]]:
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from XPolicyLab.utils.load_file import load_hdf5  # noqa: WPS433

    cfg = json.loads(task_config.read_text(encoding="utf-8"))
    mn_names = set(cfg.get("Mn") or [])
    all_names = (cfg.get("M1") or []) + (cfg.get("Mn") or [])
    inst2name = {}
    for name in all_names:
        hdf5 = hdf5_root / name / env_cfg_type / "data" / "episode_0000000.hdf5"
        if not hdf5.is_file():
            raise FileNotFoundError(f"Missing reference HDF5 for task mapping: {hdf5}")
        inst2name[load_hdf5(str(hdf5))["instruction"].strip()] = name

    mapping: Dict[int, Dict[str, str]] = {}
    for line in tasks_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        idx = int(row["task_index"])
        instruction = row["task"].strip()
        task_name = inst2name.get(instruction)
        if task_name is None:
            raise KeyError(f"No HDF5 task matches tasks.jsonl[{idx}]: {instruction[:80]!r}")
        mapping[idx] = {
            "task_name": task_name,
            "task_type": "Mn" if task_name in mn_names else "M1",
            "instruction": instruction,
        }
    return mapping


def load_mn_annotations(annotation_root: Path, mn_task_names: List[str]) -> Dict[str, dict]:
    out = {}
    for name in mn_task_names:
        path = annotation_root / name / "language_annotation.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing Mn annotation: {path}")
        out[name] = json.loads(path.read_text(encoding="utf-8"))
    return out


def _convert_episode_parquet(job: Tuple[str, str, dict, dict, dict]) -> Tuple[str, dict]:
    src_path, dst_path, task_idx_map, mn_annotations, src_video_stats = job
    df = pd.read_parquet(src_path)
    n = len(df)
    task_idx = int(df["task_index"].iloc[0])
    meta = task_idx_map[task_idx]
    global_task = meta["instruction"]
    episode_index = int(df["episode_index"].iloc[0])
    episode_id = episode_index % EPISODES_PER_TASK

    states16 = np.stack([_packed14_to_model16(np.asarray(v, dtype=np.float32)) for v in df["observation.state"]])
    actions16 = np.stack([_packed14_to_model16(np.asarray(v, dtype=np.float32)) for v in df["action"]])

    subtasks = np.empty(n, dtype=object)
    subtask_end = np.zeros(n, dtype=np.int32)
    episode_ids = np.full(n, episode_id, dtype=np.int32)
    global_tasks = np.full(n, global_task, dtype=object)

    if meta["task_type"] == "Mn":
        ann = mn_annotations[meta["task_name"]].get(f"episode_{episode_id}")
        if not ann:
            raise KeyError(
                f"Missing Mn annotation for {meta['task_name']} episode_{episode_id} "
                f"(lerobot episode_index={episode_index})"
            )
        boundaries = _segment_boundaries(ann, n)
        for i in range(n):
            subtasks[i] = global_task
            for start, end, text in boundaries:
                if start <= i <= end:
                    subtasks[i] = text
                    if (end - i) < SUBTASK_END_WINDOW:
                        subtask_end[i] = 1
                    break
    else:
        subtasks[:] = global_task
        tail = min(SUBTASK_END_WINDOW, n)
        subtask_end[n - tail:] = 1

    out = df.drop(columns=["observation.state", "action"]).copy()
    out["observation.state"] = list(states16)
    out["action"] = list(actions16)
    out["subtask"] = list(subtasks)
    out["global_task"] = list(global_tasks)
    out["subtask_end"] = list(subtask_end)
    out["episode_id"] = list(episode_ids)

    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dst_path, index=False)

    ep_stats = {
        "observation.state": _feature_stats(states16),
        "action": _feature_stats(actions16),
        "subtask_end": _feature_stats(subtask_end.astype(np.float32).reshape(-1, 1)),
        "episode_id": _feature_stats(episode_ids.astype(np.float32).reshape(-1, 1)),
    }
    if src_video_stats:
        ep_stats["observation.image.head_camera"] = src_video_stats

    return str(episode_index), ep_stats


def _as_scalar_count(count) -> int:
    if isinstance(count, np.ndarray):
        return int(count.item())
    if isinstance(count, (list, tuple)):
        return int(count[0])
    return int(count)


def _aggregate_stats(all_stats: List[dict]) -> dict:
    if not all_stats:
        return {}
    keys = all_stats[0].keys()
    agg = {}
    for key in keys:
        counts = sum(_as_scalar_count(s[key]["count"]) for s in all_stats)
        weighted_mean = sum(
            np.asarray(s[key]["mean"]) * _as_scalar_count(s[key]["count"]) for s in all_stats
        ) / counts
        agg[key] = {
            "min": np.min(np.stack([np.asarray(s[key]["min"]) for s in all_stats]), axis=0),
            "max": np.max(np.stack([np.asarray(s[key]["max"]) for s in all_stats]), axis=0),
            "mean": weighted_mean,
            "std": np.sqrt(
                sum(
                    (np.asarray(s[key]["std"]) ** 2 + (np.asarray(s[key]["mean"]) - weighted_mean) ** 2)
                    * _as_scalar_count(s[key]["count"])
                    for s in all_stats
                ) / counts
            ),
            "count": np.array([counts]),
        }
    return agg


def symlink_head_videos(source: Path, dest: Path) -> None:
    src_key = "observation.images.cam_high"
    dst_key = "observation.image.head_camera"
    src_root = source / "videos"
    dst_root = dest / "videos"
    for chunk in sorted(src_root.glob("chunk-*")):
        src_cam = chunk / src_key
        if not src_cam.is_dir():
            raise FileNotFoundError(f"Missing source camera dir: {src_cam}")
        dst_cam = dst_root / chunk.name / dst_key
        dst_cam.parent.mkdir(parents=True, exist_ok=True)
        if dst_cam.exists() or dst_cam.is_symlink():
            dst_cam.unlink()
        dst_cam.symlink_to(src_cam.resolve())


def write_info_json(source_info: dict, dest_meta: Path) -> None:
    info = json.loads(json.dumps(source_info))
    old_feats = info.pop("features")
    src_cam = old_feats.pop("observation.images.cam_high")
    info["features"] = {**MEM0_FEATURES}
    info["features"]["observation.image.head_camera"] = dict(src_cam)
    info["total_videos"] = info["total_episodes"]
    info["robot_type"] = "mem0_xpolicylab"
    dest_meta.mkdir(parents=True, exist_ok=True)
    (dest_meta / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt shared LeRobot dataset to Mem_0 format")
    parser.add_argument("--source", required=True, help="Read-only source LeRobot dataset root")
    parser.add_argument("--dest", required=True, help="Output Mem_0 LeRobot dataset root")
    parser.add_argument(
        "--annotation_root", default=str(ADAPTER_DIR / "language_annotation"),
        help="Mn language_annotation/<task>/language_annotation.json root",
    )
    parser.add_argument(
        "--task_config", default=str(ADAPTER_DIR / "task_config.json"),
        help="M1/Mn task name lists",
    )
    parser.add_argument(
        "--hdf5_root", default=str(ROOT_DIR / "data" / "RoboDojo_first100"),
        help="Used only to map tasks.jsonl instructions -> task names",
    )
    parser.add_argument("--env_cfg_type", default="arx_x5")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    args = parser.parse_args()

    source = Path(args.source).resolve()
    dest = Path(args.dest).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    task_idx_map = build_task_index_map(
        source / "meta" / "tasks.jsonl",
        Path(args.task_config),
        Path(args.hdf5_root),
        args.env_cfg_type,
    )
    mn_names = [m["task_name"] for m in task_idx_map.values() if m["task_type"] == "Mn"]
    mn_annotations = load_mn_annotations(Path(args.annotation_root), mn_names)

    src_stats_by_ep: Dict[str, dict] = {}
    stats_path = source / "meta" / "episodes_stats.jsonl"
    if stats_path.is_file():
        for line in stats_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            cam = row["stats"].get("observation.images.cam_high")
            if cam is not None:
                src_stats_by_ep[str(row["episode_index"])] = cam

    symlink_head_videos(source, dest)
    (dest / "meta").mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "meta" / "tasks.jsonl", dest / "meta" / "tasks.jsonl")
    shutil.copy2(source / "meta" / "episodes.jsonl", dest / "meta" / "episodes.jsonl")
    source_info = json.loads((source / "meta" / "info.json").read_text(encoding="utf-8"))
    write_info_json(source_info, dest / "meta")

    parquet_jobs = []
    for src_parquet in sorted(source.glob("data/chunk-*/episode_*.parquet")):
        rel = src_parquet.relative_to(source / "data")
        dst_parquet = dest / "data" / rel
        ep_idx = str(int(src_parquet.stem.split("_")[-1]))
        parquet_jobs.append(
            (
                str(src_parquet),
                str(dst_parquet),
                task_idx_map,
                mn_annotations,
                src_stats_by_ep.get(ep_idx),
            )
        )

    episodes_stats_out = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_convert_episode_parquet, job): job for job in parquet_jobs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="adapt parquet", unit="ep"):
            ep_idx, ep_stats = fut.result()
            episodes_stats_out.append({"episode_index": int(ep_idx), "stats": ep_stats})

    episodes_stats_out.sort(key=lambda x: x["episode_index"])
    with (dest / "meta" / "episodes_stats.jsonl").open("w", encoding="utf-8") as f:
        for row in episodes_stats_out:
            f.write(json.dumps(row, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x) + "\n")

    agg = _aggregate_stats([r["stats"] for r in episodes_stats_out])
    (dest / "meta" / "stats.json").write_text(
        json.dumps(agg, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x),
        encoding="utf-8",
    )

    print(f"[adapt] done -> {dest}")
    print(f"[adapt] episodes={len(episodes_stats_out)} Mn_tasks={mn_names}")


if __name__ == "__main__":
    main()
