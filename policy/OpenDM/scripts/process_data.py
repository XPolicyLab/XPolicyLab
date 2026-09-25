"""Lossless LeRobot v2.1/v3.0 joint+gripper -> OpenDM video-backed JSONL.

Numeric rows and source videos are kept unchanged for all selected tasks except dlc.
No decoding, re-encoding, resampling, terminal-row insertion or train/val shuffle.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from XPolicyLab.utils.checkpoint_resolver import build_run_dir_name
from XPolicyLab.utils.process_data import get_robot_action_dim_info
from .recipe import CAMERAS, IMAGE_KEYS

POLICY_DIR = Path(__file__).resolve().parent.parent
# Official dlc instruction in both LeRobot exports. Resolve its index from
# metadata instead of assuming that task indices remain stable across releases.
DLC_TASK_PROMPT = 'Arrange the letters to spell "RoboDojo" in a row.'


def excluded_dlc_task_indices(task_prompts):
    def normalize(text):
        return " ".join(re.findall(r"\w+", str(text).casefold()))

    excluded = {"dlc", normalize(DLC_TASK_PROMPT)}
    return {index for index, prompt in task_prompts.items() if normalize(prompt) in excluded}


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def source_episodes(source, info):
    if info["codebase_version"] == "v2.1":
        return read_jsonl(source / "meta/episodes.jsonl")
    if info["codebase_version"] == "v3.0":
        return [
            row
            for path in sorted((source / "meta/episodes").rglob("*.parquet"))
            for row in pq.read_table(path).to_pylist()
        ]
    raise ValueError(f"Unsupported LeRobot version: {info['codebase_version']}")


def source_tasks(source, info):
    """Read original instructions keyed by the source dataset's task indices."""
    if info["codebase_version"] == "v2.1":
        return {
            int(task["task_index"]): task["task"]
            for task in read_jsonl(source / "meta/tasks.jsonl")
        }
    if info["codebase_version"] == "v3.0":
        # LeRobot stores instruction text in the pandas index, whose physical
        # Parquet column name is encoded in the table's pandas metadata.
        tasks = pq.read_table(source / "meta/tasks.parquet").to_pandas()
        return {int(index): prompt for prompt, index in tasks["task_index"].items()}
    raise ValueError(f"Unsupported LeRobot version: {info['codebase_version']}")


def source_paths(info, episode):
    index = episode["episode_index"]
    is_v3 = info["codebase_version"] == "v3.0"
    values = {"episode_index": index, "episode_chunk": index // info["chunks_size"]}
    if is_v3:
        values.update(
            chunk_index=episode["data/chunk_index"], file_index=episode["data/file_index"]
        )
    data_path = info["data_path"].format(**values)
    videos = []
    for camera in CAMERAS:
        key = "observation.images." + camera
        video_values = {**values, "video_key": key}
        offset = 0
        if is_v3:
            video_values.update(
                chunk_index=episode[f"videos/{key}/chunk_index"],
                file_index=episode[f"videos/{key}/file_index"],
            )
            offset = round(episode[f"videos/{key}/from_timestamp"] * info["fps"])
        videos.append((info["video_path"].format(**video_values), offset))
    return data_path, videos


def write_episode(rows, task, videos, target, action_dim):
    """Keep source state/action pairing at t; reference the corresponding video frame."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as stream:
        for frame_index, row in enumerate(rows):
            if row["frame_index"] != frame_index:
                raise ValueError(f"Non-contiguous frame_index in {target}")
            if len(row["observation.state"]) != action_dim or len(row["action"]) != action_dim:
                raise ValueError(f"Wrong robot dimension in {target}")
            record = {
                key: {"type": "video", "url": video, "frame_idx": offset + frame_index}
                for key, (video, offset) in zip(IMAGE_KEYS, videos, strict=True)
            }
            record.update(
                state=row["observation.state"],
                action=row["action"],
                prompt=task["prompt"],
                is_robot=True,
                task_name="robodojo_" + task["name"],
            )
            stream.write(json.dumps(record, allow_nan=False) + "\n")


def convert(source, output, *, env_cfg_type="arx_x5", source_revision=None, tasks=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    info_path = source / "meta/info.json"
    info = json.loads(info_path.read_text())
    assert info["fps"] == 25, "The released history recipe requires 25 Hz data"
    names = [f"{side}_joint_{i}" for side in ("left", "right") for i in range(7)]
    for key in ("observation.state", "action"):
        assert info["features"][key]["names"] == [names], "Use the joint+gripper LeRobot export"
    dims = get_robot_action_dim_info(env_cfg_type)
    action_dim = sum(dims["arm_dim"]) + sum(dims["ee_dim"])
    assert info["features"]["action"]["shape"] == [action_dim]
    task_prompts = source_tasks(source, info)
    excluded_indices = excluded_dlc_task_indices(task_prompts)
    selected_tasks = None if tasks is None else {int(index) for index in tasks}
    if selected_tasks is not None:
        if selected_tasks & excluded_indices:
            raise ValueError("The dlc subset is excluded and cannot be selected with --tasks")
        unknown = selected_tasks - task_prompts.keys()
        if unknown:
            raise ValueError(f"Unknown source task indices: {sorted(unknown)}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "jsonl").mkdir(exist_ok=True)
    media = output / "media"
    if not media.exists():
        media.symlink_to(source, target_is_directory=True)
    assert media.resolve() == source, "Output already points to a different source"
    episodes, index = [], {}
    excluded_episodes = 0
    cached_path, table = None, None
    for episode in source_episodes(source, info):
        relative_data, videos = source_paths(info, episode)
        if relative_data != cached_path:
            table = pq.read_table(source / relative_data)
            cached_path = relative_data
        episode_index = episode["episode_index"]
        selected = table.filter(pc.equal(table["episode_index"], episode_index))
        rows = selected.to_pylist()
        assert len(rows) == episode["length"], f"Episode length differs: {episode_index}"
        task_index = int(rows[0]["task_index"])
        assert all(row["task_index"] == task_index for row in rows), "Expected one task per episode"
        if task_index in excluded_indices:
            excluded_episodes += 1
            continue
        if selected_tasks is not None and task_index not in selected_tasks:
            continue
        task = {
            "task_index": task_index,
            "name": f"task_{task_index}",
            "prompt": task_prompts[task_index],
        }
        np.testing.assert_allclose(
            [r["timestamp"] for r in rows], np.arange(len(rows)) / info["fps"], atol=1e-4
        )
        for video, _ in videos:
            if not (source / video).is_file():
                raise FileNotFoundError(source / video)
        relative = Path(task["name"]) / f"episode_{episode_index:06d}.jsonl"
        target = output / "jsonl" / relative
        write_episode(rows, task, videos, target, action_dim)
        index[str(target)] = len(rows)
        episodes.append(
            {
                "task": task["name"],
                "task_index": task_index,
                "episode_index": episode_index,
                "frames": len(rows),
                "split": "train",
                "source_data": relative_data,
                "jsonl": str(relative),
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        )
    if not episodes:
        raise ValueError("No training episodes selected")
    # Remove obsolete generated episodes, including dlc from older conversions.
    # Otherwise rebuilding the upstream index cache could reintroduce them.
    for path in (output / "jsonl").rglob("episode_*.jsonl"):
        if str(path) not in index:
            path.unlink()
    counts = Counter(e["task"] for e in episodes)
    manifest = {
        "source_repo": "RoboDojo-Benchmark/RoboDojo",
        "source_revision": source_revision,
        "source_root": str(source),
        "source_version": info["codebase_version"],
        "source_info_sha256": hashlib.sha256(info_path.read_bytes()).hexdigest(),
        "fps": info["fps"],
        "env_cfg_type": env_cfg_type,
        "action_type": "joint",
        "robot_action_dim_info": dims,
        "action_dim": action_dim,
        "image_dir": str(media),
        "selected_task_indices": None if selected_tasks is None else sorted(selected_tasks),
        "excluded_tasks": ["dlc"],
        "excluded_task_indices": sorted(excluded_indices),
        "excluded_episodes": excluded_episodes,
        "tasks": {
            str(index): task_prompts[index]
            for index in sorted({e["task_index"] for e in episodes})
        },
        "task_episode_counts": dict(counts),
        "train_episodes": len(episodes),
        "validation_episodes": 0,
        "frames": sum(e["frames"] for e in episodes),
        "resampling": False,
    }
    (output / "dataset.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))
    (output / "jsonl/index_cache.json").write_text(json.dumps({"data": index}, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="LeRobot root containing meta/, data/, videos/"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bench-name", default="RoboDojo")
    parser.add_argument("--ckpt-name", default="robodojo-mem")
    parser.add_argument("--env-cfg-type", default="arx_x5")
    parser.add_argument("--action-type", default="joint")
    parser.add_argument("--source-revision", help="Optional source revision label for provenance only")
    parser.add_argument("--tasks", nargs="+", type=int, help="Source task indices to include; dlc is always excluded")
    args = parser.parse_args()
    assert (args.bench_name, args.env_cfg_type, args.action_type) == ("RoboDojo", "arx_x5", "joint")
    output = args.output or POLICY_DIR / "data" / build_run_dir_name(vars(args), include_seed=False)
    convert(
        args.source,
        output,
        env_cfg_type=args.env_cfg_type,
        source_revision=args.source_revision,
        tasks=args.tasks,
    )


if __name__ == "__main__":
    main()
