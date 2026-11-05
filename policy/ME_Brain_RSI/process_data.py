"""Validate and link supervised LeRobot datasets without changing their targets."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def resolve_datasets(patterns: list[str]) -> list[Path]:
    """Resolve local paths, comma-separated lists, and shard globs."""
    datasets = []
    for pattern in patterns:
        for part in pattern.split(","):
            part = str(Path(part.strip()).expanduser())
            matches = sorted(glob.glob(part)) if glob.has_magic(part) else [part]
            if not matches:
                raise FileNotFoundError(f"No datasets match: {part}")
            for match in matches:
                path = Path(match).resolve()
                if path not in datasets:
                    datasets.append(path)
    if not datasets:
        raise ValueError("At least one local LeRobot dataset is required")
    return datasets


def validate_dataset(path: Path) -> dict:
    """Check metadata and required payload files before exposing a dataset."""
    info_path = path / "meta/info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Prepared LeRobot metadata not found: {info_path}")
    info = json.loads(info_path.read_text())
    version = info.get("codebase_version")
    if version not in ("v2.1", "v3.0"):
        raise ValueError(f"{path}: expected LeRobot v2.1 or v3.0, got {version!r}")
    if info.get("fps") != 25:
        raise ValueError(f"{path}: head history requires original 25 FPS data")
    if int(info.get("total_frames", 0)) < 1 or int(info.get("total_episodes", 0)) < 1:
        raise ValueError(f"{path}: dataset must contain frames and episodes")
    features = info.get("features", {})
    required = ["observation.state", "action", "episode_index", "timestamp"]
    required.extend(f"observation.images.{camera}" for camera in CAMERAS)
    missing = [key for key in required if key not in features]
    if missing:
        raise ValueError(f"{path}: missing LeRobot features: {missing}")
    shapes = {key: tuple(features[key].get("shape", ())) for key in features}
    if shapes["observation.state"] != (14,):
        raise ValueError(f"{path}: observation.state must have shape [14]")
    if shapes["action"] not in ((14,), (50, 14)):
        raise ValueError(f"{path}: action must have shape [14] or [50,14]")
    state_pair = next(
        ((state, mask) for state, mask in (("world_state", "world_state_mask"), ("s1", "s1_mask"))
         if state in features and mask in features),
        None,
    )
    if state_pair is None or not {"event_action", "event_action_mask"}.issubset(features):
        raise ValueError(
            f"{path}: missing aligned world-model supervision. Supply s1/s1_mask "
            "(or world_state/world_state_mask) and event_action/event_action_mask. "
            "Official RoboDojo converters do not generate these targets."
        )
    if shapes[state_pair[0]] != (270, 32) or shapes[state_pair[1]] != (270,):
        raise ValueError(f"{path}: world targets must be [270,32] and masks [270]")
    if shapes["event_action"] != (14,) or shapes["event_action_mask"] not in ((), (1,)):
        raise ValueError(f"{path}: event_action must be [14], event_action_mask scalar or [1]")
    if not any((path / "data").rglob("*.parquet")):
        raise FileNotFoundError(f"{path}: no dataset parquet files found under data/")
    task_file = path / ("meta/tasks.jsonl" if version == "v2.1" else "meta/tasks.parquet")
    if not task_file.is_file() and not {"task", "prompt", "prompt_pack"}.intersection(features):
        raise FileNotFoundError(f"{path}: task prompts require {task_file.name} or a prompt feature")
    for camera in CAMERAS:
        key = f"observation.images.{camera}"
        if features[key].get("dtype") != "video":
            raise ValueError(f"{path}: {key} must retain its original episode video")
        if not any(key in str(file.relative_to(path)) for file in (path / "videos").rglob("*.mp4")):
            raise FileNotFoundError(f"{path}: no episode video found for {key}")
    return {"path": str(path), "version": version, "frames": info["total_frames"]}


def prepare_datasets(datasets: list[Path], output: Path) -> dict:
    """Link validated shards without replacing existing data or renormalizing targets."""
    summaries = [validate_dataset(path) for path in datasets]
    if len({item["version"] for item in summaries}) != 1:
        raise ValueError("Do not mix LeRobot v2.1 and v3.0 in one training environment")
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists; choose another ckpt_name: {output}")
    if any(output == path or path in output.parents or output in path.parents for path in datasets):
        raise ValueError("Output and source datasets must be separate directories")
    output.mkdir(parents=True)
    shards = output / "shards"
    shards.mkdir()
    for index, source in enumerate(datasets):
        (shards / f"{index:04d}").symlink_to(source, target_is_directory=True)
    manifest = {"datasets": summaries, "dataset_pattern": str(shards / "*"), "fps": 25}
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    datasets = resolve_datasets(args.dataset)
    if args.check_only:
        summaries = [validate_dataset(path) for path in datasets]
        if len({item["version"] for item in summaries}) != 1:
            parser.error("Do not mix LeRobot v2.1 and v3.0")
        print(json.dumps(summaries, indent=2))
    else:
        if args.output_dir is None:
            parser.error("--output-dir is required unless --check-only is used")
        print(json.dumps(prepare_datasets(datasets, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
