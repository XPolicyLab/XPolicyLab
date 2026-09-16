"""Prepare per-task Evo-1 configs from official LeRobot v2.1 exports.

Only metadata is copied. Source trajectories and videos remain read-only: the
upstream statistics writer operates on the copied meta/ directories.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

import yaml


CAMERA_KEYS = (
    "observation.images.cam_high", "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def prepare(output, datasets):
    output = Path(output).resolve()
    sources = {}
    for spec in datasets:
        task, separator, raw_path = spec.partition("=")
        if not separator or not re.fullmatch(r"[a-z0-9_]+", task):
            raise ValueError("Datasets must be task_name=/absolute/path/to/lerobot_v21.")
        if task in sources:
            raise ValueError(f"Duplicate task: {task}")
        source = Path(raw_path).expanduser().resolve()
        info = json.loads((source / "meta" / "info.json").read_text())
        if info.get("codebase_version") != "v2.1":
            raise ValueError(f"{source}: requires LeRobot v2.1 (one dataset per canonical task).")
        features = info["features"]
        for key in ("observation.state", "action"):
            if features.get(key, {}).get("shape") != [14]:
                raise ValueError(f"{source}: {key} must have 14 values in left/right joint order.")
        for key in CAMERA_KEYS:
            if features.get(key, {}).get("dtype") != "video":
                raise ValueError(f"{source}: expected official video feature {key}")
        for entry in ("data", "videos", "meta/tasks.jsonl"):
            if not (source / entry).exists():
                raise FileNotFoundError(source / entry)
        sources[task] = source
    if not sources:
        raise ValueError("Provide at least one task_name=dataset_path.")
    # Refuse reuse because upstream caches are keyed by the manifest name.
    output.mkdir(parents=True, exist_ok=False)
    groups = {}
    for task, source in sorted(sources.items()):
        key = f"robotwin_{task}"
        target = output / "datasets" / key
        shutil.copytree(source / "meta", target / "meta", ignore=shutil.ignore_patterns("stats.json"))
        for name in ("data", "videos"):
            (target / name).symlink_to(source / name, target_is_directory=True)
        groups[key] = {
            "path": str(target),
            "view_map": {f"image_{i + 1}": key for i, key in enumerate(CAMERA_KEYS)},
            "use_delta_action": False, "process_suite": "aloha_joint_angle",
            "suite_config": {"state_key": "observation.state", "action_key": "action",
                             "gripper_indices": [6, 13]},
        }
    identity = hashlib.sha256(str(output).encode()).hexdigest()[:12]
    config = {
        "max_action_dim": 24, "max_state_dim": 24, "max_views": 3,
        "normalization_type": "bounds", "datasets_manifest": f"robotwin_{identity}.pkl",
        "decoder_cache_size": 16, "cache_shard_size_bytes": 268435456,
        "data_groups": {"aloha_joint": groups},
    }
    config_path = output / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "sources.json").write_text(json.dumps({k: str(v) for k, v in sources.items()}, indent=2) + "\n")
    return config_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("datasets", nargs="+")
    args = parser.parse_args()
    print(prepare(args.output, args.datasets))
