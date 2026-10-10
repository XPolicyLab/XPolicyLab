"""Validate prepared LeRobot v2 clean datasets and link them without changing data."""

import json
import runpy
import sys
from pathlib import Path


def prepare(model_root, data_root, destination):
    model_root, data_root, destination = map(Path, (model_root, data_root, destination))
    tasks = runpy.run_path(str(model_root / "starVLA/dataloader/gr00t_lerobot/mixtures.py"))["ROBOTWIN_CLEAN_TASKS"]
    with (model_root / "examples/Robotwin/train_files/modality.json").open(encoding="utf-8") as stream:
        template = json.load(stream)
    for task in tasks:
        dataset = data_root / "Clean" / task
        with (dataset / "meta/info.json").open(encoding="utf-8") as stream:
            info = json.load(stream)
        if info.get("codebase_version") not in ("v2.0", "v2.1"):
            raise ValueError(f"Expected prepared LeRobot v2.0/v2.1 data: {dataset}")
        with (dataset / "meta/modality.json").open(encoding="utf-8") as stream:
            modality = json.load(stream)
        for group, entries in template.items():
            for key, expected in entries.items():
                if modality.get(group, {}).get(key) != expected:
                    raise ValueError(f"Incompatible modality mapping {group}.{key}: {dataset}")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to replace existing data path: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(data_root.resolve(), target_is_directory=True)
    print(f"Linked {len(tasks)} clean tasks: {destination}")


if __name__ == "__main__":
    prepare(*sys.argv[1:])
