"""
Generate Mem_0 inference normalization stats from a LeRobot dataset.

Replaces the manual edit of dataset_min_max.py's __main__ (README inference
step 2). Run in the Mem_0 policy conda env:

    python gen_norm_stats.py --repo_id <lerobot_datasets/...> --task_name <task>

Writes Mem_0/assets/<task_name>/norm_stats.json (state/action min-max) and
global_instruction.txt, which deploy.yml's state_stats_path points at.
"""

import argparse
import os
import sys

ADAPTER_DIR = os.path.dirname(os.path.abspath(__file__))
UPSTREAM_DIR = os.path.dirname(ADAPTER_DIR)
if UPSTREAM_DIR not in sys.path:
    sys.path.insert(0, UPSTREAM_DIR)

from source.dataloader.dataset_min_max import LeRobot_Dataset, save_norm_stats  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Mem_0 norm_stats.json")
    parser.add_argument("--repo_id", required=True, help="path to the LeRobot dataset")
    parser.add_argument("--task_name", required=True, help="asset folder name under assets/")
    args = parser.parse_args()

    dataset = LeRobot_Dataset(
        repo_id=str(os.path.expanduser(args.repo_id)),
        features_to_load=[
            "observation.image.head_camera",
            "observation.state",
            "action",
            "subtask",
            "subtask_end",
            "episode_id",
        ],
    )
    sample = dataset[0]
    norm_path = save_norm_stats(dataset, args.task_name, sample["lang"])
    print(f"[norm] instruction: {sample['lang']}")
    print(f"[norm] saved norm stats -> {norm_path}")


if __name__ == "__main__":
    main()
