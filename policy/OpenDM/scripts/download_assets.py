"""Download the latest official model assets and LeRobot exports from main."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from .recipe import BASE_REVISION, POLICY_REVISION, SOURCE_REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", choices=["policy", "base", "norm", "data-v21", "data-v30"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    if args.asset.startswith("data-"):
        version = "v21" if args.asset == "data-v21" else "v30"
        snapshot_download(
            "RoboDojo-Benchmark/RoboDojo",
            repo_type="dataset",
            revision=SOURCE_REVISION,
            allow_patterns=[f"data/RoboDojo_lerobot_{version}_video/**"],
            local_dir=args.output or root / "data/source",
        )
    else:
        name = "DM05-MEM" if args.asset == "base" else "DM05-MEM-Robodojo-Sim"
        revision = BASE_REVISION if args.asset == "base" else POLICY_REVISION
        directory = "robodojo-norm" if args.asset == "norm" else name
        snapshot_download(
            "Dexmal/" + name,
            revision=revision,
            allow_patterns=["norm_stats.json"] if args.asset == "norm" else None,
            local_dir=args.output or root / "checkpoints" / directory,
        )


if __name__ == "__main__":
    main()
