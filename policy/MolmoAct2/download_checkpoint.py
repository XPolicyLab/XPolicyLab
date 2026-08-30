"""Download the pinned MolmoAct2-RoboDojo snapshot for offline evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_REPO_ID = "hqfang/MolmoAct2-RoboDojo"
DEFAULT_REVISION = "68964756dbfe5b455e6b4e4aa571199aa17d087c"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoints" / "MolmoAct2-RoboDojo",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=output_dir,
    )
    manifest = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "snapshot_path": str(Path(snapshot_path).resolve()),
    }
    manifest_path = output_dir / "xpolicylab_source.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[MolmoAct2] snapshot={snapshot_path}")
    print(f"[MolmoAct2] manifest={manifest_path}")


if __name__ == "__main__":
    main()
