"""Download the CogACT2 RoboDojo checkpoint from Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_REPO_ID = "microsoft/CogACT-v2-RoboDojo"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "checkpoints" / "CogACT-v2-RoboDojo"
REQUIRED_FILES = ("weights.pt", "config.yaml", "action_feature.json", "norm_config.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default="main", help="Branch, tag, or full commit hash")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Show remote files and sizes without downloading")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        parser.exit(1, "Install the downloader dependency: python -m pip install huggingface_hub\n")

    output_dir = args.output_dir.expanduser().resolve()
    # Resolve a moving branch/tag once so all four files come from one commit.
    info = HfApi().model_info(args.repo_id, revision=args.revision, files_metadata=True)
    files = {entry.rfilename: entry for entry in info.siblings or []}
    missing = [name for name in REQUIRED_FILES if name not in files]
    if missing:
        parser.exit(1, f"Incomplete checkpoint repository: missing {', '.join(missing)}\n")
    if not info.sha:
        parser.exit(1, "Could not resolve the checkpoint commit.\n")

    print(f"[CogACT2] source: {args.repo_id}@{info.sha}", flush=True)
    print(f"[CogACT2] destination: {output_dir}", flush=True)
    for name in REQUIRED_FILES:
        size = files[name].size
        print(f"  {name}: {size:,} bytes" if size is not None else f"  {name}: size unknown")
    if args.dry_run:
        return

    snapshot_download(
        repo_id=args.repo_id,
        revision=info.sha,
        local_dir=output_dir,
        allow_patterns=list(REQUIRED_FILES),
    )
    for name in REQUIRED_FILES:
        path = output_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty checkpoint file: {path}")
        expected_size = files[name].size
        if expected_size is not None and path.stat().st_size != expected_size:
            raise RuntimeError(f"Incomplete checkpoint file: {path} (expected {expected_size} bytes)")

    print(f"[CogACT2] checkpoint ready; pass this directory as ckpt_name: {output_dir}")


if __name__ == "__main__":
    main()
