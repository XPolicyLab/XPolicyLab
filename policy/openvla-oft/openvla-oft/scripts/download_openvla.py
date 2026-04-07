"""
Download the base OpenVLA model from Hugging Face into a fixed local directory.

Example:
    conda activate openvla
    python scripts/download_openvla.py
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_REPO_ID = "openvla/openvla-7b"
DEFAULT_LOCAL_DIR = "/mnt/pfs/pg4hw0/niantian/openvla_assets/models/openvla-7b"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_id",
        default=DEFAULT_REPO_ID,
        help="Hugging Face repo id for the base model.",
    )
    parser.add_argument(
        "--local_dir",
        default=DEFAULT_LOCAL_DIR,
        help="Directory to store the downloaded model snapshot.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    local_dir = Path(args.local_dir).expanduser().resolve()
    local_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading `{args.repo_id}` to `{local_dir}`")
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"Model ready at: {local_dir}")


if __name__ == "__main__":
    main()
