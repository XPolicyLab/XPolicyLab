#!/usr/bin/env python3
"""
Download DreamZero DROID dataset from Hugging Face with rate-limit handling.

Use this script when `huggingface-cli download` hits 429 (Too Many Requests).
It uses a single worker and retries with backoff to stay within the 3000 req/5min limit.

Usage:
  python scripts/data/download_droid_hf.py [--local-dir ./data/droid_lerobot] [--max-workers 1]

  # Or with env (same as CLI default):
  DROID_DATA_ROOT=./data/droid_lerobot python scripts/data/download_droid_hf.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Install huggingface_hub: pip install huggingface_hub", file=sys.stderr)
    sys.exit(1)

REPO_ID = "GEAR-Dreams/DreamZero-DROID-Data"
REPO_TYPE = "dataset"


def main() -> None:
    p = argparse.ArgumentParser(description="Download DreamZero DROID dataset with rate-limit handling.")
    p.add_argument(
        "--local-dir",
        default=os.environ.get("DROID_DATA_ROOT", "./data/droid_lerobot"),
        help="Local directory to download into (default: DROID_DATA_ROOT or ./data/droid_lerobot)",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Concurrent download threads (default 1 to reduce API requests and avoid 429)",
    )
    p.add_argument(
        "--retry-wait",
        type=int,
        default=320,
        help="Seconds to wait on 429 before retry (default 320 ≈ 5min)",
    )
    args = p.parse_args()

    local_dir = os.path.abspath(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    attempt = 0
    while True:
        attempt += 1
        try:
            print(f"Download attempt {attempt} (max_workers={args.max_workers})...")
            snapshot_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                max_workers=args.max_workers,
                resume_download=True,
            )
            print(f"Done. Dataset at: {local_dir}")
            return
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "too many requests" in err_str or "rate limit" in err_str:
                print(f"Rate limited (429). Waiting {args.retry_wait}s before retry...", file=sys.stderr)
                time.sleep(args.retry_wait)
                continue
            raise


if __name__ == "__main__":
    main()
