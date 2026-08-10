#!/usr/bin/env python3
"""Download CSU-AI-0 from a public Hugging Face dataset and verify every file."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from verify_checkpoints import DEFAULT_MANIFEST, load_manifest, sha256_file, verify_checkpoint


POLICY_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCES = POLICY_DIR / "checkpoint_sources.json"
CHUNK_SIZE = 16 * 1024 * 1024


def load_source(path: Path) -> dict[str, object]:
    source = json.loads(path.read_text(encoding="utf-8"))["CSU-AI-0"]
    if source.get("provider") != "huggingface" or source.get("repo_type") != "dataset":
        raise ValueError("Only the Hugging Face dataset direct-download provider is supported")
    return source


def build_url(base_url: str, repo_id: str, revision: str, remote_path: str) -> str:
    quoted_repo = urllib.parse.quote(repo_id, safe="/")
    quoted_revision = urllib.parse.quote(revision, safe="")
    quoted_path = urllib.parse.quote(remote_path, safe="/")
    return (
        f"{base_url.rstrip('/')}/datasets/{quoted_repo}/resolve/"
        f"{quoted_revision}/{quoted_path}?download=true"
    )


def download_one(url: str, destination: Path, expected_sha256: str, retries: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")

    if destination.is_file() and sha256_file(destination) == expected_sha256:
        print(f"[cached] {destination}")
        return

    for attempt in range(1, retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "XPolicyLab-CSU-AI-0/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"

        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                resumed = offset > 0 and getattr(response, "status", 200) == 206
                mode = "ab" if resumed else "wb"
                if offset and not resumed:
                    offset = 0
                total = offset
                with partial.open(mode) as handle:
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        handle.write(chunk)
                        total += len(chunk)
                        print(f"[download] {destination.name}: {total / (1024 ** 3):.2f} GiB", flush=True)

            actual_sha256 = sha256_file(partial)
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    f"SHA-256 mismatch after download: expected {expected_sha256}, got {actual_sha256}"
                )
            os.replace(partial, destination)
            return
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            if attempt == retries:
                raise RuntimeError(f"Failed to download {url} after {retries} attempts") from exc
            wait_seconds = min(30, 2**attempt)
            print(f"[retry {attempt}/{retries}] {exc}; waiting {wait_seconds}s", flush=True)
            time.sleep(wait_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--repo-id")
    parser.add_argument("--revision")
    parser.add_argument("--base-url", default="https://huggingface.co")
    parser.add_argument("--target-dir", type=Path)
    parser.add_argument("--force", action="store_true", help="Back up an invalid existing checkpoint")
    parser.add_argument("--retries", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = load_source(args.sources)
    repo_id = args.repo_id or os.environ.get("CSU_AI_0_HUGGINGFACE_REPO") or str(source["repo_id"])
    revision = args.revision or str(source["revision"])
    source_subdir = str(source["source_subdir"]).strip("/")
    target_dir = (args.target_dir or POLICY_DIR / str(source["local_dir"])).resolve()

    if target_dir.exists():
        try:
            verify_checkpoint(target_dir, args.manifest)
            print(f"Checkpoint is already valid: {target_dir}")
            return
        except Exception:
            if not args.force:
                raise RuntimeError(
                    f"Existing checkpoint is invalid: {target_dir}. Re-run with --force to preserve it as a backup."
                )
            backup = target_dir.with_name(f"{target_dir.name}.invalid-{int(time.time())}")
            target_dir.rename(backup)
            print(f"Preserved invalid checkpoint as {backup}")

    stage_dir = target_dir.with_name(f".{target_dir.name}.download")
    stage_dir.mkdir(parents=True, exist_ok=True)
    entries = load_manifest(args.manifest)
    for entry in entries:
        remote_path = f"{source_subdir}/{entry.relative_path.as_posix()}"
        url = build_url(args.base_url, repo_id, revision, remote_path)
        download_one(url, stage_dir / entry.relative_path, entry.sha256, args.retries)

    verify_checkpoint(stage_dir, args.manifest)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(stage_dir, target_dir)
    print(f"Downloaded and verified CSU-AI-0 at {target_dir}")


if __name__ == "__main__":
    main()
