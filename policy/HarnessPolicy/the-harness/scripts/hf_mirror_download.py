#!/usr/bin/env python3
"""Download a Hugging Face repo through hf-mirror.com with parallel ranges.

    python3 scripts/hf_mirror_download.py OpenWAM/OpenWAM-Alpha-Sim-RoboDojo /mnt/.../OpenWAM-Alpha-Sim-RoboDojo
    python3 scripts/hf_mirror_download.py TianxingChen/RoboTwin2.0 DEST --repo-type dataset --include "*.zip"

For GPU hosts that reach hf-mirror.com (and the Xet bridge it redirects to) but
not huggingface.co. A single connection runs at about 1.6 MB/s from the lab, so
large files are fetched as byte ranges by several curl processes, each resumed on
stalls, then joined and checked against the LFS sha256 from the repo listing.
Files that already match are skipped. Standard library and curl only.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.parse import quote
from urllib.request import Request, urlopen

MIRROR = "https://hf-mirror.com"


def list_files(repo, revision, repo_type="model"):
    # The mirror refuses Python's default User-Agent.
    request = Request(f"{MIRROR}/api/{repo_type}s/{repo}/tree/{revision}?recursive=true",
                      headers={"User-Agent": "curl/8"})
    with urlopen(request, timeout=120) as response:
        return [item for item in json.load(response) if item["type"] == "file"]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def curl(url, target, byte_range=None, size=None):
    """Fetch url (or a byte range of it) into target, resuming until it has `size` bytes."""
    for _attempt in range(200):
        have = target.stat().st_size if target.exists() else 0
        if size is not None and have >= size:
            return have == size
        command = ["curl", "-L", "--fail", "-sS", "--speed-limit", "51200", "--speed-time", "60"]
        if byte_range is not None:
            start, end = byte_range
            command += ["-r", f"{start + have}-{end}", "-o", "-"]
            with open(target, "ab") as handle:
                subprocess.run(command + [url], stdout=handle)
        else:
            subprocess.run(command + ["-C", "-", "-o", str(target), url])
            if size is None:
                return target.exists()
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo")
    parser.add_argument("dest", type=Path)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--repo-type", choices=["model", "dataset"], default="model")
    parser.add_argument("--include", action="append", default=[], help="glob of paths to fetch (repeatable)")
    parser.add_argument("--chunk-mb", type=int, default=512, help="range size for large files")
    parser.add_argument("--parallel", type=int, default=8, help="concurrent ranges")
    args = parser.parse_args()

    from fnmatch import fnmatch

    files = [f for f in list_files(args.repo, args.revision, args.repo_type)
             if not args.include or any(fnmatch(f["path"], pattern) for pattern in args.include)]
    print(f"{len(files)} files, {sum(f['size'] for f in files) / 1e9:.2f} GB", flush=True)
    failures = 0
    for item in files:
        path, size = item["path"], int(item["size"])
        expected = (item.get("lfs") or {}).get("oid")
        target = args.dest / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == size and (not expected or sha256(target) == expected):
            print(f"ok   {path}", flush=True)
            continue
        prefix = "datasets/" if args.repo_type == "dataset" else ""
        url = f"{MIRROR}/{prefix}{args.repo}/resolve/{args.revision}/{quote(path)}"
        chunk = args.chunk_mb << 20
        if size <= chunk:
            target.unlink(missing_ok=True)
            ok = curl(url, target, size=size)
        else:
            parts = target.with_name(target.name + ".parts")
            parts.mkdir(exist_ok=True)
            ranges = [(start, min(start + chunk, size) - 1) for start in range(0, size, chunk)]

            def fetch(index):
                start, end = ranges[index]
                return curl(url, parts / f"{index:05d}", (start, end), end - start + 1)

            with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
                done = list(pool.map(fetch, range(len(ranges))))
            ok = all(done)
            if ok:
                with open(target, "wb") as out:
                    for index in range(len(ranges)):
                        with open(parts / f"{index:05d}", "rb") as part:
                            shutil.copyfileobj(part, out, 1 << 24)
                shutil.rmtree(parts)
        if not ok or target.stat().st_size != size:
            print(f"FAIL {path} (download)", flush=True)
            failures += 1
        elif expected and sha256(target) != expected:
            print(f"FAIL {path} (sha256 mismatch)", flush=True)
            failures += 1
        else:
            print(f"done {path} {size}", flush=True)
    print("COMPLETE" if not failures else f"{failures} FAILED", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
