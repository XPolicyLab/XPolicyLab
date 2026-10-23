"""Acquire and verify the complete, fixed official Wan asset set used by RAP-WAM."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys


REPO_ID = "Wan-AI/Wan2.2-TI2V-5B"
REVISION = "921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
MANIFEST_PATH = Path(__file__).with_name("WAN_ASSETS.json")
INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"


def load_manifest():
    manifest = json.loads(MANIFEST_PATH.read_text())
    if (manifest.get("format"), manifest.get("repo_id"), manifest.get("revision")) != (
        "rap-wam-wan-assets-v1", REPO_ID, REVISION
    ):
        raise ValueError("The Wan manifest does not identify the pinned official release")
    names = set()
    for item in manifest["files"]:
        name, digest = item["path"], item["sha256"]
        relative = PurePosixPath(name)
        if (not name or relative.is_absolute() or ".." in relative.parts
                or str(relative) != name or "\\" in name or name in names
                or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
                or not isinstance(item["size_bytes"], int) or item["size_bytes"] <= 0):
            raise ValueError(f"Invalid Wan manifest entry: {name!r}")
        names.add(name)
    checksums = Path(__file__).with_name("WAN_SHA256SUMS").read_text()
    expected = "".join(f"{item['sha256']}  {item['path']}\n" for item in manifest["files"])
    if checksums != expected or len(names) != 11 or INDEX_NAME not in names:
        raise ValueError("Wan manifest and checksum list disagree or are incomplete")
    return manifest


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def asset_path(root, name):
    path = root / name
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Asset escapes its directory through a symlink: {path}")
    return path


def verify_file(root, item):
    path = asset_path(root, item["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != item["size_bytes"]:
        raise ValueError(f"Size mismatch: {path}")
    if sha256_file(path) != item["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {path}")


def verify_assets(root, manifest):
    root = Path(root).expanduser().resolve(strict=True)
    for item in manifest["files"]:
        verify_file(root, item)
        print(f"Verified {item['path']}", file=sys.stderr, flush=True)
    verify_index(root, manifest)
    return root


def verify_index(root, manifest):
    index = json.loads((root / INDEX_NAME).read_text())
    shards = {item["path"] for item in manifest["files"] if item["path"].endswith(".safetensors")}
    if not index.get("weight_map") or set(index["weight_map"].values()) != shards:
        raise ValueError("Wan transformer index does not reference exactly the pinned shards")


def work_path(path, work, label):
    result = Path(path).expanduser().resolve()
    if result == work or not result.is_relative_to(work):
        raise ValueError(f"{label} must be inside RAP_WAM_WORK_DIR: {result}")
    return result


def configure_workspace():
    raw = os.environ.get("RAP_WAM_WORK_DIR", "")
    if not raw or not Path(raw).is_absolute():
        raise ValueError("Set RAP_WAM_WORK_DIR to an absolute directory on shared storage")
    work = Path(raw).resolve()
    if work == Path("/"):
        raise ValueError("RAP_WAM_WORK_DIR must not be the filesystem root")
    paths = {
        "TMPDIR": work / "tmp", "TMP": work / "tmp", "TEMP": work / "tmp",
        "HF_HOME": work / "cache/hf", "HF_HUB_CACHE": work / "cache/hf/hub",
        "HF_XET_CACHE": work / "cache/hf/xet", "XDG_CACHE_HOME": work / "cache/xdg",
    }
    for name, default in paths.items():
        path = work_path(os.environ.get(name, str(default)), work, name)
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    return work


def acquire_assets(output, work, manifest, source=None):
    output = work_path(output, work, "Output")
    if output.exists():
        # Re-running a completed download is read-only and still verifies every byte.
        return verify_assets(output, manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = work_path(output.with_name(output.name + ".partial"), work, "Staging directory")
    lock_path = work_path(output.with_name(output.name + ".download.lock"), work, "Download lock")
    try:
        lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"Download lock exists: {lock_path}. Check for another download before removing a stale lock.") from exc
    try:
        with os.fdopen(lock_fd, "w") as lock:
            lock.write(f"pid={os.getpid()}\n")
        stage.mkdir(parents=True, exist_ok=True)
        work_path(stage / ".cache/huggingface/download", work, "Download metadata")
        if source:
            source = verify_assets(source, manifest)
            if source == stage or source == output:
                raise ValueError("Source and destination must be different directories")
        else:
            # Import only after shared-storage cache and temporary paths are configured.
            from huggingface_hub import hf_hub_download
        for item in manifest["files"]:
            destination = asset_path(stage, item["path"])
            if destination.exists():
                verify_file(stage, item)
                print(f"Verified existing {item['path']}", file=sys.stderr, flush=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source:
                shutil.copyfile(asset_path(source, item["path"]), destination)
            else:
                hf_hub_download(
                    repo_id=REPO_ID, revision=REVISION, filename=item["path"],
                    local_dir=str(stage), cache_dir=os.environ["HF_HUB_CACHE"],
                )
            verify_file(stage, item)
            print(f"Verified {item['path']}", file=sys.stderr, flush=True)
        verify_index(stage, manifest)
        # Publish a usable directory only after all files and the shard index pass.
        stage.rename(output)
        return output
    finally:
        lock_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="Copy only required assets from an existing local directory")
    parser.add_argument("--output", help="Destination inside RAP_WAM_WORK_DIR; defaults to assets/Wan2.2-TI2V-5B")
    parser.add_argument("--verify-only", action="store_true", help="Verify --source in place without copying or downloading")
    args = parser.parse_args()
    if args.verify_only and (not args.source or args.output):
        parser.error("--verify-only requires --source and does not accept --output")
    work = configure_workspace()
    manifest = load_manifest()
    if args.verify_only:
        output = verify_assets(args.source, manifest)
    else:
        output = acquire_assets(args.output or work / "assets/Wan2.2-TI2V-5B", work, manifest, args.source)
    print(output)


if __name__ == "__main__":
    main()
