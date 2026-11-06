"""Download and verify a HumanWAM-Goal release from Hugging Face into this policy directory.

The Hugging Face repository and revision are pinned in ``release.json``. Release layout::

    <hf_path>/        inference checkpoint (transformer/, vae/, text_tokenizer/, processor.json,
                      sampling.json, robot.json, export.json); "." is the repository root
    goal_cache.tar    the released 10,000-case RoboTwin goal cache (extracts to goal_cache/)

Every checkpoint file is checked against ``export.json``, whose own SHA-256 is
pinned in ``release.json``; the goal cache is unpacked into ``assets/`` only after
its extracted manifest matches the pinned digest. Nothing is overwritten.

    python download_checkpoint.py --ckpt-name HumanWAM-Goal-RoboTwin
"""

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path


POLICY_DIR = Path(__file__).resolve().parent


def checksum(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def safe_path(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Archive paths must be relative")
    path = (root / relative).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("Archive path escapes its destination")
    return path


def verify_export(root):
    """Check an exported checkpoint against its export.json; return that file's SHA-256."""
    root = Path(root).resolve()
    record = json.loads((root / "export.json").read_text())
    if record.get("schema") != "humanwam_goal.export.v1" or not record.get("files"):
        raise ValueError("Expected a HumanWAM-Goal checkpoint export (export.json)")
    for name, expected in record["files"].items():
        if checksum(safe_path(root, name)) != expected:
            raise ValueError("Checkpoint file checksum mismatch: " + name)
    return checksum(root / "export.json")


def unpack(archive, destination, manifest_name, expected_sha256):
    """Extract regular files only, then require the pinned manifest digest."""
    destination = Path(destination).resolve()
    if destination.exists():
        if checksum(destination / manifest_name) != expected_sha256:
            raise ValueError(f"{destination} exists with a different {manifest_name}; remove it first")
        return destination
    stage = destination.with_name(destination.name + ".extracting")
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    try:
        with tarfile.open(archive, "r:*") as source:
            prefix = destination.name + "/"
            for member in source:
                name = member.name.rstrip("/")
                if name in (".", "", destination.name):
                    continue
                if not name.startswith(prefix):
                    raise ValueError(f"Archive member outside {prefix}: {member.name}")
                path = safe_path(stage, name[len(prefix) :])
                if member.isdir():
                    path.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with source.extractfile(member) as src, path.open("xb") as dst:
                        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
                else:
                    raise ValueError("Links and special files are not allowed in release archives")
        if checksum(stage / manifest_name) != expected_sha256:
            raise ValueError(f"{archive.name}: {manifest_name} digest differs from release.json")
        stage.rename(destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt-name", required=True, help="Checkpoint key in release.json")
    parser.add_argument("--release", type=Path, default=POLICY_DIR / "release.json")
    parser.add_argument(
        "--download-dir", type=Path, default=POLICY_DIR / "hf_release", help="Hugging Face local_dir (resumable)"
    )
    parser.add_argument("--skip-goal-cache", action="store_true")
    args = parser.parse_args()
    release = json.loads(args.release.read_text())
    hf = release["huggingface"]
    if not hf.get("repo_id") or not hf.get("revision"):
        parser.error("release.json does not pin a published Hugging Face repository and revision yet")
    checkpoint = release["checkpoints"].get(args.ckpt_name)
    if checkpoint is None:
        parser.error(f"Unknown checkpoint {args.ckpt_name!r}; release.json lists {sorted(release['checkpoints'])}")
    if not checkpoint.get("export_sha256"):
        parser.error("release.json does not pin the checkpoint's export.json digest yet")
    hf_path = checkpoint["hf_path"].strip("/")
    goal_file = release["goal_cache"]["hf_file"]
    if hf_path in ("", "."):
        # The checkpoint is the repository root: take everything except, if skipped, the cache.
        patterns, ignore = None, [goal_file] if args.skip_goal_cache else None
    else:
        patterns, ignore = [f"{hf_path}/**"] + ([] if args.skip_goal_cache else [goal_file]), None

    from huggingface_hub import snapshot_download

    root = Path(
        snapshot_download(
            repo_id=hf["repo_id"],
            repo_type=hf.get("repo_type", "model"),
            revision=hf["revision"],
            allow_patterns=patterns,
            ignore_patterns=ignore,
            local_dir=args.download_dir,
        )
    )
    exported = root / hf_path if hf_path not in ("", ".") else root
    if verify_export(exported) != checkpoint["export_sha256"]:
        raise ValueError("Downloaded checkpoint export differs from release.json")
    link = POLICY_DIR / checkpoint["path"]
    link.parent.mkdir(parents=True, exist_ok=True)
    if not link.exists():
        link.symlink_to(exported.resolve(), target_is_directory=True)
    elif link.resolve() != exported.resolve():
        raise ValueError(f"{link} already points elsewhere; remove it first")
    done = dict(checkpoint=str(link))
    if not args.skip_goal_cache:
        item = release["goal_cache"]
        done["goal_cache"] = str(
            unpack(root / item["hf_file"], POLICY_DIR / "assets" / item["path"], "manifest.json", item["manifest_sha256"])
        )
    print(json.dumps(dict(verified=True, **done), indent=2))


if __name__ == "__main__":
    main()
