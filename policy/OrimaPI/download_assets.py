"""Fetch pinned formal skill assets from the OrimaPI manifests and verify hashes."""

import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    source = Path(__file__).resolve().parent / "OrimaPI"
    asset_root = source / "checkpoint"
    manifests = sorted(
        (source / "orimapi/skills/code_skills").glob("*/vision_resources.json")
    )
    if not manifests:
        raise ValueError("No formal skill manifests found in the bundled OrimaPI source")

    assets = {}
    for manifest in manifests:
        for entry in json.loads(manifest.read_text(encoding="utf-8")).values():
            if assets.setdefault(entry["path"], entry) != entry:
                raise ValueError(f"Conflicting assets at {entry['path']}")
    for path, entry in assets.items():
        destination = asset_root / path
        snapshot_download(
            repo_id=entry["repository"], revision=entry["revision"],
            allow_patterns=list(entry["files"]), local_dir=destination,
        )
        for name, expected in entry["files"].items():
            with (destination / name).open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                    raise ValueError(f"Asset hash mismatch: {destination / name}")

    snapshot_download(
        repo_id="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        revision="8078d68a9c75a972131914f6afd0c1723be0da7f",
        allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"],
        local_dir=asset_root / "depth-anything-v2-metric-indoor-small",
    )


if __name__ == "__main__":
    main()
