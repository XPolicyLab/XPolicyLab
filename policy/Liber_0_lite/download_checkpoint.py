"""Download and verify the pinned public checkpoint release."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="LiberAI/Liber0-Lite-Robodojo")
    parser.add_argument("--revision", default="270e1146e9f278ed1122c1ea614e5600dcf33ec3")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    args = parser.parse_args()
    for key in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        os.environ.pop(key, None)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    from huggingface_hub import snapshot_download
    files = ("model.pt", "config.yaml", "dataset_stats.json", "README.md")
    snapshot_download(repo_id=args.repo_id, revision=args.revision,
                      local_dir=str(args.destination), allow_patterns=[*files, "manifest.json", "SHA256SUMS"])
    manifest = json.loads((args.destination / "manifest.json").read_text())
    for name in files:
        digest = hashlib.sha256()
        with (args.destination / name).open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != manifest["sha256"][name]:
            raise ValueError(f"Checkpoint bundle checksum mismatch: {name}")
    print("Verified checkpoint bundle:", args.destination)
    snapshot_download(
        repo_id="HiDream-ai/HiDream-O1-Image",
        revision="0b0901d99f200389e138c61946af1185f5f49a13",
        local_dir=str(args.assets_dir),
        allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json"],
    )
    print("Downloaded model assets:", args.assets_dir)


if __name__ == "__main__":
    main()
