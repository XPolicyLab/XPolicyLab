"""Download the pinned public CP1000 artifact and encoder assets; no token required."""
from pathlib import Path
import argparse
import json
from assets import MANIFEST, POLICY_DIR, verify_bundle, verify_wan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=POLICY_DIR / "checkpoints/cp1000")
    parser.add_argument("--wan-root", type=Path, default=POLICY_DIR / "checkpoints/Wan2.1-I2V-14B-480P")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--metadata-only", action="store_true",
                        help="Download/check small checkpoint metadata; does not qualify tensor weights")
    args = parser.parse_args()
    if args.metadata_only and args.verify_only:
        parser.error("Choose metadata-only or verify-only")
    if not args.verify_only:
        from huggingface_hub import hf_hub_download
        for filename in MANIFEST["files"]:
            if args.metadata_only and filename in {"action.pt", "video.pt"}:
                continue
            hf_hub_download(MANIFEST["repo_id"], filename, revision=MANIFEST["revision"],
                            local_dir=args.bundle)
        if args.metadata_only:
            from assets import file_digest
            for name, expected in MANIFEST["files"].items():
                if name not in {"action.pt", "video.pt"}:
                    if file_digest(args.bundle / name) != expected["sha256"]:
                        raise ValueError(f"Metadata mismatch: {name}")
            print("Pinned metadata verified. Tensor weights and encoders NOT checked.")
            return
        prefix = "checkpoints/Wan2.1-I2V-14B-480P/"
        # Download just the encoders/tokenizer, never the original dense video model.
        for entry in MANIFEST["wan"]["files"]:
            source = hf_hub_download(MANIFEST["wan"]["repo"], entry["rfilename"],
                                     revision=MANIFEST["wan"]["revision"])
            import shutil
            destination = args.wan_root / entry["rfilename"].removeprefix(prefix)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copyfile(source, destination)
    print(json.dumps({"checkpoint": verify_bundle(args.bundle),
                      "encoders": verify_wan(args.wan_root)}, indent=2))


if __name__ == "__main__":
    main()
