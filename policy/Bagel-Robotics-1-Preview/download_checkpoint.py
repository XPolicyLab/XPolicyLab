"""Download or verify the recovered Bagel-Robotics-1-Preview bundle and Wan encoder assets."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


POLICY_DIR = Path(__file__).resolve().parent
BUNDLE_FILES = ("action.pt", "video.pt", "dataset_stats.json", "manifest.json",
                "pruning_architecture.json", "recovery_metadata.json")
WAN_FILES = ("Wan2.1_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth",
             "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
             "google/umt5-xxl/tokenizer_config.json")
REMOVED_VIDEO_PARAMETERS = 6003498177
DEFAULT_REPO = "bageldotcom/Bagel-Robotics-1-Preview"
DEFAULT_REVISION = "315f50f5c56511d52a266af81fa26c7c0b196908"


def verify(bundle: Path, wan: Path) -> None:
    manifest = json.loads((bundle / "manifest.json").read_text())
    architecture = json.loads((bundle / "pruning_architecture.json").read_text())
    if (manifest.get("format") != "vpp2-robodojo-bundle-v1" or
            manifest.get("step") != 100000 or
            manifest.get("pruning", {}).get("removed_video_parameters") !=
            REMOVED_VIDEO_PARAMETERS or
            architecture.get("removed_video_parameters") != REMOVED_VIDEO_PARAMETERS or
            architecture.get("original_video_parameters") != 16394878784 or
            architecture.get("compact_video_parameters") !=
            16394878784 - REMOVED_VIDEO_PARAMETERS or
            len(architecture.get("layers", [])) != 40):
        raise ValueError("Bundle is not the recovered 6.003B Bagel-Robotics-1-Preview checkpoint")
    for filename in BUNDLE_FILES:
        path = bundle / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        if filename in ("action.pt", "video.pt", "dataset_stats.json") and \
                path.stat().st_size != manifest.get("files", {}).get(filename):
            raise ValueError(f"Bundle file size differs from manifest: {filename}")
    recovery = json.loads((bundle / "recovery_metadata.json").read_text())
    if (recovery.get("schema") != 2 or not recovery.get("best_epoch") or
            recovery.get("exact_video_save_reload") is not True or
            recovery.get("exact_action_checkpoint_copy") is not True):
        raise ValueError("Bundle has no verified video recovery export")
    for filename in WAN_FILES:
        if not (wan / filename).is_file():
            raise FileNotFoundError(wan / filename)
    print(f"Verified compact bundle: {bundle}")
    print(f"Verified Wan encoder assets: {wan}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.environ.get("BAGEL_ROBOTICS_1_HF_REPO", DEFAULT_REPO),
                        help="Published Hugging Face model repository for the compact bundle")
    parser.add_argument("--revision", default=os.environ.get("BAGEL_ROBOTICS_1_HF_REVISION"),
                        help="Optional immutable checkpoint commit")
    parser.add_argument("--bundle", type=Path,
                        default=POLICY_DIR / "checkpoints/bagel_robotics_1_preview")
    parser.add_argument("--wan-root", type=Path,
                        default=POLICY_DIR / "checkpoints/Wan2.1-I2V-14B-480P")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    bundle, wan = args.bundle.expanduser().resolve(), args.wan_root.expanduser().resolve()
    if not args.verify_only:
        if not args.repo or args.repo.startswith("YOUR_ORG/"):
            parser.error("Set --repo to the published Bagel-Robotics-1-Preview Hugging Face model ID")
        from huggingface_hub import snapshot_download

        revision = args.revision or (DEFAULT_REVISION if args.repo == DEFAULT_REPO else None)
        snapshot_download(repo_id=args.repo, revision=revision, local_dir=bundle,
                          allow_patterns=list(BUNDLE_FILES))
        # The official VPP2 repository supplies VAE, text encoder, CLIP, and tokenizer.
        if any(not (wan / filename).is_file() for filename in WAN_FILES):
            if wan != (POLICY_DIR / "checkpoints/Wan2.1-I2V-14B-480P").resolve():
                raise FileNotFoundError(f"Custom Wan directory is incomplete: {wan}")
            snapshot_download(
                repo_id="Haodong082399/VPP2", local_dir=POLICY_DIR,
                allow_patterns=["checkpoints/Wan2.1-I2V-14B-480P/**"],
            )
    verify(bundle, wan)


if __name__ == "__main__":
    main()
