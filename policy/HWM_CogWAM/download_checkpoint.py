#!/usr/bin/env python3
"""Download the assets HWM_CogWAM needs for inference and stage them for
`setup_eval_policy_server.sh`.

Three independent downloads, one per environment variable the adapter reads:

1. ``COGWAM_ARTIFACT_DIR`` — the released CogWAM checkpoint (this policy's own
   weights), from ``HorizonRobotics/CogWAM``. Public, no token needed. The
   repository root *is* the artifact directory layout the policy server
   expects, so the download target can be passed to ``COGWAM_ARTIFACT_DIR``
   verbatim.
2. ``COGWAM_BASE_VLM`` — RynnBrain1.1-2B (Alibaba-DAMO-Academy, Apache-2.0).
   Public, no token needed.
3. ``COGWAM_DINO_MODEL`` — DINOv3 ViT-B/16 (Meta ``dinov3-license``). **GATED
   with manual approval**: you must request access on the model page with your
   own Hugging Face account, wait for Meta to grant it, and be logged in
   (``hf auth login``) before this download can succeed. DINOv3 is not
   optional — ``CogWAM.predict_action`` encodes the current observation with it
   on every step (``cogwam/models/cogwam.py``), so evaluation cannot run
   without these weights.

Usage:
    pip install "huggingface_hub[cli]"
    hf auth login              # required for the gated DINOv3 download
    python download_checkpoint.py --dest /path/to/assets
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

COGWAM_CHECKPOINT_REPO_ID = "HorizonRobotics/CogWAM"
# Pinned to the commit the leaderboard results were produced with, so a later
# push to the model repo cannot silently change what gets evaluated.
COGWAM_CHECKPOINT_REVISION = "017926b231a681944dc90ccacc77862847b10140"
RYNNBRAIN_REPO_ID = "Alibaba-DAMO-Academy/RynnBrain1.1-2B"
DINOV3_REPO_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"

# The four files cogwam.serve.policy_server hash-checks on every load, plus the
# manifest that carries those hashes.
REQUIRED_ARTIFACT_FILES = (
    "model.safetensors",
    "dataset_statistics.json",
    "checkpoint_keys.json",
    "inference_config.yaml",
    "artifact_manifest.json",
)

DINOV3_HELP = f"""
DINOv3 download failed. {DINOV3_REPO_ID} is gated under Meta's
`dinov3-license` and access is granted manually, so an anonymous or
unapproved download returns 401/403.

  1. Open https://huggingface.co/{DINOV3_REPO_ID}
  2. Accept the license and request access with your own account.
  3. Wait for Meta to approve (this is not instant).
  4. `hf auth login` with that account, then re-run this script.

DINOv3 cannot be skipped: CogWAM encodes every current observation with it
inside `predict_action`, and the released `inference_config.yaml` pins
`framework.dino.load_live_backbone: true` as part of the frozen reproduction
recipe.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", required=True, type=Path, help="Root directory to download into")
    parser.add_argument(
        "--revision",
        default=COGWAM_CHECKPOINT_REVISION,
        help=f"Revision of {COGWAM_CHECKPOINT_REPO_ID} (default: the evaluated commit)",
    )
    parser.add_argument("--skip-checkpoint", action="store_true", help="Skip the CogWAM checkpoint download")
    parser.add_argument("--skip-vlm", action="store_true", help="Skip the RynnBrain1.1-2B download")
    parser.add_argument("--skip-dino", action="store_true", help="Skip the DINOv3 download (evaluation will not run)")
    return parser.parse_args()


def download_checkpoint(dest: Path, revision: str) -> Path:
    artifact_dir = Path(
        snapshot_download(
            repo_id=COGWAM_CHECKPOINT_REPO_ID,
            revision=revision,
            local_dir=dest / "cogwam-robodojo-h25-50k",
        )
    )
    missing = [name for name in REQUIRED_ARTIFACT_FILES if not (artifact_dir / name).is_file()]
    if missing:
        raise SystemExit(f"[HWM_CogWAM] incomplete artifact in {artifact_dir}: missing {', '.join(missing)}")

    # Cheap integrity gate on the metadata the server needs before it can even
    # start hashing the weights; the full sha256 check happens at load time in
    # cogwam.serve.policy_server.
    manifest = json.loads((artifact_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    for name, meta in manifest.get("files", {}).items():
        expected = meta.get("size_bytes")
        actual = (artifact_dir / name).stat().st_size
        if expected is not None and actual != expected:
            raise SystemExit(f"[HWM_CogWAM] {name} is {actual} bytes, manifest expects {expected} — download is incomplete")
    return artifact_dir


def main() -> None:
    args = parse_args()
    dest = args.dest.expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    exports: list[str] = []

    if not args.skip_checkpoint:
        artifact_dir = download_checkpoint(dest, args.revision)
        exports.append(f"export COGWAM_ARTIFACT_DIR={artifact_dir}")

    if not args.skip_vlm:
        vlm_dir = snapshot_download(repo_id=RYNNBRAIN_REPO_ID, local_dir=dest / "rynnbrain1.1-2B")
        exports.append(f"export COGWAM_BASE_VLM={vlm_dir}")

    if not args.skip_dino:
        try:
            dino_dir = snapshot_download(repo_id=DINOV3_REPO_ID, local_dir=dest / "dinov3-vitb16")
        except Exception as exc:
            print(f"[HWM_CogWAM] {exc}\n\n{DINOV3_HELP}", file=sys.stderr)
            raise SystemExit(1) from exc
        exports.append(f"export COGWAM_DINO_MODEL={dino_dir}")

    print("\n[HWM_CogWAM] done. Export these before running eval.sh:\n")
    for line in exports:
        print(f"  {line}")


if __name__ == "__main__":
    main()
