#!/usr/bin/env python
# coding=utf-8
# Copyright (C) 2026 Tencent.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Overlay RoboDojo post-training support onto a Hy-Embodied-0.5-VLA clone.

The public Hy-Embodied-0.5-VLA repo (https://github.com/Tencent-Hunyuan/
Hy-Embodied-0.5-VLA) does not ship RoboDojo dataset support. XPolicyLab carries
the RoboDojo files under ``policy/hy_vla/robodojo/`` and this script overlays
them onto a local clone of that repo. We never push to the public repo.

The overlay is two steps:

  1. Copy the RoboDojo payload files (dataset loader, Hydra config, norm-stats
     computer, training launcher) into their matching paths inside the clone.
  2. Insert a ``source == "robodojo"`` branch into the clone's
     ``hy_vla/data/vla_dataset.py`` dataset dispatcher.

Both steps are idempotent: re-running against an already-overlaid clone is a
no-op. A preflight verifies the upstream transforms the RoboDojo code relies on
are still present, so a breaking upstream change fails here with a clear message
rather than deep inside training.

Usage::

    python apply_robodojo_overlay.py <HY_VLA_ROOT>
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# This script lives at policy/hy_vla/; the payload is under policy/hy_vla/robodojo/.
_POLICY_DIR = Path(__file__).resolve().parent
_PAYLOAD_DIR = _POLICY_DIR / "robodojo"

# Payload file -> destination, both relative to their respective roots
# (payload dir mirrors the clone layout, so the relative paths are identical).
_PAYLOAD_FILES = [
    "hy_vla/data/robodojo_dataset.py",
    "hy_vla/config/dataset/robodojo_hdf5.yaml",
    "scripts/compute_norm_robodojo.py",
    "scripts/train_robodojo_umi.sh",
]

# Upstream transforms the RoboDojo dataset / norm-stats code imports from
# ``hy_vla.utils.transform_utils``. Present in the public repo today; if a
# future upstream drops one, we want to fail here with a clear message.
_REQUIRED_TRANSFORMS = (
    "convert_frame_robo_to_umi",
    "dual_arm_poses_to_relative",
    "convert_PosQuat2PosRotationMatrix_batch",
)

# Dataset dispatcher (in the clone) whose source-routing we extend.
_DISPATCH_REL = "hy_vla/data/vla_dataset.py"

# The upstream anchor line we splice in front of. Kept as the exact source
# text so a change in the upstream dispatcher surfaces as a hard error.
_ANCHOR = '        if dataset_source == "umi":'

# What the anchor becomes: a new robodojo branch, then the original umi test
# demoted to elif. Indentation matches the 8-space method body.
_REPLACEMENT = (
    '        if dataset_source == "robodojo":\n'
    "            from .robodojo_dataset import RoboDojoVLADataset\n"
    "            self.hdf5_dataset = RoboDojoVLADataset(config)\n"
    '        elif dataset_source == "umi":'
)

# Comment line documenting the umi source; we add a robodojo sibling above it
# so the dispatcher's docstring-comment stays accurate. Optional (best effort).
_COMMENT_ANCHOR = '        #   source="umi"       → LanceVLADataset  (UMI / Hy-Embodied)'
_COMMENT_ADD = (
    '        #   source="umi"       → LanceVLADataset  (UMI / Hy-Embodied)\n'
    "        #   source=\"robodojo\"  → RoboDojoVLADataset"
)


def _fail(msg: str) -> "None":
    print(f"[robodojo-overlay][ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def _preflight(hy_root: Path) -> None:
    """Verify the clone looks like Hy-Embodied and still exposes the
    transforms the RoboDojo code depends on."""
    transforms_py = hy_root / "hy_vla" / "utils" / "transform_utils.py"
    if not transforms_py.is_file():
        _fail(
            f"{transforms_py} not found. Is {hy_root} a Hy-Embodied-0.5-VLA "
            f"checkout? Clone it from "
            f"https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA"
        )
    src = transforms_py.read_text(encoding="utf-8")
    missing = [name for name in _REQUIRED_TRANSFORMS if f"def {name}" not in src]
    if missing:
        _fail(
            "the public Hy-Embodied repo changed; the RoboDojo overlay needs "
            f"these transforms in hy_vla/utils/transform_utils.py: {missing}. "
            "Update policy/hy_vla/robodojo/ to match upstream."
        )


def _copy_payload(hy_root: Path) -> None:
    for rel in _PAYLOAD_FILES:
        src = _PAYLOAD_DIR / rel
        if not src.is_file():
            _fail(f"payload file missing from XPolicyLab: {src}")
        dst = hy_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"[robodojo-overlay] copied {rel}")


def _patch_dispatch(hy_root: Path) -> None:
    dispatch = hy_root / _DISPATCH_REL
    if not dispatch.is_file():
        _fail(f"{dispatch} not found; cannot wire the robodojo dataset branch.")
    src = dispatch.read_text(encoding="utf-8")

    if "robodojo" in src:
        print(f"[robodojo-overlay] {_DISPATCH_REL} already patched; skipping")
        return

    if _ANCHOR not in src:
        _fail(
            f"dispatch anchor not found in {_DISPATCH_REL}:\n    {_ANCHOR!r}\n"
            "The upstream dataset dispatcher changed. Update the anchor in "
            "apply_robodojo_overlay.py to match the current source."
        )
    if src.count(_ANCHOR) != 1:
        _fail(f"dispatch anchor is not unique in {_DISPATCH_REL}; aborting.")

    patched = src.replace(_ANCHOR, _REPLACEMENT, 1)
    # Best-effort: extend the explanatory comment block, too.
    if _COMMENT_ANCHOR in patched:
        patched = patched.replace(_COMMENT_ANCHOR, _COMMENT_ADD, 1)

    dispatch.write_text(patched, encoding="utf-8")
    print(f"[robodojo-overlay] patched {_DISPATCH_REL} (added robodojo branch)")


def main() -> None:
    if len(sys.argv) != 2:
        _fail("usage: python apply_robodojo_overlay.py <HY_VLA_ROOT>")
    hy_root = Path(sys.argv[1]).expanduser().resolve()
    if not hy_root.is_dir():
        _fail(f"HY_VLA_ROOT is not a directory: {hy_root}")

    print(f"[robodojo-overlay] target clone: {hy_root}")
    _preflight(hy_root)
    _copy_payload(hy_root)
    _patch_dispatch(hy_root)
    print("[robodojo-overlay] done: RoboDojo post-training support is available "
          "(dataset=robodojo_hdf5).")


if __name__ == "__main__":
    main()
