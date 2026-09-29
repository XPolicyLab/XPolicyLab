#!/usr/bin/env python3
"""Validate a released CogWAM artifact directory.

Two levels, because the expensive one needs the backbones on disk:

* default -- verify the manifest digests, the key inventory and the geometry.
  Reads only the safetensors header, so it costs a few milliseconds and no GPU.
* ``--build`` -- additionally construct the framework from
  ``inference_config.yaml`` and load the weights with ``strict=True``. This is
  the check that actually proves the release loads; it needs
  ``COGWAM_BASE_VLM`` and ``COGWAM_DINO_MODEL`` and roughly 16 GB of RAM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any

MANIFEST_FILENAME = "artifact_manifest.json"
MODEL_WEIGHTS_FILENAME = "model.safetensors"
KEY_INVENTORY_FILENAME = "checkpoint_keys.json"
INFERENCE_CONFIG_FILENAME = "inference_config.yaml"

EXPECTED_GEOMETRY = {
    "action_horizon": 25,
    "num_world_queries": 16,
    "planner_dim": 2048,
    "vocab_size": 248081,
    "world_hidden_size": 512,
    "dino_embed_dim": 768,
    "action_hidden_size": 1024,
    "action_dim": 14,
    "num_mot_layers": 30,
}

_CHUNK = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def read_safetensors_header(path: Path) -> dict[str, Any]:
    """Parse only the JSON header: 8-byte little-endian length, then that many bytes."""
    with path.open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length))
    header.pop("__metadata__", None)
    return header


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, condition: bool, message: str) -> None:
        if condition:
            print(f"  ok    {message}")
        else:
            print(f"  FAIL  {message}")
            self.failures.append(message)


def verify_static(artifact: Path, report: Report, *, skip_digests: bool) -> dict[str, Any]:
    manifest_path = artifact / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise SystemExit(f"{artifact}: missing {MANIFEST_FILENAME}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    print("manifest")
    report.check(manifest.get("model", {}).get("framework") == "CogWAM", "framework is CogWAM")
    declared = manifest.get("files", {})

    print("files")
    for name, entry in sorted(declared.items()):
        path = artifact / name
        if not path.is_file():
            report.check(False, f"{name} present")
            continue
        report.check(path.stat().st_size == entry["size_bytes"], f"{name} size matches manifest")
        if skip_digests:
            print(f"  skip  {name} sha256 (--skip-digests)")
        else:
            report.check(sha256_file(path) == entry["sha256"], f"{name} sha256 matches manifest")

    print("weights")
    header = read_safetensors_header(artifact / MODEL_WEIGHTS_FILENAME)
    expected_tensors = manifest["model"]["tensor_count"]
    report.check(len(header) == expected_tensors, f"tensor count is {expected_tensors}")

    prefixes: dict[str, int] = {}
    for key in header:
        prefixes[key.split(".", 1)[0]] = prefixes.get(key.split(".", 1)[0], 0) + 1
    report.check(prefixes == manifest["model"]["prefix_counts"], f"prefix counts match manifest {prefixes}")

    inventory_path = artifact / KEY_INVENTORY_FILENAME
    if inventory_path.is_file():
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        from_header = {key: [value["dtype"].lower(), value["shape"]] for key, value in header.items()}
        report.check(set(inventory) == set(from_header), "key inventory matches the weight file")
        mismatched = [key for key in inventory if key in from_header and inventory[key][1] != from_header[key][1]]
        report.check(not mismatched, f"all shapes match the inventory (differing: {mismatched[:3]})")

    print("geometry")
    geometry = manifest["model"].get("geometry", {})
    for key, expected in EXPECTED_GEOMETRY.items():
        report.check(geometry.get(key) == expected, f"{key} == {expected} (got {geometry.get(key)})")

    return manifest


def verify_build(artifact: Path, report: Report) -> None:
    config_path = artifact / INFERENCE_CONFIG_FILENAME
    if not config_path.is_file():
        raise SystemExit(
            f"{artifact}: --build needs {INFERENCE_CONFIG_FILENAME}. "
            "Re-run tools/convert_checkpoint.py with --config."
        )

    import torch  # noqa: F401  (imported for its side effects on the model build)
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    from cogwam.models.base import build_model
    from cogwam.training.config import apply_config_compat

    print("build")
    cfg = apply_config_compat(OmegaConf.load(config_path))
    model = build_model(cfg)
    report.check(True, "framework constructed from inference_config.yaml")

    state = load_file(str(artifact / MODEL_WEIGHTS_FILENAME))
    missing, unexpected = model.load_state_dict(state, strict=False)
    report.check(not missing, f"no missing keys (first few: {list(missing)[:3]})")
    report.check(not unexpected, f"no unexpected keys (first few: {list(unexpected)[:3]})")
    if not missing and not unexpected:
        model.load_state_dict(state, strict=True)
        report.check(True, "strict=True load succeeded")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--build", action="store_true", help="also construct the model and load with strict=True")
    parser.add_argument("--skip-digests", action="store_true", help="skip the 7.8 GiB sha256 re-read")
    args = parser.parse_args()

    report = Report()
    verify_static(args.artifact, report, skip_digests=args.skip_digests)
    if args.build:
        verify_build(args.artifact, report)

    if report.failures:
        print(f"\n{len(report.failures)} check(s) FAILED")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
