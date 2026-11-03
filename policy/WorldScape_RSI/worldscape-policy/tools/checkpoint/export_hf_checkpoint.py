#!/usr/bin/env python3
"""Export a validated native checkpoint as a Hugging Face model directory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from worldscape_policy.checkpoint.loader import validate_native_checkpoint_artifacts


def export_checkpoint(source: Path, destination: Path) -> None:
    manifest = validate_native_checkpoint_artifacts(source)
    if destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        raise FileExistsError(f"output directory must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    names = set(manifest["artifacts"]) | {
        "tokenizer_reference.json",
        "README.md",
    }
    for name in sorted(names):
        candidate = source / name
        if candidate.is_file():
            shutil.copy2(candidate, destination / name)
    tokenizer = source / "tokenizer"
    if tokenizer.is_dir():
        shutil.copytree(tokenizer, destination / "tokenizer")
    if not (destination / "README.md").exists():
        (destination / "README.md").write_text(
            "---\n"
            "library_name: worldscape-policy\n"
            "license: apache-2.0\n"
            "tags:\n"
            "- robotics\n"
            "- world-action-model\n"
            "---\n\n"
            "# WorldScape Policy checkpoint\n\n"
            "This directory is a validated native WorldScape Policy checkpoint.\n"
            "See `checkpoint_manifest.json` for exact provenance and dependencies.\n"
        )
    (destination / "hf_export.json").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "format": "worldscape-native",
                "model_variant": manifest["model_variant"],
                "model_sha256": manifest["model_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    export_checkpoint(args.checkpoint, args.output_dir)
    print(f"Exported validated checkpoint to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
