"""Freeze a RoboCasa365 review policy template after source and model preflight.

    python -m oeh_policy.robocasa365_freeze <template.json> <frozen.json>

Records the robocasa and robosuite checkouts as they stand (commit and the sha256 of their tracked diff), checks the
released checkpoint and normalizer against the expected identities, hashes every file the runtime reads into
``identity_files`` (the official loop and released client, OpenWAM's utilities and server, the pinned robocasa and
robosuite files, this package, OpenWAM's own sources, the kernel, the checkpoint, the teacher's prompt files) and
validates the result with ``RoboCasa365Contract`` before writing it, once (``xb``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from .contracts import encoded, file_digest
from .robocasa365 import RoboCasa365Contract, required_identity_paths, review_prompt_paths


def _checkout(root: Path) -> tuple[str, str]:
    """A checkout's HEAD and the sha256 of its tracked diff against it."""
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    diff = hashlib.sha256(subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=root)).hexdigest()
    return head, diff


def freeze(template: Path, output: Path):
    cfg = json.loads(Path(template).read_text(encoding="utf-8"))
    native = cfg["robocasa365"]
    native["source_commit"], native["tracked_diff_sha256"] = _checkout(Path(native["root"]).resolve())
    native["robosuite_commit"], native["robosuite_tracked_diff_sha256"] = _checkout(
        Path(native["robosuite_root"]).resolve())
    cfg["checkpoint_sha256"] = file_digest(Path(native["checkpoint_file"]))
    cfg["normalization_sha256"] = file_digest(Path(native["normalization_file"]))
    if cfg["checkpoint_sha256"] != native["expected_checkpoint_sha256"]:
        raise ValueError("checkpoint does not match the selected released model")
    if cfg["normalization_sha256"] != native["expected_normalization_sha256"]:
        raise ValueError("normalizer does not match the selected released model")
    source = Path(__file__).resolve().parent
    paths = required_identity_paths(native) + list(source.glob("*.py"))
    paths.extend((Path(native["openwam_root"]) / "openwam").rglob("*.py"))
    paths.extend((Path(cfg["kernel"]).resolve(), Path(native["checkpoint_file"]).resolve(),
                  Path(native["normalization_file"]).resolve()))
    paths.extend(Path(path).resolve() for path in cfg.pop("identity_paths", []))
    paths.extend(review_prompt_paths())  # what the teacher reads: re-verified at every validation
    known = {str(Path(native["checkpoint_file"]).resolve()): cfg["checkpoint_sha256"]}  # hashed once above
    cfg["identity_files"] = {str(path): known.get(str(path)) or file_digest(path) for path in sorted(set(paths))}
    cfg = RoboCasa365Contract().validate_config(cfg)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(encoded(cfg))
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    freeze(args.template, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
