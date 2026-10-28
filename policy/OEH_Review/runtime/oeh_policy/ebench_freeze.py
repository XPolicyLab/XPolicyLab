"""Freeze an EBench review policy template after source and asset preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from .contracts import encoded, file_digest
from .ebench import EBenchContract, required_identity_paths, review_prompt_paths
from .ebench_bounds import apply_pinned_bounds


def freeze(template: Path, output: Path):
    cfg = json.loads(template.read_text(encoding="utf-8"))
    # The review workspace comes from the pinned GenManip files and EBench assets, never from the template; it is
    # filled before the contract check, which refuses a placeholder.
    bounds_files = apply_pinned_bounds(cfg)
    native = cfg["ebench"]
    root = Path(native["root"]).resolve()
    native["source_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    native["tracked_diff_sha256"] = hashlib.sha256(
        subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=root)).hexdigest()
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
                  Path(native["normalization_file"]).resolve(), Path(native["checkpoint_config"]).resolve()))
    paths.extend(Path(path).resolve() for path in cfg.pop("identity_paths", []))
    paths.extend(path.resolve() for path in bounds_files)  # re-verified at every validation
    paths.extend(review_prompt_paths())  # what the teacher reads: re-verified at every validation
    known = {str(Path(native["checkpoint_file"]).resolve()): cfg["checkpoint_sha256"]}  # hashed once above
    cfg["identity_files"] = {str(path): known.get(str(path)) or file_digest(path) for path in sorted(set(paths))}
    cfg = EBenchContract().validate_config(cfg)
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
