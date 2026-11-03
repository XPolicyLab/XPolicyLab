"""Freeze a RoboTwin policy template after source and asset preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from .brain_preflight import _providers
from .contracts import encoded, file_digest
from .profiles import DUAL_ARM_EE16, check_primitive_limits
from .robotwin import RoboTwinContract, required_identity_paths, review_prompt_paths
from .robotwin_bounds import apply_pinned_bounds


def freeze(template: Path, output: Path):
    cfg = json.loads(template.read_text(encoding="utf-8"))
    # Slow-primitive and review workspace bounds come from the pinned RoboTwin files, never from the
    # template; they are filled before the profile check, which refuses a placeholder.
    bounds_files = apply_pinned_bounds(cfg)
    hybrid = cfg.get("mode") == "hybrid"
    if hybrid:
        check_primitive_limits(DUAL_ARM_EE16, cfg.get("primitives"))
        brain = cfg["brain"]
        brain["providers"] = str(Path(brain["providers"]).resolve())
        # The provider's declaration (for codex_exec: the variable naming the Codex login directory), so
        # the campaign can hand that variable to this runtime's process only, without parsing YAML.
        declared = _providers(Path(brain["providers"])).get(brain["provider"])
        if not isinstance(declared, dict):
            raise ValueError(f"brain provider {brain['provider']!r} is not declared in {brain['providers']}")
        brain["provider_declaration"] = declared
    native = cfg["robotwin"]
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
                  Path(native["normalization_file"]).resolve()))
    paths.extend(Path(path).resolve() for path in cfg.pop("identity_paths", []))
    paths.extend(path.resolve() for path in bounds_files)  # re-verified at every validation
    if hybrid:
        paths.extend((Path(cfg["brain"]["providers"]), DUAL_ARM_EE16.persona_path))
    if cfg.get("mode") == "review":
        paths.extend(review_prompt_paths())  # what the teacher reads: re-verified at every validation
    known = {str(Path(native["checkpoint_file"]).resolve()): cfg["checkpoint_sha256"]}  # hashed once above
    cfg["identity_files"] = {str(path): known.get(str(path)) or file_digest(path) for path in sorted(set(paths))}
    cfg = RoboTwinContract().validate_config(cfg)
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
