"""Verify both external WSP checkpoint bindings without model inference."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

if __package__:
    from .wam import _verify_files
    from .release import FrozenRelease, ROOT
else:
    import sys
    checkout = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(checkout.parent), str(checkout)]
    from XPolicyLab.policy.WorldScape_RSI.harness.wam import _verify_files
    from XPolicyLab.policy.WorldScape_RSI.harness.release import FrozenRelease, ROOT


def verify():
    release = FrozenRelease(ROOT / "release.json")
    receipt = {"release_sha256": release.sha256, "model_inference": False, "checkpoints": {}}
    for skill, spec in release.data["wam_skills"].items():
        path = os.environ.get(spec["checkpoint_env"], "") or os.environ.get(spec.get("legacy_checkpoint_env", ""), "")
        if not path or not Path(path).is_absolute():
            raise ValueError("Set an absolute directory in " + spec["checkpoint_env"])
        _verify_files(path, spec["checkpoint_files"], skill)
        _verify_files(ROOT.parent.parent / "WorldScape_Policy_2", spec["source_files"], "Adjacent WSP adapter")
        receipt["checkpoints"][skill] = {"path": str(Path(path).resolve()), "profile": spec["profile"],
                                       "verified_files": len(spec["checkpoint_files"])}
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = verify()
    if args.output:
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))
