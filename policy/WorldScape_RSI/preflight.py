"""Verify the active release and optionally both external WAM checkpoints."""
from pathlib import Path
import sys
checkout=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(checkout.parent),str(checkout)]
from XPolicyLab.policy.WorldScape_RSI.harness.release import FrozenRelease, ROOT
import argparse
import json
if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-checkpoints",action="store_true")
    args=parser.parse_args()
    release=FrozenRelease(ROOT/"release.json")
    result={"release_sha256":release.sha256,"mode":release.data["mode"],
            "official_single_entry_eligible":False,"checkpoint_inference":False}
    if args.check_checkpoints:
        from XPolicyLab.policy.WorldScape_RSI.harness.verify_checkpoints import verify
        result["checkpoint_verification"]=verify()
    print(json.dumps(result))
