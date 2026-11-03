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
    parser.add_argument("--check-runtime",action="store_true")
    parser.add_argument("--check-robot-assets",action="store_true")
    parser.add_argument("--check-cuda",action="store_true")
    args=parser.parse_args()
    release=FrozenRelease(ROOT/"release.json")
    result={"release_sha256":release.sha256,"mode":release.data["mode"],
            "official_single_entry_eligible":False,"checkpoint_inference":False}
    if args.check_runtime:
        from XPolicyLab.policy.WorldScape_RSI.harness.backends import register_vendor
        register_vendor()
        from XPolicyLab.policy.WorldScape_RSI.harness.vendor.runtime_support.native_runtime.env.planner_manager.curobo_planner import CuroboPlanner
        import torch, warp, yourdfpy, trimesh, importlib.metadata
        result["runtime_imports"] = {"passed": True, "torch": torch.__version__,
                                    "warp": importlib.metadata.version("warp-lang"),
                                    "gpu_inference": False, "planner_execution": False}
    if args.check_cuda:
        import torch, warp
        warp.init()
        if not torch.cuda.is_available() or warp.get_cuda_device_count() < 1:
            raise RuntimeError("Torch and Warp both need a visible CUDA device. On CUDA 12 drivers, install the pinned CUDA12-compatible Warp 1.17.0 wheel in requirements-runtime.txt.")
        result["cuda_runtime"]={"torch_available":True,"warp_available":True,
                                "model_inference":False,"planner_execution":False}
    if args.check_robot_assets:
        from XPolicyLab.policy.WorldScape_RSI.harness.backends import register_vendor
        register_vendor()
        from XPolicyLab.policy.WorldScape_RSI.harness.robot_spec import load_robot_spec
        robot=load_robot_spec()
        result["robot_assets"]={"passed":True,"joint_channels":len(robot.labels),
                                "control_hz":robot.control_hz}
    if args.check_checkpoints:
        from XPolicyLab.policy.WorldScape_RSI.harness.verify_checkpoints import verify
        result["checkpoint_verification"]=verify()
    print(json.dumps(result))
