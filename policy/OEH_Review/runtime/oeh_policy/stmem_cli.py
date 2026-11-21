"""Offline replay/preflight for the same worker used by review-stmem."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys


def preflight(config):
    import importlib.metadata
    from .stmem_geometry import backproject
    import numpy as np
    cfg = json.loads(Path(config).read_text(encoding="utf-8"))
    cfg = cfg.get("stmem", cfg)
    missing = []
    for key in ("da3_checkpoint", "sam3_checkpoint", "calibration_file"):
        if not Path(cfg[key]).exists():
            missing.append(key)
    versions = {}
    for package in ("torch", "depth-anything-3", "sam3", "scipy", "Pillow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            missing.append(package)
    if Path(cfg["calibration_file"]).exists():
        cameras = json.loads(Path(cfg["calibration_file"]).read_text(encoding="utf-8"))
        camera = cameras[cfg.get("camera", "cam_head")]
        if camera.get("convention") != "opencv" or camera.get("units") != "m" or not camera.get("frame_id"):
            raise ValueError("calibration must name OpenCV convention, metres and world frame")
        if camera.get("source") != "observation":
            backproject(np.ones((4, 4)), np.ones((4, 4), bool), camera["K"], camera["world_from_camera"])
    if "torch" not in missing:
        import torch
        if not torch.cuda.is_available():
            missing.append("CUDA device")
    result = {"ready_for_model_smoke": not missing, "missing": missing, "versions": versions,
              "note": "This checks dependencies and calibration; model loading/inference latency still require smoke/replay."}
    print(json.dumps(result, indent=2))
    return 1 if missing else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["preflight", "replay"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--frames", help="JSONL requests with input npz, time, frame_id and camera")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.command == "preflight":
        return preflight(args.config)
    if not args.frames or not args.output:
        parser.error("replay requires --frames and --output")
    from .stmem_perception import PerceptionWorker
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    cfg = cfg.get("stmem", cfg)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    worker = PerceptionWorker(cfg, output)
    with (output/"contexts.jsonl").open("w", encoding="utf-8") as dest:
        for line in Path(args.frames).read_text(encoding="utf-8").splitlines():
            if line.strip():
                dest.write(json.dumps(worker.process(json.loads(line)))+"\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
