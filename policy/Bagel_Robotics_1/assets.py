"""Pinned CP1000 artifact and inference contract; safe to import without torch."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

POLICY_DIR = Path(__file__).resolve().parent
MANIFEST = json.loads((POLICY_DIR / "artifact-manifest.json").read_text())
FIXED_SETTINGS = {
    "bench_name": "RoboDojo", "env_cfg_type": "arx_x5", "action_type": "ee",
    "protocol": "ws", "eval_batch": False, "mixed_precision": "bf16",
    "action_horizon": 32, "replan_steps": 16, "num_inference_steps": 10,
    "sigma_shift": 1.0, "deployment_adapter": "robodojo_ee16",
    "temporal_ensemble": {"enabled": True, "new_weight": 0.9, "old_weight": 0.1},
}


def file_digest(path, algorithm="sha256", git_blob=False):
    digest = hashlib.new(algorithm)
    if git_blob:
        digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle(bundle, *, full_hash=True):
    bundle = Path(bundle).expanduser().resolve()
    for name, expected in MANIFEST["files"].items():
        path = bundle / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint artifact: {path}")
        if path.stat().st_size != expected["size"]:
            raise ValueError(f"Checkpoint size mismatch: {name}")
        if full_hash or name not in {"action.pt", "video.pt"}:
            if file_digest(path) != expected["sha256"]:
                raise ValueError(f"Checkpoint SHA-256 mismatch: {name}")
    receipt = json.loads((bundle / "COMPLETE.json").read_text())
    if receipt.get("identity") != MANIFEST["checkpoint_identity"] or receipt.get("update") != 1000:
        raise ValueError("Not the approved recovery checkpoint 1000")
    architecture = json.loads((bundle / "compact_architecture.json").read_text())
    if architecture.get("schema") != "vpp2-compact-channels-v1" or architecture.get("parameters") != MANIFEST["video_parameters"]:
        raise ValueError("Unexpected compact architecture")
    return {"checkpoint_identity": receipt["identity"], "update": 1000,
            "files": len(MANIFEST["files"]), "full_hash": full_hash}


def verify_wan(wan_root, *, full_hash=True):
    wan_root = Path(wan_root).expanduser().resolve()
    prefix = "checkpoints/Wan2.1-I2V-14B-480P/"
    for entry in MANIFEST["wan"]["files"]:
        path = wan_root / entry["rfilename"].removeprefix(prefix)
        if not path.is_file():
            raise FileNotFoundError(f"Missing encoder asset: {path}")
        if path.stat().st_size != entry["size"]:
            raise ValueError(f"Encoder asset size mismatch: {path.name}")
        if full_hash:
            lfs = entry.get("lfs")
            actual = file_digest(path, "sha256" if lfs else "sha1", git_blob=not bool(lfs))
            expected = lfs["sha256"] if lfs else entry["blobId"]
            if actual != expected:
                raise ValueError(f"Encoder asset checksum mismatch: {path.name}")
    return {"repo": MANIFEST["wan"]["repo"], "revision": MANIFEST["wan"]["revision"],
            "files": len(MANIFEST["wan"]["files"]), "full_hash": full_hash}


def validate_settings(model_cfg):
    cfg = dict(model_cfg)
    for key, expected in FIXED_SETTINGS.items():
        actual = cfg.get(key, expected)
        if actual != expected:
            raise ValueError(f"CP1000 requires {key}={expected!r}; got {actual!r}")
        cfg[key] = expected
    seed = cfg.get("seed", 1)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer supplied by the evaluator")
    cfg["seed"] = seed
    return cfg


def prepare_model_config(model_cfg, policy_dir=POLICY_DIR):
    from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root

    cfg = validate_settings(model_cfg)
    explicit = os.environ.get("BAGEL_CP1000_BUNDLE") or cfg.get("bundle_path")
    lookup = dict(cfg)
    if explicit:
        lookup["model_path"] = explicit
    bundle = resolve_checkpoint_root(lookup, Path(policy_dir) / "checkpoints",
                                     policy_dir=policy_dir)
    wan_root = os.environ.get("BAGEL_CP1000_WAN_ROOT") or cfg.get("wan_model_dir")
    if not wan_root:
        wan_root = Path(policy_dir) / "checkpoints/Wan2.1-I2V-14B-480P"
    # Verify before loading any tensors. No silently accepted alternate weights.
    verification = {"checkpoint": verify_bundle(bundle), "encoders": verify_wan(wan_root)}
    print("[Bagel-Robotics-1] " + json.dumps(verification), flush=True)
    cfg.update(checkpoint_path=str(bundle / "action.pt"),
               video_checkpoint_path=str(bundle / "video.pt"),
               dataset_stats_path=str(bundle / "dataset_stats.json"),
               wan_model_dir=str(Path(wan_root).expanduser().resolve()))
    return cfg
