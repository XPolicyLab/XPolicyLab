"""Losslessly prepare inference assets; never instantiate a training dataset."""
import argparse
import json
import shutil
from pathlib import Path
from .asset_storage import configure_workspace, require_work_path, staged_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task-context", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    work = configure_workspace()
    output = require_work_path(args.output, work)
    if output.exists():
        raise FileExistsError(f"Will not overwrite an existing bundle: {output}")
    import dill
    import torch
    from omegaconf import OmegaConf
    from .assets import (MODEL_FINGERPRINT, SOURCE_SHA256, manifest_fingerprint,
                         sha256_file, tensor_manifest, validate_bundle)
    # Trusted historical files use dill/OmegaConf. The output is tensors + JSON only.
    payload = torch.load(args.checkpoint, map_location="cpu", mmap=True, pickle_module=dill, weights_only=False)
    state = payload["state_dicts"]["model"]
    manifest = tensor_manifest(state)
    if len(manifest) != 886 or manifest_fingerprint(manifest) != MODEL_FINGERPRINT:
        raise ValueError("Checkpoint is not the verified historical s742500 model")
    cfg = OmegaConf.to_container(payload["cfg"], resolve=False)
    policy = dict(cfg["policy"])
    policy.pop("obs_encoder")
    for key in ("wan_checkpoint_dir", "task_ctx_path", "latent_cache_dir", "prompt_embedding_path"):
        policy.pop(key, None)
    config = {
        "format": "rap-wam-v58-light-inference-v1",
        "source_checkpoint_sha256": SOURCE_SHA256,
        "model_fingerprint": MODEL_FINGERPRINT,
        "policy": policy,
        "action_dim": 14, "action_horizon": 16, "observation_horizon": 1,
        "video_horizon": 16, "image_size": [384, 320],
        "protocol": {"num_inference_steps": 30, "replan_steps": 8,
                     "savgol_window": 9, "savgol_polyorder": 2, "boundary_blend_steps": 4},
    }
    table = torch.load(args.task_context, map_location="cpu", weights_only=True)
    if tuple(table["embedding_table"].shape) != (35, 512, 4096) or len(table["tasks"]) != 35:
        raise ValueError("Expected the original trained 35-task table")
    # Preserve every tensor's original dtype and value; omit optimizer/scheduler only.
    with staged_output(output, work) as root:
        torch.save(state, root / "model.pt")
        shutil.copyfile(args.task_context, root / "task_context.pt")
        (root / "task_strings.json").write_text(json.dumps(table["tasks"], ensure_ascii=False, indent=2) + "\n")
        (root / "inference_config.json").write_text(json.dumps(config, indent=2) + "\n")
        (root / "model_tensors.json").write_text(json.dumps(manifest, indent=2) + "\n")
        names = ["model.pt", "task_context.pt", "task_strings.json", "inference_config.json", "model_tensors.json"]
        (root / "SHA256SUMS").write_text("".join(f"{sha256_file(root / name)}  {name}\n" for name in names))
        validate_bundle(root)
    print(json.dumps({"bundle": str(output), "model_tensors": len(manifest), "model_fingerprint": MODEL_FINGERPRINT}))


if __name__ == "__main__":
    main()
