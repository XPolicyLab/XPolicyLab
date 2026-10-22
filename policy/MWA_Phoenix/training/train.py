"""XPolicyLab train convention backed by the upstream RoboTwin-only recipe."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from XPolicyLab.utils.checkpoint_resolver import build_run_dir_name
from ..runtime.loader import SOURCE, POLICY_DIR, activate_source
from ..runtime.checkpoints import verify_local, manifest


def prepare(args):
    if (args.bench_name, args.env_cfg_type, args.action_type) != (
        "RoboTwin",
        "arx_x5",
        "joint",
    ):
        raise ValueError("Training supports RoboTwin arx_x5 joint only")
    for key in ("ckpt_name",):
        value = getattr(args, key)
        if not value or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
            for c in value
        ):
            raise ValueError("Use a simple checkpoint name")
    cfg = vars(args)
    run = build_run_dir_name(cfg)
    output = (POLICY_DIR / "checkpoints" / run).resolve()
    if output.exists():
        raise FileExistsError(
            f"Choose a new checkpoint name; refusing existing {output}"
        )
    if args.nnodes < 1 or args.nproc < 1 or not 0 <= args.node_rank < args.nnodes:
        raise ValueError("Invalid distributed topology")
    if args.nnodes > 1 and not args.master_addr:
        raise ValueError("Multi-node training needs --master-addr")
    init = args.init.expanduser().resolve(strict=True)
    weights = args.weights.expanduser().resolve(strict=True)
    data = args.data.expanduser().resolve(strict=True)
    text = args.text_cache.expanduser().resolve(strict=True)
    verify_local(weights, full_hash=False, include_policy=False)
    for i in range(1, 4):
        path = (
            weights
            / "Wan-AI/Wan2.2-TI2V-5B"
            / f"diffusion_pytorch_model-{i:05d}-of-00003.safetensors"
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing training backbone {path}; download_checkpoint.sh --training-backbones"
            )
    activate_source()
    import torch
    from wam.training.checkpoint import _joint96_identity

    payload = torch.load(init, map_location="cpu", mmap=True, weights_only=True)
    _joint96_identity(payload.get("joint96_metadata"))
    print("Initialization step:", payload.get("step"))
    del payload
    from .data import validate_dataset, validate_cache

    validate_dataset(data)
    validate_cache(data, text)
    env = dict(
        os.environ,
        PYTHONPATH=str(SOURCE / "src") + ":" + str(SOURCE),
        CUDA_VISIBLE_DEVICES=args.gpu_id,
        WAM_PRETRAIN_CHECKPOINT=str(init),
        DINO_ENCODER_CHECKPOINT=str(weights / "encoders/dino110k.pt"),
        STVAE_CHECKPOINT=str(weights / "encoders/stvae150k.pt"),
        DIFFSYNTH_MODEL_BASE_PATH=str(weights),
        ROBOTWIN_DATA_ROOT=str(data),
        ROBOTWIN_TEXT_CACHE=str(text),
        ROBOTWIN_NORM_STATS=str(SOURCE / "assets/stats/robotwin.json"),
        EMBODIMENT_REGISTRY=str(
            SOURCE
            / "configs/embodiment_registry/local_robot37_arm18_capacity28_v1.json"
        ),
        RUN_ID=run,
        HF_HUB_OFFLINE="1",
        DIFFSYNTH_SKIP_DOWNLOAD="true",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
    )
    overrides = [
        "task=robotwin_posttrain",
        f"output_dir={output}",
        f"seed={args.seed}",
        f"max_steps={args.max_steps}",
        f"batch_size={args.batch_size}",
        f"gradient_accumulation_steps={args.accumulation}",
        "wandb.enabled=false",
    ]
    command = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--config_file",
        "scripts/accelerate_configs/accelerate_joint96_posttrain_zero1.yaml",
        "--num_machines",
        str(args.nnodes),
        "--num_processes",
        str(args.nnodes * args.nproc),
        "--machine_rank",
        str(args.node_rank),
        "--main_process_ip",
        args.master_addr or "127.0.0.1",
        "--main_process_port",
        str(args.master_port),
        "--deepspeed_multinode_launcher",
        "standard",
        "scripts/train.py",
        *overrides,
    ]
    # Resolve the actual Hydra recipe before launching workers.
    from hydra import initialize_config_dir, compose
    from wam.utils.config_resolvers import register_default_resolvers
    from wam.posttrain.guard import validate_scope
    from omegaconf import OmegaConf

    register_default_resolvers()
    old = dict(os.environ)
    try:
        os.environ.update(env)
        with initialize_config_dir(
            config_dir=str(SOURCE / "configs"), version_base="1.3"
        ):
            composed = compose(config_name="train", overrides=overrides)
            OmegaConf.resolve(composed)
            validate_scope(composed)
    finally:
        os.environ.clear()
        os.environ.update(old)
    return output, weights, env, command


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bench_name", "ckpt_name", "env_cfg_type", "action_type"):
        p.add_argument(name)
    p.add_argument("seed", type=int)
    p.add_argument("gpu_id")
    p.add_argument(
        "--init",
        type=Path,
        required=True,
        help="Joint96 pretrained checkpoint; never implicitly replaced with FT35k",
    )
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--text-cache", type=Path, required=True)
    p.add_argument("--nnodes", type=int, default=1)
    p.add_argument("--nproc", type=int, default=1)
    p.add_argument("--node-rank", type=int, default=int(os.environ.get("NODE_RANK", 0)))
    p.add_argument("--master-addr", default=os.environ.get("MASTER_ADDR"))
    p.add_argument("--master-port", type=int, default=24491)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--accumulation", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=35000)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    output, weights, env, command = prepare(args)
    print(
        json.dumps(
            dict(
                output=str(output),
                command=command,
                global_batch=args.nnodes
                * args.nproc
                * args.batch_size
                * args.accumulation,
            ),
            indent=2,
        )
    )
    if args.dry_run:
        return
    subprocess.run(command, cwd=SOURCE, env=env, check=True)
    if args.node_rank != 0:
        return
    m = json.loads((output / "checkpoints/weights/weights_manifest.json").read_text())
    last = output / "checkpoints/weights" / m["latest"]["path"]
    shutil.copyfile(last, output / "policy.pt")
    for spec in manifest()["files"]:
        if spec["path"] == "policy.pt":
            continue
        dest = output / spec["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(weights / spec["path"])
    tok = manifest()["tokenizer"]["path"]
    dest = output / tok
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.symlink_to(weights / tok)
    (output / "training_export.json").write_text(
        json.dumps(
            dict(
                step=m["latest"]["step"],
                initialization=str(args.init),
                shared_weights=str(weights),
            ),
            indent=2,
        )
    )
    print("Training complete; evaluate with ckpt_name=" + output.name)


if __name__ == "__main__":
    main()
