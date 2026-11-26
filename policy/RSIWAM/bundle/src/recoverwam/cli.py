"""Small command surface for the released RoboDojo training/evaluation stage."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def load_config(path, overrides=()):
    from omegaconf import OmegaConf

    def read(path, parents=()):
        path = Path(path).resolve()
        if path in parents:
            raise ValueError(f"Cyclic configuration inheritance: {path}")
        value = OmegaConf.load(path)
        base = value.pop("_base_", None)
        return value if base is None else OmegaConf.merge(
            read(path.parent / base, (*parents, path)), value
        )

    cfg = OmegaConf.merge(read(path), OmegaConf.from_dotlist(list(overrides)))
    if cfg.train_mode == "video_only":
        from .video_training import validate_config

        validate_config(cfg)
        return cfg
    if cfg.train_mode != "joint" or cfg.checkpoint_mode != "full":
        raise ValueError("Only full joint training is supported")
    m, d = cfg.model, cfg.data.train
    if (
        m.action_dit_config.hidden_dim,
        m.action_dit_config.ffn_dim,
        m.action_dit_config.num_layers,
    ) != (1024, 4096, 40):
        raise ValueError("Expected the reference Action2B architecture")
    if m.get("joint_denoising", False) or m.video_lora.enabled:
        raise ValueError("Joint denoising and LoRA are not part of this release")
    if (d.action_horizon, d.condition_history_frames, d.condition_history_stride) != (32, 8, 25):
        raise ValueError("Reference contract is horizon32/history8/stride25")
    if cfg.get("lr_tail"):
        from .utils.lr_continuation import joint_multiplier

        if cfg.get("lr_continuation") or cfg.lr_scheduler_type != "cosine":
            raise ValueError("lr_tail requires cosine training without lr_continuation")
        joint_multiplier(0, cfg.warmup_steps, cfg.lr_tail)
        if not 0 < int(cfg.max_steps) <= int(cfg.lr_tail.end_step):
            raise ValueError("max_steps must be inside the joint LR schedule")
        if float(cfg.lr_min_ratio) != float(cfg.lr_tail.start_ratio):
            raise ValueError("lr_min_ratio must match lr_tail.start_ratio")
    if cfg.get("lr_continuation"):
        from .utils.lr_continuation import continuation_multiplier

        spec = cfg.lr_continuation
        continuation_multiplier(int(spec.start_step), spec)
        if not cfg.resume or cfg.resume_ckpt:
            raise ValueError("Continuation requires resume=<full training-state directory>")
        if not int(spec.start_step) < int(cfg.max_steps) <= int(spec.end_step):
            raise ValueError("max_steps must be inside the LR continuation interval")
    return cfg


def train(args):
    cfg = load_config(args.config, args.overrides)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    global_batch = cfg.batch_size * cfg.gradient_accumulation_steps * world
    video_only = cfg.train_mode == "video_only"
    summary = dict(
        train_mode=str(cfg.train_mode),
        video_initializer=str(cfg.model.dit_checkpoint_path),
        world_size=world,
        global_batch_size=global_batch,
        reference_global_batch_size=int(cfg.reference_global_batch_size),
        schedule_steps=int(cfg.max_steps),
        resume=cfg.resume,
    )
    if video_only:
        summary.update(
            training_video_shift=float(cfg.model.video_scheduler.train_shift),
            video_frames=25,
            condition_rgb_frames=9,
            future_rgb_frames=16,
        )
    else:
        summary.update(
            action_initializer=str(cfg.model.action_dit_pretrained_path),
            action_hidden=1024,
            action_ffn=4096,
            layers=40,
            training_action_shift=float(cfg.model.action_scheduler.train_shift),
            validation_action_shift=float(cfg.model.action_scheduler.infer_shift),
            benchmark_action_shift=1,
            lr_tail=dict(cfg.lr_tail) if cfg.get("lr_tail") else None,
            lr_continuation=(
                dict(cfg.lr_continuation) if cfg.get("lr_continuation") else None
            ),
        )
    print(json.dumps(summary, indent=2), flush=True)
    if args.dry_run:
        return
    if global_batch != cfg.reference_global_batch_size and not args.allow_batch_change:
        raise ValueError(
            f"Global batch differs from {cfg.reference_global_batch_size}. "
            "Set accumulation or explicitly use --allow-batch-change for a diagnostic."
        )
    if cfg.get("lr_continuation"):
        state_dir = Path(str(cfg.resume))
        state = json.loads((state_dir / "trainer_state.json").read_text())
        if not int(cfg.lr_continuation.start_step) <= int(state["global_step"]) < int(cfg.max_steps):
            raise ValueError("Restored trainer step is outside the requested continuation")
    required_paths = [
        cfg.model.model_id,
        cfg.model.dit_checkpoint_path,
        cfg.data.train.metadata_path,
    ]
    if not video_only:
        required_paths.extend(
            [
                cfg.model.action_dit_pretrained_path,
                cfg.data.train.pretrained_norm_stats,
                cfg.evaluation.metadata_path,
            ]
        )
    for value in required_paths:
        if not Path(value).exists():
            raise FileNotFoundError(value)
    out = Path(cfg.output_dir)
    # Rank zero owns the shared run configuration. A late worker must not reject
    # the config that rank zero has just created for this same distributed launch.
    if int(os.environ.get("RANK", "0")) == 0 and (out / "config.yaml").exists() and not cfg.resume:
        raise FileExistsError(
            "Output already contains a run; choose a new output_dir or set resume=..."
        )
    from .runtime import run_training

    run_training(cfg)


def server(args):
    from .robodojo import server_settings, validate_bundle

    settings = server_settings(args)
    evaluation_config = json.loads(Path(args.eval_config).read_text()) if args.eval_config else {}
    if evaluation_config.get("recovery_adapter_sha256") and not args.recovery_adapter:
        raise ValueError("This evaluation requires its pinned recovery adapter")
    root = Path(args.robodojo_root).resolve()
    bundle = Path(args.bundle).resolve()
    validate_bundle(bundle, settings["checkpoint_step"])
    config = root / "XPolicyLab/policy/RecoverWAM/deploy.yml"
    if not config.is_file():
        raise FileNotFoundError(
            "Install the adapter first: bash scripts/robodojo/install_adapter.sh --robodojo-root ..."
        )
    cmd = [
        sys.executable,
        "-u",
        str(root / "XPolicyLab/setup_policy_server.py"),
        "--config_path",
        str(config),
        "--overrides",
        f"port={args.port}",
        f"host={args.host}",
        f"checkpoint_path={bundle / 'action.pt'}",
        f"dataset_stats_path={bundle / 'dataset_stats.json'}",
        f"wan_model_dir={Path(args.wan).resolve()}",
        f"sigma_shift={settings['shift']}",
        f"num_inference_steps={settings['steps']}",
        f"seed={settings['seed']}",
        f"ckpt_name={settings['checkpoint_label']}",
        "action_horizon=32",
        "replan_steps=24",
    ]
    if args.recovery_adapter:
        adapter = Path(args.recovery_adapter).resolve()
        digest = hashlib.sha256(adapter.read_bytes()).hexdigest()
        base_identity = hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest()
        suffix = "-recovery-" + digest[:12]
        if args.eval_config:
            if evaluation_config.get("recovery_adapter_sha256") != digest or not settings["checkpoint_label"].endswith(suffix):
                raise ValueError("Evaluation config must pin the same recovery adapter hash and label")
            label = settings["checkpoint_label"]
        else:
            label = settings["checkpoint_label"] + suffix
        settings.update(checkpoint_label=label, adapter_sha256=digest,
                        base_manifest_sha256=base_identity)
        cmd.extend([f"recovery_adapter={adapter}", f"base_identity={base_identity}",
                    f"ckpt_name={label}"])
    print(json.dumps(dict(settings=settings, command=cmd), indent=2), flush=True)
    if args.dry_run:
        return
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        PYTHONPATH=str(root)
        + os.pathsep
        + str(root / "XPolicyLab")
        + os.pathsep
        + os.environ.get("PYTHONPATH", ""),
        RecoverWAM_TORCH_LOAD_MMAP="1",
    )
    subprocess.run(cmd, cwd=root, env=env, check=True)


def evaluate(args):
    config = json.loads(Path(args.config).read_text())
    from .robodojo import evaluation_settings

    settings = evaluation_settings(config)
    from .benchmark import scheduler_io as io

    tasks = io.load_tasks(Path(config["tasks_file"]))
    if (
        len(tasks) != 54
        or len(set(tasks)) != 54
        or sum(io.native_target(t, set(tasks)) for t in tasks) != 2100
    ):
        raise ValueError("Expected the complete native 54-entry/2100-episode inventory")
    if config.get("seed") != 1 or config.get("action_type") != "ee":
        raise ValueError("Reference evaluation requires seed1 and EE16")
    if not config.get("slots"):
        raise ValueError("Configure simulator GPU/policy-port slots")
    print(
        json.dumps(
            dict(
                tasks=54,
                official_cells=42,
                episodes=2100,
                slots=len(config["slots"]),
                run_id=config["run_id"],
                inference=settings,
                checkpoint_step=config.get("checkpoint_step"),
            ),
            indent=2,
        )
    )
    if args.dry_run:
        return
    for key in ("robodojo_root", "output_root", "tasks_file", "official_tasks_file", "sim_python"):
        config[key] = str(Path(config[key]).resolve())
    for task in tasks:
        if io.layout_count(config, task) < io.native_target(task, set(tasks)):
            raise ValueError(f"Missing seed1 layouts for {task}")
    output = Path(config["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    run_lock = (output / "eval.lock").open("a")
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous = output / "eval_config.json"
    if previous.exists() and io.read_json(previous) != config:
        raise ValueError(
            "Existing run has a different configuration; use a new run ID and output directory"
        )
    io.atomic_json(output / "eval_config.json", config)
    from .benchmark.packed import run

    if run(config, 15):
        raise RuntimeError("Evaluation has failed tasks; see summary.json")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "recoverwam.benchmark.audit",
            "--robodojo-root",
            config["robodojo_root"],
            "--tasks-file",
            config["tasks_file"],
            "--run-id",
            config["run_id"],
            "--ckpt-name",
            config["ckpt_name"],
            "--policy",
            "RecoverWAM",
            "--seed",
            "1",
            "--runner-summary",
            str(output / "summary.json"),
            "--output-json",
            str(output / "audit.json"),
            "--output-markdown",
            str(output / "audit.md"),
        ],
        check=True,
    )


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "libero":
        from .libero.cli import main as libero_main

        return libero_main(sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("libero", help="LIBERO training and standard/OOD/PRO evaluation")
    p = sub.add_parser("train")
    p.add_argument("--config", default="configs/robodojo/train_100k.yaml")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-batch-change", action="store_true")
    p.add_argument("overrides", nargs="*", help="OmegaConf key=value overrides")
    p = sub.add_parser("prepare")
    for name in ("metadata", "media-root", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--stats", default="configs/robodojo/normalization_ee16.json")
    p.add_argument("--holdout", default="configs/robodojo/holdout.json")
    p = sub.add_parser("convert", help="Convert official EE16 LeRobot v3 data to RecoverWAM episodes")
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--tasks", default="configs/robodojo/source_tasks.csv")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, help="Convert only the first N episodes for a smoke test")
    p = sub.add_parser("text-cache")
    p.add_argument("--data", required=True)
    p.add_argument("--wan", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p = sub.add_parser("init-action")
    p.add_argument("--video", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", default="configs/robodojo/train_100k.yaml")
    p = sub.add_parser("export")
    for name in ("checkpoint", "config", "stats", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--step", type=int, default=100000)
    p = sub.add_parser("install-adapter")
    p.add_argument("--robodojo-root", required=True)
    p.add_argument("--source", default="policy/RecoverWAM")
    p = sub.add_parser("server")
    for name in ("robodojo-root", "bundle", "wan"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--gpu", default="0")
    p.add_argument("--port", type=int, default=19960)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--shift", type=float)
    p.add_argument("--steps", type=int)
    p.add_argument("--expected-step", type=int, default=100000)
    p.add_argument("--eval-config", help="Use the exact inference settings and label of this evaluation")
    p.add_argument("--recovery-adapter", help="Load one global recovery LoRA, verified against the base manifest")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("eval-config", help="Write a portable local 54-task/2100-episode config")
    for name in ("robodojo-root", "sim-python", "run-id", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--output-root")
    p.add_argument("--template", default="configs/robodojo/eval_reference54.json")
    p.add_argument("--sim-gpu", type=int, default=1)
    p.add_argument("--policy-port", type=int, default=19960)
    p.add_argument("--step", type=int, default=100000)
    p.add_argument("--shift", type=float, default=1)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--recovery-adapter", help="Pin the recovery adapter hash and checkpoint label")
    p.add_argument("--sim-bin-dir", action="append", default=[])
    p = sub.add_parser("eval")
    p.add_argument("--config", default="configs/robodojo/eval_reference54.json")
    p.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command == "train":
        return train(args)
    if args.command == "server":
        return server(args)
    if args.command == "eval":
        return evaluate(args)
    if args.command == "eval-config":
        from .robodojo import write_eval_config

        return write_eval_config(args)
    if args.command == "install-adapter":
        destination = Path(args.robodojo_root) / "XPolicyLab/policy/RecoverWAM"
        shutil.copytree(args.source, destination)  # refuse to overwrite another adapter
        print(destination.resolve())
        return
    if args.command == "convert":
        from .convert_robodojo import convert

        result = convert(args.source, args.output, args.tasks, args.workers, args.limit)
    elif args.command == "prepare":
        from .data import prepare

        result = prepare(args.metadata, args.media_root, args.stats, args.output, args.holdout)
    elif args.command == "text-cache":
        from .data import text_cache

        result = text_cache(args.data, args.wan, args.device, args.batch_size)
    elif args.command == "init-action":
        from .weights import initialize_action

        result = initialize_action(args.video, args.config, args.output)
    elif args.command == "export":
        from .weights import export

        result = export(args.checkpoint, args.config, args.stats, args.output, args.step)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
