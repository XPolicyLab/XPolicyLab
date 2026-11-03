"""RecoverWAM LIBERO two-stage training and paper-protocol evaluations."""

import argparse
import json
import os
from pathlib import Path
from omegaconf import OmegaConf


def validate_training_config(cfg, require_portable_target=True):
    data, model = cfg.data.train, cfg.model
    if (
        list(data.video_size),
        int(data.num_frames),
        int(data.action_horizon),
        int(data.action_video_freq_ratio),
        data.concat_multi_camera,
    ) != ([224, 448], 65, 32, 4, "horizontal"):
        raise ValueError(
            "LIBERO release requires horizontal224x448, 17 video frames, action horizon32"
        )
    if (
        int(data.context_len) != 128
        or float(data.val_set_proportion) != 0
        or len(data.dataset_dirs) != 4
    ):
        raise ValueError("Expected four-suite natural frame mixture and cached context128")
    if data.processor.norm_default_mode != "min/max":
        raise ValueError("LIBERO reference normalization is min_max")
    if cfg.train_mode == "action_only":
        a = model.action_dit_config
        if (a.action_dim, a.hidden_dim, a.ffn_dim, a.num_layers, model.proprio_dim) != (
            7,
            512,
            2048,
            40,
            8,
        ):
            raise ValueError("Expected Action1B (hidden512) and proprio8")
        if (
            model.action_visible_video_frames,
            model.action_chunk_size,
            model.video_dit_config.video_attention_mask_mode,
        ) != (3, 1, "bidirectional"):
            raise ValueError(
                "Expected 5 bidirectional video latent frames with direct action visibility of first3"
            )
        if (
            model.video_scheduler.train_sampling_strategy != "pure_noise"
            or not model.action_only_first_frame_fast_path
        ):
            raise ValueError(
                "Action stage requires frozen pure-noise video and first-frame VAE fast path"
            )
        if (
            float(model.action_scheduler.train_shift),
            float(model.action_scheduler.infer_shift),
        ) != (5, 5):
            raise ValueError("LIBERO action shift is 5")
        if cfg.checkpoint_mode != "action_only":
            raise ValueError("Action stage saves compact checkpoints")
    elif cfg.train_mode == "video_only":
        if "action_dit_config" in model or cfg.checkpoint_mode != "full":
            raise ValueError("Video stage must instantiate video only")
        if (
            model.video_scheduler.train_sampling_strategy,
            float(model.video_scheduler.beta_alpha),
            float(model.video_scheduler.beta_beta),
        ) != ("beta", 7, 1):
            raise ValueError("Video training requires Beta(7,1)")
    else:
        raise ValueError("Unsupported LIBERO training mode")
    if require_portable_target and not model._target_.startswith("recoverwam.libero.runtime."):
        raise ValueError("Use the dedicated recoverwam.libero model factory")
    return cfg


def train(args):
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))
    validate_training_config(cfg)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    gbs = int(cfg.batch_size) * int(cfg.gradient_accumulation_steps) * world
    print(
        json.dumps(
            dict(
                stage=cfg.train_mode,
                global_batch=gbs,
                reference_global_batch=int(cfg.reference_global_batch_size),
                max_steps=int(cfg.max_steps),
                normalization="min_max",
                action_shift=5,
                video_size=[224, 448],
            ),
            indent=2,
        )
    )
    if args.dry_run:
        return
    if gbs != int(cfg.reference_global_batch_size) and not args.allow_batch_change:
        raise ValueError(
            "Batch differs from paper; use --allow-batch-change only for a labelled diagnostic"
        )
    from .trainer import validate_preflight_config

    validate_preflight_config(cfg)
    for path in [
        cfg.model.model_id,
        cfg.model.dit_checkpoint_path,
        cfg.data.train.pretrained_norm_stats,
        *cfg.data.train.dataset_dirs,
        cfg.data.train.text_embedding_cache_dir,
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(path)
    if (
        cfg.train_mode == "action_only"
        and not cfg.resume_ckpt
        and not Path(cfg.model.action_dit_pretrained_path).is_file()
    ):
        raise FileNotFoundError(cfg.model.action_dit_pretrained_path)
    if (Path(cfg.output_dir) / "config.yaml").exists() and not cfg.resume:
        raise FileExistsError(
            "Output contains an existing run; set resume=... or choose a fresh output_dir"
        )
    from .runtime import run_training

    run_training(cfg)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--config", required=True)
    t.add_argument("--dry-run", action="store_true")
    t.add_argument("--allow-batch-change", action="store_true")
    t.add_argument("overrides", nargs="*")
    e = sub.add_parser("eval")
    e.add_argument("--config", required=True)
    e.add_argument("--gpus", default="0")
    e.add_argument("--dry-run", action="store_true")
    e.add_argument("--resume", action="store_true")
    e.add_argument("overrides", nargs="*")
    a = sub.add_parser("audit")
    a.add_argument("--config", required=True)
    a.add_argument("overrides", nargs="*")
    i = sub.add_parser("init-action")
    i.add_argument("--video", required=True)
    i.add_argument("--config", default="configs/libero/train_action.yaml")
    i.add_argument("--output", required=True)
    c = sub.add_parser("text-cache")
    c.add_argument("--config", default="configs/libero/train_action.yaml")
    c.add_argument("--device", default="cuda")
    c.add_argument("--batch-size", type=int, default=16)
    c.add_argument("overrides", nargs="*")
    d = sub.add_parser("data-check")
    d.add_argument("--config", default="configs/libero/train_action.yaml")
    d.add_argument("overrides", nargs="*")
    pro = sub.add_parser("prepare-pro")
    pro.add_argument("--runtime", default="third_party/libero_pro_runtime")
    pro.add_argument("--assets", required=True)
    args = p.parse_args(argv)
    if args.command == "prepare-pro":
        from .prepare_pro import prepare

        print(json.dumps(prepare(args.runtime, args.assets), indent=2))
        return
    if args.command == "train":
        return train(args)
    if args.command == "init-action":
        validate_training_config(OmegaConf.load(args.config))
        from recoverwam.weights import initialize_action

        print(initialize_action(args.video, args.config, args.output, expected_step=10000))
        return
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))
    if args.command in {"data-check", "text-cache"}:
        validate_training_config(cfg)
        from .data import check_data, text_cache

        print(
            json.dumps(
                check_data(cfg)
                if args.command == "data-check"
                else text_cache(cfg, args.device, args.batch_size),
                indent=2,
            )
        )
        return
    from .evaluate import run, audit

    if args.command == "eval":
        return run(cfg, [int(x) for x in args.gpus.split(",")], args.dry_run, args.resume)
    report = audit(cfg)
    root = Path(cfg.output_dir)
    if (
        not (root / "full_run.status").is_file()
        or (root / "full_run.status").read_text().strip() != "0"
    ):
        report["issues"].append("full_run.status is not zero")
        report["complete"] = False
    print(json.dumps(report, indent=2))
    if not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
