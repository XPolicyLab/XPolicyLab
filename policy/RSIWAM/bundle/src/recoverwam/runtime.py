import logging
import os
import inspect
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image
import numpy as np
from einops import repeat
from omegaconf import OmegaConf
from torch.utils.data import default_collate

from .trainer import JointTrainer
from .utils.logging_config import get_logger, setup_logging
from .utils.training_profiler import TrainingProfiler
from .utils.video_io import save_mp4
from .utils import misc

logger = get_logger(__name__)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    if not isinstance(mixed_precision, str):
        raise ValueError(f"`mixed_precision` must be str, got {type(mixed_precision)}")
    key = mixed_precision.strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def create_recoverwam_wan21_14b(
    model_id: str = "weights/Wan2.1-I2V-14B-480P",
    dit_checkpoint_path: str | None = None,
    tokenizer_model_id: str | None = None,
    video_dit_config=None,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    load_clip_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    action_chunk_size: int = 1,
    action_visible_video_frames: int = 1,
    action_only_first_frame_fast_path: bool = False,
    action_clean_first_frame_start_layer: int | None = None,
    action_clean_first_frame_timestep_cutoff: float | None = None,
    action_first_frame_kv_source: str = "clean_prefill",
    action_video_timestep_cutoff: float | None = None,
    video_lora=None,
    use_fixed_video_layers: int | None = None,
    use_video_tokens: bool = False,
    video_token_latent_dim: int = 2560,
    joint_denoising: bool = False,
    joint_denoising_mode_probs=None,
    joint_denoising_sampler: str = "euler",
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.wan21_14b.recoverwam import RecoverWAM

    if joint_denoising or video_lora and bool(video_lora.get("enabled", False)):
        raise ValueError(
            "This repository supports full joint training and cached-video action inference only."
        )
    if action_chunk_size != 1 or use_fixed_video_layers is not None or use_video_tokens:
        raise ValueError("Unsupported experimental architecture option")

    if isinstance(video_dit_config, DictConfig):
        video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)
    if video_dit_config is None:
        video_dit_config = {}
    if not isinstance(video_dit_config, dict):
        raise ValueError(f"`video_dit_config` must resolve to a dict, got {type(video_dit_config)}")

    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if action_dit_config is None:
        action_dit_config = {}
    if not isinstance(action_dit_config, dict):
        raise ValueError(
            f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}"
        )
    action_chunk_size = int(action_chunk_size)
    if action_chunk_size <= 0:
        raise ValueError(f"`action_chunk_size` must be positive, got {action_chunk_size}")
    if action_chunk_size > 1:
        action_dit_config = dict(action_dit_config)
        action_dit_config["action_dim"] = int(action_dit_config["action_dim"]) * action_chunk_size

    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if video_scheduler is None:
        video_scheduler = {}
    if not isinstance(video_scheduler, dict):
        raise ValueError(f"`video_scheduler` must be dict-like, got {type(video_scheduler)}")

    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if action_scheduler is None:
        raise ValueError("`action_scheduler` is required for RecoverWAM Wan2.1-14B.")
    if not isinstance(action_scheduler, dict):
        raise ValueError(f"`action_scheduler` must be dict-like, got {type(action_scheduler)}")
    required_action_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_keys = required_action_scheduler_keys - set(action_scheduler.keys())
    if missing_keys:
        raise ValueError(
            f"`action_scheduler` missing required keys: {sorted(missing_keys)}. "
            "Expected keys: train_shift, infer_shift, num_train_timesteps."
        )

    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)
    if loss is None:
        loss = {}
    if not isinstance(loss, dict):
        raise ValueError(f"`loss` must be dict-like, got {type(loss)}")

    if isinstance(video_lora, DictConfig):
        video_lora = OmegaConf.to_container(video_lora, resolve=True)
    if video_lora is None:
        video_lora = {}
    if not isinstance(video_lora, dict):
        raise ValueError(f"`video_lora` must be dict-like, got {type(video_lora)}")

    # Explicit checkpoint paths in the release YAML are relative to the checkout.
    if dit_checkpoint_path is not None and Path(dit_checkpoint_path).is_file():
        dit_checkpoint_path = str(Path(dit_checkpoint_path).resolve())

    return RecoverWAM.from_wan21_14b_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        dit_checkpoint_path=dit_checkpoint_path,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        load_clip_encoder=bool(load_clip_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        action_chunk_size=action_chunk_size,
        action_visible_video_frames=int(action_visible_video_frames),
        action_only_first_frame_fast_path=bool(action_only_first_frame_fast_path),
        action_clean_first_frame_start_layer=(
            None
            if action_clean_first_frame_start_layer is None
            else int(action_clean_first_frame_start_layer)
        ),
        action_clean_first_frame_timestep_cutoff=(
            None
            if action_clean_first_frame_timestep_cutoff is None
            else float(action_clean_first_frame_timestep_cutoff)
        ),
        action_first_frame_kv_source=str(action_first_frame_kv_source),
        action_video_timestep_cutoff=(
            None if action_video_timestep_cutoff is None else float(action_video_timestep_cutoff)
        ),
        video_lora_config=video_lora,
        use_fixed_video_layers=(
            None if use_fixed_video_layers is None else int(use_fixed_video_layers)
        ),
        use_video_tokens=bool(use_video_tokens),
        video_token_latent_dim=int(video_token_latent_dim),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        video_train_sampling_strategy=str(
            video_scheduler.get("train_sampling_strategy", "shifted_uniform")
        ),
        video_train_beta_alpha=float(video_scheduler.get("beta_alpha", 7.0)),
        video_train_beta_beta=float(video_scheduler.get("beta_beta", 1.0)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        video_lora_joint_video_loss=bool(loss.get("video_lora_joint_video_loss", False)),
        joint_denoising=bool(joint_denoising),
        joint_denoising_mode_probs=(
            OmegaConf.to_container(joint_denoising_mode_probs, resolve=True)
            if isinstance(joint_denoising_mode_probs, DictConfig)
            else joint_denoising_mode_probs
        ),
        joint_denoising_sampler=str(joint_denoising_sampler),
    )


def build_datasets(data_cfg: DictConfig):
    train_ds = instantiate(data_cfg.train)
    if data_cfg.get("val") is None:
        val_ds = train_ds
    else:
        train_uses_relative_action = bool(data_cfg.train.get("use_relative_action", False))
        val_uses_relative_action = bool(
            data_cfg.val.get("use_relative_action", train_uses_relative_action)
        )
        train_stats_path = data_cfg.train.get(
            "pretrained_norm_stats_relative"
            if train_uses_relative_action
            else "pretrained_norm_stats"
        )
        default_stats_path = os.path.join(misc.get_work_dir(), "dataset_stats.json")
        val_stats_key = (
            "pretrained_norm_stats_relative"
            if val_uses_relative_action
            else "pretrained_norm_stats"
        )
        val_stats_path = data_cfg.val.get(val_stats_key)
        pretrained_norm_stats = val_stats_path or train_stats_path
        if not pretrained_norm_stats:
            processor = getattr(train_ds, "processor", None)
            normalizer = getattr(processor, "normalizer", None) if processor is not None else None
            pretrained_norm_stats = (
                getattr(normalizer, "stats", None) if normalizer is not None else None
            )
        if pretrained_norm_stats is None:
            pretrained_norm_stats = default_stats_path
        log_stats = (
            "in-memory stats" if isinstance(pretrained_norm_stats, dict) else pretrained_norm_stats
        )
        logger.info("Building val dataset with %s: %s", val_stats_key, log_stats)
        val_ds = instantiate(data_cfg.val, **{val_stats_key: pretrained_norm_stats})
    return train_ds, val_ds


def _resolve_train_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    device_count = torch.cuda.device_count()
    if device_count <= 1:
        return "cuda:0"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank < 0 or local_rank >= device_count:
        return "cuda:0"
    return f"cuda:{local_rank}"


def run_training(cfg: DictConfig):
    from .preflight import (
        validate_batch_contract,
        validate_preflight_config,
    )

    preflight_settings = validate_preflight_config(cfg)
    if preflight_settings is not None:
        expected_mode = {
            "video": "video_only",
            "action": "action_only",
            "alternating": "alternating",
            "joint": "joint",
        }[preflight_settings["stage"]]
        if str(cfg.train_mode).strip().lower() != expected_mode:
            raise ValueError(
                f"Preflight stage={preflight_settings['stage']} requires "
                f"train_mode={expected_mode}, got {cfg.train_mode}"
            )

    profiler = TrainingProfiler.from_config(cfg)
    with profiler.measure_startup("runtime_setup_ms"):
        output_dir = Path(str(cfg.output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(
            log_level=logging.INFO,
            is_main_process=torch.distributed.get_rank() == 0
            if torch.distributed.is_initialized()
            else True,
            log_file=output_dir / "train.log",
        )
        misc.register_work_dir(cfg.output_dir)
        config_payload = OmegaConf.to_container(cfg, resolve=True)
        if int(os.environ.get("RANK", "0")) == 0:
            with open(output_dir / "config.yaml", "w") as f:
                OmegaConf.save(config_payload, f)

    model_device = _resolve_train_device()
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    # Seed before constructing fresh Action/proprio projections on every rank.
    from accelerate.utils import set_seed

    set_seed(int(cfg.seed))
    with profiler.measure_startup("model_instantiate_ms", cuda=True):
        model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    with profiler.measure_startup("dataset_build_ms"):
        train_ds, val_ds = build_datasets(cfg.data)
    if preflight_settings is not None:
        with profiler.measure_startup("preflight_batch_ms", cuda=True):
            observed_contract = validate_batch_contract(
                default_collate([train_ds[0]]),
                preflight_settings["stage"],
                video_size=preflight_settings["video_size"],
                **preflight_settings["contract_kwargs"],
            )
        logger.info(
            "Preflight real-batch contract validated: %s",
            observed_contract,
        )
    evaluator = None
    evaluation_cfg = cfg.get("evaluation")
    if evaluation_cfg is not None:
        evaluation_kind = str(evaluation_cfg.get("kind", "")).strip().lower()
        if evaluation_kind == "heldout_action":
            from .evaluation import HeldoutActionEvaluator

            # Same dataset contract as training (window, history, stats, processor),
            # read from the held-out metadata with training-time augmentation and task
            # balancing off, so every window is a fixed, deterministic sample.
            heldout_ds = instantiate(
                cfg.data.train,
                metadata_path=str(evaluation_cfg.metadata_path),
                is_training_set=False,
                val_set_proportion=0.0,
                balance_column=None,
                balance_weight_overrides=None,
            )
            heldout_steps = evaluation_cfg.get("num_inference_steps", 10)
            heldout_samplers = evaluation_cfg.get("samplers")
            evaluator = HeldoutActionEvaluator(
                dataset=heldout_ds,
                windows_per_episode=int(evaluation_cfg.get("windows_per_episode", 16)),
                num_inference_steps=(
                    int(heldout_steps)
                    if isinstance(heldout_steps, (int, float, str))
                    else [int(steps) for steps in heldout_steps]
                ),
                samplers=(None if heldout_samplers is None else [str(x) for x in heldout_samplers]),
                seed=int(evaluation_cfg.get("seed", 42)),
                task_column=str(evaluation_cfg.get("task_column", "task_name")),
                output_path=output_dir / "eval" / "heldout_action.jsonl",
            )
            logger.info(
                "Held-out action evaluation: %d windows from %d episodes (%s)",
                len(evaluator.items),
                heldout_ds.num_trajectories,
                evaluation_cfg.metadata_path,
            )
        elif evaluation_kind:
            raise ValueError(f"Unsupported evaluation kind: {evaluation_kind}")

    with profiler.measure_startup("trainer_init_total_ms", cuda=True):
        trainer_type = JointTrainer
        if cfg.train_mode == "video_only":
            from .video_training import trainer_class

            trainer_type = trainer_class()
        trainer = trainer_type(
            cfg=cfg,
            model=model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            evaluator=evaluator,
            profiler=profiler,
        )
    trainer.train()
