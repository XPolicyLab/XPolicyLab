import logging
import os
from pathlib import Path

import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from .cache import CacheManager, CacheMode
from .model.factory import create_wam
from .trainer import WAMTrainer
from .training.config_validation import validate_training_config
from .training.infra import configure_compiler_cache
from .training.distributed import deepspeed_zero_stage, model_init_device
from .utils import misc
from .utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def normalize_video_latent_cache_entry_mode(cfg: DictConfig) -> str:
    """Validate the global cache switch without mutating dataset config."""

    return CacheMode.parse(cfg.get("video_latent_cache", "off")).value


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


def build_datasets(data_cfg: DictConfig):
    train_ds = instantiate(data_cfg.train)
    if data_cfg.get("val") is None:
        val_ds = train_ds
    else:
        train_stats_path = data_cfg.train.get("pretrained_norm_stats")
        default_stats_path = os.path.join(misc.get_work_dir(), "dataset_stats.json")
        val_cfg = data_cfg.val
        if val_cfg.get("_target_") is None:
            # Some task configs only override validation knobs (e.g. history sizes).
            # Use the train dataset config as a template so Hydra still instantiates
            # an actual Dataset instead of returning the partial DictConfig.
            val_cfg = OmegaConf.merge(data_cfg.train, val_cfg)
            val_cfg.is_training_set = False
        val_stats_path = val_cfg.get("pretrained_norm_stats")
        pretrained_norm_stats = val_stats_path or train_stats_path or default_stats_path
        if hasattr(train_ds, "make_validation_dataset"):
            logger.info("Building val dataset by reusing train dataset metadata.")
            val_ds = train_ds.make_validation_dataset(
                val_cfg, pretrained_norm_stats=pretrained_norm_stats
            )
        else:
            logger.info(
                "Building val dataset with pretrained_norm_stats: %s",
                pretrained_norm_stats,
            )
            val_ds = instantiate(val_cfg, pretrained_norm_stats=pretrained_norm_stats)
    return train_ds, val_ds


def run_training(
    cfg: DictConfig,
    *,
    cache_manager: CacheManager | None = None,
):
    setup_logging(
        log_level=logging.INFO,
        is_main_process=torch.distributed.get_rank() == 0
        if torch.distributed.is_initialized()
        else True,
    )
    cache_manager = cache_manager or CacheManager.from_config(cfg)
    if cache_manager.mode is CacheMode.GENERATE:
        raise ValueError(
            "Cache generate mode is offline; dispatch it through scripts/train.py."
        )
    validate_training_config(cfg)
    configure_compiler_cache(cfg)
    misc.register_work_dir(cfg.output_dir)
    config_payload = OmegaConf.to_container(cfg, resolve=True)
    with open(Path(cfg.output_dir) / "config.yaml", "w") as f:
        OmegaConf.save(config_payload, f)

    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    accelerator = Accelerator(
        gradient_accumulation_steps=int(cfg.gradient_accumulation_steps),
        mixed_precision=mixed_precision,
        step_scheduler_with_optimizer=False,
    )
    runtime_device = torch.device(accelerator.device)
    if runtime_device.type == "cuda":
        if runtime_device.index is None:
            runtime_device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(runtime_device)
    init_device = model_init_device(accelerator)
    zero_stage = deepspeed_zero_stage(accelerator)
    if zero_stage == 3 and str(cfg.model.get("mot_compile_mode", "off")) != "off":
        raise ValueError("MoT layer compilation supports ZeRO stages 0, 1 and 2. Use the eager model with ZeRO-3.")
    logger.info(
        "Resolved training devices: init=%s runtime=%s zero_stage=%d local_rank=%s "
        "cuda_visible_devices=%s current_cuda=%s",
        init_device,
        runtime_device,
        zero_stage,
        os.environ.get("LOCAL_RANK", "0"),
        os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        torch.cuda.current_device() if torch.cuda.is_available() else "cpu",
    )
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    train_ds, val_ds = build_datasets(cfg.data)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=str(init_device))
    cache_runtime = cache_manager.prepare_training(
        model=model,
        datasets={"train": train_ds, "validation": val_ds},
        distributed=accelerator,
    )

    trainer = WAMTrainer(
        cfg=cfg,
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        accelerator=accelerator,
        cache_runtime=cache_runtime,
    )
    trainer.train()
