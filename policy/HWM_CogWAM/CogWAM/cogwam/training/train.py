"""Training entry point: ``python -m cogwam.training.train --config_yaml <cfg>``.

Order matters here and is load-bearing:

1. The YAML is merged with CLI dotlist overrides *before* the Accelerator is
   built, because the DeepSpeed config leaves ``gradient_accumulation_steps``
   as ``"auto"`` and Accelerate would otherwise resolve it to its own default
   of 1 without ever seeing the run's value.
2. The frozen recipe contract is checked before a single parameter is
   allocated, so a config drift costs seconds instead of a GPU-hour.
3. The model is constructed under one common seed on every rank when
   ``trainer.seed_before_model_init`` is set; the trainer later restores the
   rank-offset runtime seed for data and noise.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import GradientAccumulationPlugin, set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from cogwam.data import build_dataloader, build_event_memory_dataloader
from cogwam.models.base import build_model
from cogwam.recipe import RECIPE_PROFILE, validate_recipe, validate_runtime_assets
from cogwam.training.config import apply_config_compat
from cogwam.training.config_tracker import wrap_config
from cogwam.training.optim import setup_optimizer_and_scheduler
from cogwam.training.trainer import CogWAMTrainer
from cogwam.training.utils import normalize_dotlist_args

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = get_logger(__name__)


def configure_torch_runtime() -> None:
    """Configure optional CUDA speed knobs controlled by launch env vars."""
    allow_tf32 = os.getenv("COGWAM_ALLOW_TF32", "0").strip().lower() in {"1", "true", "yes", "on"}
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
    if allow_tf32:
        precision = os.getenv("COGWAM_FLOAT32_MATMUL_PRECISION", "high")
        torch.set_float32_matmul_precision(precision)
        logger.info("CogWAM torch runtime: TF32 enabled, float32_matmul_precision=%s", precision)


def build_accelerator(cfg) -> Accelerator:
    """Build Accelerate/DeepSpeed with the accumulation value from this run."""

    accumulation_steps = int(cfg.trainer.get("gradient_accumulation_steps", 1))
    if accumulation_steps <= 0:
        raise ValueError(f"trainer.gradient_accumulation_steps must be positive, got {accumulation_steps}.")
    clipping = cfg.trainer.get("gradient_clipping", None)
    plugin = DeepSpeedPlugin(
        gradient_accumulation_steps=accumulation_steps,
        gradient_clipping=float(clipping) if clipping is not None else None,
    )
    # DeepSpeed ZeRO-2 partitions gradients and explicitly rejects
    # ``engine.no_sync()``. Accelerate normally enters no_sync on non-boundary
    # micro-batches, which crashes before the first E2E forward when
    # gradient_accumulation_steps > 1. Keep the same numerical accumulation,
    # but synchronize each micro-batch so DeepSpeed can own accumulation and
    # optimizer-boundary handling without the unsupported context.
    zero_no_sync_guard = accumulation_steps > 1
    accumulation_plugin = GradientAccumulationPlugin(
        num_steps=accumulation_steps,
        sync_each_batch=zero_no_sync_guard,
    )
    result = Accelerator(
        deepspeed_plugin=plugin,
        gradient_accumulation_plugin=accumulation_plugin,
    )
    if int(result.gradient_accumulation_steps) != accumulation_steps:
        raise RuntimeError(
            "Accelerate ignored trainer.gradient_accumulation_steps: "
            f"requested {accumulation_steps}, active {result.gradient_accumulation_steps}."
        )
    deepspeed_accumulation = plugin.get_value("gradient_accumulation_steps")
    if deepspeed_accumulation != "auto" and int(deepspeed_accumulation) != accumulation_steps:
        raise RuntimeError(
            "DeepSpeed accumulation disagrees with the training YAML: "
            f"requested {accumulation_steps}, active {deepspeed_accumulation}."
        )
    accumulation_kwargs = result.gradient_state.plugin_kwargs
    if zero_no_sync_guard and not bool(accumulation_kwargs.get("sync_each_batch", False)):
        raise RuntimeError(
            "DeepSpeed ZeRO accumulation requires GradientAccumulationPlugin(sync_each_batch=True) "
            "to avoid the unsupported engine.no_sync() path."
        )
    result.print(result.state)
    result.print(
        f"[train] gradient_accumulation_steps={result.gradient_accumulation_steps} "
        f"deepspeed={deepspeed_accumulation} "
        f"zero_no_sync_guard={str(zero_no_sync_guard).lower()}"
    )
    return result


def setup_directories(cfg) -> Path:
    """Create output directory and checkpoint directory."""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def prepare_data(cfg, accelerator, output_dir) -> tuple[DataLoader, DataLoader | None]:
    """Prepare the physical stream and the event-memory semantic NTP stream."""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    event_cfg = cfg.datasets.vla_data.get("text_annotations", {}).get("event_memory", {})
    semantic_train_dataloader = None
    if bool(event_cfg.get("enabled", False)):
        logger.info("Creating event-memory semantic NTP stream (phase-aligned 2/2/2 sampler)")
        semantic_train_dataloader = build_event_memory_dataloader(cfg, output_dir=output_dir)

    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()
    return vla_train_dataloader, semantic_train_dataloader


def main(cfg, accelerator: Accelerator) -> None:
    logger.info("CogWAM Training :: Warming Up")

    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    reproduction_profile = str(cfg.framework.get("reproduction_profile", "") or "")
    if reproduction_profile != RECIPE_PROFILE:
        raise ValueError(
            "This repository reproduces exactly one recipe; expected "
            f"framework.reproduction_profile={RECIPE_PROFILE!r}, got {reproduction_profile!r}"
        )
    contract = validate_recipe(cfg)
    runtime_contract = validate_runtime_assets(cfg)
    logger.info("Frozen reproduction contract verified: %s", contract)

    output_dir = setup_directories(cfg=cfg)
    if accelerator.is_main_process:
        (output_dir / "cogwam_runtime_audit.json").write_text(
            json.dumps({"contract": contract, "assets": runtime_contract}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if bool(cfg.trainer.get("seed_before_model_init", False)):
        # Use one common construction seed on every rank so optional
        # zero-initialized modules can be proven not to perturb the baseline
        # policy initialization.  prepare_training later restores the
        # rank-offset runtime seed used for data and flow-matching noise.
        construction_seed = int(getattr(cfg, "seed", 42))
        set_seed(construction_seed)
        logger.info("Deterministic model construction seed=%d", construction_seed)

    model = build_model(cfg)
    vla_train_dataloader, semantic_train_dataloader = prepare_data(
        cfg=cfg, accelerator=accelerator, output_dir=output_dir
    )
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=model, cfg=cfg)

    trainer = CogWAMTrainer(
        cfg=cfg,
        model=model,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        semantic_train_dataloader=semantic_train_dataloader,
    )

    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="configs/cogwam_robodojo_h25_eventmem_dino_multilayer_50k.yaml",
        help="Path to YAML config",
    )
    args, overrides = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    cli_cfg = OmegaConf.from_dotlist(normalize_dotlist_args(overrides))
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # Normalise legacy YAML keys into the current `version_id == "0.21"` schema.
    # This is idempotent and does not modify framework class signatures.
    cfg = apply_config_compat(cfg)

    # Store source config path for later copying to the output dir.
    cfg.config_yaml = args.config_yaml
    return cfg


if __name__ == "__main__":
    config = parse_config()
    # Must happen after YAML/CLI merging: the DeepSpeed config uses "auto" and
    # therefore needs the run-specific accumulation value at construction.
    accelerator = build_accelerator(config)
    configure_torch_runtime()
    main(config, accelerator)
