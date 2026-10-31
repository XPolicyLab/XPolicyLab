import logging
import os
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf

from .trainer import Wan22Trainer
from .utils.logging_config import get_logger, setup_logging
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


def _parse_hbridge_config(hbridge) -> tuple[bool, float, float]:
    """Parse the optional ``hbridge`` model-config block.

    Returns ``(hbridge_enabled, hbridge_bottom_ratio, hbridge_top_ratio)``.
    ``hbridge=None`` (the default in every config) yields ``(False, 0.25, 0.25)``,
    which preserves baseline behavior end-to-end.
    """
    if isinstance(hbridge, DictConfig):
        hbridge = OmegaConf.to_container(hbridge, resolve=True)
    if hbridge is None:
        hbridge = {}
    if not isinstance(hbridge, dict):
        raise ValueError(f"`hbridge` must be dict-like or None, got {type(hbridge)}")
    return (
        bool(hbridge.get("enabled", False)),
        float(hbridge.get("bottom_ratio", 0.25)),
        float(hbridge.get("top_ratio", 0.25)),
    )


def _normalize_flexpi_lists(
    dino_cam_patches, dino_cam_regions, pointmap_norm_bounds,
):
    if isinstance(dino_cam_patches, (DictConfig, list)):
        dino_cam_patches = OmegaConf.to_container(dino_cam_patches, resolve=True) if isinstance(dino_cam_patches, DictConfig) else dino_cam_patches
        dino_cam_patches = [tuple(p) for p in dino_cam_patches]
    if isinstance(dino_cam_regions, (DictConfig, list)):
        dino_cam_regions = OmegaConf.to_container(dino_cam_regions, resolve=True) if isinstance(dino_cam_regions, DictConfig) else dino_cam_regions
        dino_cam_regions = [tuple(r) for r in dino_cam_regions]
    if isinstance(pointmap_norm_bounds, DictConfig):
        pointmap_norm_bounds = OmegaConf.to_container(pointmap_norm_bounds, resolve=True)
    return dino_cam_patches, dino_cam_regions, pointmap_norm_bounds


def create_flexpi(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    dino_scheduler=None,
    pointmap_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    # DINO-specific
    dino_dim: int = 768,
    dino_model_name: str = "vit_base_patch16_dinov3.lvd1689m",
    dino_cam_patches=None,
    dino_cam_regions=None,
    dino_temporal_stride: int = 1,
    dino_pool_mode: str = "avg",
    freeze_dino_encoder: bool = True,
    dino_pixel_unshuffle: int = 0,
    dino_stride_keep_far: bool = False,
    dino_pred_x0: bool = False,
    # Pointmap-specific
    pointmap_norm_bounds=None,
    pointmap_max_depth_m: float = 2.0,
    pointmap_depth_vis_mode: str = "turbo",
    # Joint flags unique to this variant
    joint_video: bool = False,
    joint_dino: bool = False,
    joint_pointmap: bool = False,
    hbridge=None,
    composite_layout=None,
    composite_layout_slot_key_map=None,
    # Clean alternative to overriding `dino_cam_patches`/`dino_cam_regions`
    # directly. `dino_pool_factor=N` halves every cam's patch grid by N.
    # Default 1 = layout defaults; ignored when an explicit `dino_cam_patches`
    # override is also set.
    dino_pool_factor: int = 1,
    # Flex-joint config — per-sample randomization of stream presence + joint
    # flags at training time. When enabled, the trained model accepts runtime
    # ``joint_video / joint_dino / joint_pointmap`` overrides on ``infer_action``
    # to dispatch into any of 8 inference regimes (action-only, video+action,
    # dino+3d+action, full joint, etc.) without re-instantiation. Default off
    # is bit-identical to the legacy joint behavior.
    flex_joint=None,
    # Whether this run carries a pointmap (3D) stream. Requires a data config
    # that supplies depth; set False to train without it.
    enable_pointmap: bool = True,
):
    from .models.flexpi import FlexPi
    from .models.helpers.flex_joint import FlexJointConfig

    if isinstance(video_dit_config, DictConfig):
        video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)
    if not isinstance(video_dit_config, dict):
        raise ValueError(f"`video_dit_config` must resolve to a dict, got {type(video_dit_config)}")

    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if action_dit_config is None:
        action_dit_config = {}
    if not isinstance(action_dit_config, dict):
        raise ValueError(f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}")

    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if video_scheduler is None:
        video_scheduler = {}

    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if action_scheduler is None:
        raise ValueError("`action_scheduler` is required for FlexPi.")
    if not isinstance(action_scheduler, dict):
        raise ValueError(f"`action_scheduler` must be dict-like, got {type(action_scheduler)}")
    required_action_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_keys = required_action_scheduler_keys - set(action_scheduler.keys())
    if missing_keys:
        raise ValueError(f"`action_scheduler` missing required keys: {sorted(missing_keys)}.")

    if isinstance(dino_scheduler, DictConfig):
        dino_scheduler = OmegaConf.to_container(dino_scheduler, resolve=True)
    if dino_scheduler is None:
        dino_scheduler = {}
    if not isinstance(dino_scheduler, dict):
        raise ValueError(f"`dino_scheduler` must be dict-like, got {type(dino_scheduler)}")

    if isinstance(pointmap_scheduler, DictConfig):
        pointmap_scheduler = OmegaConf.to_container(pointmap_scheduler, resolve=True)
    if pointmap_scheduler is None:
        pointmap_scheduler = {}
    if not isinstance(pointmap_scheduler, dict):
        raise ValueError(f"`pointmap_scheduler` must be dict-like, got {type(pointmap_scheduler)}")

    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)
    if loss is None:
        loss = {}

    dino_cam_patches, dino_cam_regions, pointmap_norm_bounds = _normalize_flexpi_lists(
        dino_cam_patches, dino_cam_regions, pointmap_norm_bounds,
    )

    hbridge_enabled, hbridge_bottom_ratio, hbridge_top_ratio = _parse_hbridge_config(hbridge)

    # Resolve flex_joint config (Hydra DictConfig | dict | FlexJointConfig | None).
    if flex_joint is None:
        flex_joint_obj = FlexJointConfig()
    elif isinstance(flex_joint, FlexJointConfig):
        flex_joint_obj = flex_joint
    else:
        if isinstance(flex_joint, DictConfig):
            flex_joint_dict = OmegaConf.to_container(flex_joint, resolve=True)
        else:
            flex_joint_dict = dict(flex_joint)
        flex_joint_obj = FlexJointConfig(**flex_joint_dict)

    return FlexPi.from_wan22_pretrained(
        enable_pointmap=bool(enable_pointmap),
        joint_video=bool(joint_video),
        joint_dino=bool(joint_dino),
        joint_pointmap=bool(joint_pointmap),
        flex_joint=flex_joint_obj,
        dino_dim=int(dino_dim),
        dino_model_name=str(dino_model_name),
        dino_train_shift=float(dino_scheduler.get("train_shift", 5.0)),
        dino_infer_shift=float(dino_scheduler.get("infer_shift", 5.0)),
        dino_num_train_timesteps=int(dino_scheduler.get("num_train_timesteps", 1000)),
        loss_lambda_dino=float(loss.get("lambda_dino", 1.0)),
        dino_cam_patches=dino_cam_patches,
        dino_cam_regions=dino_cam_regions,
        dino_temporal_stride=int(dino_temporal_stride),
        dino_pool_mode=str(dino_pool_mode),
        freeze_dino_encoder=bool(freeze_dino_encoder),
        dino_pixel_unshuffle=int(dino_pixel_unshuffle),
        dino_stride_keep_far=bool(dino_stride_keep_far),
        dino_pred_x0=bool(dino_pred_x0),
        pointmap_norm_bounds=pointmap_norm_bounds,
        pointmap_max_depth_m=float(pointmap_max_depth_m),
        pointmap_train_shift=float(pointmap_scheduler.get("train_shift", 5.0)),
        pointmap_infer_shift=float(pointmap_scheduler.get("infer_shift", 5.0)),
        pointmap_num_train_timesteps=int(pointmap_scheduler.get("num_train_timesteps", 1000)),
        loss_lambda_pointmap=float(loss.get("lambda_pointmap", 1.0)),
        pointmap_depth_vis_mode=str(pointmap_depth_vis_mode),
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        hbridge_enabled=hbridge_enabled,
        hbridge_bottom_ratio=hbridge_bottom_ratio,
        hbridge_top_ratio=hbridge_top_ratio,
        composite_layout=composite_layout,
        composite_layout_slot_key_map=(
            OmegaConf.to_container(composite_layout_slot_key_map, resolve=True)
            if isinstance(composite_layout_slot_key_map, DictConfig)
            else composite_layout_slot_key_map
        ),
        dino_pool_factor=int(dino_pool_factor),
    )


def build_datasets(data_cfg: DictConfig):
    train_ds = instantiate(data_cfg.train)
    # Opt-out: when val_set_proportion is effectively zero, treat val as
    # "disabled" and reuse train_ds for eval — even if the data yaml carries
    # a val block. This preserves the pre-2026-05-21 behavior where the YAM
    # yaml had no val block and `val_ds = train_ds` was the implicit fallback.
    # Slurms set VAL_SET_PROPORTION=0.0 → wrapper forwards to data.train; the
    # yaml val block interpolates val_set_proportion from train, so this
    # short-circuit fires consistently.
    train_val_proportion = float(data_cfg.train.get("val_set_proportion", 0.0) or 0.0)
    val_disabled = train_val_proportion < 1e-6
    if data_cfg.get("val") is None or val_disabled:
        if val_disabled and data_cfg.get("val") is not None:
            logger.info(
                "val_set_proportion=%.3g < 1e-6; skipping separate val build, "
                "eval will reuse train_ds.", train_val_proportion,
            )
        val_ds = train_ds
    else:
        train_stats_path = data_cfg.train.get("pretrained_norm_stats")
        default_stats_path = os.path.join(misc.get_work_dir(), "dataset_stats.json")
        val_stats_path = data_cfg.val.get("pretrained_norm_stats")
        pretrained_norm_stats = val_stats_path or train_stats_path or default_stats_path
        logger.info("Building val dataset with pretrained_norm_stats: %s", pretrained_norm_stats)
        val_ds = instantiate(data_cfg.val, pretrained_norm_stats=pretrained_norm_stats)
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


def _maybe_swap_with_saved_config(cfg: DictConfig) -> DictConfig:
    """Optionally replace cfg with the saved phase-1 config when resuming.

    Controlled by `cfg.resume_config_source`:
      - "current" (default): no-op, return cfg as-is.
      - "saved": load <run_dir>/config.yaml from phase-1, replace cfg verbatim,
        then force-inject current launch-time fields (output_dir, wandb, resume)
        so the new launch writes to its own dir / wandb run while otherwise
        reproducing phase-1 exactly.
      - "saved_with_overrides": "saved" + apply current Hydra CLI leaf overrides
        on top (e.g. num_epochs=30). Group overrides (data=..., model=...,
        task=...) are skipped — they don't apply as leaf merges.

    The run_dir is inferred from `cfg.resume`: both
    `runs/.../checkpoints/state/step_NNNNNN` and
    `runs/.../checkpoints/weights/step_NNNNNN.pt` have the run_dir three
    parents up.
    """
    source = str(getattr(cfg, "resume_config_source", "current")).strip().lower()
    if source == "current":
        return cfg
    if source not in ("saved", "saved_with_overrides"):
        raise ValueError(
            "resume_config_source must be one of "
            "['current', 'saved', 'saved_with_overrides']; "
            f"got {source!r}"
        )
    if not cfg.resume:
        raise ValueError(
            f"resume_config_source={source!r} requires a non-empty cfg.resume "
            "(set RESUME or pass resume=<path>)."
        )

    resume_path = Path(str(cfg.resume))
    # state/step_NNNNNN/  → parents: state, checkpoints, run_dir, ...
    # weights/step_NNNNNN.pt → parents: weights, checkpoints, run_dir, ...
    run_dir = resume_path.parent.parent.parent
    saved_path = run_dir / "config.yaml"
    if not saved_path.exists():
        raise FileNotFoundError(
            f"resume_config_source={source!r} requires a saved phase-1 cfg at "
            f"{saved_path} (inferred from cfg.resume={cfg.resume!r}). "
            "If the saved cfg is missing, fall back to resume_config_source=current."
        )

    saved_cfg = OmegaConf.load(saved_path)
    OmegaConf.set_struct(saved_cfg, False)

    # Force-inject launch-time runtime fields from current cfg. The saved cfg
    # has resume=null (saved at phase-1 launch when nothing to resume from)
    # and points at phase-1's output_dir / wandb run; we want the new launch
    # to write to its own dir and have the user's current wandb settings.
    saved_cfg.resume = cfg.resume
    saved_cfg.output_dir = cfg.output_dir
    if "wandb" in cfg:
        saved_cfg.wandb = OmegaConf.to_container(cfg.wandb, resolve=True)

    # Carry the meta-flag through so any downstream readers see consistent
    # values (the saved cfg pre-dates this field for old runs).
    saved_cfg.resume_config_source = source

    if source == "saved_with_overrides":
        try:
            from hydra.core.hydra_config import HydraConfig
            cli_overrides = list(HydraConfig.get().overrides.task)
        except Exception as exc:
            logger.warning(
                "saved_with_overrides: could not access HydraConfig overrides "
                "(%s); falling back to plain saved-cfg load.", exc,
            )
            cli_overrides = []
        # Skip:
        #   - meta-control flags that don't belong in cfg
        #   - Hydra group overrides (data=..., model=..., task=...) — these
        #     would clobber the corresponding subtree as a string leaf, not
        #     re-trigger Hydra's group-config loading.
        _SKIP_LEAF = ("resume_config_source",)
        _SKIP_GROUP = ("data", "model", "task")
        kept = []
        for ov in cli_overrides:
            stripped = ov.lstrip("+")
            if "=" not in stripped:
                continue
            key, _ = stripped.split("=", 1)
            key = key.strip()
            if key in _SKIP_LEAF:
                continue
            if key in _SKIP_GROUP:
                logger.warning(
                    "saved_with_overrides: skipping group override %r "
                    "(group overrides only apply at compose time).", ov,
                )
                continue
            kept.append(stripped)
        if kept:
            try:
                override_cfg = OmegaConf.from_dotlist(kept)
                saved_cfg = OmegaConf.merge(saved_cfg, override_cfg)
                logger.info(
                    "saved_with_overrides: applied %d override(s) on top of "
                    "saved cfg: %s", len(kept), kept,
                )
            except Exception as exc:
                raise ValueError(
                    f"saved_with_overrides: failed to apply overrides {kept!r}: {exc}"
                ) from exc

    logger.info(
        "Resume cfg source: %s. Loaded saved cfg from %s. Forced-injected "
        "output_dir=%s, resume=%s, wandb (from current launch).",
        source, saved_path, saved_cfg.output_dir, saved_cfg.resume,
    )
    return saved_cfg


def run_training(cfg: DictConfig):
    setup_logging(
        log_level=logging.INFO,
        is_main_process=torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True,
    )
    cfg = _maybe_swap_with_saved_config(cfg)
    misc.register_work_dir(cfg.output_dir)
    config_payload = OmegaConf.to_container(cfg, resolve=True)
    with open(Path(cfg.output_dir) / "config.yaml", "w") as f:
        OmegaConf.save(config_payload, f)

    model_device = _resolve_train_device()
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    train_ds, val_ds = build_datasets(cfg.data)

    trainer = Wan22Trainer(
        cfg=cfg,
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
    )
    trainer.train()
