from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import torch

from policy_core.model.WAM_policy.task import WAM_policyTaskFlow, WAM_policyTaskModel
from policy_core.utils.WAM_policy import import_custom_class
from policy_core.utils.WAM_policy.data_utils import get_text_conditions
from policy_core.utils.WAM_policy.WAM_policy_config import (
    as_dataset_kwargs,
    get_action_model_kwargs,
    get_action_model_path,
    get_video_model_kwargs,
    get_video_model_path,
)
from policy_core.utils.WAM_policy.model_utils import (
    load_condition_models,
    load_diffusion_model,
    load_latent_models,
    load_vae_models,
)
from policy_core.utils.WAM_policy.vae_normal_utils import LatentStatsTracker
from policy_core.utils.WAM_policy.video_sparse_utils import (
    normalize_sparse_downsample_factor,
    normalize_sparse_spatial_rope_mode,
    validate_mixed_video_sparse_mode,
)
from policy_core.utils.common_utils import get_device

logger = logging.getLogger(__name__)


def resolve_multi_view_schedule_cfg(cfg: Any) -> Mapping[str, Any] | None:
    schedule_cfg = cfg.data.multi_view_schedule
    if schedule_cfg is None:
        return None
    if not isinstance(schedule_cfg, dict):
        raise TypeError("multi_view_schedule must be a mapping in the config file")
    if not schedule_cfg.get("enabled", False):
        return None
    if cfg.train.train_steps is None:
        raise ValueError(
            "multi_view_schedule requires train_steps to be set explicitly"
        )

    modes = schedule_cfg.get("modes", [])
    schedule = [int(span) for span in schedule_cfg.get("schedule", [])]
    if not modes:
        raise ValueError("multi_view_schedule.modes must not be empty")
    if len(modes) != len(schedule):
        raise ValueError(
            "multi_view_schedule.modes and schedule must have the same length"
        )
    return schedule_cfg


def _get_video_transformer_patch_sizes(video_diffusion_model) -> tuple[int, int]:
    if video_diffusion_model is None:
        return 1, 1

    patch_size = getattr(video_diffusion_model, "patch_size", None)
    patch_size_t = getattr(video_diffusion_model, "patch_size_t", None)
    config = getattr(video_diffusion_model, "config", None)
    if patch_size is None and config is not None:
        patch_size = getattr(config, "patch_size", 1)
    if patch_size_t is None and config is not None:
        patch_size_t = getattr(config, "patch_size_t", 1)

    if isinstance(patch_size, (list, tuple)):
        if len(patch_size) == 3:
            patch_size_t = patch_size[0]
            patch_size = patch_size[1]
        elif len(patch_size) == 2:
            patch_size = patch_size[0]

    if patch_size_t is None:
        patch_size_t = 1

    return int(patch_size), int(patch_size_t)


def build_WAM_policy_task_flow(
    *,
    cfg: Any,
    cotrain_schedule: Mapping[str, int] | None,
    transforms: Sequence[Any] = (),
    output_transforms: Sequence[Any] = (),
) -> WAM_policyTaskFlow:
    if cotrain_schedule:
        raise ValueError("WAM_policy runtime supports inference only, not cotrain schedules")
    if cfg.vlm_rm or cfg.dsrl:
        raise ValueError("Reward-model and DSRL training are not supported by the inference runtime")
    logger.info("Initializing models")
    device = get_device()
    dtype = (
        torch.bfloat16 if cfg.train.mixed_precision == "bf16" else torch.float32
    )


    _tk_path = cfg.model.tokenizer_class_path
    if _tk_path is not None and str(_tk_path).lower() != "none":
        tokenizer_class = import_custom_class(cfg.model.tokenizer_class,
            cfg.model.tokenizer_class_path,
        )
        textenc_class = import_custom_class(cfg.model.textenc_class,
            cfg.model.textenc_class_path,
        )
        cond_models = load_condition_models(
            tokenizer_class,
            textenc_class, cfg.model.pretrained_model_name_or_path
            if not hasattr(cfg, "tokenizer_pretrained_model_name_or_path")
            else cfg.tokenizer_pretrained_model_name_or_path,
            load_weights=True,
        )
        tokenizer, text_encoder = (
            cond_models["tokenizer"],
            cond_models["text_encoder"],
        )
        text_encoder = text_encoder.to(device, dtype=dtype).eval()
        uncond_prompt_embeds = get_text_conditions(
            tokenizer, text_encoder, prompt=""
        )["prompt_embeds"]
    else:
        print("Got None Tokenizer")
        tokenizer = None
        text_encoder = None
        uncond_prompt_embeds = None

    vae_class = import_custom_class(cfg.model.vae_class, cfg.model.vae_class_path
    )
    if cfg.model.vae_path:
        vae = (
            load_vae_models(
                vae_class, cfg.model.vae_path,
                load_weights=cfg.model.vae_load_weights,
            )
            .to(device, dtype=dtype)
            .eval()
        )
    else:
        if not cfg.model.vae_load_weights:
            raise ValueError(
                "vae_load_weights=false requires vae_path to construct the VAE from config."
            )
        vae = (
            load_latent_models(vae_class, cfg.model.pretrained_model_name_or_path)["vae"]
            .to(device, dtype=dtype)
            .eval()
        )
    if hasattr(vae, "latents_mean") and isinstance(vae.latents_mean, list):
        vae.latents_mean = torch.FloatTensor(vae.latents_mean)
    if hasattr(vae, "latents_std") and isinstance(vae.latents_std, list):
        vae.latents_std = torch.FloatTensor(vae.latents_std)
    if vae is not None:
        if cfg.model.enable_slicing:
            vae.enable_slicing()
        if cfg.model.enable_tiling:
            vae.enable_tiling()
    spatial_down_ratio = vae.spatial_compression_ratio
    temporal_down_ratio = vae.temporal_compression_ratio
    logger.info(f"SPATIAL_DOWN_RATIO of VAE :{spatial_down_ratio}")
    logger.info(f"TEMPORAL_DOWN_RATIO of VAE :{temporal_down_ratio}")

    vae.vae_encoding_mode = cfg.model.vae_encoding_mode

    _train_cfg = as_dataset_kwargs(cfg.data.train)
    _n_prev = int(_train_cfg["n_previous"])
    _chunk = int(
        _train_cfg.get(
            "chunk",
            int(_train_cfg.get("continious_vid_length", 0))
            + int(_train_cfg.get("sparse_length", 0)),
        )
    )
    video_frame_indices = list(range(_n_prev)) + list(
        range(_n_prev + 1, _n_prev + 1 + _chunk)
    )
    logger.info(
        f"VIDEO_FRAME_INDICES derived (n_previous={_n_prev}, chunk={_chunk}): {video_frame_indices}"
    )

    ae_video_scaling_factor = cfg.model.ae_video_scaling_factor


    video_sparse_downsample_factor = normalize_sparse_downsample_factor(
        cfg.cotrain.video_sparse_downsample_factor
    )
    video_force_mixed_path = cfg.cotrain.video_force_mixed_path
    if isinstance(video_force_mixed_path, str):
        video_force_mixed_path = video_force_mixed_path.lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
    video_force_mixed_path = bool(video_force_mixed_path)
    video_sparse_spatial_rope_mode = normalize_sparse_spatial_rope_mode(
        cfg.cotrain.video_sparse_spatial_rope_mode
    )
    use_video_sparse_mode = video_sparse_downsample_factor > 1
    use_video_mixed_path = video_force_mixed_path or use_video_sparse_mode

    stats_tracker = LatentStatsTracker(
        num_channels=512, window_size=100000
    ).to(device, dtype=dtype)
    if cfg.model.load_stats_tracker_weight and cfg.model.stats_tracker_path:
        logger.info(f"Loading latent statistics from: {cfg.model.stats_tracker_path}")
        stats_sd = torch.load(cfg.model.stats_tracker_path, map_location="cpu")
        stats_sd = {k.replace("module.", ""): v for k, v in stats_sd.items()}
        stats_tracker.load_state_dict(stats_sd, strict=True)

    video_diffusion_model = None
    video_diffusion_model_high = None
    _expert_cfg = cfg.cotrain.expert or {}
    expert_dual_train = bool(_expert_cfg.get("dual_train", False))
    _expert_high_cfg = _expert_cfg.get("high") or {}
    _expert_low_cfg = _expert_cfg.get("low") or {}
    _high_init_ckpt = _expert_high_cfg.get("init_ckpt")
    _low_init_ckpt = _expert_low_cfg.get("init_ckpt")
    if cfg.model.use_vgen:
        v_class = import_custom_class(
            cfg.train.video_diffusion_model_class,
            cfg.model.video_diffusion_model_class_path,
        )
        _goal_mode = cfg.goal_mode
        if _goal_mode is not None and _goal_mode.enabled:
            if hasattr(cfg.model.video, "goal_mode_null_text_tokens"):
                null_tokens = int(cfg.model.video.goal_mode_null_text_tokens)
            else:
                null_tokens = int(_goal_mode.null_text_tokens)
            if null_tokens <= 0:
                raise ValueError(
                    "goal_mode null-text conditioning requires at least one token"
                )
            setattr(cfg.model.video, "goal_mode_null_text_tokens", null_tokens)
            logger.info(
                "goal_mode enabled: video.goal_mode_null_text_tokens=%s",
                null_tokens,
            )
        video_kwargs = get_video_model_kwargs(cfg)
        _low_path = _low_init_ckpt or get_video_model_path(cfg)
        video_diffusion_model = load_diffusion_model(
            model_cls=v_class,
            model_dir=_low_path,
            load_weights=cfg.model.vgen_load_weights,
            **video_kwargs,
        ).to(device, dtype=dtype)

        if expert_dual_train or _high_init_ckpt:
            _high_path = _high_init_ckpt or get_video_model_path(cfg)
            logger.info(
                "Loading HIGH-expert VGen ckpt (%s): %s",
                "trainable" if expert_dual_train else "val-only frozen",
                _high_path,
            )
            video_diffusion_model_high = load_diffusion_model(
                model_cls=v_class,
                model_dir=_high_path,
                load_weights=cfg.model.vgen_load_weights,
                **video_kwargs,
            ).to(device, dtype=dtype)
            if not expert_dual_train:
                video_diffusion_model_high.eval()
                for _p in video_diffusion_model_high.parameters():
                    _p.requires_grad_(False)

        multi_view_schedule_cfg = resolve_multi_view_schedule_cfg(cfg)
        if multi_view_schedule_cfg is not None:
            max_requested_views = max(
                len(mode_cfg.get("positions") or mode_cfg.get("cams") or [])
                for mode_cfg in multi_view_schedule_cfg["modes"]
            )
            if max_requested_views > video_diffusion_model.max_view:
                raise ValueError(
                    f"multi_view_schedule requests up to {max_requested_views} views, "
                    f"but model max_view={video_diffusion_model.max_view}"
                )

        if cfg.model.reinit_view_emb:
            video_diffusion_model.make_view_emb(
                video_diffusion_model.max_view,
                video_diffusion_model.inner_dim,
            )

        if use_video_mixed_path:
            patch_size, patch_size_t = _get_video_transformer_patch_sizes(
                video_diffusion_model
            )
            validate_mixed_video_sparse_mode(
                sparse_downsample_factor=video_sparse_downsample_factor,
                sparse_spatial_rope_mode=video_sparse_spatial_rope_mode,
                force_mixed_path=video_force_mixed_path,
                vae_encoding_mode=vae.vae_encoding_mode,
                patch_size=patch_size,
                patch_size_t=patch_size_t,
                use_joint_ae_handoff=False,
            )

    action_diffusion_model = None
    if cfg.model.use_ae:
        a_class = import_custom_class(
            cfg.train.action_diffusion_model_class,
            cfg.model.action_diffusion_model_class_path,
        )
        action_diffusion_model = load_diffusion_model(
            model_cls=a_class,
            model_dir=get_action_model_path(cfg),
            load_weights=cfg.model.ae_load_weights,
            **get_action_model_kwargs(cfg),
        ).to(device, dtype=dtype)

    video_pipeline_class = import_custom_class(
        cfg.model.video_pipeline_class,
        cfg.model.video_pipeline_class_path,
    )
    action_pipeline_class = import_custom_class(
        cfg.model.action_pipeline_class,
        cfg.model.action_pipeline_class_path,
    )

    task_model = WAM_policyTaskModel(
        vgen=video_diffusion_model,
        vgen_high=(
            video_diffusion_model_high if expert_dual_train else None
        ),
        action_expert=action_diffusion_model,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        uncond_prompt_embeds=uncond_prompt_embeds,
    )
    return WAM_policyTaskFlow(
        task_model=task_model, cfg=cfg,
        vae=vae,
        spatial_down_ratio=spatial_down_ratio,
        temporal_down_ratio=temporal_down_ratio,
        video_frame_indices=video_frame_indices,
        ae_video_scaling_factor=ae_video_scaling_factor,
        video_sparse_downsample_factor=video_sparse_downsample_factor,
        video_force_mixed_path=video_force_mixed_path,
        video_sparse_spatial_rope_mode=video_sparse_spatial_rope_mode,
        use_video_sparse_mode=use_video_sparse_mode,
        use_video_mixed_path=use_video_mixed_path,
        stats_tracker=stats_tracker,
        expert_dual_train=expert_dual_train,
        video_pipeline_class=video_pipeline_class,
        action_pipeline_class=action_pipeline_class,
        transforms=transforms,
        output_transforms=output_transforms,
    )
