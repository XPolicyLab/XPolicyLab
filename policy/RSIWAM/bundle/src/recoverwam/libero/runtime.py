import torch
from omegaconf import DictConfig, OmegaConf


def create_action_model(
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
    from .action_model import LiberoActionModel

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
            f"`action_scheduler` missing required keys: {sorted(missing_keys)}. Expected keys: train_shift, infer_shift, num_train_timesteps."
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
    return LiberoActionModel.from_wan21_14b_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        dit_checkpoint_path=dit_checkpoint_path,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        load_clip_encoder=bool(load_clip_encoder),
        proprio_dim=None if proprio_dim is None else int(proprio_dim),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        action_chunk_size=action_chunk_size,
        action_visible_video_frames=int(action_visible_video_frames),
        action_only_first_frame_fast_path=bool(action_only_first_frame_fast_path),
        action_clean_first_frame_start_layer=None
        if action_clean_first_frame_start_layer is None
        else int(action_clean_first_frame_start_layer),
        action_clean_first_frame_timestep_cutoff=None
        if action_clean_first_frame_timestep_cutoff is None
        else float(action_clean_first_frame_timestep_cutoff),
        action_first_frame_kv_source=str(action_first_frame_kv_source),
        action_video_timestep_cutoff=None
        if action_video_timestep_cutoff is None
        else float(action_video_timestep_cutoff),
        video_lora_config=video_lora,
        use_fixed_video_layers=None
        if use_fixed_video_layers is None
        else int(use_fixed_video_layers),
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
        joint_denoising_mode_probs=OmegaConf.to_container(joint_denoising_mode_probs, resolve=True)
        if isinstance(joint_denoising_mode_probs, DictConfig)
        else joint_denoising_mode_probs,
        joint_denoising_sampler=str(joint_denoising_sampler),
    )


def create_wan21_video_model(
    model_id: str,
    dit_checkpoint_path: str,
    tokenizer_model_id: str,
    video_dit_config,
    video_scheduler,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = False,
    load_clip_encoder: bool = True,
    condition_clip_frame: str = "first",
    condition_latent_loss_weight: float = 1.0,
    redirect_common_files: bool = False,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from recoverwam.models.wan21_14b.wan21 import Wan21VideoModel

    if isinstance(video_dit_config, DictConfig):
        video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)
    if not isinstance(video_dit_config, dict):
        raise ValueError(f"`video_dit_config` must resolve to a dict, got {type(video_dit_config)}")
    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if not isinstance(video_scheduler, dict):
        raise ValueError(f"`video_scheduler` must resolve to a dict, got {type(video_scheduler)}")
    required_scheduler_keys = {
        "train_shift",
        "infer_shift",
        "num_train_timesteps",
        "train_sampling_strategy",
        "beta_alpha",
        "beta_beta",
    }
    missing_keys = required_scheduler_keys - set(video_scheduler)
    if missing_keys:
        raise ValueError(f"`video_scheduler` missing required keys: {sorted(missing_keys)}")
    return Wan21VideoModel.from_wan21_14b_pretrained(
        model_id=model_id,
        dit_checkpoint_path=dit_checkpoint_path,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        load_clip_encoder=bool(load_clip_encoder),
        condition_clip_frame=str(condition_clip_frame),
        condition_latent_loss_weight=float(condition_latent_loss_weight),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        video_scheduler=video_scheduler,
        torch_dtype=model_dtype,
        device=device,
    )


def run_training(cfg):
    from pathlib import Path
    import os
    from hydra.utils import instantiate
    from recoverwam.runtime import build_datasets, _resolve_train_device, _mixed_precision_to_model_dtype
    from recoverwam.utils import misc
    from recoverwam.utils.logging_config import setup_logging
    from recoverwam.utils.training_profiler import TrainingProfiler
    from .trainer import LiberoTrainer, validate_preflight_config
    from recoverwam.preflight import validate_batch_contract
    from .diagnostics import VideoComparisonEvaluator, ActionDiagnosticEvaluator
    from torch.utils.data import default_collate

    setup_logging()
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(out)
    if int(os.environ.get("RANK", "0")) == 0:
        OmegaConf.save(cfg, out / "config.yaml", resolve=True)
    model = instantiate(
        cfg.model,
        model_dtype=_mixed_precision_to_model_dtype(cfg.mixed_precision),
        device=_resolve_train_device(),
    )
    train_ds, val_ds = build_datasets(cfg.data)
    probe = validate_preflight_config(cfg)
    if probe:
        validate_batch_contract(
            default_collate([train_ds[0]]),
            probe["stage"],
            video_size=probe["video_size"],
            **probe["contract_kwargs"],
        )
    ev = cfg.evaluation
    if cfg.train_mode == "video_only":
        evaluator = VideoComparisonEvaluator(
            train_ds,
            out / "eval/video_comparison",
            fixed_indices=ev.fixed_indices,
            inference_steps=ev.inference_steps,
            seed=int(ev.seed),
            fps=int(ev.get("fps", 8)),
        )
    else:
        evaluator = ActionDiagnosticEvaluator(
            train_ds,
            fixed_indices=ev.fixed_indices,
            num_inference_steps=int(ev.num_inference_steps),
            seed=int(ev.seed),
        )
    LiberoTrainer(
        model,
        train_ds,
        val_ds,
        cfg=cfg,
        evaluator=evaluator,
        profiler=TrainingProfiler.from_config(cfg),
    ).train()
