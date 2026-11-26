"""E2E60K inference profile; keep the KA/memory base config independent."""

from __future__ import annotations

import os

from collections.abc import Callable

from robodojo_geact.settings import settings
from robodojo_geact.utils.geact.geact_config import GEActConfig

BASIC_PRETRAINED_ROOT=""

class VideoModelConfig:
    """VGen transformer weights + constructor kwargs (``cfg.model.video.*``)."""

    # Video-model weights path; set CKPT for the published asset layout.
    path = f"{BASIC_PRETRAINED_ROOT}/diffusion_pytorch_model.bin"
    activation_fn: str = "gelu-approximate"
    attention_bias: bool = True
    attention_head_dim: int = 64
    attention_out_bias: bool = True
    caption_channels: int = 4096
    cross_attention_dim: int = 2048
    in_channels: int = 512
    norm_elementwise_affine: bool = False
    norm_eps: float = 1e-06
    num_attention_heads: int = 32
    num_layers: int = 28
    out_channels: int = 512
    patch_size: int = 1
    patch_size_t: int = 1
    qk_norm: str = "rms_norm_across_heads"
    vlm_interface_class_name: str = "_QWen3_5_Interface"
    vlm_interface_class_path: str = (
        "robodojo_geact.model.geact.vlm_patches.qwen3_5_interface"
    )
    vlm_model_id: str = os.environ.get("WAM_QWEN_DIR", "")
    v_head_depth: int = 8
    fuser_normalize_gate: bool = True
    vlm_head_view_only: bool = True
    vlm_head_view_index: int = 0
    vlm_text_tokens_only: bool = True
    vlm_fixed_sequence_length: int = 192
    # FLA / causal-conv1d Triton kernels for the Qwen3.5 GatedDeltaNet blocks.
    # NOT bit-exact in bf16, so a run with this on is not a bitwise reference.
    use_fla_kernels: bool = False
    use_t_and_r_embedding: bool = True
    time_embed_zero_init: bool = True
    num_view_identities: int = 8
    # Preserve the outer memory-frame contract while the transformer predicts
    # only the eight future frames.
    t2v_mode: bool = False
    spatial_rope_grid_mode: str = "legacy"
    goal_mode_null_text_tokens: int = 0


class ActionModelConfig:
    """Action-expert weights + constructor kwargs (``cfg.model.action.*``)."""

    # Action-expert weights path; set CKPT for the published asset layout.
    path = f"{BASIC_PRETRAINED_ROOT}/action_model/diffusion_pytorch_model.bin"
    num_layers: int = 28
    attention_bias: bool = True
    norm_elementwise_affine: bool = False
    norm_eps: float = 1e-06
    qk_norm: str = "rms_norm_across_heads"
    activation_fn: str = "gelu-approximate"
    attention_out_bias: bool = True
    action_in_channels: int = 32
    action_num_attention_heads: int = 16
    action_attention_head_dim: int = 72
    vision_in_dim: int = 512
    # Must cover max camera view id used by data (see geact_camera.CAMERA_VIEW_ID_MAP).
    # head/hand_* use 0..2 → 3 is enough; libero external_1/waist use 3/4 → need ≥5.
    num_view_identities: int = 8
    n_view: int = 3
    latent_base_f: int = 9
    latent_base_h: int = 4
    latent_base_w: int = 6
    body_embedding_enabled: bool = True
    num_body_embeddings: int = 12
    body_embedding_zero_init: bool = True


class TrainDatasetConfig:
    """Train MixMultiFrame kwargs (``cfg.data.train.*``)."""
    dataset_type: str = "mix_multi_frame"

    jsonl_path_list: list = ['']
    video_folder_list: list = ['']
    dataset_source_list: list = ["lerobot"]
    dataset_name_list: list = ["lerobot_robodojo"]
    # Match agilex_sim_robotwin's 6-DoF-per-arm layout and body embedding (ID 4).
    body_type_list: list = ["sl_agilex"]
    cam_use: list = [[
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ]]
    repeat: list = [1]
    fps_list: list = [25]
    action_space_list: list = ["abs_joint"]
    stat_file_list: list = ['']
    n_view: int = 3
    sample_size: list = [256, 384]
    preprocess: str = "resize"
    # H100 NVDEC does not support the dataset's AV1 codec; use PyAV/libdav1d.
    decode_type: str = "cpu"
    max_stride: int = 1
    random_crop: bool = False
    return_action: bool = True
    return_video: bool = True
    return_caption: bool = True
    min_gap: int = 10
    n_previous: int = 1
    n_previous_action: int = 1
    strip_goal_frame: bool = False
    step_goal_prob: int = 1
    next_step_prob: int = 0
    p_hand_drop: float = 0.0
    p_color_drop: float = 0.0
    speed_cue_prob: float = 0.0
    speed_cue_thresholds: list | None = None
    embodiment_cue_prob: float = 0.7
    embodiment_sim_drop_prob: float = 0.2
    goal_end_jitter: int = 0
    chunk: int = 6
    action_chunk: int = 54
    continious_vid_length: int = 4
    continious_action_length: int = 52
    sparse_length: int = 2
    mix_captions: bool = True
    use_step_caption: bool = True
    use_recaption: bool = False
    use_frame_cache: bool = False
    filter_action: bool = True
    filter_static_action_chunks: bool = True
    norm_action: bool = True
    choose_start_end: bool = False
    sampling_statagy: str = "msth"

    allow_rnd_instep_goal: bool = True
    allow_rnd_instep_goal_range: int = 100

class ValDatasetConfig:
    """Val MixMultiFrame kwargs (``cfg.data.val.*``)."""
    dataset_type: str = "mix_multi_frame"
    jsonl_path_list: list = ['']
    video_folder_list: list = ['']
    dataset_source_list: list = ["lerobot"]
    dataset_name_list: list = ["lerobot_robodojo"]
    # Match agilex_sim_robotwin's 6-DoF-per-arm layout and body embedding (ID 4).
    body_type_list: list = ["sl_arxvr"]
    cam_use: list = [[
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ]]
    repeat: list = [1]
    fps_list: list = [25]
    action_space_list: list = ["abs_joint"]
    stat_file_list: list = ['']
    n_view: int = 3
    sample_size: list = [256, 384]
    preprocess: str = "resize"
    decode_type: str = "cpu"
    max_stride: int = 1
    random_crop: bool = False
    return_action: bool = True
    return_video: bool = True
    return_caption: bool = True
    min_gap: int = 10
    n_previous: int = 1
    n_previous_action: int = 1
    strip_goal_frame: bool = False
    p_hand_drop: float = 0.0
    p_color_drop: float = 0.0
    speed_cue_prob: float = 0.0
    speed_cue_thresholds: list | None = None
    embodiment_cue_prob: float = 1.0
    embodiment_sim_drop_prob: float = 0.2
    goal_end_jitter: int = 0
    chunk: int = 6
    action_chunk: int = 54
    continious_vid_length: int = 4
    continious_action_length: int = 52
    sparse_length: int = 2
    use_frame_cache: bool = False
    use_step_caption: bool = True
    filter_action: bool = True
    norm_action: bool = True
    choose_start_end: bool = False
    sampling_statagy: str = "msth"

class GEActModelConfig:
    model_name: str = "ltx_train"
    is_i2v: bool = True
    pretrained_model_name_or_path: str = ""
    tokenizer_class_path: str = "None"
    tokenizer_class: str = "T5Tokenizer"
    textenc_class_path: str = "None"
    textenc_class: str = "T5EncoderModel"
    vae_class_path: str = "diffusers"
    vae_class: str = "AutoencoderDC"
    vae_path: str = os.environ.get("WAM_VAE_DIR", "")
    vae_encoding_mode: str = "2dvae"
    batch_size_2dvae: int = 256
    ae_video_scaling_factor: float = 0.6337
    load_stats_tracker_weight: bool = True
    stats_tracker_path: str = (
        os.environ.get("WAM_LATENT_STATS_PATH", "")
    )
    use_vlm: bool = True
    use_vgen: bool = True
    use_ae: bool = True
    gradient_vlm2vgen: bool = False
    gradient_vgen2ae: bool = False
    video_diffusion_model_class_path: str = (
        "robodojo_geact.model.geact.ltx_models.transformer_ltx_multiview_vlm_imf_v2"
    )
    action_diffusion_model_class_path: str = (
        "robodojo_geact.model.geact.ltx_models.transformer_ltx_multiview_action_expert"
    )
    diffusion_scheduler_class_path: str = "diffusers"
    diffusion_scheduler_class: str = "FlowMatchEulerDiscreteScheduler"
    diffusion_mode: str = "FM"
    diffusion_scheduler_args: dict = {"shift": 2.7}
    video_pipeline_class_path: str = (
        "robodojo_geact.model.geact.pipeline.custom_pipeline"
    )
    video_pipeline_class: str = "CustomPipeline"
    action_pipeline_class_path: str = (
        "robodojo_geact.model.geact.pipeline.va_binding_pipeline"
    )
    action_pipeline_class: str = "VABindingPipe"
    return_action: bool = True
    return_video: bool = True
    train_mode: list = ["video", "action"]
    enable_slicing: bool = False
    enable_tiling: bool = False
    # Needs the torch==2.7.1 + triton==3.3.1 pair pinned in wam/requirements-infer.txt.
    vae_compile_encode: bool = True
    vae_compile_mode: str = "default"
    vae_compile_dynamic: bool = False
    vae_compile_fullgraph: bool = False
    vgen_compile_forward: bool = False
    vgen_compile_mode: str = "default"
    vgen_compile_dynamic: bool = False
    vgen_compile_fullgraph: bool = False
    vgen_compile_jvp: bool = False
    vlm_compile_forward: bool = False
    gradient_checkpointing: bool = False
    video: VideoModelConfig = VideoModelConfig()
    action: ActionModelConfig = ActionModelConfig()
    vgen_load_weights: bool = False
    ae_load_weights: bool = False
    vae_load_weights: bool = True
    vlm_text_only: bool = False
    reinit_view_emb: bool = False
    frame_concat_conditioning: bool = False
    frame_concat_curriculum: dict = {}
    # None → call sites fall back to memory-frame count (mem_size).
    num_frames_wo_dropout: int | None = None


class GEActTrainConfig:
    video_diffusion_model_class: str = "LTXVideoTransformer3DModel"
    action_diffusion_model_class: str = "ActionExpert"
    train_steps: int = 10000000
    train_epochs: int = 10000
    drop_last: bool = True
    steps_to_save: int = 10000
    steps_to_save_train_video: int = 2000
    steps_to_log: int = 20
    save_optimizer_state: bool = True
    resume_optimizer_state: str | None = None
    initial_global_step: int = 0
    # When cotrain_schedule is disabled, single-phase mode name.
    separate_training_mode: str = "vgen"
    probe_train_batches_before_start: int = 0
    mixed_precision: str = "bf16"
    allow_tf32: bool = False
    nccl_timeout: int = 6000
    noise_to_first_frame: float = 0.1
    optimizer: str = "adamw"
    lr: float = 5e-5
    beta1: float = 0.9
    beta2: float = 0.95
    beta3: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 1e-5
    optimizer_8bit: bool = False
    optimizer_torchao: bool = False
    scale_lr: bool = False
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 3
    log_per_model_grad_norm: bool = False
    log_fuser_gradients: bool = False
    trainable_param_filter: dict = {"enabled": False}
    lr_scheduler: str = "constant_with_warmup"
    lr_warmup_steps: int = 1000
    lr_num_cycles: int = 1
    lr_power: float = 1.0
    use_deepspeed: bool = True
    deepspeed: dict = {
        "zero_optimization": {"stage": 2},
        "fp16": {"enabled": False},
        "bf16": {"enabled": True},
        "gradient_clipping": 1.0,
    }
    profiling: dict = {
    "enabled": True,
    "profile_all_steps": False,
    "wait_steps": 5,
    "warmup_steps": 2,
    "active_steps": 15,
    "sync_cuda": True,
    "trace": False,
    "trace_ranks": [0],
    "record_shapes": False,
    "profile_memory": False,
    "with_stack": False,
}

class GEActDataConfig:
    add_state: bool = True
    rand_init_action: bool = False
    caption_dropout_p: float = 0.06
    share_dataloader: bool = True
    batch_size: int = 12
    dataloader_num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    dataloader_prefetch_factor: int = 4
    dataloader_timeout: int = 6000
    infinite_batch_sampler: bool = True
    repeat_in_batch: int = 1
    token_dropout: bool = False
    mask_ratio: int = 1
    mask_cur_obs: bool = False
    aug_vision_token: bool = True
    drop_cur_obs: bool = False
    multi_view_schedule: dict = {
        "enabled": False,
        "schedule": [10, 10, 10, 10, 60],
        "modes": [
            {"name": "first_second", "positions": [0, 1]},
            {"name": "first_third", "positions": [0, 2]},
            {"name": "second_third", "positions": [1, 2]},
            {"name": "first_only", "positions": [0]},
            {"name": "full_3views", "positions": [0, 1, 2]},
        ],
    }
    train: TrainDatasetConfig = TrainDatasetConfig()
    val: ValDatasetConfig = ValDatasetConfig()
    cotrain: TrainDatasetConfig | None = None
    ae_train: TrainDatasetConfig | None = None
    use_color_jitter: bool = True
    force_goal_on_empty_caption: bool = False
    frame_proportional_sampling: bool = True
    # Equalize total sampling mass across task_caption values while retaining
    # frame-proportional sampling within each task.
    task_level_balancing: bool = True

class GEActCotrainConfig:
    gen_loss_scale: float = 1.0
    action_loss_scale: float = 1.0
    cotrain_video_loss: bool = False
    cotrain_video_loss_scale: float = 0.1
    cotrain_latent_l1_scale: float = 1.0
    gt_latent_supervision_scale: float = 1.0
    gt_latent_supervision_loss_type: str = "mse"
    cotrain_pixel_lpips_scale: int = 0
    cotrain_video_loss_n_frames: int = 0
    cotrain_video_decode_batch: int = 4
    cotrain_video_sparse_loss_weight: float = 0.1
    ae_flow_weighting_scheme: str = "uniform"
    ae_flow_logit_mean: float = 0.0
    ae_flow_logit_std: float = 1.0
    ae_flow_mode_scale: float = 1.29
    imf_dist: str = "lognorm"
    imf_r_mode: str = "uniform_to_zero"
    imf_mu: float = 0.0
    imf_sigma: float = 1.0
    imf_flow_ratio: float = 0.75
    imf_norm_p: float = 1.0
    imf_norm_eps: float = 0.1
    use_partial_batch_mf: bool = True
    uniform_future_loss_weight: bool = True
    loss_normalize_by_total_tokens: bool = True
    pixel_wise_timestep: bool = True
    timestep_scale: float = 32.0
    video_sparse_downsample_factor: int = 1
    video_force_mixed_path: bool = False
    video_sparse_spatial_rope_mode: str = "center"
    cotrain_schedule: dict = {
        "enabled": True,
        "startup_vgen_steps": 0,
        "startup_ae_steps": 0,
        "vgen_steps": 50,
        "ae_steps": 50,
        "cotrain_steps": 100,
        "vlm_rm_steps": 0,
        "phase_order": ["cotrain", "vgen", "ae"],
    }
    combined_cotrain: dict = {
        "enabled": True,
        "order": ["vgen", "ae", "cotrain", ],
        "weights": {"vgen": 1.0, "ae": 0.1, "cotrain": 1.0, },
        "warmup_steps": 0,
        "warmup_substitute": "ae",
        "share_batch": True,
        "share_features": True,
    }
    one_step_start_t: dict = {
        "enabled": False,
        "noise_min": 0.85,
        "noise_max": 0.92,
    }
    cotrain_freeze_vgen: dict = {"enabled": False}
    expert: dict = {
        "enabled": False,
        "role": "low",
        "high": {"band": [0.85, 1.00], "mu": -1.0, "sigma": 2.0},
        "low": {"band": [0.00, 0.92], "mu": -0.9, "sigma": 1.8},
    }
    two_step_cotrain: dict = {
        "enabled": True,
        "group_size": 4,
        "judge_sigma": 0.95,
        "backward_topk": 1,
        "debug_decode_chunk_members": 1,
        "ref_mode": "gt",
        "ref_gt_weight": 0.1,
        "filter_type": "separability",
        "sep_threshold": 2.0,
        "sep_sigma_floor": 0.3,
        "weight_mode": "gt_anchored",
        "weight_temp": 3.0,
        "eref_floor": 0.1,
    }
    du_dt_t_bin_edges: list = [0.0]
    imf_dudt_limit: bool = True
    imf_dudt_limit_mode: str = "hard"
    imf_dudt_limit_budget: str = "none"
    imf_dudt_limit_clamp: str = "element"
    imf_dudt_limit_kappa: float = 0.9
    imf_dudt_limit_beta: float = 0.3
    imf_dudt_limit_c: float = 5.0
    imf_dudt_limit_log_only: bool = False
    # Analytic JVP instead of double-backward. Verified numerically equal to
    # "exact" on this config; required for vgen_compile_jvp to be worth enabling.
    imf_jvp_impl: str = "fast"
    imf_jvp_term_mode: str = "combined"
    imf_jvp_fp32: bool = False


class GEActGoalModeConfig:
    """Goal-frame conditioning (``cfg.goal_mode.*``).

    When ``enabled=True``, train batches are tagged by ``GoalModeBatchSampler``
    and the video model allocates learnable null-text tokens. Only wired for
    vgen-phase training (ae/cotrain/dsrl/vlm_rm steps must be 0).
    """

    enabled: bool = False
    null_text_tokens: int = 16
    p: float = 0.0
    curriculum: dict = {
        "p_start": 0.0,
        "hold_steps": 0,
        "decay": "cosine",
        "decay_steps": 0,
        "p_end": 0.0,
    }


    ### NOTE: When applying goal-swapping, the last latent of vgen input is clean! The final frame/latent is the end of each sub-step or the end of episod.
    # Hard-swap last future frame onto goal conditioning (0 = off).
    goal_gt_swap_prob: float = 0.0
    goal_gt_swap_curriculum: dict | None = None
    val_goal_swap: bool = False
    replace_after_vgen: bool = False
    goal_gt_swap_max_range: int = 0


class GEActValidationConfig:
    # Probe runs skip val (start + interval). Restore 1000 / True for real training.
    steps_to_val: int = 1000
    run_validation_at_start: bool = True
    multi_view_val: dict = {
        "enabled": False,
        "modes": [
            {"name": "full_3views", "positions": [0, 1, 2]},
            {"name": "first_second", "positions": [0, 1]},
            {"name": "first_third", "positions": [0, 2]},
            {"name": "second_third", "positions": [1, 2]},
            {"name": "first_only", "positions": [0]},
        ],
    }
    num_inference_step: int = 5
    noisy_video: bool = True
    num_val_episodes: int = 35
    num_val_segments: int = 3
    val_group_by: str = "task_caption"
    stat_path: str = os.environ.get("STAT_PATH", "")
    expand_action_to_raw: bool = True
    dense_action_steps: int = 52
    ae_rescale: bool = True
    ae_rescale_ratio: float | None = None
    start_from_first_step: bool = True
    # None → use data.val.continious_action_length at build time.
    val_segment_stride: int | None = None
    val_sample_by_embodiment: bool = True
    val_respect_forbidden_sampling: bool = True
    num_val_episodes_per_embodiment: dict | int | None = None
    val_forbidden_window: int = 100
    fixed_val_loss: dict = {
        "enabled": False,
        "num_samples": 64,
        "batch_size": 8,
        "seed": 1234,
        "t_grid": [0.2, 0.428, 0.538, 0.677, 0.777, 0.862, 0.907, 1.0],
        "sources": ["val"],
        "sampling": {
            "mode": "stratified",
            "group_by": "task_caption",
            "episodes_per_group": 1,
        },
    }


class GEActInferConfig:
    """Offline GenieSim / openloop binding (``cfg.infer.*``).

    Matches ``main_server_all.py`` abs_joint default for cotrain_3w.
    Override with ``ROBOT_NAME`` / ``STAT_PATH`` / ``--robot-name`` / ``--stat-path``.
    """

    robot_name: str = "g2_90d"
    # Explicit path required (CLI/env ``STAT_PATH`` / ``--stat-path`` override).
    stat_path: str = (
        os.environ.get("STAT_PATH", "")
    )
    joint_dim: int = 7
    action_mode: str = "abs"
    # Deploy/store-guard switches (same semantics as ge-act cotrain_3w top-level).
    skip_future_vae_encoding: bool = True
    vlm_disable_cache: bool = True
    vlm_logits_to_keep: int = 1
    use_action_condition_cache: bool = True
    compute_v_head: bool = False
    return_head_waist: bool = False
    dense_action_steps: int = settings.geact_infer_dense_action_steps or 52


_model = GEActModelConfig()
_model.video = VideoModelConfig()
_model.action = ActionModelConfig()
_data = GEActDataConfig()
_data.train = TrainDatasetConfig()
_data.val = ValDatasetConfig()
cfg = GEActConfig(
    model=_model,
    train=GEActTrainConfig(),
    data=_data,
    cotrain=GEActCotrainConfig(),
    validation=GEActValidationConfig(),
    goal_mode=GEActGoalModeConfig(),
    infer=GEActInferConfig(),
    ema={},
    dsrl={},
    vlm_rm={},
    basic_fps=30,
)

if settings.mode == "infer":

    import torch
    from robodojo_geact.inferrer.deploy_inferrer import DeployInferrer

    from robodojo_geact.data.geact.transforms.geact_infer_transforms import (
        GeactCorobotFormat,
        GeactCorobotPrepare,
    )
    from robodojo_geact.inferrer.geact_deploy_helpers import (
        load_geact_infer_stats,
        resolve_geact_infer_settings,
    )
    from robodojo_geact.model.geact.build_models import build_geact_task_flow
    from robodojo_geact.utils.common_utils import get_device

    if settings.geact_infer_return_head_waist:
        cfg.infer.return_head_waist = True

    _resolved = resolve_geact_infer_settings(
        cfg,
        robot_name=settings.geact_infer_robot_name or None,
        stat_path=settings.geact_infer_stat_path or None,
    )
    cfg.infer.robot_name = _resolved["robot_name"]
    cfg.infer.stat_path = _resolved["stat_path"]
    cfg.infer.joint_dim = (
        6
        if cfg.infer.robot_name in {
            "sl_acone", "sl_arxvr", "sl_agilex", "robodojo", "robodojo_real"
        }
        else _resolved["joint_dim"]
    )
    cfg.infer.action_mode = _resolved["action_mode"]

    _stats = load_geact_infer_stats(cfg.infer.stat_path, get_device())
    _val = cfg.data.val
    _cam_use = list(_val.cam_use[0]) if _val.cam_use else []
    if not _cam_use:
        raise ValueError("cfg.data.val.cam_use must define inference cameras")
    if cfg.infer.dense_action_steps <= 0:
        raise ValueError("cfg.infer.dense_action_steps must be positive")

    transforms = [
        GeactCorobotPrepare(
            cam_order=_cam_use,
            target_h=_val.sample_size[0],
            target_w=_val.sample_size[1],
            s_mean=_stats["s_mean"],
            s_std=_stats["s_std"],
            device=get_device(),
            dtype=(
                torch.bfloat16
                if cfg.train.mixed_precision == "bf16"
                else torch.float32
            ),
            default_num_ae_steps=cfg.validation.num_inference_step,
            default_dense_action_steps=cfg.infer.dense_action_steps,
            legacy_state_layout=os.environ.get(
                "GEACT_INFER_LEGACY_STATE_LAYOUT", "identity"
            ),
        ),
    ]
    output_transforms = [
        GeactCorobotFormat(
            robot_name=cfg.infer.robot_name,
            action_dim=_stats["a_mean"].shape[0],
            default_dense_action_steps=cfg.infer.dense_action_steps,
            return_head_waist=cfg.infer.return_head_waist,
        ),
    ]
    task_flow = build_geact_task_flow(
        cfg=cfg,
        cotrain_schedule=None,
        transforms=transforms,
        output_transforms=output_transforms,
    )
    if settings.infer_checkpoint_dir:
        inferrer = DeployInferrer(
            run_name=settings.run_name,
            infer_ckpt_cfg={
                "pretrain_model": settings.infer_checkpoint_dir,
                "to_cuda": True,
            },
            infer_task_flow=task_flow,
            infer_type="deploy",
        )
