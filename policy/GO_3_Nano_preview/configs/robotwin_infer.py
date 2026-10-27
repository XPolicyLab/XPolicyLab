"""Robotwin WAM_policy inference configuration.

The evaluation server loads this module after configs/runtime.conf.
Use eval.sh or setup_eval_policy_server.sh; paths come from environment variables.
"""
from __future__ import annotations
from go3_nano_preview.runtime_paths import path_from_env, paths_from_env, robotwin_stats_path, inference_weight_path
from collections.abc import Callable
from policy_core.settings import settings
from policy_core.utils.WAM_policy.WAM_policy_config import WAM_policyConfig

class VideoModelConfig:
    """VGen transformer weights + constructor kwargs (``cfg.model.video.*``)."""
    path: str = inference_weight_path('VIDEO_PRETRAIN_PATH', settings.infer_checkpoint_dir, 'diffusion_pytorch_model.bin')
    activation_fn: str = 'gelu-approximate'
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
    qk_norm: str = 'rms_norm_across_heads'
    vlm_interface_class_name: str = '_QWen3_5_Interface'
    vlm_interface_class_path: str = 'policy_core.model.WAM_policy.vlm_patches.qwen3_5_interface'
    vlm_model_id: str = path_from_env('VLM_MODEL_PATH')
    v_head_depth: int = 8
    fuser_normalize_gate: bool = True
    vlm_head_view_only: bool = True
    vlm_head_view_index: int = 0
    vlm_text_tokens_only: bool = True
    vlm_fixed_sequence_length: int = 192
    use_t_and_r_embedding: bool = True
    time_embed_zero_init: bool = True
    num_view_identities: int = 10
    t2v_mode: bool = False
    spatial_rope_grid_mode: str = 'canonical_center'
    goal_mode_null_text_tokens: int = 16

class ActionModelConfig:
    """Action-expert weights + constructor kwargs (``cfg.model.action.*``)."""
    path: str = inference_weight_path('ACTION_PRETRAIN_PATH', settings.infer_checkpoint_dir, 'action_model/diffusion_pytorch_model.bin')
    num_layers: int = 28
    attention_bias: bool = True
    norm_elementwise_affine: bool = False
    norm_eps: float = 1e-06
    qk_norm: str = 'rms_norm_across_heads'
    activation_fn: str = 'gelu-approximate'
    attention_out_bias: bool = True
    action_in_channels: int = 32
    action_num_attention_heads: int = 16
    action_attention_head_dim: int = 72
    vision_in_dim: int = 512
    num_view_identities: int = 10
    n_view: int = 3
    latent_base_f: int = 9
    latent_base_h: int = 4
    latent_base_w: int = 6
    body_embedding_enabled: bool = True
    num_body_embeddings: int = 14
    body_embedding_zero_init: bool = True

class TrainDatasetConfig:
    """Train MixMultiFrame kwargs (``cfg.data.train.*``); kept for schema parity."""
    jsonl_path_list: list = paths_from_env('TRAIN_JSONL_PATHS')
    video_folder_list: list = paths_from_env('DATA_VIDEO_ROOTS')
    dataset_source_list: list = ['processed']
    dataset_name_list: list = ['robotwin']
    repeat: list = [1]
    fps_list: list = [30]
    cam_use: list = [['head', 'hand_left', 'hand_right']]
    action_space_list: list = ['abs_joint']
    stat_file_list: list = [robotwin_stats_path()]
    p_hand_drop: float = 0.3
    p_color_drop: float = 0.3
    speed_cue_prob: float = 0.7
    speed_cue_thresholds: list | None = [0.03, 0.09]
    embodiment_cue_prob: float = 1.0
    embodiment_sim_drop_prob: float = 0.2
    n_view: int = 3
    sample_size: list = [256, 384]
    preprocess: str = 'resize'
    decode_type: str = 'cpu'
    max_stride: int = 1
    random_crop: bool = False
    return_action: bool = True
    return_video: bool = True
    return_caption: bool = True
    min_gap: int = 10
    n_previous: int = 1
    n_previous_action: int = 1
    strip_goal_frame: bool = False
    step_goal_prob: float = 0.95
    next_step_prob: float = 0.05
    goal_end_jitter: int = 5
    chunk: int = 6
    action_chunk: int = 54
    continious_vid_length: int = 4
    continious_action_length: int = 52
    sparse_length: int = 2
    mix_captions: bool = True
    use_step_caption: bool = True
    use_recaption: bool = True
    task_recap_file: str = path_from_env('TASK_RECAP_PATH', required=False)
    step_recap_file: str = path_from_env('STEP_RECAP_PATH', required=False)
    use_frame_cache: bool = False
    filter_action: bool = True
    norm_action: bool = True
    max_samples_per_source: int | None = None

class ValDatasetConfig:
    """Val MixMultiFrame kwargs (``cfg.data.val.*``) — cameras / chunk for serve."""
    jsonl_path_list: list = paths_from_env('TRAIN_JSONL_PATHS')
    video_folder_list: list = paths_from_env('DATA_VIDEO_ROOTS')
    dataset_source_list: list = ['processed']
    dataset_name_list: list = ['robotwin']
    cam_use: list = [['head', 'hand_left', 'hand_right']]
    action_space_list: list = ['abs_joint']
    stat_file_list: list = [robotwin_stats_path()]
    speed_cue_prob: float = 1.0
    speed_cue_thresholds: list | None = [0.03, 0.09]
    embodiment_cue_prob: float = 1.0
    embodiment_sim_drop_prob: float = 0.2
    p_hand_drop: float = 0.0
    p_color_drop: float = 0.0
    goal_end_jitter: int = 0
    n_view: int = 3
    sample_size: list = [256, 384]
    preprocess: str = 'resize'
    decode_type: str = 'cpu'
    max_stride: int = 1
    random_crop: bool = False
    return_action: bool = True
    return_video: bool = True
    return_caption: bool = True
    min_gap: int = 5
    n_previous: int = 1
    n_previous_action: int = 1
    strip_goal_frame: bool = False
    chunk: int = 6
    action_chunk: int = 54
    continious_vid_length: int = 4
    continious_action_length: int = 52
    sparse_length: int = 2
    use_frame_cache: bool = False
    use_step_caption: bool = True
    filter_action: bool = True
    norm_action: bool = True
    max_samples_per_source: int | None = None

class WAM_policyModelConfig:
    model_name: str = 'ltx_train'
    is_i2v: bool = True
    pretrained_model_name_or_path: str = ''
    tokenizer_class_path: str = 'None'
    tokenizer_class: str = 'T5Tokenizer'
    textenc_class_path: str = 'None'
    textenc_class: str = 'T5EncoderModel'
    vae_class_path: str = 'diffusers'
    vae_class: str = 'AutoencoderDC'
    vae_path: str = path_from_env('VAE_PATH')
    vae_encoding_mode: str = '2dvae'
    batch_size_2dvae: int = 256
    ae_video_scaling_factor: float = 0.6337
    load_stats_tracker_weight: bool = True
    stats_tracker_path: str = path_from_env('STATS_TRACKER_PATH')
    use_vlm: bool = True
    use_vgen: bool = True
    use_ae: bool = True
    gradient_vlm2vgen: bool = False
    gradient_vgen2ae: bool = False
    video_diffusion_model_class_path: str = 'policy_core.model.WAM_policy.ltx_models.transformer_ltx_multiview_vlm_imf_v2'
    action_diffusion_model_class_path: str = 'policy_core.model.WAM_policy.ltx_models.transformer_ltx_multiview_action_expert'
    diffusion_scheduler_class_path: str = 'diffusers'
    diffusion_scheduler_class: str = 'FlowMatchEulerDiscreteScheduler'
    diffusion_mode: str = 'FM'
    diffusion_scheduler_args: dict = {'shift': 2.7}
    video_pipeline_class_path: str = 'policy_core.model.WAM_policy.pipeline.custom_pipeline'
    video_pipeline_class: str = 'CustomPipeline'
    action_pipeline_class_path: str = 'policy_core.model.WAM_policy.pipeline.va_binding_pipeline'
    action_pipeline_class: str = 'VABindingPipe'
    return_action: bool = True
    return_video: bool = True
    train_mode: list = ['video', 'action']
    enable_slicing: bool = False
    enable_tiling: bool = False
    gradient_checkpointing: bool = False
    vae_before_vlm_input_prepare: bool = True
    video: VideoModelConfig = VideoModelConfig()
    action: ActionModelConfig = ActionModelConfig()
    vgen_load_weights: bool = True
    ae_load_weights: bool = True
    vae_load_weights: bool = True
    vlm_text_only: bool = False
    reinit_view_emb: bool = False
    frame_concat_conditioning: bool = False
    frame_concat_curriculum: dict = {}
    num_frames_wo_dropout: int | None = None

class WAM_policyTrainConfig:
    save_checkpoints: bool = True
    zero_copy_allreduce: bool = True
    video_diffusion_model_class: str = 'LTXVideoTransformer3DModel'
    action_diffusion_model_class: str = 'ActionExpert'
    train_steps: int = 10000000
    train_epochs: int = 10000
    drop_last: bool = True
    steps_to_save: int = 2000
    steps_to_save_train_video: int = 2000
    steps_to_log: int = 20
    save_optimizer_state: bool = True
    resume_optimizer_state: str | None = None
    initial_global_step: int = 0
    separate_training_mode: str = 'vgen'
    probe_train_batches_before_start: int = 0
    mixed_precision: str = 'bf16'
    allow_tf32: bool = False
    nccl_timeout: int = 6000
    noise_to_first_frame: float = 0.1
    optimizer: str = 'adamw'
    lr: float = 0.0001
    beta1: float = 0.9
    beta2: float = 0.95
    beta3: float = 0.999
    epsilon: float = 1e-08
    weight_decay: float = 1e-05
    optimizer_8bit: bool = False
    optimizer_torchao: bool = False
    scale_lr: bool = False
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 3
    log_per_model_grad_norm: bool = True
    log_fuser_gradients: bool = False
    trainable_param_filter: dict = {'enabled': False}
    lr_scheduler: str = 'constant_with_warmup'
    lr_warmup_steps: int = 5000
    lr_num_cycles: int = 1
    lr_power: float = 1.0
    use_deepspeed: bool = True
    deepspeed: dict = {'zero_optimization': {'stage': 2}, 'fp16': {'enabled': False}, 'bf16': {'enabled': True}, 'gradient_clipping': 1.0}
    profiling: dict = {'enabled': True, 'wait_steps': 100, 'warmup_steps': 10, 'active_steps': 30, 'sync_cuda': True, 'trace': False, 'trace_ranks': [0], 'record_shapes': False, 'profile_memory': False, 'with_stack': False}

class WAM_policyDataConfig:
    add_state: bool = True
    rand_init_action: bool = False
    caption_dropout_p: float = 0.06
    share_dataloader: bool = True
    batch_size: int = 8
    dataloader_num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    dataloader_prefetch_factor: int = 4
    multiprocessing_context: str | None = None
    worker_init_fn: Callable[[int], None] | None = None
    dataloader_timeout: int = 6000
    infinite_batch_sampler: bool = False
    repeat_in_batch: int = 1
    token_dropout: bool = False
    fixed_token_dropout_ratio: float | None = None
    token_dropout_ratio_start: float = 0.5
    token_dropout_ratio_max: float = 1.0 - 0.0625
    token_dropout_ratio_min: float = 0.5
    token_dropout_ratio_step: float = 1.09375e-05
    token_dropout_ratio_std: float = 0.3
    mask_ratio: int = 1
    mask_cur_obs: bool = False
    aug_vision_token: bool = True
    drop_cur_obs: bool = False
    multi_view_schedule: dict = {'enabled': True, 'schedule': [10, 10, 10, 10, 60], 'modes': [{'name': 'first_second', 'cams': ['head', 'hand_left'], 'view_ids': [0, 1]}, {'name': 'first_third', 'cams': ['head', 'hand_right'], 'view_ids': [0, 2]}, {'name': 'second_third', 'cams': ['hand_left', 'hand_right'], 'view_ids': [1, 2]}, {'name': 'first_only', 'cams': ['head'], 'view_ids': [0]}, {'name': 'full_3views', 'cams': ['head', 'hand_left', 'hand_right'], 'view_ids': [0, 1, 2]}]}
    train: TrainDatasetConfig = TrainDatasetConfig()
    val: ValDatasetConfig = ValDatasetConfig()
    cotrain: TrainDatasetConfig | None = None
    ae_train: TrainDatasetConfig | None = None
    use_color_jitter: bool = False
    force_goal_on_empty_caption: bool = True
    frame_proportional_sampling: bool = True

class WAM_policyCotrainConfig:
    gen_loss_scale: float = 1.0
    action_loss_scale: float = 1.0
    cotrain_video_loss: bool = False
    cotrain_video_loss_scale: float = 0.1
    cotrain_latent_l1_scale: float = 1.0
    cotrain_pixel_lpips_scale: int = 0
    cotrain_video_loss_n_frames: int = 0
    cotrain_video_decode_batch: int = 4
    cotrain_video_sparse_loss_weight: float = 0.1
    ae_flow_weighting_scheme: str = None
    ae_flow_logit_mean: float = 0.0
    ae_flow_logit_std: float = 1.0
    ae_flow_mode_scale: float = 1.29
    imf_dist: str = 'lognorm'
    imf_r_mode: str = 'uniform_to_zero'
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
    video_sparse_spatial_rope_mode: str = 'center'
    cotrain_schedule: dict = {'enabled': True, 'startup_vgen_steps': 0, 'startup_ae_steps': 0, 'vgen_steps': 50, 'ae_steps': 50, 'cotrain_steps': 100, 'vlm_rm_steps': 0, 'phase_order': ['cotrain', 'vgen', 'ae']}
    combined_cotrain: dict = {'enabled': True, 'order': ['vgen', 'ae', 'cotrain'], 'weights': {'vgen': 1.0, 'ae': 0.1, 'cotrain': 1.0}, 'warmup_steps': 0, 'warmup_substitute': 'ae', 'share_batch': True, 'share_features': True}
    one_step_start_t: dict = {'enabled': False, 'noise_min': 0.85, 'noise_max': 0.92}
    cotrain_freeze_vgen: dict = {'enabled': False}
    expert: dict = {'enabled': False, 'role': 'low', 'high': {'band': [0.85, 1.0], 'mu': -1.0, 'sigma': 2.0}, 'low': {'band': [0.0, 0.92], 'mu': -0.9, 'sigma': 1.8}}
    two_step_cotrain: dict = {'enabled': True, 'group_size': 4, 'judge_sigma': 0.95, 'backward_topk': 1, 'debug_decode_chunk_members': 1, 'ref_mode': 'gt', 'ref_gt_weight': 0.1, 'filter_type': 'separability', 'sep_threshold': 2.0, 'sep_sigma_floor': 0.3, 'weight_mode': 'gt_anchored', 'weight_temp': 3.0, 'eref_floor': 0.1}
    du_dt_t_bin_edges: list = [0.0]
    imf_dudt_limit: bool = True
    imf_dudt_limit_mode: str = 'hard'
    imf_dudt_limit_budget: str = 'none'
    imf_dudt_limit_clamp: str = 'element'
    imf_dudt_limit_kappa: float = 0.9
    imf_dudt_limit_beta: float = 0.3
    imf_dudt_limit_c: float = 5.0
    imf_dudt_limit_log_only: bool = False
    imf_jvp_term_mode: str = 'combined'
    imf_jvp_fp32: bool = False
    imf_jvp_impl: str = 'exact'

class WAM_policyGoalModeConfig:
    enabled: bool = False
    null_text_tokens: int = 16
    p: float = 0.05
    curriculum: dict = {'p_start': 0.05, 'hold_steps': 0, 'decay': 'cosine', 'decay_steps': 0, 'p_end': 0.05}
    goal_gt_swap_prob: float = 0.0
    goal_gt_swap_curriculum: dict | None = None

class WAM_policyValidationConfig:
    steps_to_val: int = 2000
    run_validation_at_start: bool = True
    multi_view_val: dict = {'enabled': False, 'modes': [{'name': 'full_3views', 'cams': ['head', 'hand_left', 'hand_right'], 'view_ids': [0, 1, 2]}, {'name': 'first_second', 'cams': ['head', 'hand_left'], 'view_ids': [0, 1]}, {'name': 'first_third', 'cams': ['head', 'hand_right'], 'view_ids': [0, 2]}, {'name': 'second_third', 'cams': ['hand_left', 'hand_right'], 'view_ids': [1, 2]}, {'name': 'first_only', 'cams': ['head'], 'view_ids': [0]}]}
    num_inference_step: int = 5
    noisy_video: bool = True
    num_val_episodes: int = 1
    num_val_segments: int = 4
    stat_path: str = robotwin_stats_path()
    save_debug_video: bool = True
    expand_action_to_raw: bool = True
    dense_action_steps: int = 52
    ae_rescale: bool = True
    ae_rescale_ratio: float | None = None
    start_from_first_step: bool = True
    val_segment_stride: int | None = None
    val_sample_by_embodiment: bool = True
    val_respect_forbidden_sampling: bool = True
    num_val_episodes_per_embodiment: dict | int | None = None
    val_forbidden_window: int = 100
    fixed_val_loss: dict = {'enabled': True, 'num_samples': 64, 'batch_size': 8, 'seed': 1234, 't_grid': [0.2, 0.428, 0.538, 0.677, 0.777, 0.862, 0.907, 1.0], 'sources': ['val']}

class WAM_policyInferConfig:
    """Robotwin inference binding (``cfg.infer.*``).

    Matches ``main_server_all.py`` abs_joint default for cotrain_3w.
    Robotwin state/action statistics are bundled with this policy.
    """
    robot_name: str = 'robotwin'
    stat_path: str = robotwin_stats_path()
    joint_dim: int = 6
    action_mode: str = 'abs'
    save_debug_video: bool = False
    debug_video_dir: str = './debug_videos'
    debug_video_fps: int = 4
    skip_future_vae_encoding: bool = True
    vlm_disable_cache: bool = True
    vlm_logits_to_keep: int = 1
    compute_v_head: bool = False
    return_head_waist: bool = False
    dense_action_steps: int = settings.WAM_policy_infer_dense_action_steps or 52
_model = WAM_policyModelConfig()
_model.video = VideoModelConfig()
_model.action = ActionModelConfig()
_data = WAM_policyDataConfig()
_data.train = TrainDatasetConfig()
_data.val = ValDatasetConfig()
cfg = WAM_policyConfig(model=_model, train=WAM_policyTrainConfig(), data=_data, cotrain=WAM_policyCotrainConfig(), validation=WAM_policyValidationConfig(), goal_mode=WAM_policyGoalModeConfig(), infer=WAM_policyInferConfig(), ema={}, dsrl={}, vlm_rm={}, basic_fps=30)
if settings.mode == 'infer':
    import torch
    from policy_core.inferrer.deploy_inferrer import DeployInferrer
    from policy_core.data.WAM_policy.transforms.WAM_policy_infer_transforms import WAM_policyCorobotFormat, WAM_policyCorobotPrepare
    from policy_core.inferrer.WAM_policy_deploy_helpers import load_WAM_policy_infer_stats, resolve_WAM_policy_infer_settings
    from policy_core.model.WAM_policy.build_models import build_WAM_policy_task_flow
    from policy_core.utils.common_utils import get_device
    if settings.WAM_policy_infer_return_head_waist:
        cfg.infer.return_head_waist = True
    _resolved = resolve_WAM_policy_infer_settings(cfg, robot_name=settings.WAM_policy_infer_robot_name or None, stat_path=settings.WAM_policy_infer_stat_path or None)
    cfg.infer.robot_name = _resolved['robot_name']
    cfg.infer.stat_path = _resolved['stat_path']
    cfg.infer.joint_dim = _resolved['joint_dim']
    cfg.infer.action_mode = _resolved['action_mode']
    if not settings.infer_checkpoint_dir:
        raise ValueError("settings.infer_checkpoint_dir must be set before importing this cfg with settings.mode='infer'")
    _stats = load_WAM_policy_infer_stats(cfg.infer.stat_path, get_device())
    _val = cfg.data.val
    _cam_use = list(_val.cam_use[0]) if _val.cam_use else []
    if not _cam_use:
        raise ValueError('cfg.data.val.cam_use must define inference cameras')
    if cfg.infer.dense_action_steps <= 0:
        raise ValueError('cfg.infer.dense_action_steps must be positive')
    transforms = [WAM_policyCorobotPrepare(cam_order=_cam_use, target_h=_val.sample_size[0], target_w=_val.sample_size[1], s_mean=_stats['s_mean'], s_std=_stats['s_std'], device=get_device(), dtype=torch.bfloat16 if cfg.train.mixed_precision == 'bf16' else torch.float32, default_num_ae_steps=cfg.validation.num_inference_step, default_dense_action_steps=cfg.infer.dense_action_steps)]
    output_transforms = [WAM_policyCorobotFormat(robot_name=cfg.infer.robot_name, action_dim=_stats['a_mean'].shape[0], default_dense_action_steps=cfg.infer.dense_action_steps, return_head_waist=cfg.infer.return_head_waist, joint_dim=cfg.infer.joint_dim)]
    task_flow = build_WAM_policy_task_flow(cfg=cfg, cotrain_schedule=None, transforms=transforms, output_transforms=output_transforms)
    inferrer = DeployInferrer(run_name=settings.run_name, infer_ckpt_cfg=dict(pretrain_model=settings.infer_checkpoint_dir, to_cuda=True), infer_task_flow=task_flow, infer_type='deploy')
