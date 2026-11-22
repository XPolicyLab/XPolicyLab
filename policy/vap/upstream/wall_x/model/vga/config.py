"""Typed config for the internal VGA model family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from wall_x.config.model_config import ModelConfig
from wall_x.config.registry import register_model_config


@register_model_config("wan")
@dataclass
class VGAModelConfig(ModelConfig):
    """VGA/WAN model config."""

    model_paths: Optional[List[str]] = None
    model_id_with_origin_paths: Optional[List[str]] = None
    trainable_models: Optional[List[str]] = None
    enable_action: bool = False
    action_dim: int = 14
    action_transformer_dim: Optional[int] = None
    action_loss_weight: float = 1.0
    video_loss_weight: float = 1.0
    video_loss_weight_start_step: int = 15000
    ignore_video_loss: bool = False
    action_video_block_mapping: Optional[str] = None
    constraint_loss_weight: float = 0.0
    dct_loss_weight: float = 0.0
    separate_flow_matching: bool = False
    action_batch_size: int = 4
    action_pred_type: str = "v_pred"
    second_head_loss_weight: float = 0.0
    timestep_sample_mode: str = "uniform"
    action_beta_alpha: float = 1.0
    action_beta_beta: float = 3.0
    enable_action_training_weight: bool = False
    sync_action_denoising: bool = False
    skip_vae_decode: bool = False
    scheduler_shift: float = 5.0
    share_state_action_encoder: bool = True
    local_alignment_pe: bool = False
    state_attn: bool = False
    action_as_condition: bool = False
    action_tokenizer_ckpt_path: Optional[str] = None
    action_timestep_mapping: Optional[str] = None
    video_early_stop_for_action: bool = True
    video_train_anchors: Optional[Union[str, List[int]]] = None
    video_train_anchor_inference_steps: int = 50
    video_train_window: int = 20
    vga_post_training: bool = False
    vga_post_action_updates_per_video: int = 1
    detach_video_features: bool = True
    fixed_resolution: bool = True
    height: int = 208
    width: int = 224
    num_views: int = 3
    multi_views: bool = True
    extra_inputs: Optional[List[str]] = None
    view_as_tokens: bool = False
    view_attn_block_interval: int = 1
    view_keep_all_prob: float = 1.0
    use_sightcone_matching: bool = False
    cross_view_masking_prob: float = 0.0
    cross_view_masking_condition_prob: float = 0.0
    cross_view_masking_max_len: int = 5
    mask_region_loss_weight: float = 1.0
    physical_view_ids: Optional[Dict[str, int]] = None
    only_train_view_attention: bool = False
    init_action_from_dit: bool = False
    init_cross_attn_q_from_dit: bool = False
    zero_init_cross_attn_o: bool = False
    zero_init_action_output: bool = False
    max_timestep_boundary: float = 1.0
    min_timestep_boundary: float = 0.0
    use_qwen25: bool = False
    qwen25_path: Optional[str] = None
    qwen_proj_out_dim: int = 4096
    qwen25_cascade_t5: bool = False
    qwen25_prompt_template: Optional[str] = None
    print_qwen_output: bool = False
    use_qwen35: bool = False
    qwen35_path: Optional[str] = None
    qwen35_cascade_t5: bool = False
    qwen35_system_prompt: Optional[str] = None
    prompt_template: Optional[str] = None
    prompt_priority_order: Optional[List[str]] = None
    text_feature_cache_path: Optional[str] = None
    lora_base_model: Optional[str] = None
    lora_target_modules: str = "q,k,v,o,ffn.0,ffn.2"
    lora_rank: int = 32
    lora_checkpoint: Optional[str] = None
    use_custom_lr_groups: bool = False
    inference: bool = False
    event_level_weighted_sampling: bool = False
    validation: Optional[Dict[str, Any]] = None
    policy: Optional[Dict[str, Any]] = None


__all__ = ["VGAModelConfig"]
