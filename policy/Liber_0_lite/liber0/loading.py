"""Load the released model without external source paths."""
from collections import OrderedDict
from pathlib import Path

import torch
from transformers import AutoProcessor

from .backbone import (
    JointModel, LiberModel, _configure_ieee_float32_matmul,
    _install_portable_inference_rope, _install_target_only_global_norm_patch16,
)
from .scheduler import FlowScheduler
from .vendor.qwen3_vl_transformers import Qwen3VLForConditionalGeneration
from .vision_attention import install_grouped_vision_attention


def load_model(config, assets, checkpoint, device):
    expected = {
        'mask_scheme': 'default', 'proprio_dim': 14, 'action_dim': 14,
        'model_dtype': 'bf16', 'numerical_mode': 'legacy_bf16',
        'joint_layout': 'legacy', 'semantic_image_mode': 'legacy_384',
        'target_image_mode': 'mosaic', 'action_backbone': 'shared',
        'sdpa_backend': 'auto', 'patch_size': 16,
        'patch_weight_pooling': 'target_only_global_norm',
        'use_gradient_checkpointing': False, 'action_prediction_mode': 'velocity',
        'reference_image_mode': 'mosaic', 'copy_gate': {'enabled': False},
        'portable_inference_rope': True, 'vision_attention_mode': 'grouped_cached',
        'action_expert_stability_fp32': True,
    }
    for name, value in expected.items():
        if config[name] != value:
            raise ValueError(f'Unsupported model setting: {name}={config[name]!r}')
    allowed = set(expected) | {'max_action_horizon', 'forward_chunk_size', 'video_scheduler', 'action_scheduler'}
    if set(config) != allowed:
        raise ValueError(f'Unexpected model fields: {set(config) ^ allowed}')
    _configure_ieee_float32_matmul()
    device = torch.device(device)
    processor = AutoProcessor.from_pretrained(str(assets), local_files_only=True)
    tokenizer = processor.tokenizer
    for name in ('boi', 'bor', 'eor', 'bot', 'tms'):
        setattr(tokenizer, name + '_token', '<|' + name + '_token|>')
    tokenizer.padding_side = 'left'
    backbone = Qwen3VLForConditionalGeneration.from_pretrained(
        str(assets), dtype=torch.bfloat16, device_map={'': device.index or 0}, local_files_only=True,
    )
    scale = _install_target_only_global_norm_patch16(backbone)
    backbone.model.patch_size = 16
    _install_portable_inference_rope(backbone)
    dit = JointModel(backbone, config['action_dim'], config['proprio_dim'],
                     config['max_action_horizon']).to(device=device, dtype=torch.bfloat16)
    model = LiberModel()
    model.dit, model.processor = dit, processor
    model.device, model.torch_dtype, model.sample_dtype = device, torch.bfloat16, torch.bfloat16
    model.patch_size, model.target_projection_scale = 16, scale
    model.noise_scale, model.timestep_eps = 8.0, 1e-3
    model.joint_token_padding_multiple = 1
    model.action_expert_stability_fp32 = True
    model._rope_index_cache = OrderedDict()
    video = config['video_scheduler']
    action = config['action_scheduler']
    model.infer_video_scheduler = FlowScheduler(video['num_train_timesteps'], video['infer_shift'])
    model.infer_action_scheduler = FlowScheduler(action['num_train_timesteps'], action['infer_shift'])
    install_grouped_vision_attention(backbone.model.visual, cache_geometry=True)
    backbone.gradient_checkpointing_disable()
    payload = torch.load(Path(checkpoint), map_location='cpu', weights_only=True, mmap=True)
    contract = dict(action_expert_stability_fp32=True, action_backbone='shared',
                    target_image_mode='mosaic', semantic_image_mode='legacy_384',
                    joint_layout='legacy', numerical_mode='legacy_bf16')
    for name, value in contract.items():
        if payload[name] != value:
            raise ValueError(f'Incompatible checkpoint: {name}')
    model.dit.load_state_dict(payload['dit'], strict=True)
    return model.to(device).eval()
