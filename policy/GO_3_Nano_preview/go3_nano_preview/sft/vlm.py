from __future__ import annotations
from typing import List, Optional, Tuple
import torch
from einops import rearrange
from policy_core.model.WAM_policy.vlm_patches.vlm_token_select import slice_vlm_text_part, compute_fixed_text_length
from policy_core.trainer.WAM_policy_helpers.vlm_bbox_inputs import build_bbox_vlm_inputs

def forward_vlm_features(interface, **kwargs):
    from policy_core.model.WAM_policy.vlm_patches.qwen3_5_interface import _QWen3_5_Interface
    if isinstance(interface, _QWen3_5_Interface):
        return interface.forward_features(**kwargs)
    return interface(**kwargs)

def compute_vlm_hidden_states(video_diffusion_model, video: torch.Tensor, captions, *, use_vlm: bool, vlm_text_only: bool, batch_size: int, n_view: int, mem_size: int, with_grad: bool=False, bboxes: torch.Tensor | None=None, bbox_crop_keep_mask: torch.Tensor | None=None, bbox_crop_images: torch.Tensor | None=None, bbox_reference_images: torch.Tensor | None=None, cpu_inputs: Optional[dict]=None, use_fixed_text_length: bool=True) -> Tuple[Optional[dict], Optional[List[torch.Tensor]], Optional[torch.Tensor]]:
    if not use_vlm or video_diffusion_model is None:
        return (None, None, None)
    vgen_unwrapped = video_diffusion_model.module if hasattr(video_diffusion_model, 'module') else video_diffusion_model
    if getattr(vgen_unwrapped, 'vlm_interface', None) is None:
        return (None, None, None)
    if bool(getattr(vgen_unwrapped, 'use_bbox_condition', False)):
        if vlm_text_only:
            raise ValueError('bbox conditioning requires VLM image inputs')
        vlm_inputs = build_bbox_vlm_inputs(vgen_unwrapped.vlm_interface, video=video, bboxes=bboxes, frame_index=mem_size - 1, head_view_index=int(getattr(vgen_unwrapped, 'vlm_head_view_index', 0)), instructions=captions, crop_keep_mask=bbox_crop_keep_mask, crop_images=bbox_crop_images, bbox_reference_images=bbox_reference_images, bbox_add_image_description=bool(getattr(vgen_unwrapped, 'bbox_add_image_description', True)))
    elif cpu_inputs is not None:
        from transformers.feature_extraction_utils import BatchFeature
        vlm_inputs = BatchFeature(data=cpu_inputs).to(vgen_unwrapped.vlm_interface.model.device)
    elif vlm_text_only:
        vlm_inputs = vgen_unwrapped.vlm_interface.build_qwenvl_inputs(images=None, instructions=captions, drop_images=True)
    else:
        vlm_images = rearrange(video[:, :, :, mem_size - 1, :, :], 'b c v h w -> b v h w c')
        if bool(getattr(vgen_unwrapped, 'vlm_head_view_only', False)):
            _head_idx = int(getattr(vgen_unwrapped, 'vlm_head_view_index', 0))
            vlm_images = vlm_images[:, _head_idx:_head_idx + 1]
        vlm_images = (vlm_images.data.cpu() + 1) / 2 * 255.0
        vlm_inputs = vgen_unwrapped.vlm_interface.build_qwenvl_inputs(images=vlm_images, instructions=captions)
    grad_ctx = torch.enable_grad() if with_grad else torch.no_grad()
    with grad_ctx:
        vlm_outputs = forward_vlm_features(vgen_unwrapped.vlm_interface, **vlm_inputs, output_attentions=False, output_hidden_states=True, return_dict=True)
        _vlm = vgen_unwrapped.vlm_interface
        if hasattr(_vlm, '_hs_hook_collector') and _vlm._hs_hook_collector:
            hs = tuple(_vlm._hs_hook_collector)
            _vlm._hs_hook_collector = []
        else:
            hs = vlm_outputs.hidden_states
        bv = batch_size * n_view
        vlm_attn_mask = vlm_inputs.get('attention_mask', None) if hasattr(vlm_inputs, 'get') else None
        if bool(getattr(vgen_unwrapped, 'vlm_text_tokens_only', False)):
            _fixed_tl = None
            _vlm_fsl = getattr(vgen_unwrapped.vlm_interface, 'fixed_sequence_length', None)
            if use_fixed_text_length and _vlm_fsl is not None:
                _fixed_tl = compute_fixed_text_length(vlm_inputs['input_ids'], _vlm_fsl)
            (hs, vlm_attn_mask) = slice_vlm_text_part(hs, vlm_attn_mask, vlm_inputs['input_ids'], vgen_unwrapped.vlm_interface.model.config.vision_end_token_id, fixed_text_length=_fixed_tl)
        if bv // hs[0].shape[0] != 1:
            hs = [_h for _h in hs]
        else:
            hs = [_h.view(bv // n_view, -1, _h.size(-1)) for _h in hs]
            if vlm_attn_mask is not None:
                vlm_attn_mask = vlm_attn_mask.reshape(bv // n_view, -1)
        if vlm_attn_mask is not None:
            vlm_attn_mask = vlm_attn_mask.detach()
        if with_grad:
            precomputed_hs = list(hs)
        else:
            precomputed_hs = [h.detach() for h in hs]
    return (vlm_inputs, precomputed_hs, vlm_attn_mask)
