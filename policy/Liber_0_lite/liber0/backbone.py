from __future__ import annotations
import copy
import os
import types
from typing import Any, Optional, Sequence
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
PRETRAINED_PATCH_SIZE = 32
POOLED_PATCH_SIZE = 16
RGB_CHANNELS = 3
def _linear_fp32(linear: nn.Linear, value: torch.Tensor) -> torch.Tensor:
    bias = None if linear.bias is None else linear.bias.float()
    return F.linear(value.float(), linear.weight.float(), bias)

def _validate_patch_weight_pooling(patch_size: int, mode: str) -> tuple[int, str]:
    patch_size = int(patch_size)
    mode = str(mode)
    if (patch_size, mode) in {(PRETRAINED_PATCH_SIZE, 'none'), (POOLED_PATCH_SIZE, 'target_only_global_norm')}:
        return (patch_size, mode)
    raise ValueError(f'Liber_0_lite patch configuration must be patch32 with no pooling or patch16 with target_only_global_norm, got patch_size={patch_size}, patch_weight_pooling={mode!r}.')

def _pool_patch32_weights_to_patch16(input_weight: torch.Tensor, output_weight: torch.Tensor, output_bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    old_area = PRETRAINED_PATCH_SIZE * PRETRAINED_PATCH_SIZE
    new_area = POOLED_PATCH_SIZE * POOLED_PATCH_SIZE
    if input_weight.shape[1] != RGB_CHANNELS * old_area:
        raise ValueError(f'Unexpected patch32 input weight shape: {tuple(input_weight.shape)}')
    if output_weight.shape[0] != RGB_CHANNELS * old_area:
        raise ValueError(f'Unexpected patch32 output weight shape: {tuple(output_weight.shape)}')
    if output_bias.shape != (RGB_CHANNELS * old_area,):
        raise ValueError(f'Unexpected patch32 output bias shape: {tuple(output_bias.shape)}')
    pooled_input = input_weight.float().reshape(input_weight.shape[0], RGB_CHANNELS, POOLED_PATCH_SIZE, 2, POOLED_PATCH_SIZE, 2).sum(dim=(3, 5)).reshape(input_weight.shape[0], RGB_CHANNELS * new_area)
    target_projection_scale = float(input_weight.float().norm().div(pooled_input.norm()).item())
    pooled_output = output_weight.float().reshape(RGB_CHANNELS, POOLED_PATCH_SIZE, 2, POOLED_PATCH_SIZE, 2, output_weight.shape[1]).mean(dim=(2, 4)).reshape(RGB_CHANNELS * new_area, output_weight.shape[1])
    pooled_bias = output_bias.float().reshape(RGB_CHANNELS, POOLED_PATCH_SIZE, 2, POOLED_PATCH_SIZE, 2).mean(dim=(2, 4)).reshape(RGB_CHANNELS * new_area)
    return (pooled_input.to(input_weight.dtype), pooled_output.to(output_weight.dtype), pooled_bias.to(output_bias.dtype), target_projection_scale)

def _install_target_only_global_norm_patch16(backbone: nn.Module) -> float:
    old_proj1 = backbone.model.x_embedder.proj1
    old_final = backbone.model.final_layer2.linear
    (pooled_input, pooled_output, pooled_bias, target_projection_scale) = _pool_patch32_weights_to_patch16(old_proj1.weight.detach(), old_final.weight.detach(), old_final.bias.detach())
    new_proj1 = nn.Linear(RGB_CHANNELS * POOLED_PATCH_SIZE * POOLED_PATCH_SIZE, old_proj1.out_features, bias=False, device=old_proj1.weight.device, dtype=old_proj1.weight.dtype)
    new_final = nn.Linear(old_final.in_features, RGB_CHANNELS * POOLED_PATCH_SIZE * POOLED_PATCH_SIZE, bias=True, device=old_final.weight.device, dtype=old_final.weight.dtype)
    with torch.no_grad():
        new_proj1.weight.copy_(pooled_input)
        new_final.weight.copy_(pooled_output)
        new_final.bias.copy_(pooled_bias)
    backbone.model.x_embedder.proj1 = new_proj1
    backbone.model.final_layer2.linear = new_final
    backbone.model.patch_size = POOLED_PATCH_SIZE
    return target_projection_scale

def _build_raw_image_rope_layout(*, target_grid: torch.Tensor, reference_grids: Sequence[torch.Tensor], target_len: int, reference_lengths: Sequence[int], image_token_id: int, vision_start_token_id: int, batch_size: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    if len(reference_grids) != len(reference_lengths):
        raise ValueError(f'Reference grid/length mismatch: {len(reference_grids)} vs {len(reference_lengths)}.')
    target_grids = target_grid.reshape(-1, 3)
    target_lengths = tuple((int(grid.prod()) for grid in target_grids))
    if sum(target_lengths) != int(target_len):
        raise ValueError('Target grids do not match the total target patch length.')
    raw_grids = torch.stack([*target_grids, *reference_grids])
    total_ref_len = sum((int(value) for value in reference_lengths))
    raw_tokens = torch.full((batch_size, int(target_len) + total_ref_len), int(image_token_id), dtype=dtype)
    offset = 0
    for image_len in (*target_lengths, *reference_lengths):
        raw_tokens[:, offset] = int(vision_start_token_id)
        offset += int(image_len)
    reference_count = len(reference_lengths)
    skip_vision_start_token = [0] * reference_count + [1] * (len(target_lengths) + reference_count)
    return (raw_tokens, raw_grids, skip_vision_start_token)

def _configure_ieee_float32_matmul() -> None:
    if os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE') != '0':
        raise RuntimeError('Export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 before starting Python.')
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.get_float32_matmul_precision() == 'highest'
    assert not torch.backends.cuda.matmul.allow_tf32

def _install_portable_inference_rope(backbone: nn.Module) -> None:
    rotary = backbone.model.language_model.rotary_emb
    original_forward = rotary.forward
    inv_freq_cpu = rotary.original_inv_freq.detach().to(device='cpu', dtype=torch.float32)

    def forward(module, x: torch.Tensor, position_ids: torch.Tensor):
        if torch.is_grad_enabled():
            return original_forward(x, position_ids)
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq = inv_freq_cpu[None, None, :, None].expand(3, position_ids.shape[1], -1, 1)
        positions = position_ids.to(device='cpu', dtype=torch.float32)[:, :, None, :]
        freqs = (inv_freq @ positions).transpose(2, 3)
        freqs = module.apply_interleaved_mrope(freqs, module.mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = (emb.cos() * module.attention_scaling).to(device=x.device, dtype=x.dtype)
        sin = (emb.sin() * module.attention_scaling).to(device=x.device, dtype=x.dtype)
        return (cos, sin)
    rotary.forward = types.MethodType(forward, rotary)

class JointModel(nn.Module):
    """Liber_0_lite backbone plus action/proprio interfaces."""

    def __init__(self, backbone: nn.Module, action_dim: int, proprio_dim: int, max_action_horizon: int) -> None:
        super().__init__()
        hidden_dim = int(backbone.config.text_config.hidden_size)
        self.backbone = backbone
        self.action_dim = int(action_dim)
        self.proprio_dim = int(proprio_dim)
        self.max_action_horizon = int(max_action_horizon)
        self.action_encoder = nn.Linear(self.action_dim, hidden_dim)
        self.proprio_encoder = nn.Linear(self.proprio_dim, hidden_dim)
        self.action_time_embedder = copy.deepcopy(backbone.model.t_embedder1)
        self.action_type_embedding = nn.Parameter(torch.zeros(hidden_dim))
        self.proprio_type_embedding = nn.Parameter(torch.zeros(hidden_dim))
        self.action_norm = nn.LayerNorm(hidden_dim, eps=1e-06)
        self.action_head = nn.Linear(hidden_dim, self.action_dim)

class LiberModel(nn.Module):

    @property
    def backbone(self):
        return self.dit.backbone

    def _language_model_forward(self, **kwargs):
        return self.backbone.model.language_model(**kwargs)

    @staticmethod
    def _to_pil(image: torch.Tensor) -> Image.Image:
        value = ((image.detach().float().cpu() + 1.0) * 127.5).round().clamp(0, 255)
        return Image.fromarray(value.permute(1, 2, 0).byte().numpy())

    @staticmethod
    def _patchify(images: torch.Tensor, patch_size: int=PRETRAINED_PATCH_SIZE) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f'Images must be [B,C,H,W], got {tuple(images.shape)}.')
        (batch_size, channels, height, width) = images.shape
        patch_size = int(patch_size)
        if channels != RGB_CHANNELS or height % patch_size or width % patch_size:
            raise ValueError(f'Liber_0_lite images must be RGB and divisible by {patch_size}, got {tuple(images.shape)}.')
        return images.reshape(batch_size, channels, height // patch_size, patch_size, width // patch_size, patch_size).permute(0, 2, 4, 1, 3, 5).reshape(batch_size, height // patch_size * (width // patch_size), channels * patch_size * patch_size)

    @staticmethod
    def _unpatchify(patches: torch.Tensor, height: int, width: int, patch_size: int=PRETRAINED_PATCH_SIZE) -> torch.Tensor:
        patch_size = int(patch_size)
        return patches.reshape(patches.shape[0], height // patch_size, width // patch_size, RGB_CHANNELS, patch_size, patch_size).permute(0, 3, 1, 4, 2, 5).reshape(patches.shape[0], RGB_CHANNELS, height, width)

    def _embed_video_tokens(self, noisy_video: torch.Tensor, reference_patches: torch.Tensor) -> torch.Tensor:
        embedder = self.backbone.model.x_embedder
        target_projection = embedder.proj1(noisy_video) * self.target_projection_scale
        reference_projection = embedder.proj1(reference_patches)
        return embedder.proj2(torch.cat([target_projection, reference_projection], dim=1))

    def _apply_copy_gate(self, target_hidden: torch.Tensor, generated_x0: torch.Tensor, reference_patches: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        return (generated_x0, None)

    def _select_reference_images(self, mosaic_reference: torch.Tensor, reference_images: Optional[dict[str, torch.Tensor]]) -> tuple[torch.Tensor, ...]:
        if reference_images is not None:
            raise ValueError("reference_images must be omitted when reference_image_mode='mosaic'.")
        return (mosaic_reference,)

    def _target_grids(self, reference_images, height, width, target_len):
        sizes = [(height, width)]
        grids = torch.tensor([[1, h // self.patch_size, w // self.patch_size] for (h, w) in sizes])
        if any((h % self.patch_size or w % self.patch_size for (h, w) in sizes)):
            raise ValueError('Target image sizes must be divisible by the model patch size.')
        if int(grids.prod(-1).sum()) != int(target_len):
            raise ValueError('Target image grids do not match the packed target length.')
        return grids

    def _patch_reference_images(self, reference_images: Sequence[torch.Tensor]) -> tuple[torch.Tensor, tuple[int, ...]]:
        patch_groups = tuple((self._patchify(image, self.patch_size) for image in reference_images))
        lengths = tuple((int(patches.shape[1]) for patches in patch_groups))
        return (torch.cat(patch_groups, dim=1), lengths)

    def _conditioning_prompt(self, prompt, reference_count):
        return prompt

    def _process_semantic_references(self, templates, reference_images, *, padding=False):
        reference_images = tuple((image.detach().cpu() for image in reference_images))
        images = []
        for batch_index in range(len(templates)):
            for reference in reference_images:
                (height, width) = reference.shape[-2:]
                image = self._to_pil(reference[batch_index])
                from .vendor.rope import calculate_dimensions
                (cond_width, cond_height) = calculate_dimensions(384, width / height)
                image = image.resize((cond_width, cond_height), Image.Resampling.LANCZOS)
                images.append(image)
        processed = self.processor(text=templates, images=images, padding=padding, return_tensors='pt')
        return processed

    def _get_rope_index(self, *args, **kwargs):
        from .vendor.rope import get_rope_index_fix_point

        def freeze(value):
            if isinstance(value, torch.Tensor):
                assert value.device.type == 'cpu'
                return (tuple(value.shape), str(value.dtype), value.numpy().tobytes())
            if isinstance(value, (list, tuple)):
                return tuple((freeze(item) for item in value))
            return value
        key = (freeze(args), tuple(((key, freeze(value)) for (key, value) in sorted(kwargs.items()))))
        cache = self._rope_index_cache
        if key in cache:
            cache.move_to_end(key)
            return tuple((value.clone() for value in cache[key]))
        result = get_rope_index_fix_point(*args, **kwargs)
        cache[key] = tuple((value.clone() for value in result))
        if len(cache) > 256:
            cache.popitem(last=False)
        return result

    def _prepare_condition_static(self, prompt: str, reference_images: Sequence[torch.Tensor], target_height: int, target_width: int, target_len: int, ref_lengths: Sequence[int]) -> dict[str, Any]:
        reference_count = len(reference_images)
        if reference_count != len(ref_lengths):
            raise ValueError(f'Reference image/length mismatch: {reference_count} vs {len(ref_lengths)}.')
        raw_ref_grids = []
        for (image, ref_len) in zip(reference_images, ref_lengths):
            (height, width) = (int(image.shape[-2]), int(image.shape[-1]))
            raw_ref_grids.append(torch.tensor([1, height // self.patch_size, width // self.patch_size]))
            expected_ref_len = height // self.patch_size * (width // self.patch_size)
            if int(ref_len) != expected_ref_len:
                raise ValueError(f'Reference patch length mismatch: {ref_len} vs {expected_ref_len}.')
        tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else self.processor
        content = [{'type': 'image'} for _ in reference_images]
        content.append({'type': 'text', 'text': self._conditioning_prompt(prompt, reference_count)})
        template = self.processor.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)
        processed = self._process_semantic_references([template], reference_images)
        suffix = tokenizer.encode(tokenizer.boi_token + tokenizer.tms_token, return_tensors='pt', add_special_tokens=False)
        input_ids_cpu = torch.cat([processed.input_ids, suffix], dim=-1)
        spatial_merge = int(self.backbone.config.vision_config.spatial_merge_size)
        semantic_grids = processed.image_grid_thw.clone()
        semantic_grids[:, 1:] //= spatial_merge
        if int(semantic_grids.shape[0]) != reference_count:
            raise ValueError(f'Processor returned {semantic_grids.shape[0]} image grids for {reference_count} reference images.')
        target_grid = self._target_grids(reference_images, target_height, target_width, target_len)
        image_token = int(self.backbone.config.image_token_id)
        vision_start = int(self.backbone.config.vision_start_token_id)
        (raw_tokens, raw_grids, skip_vision_start_token) = _build_raw_image_rope_layout(target_grid=target_grid, reference_grids=raw_ref_grids, target_len=target_len, reference_lengths=ref_lengths, image_token_id=image_token, vision_start_token_id=vision_start, batch_size=1, dtype=input_ids_cpu.dtype)
        all_grids = torch.cat([semantic_grids, raw_grids], dim=0)
        (position_ids, _) = self._get_rope_index(1, image_token, int(self.backbone.config.video_token_id), vision_start, input_ids=torch.cat([input_ids_cpu, raw_tokens], dim=-1), image_grid_thw=all_grids, video_grid_thw=None, attention_mask=None, skip_vision_start_token=skip_vision_start_token)
        input_ids = input_ids_cpu.to(self.device)
        text_embeds = self.backbone.model.get_input_embeddings()(input_ids)
        (image_embeds, deepstack) = self.backbone.model.get_image_features(processed.pixel_values.to(self.device, self.torch_dtype), processed.image_grid_thw.to(self.device))
        image_embeds = torch.cat(image_embeds, dim=0).to(text_embeds.dtype)
        (image_mask, _) = self.backbone.model.get_placeholder_mask(input_ids, inputs_embeds=text_embeds, image_features=image_embeds)
        text_embeds = text_embeds.masked_scatter(image_mask, image_embeds)
        return {'text_embeds_base': text_embeds, 'tms_mask': input_ids == int(self.backbone.model.tms_token_id), 'positions': position_ids.to(self.device), 'visual_mask': image_mask[..., 0].to(torch.bool), 'deepstack': [value.to(self.device, self.torch_dtype) for value in deepstack]}

    def _condition_with_timestep(self, condition_static: dict[str, Any], video_clean_t: torch.Tensor) -> dict[str, Any]:
        text_embeds_base = condition_static['text_embeds_base']
        time_embed = self.backbone.model.t_embedder1(video_clean_t.reshape(1)).to(text_embeds_base.dtype)
        return {'text_embeds': torch.where(condition_static['tms_mask'].unsqueeze(-1), time_embed.unsqueeze(1).expand_as(text_embeds_base), text_embeds_base), 'positions': condition_static['positions'], 'visual_mask': condition_static['visual_mask'], 'deepstack': condition_static['deepstack']}

    def _prepare_condition(self, prompt: str, reference_images: Sequence[torch.Tensor], video_clean_t: torch.Tensor, target_height: int, target_width: int, target_len: int, ref_lengths: Sequence[int]) -> dict[str, Any]:
        condition_static = self._prepare_condition_static(prompt, reference_images, target_height, target_width, target_len, ref_lengths)
        return self._condition_with_timestep(condition_static, video_clean_t)

    @staticmethod
    def _joint_mask(text_len: int, target_len: int, ref_len: int, action_len: int, device: torch.device, dtype: torch.dtype, joint_layout: str='legacy') -> torch.Tensor:
        from .attention import build_batched_joint_mask
        return build_batched_joint_mask(torch.ones(1, text_len, device=device, dtype=torch.bool), target_len, ref_len, action_len, dtype, joint_layout=joint_layout)

    def _joint_positions(self, base_positions, text_valid, target_len, ref_len, action_len):
        text_len = text_valid.shape[1]
        text_positions = base_positions[:, :, :text_len]
        video_positions = base_positions[:, :, text_len:text_len + target_len + ref_len]
        video_positions = video_positions + 1
        valid_positions = text_positions.masked_fill(~text_valid.unsqueeze(0), 0)
        proprio_position = valid_positions.amax(dim=2, keepdim=True) + 1
        action_base = video_positions.amax(dim=(0, 2)) + 1
        action_axis = torch.arange(action_len, device=base_positions.device).unsqueeze(0)
        action_time = action_base.unsqueeze(1) + action_axis
        action_space = action_base.unsqueeze(1).expand(-1, action_len)
        action_positions = torch.stack([action_time, action_space, action_space], dim=0)
        return torch.cat([text_positions, proprio_position, video_positions, action_positions], dim=2)

    def _pad_joint_sequence(self, embeds, positions, attention_mask, visual_mask):
        padding = -embeds.shape[1] % self.joint_token_padding_multiple
        if not padding:
            return (embeds, positions, attention_mask, visual_mask)
        embeds = F.pad(embeds, (0, 0, 0, padding))
        positions = F.pad(positions, (0, padding))
        visual_mask = F.pad(visual_mask, (0, padding), value=False)
        attention_mask = F.pad(attention_mask, (0, padding, 0, padding), value=torch.finfo(attention_mask.dtype).min)
        attention_mask[..., -padding:, -padding:].diagonal(dim1=-2, dim2=-1).zero_()
        return (embeds, positions, attention_mask, visual_mask)

    def _embed_action_tokens(self, noisy_action, action_clean_t):
        with torch.autocast(device_type=noisy_action.device.type, enabled=False):
            tokens = _linear_fp32(self.dit.action_encoder, noisy_action)
            embedder = self.dit.action_time_embedder
            frequency = embedder.timestep_embedding(action_clean_t.reshape(-1).float() * 1000, embedder.frequency_embedding_size)
            time = _linear_fp32(embedder.mlp[2], F.silu(_linear_fp32(embedder.mlp[0], frequency)))
            tokens = tokens + time.unsqueeze(1)
            tokens = tokens + self.dit.action_type_embedding.float()
        return tokens.to(self.torch_dtype)

    def _predict_action(self, hidden):
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            norm = self.dit.action_norm
            normalized = F.layer_norm(hidden.float(), norm.normalized_shape, norm.weight.float(), norm.bias.float(), norm.eps)
            return _linear_fp32(self.dit.action_head, normalized)

    def _joint_forward_single(self, prompt: str, reference_images: Sequence[torch.Tensor], target_height: int, target_width: int, proprio: torch.Tensor, noisy_video: torch.Tensor, video_clean_t: torch.Tensor, noisy_action: torch.Tensor, action_clean_t: torch.Tensor, condition_static: Optional[dict[str, Any]]=None) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        target_len = int(noisy_video.shape[1])
        (ref_patches, ref_lengths) = self._patch_reference_images(reference_images)
        ref_len = int(ref_patches.shape[1])
        action_len = int(noisy_action.shape[1])
        if action_len > self.dit.max_action_horizon:
            raise ValueError(f'Action length {action_len} exceeds {self.dit.max_action_horizon}.')
        if condition_static is None:
            condition = self._prepare_condition(prompt, reference_images, video_clean_t, target_height, target_width, target_len, ref_lengths)
        else:
            condition = self._condition_with_timestep(condition_static, video_clean_t)
        text_len = int(condition['text_embeds'].shape[1])
        video_tokens = self._embed_video_tokens(noisy_video.to(self.torch_dtype), ref_patches.to(self.torch_dtype))
        action_tokens = self._embed_action_tokens(noisy_action, action_clean_t)
        proprio_token = self.dit.proprio_encoder(proprio.to(self.torch_dtype)).unsqueeze(1) + self.dit.proprio_type_embedding
        embeds = torch.cat([condition['text_embeds'], proprio_token, video_tokens, action_tokens], dim=1)
        positions = self._joint_positions(condition['positions'], torch.ones(1, text_len, device=self.device, dtype=torch.bool), target_len, ref_len, action_len)
        visual_mask = torch.cat([condition['visual_mask'], torch.zeros(1, 1 + target_len + ref_len + action_len, device=self.device, dtype=torch.bool)], dim=1)
        attention_mask = self._joint_mask(text_len, target_len, ref_len, action_len, self.device, self.torch_dtype, joint_layout='legacy')
        (embeds, positions, attention_mask, visual_mask) = self._pad_joint_sequence(embeds, positions, attention_mask, visual_mask)
        output = self._language_model_forward(input_ids=None, position_ids=positions, attention_mask=attention_mask, inputs_embeds=embeds, use_cache=False, visual_pos_masks=visual_mask, deepstack_visual_embeds=condition['deepstack']).last_hidden_state
        video_start = text_len + 1
        action_start = video_start + target_len + ref_len
        target_hidden = output[:, video_start:video_start + target_len]
        generated_x0 = self.backbone.model.final_layer2(target_hidden)
        (pred_x0, copy_gate) = self._apply_copy_gate(target_hidden, generated_x0, ref_patches)
        pred_action = self._predict_action(output[:, action_start:action_start + action_len])
        return (pred_x0, pred_action, copy_gate)

    def _action_model_output_to_velocity(self, model_output: torch.Tensor, noisy_action: torch.Tensor, action_sigma: torch.Tensor) -> torch.Tensor:
        return model_output.to(noisy_action.dtype)

    @torch.no_grad()
    def infer_joint(self, prompt: Optional[str], input_image: Optional[torch.Tensor], action_horizon: int, proprio: Optional[torch.Tensor]=None, reference_images: Optional[dict[str, torch.Tensor]]=None, context: Optional[torch.Tensor]=None, context_mask: Optional[torch.Tensor]=None, num_inference_steps: int=30, sigma_shift: Optional[float]=None, seed: Optional[int]=None, rand_device: str='cpu', text_cfg_scale: float=1.0, negative_prompt: str='', **_: Any) -> dict[str, Any]:
        if float(text_cfg_scale) != 1.0 or negative_prompt:
            raise ValueError('Liber_0_lite action inference supports text_cfg_scale=1 and an empty negative_prompt only; CFG is not trained for this action head.')
        if context is not None or context_mask is not None:
            raise ValueError('Liber_0_lite inference requires a raw prompt.')
        if prompt is None:
            raise ValueError('Liber_0_lite inference requires prompt.')
        if input_image is None:
            raise ValueError('input_image is required for mosaic target inference.')
        else:
            if input_image.ndim == 3:
                input_image = input_image.unsqueeze(0)
            mosaic_reference = input_image.to(self.device, self.sample_dtype)
            selected_reference_images = self._select_reference_images(mosaic_reference, reference_images)
            (height, width) = mosaic_reference.shape[-2:]
        proprio_tensor = torch.zeros(1, self.dit.proprio_dim, device=self.device, dtype=self.sample_dtype)
        if proprio is not None:
            proprio_tensor = proprio.reshape(1, -1).to(self.device, self.sample_dtype)
        generator = None if seed is None else torch.Generator(rand_device).manual_seed(seed)
        video = torch.randn((1, 3, height, width), generator=generator, device=rand_device).to(self.device, self.sample_dtype) * self.noise_scale
        noisy_patches = self._patchify(video, self.patch_size)
        action = torch.randn((1, action_horizon, self.dit.action_dim), generator=generator, device=rand_device).to(self.device, self.sample_dtype)
        (video_t, video_delta) = self.infer_video_scheduler.build_inference_schedule(num_inference_steps, self.device, self.sample_dtype, sigma_shift)
        (action_t, action_delta) = self.infer_action_scheduler.build_inference_schedule(num_inference_steps, self.device, self.sample_dtype, sigma_shift)
        target_len = noisy_patches.shape[1]
        (_, ref_lengths) = self._patch_reference_images(selected_reference_images)
        condition_static = self._prepare_condition_static(prompt, selected_reference_images, int(height), int(width), target_len, ref_lengths)
        for i in range(num_inference_steps):
            video_sigma = video_t[i:i + 1] / self.infer_video_scheduler.num_train_timesteps
            action_sigma = action_t[i:i + 1] / self.infer_action_scheduler.num_train_timesteps
            (pred_x0, action_model_output, _) = self._joint_forward_single(prompt, selected_reference_images, int(height), int(width), proprio_tensor, noisy_patches, 1 - video_sigma, action, 1 - action_sigma, condition_static=condition_static)
            pred_video = (noisy_patches - pred_x0) / video_sigma[:, None, None].clamp_min(self.timestep_eps)
            noisy_patches = self.infer_video_scheduler.step(pred_video, video_delta[i], noisy_patches)
            pred_action = self._action_model_output_to_velocity(action_model_output, action, action_sigma)
            action = self.infer_action_scheduler.step(pred_action, action_delta[i], action)
        output = {'action': action[0].cpu().float()}
        video = self._unpatchify(noisy_patches, height, width, self.patch_size)
        output['image'] = video[0].cpu().float().clamp(-1, 1)
        return output

    @torch.no_grad()
    def infer_action(self, **kwargs) -> dict[str, Any]:
        return {'action': self.infer_joint(**kwargs)['action']}
