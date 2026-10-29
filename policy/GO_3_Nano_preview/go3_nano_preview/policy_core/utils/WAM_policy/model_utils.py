from typing import Dict, Optional, Union
import json
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from policy_core.settings import settings
from safetensors.torch import save_model, load_file, save_file


def reduce_tensor(
    tensor: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()) or settings.world_size == 1:
        return tensor
    out = tensor.detach().clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    if reduction == "mean":
        out = out / settings.world_size
    elif reduction != "sum":
        raise ValueError(f"Unsupported reduction: {reduction}")
    return out


def count_model_parameters(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def load_index_file(index_filename):
    checkpoint_folder = os.path.split(index_filename)[0]
    with open(index_filename) as f:
        index = json.loads(f.read())

    if "weight_map" in index:
        index = index["weight_map"]
    checkpoint_files = sorted(list(set(index.values())))
    checkpoint_files = [os.path.join(checkpoint_folder, f) for f in checkpoint_files]
    state_dict = {}
    for checkpoint_file in checkpoint_files:
        state_dict.update(load_file(checkpoint_file))
    return state_dict


def _find_mismatched_keys(
    state_dict,
    model_state_dict,
    loaded_keys,
):
    mismatched_keys = []
    for checkpoint_key in loaded_keys:
        model_key = checkpoint_key

        if (
            model_key in model_state_dict
            and state_dict[checkpoint_key].shape != model_state_dict[model_key].shape
        ):
            mismatched_keys.append(
                (checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[model_key].shape)
            )
            del state_dict[checkpoint_key]

    return mismatched_keys


def load_checkpoints(model, pretrained_ckpt, strict=False, ignore_mismatched_sizes=True):
    if os.path.isdir(pretrained_ckpt):
        st_index = os.path.join(
            pretrained_ckpt, "diffusion_pytorch_model.safetensors.index.json"
        )
        st_single = os.path.join(pretrained_ckpt, "diffusion_pytorch_model.safetensors")
        bin_single = os.path.join(pretrained_ckpt, "diffusion_pytorch_model.bin")
        if os.path.isfile(st_index):
            state_dict = load_index_file(st_index)
        elif os.path.isfile(st_single):
            state_dict = load_file(st_single)
        elif os.path.isfile(bin_single):
            state_dict = torch.load(bin_single, map_location="cpu")
        else:
            raise FileNotFoundError(
                f"No diffusion weights under {pretrained_ckpt}: expected "
                "diffusion_pytorch_model.bin, .safetensors, or "
                ".safetensors.index.json"
            )
    elif pretrained_ckpt.endswith("safetensors"):
        state_dict = load_file(pretrained_ckpt)
    else:
        state_dict = torch.load(pretrained_ckpt, map_location="cpu")

    if strict:
        model.load_state_dict(state_dict, strict=True)
    else:
        if ignore_mismatched_sizes:
            model_state_dict = model.state_dict()
            pre_mismatch = dict(state_dict)
            mismatched_keys = _find_mismatched_keys(
                state_dict,
                model_state_dict,
                list(state_dict.keys()),
            )
        else:
            mismatched_keys = []
            pre_mismatch = {}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        partial_copied = []
        with torch.no_grad():
            for key, ckpt_shape, model_shape in mismatched_keys:
                if len(ckpt_shape) != len(model_shape):
                    continue
                target = dict(model.named_parameters()).get(key)
                if target is None:
                    target = dict(model.named_buffers()).get(key)
                if target is None:
                    continue
                src = pre_mismatch[key].to(device=target.device, dtype=target.dtype)
                slices = tuple(slice(0, min(s, t)) for s, t in zip(ckpt_shape, model_shape))
                target[slices].copy_(src[slices])
                partial_copied.append((key, tuple(ckpt_shape), tuple(model_shape)))

        print(">>> mismatched_keys: %s" % mismatched_keys)
        print(">>> partial_copied (overlap preserved): %s" % partial_copied)
        print(">>> missing: %s" % missing)
        print(">>> unexpected: %s" % unexpected)

        if missing and hasattr(model, "init_missing_v_head_from_u_head"):
            model.init_missing_v_head_from_u_head(missing)
    print(">>> Loaded weights from pretrained checkpoint: %s"%pretrained_ckpt)


def load_condition_models(
    tokenizer_class,
    textenc_class,
    model_id: str = "a-r-r-o-w/LTX-Video-0.9.1-diffusers",
    text_encoder_dtype: torch.dtype = torch.bfloat16,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    load_weights: bool = True,
    **kwargs,
) -> Dict[str, nn.Module]:
    tokenizer = tokenizer_class.from_pretrained(
        model_id,
        subfolder="tokenizer",
        revision=revision,
        cache_dir=cache_dir
    )

    if load_weights:
        text_encoder = textenc_class.from_pretrained(
            model_id,
            subfolder="text_encoder",
            torch_dtype=text_encoder_dtype,
            revision=revision,
            cache_dir=cache_dir
        )
    else:
        config = textenc_class.config_class.from_pretrained(
            model_id,
            subfolder="text_encoder",
            revision=revision,
            cache_dir=cache_dir
        )
        text_encoder = textenc_class(config)

    return {"tokenizer": tokenizer, "text_encoder": text_encoder}


def load_latent_models(
    model_cls,
    model_id,
    vae_dtype: torch.dtype = torch.bfloat16,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, nn.Module]:
    vae = model_cls.from_pretrained(
        model_id, subfolder="vae", torch_dtype=vae_dtype, revision=revision, cache_dir=cache_dir
    )
    return {"vae": vae}


def load_diffusion_model(model_cls, model_dir, load_weights=True, **kwargs):
    model = model_cls(**kwargs)
    if load_weights:
        print("Load pretrained model weights: ", model_dir)
        load_checkpoints(model, pretrained_ckpt=model_dir)
    return model


def load_vae_models(model_cls, model_dir, load_weights=True):
    with open(os.path.join(model_dir, 'config.json'), 'r', encoding='utf-8') as f:
        vae_kwargs = json.load(f)
    model = model_cls(**vae_kwargs)
    if load_weights:
        load_checkpoints(model, pretrained_ckpt=os.path.join(model_dir, 'diffusion_pytorch_model.safetensors'))
    else:
        print('You are not loading the weights of the vae model, please check your code.')
        pass
    return model


def forward_pass(
    model,
    prompt_embeds: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    noisy_latents: torch.Tensor,
    timesteps: torch.LongTensor,
    num_frames: int,
    height: int,
    width: int,
    n_view: int = 1,
    frame_rate = 5,
    temporal_compression_ratio = 1,
    spatial_compression_ratio = 64,
    model_forward=None,
    **kwargs,
) -> torch.Tensor:
    latent_frame_rate = frame_rate / temporal_compression_ratio
    rope_interpolation_scale = [1.0, 1.0, 1.0]
    forward_callable = model if model_forward is None else model_forward
    denoised_latents = forward_callable(
        hidden_states=noisy_latents,
        encoder_hidden_states=prompt_embeds,
        timestep=timesteps,
        encoder_attention_mask=prompt_attention_mask,
        num_frames=num_frames,
        height=height,
        width=width,
        n_view=n_view,
        rope_interpolation_scale=rope_interpolation_scale,
        return_dict=False,
        **kwargs,
    )[0]
    return {"latents": denoised_latents}
