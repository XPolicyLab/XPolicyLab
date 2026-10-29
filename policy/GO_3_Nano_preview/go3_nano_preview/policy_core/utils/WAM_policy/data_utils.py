from typing import Dict, List, Optional, Union
import random
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from torchvision.transforms import ColorJitter

def _encode_prompt(
    tokenizer,
    text_encoder,
    prompt: List[str],
    device: torch.device,
    dtype: torch.dtype,
    max_sequence_length,
) -> torch.Tensor:
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_attention_mask = text_inputs.attention_mask
    prompt_attention_mask = prompt_attention_mask.bool().to(device)

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)

    return {"prompt_embeds": prompt_embeds, "prompt_attention_mask": prompt_attention_mask}


def prepare_conditions(
    tokenizer,
    text_encoder,
    prompt: Union[str, List[str]],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    max_sequence_length: int = 128,
    **kwargs,
) -> torch.Tensor:
    device = device or text_encoder.device
    dtype = dtype or text_encoder.dtype

    if isinstance(prompt, str):
        prompt = [prompt]

    return _encode_prompt(tokenizer, text_encoder, prompt, device, dtype, max_sequence_length)


def read_img(img_file, target_shape=(512,384)):
    image = Image.open(img_file)
    image = image.resize(target_shape)
    image_array = np.array(image)
    if image_array.shape[0] != 3:
        image_array = np.transpose(image_array, (2,0,1))
    image = torch.from_numpy(image_array)
    image = image.float()/255.0 * 2 - 1
    return image


@torch.no_grad()
def get_text_conditions(
    tokenizer,
    text_encoder,
    prompt: Union[str, List[str]],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    max_sequence_length: int = 128,
    **kwargs,
) -> torch.Tensor:
    device = device or text_encoder.device
    dtype = dtype or text_encoder.dtype

    if isinstance(prompt, str):
        prompt = [prompt]

    return _encode_prompt(tokenizer, text_encoder, prompt, device, dtype, max_sequence_length)


@torch.no_grad()
def get_3dvae_latents(
        vae,
        mem: torch.Tensor = None,
        video: torch.Tensor = None,
        patch_size: int = 1,
        patch_size_t: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        generator: Optional[torch.Generator] = None
    ):
    device = device or vae.device

    if video is not None:
        video_latents = vae.encode(video).latent_dist.sample(generator=generator)
        video_latents = video_latents.to(dtype=dtype)
        video_latents = _normalize_latents(video_latents, vae.latents_mean, vae.latents_std)
        video_latents = _pack_latents(video_latents, patch_size, patch_size_t)
    else:
        video_latents = None

    if mem is not None:
        mem = rearrange(mem, 'b c m h w -> (b m) c h w').unsqueeze(2)
        mem_latents = vae.encode(mem).latent_dist.sample(generator=generator)
        mem_latents = mem_latents.to(dtype=dtype)
        mem_latents = _normalize_latents(mem_latents, vae.latents_mean, vae.latents_std)
        mem_latents = _pack_latents(mem_latents, patch_size, patch_size_t)
    else:
        mem_latents = None

    return mem_latents, video_latents

@torch.no_grad()
def get_latents(
        vae,
        mem: torch.Tensor = None,
        video: torch.Tensor = None,
        patch_size: int = 1,
        patch_size_t: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        generator: Optional[torch.Generator] = None,
        mode="3dvae",
        batch_size_2dvae = 8,
        stats_tracker=None,
        encode_future_video: bool = True,
    ):
    mode = getattr(vae, "vae_encoding_mode", "3dvae")

    if mode.lower().strip() == "3dvae":
        return get_3dvae_latents(vae, mem, video, patch_size, patch_size_t, device, dtype, generator)
    elif mode.lower().strip() == "2dvae":
        mem = rearrange(mem, "bv c m h w -> (bv m) c h w")
        split_idx = mem.shape[0]
        if encode_future_video:
            if video is None:
                raise ValueError("video must be provided when encode_future_video=True")
            num_frames = video.shape[2]
            video = rearrange(video, "bv c f h w -> (bv f) c h w")
            entire_video = torch.cat((mem, video), dim=0)
        else:
            num_frames = 0
            entire_video = mem
        entire_latents = []
        n_batch = int(np.ceil(float(entire_video.shape[0])/batch_size_2dvae))
        for _i in range(n_batch):
            sidx = _i*batch_size_2dvae
            eidx = min((_i+1)*batch_size_2dvae, entire_video.shape[0])
            entire_latents.append(vae.encode(entire_video[sidx:eidx]).latent)
        entire_latents = torch.cat(entire_latents, dim=0)
        entire_latents = stats_tracker.normalize(entire_latents) if stats_tracker is not None else entire_latents * vae.config.scaling_factor
        mem_latents = rearrange(entire_latents[:split_idx], "bvm c h w -> bvm (h w) c")
        video_latents = (
            rearrange(entire_latents[split_idx:], "(bv f) c h w -> bv (f h w) c", f=num_frames)
            if encode_future_video
            else None
        )
        return mem_latents, video_latents
    else:
        raise NotImplementedError


@torch.no_grad()
def decode_3dvae_latents(vae, latents, decode_timestep=0.0, decode_noise_scale=None, dtype=torch.bfloat16, generator=None):
    batch_size = latents.shape[0]
    latents = _normalize_latents(latents, vae.latents_mean, vae.latents_std, vae.config.scaling_factor, reverse=True)
    latents = latents.to(dtype)

    if not vae.config.timestep_conditioning:
        timestep = None
    else:
        noise = torch.randn(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)
        if not isinstance(decode_timestep, list):
            decode_timestep = [decode_timestep] * batch_size
        if decode_noise_scale is None:
            decode_noise_scale = decode_timestep
        elif not isinstance(decode_noise_scale, list):
            decode_noise_scale = [decode_noise_scale] * batch_size

        timestep = torch.tensor(decode_timestep, device=latents.device, dtype=latents.dtype)
        decode_noise_scale = torch.tensor(decode_noise_scale, device=latents.device, dtype=latents.dtype)[
            :, None, None, None, None
        ]
        latents = (1 - decode_noise_scale) * latents + decode_noise_scale * noise

    video = vae.decode(latents, temb=timestep, return_dict=False)[0]

    return video.clamp(-1, 1)


@torch.no_grad()
def decode_latents(vae, latents, decode_timestep=0.0, decode_noise_scale=None, dtype=torch.bfloat16, generator=None, stats_tracker=None,):

    mode = getattr(vae, "vae_encoding_mode", "3dvae")

    if mode.lower().strip() == "3dvae":
        return decode_3dvae_latents(vae, latents, decode_timestep, decode_noise_scale, dtype, generator)
    elif mode.lower().strip() == "2dvae":
        batch_size = latents.shape[0]
        x = rearrange(latents, "b c t h w -> (b t) c h w")
        x = stats_tracker.denormalize(x) if stats_tracker is not None else x / vae.config.scaling_factor
        video = vae.decode(x, return_dict=False)[0]
        video = rearrange(video, "(b t) c h w -> b c t h w", b=batch_size)
        return video.clamp(-1, 1)
    else:
        raise NotImplementedError


def prepare_latents(
    vae,
    image_or_video: torch.Tensor,
    patch_size: int = 1,
    patch_size_t: int = 1,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    generator: Optional[torch.Generator] = None,
    precompute: bool = False,
) -> torch.Tensor:
    device = device or vae.device

    if image_or_video.ndim == 4:
        image_or_video = image_or_video.unsqueeze(2)
    assert image_or_video.ndim == 5, f"Expected 5D tensor, got {image_or_video.ndim}D tensor"

    image_or_video = image_or_video.to(device=device, dtype=vae.dtype)
    image_or_video = image_or_video.permute(0, 2, 1, 3, 4).contiguous()
    if not precompute:
        latents = vae.encode(image_or_video).latent_dist.sample(generator=generator)
        latents = latents.to(dtype=dtype)
        _, _, num_frames, height, width = latents.shape
        latents = _normalize_latents(latents, vae.latents_mean, vae.latents_std)
        latents = _pack_latents(latents, patch_size, patch_size_t)
        return {"latents": latents, "num_frames": num_frames, "height": height, "width": width}
    else:
        if vae.use_slicing and image_or_video.shape[0] > 1:
            encoded_slices = [vae._encode(x_slice) for x_slice in image_or_video.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = vae._encode(image_or_video)
        _, _, num_frames, height, width = h.shape

        return {
            "latents": h,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "latents_mean": vae.latents_mean,
            "latents_std": vae.latents_std,
        }


def post_latent_preparation(
    latents: torch.Tensor,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
    num_frames: int,
    height: int,
    width: int,
    patch_size: int = 1,
    patch_size_t: int = 1,
) -> torch.Tensor:
    latents = _normalize_latents(latents, latents_mean, latents_std)
    latents = _pack_latents(latents, patch_size, patch_size_t)
    return {"latents": latents, "num_frames": num_frames, "height": height, "width": width}


def _normalize_latents(
    latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0,
    reverse=False,
) -> torch.Tensor:
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    if not reverse:
        latents = (latents - latents_mean) * scaling_factor / latents_std
    else:
        latents = latents * latents_std / scaling_factor + latents_mean
    return latents


def unpack_latents(
        latents: torch.Tensor, num_frames: int, height: int, width: int, patch_size: int = 1, patch_size_t: int = 1
    ) -> torch.Tensor:
    batch_size = latents.size(0)
    latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
    latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return latents


def _pack_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
    batch_size, num_channels, num_frames, height, width = latents.shape
    post_patch_num_frames = num_frames // patch_size_t
    post_patch_height = height // patch_size
    post_patch_width = width // patch_size
    latents = latents.reshape(
        batch_size,
        -1,
        post_patch_num_frames,
        patch_size_t,
        post_patch_height,
        patch_size,
        post_patch_width,
        patch_size,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
    return latents


def gen_noise_from_condition_frame_latent(condition_frame_latent, latent_num_frames,
                                     latent_height=12, latent_width=16, generator=None, noise_to_condition_frames=0.05):
    mem_size = condition_frame_latent.shape[2]
    num_channels_latents = condition_frame_latent.shape[1]
    batch_size = condition_frame_latent.size(0)

    shape = (batch_size, num_channels_latents, latent_num_frames, latent_height, latent_width)
    mask_shape = (batch_size, 1, latent_num_frames, latent_height, latent_width)

    init_latents = condition_frame_latent[:,:,:1].repeat(1, 1, latent_num_frames, 1, 1)
    init_latents[:,:,:mem_size] = condition_frame_latent
    conditioning_mask = torch.zeros(mask_shape, device=condition_frame_latent.device, dtype=condition_frame_latent.dtype)
    conditioning_mask[:, :, :mem_size] = 1.0

    cond_indicator = torch.zeros((1, 1, latent_num_frames, 1, 1), device=condition_frame_latent.device, dtype=condition_frame_latent.dtype)
    cond_indicator[:, :, :mem_size] = 1.0

    if noise_to_condition_frames:
        if generator is not None:
            rand_noise_ff = torch.rand(
                (), device=generator.device, generator=generator
            ).item() * noise_to_condition_frames
        else:
            rand_noise_ff = random.random() * noise_to_condition_frames
    else:
        rand_noise_ff = 0.0

    first_frame_mask = conditioning_mask.clone()
    first_frame_mask[:, :, :mem_size] = 1.0 - rand_noise_ff

    noise = randn_tensor(shape, generator=generator, device=condition_frame_latent.device, dtype=condition_frame_latent.dtype)
    latents = init_latents * first_frame_mask + noise * (1 - first_frame_mask)

    conditioning_mask = _pack_latents(conditioning_mask).squeeze(-1)
    cond_indicator = _pack_latents(cond_indicator).squeeze(-1)

    latents = _pack_latents(latents)

    return latents, conditioning_mask, cond_indicator


def apply_color_jitter_to_video(tensor, jitter=None):
    batch_size, channels, _, _, _ = tensor.shape
    if channels != 3:
        raise ValueError(f"Color jitter expects 3 channels, got {channels}")

    if jitter is None:
        jitter = ColorJitter(
            brightness=0.3,
            contrast=0.4,
            saturation=0.5,
            hue=0.1,
        )

    tensor = rearrange((tensor + 1.0) / 2.0, "b c t h w -> b t c h w")
    for batch_index in range(batch_size):
        tensor[batch_index] = jitter(tensor[batch_index])
    return rearrange(tensor, "b t c h w -> b c t h w") * 2.0 - 1.0


@torch.no_grad()
def prepare_ray_map(intrinsic, c2w, H, W):
    batch_size = intrinsic.shape[0]
    fx, fy, cx, cy = intrinsic[:,0,0].unsqueeze(1).unsqueeze(2), intrinsic[:,1,1].unsqueeze(1).unsqueeze(2), intrinsic[:,0,2].unsqueeze(1).unsqueeze(2), intrinsic[:,1,2].unsqueeze(1).unsqueeze(2)
    i, j = torch.meshgrid(torch.linspace(0.5, W-0.5, W, device=c2w.device), torch.linspace(0.5, H-0.5, H, device=c2w.device))
    i = i.t()
    j = j.t()
    i = i.unsqueeze(0).repeat(batch_size,1,1)
    j = j.unsqueeze(0).repeat(batch_size,1,1)
    dirs = torch.stack([(i-cx)/fx, (j-cy)/fy, torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:,np.newaxis,np.newaxis, :3,:3], -1)
    rays_o = c2w[:, :3,-1].unsqueeze(1).unsqueeze(2).repeat(1,H,W,1)
    viewdir = rays_d/torch.norm(rays_d, dim=-1, keepdim=True)
    return rays_o, viewdir


@torch.no_grad()
def get_ray_maps(intrinsic, extrinsic, h, w, n_view, t, device, dtype):
    intrinsics = rearrange(intrinsic, "b v t i j -> (b v t) i j")
    extrinsics = rearrange(extrinsic, "b v t i j -> (b v t) i j")
    rays_o, rays_d = prepare_ray_map(intrinsics, extrinsics, H=h, W=w)
    rays = rearrange(torch.cat((rays_o, rays_d), dim=-1), "(b v t) h w c -> b c v t h w", v=n_view, t=t)
    rays = rays.to(device, dtype=dtype).contiguous()
    return rays


def resize_traj_and_ray(traj_n_ray, mem_size, future_size, height, width):
    orig_t = traj_n_ray.shape[3]
    assert orig_t > (mem_size + future_size)

    n_view = traj_n_ray.shape[2]
    traj_n_ray = rearrange(traj_n_ray, 'b c v t h w -> (b v) c t h w')

    mem = traj_n_ray[:, :, :mem_size]
    mem = rearrange(mem, 'bv c t h w -> (bv t) c h w')
    mem = F.interpolate(mem, (height, width), mode='bilinear')
    mem = rearrange(mem, '(bv t) c h w -> bv c t h w', t=mem_size)

    future = traj_n_ray[:, :, mem_size:]
    future = F.interpolate(future, (future_size, height, width), mode='trilinear')

    out = torch.cat([mem, future], dim=2)
    out = rearrange(out, '(b v) c t h w -> b c v t h w', v=n_view)
    return out