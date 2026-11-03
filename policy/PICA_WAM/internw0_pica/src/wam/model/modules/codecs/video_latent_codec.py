from typing import Any, Sequence

import torch
from PIL import Image

from .utils import normalize_input_image_tensor, video_spatial_shape


def latent_spatial_shape_from_video(model, video: torch.Tensor) -> tuple[int, int]:
    height, width = video_spatial_shape(video)
    latent_h = height // int(model.vae.upsampling_factor)
    latent_w = width // int(model.vae.upsampling_factor)
    if video.ndim == 6:
        latent_w *= int(video.shape[1])
    return latent_h, latent_w


def normalize_video_metadata(
    value: Any, batch_size: int, *, default: str = ""
) -> list[str]:
    if value is None:
        return [default] * int(batch_size)
    if isinstance(value, str):
        return [value] * int(batch_size)
    if isinstance(value, Sequence):
        values = [default if item is None else str(item) for item in value]
        if len(values) == int(batch_size):
            return values
        if len(values) == 1:
            return values * int(batch_size)
    return [str(value)] * int(batch_size)


def normalize_video_layouts(video_layout: Any, batch_size: int) -> list[str]:
    return normalize_video_metadata(video_layout, batch_size, default="")


def is_robotwin_tshape_layout(layout: str) -> bool:
    return str(layout).strip().lower() in {"robotwin", "robotwin_tshape", "tshape"}


def is_latent_split_layout(layout: str) -> bool:
    return str(layout).strip().lower() in {
        "horizontal",
        "vertical",
        "grid2x2",
        "robotwin",
        "robotwin_tshape",
        "tshape",
    }


def num_views_from_names(video_view_names: Any, *, default: int) -> int:
    if video_view_names is None:
        return int(default)
    text = str(video_view_names).strip()
    if not text:
        return int(default)
    parts = [part for part in text.split("|") if part]
    return max(1, len(parts) if parts else int(default))


def split_sizes(total: int, parts: int) -> list[int]:
    parts = max(1, int(parts))
    base = int(total) // parts
    sizes = [base] * parts
    sizes[-1] += int(total) - sum(sizes)
    return sizes


def check_vae_slot_shape(model, height: int, width: int, *, layout: str) -> None:
    factor = int(model.vae.upsampling_factor)
    if int(height) % factor != 0 or int(width) % factor != 0:
        raise ValueError(
            f"{layout} latent-split slot size must be divisible by VAE factor {factor}, "
            f"got HxW=({int(height)},{int(width)})."
        )


def robotwin_tshape_splits(
    model, height: int, width: int, *, view_count: int
) -> tuple[int, int, int | None]:
    height = int(height)
    width = int(width)
    top_h = (height * 2) // 3
    bottom_h = height - top_h
    check_vae_slot_shape(model, top_h, width, layout="robotwin_tshape high")
    check_vae_slot_shape(model, bottom_h, width, layout="robotwin_tshape wrist")
    if int(view_count) <= 2:
        return top_h, bottom_h, None
    left_w = width // 2
    right_w = width - left_w
    check_vae_slot_shape(model, bottom_h, left_w, layout="robotwin_tshape wrist-left")
    check_vae_slot_shape(model, bottom_h, right_w, layout="robotwin_tshape wrist-right")
    return top_h, bottom_h, left_w


@torch.no_grad()
def encode_plain_video_latents(
    model, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)
):
    return model.vae.encode(
        video_tensor,
        device=model.device,
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )


@torch.no_grad()
def encode_robotwin_tshape_latents(
    model,
    video_tensor,
    tiled=False,
    tile_size=(30, 52),
    tile_stride=(15, 26),
    video_view_names=None,
):
    if video_tensor.ndim != 5:
        raise ValueError(
            f"robotwin_tshape VAE encode expects [B,C,T,H,W], got {tuple(video_tensor.shape)}"
        )
    height, width = int(video_tensor.shape[-2]), int(video_tensor.shape[-1])
    view_count = num_views_from_names(video_view_names, default=3)
    top_h, _, split_w = robotwin_tshape_splits(
        model, height, width, view_count=view_count
    )
    high = video_tensor[..., :top_h, :]
    wrist = video_tensor[..., top_h:, :]
    high_z = encode_plain_video_latents(
        model,
        high,
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )
    if view_count <= 2:
        wrist_z = encode_plain_video_latents(
            model,
            wrist,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
    else:
        if split_w is None:
            raise ValueError(
                "robotwin_tshape expected a wrist split for 3-view layout."
            )
        left = wrist[..., :split_w]
        right = wrist[..., split_w:]
        wrist_video = torch.cat([left, right], dim=0)
        wrist_z = encode_plain_video_latents(
            model,
            wrist_video,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        batch_size = int(video_tensor.shape[0])
        left_z, right_z = wrist_z[:batch_size], wrist_z[batch_size:]
        wrist_z = torch.cat([left_z, right_z], dim=-1)
    return torch.cat([high_z, wrist_z], dim=-2).contiguous()


@torch.no_grad()
def encode_latent_split_video_latents(
    model,
    video_tensor,
    *,
    video_layout: str,
    video_view_names=None,
    tiled=False,
    tile_size=(30, 52),
    tile_stride=(15, 26),
):
    if video_tensor.ndim != 5:
        raise ValueError(
            f"latent split VAE encode expects [B,C,T,H,W], got {tuple(video_tensor.shape)}"
        )
    layout = str(video_layout).strip().lower()
    if is_robotwin_tshape_layout(layout):
        return encode_robotwin_tshape_latents(
            model,
            video_tensor,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
            video_view_names=video_view_names,
        )
    batch_size, channels, num_frames, height, width = video_tensor.shape
    if layout == "vertical":
        num_parts = num_views_from_names(video_view_names, default=2)
        if num_parts <= 1:
            num_parts = 2
        latents = []
        start = 0
        for slot_h in split_sizes(int(height), num_parts):
            end = start + int(slot_h)
            check_vae_slot_shape(model, slot_h, width, layout=layout)
            latents.append(
                encode_plain_video_latents(
                    model,
                    video_tensor[..., start:end, :],
                    tiled=tiled,
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                )
            )
            start = end
        return torch.cat(latents, dim=-2).contiguous()
    if layout == "horizontal":
        num_parts = num_views_from_names(video_view_names, default=2)
        latents = []
        start = 0
        for slot_w in split_sizes(int(width), num_parts):
            end = start + int(slot_w)
            check_vae_slot_shape(model, height, slot_w, layout=layout)
            latents.append(
                encode_plain_video_latents(
                    model,
                    video_tensor[..., :, start:end],
                    tiled=tiled,
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                )
            )
            start = end
        return torch.cat(latents, dim=-1).contiguous()
    if layout == "grid2x2":
        top_h = int(height) // 2
        bottom_h = int(height) - top_h
        left_w = int(width) // 2
        right_w = int(width) - left_w
        check_vae_slot_shape(model, top_h, left_w, layout=layout)
        check_vae_slot_shape(model, top_h, right_w, layout=layout)
        check_vae_slot_shape(model, bottom_h, left_w, layout=layout)
        check_vae_slot_shape(model, bottom_h, right_w, layout=layout)
        top_left = encode_plain_video_latents(
            model,
            video_tensor[..., :top_h, :left_w],
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        top_right = encode_plain_video_latents(
            model,
            video_tensor[..., :top_h, left_w:],
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        bottom_left = encode_plain_video_latents(
            model,
            video_tensor[..., top_h:, :left_w],
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        bottom_right = encode_plain_video_latents(
            model,
            video_tensor[..., top_h:, left_w:],
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        top = torch.cat([top_left, top_right], dim=-1)
        bottom = torch.cat([bottom_left, bottom_right], dim=-1)
        return torch.cat([top, bottom], dim=-2).contiguous()
    return encode_plain_video_latents(
        model,
        video_tensor,
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )


@torch.no_grad()
def encode_video_latents(
    model,
    video_tensor,
    tiled=False,
    tile_size=(30, 52),
    tile_stride=(15, 26),
    video_layout=None,
    video_view_names=None,
):
    if isinstance(video_tensor, torch.Tensor) and video_tensor.ndim == 6:
        batch_size, num_cameras, channels, num_frames, height, width = (
            video_tensor.shape
        )
        flat_video = video_tensor.reshape(
            batch_size * num_cameras,
            channels,
            num_frames,
            height,
            width,
        )
        flat_z = encode_plain_video_latents(
            model,
            flat_video,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        flat_z = flat_z.reshape(
            batch_size,
            num_cameras,
            flat_z.shape[1],
            flat_z.shape[2],
            flat_z.shape[3],
            flat_z.shape[4],
        )
        return torch.cat(
            [flat_z[:, cam_idx] for cam_idx in range(num_cameras)], dim=-1
        ).contiguous()
    if isinstance(video_tensor, torch.Tensor) and video_tensor.ndim == 5:
        layouts = normalize_video_layouts(video_layout, int(video_tensor.shape[0]))
        view_names = normalize_video_metadata(
            video_view_names, int(video_tensor.shape[0]), default=""
        )
        if any(is_latent_split_layout(layout) for layout in layouts):
            latents = []
            for batch_idx, layout in enumerate(layouts):
                sample_video = video_tensor[batch_idx : batch_idx + 1]
                if is_latent_split_layout(layout):
                    sample_latents = encode_latent_split_video_latents(
                        model,
                        sample_video,
                        video_layout=layout,
                        video_view_names=view_names[batch_idx],
                        tiled=tiled,
                        tile_size=tile_size,
                        tile_stride=tile_stride,
                    )
                else:
                    sample_latents = encode_plain_video_latents(
                        model,
                        sample_video,
                        tiled=tiled,
                        tile_size=tile_size,
                        tile_stride=tile_stride,
                    )
                latents.append(sample_latents)
            return torch.cat(latents, dim=0).contiguous()
    return encode_plain_video_latents(
        model,
        video_tensor,
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )


@torch.no_grad()
def encode_input_image_latents_tensor(
    model,
    input_image: torch.Tensor,
    tiled=False,
    tile_size=(30, 52),
    tile_stride=(15, 26),
    video_layout=None,
    video_view_names=None,
):
    input_image = normalize_input_image_tensor(input_image).to(device=model.device)
    if input_image.ndim == 5:
        return encode_video_latents(
            model,
            input_image.unsqueeze(3),
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
            video_layout=video_layout,
            video_view_names=video_view_names,
        )
    return encode_video_latents(
        model,
        input_image.unsqueeze(2),
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
        video_layout=video_layout,
        video_view_names=video_view_names,
    )


def decode_plain_latents_to_tensor(
    model, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)
):
    return model.vae.decode(
        latents,
        device=model.device,
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )


def decode_latent_split_latents_to_tensor(
    model,
    latents,
    *,
    video_layout: str,
    video_view_names=None,
    tiled=False,
    tile_size=(30, 52),
    tile_stride=(15, 26),
):
    layout = str(video_layout).strip().lower()
    if not is_latent_split_layout(layout):
        return decode_plain_latents_to_tensor(
            model,
            latents,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
    if int(latents.shape[0]) != 1:
        raise ValueError(
            f"latent split decode currently expects batch size 1, got {tuple(latents.shape)}"
        )
    latent_h, latent_w = int(latents.shape[-2]), int(latents.shape[-1])
    if is_robotwin_tshape_layout(layout):
        top_h = (latent_h * 2) // 3
        high = latents[..., :top_h, :]
        wrist = latents[..., top_h:, :]
        high_video = decode_plain_latents_to_tensor(
            model,
            high,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        view_count = num_views_from_names(video_view_names, default=3)
        if view_count <= 2:
            wrist_video = decode_plain_latents_to_tensor(
                model,
                wrist,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
        else:
            left_w = latent_w // 2
            left = decode_plain_latents_to_tensor(
                model,
                wrist[..., :, :left_w],
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
            right = decode_plain_latents_to_tensor(
                model,
                wrist[..., :, left_w:],
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
            wrist_video = torch.cat([left, right], dim=-1)
        return torch.cat([high_video, wrist_video], dim=-2)
    if layout == "vertical":
        num_parts = num_views_from_names(video_view_names, default=2)
        if num_parts <= 1:
            num_parts = 2
        videos = []
        start = 0
        for slot_h in split_sizes(latent_h, num_parts):
            end = start + int(slot_h)
            videos.append(
                decode_plain_latents_to_tensor(
                    model,
                    latents[..., start:end, :],
                    tiled=tiled,
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                )
            )
            start = end
        return torch.cat(videos, dim=-2)
    if layout == "horizontal":
        num_parts = num_views_from_names(video_view_names, default=2)
        videos = []
        start = 0
        for slot_w in split_sizes(latent_w, num_parts):
            end = start + int(slot_w)
            videos.append(
                decode_plain_latents_to_tensor(
                    model,
                    latents[..., :, start:end],
                    tiled=tiled,
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                )
            )
            start = end
        return torch.cat(videos, dim=-1)
    if layout == "grid2x2":
        top_h = latent_h // 2
        left_w = latent_w // 2
        top_left = decode_plain_latents_to_tensor(
            model,
            latents[..., :top_h, :left_w],
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        top_right = decode_plain_latents_to_tensor(
            model,
            latents[..., :top_h, left_w:],
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        bottom_left = decode_plain_latents_to_tensor(
            model,
            latents[..., top_h:, :left_w],
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        bottom_right = decode_plain_latents_to_tensor(
            model,
            latents[..., top_h:, left_w:],
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        top = torch.cat([top_left, top_right], dim=-1)
        bottom = torch.cat([bottom_left, bottom_right], dim=-1)
        return torch.cat([top, bottom], dim=-2)
    return decode_plain_latents_to_tensor(
        model,
        latents,
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )


def decode_latents(
    model,
    latents,
    tiled=False,
    tile_size=(30, 52),
    tile_stride=(15, 26),
    video_layout=None,
    video_view_names=None,
):
    layout = (
        normalize_video_layouts(video_layout, int(latents.shape[0]))[0]
        if isinstance(latents, torch.Tensor)
        else ""
    )
    view_names = (
        normalize_video_metadata(video_view_names, int(latents.shape[0]), default="")[0]
        if isinstance(latents, torch.Tensor)
        else ""
    )
    video_tensor = decode_latent_split_latents_to_tensor(
        model,
        latents,
        video_layout=layout,
        video_view_names=view_names,
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )
    video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
    video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
    frames = []
    for t in range(video_tensor.shape[1]):
        frame = video_tensor[:, t].permute(1, 2, 0).numpy()
        frames.append(Image.fromarray(frame))
    return frames
