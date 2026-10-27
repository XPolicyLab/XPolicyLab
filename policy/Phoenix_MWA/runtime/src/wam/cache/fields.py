"""Cache field names and user-facing normalization."""

from __future__ import annotations

from enum import Enum
from typing import Any, Iterable

from .codec import RAW_CODEC, normalize_value_codec


class CacheField(str, Enum):
    CURRENT = "current"
    MEMORY_ANCHOR = "memory_anchor"
    MEMORY_RECENT = "memory_recent"


DEFAULT_VAE_CACHE_FIELDS: tuple[CacheField, ...] = (
    CacheField.CURRENT,
    CacheField.MEMORY_ANCHOR,
    CacheField.MEMORY_RECENT,
)

FIELD_SAMPLE_KEYS: dict[CacheField, str] = {
    CacheField.CURRENT: "video_latents",
    CacheField.MEMORY_ANCHOR: "memory_video_anchor_latents",
    CacheField.MEMORY_RECENT: "memory_video_recent_latents",
}

FIELD_ARTIFACT_IDS: dict[CacheField, str] = {
    CacheField.CURRENT: "vae_latent.current",
    CacheField.MEMORY_ANCHOR: "vae_latent.memory_anchor",
    CacheField.MEMORY_RECENT: "vae_latent.memory_recent",
}

ARTIFACT_CACHE_FIELDS: dict[str, CacheField] = {
    artifact_id: field for field, artifact_id in FIELD_ARTIFACT_IDS.items()
}

VLM_CACHE_FIELD = "vlm"
VLM_ARTIFACT_ID = "vlm.current"


def codec_artifact_id(artifact_id: str, value_codec: str) -> str:
    normalize_value_codec(value_codec)
    return str(artifact_id)


def vae_artifact_id(field: CacheField, value_codec: str = RAW_CODEC) -> str:
    return codec_artifact_id(FIELD_ARTIFACT_IDS[field], value_codec)


def vlm_artifact_id(value_codec: str = RAW_CODEC) -> str:
    return codec_artifact_id(VLM_ARTIFACT_ID, value_codec)


def cache_key_for_sample(sample_idx: int) -> str:
    return f"sample:{int(sample_idx)}"


def expected_vae_latent_shapes(
    vae: Any,
    *,
    video_size: Iterable[int],
    current_video_frames: int,
    video_layout: str = "single",
    num_views: int = 1,
    fields: Any = None,
) -> dict[CacheField, tuple[int, ...]]:
    """Derive strict per-sample latent shapes from the loaded VAE contract."""

    channels = int(getattr(vae, "z_dim"))
    spatial_factor = int(getattr(vae, "upsampling_factor"))
    temporal_factor = int(getattr(vae, "temporal_downsample_factor"))
    if channels <= 0 or spatial_factor <= 0 or temporal_factor <= 0:
        raise ValueError(
            "VAE cache shape attributes must be positive: "
            f"channels={channels}, spatial={spatial_factor}, "
            f"temporal={temporal_factor}."
        )
    height, width = (int(value) for value in video_size)
    if height % spatial_factor != 0 or width % spatial_factor != 0:
        raise ValueError(
            "Video size is incompatible with the VAE spatial factor: "
            f"size={(height, width)} factor={spatial_factor}."
        )
    current_video_frames = int(current_video_frames)
    if current_video_frames <= 0:
        raise ValueError("current_video_frames must be positive.")
    num_views = int(num_views)
    if num_views <= 0:
        raise ValueError("num_views must be positive.")
    current_latent_frames = (
        current_video_frames - 1
    ) // temporal_factor + 1
    latent_height = height // spatial_factor
    latent_width = width // spatial_factor
    if str(video_layout).strip().lower() == "latent_horizontal":
        # RobotVideoDataset keeps each camera as an independent video for this
        # layout.  The codec encodes those videos independently, then joins the
        # resulting latents along width.
        latent_width *= num_views
    spatial = (latent_height, latent_width)
    all_shapes = {
        CacheField.CURRENT: (channels, current_latent_frames, *spatial),
        CacheField.MEMORY_ANCHOR: (channels, 1, *spatial),
        CacheField.MEMORY_RECENT: (channels, 1, *spatial),
    }
    selected = normalize_cache_fields(fields)
    return {field: all_shapes[field] for field in selected}


def _field_values(value: Any) -> Iterable[Any]:
    if value is None:
        return DEFAULT_VAE_CACHE_FIELDS
    if isinstance(value, CacheField):
        return (value,)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        if text.lower() == "all":
            return DEFAULT_VAE_CACHE_FIELDS
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        return tuple(item.strip() for item in text.split(",") if item.strip())
    try:
        return tuple(value)
    except TypeError as exc:
        raise ValueError(
            "video_latent_cache_fields must be a list or comma-separated string"
        ) from exc


def normalize_cache_fields(
    value: Any,
    *,
    allow_empty: bool = False,
) -> tuple[CacheField, ...]:
    """Return de-duplicated fields in their canonical order."""

    requested: set[CacheField] = set()
    invalid: list[str] = []
    for item in _field_values(value):
        try:
            requested.add(
                item if isinstance(item, CacheField) else CacheField(str(item))
            )
        except ValueError:
            invalid.append(str(item))
    if invalid:
        allowed = ", ".join(field.value for field in DEFAULT_VAE_CACHE_FIELDS)
        raise ValueError(
            f"Unsupported video latent cache fields {invalid}; allowed: {allowed}."
        )
    result = tuple(field for field in DEFAULT_VAE_CACHE_FIELDS if field in requested)
    if not result and not allow_empty:
        raise ValueError("video_latent_cache_fields must select at least one field.")
    return result


def normalize_cache_selection(value: Any) -> tuple[tuple[CacheField, ...], bool]:
    """Split the unified selection into VAE fields and the VLM artifact."""

    if value is None:
        return DEFAULT_VAE_CACHE_FIELDS, False
    raw = tuple(_field_values(value))
    vlm_enabled = any(str(item).strip().lower() == VLM_CACHE_FIELD for item in raw)
    vae_raw = [item for item in raw if str(item).strip().lower() != VLM_CACHE_FIELD]
    fields = normalize_cache_fields(vae_raw, allow_empty=True)
    if not fields and not vlm_enabled:
        raise ValueError("video_latent_cache_fields must select at least one artifact.")
    return fields, vlm_enabled
