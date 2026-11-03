"""User-facing artifact-cache configuration, isolated from datasets."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .codec import RAW_CODEC, normalize_value_codec
from .fields import CacheField, normalize_cache_selection


DEFAULT_VALIDATION_WORKERS = 8
DEFAULT_VALIDATION_MAX_INFLIGHT_BYTES = 1024**3
DEFAULT_VALIDATION_PROGRESS_INTERVAL_SECONDS = 30.0


class CacheMode(str, Enum):
    OFF = "off"
    GENERATE = "generate"
    READ = "read"

    @classmethod
    def parse(cls, value: Any) -> "CacheMode":
        text = str(value or "off").strip().lower()
        try:
            return cls(text)
        except ValueError as exc:
            allowed = ", ".join(mode.value for mode in cls)
            raise ValueError(
                f"`video_latent_cache` must be one of {allowed}; got {value!r}."
            ) from exc


class CacheValidationMode(str, Enum):
    META_ONLY = "meta-only"
    FULL = "full"

    @classmethod
    def parse(cls, value: Any) -> "CacheValidationMode":
        if isinstance(value, cls):
            return value
        text = str(value or cls.META_ONLY.value).strip().lower()
        try:
            return cls(text)
        except ValueError as exc:
            allowed = ", ".join(mode.value for mode in cls)
            raise ValueError(
                "`data.video_latent_cache.validation_mode` must be one of "
                f"{allowed}; got {value!r}."
            ) from exc


def _get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    getter = getattr(node, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(node, key, default)


@dataclass(frozen=True)
class VaeCacheConfig:
    mode: CacheMode
    root: Path | None
    fields: tuple[CacheField, ...]
    max_cached_shards: int = 32
    vlm_enabled: bool = False
    vae_value_codec: str = RAW_CODEC
    vlm_value_codec: str = RAW_CODEC
    validation_mode: CacheValidationMode = CacheValidationMode.META_ONLY
    validation_workers: int = DEFAULT_VALIDATION_WORKERS
    validation_max_inflight_bytes: int = DEFAULT_VALIDATION_MAX_INFLIGHT_BYTES
    validation_progress_interval_seconds: float = (
        DEFAULT_VALIDATION_PROGRESS_INTERVAL_SECONDS
    )

    def __post_init__(self) -> None:
        if self.mode is not CacheMode.OFF and not self.fields and not self.vlm_enabled:
            raise ValueError("Cache generate/read must select at least one artifact.")
        object.__setattr__(
            self, "max_cached_shards", max(1, int(self.max_cached_shards))
        )
        object.__setattr__(
            self, "vae_value_codec", normalize_value_codec(self.vae_value_codec)
        )
        object.__setattr__(
            self, "vlm_value_codec", normalize_value_codec(self.vlm_value_codec)
        )
        object.__setattr__(
            self,
            "validation_mode",
            CacheValidationMode.parse(self.validation_mode),
        )
        object.__setattr__(
            self,
            "validation_workers",
            max(1, int(self.validation_workers)),
        )
        object.__setattr__(
            self,
            "validation_max_inflight_bytes",
            max(1, int(self.validation_max_inflight_bytes)),
        )
        object.__setattr__(
            self,
            "validation_progress_interval_seconds",
            max(0.0, float(self.validation_progress_interval_seconds)),
        )

    @classmethod
    def from_config(cls, cfg: Any) -> "VaeCacheConfig":
        mode = CacheMode.parse(_get(cfg, "video_latent_cache", "off"))
        fields, vlm_enabled = normalize_cache_selection(
            _get(cfg, "video_latent_cache_fields", None)
        )
        data_cfg = _get(cfg, "data")
        cache_cfg = _get(data_cfg, "video_latent_cache")
        root_value = _get(cache_cfg, "root")
        max_cached_shards = _get(cache_cfg, "max_cached_shards", 32)
        codec_cfg = _get(cache_cfg, "value_codec")
        vae_value_codec = normalize_value_codec(_get(codec_cfg, "vae", RAW_CODEC))
        vlm_value_codec = normalize_value_codec(
            _get(codec_cfg, "vlm", RAW_CODEC)
        )
        validation_mode = CacheValidationMode.parse(
            _get(
                cache_cfg,
                "validation_mode",
                CacheValidationMode.META_ONLY.value,
            )
        )
        if max_cached_shards is None:
            max_cached_shards = 32
        validation_workers = _get(
            cache_cfg,
            "validation_workers",
            DEFAULT_VALIDATION_WORKERS,
        )
        if validation_workers is None:
            validation_workers = DEFAULT_VALIDATION_WORKERS
        validation_max_inflight_bytes = _get(
            cache_cfg,
            "validation_max_inflight_bytes",
            DEFAULT_VALIDATION_MAX_INFLIGHT_BYTES,
        )
        if validation_max_inflight_bytes is None:
            validation_max_inflight_bytes = DEFAULT_VALIDATION_MAX_INFLIGHT_BYTES
        validation_progress_interval_seconds = _get(
            cache_cfg,
            "validation_progress_interval_seconds",
            DEFAULT_VALIDATION_PROGRESS_INTERVAL_SECONDS,
        )
        if validation_progress_interval_seconds is None:
            validation_progress_interval_seconds = (
                DEFAULT_VALIDATION_PROGRESS_INTERVAL_SECONDS
            )
        root = (
            None
            if root_value in (None, "", "null")
            else Path(str(root_value)).expanduser().resolve()
        )
        if mode is not CacheMode.OFF and root is None:
            raise ValueError(
                "Cache generate/read requires `data.video_latent_cache.root`."
            )
        return cls(
            mode=mode,
            root=root,
            fields=fields,
            vlm_enabled=vlm_enabled,
            vae_value_codec=vae_value_codec,
            vlm_value_codec=vlm_value_codec,
            validation_mode=validation_mode,
            max_cached_shards=max(1, int(max_cached_shards)),
            validation_workers=max(1, int(validation_workers)),
            validation_max_inflight_bytes=max(
                1, int(validation_max_inflight_bytes)
            ),
            validation_progress_interval_seconds=max(
                0.0, float(validation_progress_interval_seconds)
            ),
        )


# New code can use the producer-neutral name; retain the old public name for
# callers that already construct this config directly.
CacheConfig = VaeCacheConfig
