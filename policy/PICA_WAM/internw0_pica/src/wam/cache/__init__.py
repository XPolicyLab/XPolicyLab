"""Public API for InternW0-delta's artifact cache manager."""

from .binding import (
    CacheBinding,
    CacheDataProjection,
    CacheDevicePlacement,
    CacheRuntimeBindings,
)
from .config import CacheMode, CacheValidationMode, VaeCacheConfig, CacheConfig
from .contracts import (
    ArtifactContract,
    ArtifactSnapshot,
    ArtifactValidationReport,
    PreflightReport,
    PublishReport,
    TensorSpec,
    ValidatedCacheSnapshot,
    dtype_from_name,
    dtype_name,
)
from .errors import CacheError
from .fields import (
    ARTIFACT_CACHE_FIELDS,
    DEFAULT_VAE_CACHE_FIELDS,
    FIELD_ARTIFACT_IDS,
    FIELD_SAMPLE_KEYS,
    CacheField,
    VLM_ARTIFACT_ID,
    normalize_cache_selection,
    cache_key_for_sample,
    expected_vae_latent_shapes,
    normalize_cache_fields,
)
from .manager import CacheManager


__all__ = [
    "ARTIFACT_CACHE_FIELDS",
    "DEFAULT_VAE_CACHE_FIELDS",
    "FIELD_ARTIFACT_IDS",
    "FIELD_SAMPLE_KEYS",
    "ArtifactContract",
    "ArtifactSnapshot",
    "ArtifactValidationReport",
    "CacheBinding",
    "CacheDataProjection",
    "CacheDevicePlacement",
    "CacheError",
    "CacheField",
    "CacheConfig",
    "VLM_ARTIFACT_ID",
    "normalize_cache_selection",
    "CacheManager",
    "CacheMode",
    "CacheValidationMode",
    "CacheRuntimeBindings",
    "PreflightReport",
    "PublishReport",
    "TensorSpec",
    "ValidatedCacheSnapshot",
    "VaeCacheConfig",
    "cache_key_for_sample",
    "dtype_from_name",
    "dtype_name",
    "expected_vae_latent_shapes",
    "normalize_cache_fields",
]
