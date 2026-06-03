from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version as get_package_version
from typing import Optional

from packaging.version import Version, parse as parse_version

import accelerate
import torch
import transformers


_MIN_ACCELERATE_SUPPORTED = parse_version("1.10.0")
_MIN_TORCH_NATIVE_FSDP2 = parse_version("2.6.0")
_MIN_TRANSFORMERS_NATIVE_FSDP2 = parse_version("5.0.0")
_MIN_DEEPSPEED_NEW_STACK = parse_version("0.18.0")

_TRAINER_FSDP_KEYS = {
    "activation_checkpointing",
    "backward_prefetch",
    "cpu_ram_efficient_loading",
    "forward_prefetch",
    "limit_all_gathers",
    "min_num_params",
    "reshard_after_forward",
    "state_dict_type",
    "sync_module_states",
    "transformer_layer_cls_to_wrap",
    "use_orig_params",
}

_PLUGIN_FSDP_KEYS = {
    "activation_checkpointing",
    "auto_wrap_policy",
    "backward_prefetch",
    "cpu_ram_efficient_loading",
    "forward_prefetch",
    "limit_all_gathers",
    "min_num_params",
    "reshard_after_forward",
    "sharding_strategy",
    "state_dict_type",
    "sync_module_states",
    "use_orig_params",
}


@dataclass(frozen=True)
class PackageVersions:
    torch: str
    transformers: str
    accelerate: str
    deepspeed: Optional[str] = None

    @property
    def torch_version(self) -> Version:
        return parse_version(self.torch)

    @property
    def transformers_version(self) -> Version:
        return parse_version(self.transformers)

    @property
    def accelerate_version(self) -> Version:
        return parse_version(self.accelerate)

    
    @property
    def deepspeed_version(self) -> Optional[Version]:
        if self.deepspeed is None:
            return None
        return parse_version(self.deepspeed)


@dataclass
class BackendResolution:
    requested_backend: str
    resolved_mode: str
    framework: str
    trainer_fsdp: Optional[str] = None
    trainer_fsdp_config: Optional[dict] = None
    plugin_kwargs: Optional[dict] = None
    normalized_fsdp_config: Optional[dict] = None
    package_versions: Optional[PackageVersions] = None
    warnings: list[str] = field(default_factory=list)

    @property
    def is_fsdp_backend(self) -> bool:
        return self.requested_backend in {"fsdp", "fsdp2"}


@dataclass(frozen=True)
class FSDPProfile:
    enabled: bool = False
    cpu_ram_efficient_loading: Optional[bool] = None
    transformer_layer_cls_to_wrap: Optional[tuple[str, ...]] = None
    root_only: bool = True
    cast_model_to_bf16_backends: tuple[str, ...] = ()


def _detect_optional_package_version(package_name: str) -> Optional[str]:
    try:
        return get_package_version(package_name)
    except PackageNotFoundError:
        return None


def detect_package_versions() -> PackageVersions:
    return PackageVersions(
        torch=torch.__version__,
        transformers=transformers.__version__,
        accelerate=accelerate.__version__,
        deepspeed=_detect_optional_package_version("deepspeed"),
    )


def _default_fsdp_strategy(profile: FSDPProfile, fsdp_config: dict) -> str:
    has_wrap_selector = (
        profile.transformer_layer_cls_to_wrap is not None
        or fsdp_config.get("transformer_layer_cls_to_wrap") is not None
        or fsdp_config.get("auto_wrap_policy") is not None
        or fsdp_config.get("min_num_params") is not None
    )
    return "full_shard auto_wrap" if has_wrap_selector else "full_shard"


def apply_backend_defaults(trainer_config) -> None:
    if trainer_config.train_backend not in {"fsdp", "fsdp2"}:
        return

    profile = getattr(trainer_config, "fsdp_profile", FSDPProfile())

    trainer_config.gradient_checkpointing = False

    if trainer_config.fsdp_config is None:
        trainer_config.fsdp_config = {}

    trainer_config.fsdp_config.setdefault("activation_checkpointing", True)
    # Use SHARDED_STATE_DICT to avoid incomplete optimizer state with FULL_STATE_DICT
    # under FSDP2 (pytorch/pytorch#136950).  Sharded checkpoints support reliable
    # auto-resume.  For inference, convert first:
    #   from accelerate.utils import merge_fsdp_weights
    #   merge_fsdp_weights("checkpoint-N/pytorch_model_fsdp_0", "checkpoint-N/", safe_serialization=True)
    trainer_config.fsdp_config.setdefault("state_dict_type", "SHARDED_STATE_DICT")

    if profile.enabled:
        if profile.cpu_ram_efficient_loading is not None:
            trainer_config.fsdp_config.setdefault(
                "cpu_ram_efficient_loading", profile.cpu_ram_efficient_loading
            )
        if profile.transformer_layer_cls_to_wrap is not None:
            if (
                "transformer_layer_cls_to_wrap" not in trainer_config.fsdp_config
                and "auto_wrap_policy" not in trainer_config.fsdp_config
            ):
                trainer_config.fsdp_config["transformer_layer_cls_to_wrap"] = list(
                    profile.transformer_layer_cls_to_wrap
                )
        elif profile.root_only:
            trainer_config.fsdp_config.pop("transformer_layer_cls_to_wrap", None)
            trainer_config.fsdp_config.pop("auto_wrap_policy", None)

    if trainer_config.fsdp is None:
        trainer_config.fsdp = _default_fsdp_strategy(
            profile, trainer_config.fsdp_config
        )

    if trainer_config.train_backend == "fsdp2":
        trainer_config.fsdp_config.setdefault("reshard_after_forward", True)
        if trainer_config.fsdp_version is None:
            trainer_config.fsdp_version = 2
    else:
        # Accelerate 1.10+ expects FSDP1 reshard_after_forward to use the
        # sharding-strategy spelling instead of a boolean.
        trainer_config.fsdp_config.setdefault("reshard_after_forward", "FULL_SHARD")


def normalize_fsdp_config(trainer_config) -> tuple[dict, list[str]]:
    normalized = dict(trainer_config.fsdp_config or {})
    warnings: list[str] = []

    fsdp_version = normalized.pop("fsdp_version", None)
    if fsdp_version is not None:
        normalized.setdefault("version", fsdp_version)

    plugin_wrap = normalized.pop("transformer_cls_names_to_wrap", None)
    if plugin_wrap is not None:
        normalized.setdefault("transformer_layer_cls_to_wrap", plugin_wrap)
        warnings.append(
            "Normalized fsdp_config.transformer_cls_names_to_wrap to "
            "transformer_layer_cls_to_wrap."
        )

    if trainer_config.fsdp_version is not None:
        config_version = normalized.get("version")
        if config_version is not None and config_version != trainer_config.fsdp_version:
            warnings.append(
                "trainer_config.fsdp_version overrides fsdp_config version mismatch."
            )
        normalized["version"] = trainer_config.fsdp_version

    return normalized, warnings


def to_trainer_fsdp_args(
    normalized: dict, *, fsdp_version: Optional[int] = None
) -> dict:
    mapped = {
        key: value for key, value in normalized.items() if key in _TRAINER_FSDP_KEYS
    }
    if fsdp_version != 2 and "reshard_after_forward" in mapped:
        if mapped["reshard_after_forward"] is True:
            mapped["reshard_after_forward"] = "FULL_SHARD"
        elif mapped["reshard_after_forward"] is False:
            mapped["reshard_after_forward"] = "SHARD_GRAD_OP"
    if fsdp_version is not None:
        # `fsdp_version` is the documented key in newer Transformers. Keep `version`
        # as a compatibility alias for stacks that still consume the older spelling.
        mapped["fsdp_version"] = fsdp_version
        mapped["version"] = fsdp_version
    return mapped


def to_accelerate_fsdp2_plugin_kwargs(normalized: dict) -> dict:
    mapped = {
        key: value for key, value in normalized.items() if key in _PLUGIN_FSDP_KEYS
    }
    mapped["fsdp_version"] = 2
    wrap_layers = normalized.get("transformer_layer_cls_to_wrap")
    if wrap_layers is not None:
        mapped["transformer_cls_names_to_wrap"] = wrap_layers
        mapped.setdefault("auto_wrap_policy", "transformer_based_wrap")
    return mapped


def _supports_native_trainer_fsdp2(versions: PackageVersions) -> bool:
    return (
        versions.transformers_version >= _MIN_TRANSFORMERS_NATIVE_FSDP2
        and versions.torch_version >= _MIN_TORCH_NATIVE_FSDP2
        and versions.accelerate_version >= _MIN_ACCELERATE_SUPPORTED
    )


def _supports_explicit_plugin_fsdp2(versions: PackageVersions) -> bool:
    return (
        versions.accelerate_version >= _MIN_ACCELERATE_SUPPORTED
        and versions.torch_version >= _MIN_TORCH_NATIVE_FSDP2
    )


def _requires_deepspeed_v018_plus(versions: PackageVersions) -> bool:
    return (
        versions.torch_version >= _MIN_TORCH_NATIVE_FSDP2
        and versions.transformers_version >= _MIN_TRANSFORMERS_NATIVE_FSDP2
    )


def _cpu_ram_efficient_loading_default(versions: PackageVersions) -> bool:
    # transformers>=5.0 implements cpu_ram_efficient_loading via
    # torch.set_default_device('meta'), which breaks custom model __init__ methods
    # that perform real computation (e.g. diffusion schedule precomputation).
    # Return False for transformers>=5.0 regardless of FSDP version.
    return versions.transformers_version < _MIN_TRANSFORMERS_NATIVE_FSDP2


def _fsdp2_has_wrap_selector(normalized: dict) -> bool:
    wrap_layers = normalized.get("transformer_layer_cls_to_wrap")
    if wrap_layers:
        return True
    auto_wrap_policy = normalized.get("auto_wrap_policy")
    if auto_wrap_policy == "size_based_wrap" and normalized.get("min_num_params") is not None:
        return True
    if callable(auto_wrap_policy):
        return True
    return False


def resolve_fsdp2_mode(trainer_config) -> BackendResolution:
    versions = detect_package_versions()
    normalized, warnings = normalize_fsdp_config(trainer_config)
    normalized["version"] = 2
    profile = getattr(trainer_config, "fsdp_profile", FSDPProfile())
    if normalized.get("activation_checkpointing") and not _fsdp2_has_wrap_selector(normalized):
        normalized["activation_checkpointing"] = False
        warnings.append(
            "Disabled FSDP2 activation_checkpointing because no auto-wrap selector is configured."
        )

    if not profile.enabled:
        raise ValueError(
            f"FSDP2 is not enabled for {trainer_config.__class__.__name__}."
        )
    if trainer_config.fsdp is None:
        raise ValueError("FSDP2 backend requires trainer_config.fsdp to be configured.")

    if _supports_native_trainer_fsdp2(versions):
        normalized["cpu_ram_efficient_loading"] = _cpu_ram_efficient_loading_default(versions)
        return BackendResolution(
            requested_backend="fsdp2",
            resolved_mode="trainer_fsdp2_native",
            framework="trainer",
            trainer_fsdp=trainer_config.fsdp,
            trainer_fsdp_config=to_trainer_fsdp_args(normalized, fsdp_version=2),
            normalized_fsdp_config=normalized,
            package_versions=versions,
            warnings=warnings,
        )

    if _supports_explicit_plugin_fsdp2(versions):
        return BackendResolution(
            requested_backend="fsdp2",
            resolved_mode="accelerate_fsdp2_plugin",
            framework="accelerate_plugin",
            trainer_fsdp=trainer_config.fsdp,
            trainer_fsdp_config=to_trainer_fsdp_args(normalized, fsdp_version=2),
            plugin_kwargs=to_accelerate_fsdp2_plugin_kwargs(normalized),
            normalized_fsdp_config=normalized,
            package_versions=versions,
            warnings=warnings,
        )

    raise ValueError(
        "Requested FSDP2 backend is unsupported for the current stack. "
        f"FSDP2 requires torch>={_MIN_TORCH_NATIVE_FSDP2} and accelerate>={_MIN_ACCELERATE_SUPPORTED}; "
        f"got torch={versions.torch}, transformers={versions.transformers}, accelerate={versions.accelerate}."
    )


def resolve_backend_mode(trainer_config) -> BackendResolution:
    versions = detect_package_versions()
    normalized, warnings = normalize_fsdp_config(trainer_config)

    if trainer_config.train_backend == "deepspeed":
        if trainer_config.fsdp is not None:
            raise ValueError(
                "trainer_config.fsdp is configured, but train_backend is still "
                "'deepspeed'. Please explicitly set train_backend to 'fsdp' or 'fsdp2'."
            )
        if _requires_deepspeed_v018_plus(versions):
            if versions.deepspeed_version is None:
                raise ValueError(
                    "DeepSpeed backend with torch>=2.6.0 and transformers>=5.0.0 requires deepspeed>=0.18.0, but deepspeed is not installed."
                )
            if versions.deepspeed_version < _MIN_DEEPSPEED_NEW_STACK:
                raise ValueError(
                    "DeepSpeed backend with torch>=2.6.0 and transformers>=5.0.0 requires deepspeed>=0.18.0; "
                    f"got deepspeed={versions.deepspeed}, torch={versions.torch}, transformers={versions.transformers}."
                )
        return BackendResolution(
            requested_backend="deepspeed",
            resolved_mode="deepspeed_trainer",
            framework="trainer",
            package_versions=versions,
        )

    if trainer_config.train_backend == "fsdp":
        if trainer_config.fsdp is None:
            raise ValueError("FSDP backend requires trainer_config.fsdp to be configured.")
        normalized["cpu_ram_efficient_loading"] = _cpu_ram_efficient_loading_default(versions)
        return BackendResolution(
            requested_backend="fsdp",
            resolved_mode="trainer_fsdp1",
            framework="trainer",
            trainer_fsdp=trainer_config.fsdp,
            trainer_fsdp_config=to_trainer_fsdp_args(normalized),
            normalized_fsdp_config=normalized,
            package_versions=versions,
            warnings=warnings,
        )

    if trainer_config.train_backend == "fsdp2":
        return resolve_fsdp2_mode(trainer_config)

    if trainer_config.train_backend == "ddp":
        return BackendResolution(
            requested_backend="ddp",
            resolved_mode="ddp_trainer",
            framework="trainer",
            package_versions=versions,
        )

    raise ValueError(
        f"Unsupported train_backend: {trainer_config.train_backend}. "
        "Expected one of: deepspeed, fsdp, fsdp2."
    )
