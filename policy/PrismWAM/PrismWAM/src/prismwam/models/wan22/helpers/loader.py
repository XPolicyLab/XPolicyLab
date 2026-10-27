import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import torch
import time
from .io import ModelConfig, hash_model_file, load_state_dict
from .state_dict_converters import wan_video_vae_state_dict_converter
from ..wan_video_dit import WanVideoDiT
from ..wan_video_text_encoder import HuggingfaceTokenizer, WanTextEncoder
from ..wan_video_vae import WanDepthVAE16, WanDepthVAE38, WanVideoVAE, WanVideoVAE38
from prismwam.utils.logging_config import get_logger

logger = get_logger(__name__)
SKIPPED_PRETRAIN_SENTINEL = "SKIPPED_PRETRAIN"


@dataclass
class Wan22LoadedComponents:
    dit: WanVideoDiT
    vae: WanVideoVAE
    text_encoder: WanTextEncoder | None
    tokenizer: HuggingfaceTokenizer | None
    dit_path: str
    vae_path: str
    text_encoder_path: str | None
    tokenizer_path: str | None


# Hashes of the state-dict keys and shapes (see `hash_model_file`) of the
# Wan2.2-TI2V-5B assets this package loads.
WAN22_MODEL_REGISTRY = [
    {
        "model_hash": "9c8818c2cbea55eca56c7b447df170da",
        "model_name": "wan_video_text_encoder",
        "model_class": WanTextEncoder,
    },
    {
        "model_hash": "1f5ab7703c6fc803fdded85ff040c316",
        "model_name": "wan_video_dit",
        "model_class": WanVideoDiT,
    },
    {
        "model_hash": "e1de6c02cdac79f8b739f4d3698cd216",
        "model_name": "wan_video_vae",
        "model_class": WanVideoVAE38,
        "state_dict_converter": wan_video_vae_state_dict_converter,
    },
]


def _validate_dit_config(dit_config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(dit_config, dict):
        raise ValueError(f"`dit_config` must be a dict, got {type(dit_config)}")
    validated = dict(dit_config)
    signature = inspect.signature(WanVideoDiT.__init__)
    allowed_keys = set()
    required_keys = set()
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        allowed_keys.add(name)
        if param.default is inspect.Signature.empty:
            required_keys.add(name)
    unknown_keys = sorted(set(validated) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            f"Unknown keys in `dit_config`: {unknown_keys}. Allowed keys: {sorted(allowed_keys)}"
        )
    missing_keys = sorted(required_keys - set(validated))
    if missing_keys:
        raise ValueError(
            f"Missing required keys in `dit_config`: {missing_keys}. Please specify all required WanVideoDiT constructor args."
        )
    return validated


def _load_registered_model(
    path,
    model_name: str,
    torch_dtype: torch.dtype,
    device: str,
    model_kwargs_override: dict[str, Any] | None = None,
):
    model_hash = hash_model_file(path)
    matched_config = None
    for config in WAN22_MODEL_REGISTRY:
        if config["model_hash"] == model_hash and config["model_name"] == model_name:
            matched_config = config
            break
    if matched_config is None:
        raise ValueError(
            f"Cannot detect model type for {model_name}. File: {path}. Model hash: {model_hash}. This standalone package follows DiffSynth hash-based loading."
        )
    model_class = matched_config["model_class"]
    model_kwargs = dict(matched_config.get("extra_kwargs", {}))
    if model_kwargs_override is not None:
        model_kwargs.update(model_kwargs_override)
    state_dict_converter = matched_config.get("state_dict_converter")
    model = model_class(**model_kwargs)
    state_dict = load_state_dict(path, torch_dtype=torch_dtype, device="cpu")
    if state_dict_converter is not None:
        state_dict = state_dict_converter(state_dict)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device=device, dtype=torch_dtype)
    return model


def _depth_latent_scale_from_state(
    state_dict: dict[str, Any], *, z_dim: int, require_latent_stats: bool
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not require_latent_stats:
        return None
    ready_value = state_dict.get("latent_stats_ready")
    ready = (
        bool(torch.as_tensor(ready_value).item()) if ready_value is not None else False
    )
    if not ready:
        raise RuntimeError(
            "Depth VAE latent statistics are not ready. Run `python -m vae.compute_latent_stats` and use the resulting `_with_depth_stats.pt` checkpoint."
        )
    missing = [
        key
        for key in ("latent_mean", "latent_std")
        if key not in state_dict or state_dict[key] is None
    ]
    if missing:
        raise ValueError(
            "Depth VAE checkpoint marks latent statistics ready but is missing "
            + ", ".join(missing)
            + "."
        )
    mean = torch.as_tensor(state_dict.get("latent_mean"), dtype=torch.float32)
    std = torch.as_tensor(state_dict.get("latent_std"), dtype=torch.float32)
    if tuple(mean.shape) != (int(z_dim),) or tuple(std.shape) != (int(z_dim),):
        raise ValueError(
            f"Depth VAE latent stats shape mismatch: mean={tuple(mean.shape)}, std={tuple(std.shape)}, z_dim={z_dim}."
        )
    if not bool(torch.isfinite(mean).all().item()) or not bool(
        torch.isfinite(std).all().item()
    ):
        raise ValueError("Depth VAE latent statistics contain non-finite values.")
    if bool((std <= 0.0).any().item()):
        raise ValueError("Depth VAE latent standard deviations must be positive.")
    return (mean, std)


def _load_wan_video_vae(
    path,
    torch_dtype: torch.dtype,
    device: str,
    vae_variant: str = "rgb",
    depth_vae_require_latent_stats: bool = False,
):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"VAE checkpoint not found: {path}")
    vae_variant = str(vae_variant).strip().lower()
    if vae_variant not in {"rgb", "depth"}:
        raise ValueError(
            f"Unsupported VAE variant: {vae_variant}. Expected one of: ['rgb', 'depth']."
        )
    if vae_variant == "depth":
        logger.info("Loading finetuned WAN depth VAE checkpoint: %s", path)
        payload = torch.load(path, map_location="cpu")
        state_dict = (
            payload.get("model_state", payload)
            if isinstance(payload, dict)
            else payload
        )
        if not isinstance(state_dict, dict):
            raise TypeError(
                f"Unsupported depth VAE checkpoint payload type: {type(payload)}"
            )
        latent_projection = state_dict.get("model.conv2.weight")
        z_dim = (
            int(latent_projection.shape[0])
            if isinstance(latent_projection, torch.Tensor)
            else 48
        )
        if z_dim not in (16, 48):
            raise ValueError(f"Unsupported depth VAE latent channel count: {z_dim}.")
        model = WanDepthVAE16() if z_dim == 16 else WanDepthVAE38()
        latent_scale = _depth_latent_scale_from_state(
            state_dict,
            z_dim=model.z_dim,
            require_latent_stats=bool(depth_vae_require_latent_stats),
        )
        if z_dim == 16:
            # Apart from its latent statistics, the z16 file holds exactly the
            # network tensors.
            statistics = {"latent_mean", "latent_std", "latent_stats_ready"}
            model_state = {
                key: value
                for (key, value) in state_dict.items()
                if key not in statistics
            }
        else:
            model_state = {
                key: value
                for (key, value) in state_dict.items()
                if key in model.state_dict()
            }
        load_result = model.load_state_dict(model_state, strict=z_dim == 16)
        if load_result.unexpected_keys:
            logger.warning(
                "Unexpected keys when loading finetuned depth VAE: %s",
                load_result.unexpected_keys,
            )
        if load_result.missing_keys:
            logger.warning(
                "Missing keys when loading finetuned depth VAE: %s",
                load_result.missing_keys,
            )
        if latent_scale is not None:
            (mean, std) = latent_scale
            model.mean = mean
            model.std = std
            model.scale = [model.mean, 1.0 / model.std]
            logger.info("Loaded depth-specific VAE latent statistics from %s", path)
        else:
            logger.warning(
                "Using the legacy Wan latent scale for depth VAE checkpoint %s because depth_vae_require_latent_stats=false.",
                path,
            )
        return model.to(device=device, dtype=torch_dtype)
    try:
        return _load_registered_model(
            str(path), "wan_video_vae", torch_dtype=torch_dtype, device=device
        )
    except (ValueError, TypeError, AttributeError) as exc:
        logger.info("Loading finetuned WAN video VAE checkpoint: %s (%s)", path, exc)
    payload = torch.load(path, map_location="cpu")
    state_dict = (
        payload.get("model_state", payload) if isinstance(payload, dict) else payload
    )
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported VAE checkpoint payload type: {type(payload)}")
    model = WanVideoVAE38()
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.unexpected_keys:
        logger.warning(
            "Unexpected keys when loading finetuned VAE: %s",
            load_result.unexpected_keys,
        )
    if load_result.missing_keys:
        logger.warning(
            "Missing keys when loading finetuned VAE: %s", load_result.missing_keys
        )
    return model.to(device=device, dtype=torch_dtype)


def _resolve_configs(
    model_id: str, tokenizer_model_id: str, redirect_common_files: bool = True
):
    dit_config = ModelConfig(
        model_id=model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors"
    )
    text_config = ModelConfig(
        model_id=model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"
    )
    vae_config = ModelConfig(model_id=model_id, origin_file_pattern="Wan2.2_VAE.pth")
    tokenizer_config = ModelConfig(
        model_id=tokenizer_model_id, origin_file_pattern="google/umt5-xxl/"
    )
    if redirect_common_files:
        redirect_dict = {
            "models_t5_umt5-xxl-enc-bf16.pth": (
                "DiffSynth-Studio/Wan-Series-Converted-Safetensors",
                "models_t5_umt5-xxl-enc-bf16.safetensors",
            ),
            "Wan2.2_VAE.pth": (
                "DiffSynth-Studio/Wan-Series-Converted-Safetensors",
                "Wan2.2_VAE.safetensors",
            ),
        }
        (text_config.model_id, text_config.origin_file_pattern) = redirect_dict[
            text_config.origin_file_pattern
        ]
        (vae_config.model_id, vae_config.origin_file_pattern) = redirect_dict[
            vae_config.origin_file_pattern
        ]
    return (dit_config, text_config, vae_config, tokenizer_config)


def load_wan22_ti2v_5b_components(
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
    tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
    tokenizer_max_len: int = 512,
    redirect_common_files: bool = True,
    dit_config: dict[str, Any] | None = None,
    skip_dit_load_from_pretrain: bool = False,
    load_text_encoder: bool = True,
    vae_checkpoint_path: str | None = None,
    vae_variant: str = "rgb",
    depth_vae_require_latent_stats: bool = False,
):
    logger.info("Loading Wan2.2-TI2V-5B components...")
    start = time.time()
    if dit_config is None:
        raise ValueError("`dit_config` is required for Wan2.2-TI2V-5B loading.")
    validated_dit_config = _validate_dit_config(dit_config)
    (dit_model_config, text_config, vae_config, tokenizer_config) = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=redirect_common_files,
    )
    if vae_checkpoint_path is None:
        vae_config.download_if_necessary()
    if load_text_encoder:
        text_config.download_if_necessary()
        tokenizer_config.download_if_necessary()
    if skip_dit_load_from_pretrain:
        logger.info(
            "Skipping pretrained video DiT load (`skip_dit_load_from_pretrain=True`); initializing video expert randomly and expecting checkpoint override."
        )
        dit: WanVideoDiT = WanVideoDiT(**validated_dit_config).to(
            device=device, dtype=torch_dtype
        )
        dit_path = SKIPPED_PRETRAIN_SENTINEL
    else:
        dit_model_config.download_if_necessary()
        dit = _load_registered_model(
            dit_model_config.path,
            "wan_video_dit",
            torch_dtype=torch_dtype,
            device=device,
            model_kwargs_override=validated_dit_config,
        )
        dit_path = str(dit_model_config.path)
    text_encoder: WanTextEncoder | None = None
    tokenizer: HuggingfaceTokenizer | None = None
    text_encoder_path: str | None = None
    tokenizer_path: str | None = None
    if load_text_encoder:
        text_encoder = _load_registered_model(
            text_config.path,
            "wan_video_text_encoder",
            torch_dtype=torch_dtype,
            device=device,
        )
        tokenizer = HuggingfaceTokenizer(
            name=tokenizer_config.path,
            seq_len=int(tokenizer_max_len),
            clean="whitespace",
        )
        text_encoder_path = str(text_config.path)
        tokenizer_path = str(tokenizer_config.path)
    else:
        logger.info(
            "Skipping pretrained text encoder/tokenizer load (`load_text_encoder=False`); training must provide cached `context/context_mask`."
        )
    vae_path = str(
        vae_config.path if vae_checkpoint_path is None else vae_checkpoint_path
    )
    vae: WanVideoVAE = _load_wan_video_vae(
        vae_path,
        torch_dtype=torch_dtype,
        device=device,
        vae_variant=vae_variant,
        depth_vae_require_latent_stats=depth_vae_require_latent_stats,
    )
    logger.info(
        "Finished loading Wan2.2-TI2V-5B components in %.2f seconds.",
        time.time() - start,
    )
    return Wan22LoadedComponents(
        dit=dit,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        dit_path=dit_path,
        vae_path=vae_path,
        text_encoder_path=text_encoder_path,
        tokenizer_path=tokenizer_path,
    )
