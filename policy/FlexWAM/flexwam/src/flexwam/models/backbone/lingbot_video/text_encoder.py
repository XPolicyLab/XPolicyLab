from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

LINGBOT_ENCODER_ID = "lingbotqwen3vl"


def resolve_project_relative_path(path_like: str | Path, project_root: Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return path


def load_lingbot_qwen(model_path: Path, device: str, dtype: torch.dtype):
    try:
        from transformers import Qwen3VLConfig, Qwen3VLProcessor, Qwen3VLTextModel
    except ImportError as exc:
        raise ImportError(
            "The installed transformers package does not provide the Qwen3VL text-encoder classes. "
            "Install a transformers release that includes Qwen3VL, for example `transformers==4.57.1`."
        ) from exc

    processor = Qwen3VLProcessor.from_pretrained(str(model_path / "tokenizer"), trust_remote_code=True)
    config = Qwen3VLConfig.from_pretrained(str(model_path / "tokenizer"), trust_remote_code=True).text_config
    text_encoder = Qwen3VLTextModel(config)
    state_dict = load_lingbot_qwen_text_state_dict(model_path / "text_encoder")
    missing, unexpected = text_encoder.load_state_dict(state_dict, strict=False)
    unexpected = [key for key in unexpected if not key.startswith("visual.")]
    if missing or unexpected:
        raise ValueError(
            "LingBot/Qwen text encoder state_dict mismatch after fused-key conversion: "
            f"missing={list(missing)[:20]} unexpected={list(unexpected)[:20]}"
        )
    text_encoder = text_encoder.to(device=device, dtype=dtype)
    text_encoder.eval()
    return processor, text_encoder


def load_lingbot_qwen_text_state_dict(text_encoder_dir: Path) -> dict[str, torch.Tensor]:
    index_path = text_encoder_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            index = json.load(f)
        shard_names = sorted(set(index["weight_map"].values()))
        raw_state: dict[str, torch.Tensor] = {}
        for shard_name in shard_names:
            raw_state.update(load_file(str(text_encoder_dir / shard_name), device="cpu"))
    else:
        raw_state = load_file(str(text_encoder_dir / "model.safetensors"), device="cpu")

    state: dict[str, torch.Tensor] = {}
    for key, value in raw_state.items():
        if key.endswith(".self_attn.qkv_proj.weight"):
            prefix = key.removesuffix(".self_attn.qkv_proj.weight")
            q, k, v = value.split([4096, 1024, 1024], dim=0)
            state[f"{prefix}.self_attn.q_proj.weight"] = q.contiguous()
            state[f"{prefix}.self_attn.k_proj.weight"] = k.contiguous()
            state[f"{prefix}.self_attn.v_proj.weight"] = v.contiguous()
        elif key.endswith(".mlp.gate_up_proj.weight"):
            prefix = key.removesuffix(".mlp.gate_up_proj.weight")
            gate, up = value.chunk(2, dim=0)
            state[f"{prefix}.mlp.gate_proj.weight"] = gate.contiguous()
            state[f"{prefix}.mlp.up_proj.weight"] = up.contiguous()
        else:
            state[key] = value
    return state


def extract_context(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        for key in ("prompt_embeds", "last_hidden_state"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value
        hidden_states = output.get("hidden_states")
        if hidden_states is not None:
            return hidden_states[-1]
    for attr in ("prompt_embeds", "last_hidden_state"):
        value = getattr(output, attr, None)
        if isinstance(value, torch.Tensor):
            return value
    hidden_states = getattr(output, "hidden_states", None)
    if hidden_states is not None:
        return hidden_states[-1]
    raise ValueError(f"Cannot extract text context from LingBot/Qwen output type: {type(output)}")


def encode_prompt_online(
    processor,
    text_encoder,
    prompt: str,
    *,
    context_len: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = processor(
        text=[prompt],
        padding="max_length",
        max_length=context_len,
        truncation=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    mask = inputs.get("attention_mask")
    if mask is None:
        raise ValueError("LingBot/Qwen processor did not return `attention_mask`.")
    with torch.no_grad():
        try:
            output = text_encoder(**inputs, output_hidden_states=True)
        except TypeError:
            output = text_encoder(**inputs)
    context = extract_context(output)
    if context.ndim != 3 or context.shape[0] != 1:
        raise ValueError(f"LingBot/Qwen context must be [1,L,D], got {tuple(context.shape)}")
    if context.shape[1] != context_len:
        raise ValueError(f"LingBot/Qwen context_len mismatch: expected {context_len}, got {context.shape[1]}")
    return context[0].detach(), mask[0].detach().bool()


def load_cached_text_context(
    prompt: str,
    *,
    cache_dir: Path,
    context_len: int,
    encoder_id: str,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    cache_path = cache_dir / f"{hashed}.text_len{context_len}.{encoder_id}.pt"
    legacy_cache_path = cache_dir / f"{hashed}.t5_len{context_len}.wan22ti2v5b.pt"
    if encoder_id == "wan22ti2v5b" and not cache_path.exists():
        cache_path = legacy_cache_path
    if not cache_path.exists():
        return None

    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    context = payload["context"]
    context_mask = payload["mask"].bool()
    if context.ndim != 2:
        raise ValueError(
            f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
        )
    if context_mask.ndim != 1:
        raise ValueError(
            f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
        )
    if context.shape[0] != context_len or context_mask.shape[0] != context_len:
        raise ValueError(
            f"Cached context_len mismatch in {cache_path}: expected {context_len}, "
            f"got context={context.shape[0]} mask={context_mask.shape[0]}"
        )
    logger.info("Loaded LingBot text embedding cache: %s", cache_path)
    return context, context_mask


def finalize_text_context(context: torch.Tensor, context_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    context = context.clone()
    context_mask = context_mask.clone().bool()
    context[~context_mask] = 0.0
    context_mask = torch.ones_like(context_mask)
    return context, context_mask


class LingBotTextEncoderAdapter:
    """Deploy-time LingBot text conditioning.

    Lookup order per prompt: in-memory (this process) → disk cache → online Qwen3VL.
    A closed-loop eval keeps the same instruction, so the first chunk materializes
    the embedding and later chunks reuse it (same idea as OpenWAM `prompt_embed_cache`).
    """

    def __init__(
        self,
        *,
        model_path: Path,
        cache_dir: Path | None,
        context_len: int,
        encoder_id: str,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"LingBot model path does not exist: {model_path}")
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.context_len = int(context_len)
        self.encoder_id = str(encoder_id)
        self.device = str(device)
        self.dtype = dtype
        self.processor = None
        self.text_encoder = None
        self._memory_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def _ensure_online_encoder(self) -> None:
        if self.processor is not None and self.text_encoder is not None:
            return
        logger.info(
            "Loading LingBot/Qwen text encoder from %s for online prompt encoding.",
            self.model_path,
        )
        self.processor, self.text_encoder = load_lingbot_qwen(
            self.model_path,
            device=self.device,
            dtype=self.dtype,
        )

    def _pack_context(
        self, context: torch.Tensor, context_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return context.unsqueeze(0), context_mask.unsqueeze(0)

    @torch.no_grad()
    def encode_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        mem = self._memory_cache.get(prompt)
        if mem is not None:
            context, context_mask = mem
            return self._pack_context(context.clone(), context_mask.clone())

        cached = None
        if self.cache_dir is not None:
            cached = load_cached_text_context(
                prompt,
                cache_dir=self.cache_dir,
                context_len=self.context_len,
                encoder_id=self.encoder_id,
            )
        if cached is not None:
            context, context_mask = cached
        else:
            self._ensure_online_encoder()
            assert self.processor is not None and self.text_encoder is not None
            logger.info("LingBot text cache miss; encoding prompt online with Qwen3VL.")
            context, context_mask = encode_prompt_online(
                self.processor,
                self.text_encoder,
                prompt,
                context_len=self.context_len,
                device=self.device,
            )
        context, context_mask = finalize_text_context(context, context_mask)
        self._memory_cache[prompt] = (context.detach().clone(), context_mask.detach().clone())
        logger.info("Cached LingBot text embedding in memory for this eval session.")
        return self._pack_context(context, context_mask)
