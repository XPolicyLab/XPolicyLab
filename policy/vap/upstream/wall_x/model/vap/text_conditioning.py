"""Opt-in frozen Qwen3.5 raw-text cache and trainable VAP input projection.

Cache entries are CPU bf16 [1,L,Dq], causal last-layer text features including
EOS, right-zero-padded to a fixed L. No image tokens or projector outputs are
cached. Runtime has no Qwen/Transformers dependency beyond existing VAP deps.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


FORMAT = "vap.qwen35.raw_text.v1"
PROJECTOR_KEY = "core.text_input_projector.proj.weight"
QWEN35_RECIPE_BASE = "plain_text_eos_last_hidden_v1"
QWEN35_RECIPE_HEAD = "plain_text_head_truncate_eos_last_hidden_v1"
QWEN35_TRUNCATION_MODES = ("none", "head")
CONTRACT_FIELDS = (
    "text_encoder_backend",
    "text_feature_cache_id",
    "text_feature_hidden_dim",
    "text_feature_seq_len",
    "text_projector_output_dim",
    "qwen35_truncation",
)


def cache_identity(metadata: dict, features: dict, lengths: dict) -> str:
    """Content identity independent of torch serialization or cache location."""
    digest = hashlib.sha256(
        json.dumps(
            metadata, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    )
    for prompt in sorted(features):
        key = json.dumps([prompt, lengths[prompt]], ensure_ascii=False).encode()
        digest.update(len(key).to_bytes(8, "little"))
        digest.update(key)
        digest.update(features[prompt].contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def qwen35_recipe_for_truncation(mode: str) -> str:
    """Return the immutable text recipe name for an explicit truncation mode."""
    if mode not in QWEN35_TRUNCATION_MODES:
        raise ValueError(
            f"qwen35_truncation must be one of {QWEN35_TRUNCATION_MODES}, got {mode!r}"
        )
    return QWEN35_RECIPE_HEAD if mode == "head" else QWEN35_RECIPE_BASE


def qwen35_truncation_for_recipe(recipe: str) -> str:
    if recipe == QWEN35_RECIPE_BASE:
        return "none"
    if recipe == QWEN35_RECIPE_HEAD:
        return "head"
    raise ValueError(f"Unsupported Qwen text recipe: {recipe}")


def qwen35_encode_ids(
    tokenizer, prompt: str, *, seq_len: int, eos_token_id: int, truncation: str = "none"
) -> list[int]:
    """Tokenize plain text and preserve its prefix when head truncation is enabled."""
    if seq_len < 1:
        raise ValueError("Qwen text sequence length must be positive")
    if truncation not in QWEN35_TRUNCATION_MODES:
        raise ValueError(
            f"qwen35_truncation must be one of {QWEN35_TRUNCATION_MODES}, got {truncation!r}"
        )
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    ids = [*ids, eos_token_id]
    if len(ids) > seq_len:
        if truncation != "head":
            raise ValueError(f"Prompt has {len(ids)} tokens > {seq_len}; no truncation")
        # Keep the beginning of the instruction and reserve the last slot for EOS.
        ids = [*ids[: seq_len - 1], eos_token_id]
    return ids


class Qwen35TextCache:
    """Cache-only provider with optional, lazy, rank-local CUDA residency.

    Default lookup is unchanged: concatenate CPU bf16 [1,L,Dq] entries then
    transfer the batch. VAP_QWEN35_CACHE_GPU_RESIDENT=1 instead uploads the
    complete validated bank once on the first CUDA request and gathers rows
    there. This plain object is not an nn.Module: raw features never become
    FSDP parameters/buffers or checkpoint state. Projector outputs are NOT
    cached. CPU preflight and cache identity remain independent of placement.
    """

    text_encoder = None
    skip_text_encoder = True

    def __init__(
        self,
        path,
        *,
        expected_id=None,
        hidden_dim=None,
        seq_len=None,
        gpu_resident=None,
    ):
        if gpu_resident is None:
            setting = os.environ.get("VAP_QWEN35_CACHE_GPU_RESIDENT", "0").lower()
            if setting not in {"0", "1", "false", "true", "off", "on"}:
                raise ValueError("VAP_QWEN35_CACHE_GPU_RESIDENT must be 0 or 1")
            gpu_resident = setting in {"1", "true", "on"}
        if not isinstance(gpu_resident, bool):
            raise TypeError("gpu_resident must be bool or None")
        self.gpu_resident = gpu_resident
        self._gpu_cache = None
        self._gpu_cache_pid = None
        value = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(value, dict) or value.get("format") != FORMAT:
            raise ValueError(
                "Expected versioned Qwen3.5 raw cache; T5/plain caches are incompatible"
            )
        self.metadata = value["metadata"]
        self.text_feature_cache = value["features"]
        self.lengths = value["lengths"]
        self.hidden_dim = int(self.metadata["hidden_dim"])
        self.seq_len = int(self.metadata["seq_len"])
        qwen35_truncation_for_recipe(self.metadata.get("recipe"))
        if self.metadata.get("model_type") not in {"qwen3_5", "qwen3_5_text"}:
            raise ValueError("Cache must come from a dense Qwen3.5 model")
        if self.hidden_dim <= 0 or self.seq_len <= 0:
            raise ValueError("Qwen cache dimensions must be positive")
        if not self.text_feature_cache or set(self.lengths) != set(
            self.text_feature_cache
        ):
            raise ValueError("Empty cache or mismatched prompt/length keys")
        for prompt, tensor in self.text_feature_cache.items():
            if not isinstance(prompt, str) or not isinstance(tensor, torch.Tensor):
                raise ValueError("Cache expects exact string prompts and tensors")
            if tensor.dtype != torch.bfloat16 or tensor.shape != (
                1,
                self.seq_len,
                self.hidden_dim,
            ):
                raise ValueError(f"Invalid raw cache shape/dtype for {prompt!r}")
            length = self.lengths[prompt]
            if not isinstance(length, int) or not 0 < length <= self.seq_len:
                raise ValueError(f"Invalid text length for {prompt!r}")
            if not torch.isfinite(tensor).all() or torch.count_nonzero(
                tensor[:, length:]
            ):
                raise ValueError(
                    f"Nonfinite features or nonzero right padding for {prompt!r}"
                )
        self.cache_id = cache_identity(
            self.metadata, self.text_feature_cache, self.lengths
        )
        if self.cache_id != value.get("cache_id"):
            raise ValueError("Qwen text cache content checksum mismatch")
        if expected_id is not None and expected_id != self.cache_id:
            raise ValueError("Qwen text cache identity differs from checkpoint/config")
        if hidden_dim is not None and int(hidden_dim) != self.hidden_dim:
            raise ValueError("Qwen text cache hidden dimension differs from config")
        if seq_len is not None and int(seq_len) != self.seq_len:
            raise ValueError("Qwen text cache sequence length differs from config")
        self._prompt_to_row = (
            {prompt: row for row, prompt in enumerate(self.text_feature_cache)}
            if self.gpu_resident
            else None
        )

    def _encode_gpu_resident(self, prompts, device):
        # Resolve an unspecified CUDA index only AFTER rank-local placement.
        # Do not hardcode cuda:0 or allocate in __init__/CPU-only preflight.
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if self._gpu_cache is None:
            # One temporary packed CPU bank; no per-step feature H2D copies.
            # Use normal tensors, not inference tensors: the trainable projector
            # must be able to save its input for backward.
            with torch.inference_mode(False), torch.no_grad():
                bank = torch.cat(list(self.text_feature_cache.values())).detach()
                self._gpu_cache = bank.to(device)
            self._gpu_cache_pid = os.getpid()
            logging.getLogger(__name__).info(
                "Qwen raw text GPU cache ready: device=%s prompts=%d bytes=%d "
                "shape=%s dtype=%s cache_id=%s; per-batch GPU index_select",
                device,
                len(self._prompt_to_row),
                self._gpu_cache.numel() * self._gpu_cache.element_size(),
                tuple(self._gpu_cache.shape),
                self._gpu_cache.dtype,
                self.cache_id,
            )
        elif self._gpu_cache_pid != os.getpid():
            raise RuntimeError("Qwen GPU cache cannot be reused across a process fork")
        elif self._gpu_cache.device != device:
            raise RuntimeError(
                "Qwen GPU cache is rank-local; cannot change device from "
                f"{self._gpu_cache.device} to {device}"
            )
        # Only B int64 indices travel to CUDA, not B*L*Dq feature tensors.
        rows = torch.tensor(
            [self._prompt_to_row[p] for p in prompts], dtype=torch.long, device=device
        )
        return self._gpu_cache.index_select(0, rows)

    def has_cached_prompt(self, prompt, positive=True):
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        return all(p in self.text_feature_cache for p in prompts)

    def encode_prompt(self, prompt, positive=True, device="cpu", **kwargs):
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        missing = [p for p in prompts if p not in self.text_feature_cache]
        if missing:
            raise ValueError(
                f"Qwen3.5 cache miss: {missing[:3]!r}; rebuild cache with exact prompts"
            )
        if self.gpu_resident and torch.device(device).type == "cuda":
            if not prompts:
                raise ValueError("Qwen GPU cache lookup requires at least one prompt")
            return self._encode_gpu_resident(prompts, torch.device(device))
        return torch.cat([self.text_feature_cache[p] for p in prompts]).to(device)


class TextInputProjector(nn.Module):
    """Trainable Dq->Wan text_dim; zero padding remains zero before Wan MLP.

    Input/output [B,L,D], floating point; fixed L avoids length recompilation.
    RMS normalization is parameter-free. Nonzero linear initialization lets
    gradients reach the projector on step 1. No text causal mask is modified.
    """

    def __init__(self, hidden_dim: int, output_dim: int, seq_len: int):
        super().__init__()
        self.hidden_dim, self.seq_len = hidden_dim, seq_len
        self.proj = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, hidden):
        if hidden.ndim != 3 or hidden.shape[1:] != (self.seq_len, self.hidden_dim):
            raise ValueError(
                f"Qwen raw context must be [B,{self.seq_len},{self.hidden_dim}]"
            )
        h = hidden.float()
        h = h * torch.rsqrt(h.square().mean(dim=-1, keepdim=True) + 1e-6)
        # Keep live compute in the input dtype even with fp32 optimizer masters.
        return F.linear(h.to(hidden.dtype), self.proj.weight.to(hidden.dtype))


def validate_text_config(config):
    backend = getattr(config, "text_encoder_backend", "t5")
    if backend not in {"t5", "qwen35_cache", "qwen35_online"}:
        raise ValueError(
            "text_encoder_backend must be t5, qwen35_cache or qwen35_online"
        )
    if backend == "t5":
        return
    if backend == "qwen35_cache" and (
        not config.text_feature_cache_path or not config.text_feature_cache_id
    ):
        raise ValueError(
            "qwen35_cache requires text_feature_cache_path and text_feature_cache_id"
        )
    if backend == "qwen35_cache":
        truncation = getattr(config, "qwen35_truncation", "none")
        if config.qwen35_text_recipe != qwen35_recipe_for_truncation(truncation):
            raise ValueError("Qwen cache recipe does not match qwen35_truncation")
    if backend == "qwen35_online":
        if config.text_feature_cache_path or config.text_feature_cache_id:
            raise ValueError("qwen35_online cannot read a text feature cache")
        if not config.qwen35_model_path or not config.qwen35_model_id:
            raise ValueError(
                "qwen35_online requires a local model path and immutable model identity"
            )
        truncation = getattr(config, "qwen35_truncation", "none")
        if config.qwen35_text_recipe != qwen35_recipe_for_truncation(truncation):
            raise ValueError("Qwen text recipe does not match qwen35_truncation")
        if config.qwen35_encode_micro_batch < 1:
            raise ValueError("qwen35_encode_micro_batch must be positive")
    if config.text_feature_hidden_dim <= 0 or config.text_feature_seq_len <= 0:
        raise ValueError(
            "Qwen text backends require positive hidden dimension and sequence length"
        )
    # Legacy VAP feeds Wan's 4096-wide cross-attention text bridge. Only the
    # explicit joint-v2 architecture feeds 3072-wide tokens to joint self-attn.
    architecture = str(getattr(config, "architecture_version", ""))
    joint_v2 = architecture == "vapv2.qwen_joint_ivava.v2"
    if architecture.startswith("vapv2.") and not joint_v2:
        raise ValueError(f"Unsupported VAPv2 text architecture: {architecture}")
    if (
        joint_v2
        and getattr(config, "text_injection_mode", None) != "joint_ivava_prefix"
    ):
        raise ValueError("VAPv2 requires text_injection_mode=joint_ivava_prefix")
    expected_projector_dim = 3072 if joint_v2 else 4096
    if config.text_projector_output_dim != expected_projector_dim:
        raise ValueError(
            f"{expected_projector_dim}-wide text projector required for "
            f"{getattr(config, 'architecture_version', 'legacy VAP')}"
        )
    if any(
        getattr(config, k, False)
        for k in (
            "use_qwen25",
            "use_qwen35",
            "qwen25_cascade_t5",
            "qwen35_cascade_t5",
            "text_feature_cache_online_fallback",
        )
    ):
        raise ValueError(
            "Qwen text backends cannot combine with legacy/cascade encoders or T5 fallback"
        )


def is_qwen_head_model_init(config: dict, source: dict) -> bool:
    """Allow only the explicit none->head recipe change for strict model-only VAPv2.

    Other text/architecture fields must still be checked by the caller.
    Directory/exact resumes and cached text never use this exception.
    """
    checkpoint = config.get("checkpoint", {})
    target = config["model"]
    path = Path(checkpoint.get("resume_from") or "")
    return (
        checkpoint.get("strict_model_load") is True
        and path.suffix == ".safetensors"
        and path.is_file()
        and not any(
            value
            for key, value in checkpoint.items()
            if key.startswith("resume_") and key != "resume_from"
        )
        and source.get("architecture_version")
        == target.get("architecture_version")
        == "vapv2.qwen_joint_ivava.v2"
        and source.get("text_encoder_backend")
        == target.get("text_encoder_backend")
        == "qwen35_online"
        and source.get("qwen35_truncation", "none") == "none"
        and source.get("qwen35_text_recipe") == QWEN35_RECIPE_BASE
        and target.get("qwen35_truncation") == "head"
        and target.get("qwen35_text_recipe") == QWEN35_RECIPE_HEAD
    )


def text_init_missing_keys(config: dict) -> list[str]:
    """Only a model-only T5->Qwen warm start may omit the new projector."""
    model = config["model"]
    checkpoint = config.get("checkpoint", {})
    resume = checkpoint.get("resume_from")
    if not resume:
        return []
    import yaml

    path = Path(resume)
    source_file = (path if path.is_dir() else path.parent) / "config.yml"
    if not source_file.is_file():
        raise FileNotFoundError(f"Text checkpoint contract requires {source_file}")
    source = yaml.safe_load(source_file.read_text())["model"]
    source_backend = source.get("text_encoder_backend", "t5")
    target_backend = model.get("text_encoder_backend", "t5")
    if source_backend == target_backend == "t5":
        return []
    if (
        checkpoint.get("strict_model_load", False)
        and source_backend == "t5"
        and target_backend in {"qwen35_cache", "qwen35_online"}
    ):
        if str(model.get("architecture_version", "")).startswith("vapv2."):
            raise ValueError(
                "VAPv2 cannot warm-start from a legacy T5 checkpoint; use fresh Wan initialization"
            )
        return [PROJECTOR_KEY]
    if (
        source_backend == "qwen35_online"
        and target_backend == "qwen35_cache"
        and model.get("architecture_version") == "vapv2.qwen_joint_ivava.v2"
    ):
        from wall_x.model.vapv2.model_init import validate_online_to_cache_model_init

        if validate_online_to_cache_model_init(config, source):
            return []  # Same trainable weights, no projector reset or partial load.
    fields = CONTRACT_FIELDS
    if "qwen35_online" in {source_backend, target_backend}:
        fields += ("qwen35_model_id", "qwen35_text_recipe", "prompt_template")
    head_model_init = is_qwen_head_model_init(config, source)
    if head_model_init:
        logging.warning(
            "Model-only initialization: Qwen truncation none -> head; no optimizer/state restore"
        )
    for key in fields:
        if head_model_init and key in {"qwen35_truncation", "qwen35_text_recipe"}:
            continue
        default = "none" if key == "qwen35_truncation" else None
        if source.get(key, default) != model.get(key, default):
            raise ValueError(f"Text encoder/cache changed since checkpoint: {key}")
    return []


def configure_qwen_text_cache(config: dict, path: str):
    """Materialize cache identity in config before typed loading/checkpointing."""
    cache = Qwen35TextCache(path)
    model = config.setdefault("model", {})
    model.update(
        text_encoder_backend="qwen35_cache",
        text_feature_cache_path=str(path),
        text_feature_cache_id=cache.cache_id,
        text_feature_hidden_dim=cache.hidden_dim,
        text_feature_seq_len=cache.seq_len,
        text_projector_output_dim=4096,
        qwen35_text_recipe=cache.metadata["recipe"],
        qwen35_truncation=qwen35_truncation_for_recipe(cache.metadata["recipe"]),
        use_qwen25=False,
        use_qwen35=False,
        qwen35_cascade_t5=False,
        qwen25_cascade_t5=False,
    )
    if "text_feature_cache_online_fallback" in model:
        model["text_feature_cache_online_fallback"] = False
    keys = text_init_missing_keys(config)
    checkpoint = config.setdefault("checkpoint", {})
    checkpoint["model_init_allowed_missing_keys"] = list(
        dict.fromkeys([*checkpoint.get("model_init_allowed_missing_keys", []), *keys])
    )
    print(
        f"Text: frozen Qwen3.5 raw cache {cache.cache_id}, "
        f"[B,{cache.seq_len},{cache.hidden_dim}] -> trainable projector -> 4096"
    )
