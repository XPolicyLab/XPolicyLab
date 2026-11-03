"""Optional attention kernels, with FA4 -> FA2 -> SDPA fallback."""

import importlib
import warnings
from dataclasses import dataclass

import torch
from torch.nn import functional as F


def validate_backend(name):
    if name not in ("fa4", "fa2", "sdpa", "auto"):
        raise ValueError("attn_backend must be one of: fa4, fa2, sdpa, auto")
    return name


@dataclass
class PackedMask:
    indices: torch.Tensor
    cumulative: torch.Tensor
    maximum: int
    batch: int
    length: int

    @classmethod
    def from_mask(cls, mask):
        lengths = mask.sum(-1, dtype=torch.int32)
        cumulative = F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))
        return cls(mask.flatten().nonzero().flatten(), cumulative, int(lengths.max().item()), *mask.shape)

    def pack(self, tensor):
        # (B,H,T,D) -> valid (N,H,D), preserving batch and token order.
        tensor = tensor.transpose(1, 2).flatten(0, 1)
        return tensor.index_select(0, self.indices).contiguous()

    def unpack(self, tensor):
        output = tensor.new_zeros(self.batch * self.length, *tensor.shape[1:])
        output = output.index_copy(0, self.indices, tensor)
        return output.reshape(self.batch, self.length, *tensor.shape[1:]).transpose(1, 2)


class AttentionBackend:
    def __init__(self, requested="fa4", device="cuda", dtype=torch.bfloat16):
        self.requested = validate_backend(requested)
        self.name, self.flash_varlen = "sdpa", None
        self.fallback_reasons = []
        device = torch.device(device)
        candidates = {"fa4": (4, 2), "auto": (4, 2), "fa2": (2,), "sdpa": ()}[requested]
        for generation in candidates:
            name = f"fa{generation}"
            try:
                if device.type != "cuda" or not torch.cuda.is_available():
                    raise ImportError("CUDA is unavailable")
                if dtype not in (torch.bfloat16, torch.float16):
                    raise ImportError("requires BF16 or FP16")
                minimum = 9 if generation == 4 else 8
                capability = torch.cuda.get_device_capability(device)
                if capability[0] < minimum:
                    raise ImportError(f"requires compute capability >= {minimum}.0; got {capability}")
                hf_utils = importlib.import_module("transformers.utils")
                if not getattr(hf_utils, f"is_flash_attn_{generation}_available")():
                    raise ImportError("package is not available to Transformers")
                module = importlib.import_module("flash_attn.cute" if generation == 4 else "flash_attn")
                # Check both entry points used by the vision/text backbone and MoT.
                if not callable(module.flash_attn_func) or not callable(module.flash_attn_varlen_func):
                    raise ImportError("required FlashAttention entry points are missing")
                if generation == 2:
                    importlib.import_module("flash_attn.bert_padding")
                    if not callable(module.flash_attn_with_kvcache):
                        raise ImportError("FA2 KV-cache entry point is missing")
                self.name, self.flash_varlen = name, module.flash_attn_varlen_func
                break
            except (ImportError, OSError, RuntimeError, AttributeError) as exc:
                self.fallback_reasons.append(f"{name}: {type(exc).__name__}: {exc}")
        if self.fallback_reasons:
            warnings.warn(f"[CogACT2] attention {requested} -> {self.name}: " + "; ".join(self.fallback_reasons))
        print(f"[CogACT2] attention {requested} -> {self.name}", flush=True)

    @property
    def hf_implementation(self):
        return {"fa4": "flash_attention_4", "fa2": "flash_attention_2", "sdpa": "sdpa"}[self.name]

    def layout(self, query_mask, key_mask):
        if self.name == "sdpa":
            return None
        return PackedMask.from_mask(query_mask), PackedMask.from_mask(key_mask)

    def __call__(self, q, k, v, key_mask, layout):
        if self.name == "sdpa":
            return F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask[:, None, None])
        query, key = layout
        if query.indices.numel() == 0:
            return torch.zeros_like(q)
        if key.maximum == 0:
            return torch.zeros_like(q)
        # FA uses separate query/key lengths: action queries see condition + action
        # keys. Auxiliary rows and padding must be excluded from both sequences.
        output = self.flash_varlen(
            query.pack(q), key.pack(k), key.pack(v),
            cu_seqlens_q=query.cumulative, cu_seqlens_k=key.cumulative,
            max_seqlen_q=query.maximum, max_seqlen_k=key.maximum, causal=False,
        )
        if isinstance(output, tuple):
            output = output[0]
        return query.unpack(output)


def attention_runtime(model):
    """Record kernels attached to the instantiated model, including partial FLA fallback."""
    backbone = model.backbone.model
    linear = next(layer.linear_attn for layer in backbone.language_model.layers if hasattr(layer, "linear_attn"))
    uses_fla = linear.chunk_gated_delta_rule.__module__.startswith("fla.")
    uses_causal = linear.causal_conv1d_fn is not None
    result = {
        "requested": model.attention.requested,
        "vlm_vision": backbone.visual.config._attn_implementation,
        "vlm_text": backbone.language_model.config._attn_implementation,
        "mot": model.action_head.mot.attention.name,
        "linear_attention": "fla" if uses_fla else "torch",
        "causal_conv1d": "causal-conv1d" if uses_causal else "torch",
        "gated_rms_norm": "fla" if type(linear.norm).__module__.startswith("fla.") else "torch",
        "fallback_reasons": list(model.attention.fallback_reasons),
    }
    for present, component in ((uses_fla, "flash-linear-attention"), (uses_causal, "causal-conv1d")):
        if not present:
            result["fallback_reasons"].append(f"{component}: unavailable; using PyTorch implementation")
    print(f"[CogACT2] linear attention: {result['linear_attention']}; convolution: {result['causal_conv1d']}", flush=True)
    return result
