from __future__ import annotations
from typing import Callable, Optional, Union

import torch
from torch import nn
from pathlib import Path
import numpy as np

from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.activations import ACT2FN

try:
    import sentencepiece
except ImportError as exc:  # pragma: no cover - optional dep
    raise ImportError("Please install sentencepiece to use PaligemmaTokenizer.") from exc

class PaligemmaTokenizer:
    """
    Tokenizes prompt and discretized state into Paligemma tokens.

    This is a thin wrapper around sentencepiece; the caller must provide a local
    path to `paligemma_tokenizer.model` (set `tokenizer_path` in config).
    """

    def __init__(self, max_len: int = 200, tokenizer_path: str | None = None):
        self._max_len = max_len
        if tokenizer_path is None:
            raise ValueError("tokenizer_path is required for PaligemmaTokenizer.")
        path = Path(tokenizer_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"tokenizer_path not found: {path}")
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(self, prompt: str, state: torch.Tensor | "np.ndarray"):
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        # discretize state to 256 bins assuming normalized [-1,1]
        if not isinstance(state, (torch.Tensor,)):
            state_np = state
        else:
            state_np = state.detach().cpu().numpy()
        discretized_state = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        state_str = " ".join(map(str, discretized_state))
        full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
        tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [0] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + [False] * len(padding)
            tokens = tokens + padding
        else:
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len
        return tokens, mask


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, cond_dim: Optional[int] = None):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.cond_dim = cond_dim
        
        # Dense layer for adaptive normalization (if cond_dim is provided)
        if cond_dim is not None:
            #self.dense = nn.Linear(cond_dim, dim * 3, bias=True, dtype=torch.bfloat16)
            self.dense = nn.Linear(cond_dim, dim * 3, bias=True)
            # Initialize with zeros (matches source implementation)
            nn.init.zeros_(self.dense.weight)
        else:
            self.weight = nn.Parameter(torch.zeros(dim, dtype=torch.bfloat16))
            self.dense = None


    def _norm(self, x):
        # Compute variance in float32 (like the source implementation)
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        # Compute normalization in float32
        normed_inputs = x * torch.rsqrt(var + self.eps)
        return normed_inputs

    def forward(self, x, cond=None):
        dtype = x.dtype  # original dtype, could be half-precision
        normed_inputs = self._norm(x)
        
        if cond is None or self.dense is None:
            # regular RMSNorm
            # scale by learned parameter in float32 (matches source implementation)
            normed_inputs = normed_inputs * (1.0 + self.weight.float())
            return normed_inputs.to(dtype), None  # return in original dtype with None gate
        
        # adaptive RMSNorm (if cond is provided and dense layer exists)
        if cond.shape[-1] != self.cond_dim:
            raise ValueError(f"Expected cond dimension {self.cond_dim}, got {cond.shape[-1]}")
        
        #self.dense.to(dtype=torch.bfloat16).to(dtype=torch.float32)
        modulation = self.dense(cond)
        # Reshape modulation to broadcast properly: [batch, 1, features] for [batch, seq, features]
        if len(x.shape) == 3:  # [batch, seq, features]
            modulation = modulation.unsqueeze(1)
        
        scale, shift, gate = torch.chunk(modulation, 3, dim=-1)
        
        # Apply adaptive normalization: use model weight dtype to ensure compatibility
        # model_dtype = self.dense.weight.dtype  # Use the model's dtype (bfloat16)
        # scale = scale.to(model_dtype)
        # shift = shift.to(model_dtype)
        # gate = gate.to(model_dtype)
        # normed_inputs = normed_inputs.to(model_dtype)  # Convert normed_inputs to model dtype
        
        normed_inputs = normed_inputs * (1 + scale.to(torch.float32)) + shift.to(torch.float32)

        return normed_inputs.to(dtype), gate.to(dtype)

    def extra_repr(self):
        if self.dense is not None:
            repr_str = f"adaptive=True, cond_dim={self.cond_dim}, eps={self.eps}"
        else:
            repr_str = f"{tuple(self.weight.shape)}, eps={self.eps}"
        return repr_str


class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class GemmaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

