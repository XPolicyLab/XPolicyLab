"""Frozen Qwen provider contract. No vision inputs, generation or T5 bridge."""

from dataclasses import dataclass

import torch
from torch import nn

from wall_x.model.vap.qwen35_online import Qwen35OnlinePrompter
from wall_x.model.vap.text_conditioning import Qwen35TextCache, qwen35_encode_ids


@dataclass(frozen=True)
class TextCondition:
    """Raw floating [B,L,Dq] features and bool [B,L] mask, including EOS."""

    hidden_states: torch.Tensor
    valid_mask: torch.Tensor

    def __post_init__(self):
        if self.hidden_states.ndim != 3 or not self.hidden_states.is_floating_point():
            raise ValueError("Text hidden_states must be floating [B,L,Dq]")
        if self.valid_mask.dtype != torch.bool or self.valid_mask.shape != self.hidden_states.shape[:2]:
            raise ValueError("Text valid_mask must be bool [B,L]")
        if self.valid_mask.device != self.hidden_states.device:
            raise ValueError("Text features and validity must share a device")
        if not bool(self.valid_mask.any(dim=1).all()):
            raise ValueError("Every text sample must include at least EOS")

    @property
    def lengths(self):
        return self.valid_mask.sum(dim=1)

    def to(self, *, device=None, dtype=None):
        return TextCondition(self.hidden_states.to(device=device, dtype=dtype), self.valid_mask.to(device=device))


class QwenTextProvider:
    """Owns a frozen encoder outside policy parameters, FSDP and optimizer.

    Uses the existing plain_text_eos_last_hidden_v1 recipe. Online masks come
    from tokenization; cache masks come from stored lengths, never from zeros.
    """

    manages_placement = True

    def __init__(self, config):
        self.config = config
        if config.text_encoder_backend == "qwen35_online":
            self.provider = Qwen35OnlinePrompter(config)
        elif config.text_encoder_backend == "qwen35_cache":
            self.provider = Qwen35TextCache(
                config.text_feature_cache_path, expected_id=config.text_feature_cache_id,
                hidden_dim=config.text_feature_hidden_dim, seq_len=config.text_feature_seq_len,
            )
        else:
            raise ValueError("VAPv2 accepts Qwen3.5 only")

    @property
    def text_encoder(self):
        return self.provider.text_encoder

    @torch.no_grad()
    def encode(self, prompts, *, device, dtype):
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        hidden = self.provider.encode_prompt(prompts, device=device)
        if isinstance(self.provider, Qwen35TextCache):
            lengths = [self.provider.lengths[p] for p in prompts]
        else:
            lengths = [len(qwen35_encode_ids(
                self.provider.tokenizer, p, seq_len=self.config.text_feature_seq_len,
                eos_token_id=self.provider.tokenizer.eos_token_id,
                truncation=getattr(self.config, "qwen35_truncation", "none"),
            )) for p in prompts]
        lengths = torch.tensor(lengths, device=hidden.device)
        valid = torch.arange(hidden.shape[1], device=hidden.device)[None] < lengths[:, None]
        return TextCondition(hidden.to(dtype=dtype), valid)


class JointTextProjector(nn.Module):
    """Dq -> Dvideo directly. All trainable state stays inside this boundary."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.norm = nn.RMSNorm(input_dim, eps=1e-6)
        self.proj = nn.Sequential(nn.Linear(input_dim, output_dim), nn.SiLU(), nn.Linear(output_dim, output_dim))
        self.modality = nn.Parameter(torch.zeros(1, 1, output_dim))

    def forward(self, hidden, valid=None):
        if hidden.shape[-1] != self.input_dim:
            raise ValueError("Qwen hidden dimension differs from VAPv2 projector")
        if valid is None:
            valid = torch.ones(
                hidden.shape[:2], device=hidden.device, dtype=torch.bool
            )
        if valid.dtype != torch.bool or valid.shape != hidden.shape[:2]:
            raise ValueError("Projector validity must be bool [B,L]")
        # Sanitize invalid input BEFORE projection, not only its output.
        hidden = hidden.masked_fill(~valid[..., None], 0)
        return (self.proj(self.norm(hidden)) + self.modality.to(hidden)).masked_fill(~valid[..., None], 0)
