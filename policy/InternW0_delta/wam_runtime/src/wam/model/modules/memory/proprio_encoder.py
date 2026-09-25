"""Proprioceptive token encoding used by context and memory streams."""

import math
from typing import Optional

import torch
import torch.nn as nn


class ProprioContextEncoder(nn.Linear):
    """Project proprioception into context space with an explicit token type."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        valid_input_scale: float = 1.0,
        dimension_valid_mask=None,
    ) -> None:
        super().__init__(int(in_features), int(out_features))
        self.type_embedding = nn.Parameter(torch.zeros(int(out_features)))
        self.register_buffer("feature_scale", None, persistent=False)
        scale_value = float(valid_input_scale)
        if not math.isfinite(scale_value) or scale_value <= 0.0:
            raise ValueError(
                f"valid_input_scale must be finite and positive, got {scale_value}."
            )
        if scale_value != 1.0:
            if dimension_valid_mask is None:
                raise ValueError(
                    "dimension_valid_mask is required when valid_input_scale != 1.0."
                )
            valid = torch.as_tensor(dimension_valid_mask, dtype=torch.bool)
            if valid.ndim != 1 or int(valid.numel()) != int(in_features):
                raise ValueError(
                    f"dimension_valid_mask must be [{in_features}], got {tuple(valid.shape)}"
                )
            if not bool(valid.any().item()):
                raise ValueError(
                    "dimension_valid_mask must contain at least one valid dimension."
                )
            scale = torch.ones(int(in_features), dtype=torch.float32)
            scale[valid] = scale_value
            self.feature_scale = scale

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.feature_scale is not None:
            input = input * self.feature_scale.to(
                device=input.device, dtype=input.dtype
            )
        return super().forward(input) + self.type_embedding


def append_proprio_to_context(
    model,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    proprio: Optional[torch.Tensor],
    *,
    proprio_encoder: Optional[nn.Module] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoder = model.proprio_encoder if proprio_encoder is None else proprio_encoder
    if encoder is None or proprio is None:
        return context, context_mask
    if proprio.ndim != 2:
        raise ValueError(
            f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}"
        )
    if model.proprio_dim is None or proprio.shape[1] != model.proprio_dim:
        raise ValueError(
            f"`proprio` last dim must be {model.proprio_dim}, got {proprio.shape[1]}"
        )
    proprio_token = encoder(
        proprio.to(device=model.device, dtype=context.dtype).unsqueeze(1)
    ).to(dtype=context.dtype)  # [B, 1, D]
    proprio_mask = torch.ones(
        (context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device
    )
    return (
        torch.cat([context, proprio_token], dim=1),
        torch.cat([context_mask, proprio_mask], dim=1),
    )



def encode_memory_proprio_sequence_tokens(
    model,
    proprio: Optional[torch.Tensor],
    proprio_is_pad: Optional[torch.Tensor],
    *,
    token_dtype: torch.dtype,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if proprio is None:
        return None, None
    if model.memory_proprio_token_encoder is None or model.proprio_dim is None:
        raise ValueError(
            "Memory proprio input was provided but memory proprio token encoder is disabled."
        )
    if proprio.ndim != 3 or int(proprio.shape[-1]) != int(model.proprio_dim):
        raise ValueError(f"`proprio` must be [B,T,D], got {tuple(proprio.shape)}")
    batch_size, seq_len, _ = proprio.shape
    if proprio_is_pad is None:
        valid = torch.ones(
            (batch_size, seq_len), dtype=torch.bool, device=proprio.device
        )
    else:
        proprio_is_pad = proprio_is_pad.to(device=proprio.device, dtype=torch.bool)
        if proprio_is_pad.ndim != 2 or tuple(proprio_is_pad.shape) != (
            batch_size,
            seq_len,
        ):
            raise ValueError(
                "`proprio_is_pad` shape mismatch: "
                f"got {tuple(proprio_is_pad.shape)} vs proprio {tuple(proprio.shape)}"
            )
        valid = ~proprio_is_pad
    tokens = (
        model.memory_proprio_token_encoder(
            proprio.to(device=model.device, dtype=token_dtype).reshape(
                batch_size * seq_len, -1
            )
        )
        .reshape(batch_size, seq_len, -1)
        .to(dtype=token_dtype)
    )
    tokens = tokens.masked_fill(~valid.to(device=tokens.device).unsqueeze(-1), 0.0)
    return tokens, valid.to(device=tokens.device, dtype=torch.bool)
