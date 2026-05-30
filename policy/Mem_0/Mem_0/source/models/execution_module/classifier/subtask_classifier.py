# A lightweight subtask end classifier that consumes fused memory features.
"""
SubtaskEndClassifier
Inputs:
    - fused_hidden: Tensor of shape (B, 1, H) or (B, H), fused summary token from memory bank.
    - labels (optional): Tensor of shape (B,), binary ground truth.
Outputs:
    - logits: (B,) raw logits.
    - loss: scalar when labels provided.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMLPBlock(nn.Module):
    """Two-layer MLP block with residual (projects skip if dims differ) and LayerNorm."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.skip = None if in_dim == out_dim else nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.skip is None else self.skip(x)
        z = self.norm (x)
        h = self.fc1 (z)
        h = F.gelu (h)
        h = self.dropout (h)
        h = self.fc2 (h)
        h = self.dropout (h)
        return h + residual


class SubtaskEndClassifier(nn.Module):
    def __init__(
        self,
        hidden_sizes,
        dropout: float = 0.1,
        pos_weight: Optional[float] = None,
        focal_gamma: float = 0.0,
    ):
        super().__init__()
        self.focal_gamma = focal_gamma

        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(in_dim, out_dim, dropout) for in_dim, out_dim in zip(hidden_sizes, hidden_sizes[1:])]
        )
        self.final_norm = nn.LayerNorm(hidden_sizes[-1])
        self.head = nn.Linear(hidden_sizes[-1], 1)

        # Use buffer for automatic device placement.
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))
        else:
            self.pos_weight = None

    def forward(
        self,
        fused_hidden: torch.Tensor,
        labels: torch.Tensor,
    ):
        """
        Args:
            fused_hidden: (B, 1, H) or (B, H)
            labels: (B,) binary labels
        """
        if fused_hidden.dim() == 3:
            fused_hidden = fused_hidden.squeeze(1)

        h = fused_hidden
        for block in self.blocks:
            h = block(h)
        h = self.final_norm(h)
        logits = self.head(h).squeeze(-1)
        
        # Binary cross-entropy with logits; reduction handled manually for masking.
        pos_weight = None
        if self.pos_weight is not None:
            pos_weight = self.pos_weight.to(dtype=logits.dtype, device=logits.device)
        bce = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=pos_weight,
            reduction="none",
        )

        if self.focal_gamma > 0:
            # Focal weight to down-weight easy negatives.
            prob = torch.sigmoid(logits)
            pt = prob * labels + (1 - prob) * (1 - labels)
            focal_weight = (1 - pt).pow(self.focal_gamma)
            bce = bce * focal_weight

        normalizer = max(labels.numel(), 1)

        loss = (bce.sum() / normalizer).float()  # ensure at least fp32 for stability
        return {"logits": logits, "loss": loss}

    @torch.inference_mode()
    def predict(self, fused_hidden: torch.Tensor):
        """
        Predict logits and probabilities without labels.

        Args:
            fused_hidden: (B, 1, H) or (B, H)
        Returns:
            dict with:
                logits: (B,)
                prob: (B,)
        """
        if fused_hidden.dim() == 3:
            fused_hidden = fused_hidden.squeeze(1)
        h = fused_hidden
        for block in self.blocks:
            h = block(h)
        h = self.final_norm(h)
        logits = self.head(h).squeeze(-1)
        prob = torch.sigmoid(logits)
        return {"logits": logits, "prob": prob}
