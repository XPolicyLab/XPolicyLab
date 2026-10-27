from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from awomo.awomo.utils.logging_config import get_logger

logger = get_logger(__name__)


class MoT(nn.Module):
    def __init__(
        self,
        mixtures: Dict[str, nn.Module],
        mot_checkpoint_mixed_attn: bool = True,
        gqa_implementation: str = "repeat",
        force_flash_attention: bool = False,
    ):
        super().__init__()
        if not mixtures:
            raise ValueError("`mixtures` cannot be empty.")

        self.mixtures = nn.ModuleDict(mixtures)
        self.expert_order = list(self.mixtures.keys())
        self.mot_checkpoint_mixed_attn = mot_checkpoint_mixed_attn
        self.gqa_implementation = str(gqa_implementation).strip().lower()
        if self.gqa_implementation not in {"repeat", "sdpa"}:
            raise ValueError(
                f"`gqa_implementation` must be 'repeat' or 'sdpa', got {gqa_implementation!r}."
            )
        self.force_flash_attention = bool(force_flash_attention)
        if mot_checkpoint_mixed_attn:
            logger.info("Using gradient checkpointing for mixture attention.")

        first_expert = self.mixtures[self.expert_order[0]]
        self.num_layers = len(first_expert.blocks)
        self.num_heads = int(first_expert.num_heads)
        self.num_kv_heads = int(getattr(first_expert, "num_kv_heads", first_expert.num_heads))
        self.attn_head_dim = int(first_expert.attn_head_dim)
        self.block_protocol = str(getattr(first_expert, "block_protocol", "wan22"))

        for name in self.expert_order[1:]:
            expert = self.mixtures[name]
            protocol = str(getattr(expert, "block_protocol", "wan22"))
            num_kv_heads = int(getattr(expert, "num_kv_heads", expert.num_heads))
            checks = {
                "num_layers": (len(expert.blocks), self.num_layers),
                "num_heads": (int(expert.num_heads), self.num_heads),
                "num_kv_heads": (num_kv_heads, self.num_kv_heads),
                "attn_head_dim": (int(expert.attn_head_dim), self.attn_head_dim),
                "block_protocol": (protocol, self.block_protocol),
            }
            for attr, (got, expected) in checks.items():
                if got != expected:
                    raise ValueError(f"All experts must share {attr}; got {got} vs {expected} for expert {name}.")

        logger.info(
            "Initialized MoT with experts=%s protocol=%s layers=%d heads=%d kv_heads=%d head_dim=%d",
            self.expert_order,
            self.block_protocol,
            self.num_layers,
            self.num_heads,
            self.num_kv_heads,
            self.attn_head_dim,
        )
        logger.info(
            "MoT attention config: gqa_implementation=%s force_flash_attention=%s",
            self.gqa_implementation,
            self.force_flash_attention,
        )


    @staticmethod
    def _format_attention_mask(
        attention_mask: torch.Tensor,
        batch_size: int,
        query_len: int,
        key_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        mask = attention_mask.to(device=device, dtype=torch.bool)
        if mask.ndim == 2:
            if tuple(mask.shape) != (query_len, key_len):
                raise ValueError(f"2D attention mask must be {(query_len, key_len)}, got {tuple(mask.shape)}")
            return mask.view(1, 1, query_len, key_len)
        if mask.ndim == 3:
            if mask.shape[0] != batch_size or tuple(mask.shape[1:]) != (query_len, key_len):
                raise ValueError(
                    f"3D attention mask must be {(batch_size, query_len, key_len)}, got {tuple(mask.shape)}"
                )
            return mask.unsqueeze(1)
        if mask.ndim == 4:
            if mask.shape[0] not in (1, batch_size) or tuple(mask.shape[-2:]) != (query_len, key_len):
                raise ValueError(
                    f"4D attention mask must end with {(query_len, key_len)}, got {tuple(mask.shape)}"
                )
            return mask
        raise ValueError(f"attention_mask must be 2D/3D/4D, got shape {tuple(mask.shape)}")

    def _mixed_attention(
        self,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attn_probs: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch_size, query_len, _ = q_cat.shape
        key_len = k_cat.shape[1]
        H, H_kv, D = self.num_heads, self.num_kv_heads, self.attn_head_dim
        attn_mask = self._format_attention_mask(attention_mask, batch_size, query_len, key_len, q_cat.device)

        if return_attn_probs:
            q = q_cat.view(batch_size, query_len, H, D).transpose(1, 2)
            k = k_cat.view(batch_size, key_len, H_kv, D).transpose(1, 2)
            v = v_cat.view(batch_size, key_len, H_kv, D).transpose(1, 2)
            if H_kv != H:
                if self.gqa_implementation != "repeat":
                    raise ValueError("Attention capture with GQA currently requires gqa_implementation='repeat'.")
                repeat_factor = H // H_kv
                k = k.repeat_interleave(repeat_factor, dim=1)
                v = v.repeat_interleave(repeat_factor, dim=1)
            scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (D ** -0.5)
            scores = scores.masked_fill(~attn_mask, torch.finfo(scores.dtype).min)
            attn_probs = torch.softmax(scores, dim=-1).to(dtype=v.dtype)
            out = torch.matmul(attn_probs, v)
            out = out.transpose(1, 2).reshape(batch_size, query_len, H * D)
            return out, attn_probs

        def _sdpa_context():
            if not self.force_flash_attention:
                return nullcontext()
            try:
                from torch.nn.attention import SDPBackend, sdpa_kernel
            except Exception as exc:  # pragma: no cover - depends on torch build
                raise RuntimeError("`force_flash_attention=True` requires torch.nn.attention.sdpa_kernel.") from exc
            return sdpa_kernel([SDPBackend.FLASH_ATTENTION])
            force_flash_context = sdpa_kernel([SDPBackend.FLASH_ATTENTION])

        def _forward(q_flat: torch.Tensor, k_flat: torch.Tensor, v_flat: torch.Tensor) -> torch.Tensor:
            q = q_flat.view(batch_size, query_len, H, D).transpose(1, 2)
            k = k_flat.view(batch_size, key_len, H_kv, D).transpose(1, 2)
            v = v_flat.view(batch_size, key_len, H_kv, D).transpose(1, 2)
            enable_gqa = False
            if H_kv != H and self.gqa_implementation == "repeat":
                repeat_factor = H // H_kv
                k = k.repeat_interleave(repeat_factor, dim=1)
                v = v.repeat_interleave(repeat_factor, dim=1)
            elif H_kv != H:
                enable_gqa = True
            with _sdpa_context():
                if enable_gqa:
                    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=True)
                else:
                    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            return out.transpose(1, 2).reshape(batch_size, query_len, H * D)

        if self.mot_checkpoint_mixed_attn and self.training:
            return torch.utils.checkpoint.checkpoint(_forward, q_cat, k_cat, v_cat, use_reentrant=False)
        return _forward(q_cat, k_cat, v_cat)


    def _flux2_flatten_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.transpose(1, 2).reshape(tensor.shape[0], tensor.shape[2], tensor.shape[1] * tensor.shape[3])

    def _flux2_video_single_io(self, block, x: torch.Tensor, pe: torch.Tensor, mod) -> dict:
        from awomo.flux2.model import apply_rope

        q, k, v, mlp, gate = block._qkv(x, mod)
        q, k = apply_rope(q, k, pe)
        return {
            "q": self._flux2_flatten_heads(q),
            "k": self._flux2_flatten_heads(k),
            "v": self._flux2_flatten_heads(v),
            "mlp": mlp,
            "gate": gate,
            "residual_x": x,
        }
