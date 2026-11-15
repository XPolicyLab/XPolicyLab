import math

import torch
import torch.nn as nn

from falcon.distributed import random as falcon_random
from falcon.distributed.parallel import get_cp_rank, get_cp_size


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Generates sinusoidal timestep embeddings.

    Args:
        timesteps (torch.Tensor): A 1-D tensor of shape (N,) containing indices, one per batch element.
                                  These may be fractional.
        dim (int): The dimension of the output embeddings.
        max_period (int, optional): Controls the minimum frequency of the embeddings. Default is 10000.

    Returns:
        torch.Tensor: A tensor of shape (N, dim) containing the positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepEmbed(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256, linear2_hidden_size: int = None):
        super().__init__()
        self.linear1 = nn.Linear(frequency_embedding_size, hidden_size, bias=True)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(hidden_size, linear2_hidden_size or hidden_size, bias=True)
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = timestep_embedding(t * 1000, dim=self.frequency_embedding_size)
        t_freq = t_freq.to(t)
        t_emb = self.linear2(self.act(self.linear1(t_freq)))
        return t_emb

    def init_weights(self):
        nn.init.normal_(self.linear1.weight, std=0.02)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.normal_(self.linear2.weight, std=0.02)
        nn.init.constant_(self.linear2.bias, 0)


class CaptionEmbed(nn.Module):
    def __init__(
        self,
        caption_length: int,
        text_dim: int,
        hidden_size: int,
        uncond_prob: float = 0,
    ):
        super().__init__()
        # make sure y_embedding same between all devices
        self.caption_length = caption_length
        self.text_dim = text_dim
        self.register_buffer(
            "y_embedding",
            torch.randn(caption_length, text_dim) / text_dim**0.5,
            persistent=True,
        )
        self.uncond_prob = uncond_prob
        self.y_proj = nn.Linear(text_dim, hidden_size, bias=True)
        self.cp_rank = get_cp_rank()
        self.cp_size = get_cp_size()

    def token_drop(
        self,
        caption_embed: torch.Tensor,
        cuCs: torch.Tensor,
        Cs_unpadded: torch.Tensor,
        force_drop_ids=None,
    ):
        """Replace dropped samples' tokens with unconditional embedding (interleaving CP).

        Args:
            caption_embed: this rank's slice, shape ``(L, text_dim)``.
            cuCs: cumulative padded lengths split by CP, shape ``(mbs + 1,)``.
            Cs_unpadded: per-sample unpadded lengths split by CP, shape ``(mbs,)``.
            force_drop_ids: ``True`` to force-drop all, ``None`` for random.
        """
        with falcon_random.get_rng_states_tracker().fork():  # type: ignore
            mbs = cuCs.shape[0] - 1
            if force_drop_ids is None:
                drop_ids = torch.rand(mbs, device=caption_embed.device) < self.uncond_prob
            else:
                drop_ids = [force_drop_ids == 1 for i in range(mbs)]

            for i in range(mbs):
                start, end = cuCs[i], cuCs[i] + Cs_unpadded[i]
                if drop_ids[i]:
                    caption_embed[start:end].copy_(
                        self.y_embedding[self.cp_rank * (end - start) : (self.cp_rank + 1) * (end - start)]
                    )
            return caption_embed

    def forward(
        self,
        caption_embed: torch.Tensor,
        cuCs: torch.Tensor,
        Cs_unpadded: torch.Tensor,
        force_drop_ids: bool | None = None,
    ) -> torch.Tensor:
        """

        Args:
            caption_embed (torch.Tensor): Caption embedding
            cuCs (torch.Tensor): Cumulative sequence lengths
            Cs_unpadded (torch.Tensor): Unpadded sequence lengths
            force_drop_ids (bool | None): Force drop ids

        Returns:
            caption_embed (torch.Tensor): Caption embedding
        """
        if self.training and self.uncond_prob > 0 or (force_drop_ids is not None):
            caption_embed = self.token_drop(caption_embed, cuCs, Cs_unpadded, force_drop_ids)
        caption_embed = self.y_proj(caption_embed)
        return caption_embed

    def init_weights(self):
        nn.init.normal_(self.y_proj.weight, std=0.02)
        nn.init.constant_(self.y_proj.bias, 0)

        temp_generator = torch.Generator(
            device=(self.y_embedding.device if self.y_embedding.device.type != "meta" else "cpu")
        )
        temp_generator.manual_seed(42)
        self.y_embedding.copy_(
            torch.randn(
                self.caption_length,
                self.text_dim,
                device=self.y_embedding.device,
                dtype=self.y_embedding.dtype,
                generator=temp_generator,
            )
            / self.text_dim**0.5
        )


class FlatCpCaptionEmbed(CaptionEmbed):
    """CaptionEmbed variant for flat context-parallel models.

    Under flat CP, each rank holds a contiguous slice [cp_rank*L, (cp_rank+1)*L)
    of the global caption sequence. token_drop must receive GLOBAL cuCs_padded and
    Cs, then compute the overlap with this rank's local window.
    """

    def token_drop(
        self,
        caption_embed: torch.Tensor,
        cuCs: torch.Tensor,
        Cs_unpadded: torch.Tensor,
        force_drop_ids=None,
    ):
        """Replace dropped samples' tokens with unconditional embedding (flat CP).

        Args:
            caption_embed: this rank's local slice, shape ``(L, text_dim)``.
            cuCs: GLOBAL cumulative padded lengths, shape ``(mbs + 1,)``.
                Must be ``mb_meta["cuCs_padded"]``.
            Cs_unpadded: GLOBAL per-sample unpadded lengths, shape ``(mbs,)``.
                Must be ``mb_meta["Cs"]``.
            force_drop_ids: ``True`` to force-drop all, ``None`` for random.
        """
        L = caption_embed.shape[0]
        base = self.cp_rank * L

        with falcon_random.get_rng_states_tracker().fork():  # type: ignore
            mbs = cuCs.shape[0] - 1
            if force_drop_ids is None:
                drop_ids = (torch.rand(mbs, device=caption_embed.device) < self.uncond_prob).tolist()
            else:
                drop_ids = [force_drop_ids == 1 for _ in range(mbs)]

            cuCs_list = cuCs.tolist()
            Cs_list = Cs_unpadded.tolist()
            for i in range(mbs):
                if not drop_ids[i]:
                    continue
                g_start = cuCs_list[i]
                g_end = g_start + Cs_list[i]
                ov_start = max(g_start, base)
                ov_end = min(g_end, base + L)
                if ov_end <= ov_start:
                    continue
                local_start = ov_start - base
                ov_len = ov_end - ov_start
                in_sample_offset = ov_start - g_start
                caption_embed[local_start : local_start + ov_len].copy_(
                    self.y_embedding[in_sample_offset : in_sample_offset + ov_len]
                )
            return caption_embed
