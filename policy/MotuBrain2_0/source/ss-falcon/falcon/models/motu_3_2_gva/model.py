import copy
import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import RMSNorm

from falcon.config import TrainerConfig
from falcon.distributed import get_cp_group, get_cp_rank, get_cp_size
from falcon.modules import FlatCpCaptionEmbed, TimestepEmbed
from falcon.ops import gather_from_context_parallel_region
from falcon.utils import get_accumulate_tensor
from soma.inference.models.operators import (
    build_rope_cache,
    ensure_bf16,
    ensure_fp32,
)

from .layers import FinalLayer, MotuLayer
from .model_config import GVAConfig

logger = logging.getLogger(__name__)


class MotuGVAModel(nn.Module):
    def __init__(self, cfg: TrainerConfig):
        super().__init__()
        # A single adaLN stream modulates video and audio alike, so audio can only ever be
        # conditioned on the video timestep; noising it at another level trains silently wrong.
        assert not cfg.noise_scheduler.get(
            "decouple_audio_t", False
        ), "noise_scheduler.decouple_audio_t is not supported by MotuGVAModel (single shared adaLN stream)"
        self.motu_cfg = GVAConfig(cfg)
        self.patch_size = self.motu_cfg.patch_size
        self.v_in_channels = self.motu_cfg.v_in_channels
        self.a_in_channels = self.motu_cfg.a_in_channels
        self.a_embed_in_channels = self.motu_cfg.a_embed_in_channels
        self.frames = self.motu_cfg.frames

        hidden_size = self.motu_cfg.hidden_size
        self.patch_embed = nn.Sequential(
            nn.Linear(self.patch_size**2 * self.v_in_channels, hidden_size, bias=True),
            RMSNorm(hidden_size, elementwise_affine=True, eps=1e-6),
        )
        self.a_embed = nn.Sequential(
            nn.Linear(self.a_embed_in_channels, hidden_size, bias=True),
            RMSNorm(hidden_size, elementwise_affine=True, eps=1e-6),
        )
        self.c_embed = FlatCpCaptionEmbed(
            self.motu_cfg.caption_length,
            self.motu_cfg.text_dim,
            hidden_size,
            uncond_prob=self.motu_cfg.p_uncond,
        )
        self.timestep_embed = TimestepEmbed(hidden_size)

        self.vidu_blocks = nn.ModuleList()
        for i in range(self.motu_cfg.num_layers):
            self.vidu_blocks.append(MotuLayer(i + 1, self.motu_cfg))

        self.final_layer = FinalLayer(
            hidden_size=hidden_size,
            out_channels=self.v_in_channels * self.patch_size**2,
            audio_in_channels=self.a_in_channels,
        )
        self.initial_weights()
        self.cp_size = get_cp_size()
        self.cp_rank = get_cp_rank()

        # TODO: fix this
        self.loss_weight_video = cfg.noise_scheduler.get("loss_weight_video", 2.0)
        self.loss_weight_video_audio = cfg.noise_scheduler.get("loss_weight_video_audio", 3.0)
        self.loss_weight_audio = cfg.noise_scheduler.get("loss_weight_audio", 0.5)
        assert (
            0 < self.loss_weight_audio < 1
        ), f"loss_weight_audio must be between 0 and 1, but got {self.loss_weight_audio}"
        self.loss_weight_image = cfg.noise_scheduler.get("loss_weight_image", 1.0)

    def initial_weights(self):
        """
        * Learned from TorchTitan.
        [Note: On ``init_weights`` vs. ``reset_parameters``]
        Modules may define ``reset_parameters`` to initialize parameter values.
        ``reset_parameters`` is meant to only initialize directly owned
        parameters/buffers, not those of their child modules, and it can be
        used to give the initial values for these tensors.
        Separately, users may want custom initialization for their modules,
        different from that in ``reset_parameters``. For this, we define
        ``init_weights``. We only call it in the constructor of this
        the root module to avoid reinitializing tensors.
        """
        nn.init.xavier_normal_(self.patch_embed[0].weight)
        nn.init.constant_(self.patch_embed[0].bias, 0)
        self.patch_embed[1].reset_parameters()
        nn.init.xavier_normal_(self.a_embed[0].weight)
        nn.init.constant_(self.a_embed[0].bias, 0)
        self.a_embed[1].reset_parameters()
        self.c_embed.init_weights()
        self.timestep_embed.init_weights()

        for block in self.vidu_blocks:
            block.init_weights()

        self.final_layer.init_weights()

    def prepare_mb_meta(self, **kwargs):
        mb_meta = copy.deepcopy(kwargs)
        Vs, As, Cs = kwargs["Vs"], kwargs["As"], kwargs["Cs"]
        As_padded, Cs_padded = kwargs["As_padded"], kwargs["Cs_padded"]

        # joint-modal metadata
        VACs = torch.tensor([a + b + c for a, b, c in zip(Vs, As, Cs)], dtype=torch.int32)
        VACs_padded = torch.tensor([a + b + c for a, b, c in zip(Vs, As_padded, Cs_padded)], dtype=torch.int32)
        cuVACs, cuVACs_padded = list(map(get_accumulate_tensor, [VACs, VACs_padded]))
        # joint-modal metadata
        mb_meta.update(
            dict(
                VACs=VACs,
                maxVAC=VACs.max().item(),
                cuVACs=cuVACs,
                totalVAC=cuVACs[-1].item(),
                VACs_padded=VACs_padded,
                cuVACs_padded=cuVACs_padded,
                maxVAC_padded=VACs_padded.max().item(),
                totalVAC_padded=cuVACs_padded[-1].item(),
            )
        )

        device = torch.cuda.current_device()
        for k, v in mb_meta.items():
            if isinstance(v, torch.Tensor):
                mb_meta[k] = v.to(device, non_blocking=True)
        return mb_meta

    @classmethod
    def preprocess_context_parallel(cls, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int):
        """Flat contiguous CP split: rank r gets global positions [r*L, (r+1)*L).

        ``cu_seqlens`` / ``max_seqlen`` are kept for signature compatibility but
        are unused under flat CP. Only (L, D>1) features are ever passed here
        (v/a/caption); the old per-sample mask branch was dead code and removed.
        """
        cp_rank, cp_size = get_cp_rank(), get_cp_size()
        if cp_size > 1:
            total_local = hidden_states.shape[0] // cp_size
            hidden_states = hidden_states[cp_rank * total_local : (cp_rank + 1) * total_local].contiguous()
        return hidden_states

    @classmethod
    def postprocess_context_parallel(
        cls, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int
    ) -> torch.Tensor:
        """Gather flat CP shards — result is already in global order (no blocking)."""
        cp_size = get_cp_size()
        if cp_size > 1:
            hidden_states = gather_from_context_parallel_region(hidden_states, group=get_cp_group())  # type: ignore
        return hidden_states

    def forward(
        self, v: torch.Tensor, a: torch.Tensor, caption: torch.Tensor, t: torch.Tensor, force_drop_ids=None, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            v (torch.Tensor): Video latent.
            a (torch.Tensor): Audio latent.
            caption (torch.Tensor): Caption embeddings.
            t (torch.Tensor): (micro_batch_size)
            force_drop_ids : default is None.
            kwargs: mb_meta

        The design logic for context parallel is that the input and output is full sequence. The model handles input sharding.
        The output will have half the sequence with grad and the other half without grad.
        The loss is calculated on the full sequence as no cp.
        The gradient scaling is handled in base trainer.
        """
        mb_meta = self.prepare_mb_meta(**kwargs)

        # Build RoPE caches once per forward (reused by every layer's attention).
        # axes_shapes uses PER-SAMPLE extents (Fs/Hs/Ws, As_padded), NOT the
        # cumulative cuFs/cuHs/cuWs the old QKFHWRoPE consumed; cu_tokens stays
        # GLOBAL (cuVs / cuAs_padded). Caches are aligned to this rank's flat
        # local window via cp_rank/total_local.
        head_dim = self.motu_cfg.hidden_size_per_attention_head
        v_local = mb_meta["cuVs"][-1].item() // self.cp_size
        a_local = mb_meta["cuAs_padded"][-1].item() // self.cp_size
        video_axes_offsets = [
            mb_meta.get(k, None) for k in ("f_seqlen_offsets", "h_seqlen_offsets", "w_seqlen_offsets")
        ]
        audio_axes_offsets = [mb_meta.get("a_seqlen_offsets", None)]
        mb_meta["video_rope_cache"] = build_rope_cache(
            v_local,
            head_dim,
            mb_meta["cuVs"],
            [mb_meta["Fs"], mb_meta["Hs"], mb_meta["Ws"]],
            self.motu_cfg.video_rope_axes,
            axes_seqlen_offsets=video_axes_offsets,
            cp_rank=self.cp_rank,
            total_local=v_local,
        )
        mb_meta["audio_rope_cache"] = build_rope_cache(
            a_local,
            head_dim,
            mb_meta["cuAs_padded"],
            [mb_meta["As_padded"]],
            self.motu_cfg.audio_rope_axes,
            axes_seqlen_offsets=audio_axes_offsets,
            cp_rank=self.cp_rank,
            total_local=a_local,
        )

        x = self.preprocess_context_parallel(v, mb_meta["cuVs"], mb_meta["maxV"])
        a = self.preprocess_context_parallel(a, mb_meta["cuAs_padded"], mb_meta["maxA_padded"])
        caption = self.preprocess_context_parallel(caption, mb_meta["cuCs_padded"], mb_meta["maxC_padded"])
        # Ensure inputs is in bf16
        x, a, caption, t = ensure_bf16((x, a, caption, t))

        # timestep embed
        adaln = self.timestep_embed(t)
        # caption embed (flat CP: pass GLOBAL cuCs_padded + UNPADDED Cs)
        caption = self.c_embed(caption, mb_meta["cuCs_padded"], mb_meta["Cs"], force_drop_ids)
        # patch embed
        hidden_states = torch.utils.checkpoint.checkpoint(self.patch_embed, x, use_reentrant=False)
        # audio embed
        audio_hidden_states = torch.utils.checkpoint.checkpoint(self.a_embed, a, use_reentrant=False)

        # apply the transformer blocks
        for block in self.vidu_blocks:
            hidden_states, audio_hidden_states, caption = block(
                hidden_states, audio_hidden_states, caption, adaln, mb_meta
            )

        args = (hidden_states, audio_hidden_states, adaln, mb_meta)
        hidden_states, audio_hidden_states = torch.utils.checkpoint.checkpoint(
            self.final_layer, *args, use_reentrant=False
        )

        if self.cp_size > 1:
            # gather hidden_states to full tensor
            hidden_states = self.postprocess_context_parallel(hidden_states, mb_meta["cuVs"], mb_meta["maxV"])
            audio_hidden_states = self.postprocess_context_parallel(
                audio_hidden_states, mb_meta["cuAs_padded"], mb_meta["maxA_padded"]
            )

        # Ensure outputs is in fp32
        hidden_states = ensure_fp32(hidden_states)
        audio_hidden_states = ensure_fp32(audio_hidden_states)
        return hidden_states, audio_hidden_states
