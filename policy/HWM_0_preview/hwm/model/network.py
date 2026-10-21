"""HWM0Preview model components and checkpoint loading.

Transformer arithmetic follows the StarVLA/FastWAM inference implementation
"""

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ..config import ModelConfig


def video_prompt(instruction: str, current_subtask: str | None, template: str) -> str:
    """The global task plus the planner's effective subtask, without sampling."""
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Video conditioning requires a nonempty task instruction")
    task = instruction.strip()
    if not isinstance(current_subtask, str) or current_subtask.strip() in ("", "None."):
        return task
    return template.format(task=task, subtask=current_subtask.strip())


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return value.to(x.dtype) * self.weight


class Attention(nn.Module):
    """Q/K normalization precedes the configured head splitting."""

    def __init__(self, width: int, context_width: int, cfg: ModelConfig):
        super().__init__()
        self.heads, self.head_dim = cfg.m["heads"], cfg.m["head_dim"]
        inner = self.heads * self.head_dim
        self.q = nn.Linear(width, inner)
        self.k = nn.Linear(context_width, inner)
        self.v = nn.Linear(context_width, inner)
        self.o = nn.Linear(inner, width)
        self.norm_q = RMSNorm(inner, cfg.m["norm_eps"])
        self.norm_k = RMSNorm(inner, cfg.m["norm_eps"])

    def qkv(self, x, context):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(context))
        v = self.v(context)
        return q, k, v

    def prepare_kv(self, context):
        """Project a fixed cross-attention condition once per sampling request."""
        k = self.norm_k(self.k(context))
        v = self.v(context)
        return tuple(
            t.unflatten(-1, (self.heads, self.head_dim)).transpose(1, 2) for t in (k, v)
        )

    def forward(self, x, context, mask=None, *, kv=None):
        if kv is None:
            q, k, v = self.qkv(x, context)
            q, k, v = (
                t.unflatten(-1, (self.heads, self.head_dim)).transpose(1, 2)
                for t in (q, k, v)
            )
        else:
            q = (
                self.norm_q(self.q(x))
                .unflatten(-1, (self.heads, self.head_dim))
                .transpose(1, 2)
            )
            k, v = kv
        if mask is None:
            mask = torch.ones(
                (k.shape[0], k.shape[2]), device=k.device, dtype=torch.bool
            )
        # Keep the reference's explicit [B,1,Sq,Sk] mask even when all keys
        # are visible. Replacing it with None changes CUDA SDPA selection.
        mask = mask[:, None, None, :].expand(-1, 1, x.shape[1], -1)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(attended.transpose(1, 2).flatten(2))


class DiTBlock(nn.Module):
    """One physical-expert layer, split at the joint-attention boundary."""

    def __init__(self, width: int, ffn_width: int, cfg: ModelConfig):
        super().__init__()
        self.self_attn = Attention(width, width, cfg)
        self.cross_attn = Attention(width, width, cfg)
        self.norm1 = nn.LayerNorm(
            width, eps=cfg.m["norm_eps"], elementwise_affine=False
        )
        self.norm2 = nn.LayerNorm(
            width, eps=cfg.m["norm_eps"], elementwise_affine=False
        )
        self.norm3 = nn.LayerNorm(width, eps=cfg.m["norm_eps"])
        self.ffn = nn.Sequential(
            nn.Linear(width, ffn_width),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_width, width),
        )
        self.modulation = nn.Parameter(torch.empty(1, 6, width))

    def prepare_attention(self, x, time_modulation):
        modulation = self.modulation + time_modulation
        pieces = modulation.chunk(6, dim=-2)
        if modulation.ndim == 4:
            pieces = tuple(piece.squeeze(-2) for piece in pieces)
        shift, scale, gate, ffn_shift, ffn_scale, ffn_gate = pieces
        normalized = self.norm1(x) * (1 + scale) + shift
        return self.self_attn.qkv(normalized, normalized), (
            gate,
            ffn_shift,
            ffn_scale,
            ffn_gate,
        )

    def finish_attention(
        self,
        x,
        attended,
        modulation,
        context,
        mask=None,
        visual=None,
        *,
        context_kv=None,
        visual_kv=None,
    ):
        gate, shift, scale, ffn_gate = modulation
        x = x + gate * self.self_attn.o(attended)
        cross = self.cross_attn(self.norm3(x), context, mask, kv=context_kv)
        if visual is None:
            x = x + cross
        else:
            norm, attention, world_queries = visual
            # Both cross attentions read the same pre-residual x.
            visual_cross = attention(norm(x), world_queries, kv=visual_kv)
            x = x + cross + visual_cross
        return x + ffn_gate * self.ffn(self.norm2(x) * (1 + scale) + shift)


class VideoHead(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        width = cfg.m["video_width"]
        self.norm = nn.LayerNorm(width, eps=cfg.m["norm_eps"], elementwise_affine=False)
        self.head = nn.Linear(width, cfg.channels * math.prod(cfg.m["patch_size"]))
        self.modulation = nn.Parameter(torch.empty(1, 2, width))

    def forward(self, x, time_embedding):
        shift, scale = (self.modulation[None] + time_embedding[:, :, None]).unbind(2)
        return self.head(self.norm(x) * (1 + scale) + shift)


class HistoryProjection(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        for name, kernel in zip(
            ("proj", "proj_2x", "proj_4x"), cfg.m["history_kernels"], strict=True
        ):
            setattr(
                self,
                name,
                nn.Conv3d(cfg.channels, cfg.m["video_width"], kernel, stride=kernel),
            )


class VideoDiT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        width, layers = cfg.m["video_width"], cfg.m["layers"]
        self.patch_embedding = nn.Conv3d(
            cfg.channels, width, cfg.m["patch_size"], stride=cfg.m["patch_size"]
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(cfg.backbones["text"]["d_model"], width),
            nn.GELU(approximate="tanh"),
            nn.Linear(width, width),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(cfg.m["time_width"], width), nn.SiLU(), nn.Linear(width, width)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))
        self.blocks = nn.ModuleList(
            DiTBlock(width, cfg.m["video_ffn_width"], cfg) for _ in range(layers)
        )
        self.vlm_norms = nn.ModuleList(
            nn.LayerNorm(width, eps=cfg.m["norm_eps"]) for _ in range(layers)
        )
        self.vlm_cross_attentions = nn.ModuleList(
            Attention(width, cfg.planner_width, cfg) for _ in range(layers)
        )
        self.head = VideoHead(cfg)
        self.history_packing = HistoryProjection(cfg)


class ActionDiT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        width = cfg.m["action_width"]
        self.action_encoder = nn.Linear(cfg.m["action_dim"], width)
        self.text_embedding = nn.Sequential(
            nn.Linear(cfg.planner_width, width),
            nn.GELU(approximate="tanh"),
            nn.Linear(width, width),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(cfg.m["time_width"], width), nn.SiLU(), nn.Linear(width, width)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))
        self.blocks = nn.ModuleList(
            DiTBlock(width, cfg.m["action_ffn_width"], cfg)
            for _ in range(cfg.m["layers"])
        )
        self.head = nn.Linear(width, cfg.m["action_dim"])


class HWM0Preview(nn.Module):
    """VLM planner, video/action transformers, text encoder and video VAE."""

    def __init__(self, cfg: ModelConfig, *, attention_backend="flash_attention_2"):
        super().__init__()
        from diffusers import AutoencoderKLWan
        from transformers import (
            Qwen3_5Config,
            Qwen3_5ForConditionalGeneration,
            UMT5Config,
            UMT5EncoderModel,
        )

        self.config = cfg
        qwen_config = Qwen3_5Config(**cfg.backbones["vlm"])
        qwen_config._attn_implementation = attention_backend
        self.vlm = Qwen3_5ForConditionalGeneration(qwen_config)
        self.text = UMT5EncoderModel(UMT5Config(**cfg.backbones["text"]))
        self.vae = AutoencoderKLWan(**cfg.backbones["vae"])
        self.register_buffer("vae_mean", torch.empty(1, cfg.channels, 1, 1, 1))
        self.register_buffer("vae_inv_std", torch.empty(1, cfg.channels, 1, 1, 1))
        self.world_queries = nn.Parameter(
            torch.empty(1, cfg.m["world_queries"], cfg.planner_width)
        )
        self.action_queries = nn.Parameter(
            torch.empty(1, cfg.m["action_horizon"], cfg.planner_width)
        )
        self.state_projection = nn.Linear(cfg.m["state_dim"], cfg.planner_width)
        self.video = VideoDiT(cfg)
        self.action = ActionDiT(cfg)

    @torch.no_grad()
    def encode_video_text(
        self,
        tokenizer,
        instruction: str | list[str],
        current_subtask: str | list[str | None] | None,
    ):
        device = self.world_queries.device
        prompts = (
            [
                video_prompt(
                    instruction,
                    current_subtask,
                    self.config.conditioning["video_prompt_template"],
                )
            ]
            if isinstance(instruction, str)
            else [
                video_prompt(
                    task, subtask, self.config.conditioning["video_prompt_template"]
                )
                for task, subtask in zip(instruction, current_subtask, strict=True)
            ]
        )
        tokens = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.config.conditioning["t5_max_length"],
            return_tensors="pt",
            add_special_tokens=True,
        ).to(device)
        mask = tokens.attention_mask.bool()
        context = self.text(
            input_ids=tokens.input_ids, attention_mask=mask
        ).last_hidden_state
        # Wan zeros the UMT5 padding embeddings, but intentionally exposes all
        # padded slots to cross attention (whose projections have learned biases).
        context = context.masked_fill(~mask[..., None], 0)
        return context, torch.ones_like(mask)

    @torch.no_grad()
    def encode_video(self, pixels: torch.Tensor) -> torch.Tensor:
        """Normalized BF16 pixels [B,3,T,H,W] -> scaled deterministic VAE mode."""
        latent = self.vae.encode(pixels).latent_dist.mode()
        scaled = (latent.float() - self.vae_mean.float()) * self.vae_inv_std.float()
        return scaled.to(latent.dtype)

    def validate_checkpoint(
        self, state: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        expected = self.state_dict()
        mapped = {}
        for source, tensor in state.items():
            target = source
            if target in mapped:
                raise ValueError(f"Duplicate destination for {source}: {target}")
            if target not in expected:
                raise ValueError(f"Unexpected tensor: {source} -> {target}")
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Checkpoint entry {source} is not a tensor")
            if tensor.shape != expected[target].shape:
                raise ValueError(
                    f"Shape mismatch for {source}: {tuple(tensor.shape)} != {tuple(expected[target].shape)}"
                )
            if tensor.dtype != torch.bfloat16:
                raise ValueError(f"Expected BF16 tensor {source}, got {tensor.dtype}")
            mapped[target] = tensor
        missing = sorted(expected.keys() - mapped.keys())
        if missing:
            raise ValueError(f"Missing checkpoint tensors: {missing}")
        return mapped

    @classmethod
    def load(
        cls,
        checkpoint: str | Path,
        *,
        config: ModelConfig | None = None,
        device="cuda",
        attention_backend="flash_attention_2",
    ):
        from accelerate import init_empty_weights

        checkpoint = Path(checkpoint)
        config = config or ModelConfig.load(checkpoint.parent / "config.yaml")
        with init_empty_weights(include_buffers=False):
            model = cls(config, attention_backend=attention_backend)
        state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
        model.load_state_dict(
            model.validate_checkpoint(state), strict=True, assign=True
        )
        if any(t.is_meta for t in model.state_dict().values()):
            raise RuntimeError("Checkpoint load left unresolved meta tensors")
        # Parameters were assigned as BF16. Do not cast the whole module: Qwen's
        # nonpersistent rotary-frequency buffers must retain their FP32 values.
        return model.to(device=device).requires_grad_(False).eval()
