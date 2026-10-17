# EWAM - Modular Architecture (WAN Video Model + Action Expert + VLM Direct MoT)
# Three-modal UniDiffuser: Video Model (WAN) + Action Expert + VLM (Direct MoT)
# VLM 28 layers -> WAN 30 layers (Layer 26, 27 reused for WAN Layer 28, 29)

import sys
import os
import json
import time
import torch
import logging
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

# The `wan` package (Wan2.2 official code) lives at the project root and
# is importable via PYTHONPATH (see the launch scripts in scripts/).

from utils.common import get_t_distribution
from wan.modules.model_mask import sinusoidal_embedding_1d
from transformers import Qwen3VLForConditionalGeneration, AutoConfig
from .wan_model_mask import WanVideoModel
from .action_expert import ActionExpert, ActionExpertConfig
from .qwen3_module_wan import Qwen3VLWanModule, Qwen3VLWanConfig
# Add Flow-Matching schedulers
from wan.utils.fm import FlowMatchScheduler
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

logger = logging.getLogger(__name__)


@dataclass
class EwamConfig:
    """Configuration for EwamDirect with asymmetric attention mask."""
    # Video model settings (WAN 5B)
    wan_checkpoint_path: str = ""
    vae_path: str = ""
    wan_config_path: str = ""
    video_precision: str = "bfloat16"

    # VLM settings (Qwen3-VL-2B, trainable)
    vlm_checkpoint_path: str = ""

    # Qwen3-VL Expert settings (per-layer QKV projections)
    vlm_dim: int = 2048              # Qwen3-VL-2B hidden size
    qwen3_expert_head_dim: int = 128              # Head dimension (for WAN 5B)
    qwen3_expert_num_heads: int = 24              # Number of heads (for WAN 5B)
    qwen3_expert_num_layers: int = 30             # Number of layers (matches WAN)
    qwen3_expert_norm_eps: float = 1e-5           # Layer norm epsilon

    # Action expert settings
    num_layers: int = 30
    action_state_dim: int = 14
    action_dim: int = 14
    action_expert_dim: int = 1024           # Configurable hidden dimension
    action_expert_ffn_dim_multiplier: int = 4  # FFN dimension multiplier
    action_expert_norm_eps: float = 1e-6    # Layer norm epsilon for Action Expert

    # Sampling settings
    global_downsample_rate: int = 3     # Global downsampling rate
    video_action_freq_ratio: int = 4    # Video:Action frequency ratio
    num_video_frames: int = 4           # Number of video frames to predict

    # Video dimensions
    video_height: int = 512             # Input video height
    video_width: int = 512              # Input video width

    # Training settings
    batch_size: int = 8

    # Training mode
    training_mode: str = 'finetune'  # 'pretrain' or 'finetune'

    # Loss weights
    video_loss_weight: float = 1.0
    action_loss_weight: float = 1.0

    # Loss variant: false -> plain uniform-mean-MSE flow matching (discrete uniform
    # timestep sampling); true -> Gaussian timestep weighting + shifted
    # timestep sampling. Same state dict / inference path either way.
    loss_reweight: bool = False

    # Control whether to load pretrained WAN/VLM backbones
    load_pretrained_backbones: Optional[bool] = None

    # VLM frozen setting
    vlm_frozen: bool = False  # VLM is trainable by default

    def __post_init__(self):
        """Calculate derived parameters."""
        # Action chunk size is determined by global downsample rate and frequency ratio
        self.action_chunk_size = self.num_video_frames * self.video_action_freq_ratio


class VideoModule(nn.Module):
    """Video processing module - handles WAN + T5 operations."""

    def __init__(self, video_model, dtype, device, grid_sizes):
        super().__init__()
        self.video_model = video_model
        self.dtype = dtype
        self.device = device
        self.grid_sizes = grid_sizes

    def prepare_input(self, noisy_video_latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare video tokens from pre-processed noisy latent."""
        # Through patch_embedding: 48 -> 3072 channels
        video_patched = self.video_model.wan_model.patch_embedding(noisy_video_latent)
        # Flatten and convert to tokens
        video_features = video_patched.flatten(2).transpose(1, 2)
        return video_features

    def preprocess_t5_embeddings(self, language_embeddings) -> torch.Tensor:
        """Pre-process T5 embeddings once for all layers."""
        text_len = self.video_model.wan_model.text_len  # 512
        if isinstance(language_embeddings, list):
            padded_embeddings = []
            for emb in language_embeddings:
                if emb.shape[0] <= text_len:
                    padded = torch.cat([emb, emb.new_zeros(text_len - emb.shape[0], emb.shape[1])])
                else:
                    padded = emb[:text_len]
                padded_embeddings.append(padded)
            t5_context_raw = torch.stack(padded_embeddings, dim=0)
        else:
            t5_context_raw = language_embeddings
        # Convert via text_embedding layer (4096 -> 3072)
        t5_context = self.video_model.wan_model.text_embedding(t5_context_raw)
        return t5_context

    def get_time_embedding(self, t_video: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get WAN's time embedding using WAN's own weights."""
        if t_video.dim() == 1:
            t_video = t_video.unsqueeze(1).expand(t_video.size(0), seq_len)

        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t_video.size(0)
            t_flat = t_video.flatten()

            t_emb = self.video_model.wan_model.time_embedding(
                sinusoidal_embedding_1d(self.video_model.wan_model.freq_dim, t_flat).unflatten(0, (bt, seq_len)).float()
            )
            t_emb_proj = self.video_model.wan_model.time_projection(t_emb).unflatten(2, (6, 3072))
            assert t_emb.dtype == torch.float32 and t_emb_proj.dtype == torch.float32

        return t_emb, t_emb_proj

    def process_cross_attention(self, video_tokens: torch.Tensor, video_adaln_params: torch.Tensor,
                               layer_idx: int, processed_t5_context: torch.Tensor) -> torch.Tensor:
        """Process WAN cross attention with pre-processed T5 context."""
        wan_layer = self.video_model.wan_model.blocks[layer_idx]
        ca = wan_layer.cross_attn
        ctx = processed_t5_context
        context_lens = None

        # K/V cache: `processed_t5_context` is computed once before the denoise loop and passed
        # unchanged to all 30 layers x 10 steps. K = norm_k(k(ctx)) and V = v(ctx) are pure
        # functions of ctx + per-layer weights with NO timestep/AdaLN dependence (the
        # video_adaln_params arg is unused on the K/V path), so they are bit-exact identical
        # across all 300 calls. Cache them per-layer so the k/v projection runs once per layer
        # (step 0) instead of 10x. Q (from video_tokens) still recomputes every step.
        # Keyed by the context tensor's identity: a new request builds a new ctx object ->
        # automatic cache rebuild; the same ctx object across steps -> reuse.
        cache = getattr(self, "_cross_kv_cache", None)
        if cache is None or cache.get("_ctx") is not ctx:
            cache = {"_ctx": ctx}
            self._cross_kv_cache = cache
        kv = cache.get(layer_idx)
        if kv is None:
            b = ctx.size(0)
            n = ca.num_heads
            d = ca.head_dim
            k = ca.norm_k(ca.k(ctx)).view(b, -1, n, d)
            v = ca.v(ctx).view(b, -1, n, d)
            kv = (k, v)
            cache[layer_idx] = kv

        cross_out = ca(wan_layer.norm3(video_tokens), ctx, context_lens, cached_kv=kv)
        return video_tokens + cross_out

    def compute_adaln_modulation(self, video_adaln_params: torch.Tensor, layer_idx: int) -> tuple:
        """Compute AdaLN modulation parameters for WAN (6 components)."""
        wan_layer = self.video_model.wan_model.blocks[layer_idx]
        with torch.amp.autocast('cuda', dtype=torch.float32):
            modulation = (
                wan_layer.modulation.unsqueeze(0)
                + video_adaln_params
            ).chunk(6, dim=2)
        return modulation

    def process_ffn(self, video_tokens: torch.Tensor, video_adaln_modulation: tuple, layer_idx: int) -> torch.Tensor:
        """Process WAN FFN with proper AdaLN modulation."""
        wan_layer = self.video_model.wan_model.blocks[layer_idx]

        # AdaLN params
        v_mod = video_adaln_modulation

        # WAN FFN with AdaLN (params 3,4,5 for FFN: α3, β3, γ3)
        ffn_input = wan_layer.norm2(video_tokens).float() * (1 + v_mod[4].squeeze(2)) + v_mod[3].squeeze(2)
        ffn_out = wan_layer.ffn(ffn_input)

        with torch.amp.autocast('cuda', dtype=torch.float32):
            return video_tokens + ffn_out * v_mod[5].squeeze(2)

    def apply_output_head(self, video_tokens: torch.Tensor, video_time_emb: torch.Tensor) -> torch.Tensor:
        """Apply WAN's head + unpatchify for final video output."""
        x = self.video_model.wan_model.head(video_tokens, video_time_emb)
        x = self.video_model.wan_model.unpatchify(x, self.grid_sizes)
        return torch.stack([u.float() for u in x], dim=0)

    def process_joint_attention(
        self,
        video_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        vlm_tokens: torch.Tensor,
        video_adaln_modulation: tuple,
        action_adaln_modulation: tuple,
        layer_idx: int,
        action_block: nn.Module,
        vlm_block: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Trimodal joint self-attention: WAN + Action + VLM via WAN self-attn (MoT)."""
        wan_layer = self.video_model.wan_model.blocks[layer_idx]

        # AdaLN params (already computed)
        v_mod = video_adaln_modulation
        a_mod = action_adaln_modulation

        # Pre-attn normalization with AdaLN
        norm_video = wan_layer.norm1(video_tokens).float() * (1 + v_mod[1].squeeze(2)) + v_mod[0].squeeze(2)
        norm_action = action_block.norm1(action_tokens) * (1 + a_mod[1].squeeze(2)) + a_mod[0].squeeze(2)

        # Get dimensions
        B, L_v, C = norm_video.shape
        L_a = norm_action.shape[1]
        n = self.video_model.wan_model.num_heads
        d = C // n

        # Action heads for WAN space (1024 -> 24*128)
        a_qkv = torch.einsum("BTD,KNDE->KBTNE", norm_action, action_block.wan_action_qkv)
        a_q_h, a_k_h, a_v_h = a_qkv[0], a_qkv[1], a_qkv[2]
        a_q = action_block.wan_action_norm_q(a_q_h.flatten(-2)).view(B, L_a, n, d)
        a_k = action_block.wan_action_norm_k(a_k_h.flatten(-2)).view(B, L_a, n, d)
        a_v = a_v_h.view(B, L_a, n, d)

        # VLM processing
        norm_vlm = vlm_block.norm1(vlm_tokens)
        L_vlm = norm_vlm.shape[1]

        # VLM heads for WAN space (2048 -> 24*128)
        v_qkv = torch.einsum("BTD,KNDE->KBTNE", norm_vlm, vlm_block.wan_vlm_qkv)
        v_q_h, v_k_h, v_v_h = v_qkv[0], v_qkv[1], v_qkv[2]
        v_q = vlm_block.wan_vlm_norm_q(v_q_h.flatten(-2)).view(B, L_vlm, n, d)
        v_k = vlm_block.wan_vlm_norm_k(v_k_h.flatten(-2)).view(B, L_vlm, n, d)
        v_v = v_v_h.view(B, L_vlm, n, d)

        # Meta info for WAN attention
        # seq_lens and the asymmetric attn_mask depend only on (B, L_v, L_a, L_vlm), which are
        # constant across all 30 layers x 10 steps of a single request (fixed video latent shape,
        # fixed action chunk, fixed VLM seq). Cache them to avoid rebuilding a [B, L_total, L_total]
        # bool tensor + a torch.full 300 times per request.
        cache_key = (B, L_v, L_a, L_vlm)
        cached = getattr(self, "_attn_mask_cache", None)
        if cached is None or cached[0] != cache_key:
            seq_lens = torch.full((B,), L_v + L_a + L_vlm, dtype=torch.long, device=self.device)
            # ==================== Asymmetric attention mask ====================
            # Attention pattern: Action attends to all; Video / VLM only self-attend
            L_total = L_v + L_a + L_vlm
            attn_mask = torch.ones((B, L_total, L_total), dtype=torch.bool, device=self.device)
            # Video cannot attend to Action or VLM (Video self-attention only)
            attn_mask[:, :L_v, L_v:] = False
            # VLM cannot attend to Video or Action (VLM self-attention only)
            attn_mask[:, L_v + L_a:, :L_v + L_a] = False
            # Action attends to everything (already True)

            # Print mask for verification (only once, when first built)
            # DISABLED: prints large matrices on every rank's first forward;
            # uncomment to re-enable.
            # if not hasattr(self, '_mask_printed'):
            #     logger.info("=" * 60)
            #     logger.info(f"Attention mask (L_v={L_v}, L_a={L_a}, L_vlm={L_vlm}):")
            #     logger.info("Rows=Query, Cols=Key, 1=allow, 0=mask")
            #     mask_np = attn_mask[0].cpu().numpy().astype(int)
            #     logger.info(f"Video rows (first 5): {mask_np[:L_v, :5]}")
            #     logger.info(f"Video rows (Action/VLM cols): {mask_np[:L_v, L_v:L_v+5]}")
            #     logger.info(f"Action rows (Video cols, first 5): {mask_np[L_v:L_v+1, :5]}")
            #     logger.info(f"VLM rows (Video cols, first 5): {mask_np[L_v+L_a:, :5]}")
            #     logger.info("V=Video, A=Action, V=VLM")
            #     logger.info("Expected: Video→Video✅, Action→all✅, VLM→VLM✅, cross:❌")
            #     logger.info("=" * 60)
            #     self._mask_printed = True

            self._attn_mask_cache = (cache_key, seq_lens, attn_mask)
        else:
            seq_lens, attn_mask = cached[1], cached[2]

        freqs = self.video_model.wan_model.freqs
        if freqs.device != self.device:
            freqs = freqs.to(self.device)

        # Call WAN self-attn with trimodal MoT and custom mask
        y, action_out_h, vlm_out_h = wan_layer.self_attn(
            norm_video, seq_lens, self.grid_sizes, freqs,
            action_q=a_q, action_k=a_k, action_v=a_v,
            vlm_q=v_q, vlm_k=v_k, vlm_v=v_v,
            attn_mask=attn_mask  # asymmetric attention mask
        )

        # Project VLM output back to VLM dimension (3072 -> 2048)
        vlm_out = vlm_block.wan_vlm_o(vlm_out_h.flatten(2))

        # Project back and residual connections
        action_out = action_block.wan_action_o(action_out_h.flatten(2))
        video_tokens = video_tokens + y * v_mod[2].squeeze(2)
        action_tokens = action_tokens + action_out * a_mod[2].squeeze(2)
        vlm_tokens = vlm_tokens + vlm_out  # Regular residual connection

        return video_tokens, action_tokens, vlm_tokens


class ActionModule(nn.Module):
    """Action processing module - handles Action Expert + joint attentions + masks."""

    def __init__(self, action_expert: ActionExpert, config, video_model, vlm_model, dtype, device):
        super().__init__()
        self.action_expert = action_expert
        self.config = config
        self.video_model = video_model  # For accessing WAN weights
        self.vlm_model = vlm_model      # For accessing VLM weights
        self.dtype = dtype
        self.device = device

    def get_time_embedding(self, t: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get action time embedding."""
        if t.dim() == 1:
            t = t.unsqueeze(1).expand(t.size(0), seq_len)

        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t.size(0)
            t_flat = t.flatten()

            # Create sinusoidal embedding (same pattern as VideoModule)
            a_e = self.action_expert.time_embedding(
                sinusoidal_embedding_1d(self.action_expert.freq_dim, t_flat).unflatten(0, (bt, seq_len)).float()
            )  # [B, seq_len, freq_dim]

            # Project to AdaLN parameters (6 params: 3 for WAN-Action joint attn + 3 for FFN)
            a_e0 = self.action_expert.time_projection(a_e).unflatten(2, (6, self.config.action_expert_dim))  # [B, seq_len, 6, dim]

            assert a_e.dtype == torch.float32 and a_e0.dtype == torch.float32

        return a_e, a_e0  # (basic_emb, adaln_params)

    def compute_adaln_modulation(self, action_adaln_params: torch.Tensor, layer_idx: int) -> tuple:
        """Compute AdaLN modulation parameters for 6 components (3 for WAN-Action joint attn + 3 for FFN)."""
        action_layer = self.action_expert.blocks[layer_idx]
        with torch.amp.autocast('cuda', dtype=torch.float32):
            modulation = (
                action_layer.modulation.unsqueeze(0)
                + action_adaln_params
            ).chunk(6, dim=2)
        return modulation

    def process_ffn(self, action_tokens: torch.Tensor, action_adaln_modulation: tuple, layer_idx: int) -> torch.Tensor:
        """Process Action Expert FFN with AdaLN modulation."""
        action_block = self.action_expert.blocks[layer_idx]

        # AdaLN params
        a_mod = action_adaln_modulation

        # Apply FFN with AdaLN modulation (params 3,4,5 for FFN: α3, β3, γ3)
        ffn_input = action_block.norm2(action_tokens).float() * (1 + a_mod[4].squeeze(2)) + a_mod[3].squeeze(2)
        ffn_out = action_block.ffn(ffn_input)

        with torch.amp.autocast('cuda', dtype=torch.float32):
            action_tokens = action_tokens + ffn_out * a_mod[5].squeeze(2)
        return action_tokens


class Ewam(nn.Module):
    """
    Modular Three-modal UniDiffuser with VGM (WAN), VLM (Direct MoT), and Action modules.
    Uses asymmetric attention mask: Action sees all, Video/VLM only self-attention.

    Attention Matrix:
        Query \ Key  | Video | Action | VLM
        -------------|-------|--------|-----
        Video        |   ✅  |   ❌   |  ❌
        Action       |   ✅  |   ✅   |  ✅
        VLM          |   ❌  |   ❌   |  ✅
    """

    def __init__(self, config: EwamConfig):
        super().__init__()
        self.config = config

        # Set unified data type for the model
        self.dtype = torch.bfloat16

        # Initialize video/VLM backbones
        load_backbones = True if config.load_pretrained_backbones is None else bool(config.load_pretrained_backbones)

        # Initialize video model (WAN 5B)
        logger.info("Initializing WAN 5B video model...")
        if load_backbones:
            self.video_model = WanVideoModel.from_pretrained(
                checkpoint_path=config.wan_checkpoint_path,
                vae_path=config.vae_path,
                config_path=config.wan_config_path,
                precision=config.video_precision
            )
        else:
            self.video_model = WanVideoModel.from_config(
                config_path=config.wan_config_path,
                vae_path=config.vae_path,
                device="cuda",
                precision=config.video_precision
            )

        # Initialize VLM (Qwen3-VL-2B, trainable)
        logger.info("Initializing VLM (Qwen3-VL-2B, trainable{})...".format(" - FROZEN" if config.vlm_frozen else ""))
        if load_backbones:
            self.vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
                config.vlm_checkpoint_path,
                dtype=self.dtype,
                device_map="cuda",
                trust_remote_code=True
            )
            logger.info("Load pretrained VLM...")
        else:
            vlm_cfg = AutoConfig.from_pretrained(config.vlm_checkpoint_path, trust_remote_code=True)
            self.vlm_model = Qwen3VLForConditionalGeneration._from_config(vlm_cfg, torch_dtype=self.dtype)
            self.vlm_model.to(device="cuda", dtype=self.dtype)
            logger.info("Initializing VLM from config...")

        # Freeze VLM parameters if specified
        if config.vlm_frozen:
            for param in self.vlm_model.parameters():
                param.requires_grad = False
            logger.info("VLM parameters frozen")
        else:
            logger.info("VLM parameters are TRAINABLE")

        # Keep VLM complete (do not truncate)
        vlm_num_layers = len(self.vlm_model.model.language_model.layers)
        logger.info(f"VLM kept complete with {vlm_num_layers} layers")

        # Get WAN and VLM configurations directly
        wan_dim = getattr(self.video_model.wan_model.config, 'dim', 3072)
        wan_num_heads = getattr(self.video_model.wan_model.config, 'num_heads', 24)
        wan_head_dim = wan_dim // wan_num_heads

        vlm_dim = self.vlm_model.config.text_config.hidden_size
        vlm_num_heads = self.vlm_model.config.text_config.num_attention_heads
        vlm_num_kv_heads = getattr(self.vlm_model.config.text_config if hasattr(self.vlm_model.config, 'text_config') else self.vlm_model.config, 'num_key_value_heads', vlm_num_heads)
        vlm_num_hidden_layers = self.vlm_model.config.text_config.num_hidden_layers
        vlm_head_dim = vlm_dim // vlm_num_heads

        logger.info(f"Model configurations:")
        logger.info(f"  WAN 5B: {wan_num_heads} heads × {wan_head_dim} head_dim = {wan_dim}D, {config.num_layers} layers")
        logger.info(f"  VLM 2B: {vlm_num_heads} Q heads, {vlm_num_kv_heads} KV heads × {vlm_head_dim} head_dim = {vlm_dim}D, {vlm_num_layers} layers")
        logger.info(f"  Layer mapping: VLM 0-27 -> WAN 0-27, VLM 26->WAN 28, VLM 27->WAN 29")

        # Create config dictionaries for ActionExpert
        wan_config = {
            'dim': wan_dim,
            'num_heads': wan_num_heads,
            'head_dim': wan_head_dim
        }

        # Initialize action expert with unified configs
        logger.info("Initializing Action Expert...")

        # Determine chunk_size based on training mode
        if config.training_mode == 'pretrain':
            action_chunk_size_for_expert = config.action_chunk_size
        else:
            action_chunk_size_for_expert = config.action_chunk_size + 1  # include state token

        # Configure registers by mode: no registers in pretrain, keep default (e.g., 4) in finetune
        num_registers = 0 if config.training_mode == 'pretrain' else 4

        action_config = ActionExpertConfig(
            dim=config.action_expert_dim,
            ffn_dim=config.action_expert_dim * config.action_expert_ffn_dim_multiplier,
            num_layers=config.num_layers,
            state_dim=config.action_state_dim,
            action_dim=config.action_dim,
            chunk_size=action_chunk_size_for_expert,
            num_registers=num_registers,
            video_feature_dim=wan_dim,
            causal=False,
            eps=config.action_expert_norm_eps,
            training_mode=config.training_mode,
        )

        self.action_expert = ActionExpert(action_config, wan_config)

        # Move models to device - need device before initializing qwen3_module
        self.device = next(self.video_model.parameters()).device
        self.action_expert.to(device=self.device, dtype=self.dtype)

        # Initialize Qwen3-VL Module (per-layer QKV projections for direct MoT)
        logger.info("Initializing Qwen3-VL Module for direct MoT...")
        qwen3_config = Qwen3VLWanConfig(
            vlm_dim=config.vlm_dim,
            head_dim=config.qwen3_expert_head_dim,
            num_heads=config.qwen3_expert_num_heads,
            num_layers=config.qwen3_expert_num_layers,  # 30 layers (matches WAN)
            eps=config.qwen3_expert_norm_eps,
        )

        self.qwen3_module = Qwen3VLWanModule(self.vlm_model, qwen3_config, self.dtype, self.device)

        # Set time embedding layers to float32 for numerical stability
        self.action_expert.time_embedding.to(dtype=torch.float32)
        self.action_expert.time_projection.to(dtype=torch.float32)

        # Pre-compute grid_sizes for training batch size
        lat_T = 1 + config.num_video_frames // 4
        lat_H = config.video_height // 32
        lat_W = config.video_width // 32
        batch_size = config.batch_size
        self.grid_sizes = torch.tensor(
            [lat_T, lat_H, lat_W],
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0).expand(batch_size, -1)  # [batch_size, 3] - pre-expanded

        logger.info(f"Pre-computed grid_sizes: T={lat_T}, H={lat_H}, W={lat_W}")

        # Initialize modular components
        self.video_module = VideoModule(self.video_model, self.dtype, self.device, self.grid_sizes)
        self.action_module = ActionModule(self.action_expert, self.config, self.video_model, self.vlm_model, self.dtype, self.device)

        # Initialize t distributions from config
        time_dist_config = getattr(config, 'time_distribution', {})
        model_config = {
            'timestep_sample_method': time_dist_config.get('timestep_sample_method', 'logit_normal'),
            'sigmoid_scale': time_dist_config.get('sigmoid_scale', 1.0),
            'min_t': time_dist_config.get('min_t', 0.0),
            'max_t': time_dist_config.get('max_t', 1.0)
        }

        # Flow-Matching scheduler for training (video branch only)
        try:
            self.fm_train_scheduler = FlowMatchScheduler(
                shift=5.0,
                sigma_min=0.0,
                extra_one_step=True,
                num_train_timesteps=1000
            )
            # Enable training mode to build per-timestep weights (if used)
            self.fm_train_scheduler.set_timesteps(num_inference_steps=1000, training=True)
            logger.info("Initialized FlowMatchScheduler for training (video)")
        except Exception as e:
            logger.warning(f"Failed to init FlowMatchScheduler: {e}")

        # Flow-Matching scheduler for training (action branch)
        try:
            self.fm_train_scheduler_action = FlowMatchScheduler(
                shift=5.0,
                sigma_min=0.0,
                extra_one_step=True,
                num_train_timesteps=1000
            )
            # Enable training mode for action as well
            self.fm_train_scheduler_action.set_timesteps(num_inference_steps=1000, training=True)
            logger.info("Initialized FlowMatchScheduler for training (action)")
        except Exception as e:
            logger.warning(f"Failed to init FlowMatchScheduler for action: {e}")

        # Loss variant flag (from config)
        self.loss_reweight = bool(getattr(config, "loss_reweight", False))

        if self.loss_reweight:
            # Precompute training weight stats (Gaussian weighting)
            _steps = 1000
            _shift = 5.0
            u_grid = torch.linspace(1.0, 0.0, _steps + 1, dtype=torch.float64)[:-1]
            t_grid = self._phi(u_grid, _shift) * float(_steps)
            y_grid = torch.exp(-2.0 * ((t_grid - (_steps / 2.0)) / _steps) ** 2)
            self._y_min = float(y_grid.min().item())
            self._weight_norm_const = float((y_grid - self._y_min).mean().item())

        # Log parameter counts
        self.log_parameter_counts()

    def log_parameter_counts(self):
        """Log detailed parameter counts for each component."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        video_params = sum(p.numel() for p in self.video_model.parameters())
        action_params = sum(p.numel() for p in self.action_expert.parameters())
        vlm_params = sum(p.numel() for p in self.vlm_model.parameters())
        qwen3_module_params = sum(p.numel() for p in self.qwen3_module.parameters())

        vlm_trainable_params = sum(p.numel() for p in self.vlm_model.parameters() if p.requires_grad)
        qwen3_module_trainable_params = sum(p.numel() for p in self.qwen3_module.parameters() if p.requires_grad)

        logger.info(f"EwamDirect parameter breakdown:")
        logger.info(f"  Total parameters: {total_params / 1e9:.2f}B")
        logger.info(f"  Trainable parameters: {trainable_params / 1e9:.2f}B")
        logger.info(f"  Video Model (WAN 5B): {video_params / 1e9:.2f}B")
        logger.info(f"  Action Expert: {action_params / 1e6:.1f}M")
        logger.info(f"  VLM (Qwen3-VL-2B): {vlm_params / 1e9:.2f}B (trainable: {vlm_trainable_params / 1e9:.2f}B)")
        logger.info(f"  Qwen3 Module (projections): {qwen3_module_params / 1e6:.1f}M")

    def load_checkpoint(self, path: str, strict: bool = True) -> Dict:
        """Load model checkpoint."""
        # Handle directory path
        checkpoint_path = Path(path)
        if checkpoint_path.is_dir():
            checkpoint_file = checkpoint_path / "mp_rank_00_model_states.pt"
            if not checkpoint_file.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
            path = str(checkpoint_file)

        # Load state dict
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint['module']
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=strict)
        logger.info(f"Checkpoint loaded from {path}: missing={len(missing_keys)}, unexpected={len(unexpected_keys)}")

        # Return additional state
        additional_state = {k: v for k, v in checkpoint.items()
                          if k not in ['module', 'config']}
        return additional_state

    def load_pretrain_weights(self, path: str) -> None:
        """Load weights from a pretrain checkpoint when current mode is finetune.

        Skips layers that depend on state vs action-only differences:
          - action_expert.input_encoder.*
          - action_expert.decoder.*
        """
        if self.config.training_mode != 'finetune':
            raise ValueError("load_pretrain_weights should be called only in finetune mode")

        # Handle directory path - try two possible locations
        checkpoint_path = Path(path)
        if checkpoint_path.is_dir():
            # Try two possible paths
            possible_paths = [
                checkpoint_path / "pytorch_model" / "mp_rank_00_model_states.pt",
                checkpoint_path / "mp_rank_00_model_states.pt",
            ]

            checkpoint_file = None
            for p in possible_paths:
                if p.exists():
                    checkpoint_file = p
                    logger.info(f"Found checkpoint: {checkpoint_file}")
                    break

            if checkpoint_file is None:
                raise FileNotFoundError(
                    f"Checkpoint not found. Tried:\n"
                    f"  - {possible_paths[0]}\n"
                    f"  - {possible_paths[1]}"
                )
            path = str(checkpoint_file)
        else:
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        logger.info(f"Loading pretrain weights from {path}")
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint.get('module', checkpoint)

        # Filter out duplicate keys: keep action_expert.*, skip action_module.action_expert.*
        filtered = {}
        for k, v in state_dict.items():
            if ('action_expert.input_encoder' in k or 'action_expert.decoder' in k):
                continue
            filtered[k] = v

        missing, unexpected = self.load_state_dict(filtered, strict=False)
        logger.info(f"Loaded pretrain weights (filtered). Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    # ---- Non-uniform timestep sampling and loss weighting ----
    # (loss_reweight variant only; exercised when config.loss_reweight is True)

    @staticmethod
    def _phi(u, shift):
        """Shift transform: sigma = shift * u / (1 + (shift-1) * u)"""
        return shift * u / (1.0 + (shift - 1.0) * u)

    def _sample_training_t(self, batch_size, shift=5.0):
        """Continuous timestep sampling with shift."""
        u = torch.rand((batch_size,), device=self.device, dtype=torch.float32)
        sigma = self._phi(u, shift)
        return sigma * float(self.fm_train_scheduler.num_train_timesteps)

    def _training_weight(self, timestep, shift=5.0):
        """Gaussian-shaped weight centered at midpoint timestep."""
        steps = float(self.fm_train_scheduler.num_train_timesteps)
        t = timestep.to(dtype=torch.float32)
        y = torch.exp(-2.0 * ((t - (steps / 2.0)) / steps) ** 2)
        y_shifted = y - self._y_min
        weight = y_shifted / (self._weight_norm_const + 1e-10)
        return weight

    def training_step(
        self,
        first_frame: torch.Tensor,         # [B, C, H, W] - first frame
        video_frames: torch.Tensor,       # [B, num_frames, C, H, W] - target frames
        state: torch.Tensor = None,       # [B, state_dim] - robot state
        actions: torch.Tensor = None,     # [B, chunk_size, action_dim] - actions
        language_embeddings: Optional[List[torch.Tensor]] = None,  # Pre-encoded T5 embeddings for WAN
        vlm_inputs: Optional[List] = None,  # Complete VLM inputs from dataset
        action_dim_is_pad: Optional[torch.Tensor] = None,  # [B, action_dim] bool, True = padded dim excluded from action loss
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        UniDiffuser training step with three modalities (WAN + Action + VLM direct).

        Args:
            first_frame: First video frame for Teacher Forcing
            video_frames: Target video frames
            state: Initial robot state
            actions: Target action sequence
            language_embeddings: Pre-encoded T5 embeddings for WAN model
            vlm_inputs: Complete VLM inputs from dataset
            action_dim_is_pad: Optional [B, action_dim] bool mask; positions marked
                True are padded (zero-filled) dims and excluded from the action loss.
                None falls back to a plain mean over all dims (backward compatible).
            return_dict: Whether to return detailed outputs

        Returns:
            Dictionary containing losses and metrics
        """
        import time  # Import for profiling
        # Reset timing for this step
        if not hasattr(self, '_timing'):
            self._timing = {}
        else:
            self._timing = {}

        B = video_frames.shape[0]

        # 1. Video pipeline
        # Normalize/format
        first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)  # [B, C, 1, H, W]
        video_normalized = (video_frames * 2.0 - 1.0).permute(0, 2, 1, 3, 4)  # [B, C, num_frames, H, W]
        full_video = torch.cat([first_frame_norm, video_normalized], dim=2)  # [B, C, frames+1, H, W]

        # Encode video using VAE
        t_vae_start = time.time()
        with torch.no_grad():
            clean_full_latent = self.video_model.encode_video(full_video.to(self.dtype))  # [B, 48, latent_frames, H', W']
            condition_frame_latent = self.video_model.encode_video(first_frame_norm.to(self.dtype))  # [B, 48, 1, H', W']
        self._timing['vae_encode'] = time.time() - t_vae_start

        # Flow-Matching noise mixture
        if self.loss_reweight:
            # Continuous sampling with shift
            video_t = self._sample_training_t(batch_size=B, shift=5.0)  # [B], continuous [0, 1000]
            video_t_embed = video_t.to(dtype=self.dtype, device=self.device)  # [B]
            sigma = (video_t / self.fm_train_scheduler.num_train_timesteps).to(dtype=self.dtype, device=self.device).view(B, 1, 1, 1, 1)
        else:
            timestep_id = torch.randint(0, self.fm_train_scheduler.num_train_timesteps, (B,))
            # Scalar timesteps (0..num_train_timesteps) for time embedding
            video_t_embed = self.fm_train_scheduler.timesteps[timestep_id].to(dtype=self.dtype, device=self.device)  # [B]
            # Sigma for noise mixture
            sigma = self.fm_train_scheduler.sigmas[timestep_id].to(dtype=self.dtype, device=self.device).view(B, 1, 1, 1, 1)
        video_noise = torch.randn_like(clean_full_latent, dtype=self.dtype)
        noisy_video_latent = clean_full_latent * (1 - sigma) + video_noise * sigma
        # Teacher Forcing on the first frame
        noisy_video_latent[:, :, 0:1] = condition_frame_latent
        # Flow-Matching target: noise - clean
        video_target = video_noise - clean_full_latent
        video_target[:, :, 0:1] = 0

        # Latent to Tokens
        video_tokens = self.video_module.prepare_input(noisy_video_latent.to(self.dtype))

        # 2. Action pipeline
        if self.loss_reweight:
            # Continuous sampling with shift
            action_t = self._sample_training_t(batch_size=B, shift=5.0)  # [B], continuous [0, 1000]
            action_t_embed = action_t.to(dtype=self.dtype, device=self.device)  # [B]
            sigma_action = (action_t / self.fm_train_scheduler_action.num_train_timesteps).to(dtype=self.dtype, device=self.device).view(B, 1, 1)
        else:
            timestep_id_action = torch.randint(0, self.fm_train_scheduler_action.num_train_timesteps, (B,))
            # Discrete timesteps for time embedding (0..num_train_timesteps)
            action_t_embed = self.fm_train_scheduler_action.timesteps[timestep_id_action].to(dtype=self.dtype, device=self.device)  # [B]
            # Sigma for action noise mixture
            sigma_action = self.fm_train_scheduler_action.sigmas[timestep_id_action].to(dtype=self.dtype, device=self.device).view(B, 1, 1)
        action_noise = torch.randn_like(actions, dtype=self.dtype)
        noisy_actions = actions * (1 - sigma_action) + action_noise * sigma_action
        action_target = action_noise - actions

        # Encode Action Chunk with optional Registers
        if self.action_expert.config.num_registers > 0 and self.action_expert.registers is not None:
            registers = self.action_expert.registers.expand(B, -1, -1)  # [B, num_registers, dim]
        else:
            registers = None
        if self.config.training_mode == 'pretrain':
            action_tokens = self.action_expert.input_encoder(None, noisy_actions, registers)
        else:
            state_tokens = state.unsqueeze(1).to(self.dtype)
            action_tokens = self.action_expert.input_encoder(state_tokens, noisy_actions, registers)

        # Extract VLM per-layer features (28 VLM layers -> 30 layers output)
        t_vlm_start = time.time()
        vlm_per_layer = self.qwen3_module.extract_per_layer_features(vlm_inputs)  # List of 30 tensors
        self._timing['vlm_extract'] = time.time() - t_vlm_start
        assert len(vlm_per_layer) == 30, f"Expected 30 VLM layers, got {len(vlm_per_layer)}"

        # Time embeddings
        video_head_time_emb, video_adaln_params = self.video_module.get_time_embedding(video_t_embed, video_tokens.shape[1])
        action_head_time_emb, action_adaln_params = self.action_module.get_time_embedding(action_t_embed, action_tokens.shape[1])

        # T5 preprocess
        processed_t5_context = self.video_module.preprocess_t5_embeddings(language_embeddings)

        # 3. MoT forward
        t_mot_start = time.time()
        # Initialize component timing accumulators
        wan_attn_time = 0
        wan_ffn_time = 0
        action_attn_time = 0
        action_ffn_time = 0
        vlm_attn_time = 0
        vlm_ffn_time = 0

        with torch.autocast(device_type="cuda", dtype=self.video_model.precision):
            # Process through 30 layers - modality-grouped execution
            # Initialize vlm_tokens with first layer VLM features

            for layer_idx in range(self.config.num_layers):
                # VLM tokens for this layer: base layer features + accumulated MoT updates
                # Similar to the original design where und_tokens is updated layer by layer
                vlm_base = vlm_per_layer[layer_idx]  # Base VLM features for this layer
                if layer_idx == 0:
                    vlm_tokens = vlm_base
                else:
                    # Add the difference between current and previous layer, plus accumulated updates
                    # This allows vlm_tokens to evolve through layers like video/action tokens
                    vlm_tokens = (vlm_base + vlm_tokens)/2

                # Compute AdaLN modulation once per layer using pre-computed parameters
                video_adaln_modulation = self.video_module.compute_adaln_modulation(video_adaln_params, layer_idx)
                action_adaln_modulation = self.action_module.compute_adaln_modulation(action_adaln_params, layer_idx)

                # Trimodal MoT: WAN + Action + VLM joint attention
                t_joint_attn = time.time()
                video_tokens, action_tokens, vlm_tokens = self.video_module.process_joint_attention(
                    video_tokens, action_tokens, vlm_tokens,
                    video_adaln_modulation, action_adaln_modulation, layer_idx,
                    self.action_expert.blocks[layer_idx],
                    self.qwen3_module.blocks[layer_idx]
                )
                joint_attn_time = time.time() - t_joint_attn

                # Split joint attention time proportionally
                wan_attn_time += joint_attn_time / 3
                action_attn_time += joint_attn_time / 3
                vlm_attn_time += joint_attn_time / 3

                # WAN cross
                t_cross_attn = time.time()
                video_tokens = self.video_module.process_cross_attention(video_tokens, video_adaln_params, layer_idx, processed_t5_context)
                wan_attn_time += time.time() - t_cross_attn

                # FFNs: WAN, Action (VLM FFN is internal, already computed via extract_per_layer_features)
                t_ffn = time.time()
                video_tokens = self.video_module.process_ffn(video_tokens, video_adaln_modulation, layer_idx)
                wan_ffn_time += time.time() - t_ffn

                t_ffn = time.time()
                action_tokens = self.action_module.process_ffn(action_tokens, action_adaln_modulation, layer_idx)
                action_ffn_time += time.time() - t_ffn

                # VLM has no FFN (internal FFN already computed in extract_per_layer_features)
                # Just return vlm_tokens unchanged
                vlm_tokens = self.qwen3_module.process_ffn(vlm_tokens, layer_idx)

            # 4. Heads + Losses
            t_output_start = time.time()
            video_pred = self.video_module.apply_output_head(video_tokens, video_head_time_emb)
            action_pred_full = self.action_expert.decoder(action_tokens, action_head_time_emb)
            up_len = action_pred_full.shape[1] - self.action_expert.config.num_registers
            # Slice predicted actions depending on mode
            if self.config.training_mode == 'pretrain':
                action_pred = action_pred_full[:, :up_len, :]
            else:
                action_pred = action_pred_full[:, 1:up_len, :]
            self._timing['output_heads'] = time.time() - t_output_start

        # Store component timings
        self._timing['wan_attn'] = wan_attn_time
        self._timing['wan_ffn'] = wan_ffn_time
        self._timing['action_attn'] = action_attn_time
        self._timing['action_ffn'] = action_ffn_time
        self._timing['vlm_attn'] = vlm_attn_time
        self._timing['vlm_ffn'] = vlm_ffn_time
        self._timing['mot_layers'] = time.time() - t_mot_start

        # Video loss (mask the first frame)
        video_pred_masked = video_pred.clone()
        video_pred_masked[:, :, 0:1] = 0

        if self.loss_reweight:
            # Per-sample MSE -> timestep-weighted mean
            video_loss_per_sample = torch.nn.functional.mse_loss(
                video_pred_masked.float(), video_target.float(), reduction="none"
            ).mean(dim=[1, 2, 3, 4])  # [B]
            video_weight = self._training_weight(video_t, shift=5.0).to(
                device=video_loss_per_sample.device, dtype=video_loss_per_sample.dtype
            )
            video_loss = (video_loss_per_sample * video_weight).mean()
        else:
            video_loss = torch.nn.functional.mse_loss(video_pred_masked, video_target, reduction='mean')

        # Action loss
        if self.loss_reweight:
            # Per-sample MSE -> timestep-weighted mean, masked over valid dims. When
            # action_dim_is_pad is provided (UnifyTo16 dim16), padded dims are excluded
            # from the per-sample MSE before weighting.
            if action_dim_is_pad is not None:
                action_loss_element = torch.nn.functional.mse_loss(
                    action_pred.float(), action_target.float(), reduction='none')  # [B, T, D]
                dim_valid = (~action_dim_is_pad).to(device=action_loss_element.device, dtype=action_loss_element.dtype)  # [B, D]
                dim_valid_sum = dim_valid.sum(dim=1).clamp(min=1.0)  # [B]
                action_loss_per_sample = (action_loss_element * dim_valid.unsqueeze(1)).sum(dim=2) \
                    / dim_valid_sum.unsqueeze(1)  # [B, T]
                action_loss_per_sample = action_loss_per_sample.mean(dim=1)  # [B]
            else:
                action_loss_per_sample = torch.nn.functional.mse_loss(
                    action_pred.float(), action_target.float(), reduction="none"
                ).mean(dim=[1, 2])  # [B]
            action_weight = self._training_weight(action_t, shift=5.0).to(
                device=action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
            )
            action_loss = (action_loss_per_sample * action_weight).mean()
        elif action_dim_is_pad is not None:
            # Masked mean: padded (zero-filled) dims are excluded.
            # action_pred / action_target: [B, T, D]; action_dim_is_pad: [B, D]
            action_loss_element = torch.nn.functional.mse_loss(action_pred, action_target, reduction='none')  # [B, T, D]
            dim_valid = (~action_dim_is_pad).to(device=action_loss_element.device, dtype=action_loss_element.dtype)  # [B, D]
            dim_valid_sum = dim_valid.sum(dim=1).clamp(min=1.0)  # [B]
            action_loss = (action_loss_element * dim_valid.unsqueeze(1)).sum(dim=2) / dim_valid_sum.unsqueeze(1)  # [B, T]
            action_loss = action_loss.mean()
        else:
            action_loss = torch.nn.functional.mse_loss(action_pred, action_target, reduction='mean')

        total_loss = (
            self.config.video_loss_weight * video_loss +
            self.config.action_loss_weight * action_loss
        )

        if return_dict:
            loss_dict = {
                'total_loss': total_loss,
                'video_loss': video_loss,
                'action_loss': action_loss,
            }
            if self.loss_reweight:
                loss_dict.update({
                    'video_timestep_mean': video_t.float().mean().item(),
                    'action_timestep_mean': action_t.float().mean().item(),
                    'video_weight_mean': video_weight.float().mean().item(),
                    'action_weight_mean': action_weight.float().mean().item(),
                })
            else:
                loss_dict.update({
                    'video_timestep_mean': sigma.float().mean().item(),
                    'action_timestep_mean': sigma_action.float().mean().item(),
                })
            return loss_dict

    def inference_step(
        self,
        first_frame: torch.Tensor,
        state: torch.Tensor = None,
        num_inference_steps: int = 50,
        language_embeddings: Optional[List[torch.Tensor]] = None,
        vlm_inputs: Optional[List] = None,
        skip_decode: bool = False
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Joint inference for video and action prediction.

        Args:
            first_frame: Initial frame [B, C, H, W]
            state: Initial robot state [B, state_dim]
            num_inference_steps: Number of denoising steps
            language_embeddings: Pre-encoded T5 embeddings for WAN model
            vlm_inputs: VLM inputs
            skip_decode: If True, skip the VAE decode of video_latent and return None for
                predicted_frames. Actions come from action_latent and are unaffected by this.
                Use when the caller only needs actions (not video frames) to save one VAE decode.

        Returns:
            Tuple of (predicted_frames, predicted_actions). predicted_frames is None if
            skip_decode=True.
        """
        # Per-stage profiling (off by default; enable with EWAM_PROFILE=1).
        # Instrumentation only — does not change any computed value.
        _profile = os.environ.get("EWAM_PROFILE", "0") not in ("0", "", "false", "False")
        if _profile:
            import time as _time
            _timings = {"vae_encode": 0.0, "vlm_extract": 0.0, "t5_preprocess": 0.0,
                        "loop_total": 0.0, "loop_joint_attn": 0.0, "loop_cross_attn": 0.0,
                        "loop_ffn": 0.0, "loop_heads": 0.0, "loop_prepare": 0.0,
                        "vae_decode": 0.0}
            _vlm_call_count = 0
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _t_total = _time.perf_counter()

        B = first_frame.shape[0]

        language_embeddings = [emb.to(self.device).to(self.dtype) for emb in language_embeddings]
        state = state.to(self.device).to(self.dtype)
        first_frame = first_frame.to(self.device).to(self.dtype)

        # 1. Video/Action latents init
        first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)   # [0,1] -> [-1,1], [B, C, 1, H, W]
        if _profile and torch.cuda.is_available():
            torch.cuda.synchronize(); _t = _time.perf_counter()
        with torch.no_grad():
            condition_frame_latent = self.video_model.encode_video(first_frame_norm.to(self.dtype))   # [B, C', 1, H', W']
        if _profile and torch.cuda.is_available():
            torch.cuda.synchronize(); _timings["vae_encode"] += _time.perf_counter() - _t

        # Init video/action latents
        B, C_latent, f_latent, H_latent, W_latent = condition_frame_latent.shape
        num_total_latent_frames = 1 + self.config.num_video_frames // 4
        video_latent = torch.randn((B, C_latent, num_total_latent_frames, H_latent, W_latent), device=self.device, dtype=self.dtype)
        video_latent[:, :, 0:1] = condition_frame_latent
        action_shape = (B, self.config.action_chunk_size, self.config.action_dim)
        action_latent = torch.randn(action_shape, device=self.device, dtype=self.dtype)

        # 2. VLM per-layer features and T5 context
        if _profile and torch.cuda.is_available():
            torch.cuda.synchronize(); _t = _time.perf_counter()
        vlm_per_layer = self.qwen3_module.extract_per_layer_features(vlm_inputs)
        if _profile:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _timings["vlm_extract"] += _time.perf_counter() - _t
            _vlm_call_count += 1
        assert len(vlm_per_layer) == 30, f"Expected 30 VLM layers, got {len(vlm_per_layer)}"

        # T5 preprocess
        if _profile and torch.cuda.is_available():
            torch.cuda.synchronize(); _t = _time.perf_counter()
        processed_t5_context = self.video_module.preprocess_t5_embeddings(language_embeddings)
        if _profile and torch.cuda.is_available():
            torch.cuda.synchronize(); _timings["t5_preprocess"] += _time.perf_counter() - _t

        # 3. Denoising loop: from noise (t=1) to clean (t=0)
        timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=self.device, dtype=self.dtype)
        if _profile and torch.cuda.is_available():
            torch.cuda.synchronize(); _loop_start = _time.perf_counter()
        for i in range(num_inference_steps):
            # Timesteps
            t = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t
            video_t_scaled = (t * 1000).expand(B).to(self.dtype)
            action_t_scaled = (t * 1000).expand(B).to(self.dtype)

            if _profile and torch.cuda.is_available():
                torch.cuda.synchronize(); _t_prep = _time.perf_counter()
            # Tokens with Registers
            video_tokens = self.video_module.prepare_input(video_latent.to(self.dtype))
            state_tokens = state.unsqueeze(1).to(self.dtype)
            registers = self.action_expert.registers.expand(B, -1, -1)  # [B, num_registers, dim]
            action_tokens = self.action_expert.input_encoder(state_tokens, action_latent, registers)
            if _profile and torch.cuda.is_available():
                torch.cuda.synchronize(); _timings["loop_prepare"] += _time.perf_counter() - _t_prep

            # NOTE: vlm_per_layer is computed ONCE before the loop (line ~915) and reused.
            # The VLM forward is a pure function of vlm_inputs (image+text), which is constant
            # across denoising steps; under eval()+inference_mode it is deterministic, so the
            # per-step recomputation was redundant (~10x). Verified bit-exact equal to original.
            # Only vlm_tokens (derived below via (vlm_base + vlm_tokens)/2) evolves layer-by-layer;
            # it reads vlm_per_layer read-only.
            # Trimodal MoT forward - joint denoising for WAN, Action, VLM
            with torch.autocast(device_type="cuda", dtype=self.video_model.precision):
                # Time embeddings
                video_head_time_emb, video_adaln_params = self.video_module.get_time_embedding(video_t_scaled, video_tokens.shape[1])
                action_head_time_emb, action_adaln_params = self.action_module.get_time_embedding(action_t_scaled, action_tokens.shape[1])

                # Process through all layers - trimodal denoising of WAN, Action, VLM
                # Initialize vlm_tokens with first layer VLM features
                vlm_tokens = vlm_per_layer[0]  # [B, T_vlm, 2048]

                for layer_idx in range(self.config.num_layers):
                    # VLM tokens for this layer: base layer features + accumulated MoT updates
                    # Similar to training_step and the original design where und_tokens evolves through layers
                    vlm_base = vlm_per_layer[layer_idx]  # Base VLM features for this layer
                    if layer_idx == 0:
                        vlm_tokens = vlm_base
                    else:
                        # Mix VLM layer output with accumulated updates (average)
                        vlm_tokens = (vlm_base + vlm_tokens) / 2

                    # Compute AdaLN modulation using pre-computed parameters
                    video_adaln_modulation = self.video_module.compute_adaln_modulation(video_adaln_params, layer_idx)
                    action_adaln_modulation = self.action_module.compute_adaln_modulation(action_adaln_params, layer_idx)

                    # Trimodal joint attention: WAN + Action + VLM
                    if _profile and torch.cuda.is_available():
                        torch.cuda.synchronize(); _t_a = _time.perf_counter()
                    video_tokens, action_tokens, vlm_tokens = self.video_module.process_joint_attention(
                        video_tokens, action_tokens, vlm_tokens,
                        video_adaln_modulation, action_adaln_modulation, layer_idx,
                        self.action_expert.blocks[layer_idx],
                        self.qwen3_module.blocks[layer_idx]
                    )
                    if _profile and torch.cuda.is_available():
                        torch.cuda.synchronize(); _timings["loop_joint_attn"] += _time.perf_counter() - _t_a; _t_a = _time.perf_counter()

                    # WAN cross-attention with T5 embeddings
                    video_tokens = self.video_module.process_cross_attention(
                        video_tokens, video_adaln_params, layer_idx, processed_t5_context
                    )
                    if _profile and torch.cuda.is_available():
                        torch.cuda.synchronize(); _timings["loop_cross_attn"] += _time.perf_counter() - _t_a; _t_a = _time.perf_counter()

                    # FFNs: WAN, Action (VLM FFN is internal)
                    video_tokens = self.video_module.process_ffn(video_tokens, video_adaln_modulation, layer_idx)
                    action_tokens = self.action_module.process_ffn(action_tokens, action_adaln_modulation, layer_idx)
                    vlm_tokens = self.qwen3_module.process_ffn(vlm_tokens, layer_idx)
                    if _profile and torch.cuda.is_available():
                        torch.cuda.synchronize(); _timings["loop_ffn"] += _time.perf_counter() - _t_a

                # Heads (velocities)
                if _profile and torch.cuda.is_available():
                    torch.cuda.synchronize(); _t_h = _time.perf_counter()
                video_velocity = self.video_module.apply_output_head(video_tokens, video_head_time_emb)
                action_pred_full = self.action_expert.decoder(action_tokens, action_head_time_emb)
                # Extract middle action chunk (skip first state token and last register tokens)
                action_velocity = action_pred_full[:, 1:-self.action_expert.config.num_registers, :]
                if _profile and torch.cuda.is_available():
                    torch.cuda.synchronize(); _timings["loop_heads"] += _time.perf_counter() - _t_h

                # Euler integration
                video_latent = video_latent + video_velocity * dt
                action_latent = action_latent + action_velocity * dt

                # Teacher Forcing
                video_latent[:, :, 0:1] = condition_frame_latent

        if _profile and torch.cuda.is_available():
            torch.cuda.synchronize(); _timings["loop_total"] = _time.perf_counter() - _loop_start

        # 4. Decode outputs (skipped when caller only needs actions — VAE decode is
        #    independent of action_latent, so skipping is output-preserving for actions)
        if skip_decode:
            predicted_frames = None
        else:
            if _profile and torch.cuda.is_available():
                torch.cuda.synchronize(); _t = _time.perf_counter()
            with torch.no_grad():
                decoded_frames = self.video_model.decode_video(video_latent)
                predicted_frames = decoded_frames[:, :, 1:]  # Skip first frame (condition)
                predicted_frames = (predicted_frames + 1.0) / 2.0  # [-1,1] to [0,1]
                predicted_frames = torch.clamp(predicted_frames, 0, 1).float()
            if _profile and torch.cuda.is_available():
                torch.cuda.synchronize(); _timings["vae_decode"] += _time.perf_counter() - _t

        predicted_actions = action_latent.float()  # [B, action_chunk_size, 14]

        if _profile:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _total = _time.perf_counter() - _t_total
            _timings["loop_other"] = max(0.0, _timings["loop_total"]
                                         - _timings["loop_prepare"] - _timings["vlm_extract"]
                                         - _timings["loop_joint_attn"] - _timings["loop_cross_attn"]
                                         - _timings["loop_ffn"] - _timings["loop_heads"])
            _sum_stages = (_timings["vae_encode"] + _timings["vlm_extract"] + _timings["t5_preprocess"]
                           + _timings["loop_total"] + _timings["vae_decode"])
            logger.info("=" * 70)
            logger.info(f"[EWAM_PROFILE] inference_step total: {_total*1000:.1f} ms "
                        f"(stages sum: {_sum_stages*1000:.1f} ms) | steps={num_inference_steps} layers={self.config.num_layers}")
            logger.info(f"[EWAM_PROFILE]   vae_encode:        {_timings['vae_encode']*1000:8.1f} ms")
            logger.info(f"[EWAM_PROFILE]   vlm_extract:       {_timings['vlm_extract']*1000:8.1f} ms  "
                        f"(calls={_vlm_call_count} -> {(_timings['vlm_extract']/max(_vlm_call_count,1))*1000:.1f} ms/call)")
            logger.info(f"[EWAM_PROFILE]   t5_preprocess:     {_timings['t5_preprocess']*1000:8.1f} ms")
            logger.info(f"[EWAM_PROFILE]   denoise loop:      {_timings['loop_total']*1000:8.1f} ms  (per step: {(_timings['loop_total']/num_inference_steps)*1000:.1f} ms)")
            logger.info(f"[EWAM_PROFILE]     prepare/tok:     {_timings['loop_prepare']*1000:8.1f} ms")
            logger.info(f"[EWAM_PROFILE]     vlm_extract(loop):{_timings['vlm_extract']*1000:8.1f} ms  <- P0 target (redundant)")
            logger.info(f"[EWAM_PROFILE]     joint_attn:      {_timings['loop_joint_attn']*1000:8.1f} ms")
            logger.info(f"[EWAM_PROFILE]     cross_attn:      {_timings['loop_cross_attn']*1000:8.1f} ms")
            logger.info(f"[EWAM_PROFILE]     ffn:             {_timings['loop_ffn']*1000:8.1f} ms")
            logger.info(f"[EWAM_PROFILE]     heads:           {_timings['loop_heads']*1000:8.1f} ms")
            logger.info(f"[EWAM_PROFILE]     loop_other:      {_timings['loop_other']*1000:8.1f} ms")
            logger.info(f"[EWAM_PROFILE]   vae_decode:        {_timings['vae_decode']*1000:8.1f} ms")
            logger.info("=" * 70)

        return predicted_frames, predicted_actions
