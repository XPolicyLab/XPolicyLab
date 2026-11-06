# SkyrobotWAM inference model with KV cache


import os
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange

from skyrobot.models.attention import WanLayerNorm, WanSelfAttention, WanCrossAttention
from skyrobot.models.rotary import sinusoidal_embedding_1d, WanRotaryPosEmbed
from skyrobot.models.model import SkyrobotWAM, Head, Action_Head
from skyrobot.utils.scheduler import FlowMatchScheduler


# ---------------------------------------------------------------------------
# KV Cache
# ---------------------------------------------------------------------------

class KVCache:
    """Per-layer self-attention KV cache that grows by appending clean tokens."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.k: list[torch.Tensor | None] = [None] * num_layers
        self.v: list[torch.Tensor | None] = [None] * num_layers

    def append(self, layer_idx: int, new_k: torch.Tensor, new_v: torch.Tensor):
        if self.k[layer_idx] is None:
            self.k[layer_idx] = new_k
            self.v[layer_idx] = new_v
        else:
            self.k[layer_idx] = torch.cat([self.k[layer_idx], new_k], dim=1)
            self.v[layer_idx] = torch.cat([self.v[layer_idx], new_v], dim=1)

    def get(self, layer_idx: int):
        return self.k[layer_idx], self.v[layer_idx]

    def reset(self):
        self.k = [None] * self.num_layers
        self.v = [None] * self.num_layers


# ---------------------------------------------------------------------------
# Inference Attention Block (shares weights with WanAttentionBlock)
# ---------------------------------------------------------------------------

class InferenceBlock(nn.Module):
    """WanAttentionBlock with two inference forward paths sharing the same weights.

    - forward_append:  process new clean tokens, update KV cache.
    - forward_denoise: process noisy tokens with cached KV, do NOT update cache.
    """

    def __init__(self, block):
        """Wrap an existing WanAttentionBlock (zero-copy, shares parameters)."""
        super().__init__()
        self.norm1 = block.norm1
        self.self_attn = block.self_attn
        self.norm3 = block.norm3
        self.cross_attn = block.cross_attn
        self.norm2 = block.norm2
        self.ffn = block.ffn
        self.modulation = block.modulation

    def _modulate(self, hidden_states, temb):
        dt = hidden_states.dtype
        temb_mod = self.modulation[None].to(dt) + temb.to(dt)
        shift_msa, scale_msa, gate_msa, c_shift, c_scale, c_gate = \
            rearrange(temb_mod, 'b l n c -> b n l c').chunk(6, dim=1)
        # Mirror training: modulation arithmetic done in fp32 (see
        # skyrobot/models/attention_block.py WanAttentionBlock.forward).
        # Doing it in bf16 here was a 0.5%/layer drift that compounds across
        # ~30 layers and grows with training as modulation magnitudes sharpen.
        shift_msa = shift_msa.squeeze(1).float()
        scale_msa = scale_msa.squeeze(1).float()
        gate_msa = gate_msa.squeeze(1).float()
        c_shift = c_shift.squeeze(1).float()
        c_scale = c_scale.squeeze(1).float()
        c_gate = c_gate.squeeze(1).float()
        return shift_msa, scale_msa, gate_msa, c_shift, c_scale, c_gate

    def forward_append(self, hidden_states, encoder_hidden_states, temb,
                       rotary_emb, kv_cache, layer_idx):
        """Process clean tokens -> update KV cache."""
        dt = hidden_states.dtype
        shift_msa, scale_msa, gate_msa, c_shift, c_scale, c_gate = \
            self._modulate(hidden_states, temb)

        # 1. Self-attention with cache (update_cache=True)
        x = self.norm1(hidden_states).float() * (1. + scale_msa) + shift_msa
        x = x.to(dt)
        cached_k, cached_v = kv_cache.get(layer_idx)
        x, new_k, new_v = self.self_attn.forward_with_cache(
            x, rotary_emb, cached_k, cached_v, update_cache=True)
        kv_cache.k[layer_idx] = new_k
        kv_cache.v[layer_idx] = new_v
        hidden_states = hidden_states.float() + x.float() * gate_msa
        hidden_states = hidden_states.to(dt)

        # 2. Cross-attention (standard SDPA, no FlexAttn)
        x = self.cross_attn.forward_inference(
            self.norm3(hidden_states), encoder_hidden_states)
        hidden_states = hidden_states + x

        # 3. FFN
        x = self.norm2(hidden_states).float() * (1. + c_scale) + c_shift
        x = self.ffn(x.to(dt)).float()
        hidden_states = hidden_states.float() + x * c_gate
        hidden_states = hidden_states.to(dt)

        return hidden_states

    def forward_denoise(self, hidden_states, encoder_hidden_states, temb,
                        rotary_emb, kv_cache, layer_idx):
        """Process noisy tokens using cached KV, do NOT update cache."""
        dt = hidden_states.dtype
        shift_msa, scale_msa, gate_msa, c_shift, c_scale, c_gate = \
            self._modulate(hidden_states, temb)

        # 1. Self-attention with cache (update_cache=False)
        x = self.norm1(hidden_states).float() * (1. + scale_msa) + shift_msa
        x = x.to(dt)
        cached_k, cached_v = kv_cache.get(layer_idx)
        x, _, _ = self.self_attn.forward_with_cache(
            x, rotary_emb, cached_k, cached_v, update_cache=False)
        hidden_states = hidden_states.float() + x.float() * gate_msa
        hidden_states = hidden_states.to(dt)

        # 2. Cross-attention (standard SDPA, no FlexAttn)
        x = self.cross_attn.forward_inference(
            self.norm3(hidden_states), encoder_hidden_states)
        hidden_states = hidden_states + x

        # 3. FFN
        x = self.norm2(hidden_states).float() * (1. + c_scale) + c_shift
        x = self.ffn(x.to(dt)).float()
        hidden_states = hidden_states.float() + x * c_gate
        hidden_states = hidden_states.to(dt)

        return hidden_states


# ---------------------------------------------------------------------------
# Inference Model
# ---------------------------------------------------------------------------

class SkyrobotWAMForInference(nn.Module):
    """SkyrobotWAM inference model with VAE, T5, and KV cache.

    Loads weights from a trained SkyrobotWAM and provides:
      - encode_video:  raw video frames -> VAE latents
      - encode_text:   text prompts -> T5 embeddings
      - predict_action: full action-only inference loop (given video + states)
    """

    def __init__(self, model: SkyrobotWAM, video_height: int = 320,
                 video_width: int = 480):
        super().__init__()

        # Copy config
        self.patch_size = model.patch_size
        self.text_len = model.text_len
        self.in_dim = model.in_dim
        self.dim = model.dim
        self.freq_dim = model.freq_dim
        self.action_dim = model.action_dim
        self.action_per_latent = model.action_per_latent
        self.state_dim = model.state_dim
        self.out_dim = model.out_dim
        self.num_heads = model.num_heads
        self.head_dim = model.head_dim
        self.num_layers = model.num_layers

        # Video target resolution (must match training).
        # Must be divisible by VAE spatial stride (=16) * patch_size[1:].
        vae_spatial_stride = 16  # Wan2.2 VAE
        align_h = vae_spatial_stride * model.patch_size[1]
        align_w = vae_spatial_stride * model.patch_size[2]
        assert video_height % align_h == 0, (
            f"video_height={video_height} must be divisible by {align_h} "
            f"(= vae_stride=16 * patch_size[1]={model.patch_size[1]})."
        )
        assert video_width % align_w == 0, (
            f"video_width={video_width} must be divisible by {align_w} "
            f"(= vae_stride=16 * patch_size[2]={model.patch_size[2]})."
        )
        self.video_height = video_height
        self.video_width = video_width

        # Share embedding weights (zero-copy)
        self.text_embedding = model.text_embedding
        self.patch_embedding = model.patch_embedding
        self.time_embedding = model.time_embedding
        self.time_projection = model.time_projection
        self.action_embedding = model.action_embedding
        self.action_time_embedding = model.action_time_embedding
        self.action_time_projection = model.action_time_projection
        self.state_embedding = model.state_embedding
        self.state_time_embedding = model.state_time_embedding
        self.state_time_projection = model.state_time_projection
        self.rope = model.rope

        # Wrap each block for inference
        self.blocks = nn.ModuleList([
            InferenceBlock(block) for block in model.blocks
        ])

        # Heads
        self.head = model.head
        self.action_head = model.action_head

        # Embodiment routing (CategorySpecificLinear) — copy from trained model.
        # Needed by _embed_action / _embed_state / action_head at inference.
        self.supported_embodiments = list(model.supported_embodiments)
        self.embodiments_table = dict(model.embodiments_table)

        # VAE and T5 (set via load_encoders)
        self.vae = None
        self.text_encoder = None
        # T5 placement/reload bookkeeping (see load_text_encoder).
        self._t5_run_device = 'cpu'
        self._t5_load_cfg = None
        self._t5_tokenizer = None

    # ------------------------------------------------------------------
    # Embodiment helpers
    # ------------------------------------------------------------------

    def resolve_cat_ids(self, embodiment_id, batch_size: int = 1,
                        device=None) -> torch.Tensor:
        """Resolve a raw embodiment_id to a cat_ids tensor with friendly errors.

        Args:
            embodiment_id: int, EmbodimentId, or None. If None and the model
                           was trained with exactly one embodiment, implicit
                           slot 0 is used; otherwise raises.
            batch_size: size of the returned [B] tensor.
            device: target device; defaults to the model's device.

        Returns:
            cat_ids: [batch_size] long on `device`.
        """
        if device is None:
            device = next(self.parameters()).device
        if embodiment_id is None:
            if len(self.supported_embodiments) == 1:
                return torch.zeros(batch_size, dtype=torch.long, device=device)
            raise ValueError(
                f"embodiment_id is required — model was trained with "
                f"{len(self.supported_embodiments)} embodiments "
                f"(supported: {self.supported_embodiments}). "
                f"Pass embodiment_id=<int> (e.g. DATA_CFG.embodiment_id)."
            )
        raw = int(embodiment_id)
        if raw not in self.embodiments_table:
            raise ValueError(
                f"embodiment_id={raw} not in supported_embodiments="
                f"{self.supported_embodiments}. Model wasn't trained on this "
                f"body, so no CategorySpecificLinear slot exists for it. "
                f"Either retrain with this body included, or use a supported "
                f"embodiment_id."
            )
        slot = self.embodiments_table[raw]
        return torch.full((batch_size,), slot, dtype=torch.long, device=device)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, pretrained_path: str, device='cuda',
                        dtype=torch.bfloat16,
                        video_height: int = 320, video_width: int = 480,
                        **config_kwargs):
        """Load from a save_pretrained directory (config.json + safetensors)."""
        model = SkyrobotWAM.from_pretrained(pretrained_path,
                                            torch_dtype=dtype, **config_kwargs)
        model = model.to(device).to(dtype)
        infer_model = cls(model, video_height=video_height,
                          video_width=video_width)
        return infer_model.to(device).eval()

    @classmethod
    def from_trained_model(cls, model: SkyrobotWAM, device='cuda',
                           dtype=torch.bfloat16,
                           video_height: int = 320, video_width: int = 480):
        """Wrap an already-loaded SkyrobotWAM (zero-copy, shares weights)."""
        infer_model = cls(model, video_height=video_height,
                          video_width=video_width)
        return infer_model.to(device).to(dtype).eval()

    def compile_for_inference(self, mode="reduce-overhead"):
        """Compile hot inference paths with torch.compile for speedup.

        Args:
            mode: torch.compile mode. "reduce-overhead" uses CUDA graphs
                  for best latency; "default" for safer compatibility.
        """
        for i, block in enumerate(self.blocks):
            block.forward_denoise = torch.compile(
                block.forward_denoise, mode=mode, dynamic=True)
            block.forward_append = torch.compile(
                block.forward_append, mode=mode, dynamic=True)
        # Action head is called every denoise step
        self.action_head = torch.compile(
            self.action_head, mode=mode, dynamic=True)
        return self

    def load_vae(self, vae_path: str, device='cuda'):
        """Load Wan2.2 VAE for video encoding/decoding.

        Args:
            vae_path: path to directory containing Wan2.2_VAE.pth
        """
        from skyrobot.models.vae.vae2_2 import Wan2_2_VAE
        self.vae = Wan2_2_VAE(
            vae_pth=os.path.join(vae_path, "Wan2.2_VAE.pth"), device=device)

    @staticmethod
    def resolve_t5_device(device='cuda', t5_cpu: bool = False,
                          t5_device=None) -> str:
        """Pick the device the umt5-xxl encoder lives/runs on.

        Priority: explicit ``t5_device`` > legacy ``t5_cpu`` flag > ``device``.
        Returned as a plain string so it can be compared/logged cheaply.
        """
        if t5_device is not None and str(t5_device).strip() != "":
            return str(t5_device)
        return 'cpu' if t5_cpu else str(device)

    def load_text_encoder(self, t5_path: str, device='cuda',
                          text_len: int = 512, dtype=torch.bfloat16,
                          t5_cpu: bool = False, t5_device=None,
                          lazy: bool = False):
        """Load T5 encoder for text embedding.

        Args:
            t5_path: path to directory containing models_t5_umt5-xxl-enc-bf16.pth
                     and google/umt5-xxl tokenizer.
            t5_cpu: legacy flag — keep the ~11.4GB T5 weights on CPU. Frees the
                    VRAM but each encode costs seconds-to-minutes of CPU compute;
                    prefer ``t5_device`` to park T5 on a SECOND GPU instead.
            t5_device: explicit device for T5, e.g. ``"cuda:1"`` or ``"cpu"``.
                    Overrides ``t5_cpu``. Use this on 24GB cards to split the
                    encoder off the card running the 5B diffusion backbone.
            lazy: record the load config but do not read the weights yet; the
                    encoder is built on the first encode (and rebuilt after
                    ``unload_text_encoder``).
        """
        # Where T5 lives AND runs. encode_text* reads self._t5_run_device and
        # moves the resulting embedding to the transformer's device afterwards.
        self._t5_run_device = self.resolve_t5_device(device, t5_cpu, t5_device)
        self._t5_load_cfg = dict(t5_path=str(t5_path), text_len=int(text_len),
                                 dtype=dtype)
        self.text_encoder = None
        if not lazy:
            self._build_text_encoder()

    def _build_text_encoder(self):
        """Materialize the T5 weights on ``self._t5_run_device`` (idempotent)."""
        if self.text_encoder is not None:
            return self.text_encoder
        cfg = getattr(self, "_t5_load_cfg", None)
        if cfg is None:
            raise RuntimeError("Call load_text_encoder()/load_encoders() first")
        from skyrobot.models.t5.t5 import T5EncoderModel

        t5_path = cfg["t5_path"]
        run_dev = self._t5_run_device
        # Always build on CPU first: constructing directly on the target GPU
        # would hold the meta-init copy AND the checkpoint at the same time.
        enc = T5EncoderModel(
            text_len=cfg["text_len"],
            dtype=cfg["dtype"],
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(t5_path,
                                         "models_t5_umt5-xxl-enc-bf16.pth"),
            tokenizer_path=os.path.join(t5_path, "google/umt5-xxl"),
            shard_fn=None)
        if str(run_dev) == 'cpu':
            # bf16 matmul on CPUs without AMX falls back to very slow kernels
            # (measured 3x slower than fp32 even on AVX512; far worse on older
            # CPUs) — enough to blow the ws request timeout on episode reset.
            # Run the offloaded T5 in fp32: ~2x RAM (11G -> 23G), much faster.
            enc.model.float()
        else:
            enc.model.to(run_dev)
        self.text_encoder = enc
        # Survives unload_text_encoder(): the tokenizer is tiny and the
        # key-padding mask in encode_text_with_mask needs it either way.
        self._t5_tokenizer = enc.tokenizer
        print(f"[model_dp] T5 encoder resident on {run_dev}")
        return enc

    def unload_text_encoder(self):
        """Free the T5 weights (~11.4GB) but keep the tokenizer + load config.

        The next ``encode_text*`` transparently rebuilds it. Use when the T5
        card is also running something else (e.g. Isaac Sim) and the encoder is
        only needed at episode reset.
        """
        if self.text_encoder is None:
            return
        run_dev = getattr(self, "_t5_run_device", 'cpu')
        self._t5_tokenizer = self.text_encoder.tokenizer
        self.text_encoder.model = None
        self.text_encoder = None
        import gc
        gc.collect()
        if 'cuda' in str(run_dev):
            with torch.cuda.device(run_dev):
                torch.cuda.empty_cache()
        print(f"[model_dp] T5 encoder unloaded from {run_dev}")

    def load_encoders(self, encoder_path: str, device='cuda',
                      text_len: int = 512, dtype=torch.bfloat16,
                      t5_cpu: bool = False, t5_device=None,
                      t5_lazy: bool = False):
        """Load both VAE and T5 from a single base directory.

        Expects:
            encoder_path/Wan2.2_VAE.pth
            encoder_path/models_t5_umt5-xxl-enc-bf16.pth
            encoder_path/google/umt5-xxl/

        The VAE always follows the transformer (``device``) — it runs every
        observe(). Only T5 is placeable via ``t5_device`` / ``t5_cpu``.
        """
        self.load_vae(encoder_path, device=device)
        self.load_text_encoder(encoder_path, device=device,
                               text_len=text_len, dtype=dtype, t5_cpu=t5_cpu,
                               t5_device=t5_device, lazy=t5_lazy)

    # ------------------------------------------------------------------
    # Encoding helpers (VAE + T5)
    # ------------------------------------------------------------------

    def _resize_frames(self, frames):
        """Resize (保持宽高比) + 黑色 letterbox padding 到 (video_height, video_width).

        Mirrors training's `SkyRobotLeRobotDataset._resize_frames`
        (skyrobot/data/lerobot_datasets.py): fit-scale (long side touches
        target) + zero-pad on the short side. NO cropping.

        Args:
            frames: [F, C, H, W] float32
        Returns:
            frames: [F, C, video_height, video_width]
        """
        th, tw = self.video_height, self.video_width
        _, _, H, W = frames.shape
        if H == th and W == tw:
            return frames
        scale = min(tw / W, th / H)
        new_h = min(round(H * scale), th)
        new_w = min(round(W * scale), tw)
        frames = torch.nn.functional.interpolate(
            frames, size=(new_h, new_w), mode="bilinear", align_corners=False)
        pad_h = th - new_h
        pad_w = tw - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        # F.pad on last 2 dims: (left, right, top, bottom)
        frames = torch.nn.functional.pad(
            frames, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
        assert frames.shape[-2] == th and frames.shape[-1] == tw, (
            f"Padded shape {frames.shape} does not match target ({th}, {tw})."
        )
        return frames

    @torch.no_grad()
    def encode_text(self, prompts: list[str], device='cuda') -> torch.Tensor:
        """Encode text prompts to T5 embeddings.

        Pad-token outputs are zeroed and a key-padding mask is installed on
        WanCrossAttention so inference cross-attention hard-excludes pad
        tokens (parity with the training-side FlexAttn cross mask). Return
        type unchanged (just the embedding) so existing callers keep working.

        Args:
            prompts: list of B text strings.

        Returns:
            text_emb: [B, text_len, 4096]  (pad positions zeroed)
        """
        context, kpm = self.encode_text_with_mask(prompts, device)
        # Backward-compat side effect: install the cross-attn key-padding mask
        # so single-pass callers (no CFG) still hard-exclude pad tokens.
        from skyrobot.models.attention import WanCrossAttention
        WanCrossAttention.infer_text_kpm = kpm
        return context

    def encode_text_with_mask(self, prompts: list[str], device='cuda'):
        """Same as encode_text but ALSO returns the cross-attn key-padding
        mask as a value (does NOT install the classvar). Used by CFG so the
        caller can swap cond/uncond masks per pass.

        Returns:
            context: [B, text_len, 4096]  (pad positions zeroed)
            kpm:     [B, 1, 1, text_len] bool  (True = attend, False = pad)
        """
        if getattr(self, "_t5_load_cfg", None) is None:
            raise AssertionError("Call load_text_encoder() first")
        # Rebuilds the weights when running lazily / after unload_text_encoder.
        encoder = self._build_text_encoder()
        # Run T5 on its resident device (a second GPU, or CPU when offloaded),
        # then move the embedding to the device the transformer runs on.
        run_dev = getattr(self, "_t5_run_device", device)
        context = encoder(prompts, run_dev)  # [B, text_len, 4096]
        if str(run_dev) != str(device):
            context = context.to(device)
        # Recover the same tokenizer mask T5 used internally (deterministic).
        _, mask = self._t5_tokenizer(
            prompts, return_mask=True, add_special_tokens=True)
        keep = mask.to(context.device).bool()  # [B, text_len], True = real
        empty = ~keep.any(dim=1)
        if empty.any():
            keep[empty, 0] = True
        context = context * keep.unsqueeze(-1).to(context.dtype)  # zero pads
        return context, keep[:, None, None, :]  # [B,1,1,L]

    # ------------------------------------------------------------------
    # Embedding helpers (transformer internal)
    # ------------------------------------------------------------------

    def _embed_latent(self, latent):
        """latent: [B, C, F, H, W] -> [B, L, dim]"""
        x = self.patch_embedding(latent)
        return x.flatten(2).transpose(1, 2)

    def _embed_action(self, action, cat_ids):
        """action: [B, C_a, F, apf, 1] -> [B, F*apf, dim]"""
        x = rearrange(action, 'b c f h w -> b (f h w) c')
        return self.action_embedding(x, cat_ids)

    def _embed_state(self, state, cat_ids):
        """state: [B, C_s, F, 1, 1] -> [B, F, dim]"""
        x = rearrange(state, 'b c f h w -> b (f h w) c')
        return self.state_embedding(x, cat_ids)

    def _embed_text(self, text):
        """text: [B, text_len, 4096] -> [B, text_len, dim]"""
        return self.text_embedding(text)

    def _latent_temb(self, timesteps, H, W):
        """Compute latent time embedding + projection.
        timesteps: [B, F_latent].
        Returns: temb [B, L, dim], proj [B, L, 6, dim]
        """
        ph, pw = self.patch_size[1], self.patch_size[2]
        spatial_tokens = (H // ph) * (W // pw)
        ts = torch.repeat_interleave(timesteps, spatial_tokens, dim=1)
        emb = sinusoidal_embedding_1d(self.freq_dim, ts).to(
            device=timesteps.device, dtype=self.time_embedding[0].weight.dtype)
        temb = self.time_embedding(emb)
        proj = self.time_projection(temb).unflatten(2, (6, -1))
        return temb, proj

    def _action_temb(self, timesteps, apf):
        """timesteps: [B, F] -> (temb [B, F*apf, dim], proj [B, F*apf, 6, dim]).

        `temb` is the raw time embedding consumed by Action_Head's AdaLN
        modulation (mirrors model.py: action_head(x, cat_ids, action_temb));
        `proj` is the 6-chunk projection consumed by the transformer blocks.
        """
        ts = torch.repeat_interleave(timesteps, apf, dim=1)
        emb = sinusoidal_embedding_1d(self.freq_dim, ts).to(
            device=timesteps.device, dtype=self.action_time_embedding[0].weight.dtype)
        temb = self.action_time_embedding(emb)
        proj = self.action_time_projection(temb).unflatten(2, (6, -1))
        return temb, proj

    def _state_temb(self, timesteps):
        """timesteps: [B, F_state] -> proj [B, F_state, 6, dim]"""
        emb = sinusoidal_embedding_1d(self.freq_dim, timesteps).to(
            device=timesteps.device, dtype=self.state_time_embedding[0].weight.dtype)
        temb = self.state_time_embedding(emb)
        proj = self.state_time_projection(temb).unflatten(2, (6, -1))
        return proj

    def _build_rope(self, grid_ids):
        """grid_ids: [B, 3, L] -> rotary_emb [B, L, 1, head_dim/2] complex."""
        return self.rope(grid_ids)[:, :, None]

    # ------------------------------------------------------------------
    # Grid id helpers (for RoPE)
    # ------------------------------------------------------------------

    @staticmethod
    def _latent_grid_id(F_latent, H_latent, W_latent, patch_size, f_shift=0,
                        device='cuda'):
        from skyrobot.train.utils import get_mesh_id
        pf, ph, pw = patch_size
        return get_mesh_id(
            F_latent // pf, H_latent // ph, W_latent // pw,
            f_w=1, f_shift=f_shift, action=False
        ).to(device)

    @staticmethod
    def _action_grid_id(F, apf, f_shift=0, device='cuda'):
        from skyrobot.train.utils import get_mesh_id
        return get_mesh_id(F, apf, 1, f_w=1, f_shift=f_shift,
                           action=False).to(device)

    @staticmethod
    def _state_grid_id(F_state, f_shift=0, device='cuda'):
        from skyrobot.train.utils import get_mesh_id
        return get_mesh_id(F_state, 1, 1, f_w=1, f_shift=f_shift,
                           action=False).to(device)

    # ------------------------------------------------------------------
    # Core inference primitives
    # ------------------------------------------------------------------

    def _run_blocks_append(self, hidden, text_emb, temb_proj, rotary_emb,
                           kv_cache):
        """Pass clean tokens through all layers, updating KV cache."""
        for i, block in enumerate(self.blocks):
            hidden = block.forward_append(
                hidden, text_emb, temb_proj, rotary_emb, kv_cache, i)
        return hidden

    def _run_blocks_denoise(self, hidden, text_emb, temb_proj, rotary_emb,
                            kv_cache):
        """Pass noisy tokens through all layers using cached KV."""
        for i, block in enumerate(self.blocks):
            hidden = block.forward_denoise(
                hidden, text_emb, temb_proj, rotary_emb, kv_cache, i)
        return hidden

    # ------------------------------------------------------------------
    # Step-level API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def init_cache(self, context_latent, state_0, text_emb, H_latent,
                   W_latent, kv_cache, cat_ids):
        text_embed = self._embed_text(text_emb)            # [B, text_len, dim]

        # 1. Context (frame 0 latent) alone.
        self._append_clean_latent_only(
            context_latent, text_embed, frame_idx=0,
            H_latent=H_latent, W_latent=W_latent, kv_cache=kv_cache)

        # 2. state_0 alone, after context is in the cache.
        self._append_clean_state_only(
            state_0, text_embed, frame_idx=0, kv_cache=kv_cache,
            cat_ids=cat_ids)

        return text_embed

    @torch.no_grad()
    def denoise_action_step(self, noisy_action, timestep, text_emb,
                            frame_idx, kv_cache, cat_ids):
        B = noisy_action.shape[0]
        device = noisy_action.device
        apf = noisy_action.shape[3]

        action_embed = self._embed_action(noisy_action, cat_ids)  # [B, apf, dim]

        # Ensure timestep is on the correct device (scheduler.timesteps is CPU)
        if isinstance(timestep, (int, float)):
            ts = torch.full((B, 1), timestep, device=device)
        elif isinstance(timestep, torch.Tensor):
            ts = timestep.to(device).reshape(1).expand(B).view(B, 1)
        else:
            ts = torch.full((B, 1), float(timestep), device=device)
        action_temb, action_temb_proj = self._action_temb(ts, apf)

        # Training: action f_shift=0, frames [0..F-1]; frame_idx is 1-based → f_shift = frame_idx - 1
        action_grid = self._action_grid_id(
            1, apf, f_shift=frame_idx - 1, device=device)
        grid_ids = action_grid[None].expand(B, -1, -1)
        rotary_emb = self._build_rope(grid_ids)

        hidden = self._run_blocks_denoise(
            action_embed, text_emb, action_temb_proj, rotary_emb, kv_cache)

        # action_head needs the raw time emb `e` for AdaLN modulation
        # (mirrors model.py: action_head(x, cat_ids, action_temb)).
        pred = self.action_head(hidden, cat_ids, action_temb)  # [B, apf, action_dim]
        pred = rearrange(pred, 'b (f h w) c -> b c f h w', f=1, h=apf, w=1)
        return pred

    @torch.no_grad()
    def denoise_latent_step(self, noisy_latent, timestep, text_emb, frame_idx,
                            H_latent, W_latent, kv_cache, cat_ids):

        B = noisy_latent.shape[0]
        device = noisy_latent.device
        ph, pw = self.patch_size[1], self.patch_size[2]
        h, w = H_latent // ph, W_latent // pw

        lat_embed = self._embed_latent(noisy_latent)  # [B, h*w, dim]

        if isinstance(timestep, (int, float)):
            ts = torch.full((B, 1), timestep, device=device)
        elif isinstance(timestep, torch.Tensor):
            ts = timestep.to(device).reshape(1).expand(B).view(B, 1)
        else:
            ts = torch.full((B, 1), float(timestep), device=device)
        lat_temb, lat_temb_proj = self._latent_temb(ts, H_latent, W_latent)

        # Same f_shift as `append_clean_latent_state` for this frame.
        lat_grid = self._latent_grid_id(
            1, H_latent, W_latent, self.patch_size,
            f_shift=frame_idx, device=device)
        grid_ids = lat_grid[None].expand(B, -1, -1)
        rotary_emb = self._build_rope(grid_ids)

        hidden = self._run_blocks_denoise(
            lat_embed, text_emb, lat_temb_proj, rotary_emb, kv_cache)

        # Head applies AdaLN modulation conditioned on the latent time emb,
        # then projects to prod(patch) * out_dim per token.
        pred = self.head(hidden, lat_temb)  # [B, h*w, prod(patch)*out_dim]
        pred = rearrange(
            pred, 'b (f h w) (pf ph pw c) -> b c (f pf) (h ph) (w pw)',
            f=1, h=h, w=w,
            pf=self.patch_size[0], ph=ph, pw=pw,
        )
        return pred

    @torch.no_grad()
    def append_clean_action(self, action_clean, text_emb, frame_idx, kv_cache,
                            cat_ids):
        B = action_clean.shape[0]
        device = action_clean.device
        apf = action_clean.shape[3]

        action_embed = self._embed_action(action_clean, cat_ids)
        ts = torch.zeros(B, 1, device=device)
        _, action_temb_proj = self._action_temb(ts, apf)

        # Training: action f_shift=0, frames [0..F-1]; frame_idx is 1-based → f_shift = frame_idx - 1
        action_grid = self._action_grid_id(
            1, apf, f_shift=frame_idx - 1, device=device)
        grid_ids = action_grid[None].expand(B, -1, -1)
        rotary_emb = self._build_rope(grid_ids)

        self._run_blocks_append(
            action_embed, text_emb, action_temb_proj, rotary_emb, kv_cache)

    @torch.no_grad()
    def _append_clean_latent_only(self, latent_clean, text_emb,
                                  frame_idx, H_latent, W_latent, kv_cache):
        B = latent_clean.shape[0]
        device = latent_clean.device

        lat_embed = self._embed_latent(latent_clean)
        ts_lat = torch.zeros(B, 1, device=device)
        _, lat_temb_proj = self._latent_temb(ts_lat, H_latent, W_latent)
        lat_grid = self._latent_grid_id(
            1, H_latent, W_latent, self.patch_size,
            f_shift=frame_idx, device=device)
        grid_ids = lat_grid[None].expand(B, -1, -1)
        rotary_emb = self._build_rope(grid_ids)

        self._run_blocks_append(
            lat_embed, text_emb, lat_temb_proj, rotary_emb, kv_cache)

    @torch.no_grad()
    def _append_clean_state_only(self, state, text_emb, frame_idx, kv_cache,
                                 cat_ids):
        B = state.shape[0]
        device = state.device

        state_embed = self._embed_state(state, cat_ids)
        ts_state = torch.zeros(B, 1, device=device)
        state_temb_proj = self._state_temb(ts_state)
        state_grid = self._state_grid_id(1, f_shift=frame_idx, device=device)
        grid_ids = state_grid[None].expand(B, -1, -1)
        rotary_emb = self._build_rope(grid_ids)

        self._run_blocks_append(
            state_embed, text_emb, state_temb_proj, rotary_emb, kv_cache)

    @torch.no_grad()
    def append_clean_latent_state(self, latent_clean, state, text_emb,
                                  frame_idx, H_latent, W_latent, kv_cache,
                                  cat_ids):
        self._append_clean_latent_only(
            latent_clean, text_emb, frame_idx, H_latent, W_latent, kv_cache)
        self._append_clean_state_only(
            state, text_emb, frame_idx, kv_cache, cat_ids)

    # ------------------------------------------------------------------
    # Online interactive session (env loop)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def create_session(self, first_frame, state_0, text_emb,
                       scheduler, num_inference_steps=16, max_frames=32,
                       embodiment_id=None, active_action_dim: int | None = None,
                       cond_text_kpm=None, uncond_text_emb=None,
                       uncond_text_kpm=None, cfg_scale: float = 1.0):
        return OnlineSession(
            model=self,
            first_frame=first_frame,
            state_0=state_0,
            text_emb=text_emb,
            scheduler=scheduler,
            num_inference_steps=num_inference_steps,
            max_frames=max_frames,
            embodiment_id=embodiment_id,
            active_action_dim=active_action_dim,
            cond_text_kpm=cond_text_kpm,
            uncond_text_emb=uncond_text_emb,
            uncond_text_kpm=uncond_text_kpm,
            cfg_scale=cfg_scale,
        )


class OnlineSession:
    """Stateful online inference session with KV cache.

    Windowed inference: KV cache holds at most `max_frames` (default 16)
    latent frames. When the window is full, the cache is reset and the
    latest observation becomes the new context frame.

    Lifecycle:
      1. __init__: encode first frame + state_0 → KV cache
      2. predict_next_action(): denoise action for current frame
      3. observe(frames, state): VAE encode new frames → append to cache
      4. Repeat 2-3. Cache auto-resets every max_frames.
    """

    def __init__(self, model, first_frame, state_0, text_emb,
                 scheduler, num_inference_steps=16, max_frames=32,
                 embodiment_id=None, active_action_dim: int | None = None,
                 cond_text_kpm=None, uncond_text_emb=None,
                 uncond_text_kpm=None, cfg_scale: float = 1.0):
        self.model : SkyrobotWAMForInference = model
        self.scheduler = scheduler
        self.num_inference_steps = num_inference_steps
        self.max_frames = max_frames
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.active_action_dim = (
            active_action_dim if active_action_dim is not None
            else model.action_dim)
        assert 0 < self.active_action_dim <= model.action_dim, (
            f"active_action_dim={self.active_action_dim} must be in "
            f"(0, {model.action_dim}]")

        # Resolve embodiment → category slot once; reused for every forward.
        self.cat_ids = model.resolve_cat_ids(
            embodiment_id, batch_size=1, device=self.device)

        # Setup scheduler
        scheduler.set_timesteps(num_inference_steps)

        # Save text embedding (reused across resets)
        text_emb = text_emb.to(self.device).to(self.dtype)
        self._raw_text_emb = text_emb

        # Classifier-free guidance state. The KV cache is built with the COND
        # text (init_cache/observe); CFG is applied only on the action-flow
        # prediction at the denoise step (cond/uncond cross-text, shared
        # cache). cfg_scale==1.0 -> single pass (legacy behaviour).
        self.cfg_scale = float(cfg_scale)
        self._cond_kpm = cond_text_kpm
        self._uncond_kpm = uncond_text_kpm
        if uncond_text_emb is not None and self.cfg_scale != 1.0:
            u = uncond_text_emb.to(self.device).to(self.dtype)
            self._uncond_text_embed = self.model._embed_text(u)  # [B,text_len,dim]
        else:
            self._uncond_text_embed = None

        # Stream-encode the first frame (resets VAE feat cache)
        if not (isinstance(first_frame, torch.Tensor) and first_frame.ndim == 4):
            raise ValueError(
                "first_frame must be raw video tensor [F, C, H, W] "
                "for streaming VAE encode.")
        first_frame = first_frame.to(self.device).float()
        context_latent = self._stream_encode(
            first_frame, reset_cache=True, is_first_chunk=True)

        self.H_latent = context_latent.shape[3]
        self.W_latent = context_latent.shape[4]

        # Init
        self.kv_cache = None
        self.text_embed = None
        self.frame_idx = 0  # frames within current window (0 = just context)

        # Start first window
        state_0_t = self._to_state_tensor(state_0)
        self._init_window(context_latent, state_0_t)

    def _stream_encode(self, frames, reset_cache: bool = False,
                       is_first_chunk: bool = False) -> torch.Tensor:
        model = self.model
        # Resize + [0,1] → [-1,1] + permute to [C, F, H, W]
        frames_resized = model._resize_frames(frames)
        frames_vae = frames_resized.mul(2).sub(1).permute(1, 0, 2, 3)
        # Streaming encode (preserves feat cache across calls)
        mu = model.vae.encode_streaming(
            frames_vae, reset_cache=reset_cache, is_first_chunk=is_first_chunk)
        # mu: [C_lat, T_lat, H_lat, W_lat] → [1, C_lat, T_lat, H_lat, W_lat]
        return mu.unsqueeze(0).to(self.dtype)

    def _to_state_tensor(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        if state.ndim == 1:
            state = state.view(1, -1, 1, 1, 1)
        state = state.to(self.device).to(self.dtype)
        cur_c = state.shape[1]
        target_c = self.model.state_dim
        if cur_c < target_c:
            pad = torch.zeros(
                state.shape[0], target_c - cur_c, *state.shape[2:],
                device=state.device, dtype=state.dtype)
            state = torch.cat([state, pad], dim=1)
        elif cur_c > target_c:
            raise ValueError(
                f"state has {cur_c} channels but model.state_dim={target_c}"
            )
        return state

    def _init_window(self, context_latent, state_0):
        """(Re-)initialize KV cache with context frame + state_0."""
        self.kv_cache = KVCache(self.model.num_layers)
        self.frame_idx = 0
        self.text_embed = self.model.init_cache(
            context_latent, state_0, self._raw_text_emb,
            self.H_latent, self.W_latent, self.kv_cache, self.cat_ids)

    @torch.no_grad()
    def predict_next_action(self):
        self.frame_idx += 1
        model = self.model
        apf = model.action_per_latent
        ad = self.active_action_dim
        B = 1

        noisy_action = torch.zeros(
            B, model.action_dim, 1, apf, 1,
            device=self.device, dtype=self.dtype)
        noisy_action[:, :ad] = torch.randn(
            B, ad, 1, apf, 1, device=self.device, dtype=self.dtype)

        from skyrobot.models.attention import WanCrossAttention
        use_cfg = (self.cfg_scale != 1.0
                   and self._uncond_text_embed is not None)

        for t in self.scheduler.timesteps:
            # Conditional pass (real instruction). Install the cond text mask
            # so cross-attn hard-excludes pad (parity with training).
            WanCrossAttention.infer_text_kpm = self._cond_kpm
            pred_c = model.denoise_action_step(
                noisy_action, t, self.text_embed,
                self.frame_idx, self.kv_cache, self.cat_ids)
            if use_cfg:
                # Unconditional pass (neutral instruction), shared KV cache.
                WanCrossAttention.infer_text_kpm = self._uncond_kpm
                pred_u = model.denoise_action_step(
                    noisy_action, t, self._uncond_text_embed,
                    self.frame_idx, self.kv_cache, self.cat_ids)
                pred = pred_u + self.cfg_scale * (pred_c - pred_u)
            else:
                pred = pred_c
            noisy_action = self.scheduler.step(
                pred, t, noisy_action).to(
                    device=self.device, dtype=self.dtype)
            if ad < model.action_dim:
                noisy_action[:, ad:] = 0

        # Append denoised action as clean to cache (full padded shape — the
        # cache stores model-internal tokens, not the truncated output).
        # Cache is built with COND text -> restore cond mask first.
        WanCrossAttention.infer_text_kpm = self._cond_kpm
        model.append_clean_action(
            noisy_action, self.text_embed,
            self.frame_idx, self.kv_cache, self.cat_ids)

        # Return only the active channels so the caller can directly
        # denormalize with its (active_action_dim,)-sized mean/std stats.
        action = noisy_action[0, :ad, 0, :, 0]   # [active_action_dim, apf]
        action = action.permute(1, 0)             # [apf, active_action_dim]
        return action.float().cpu().numpy()

    @torch.no_grad()
    def observe(self, frames, state):
        state_t = self._to_state_tensor(state)

        if isinstance(frames, np.ndarray):
            frames = torch.from_numpy(frames)
        frames = frames.to(self.device).float()

        # Check if window is full → reset
        if self.frame_idx >= self.max_frames:
            # New window: use last frame of this chunk as new context.
            # Reset VAE feat cache so it behaves like a new video.
            new_context = self._stream_encode(
                frames[-1:], reset_cache=True, is_first_chunk=True)
            self._init_window(new_context, state_t)
        else:
            # Stream-encode this chunk (4 raw frames → 1 latent), feat cache preserved
            latent = self._stream_encode(frames, reset_cache=False, is_first_chunk=False)
            # Append latent + state to cache within current window
            self.model.append_clean_latent_state(
                latent, state_t, self.text_embed,
                self.frame_idx, self.H_latent, self.W_latent,
                self.kv_cache, self.cat_ids)
