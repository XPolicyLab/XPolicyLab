"""Hand-rolled forward-mode JVP for LTX-DiT.

Bypasses torch.func.jvp by manually propagating (primal, tangent) pairs
through each layer.  Mathematically equivalent to torch.func.jvp but avoids:
  1. SDPA fallback to math backend (flash attn doesn't support forward AD)
  2. ~10K micro-kernel launches from forward-AD tape
  3. torch.compile + forward-AD view-op bug (PT 2.7/2.8)

Two backends:
  "fast"   — stacked GEMM trick for linear layers (max throughput)
  "strict" — separate GEMMs (bit-exact match with torch.func.jvp)
Both use bf16 materialized attention (matching SDPA math backend numerics).

Reference: DowneyFlyfan/iMFVideo mla_jvp_fast.py (MLA arch), adapted for
standard MHA LTX-DiT.  Sequence length is ~168 tokens so materialized
attention weights are used instead of a flash-JVP Triton kernel.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from einops import rearrange
# ---------------------------------------------------------------------------
# Primitive JVP functions
# ---------------------------------------------------------------------------

def linear_jvp(
    x: torch.Tensor,
    dx: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Linear layer JVP — separate GEMMs (bit-exact with functorch)."""
    y = F.linear(x, weight, bias)
    dy = F.linear(dx, weight)
    return y, dy


def linear_jvp_stacked(
    x: torch.Tensor,
    dx: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Linear layer JVP — stacked batch GEMM (one kernel launch, max speed).

    Concatenates [x; dx] along batch dim and does a single F.linear call.
    Slightly different bf16 accumulation pattern vs two separate GEMMs.
    """
    stacked = torch.cat([x, dx], dim=0)
    out = F.linear(stacked, weight)
    y, dy = out.chunk(2, dim=0)
    if bias is not None:
        y = y + bias
    return y, dy


def rmsnorm_jvp(
    x: torch.Tensor,
    dx: torch.Tensor,
    weight: Optional[torch.Tensor],
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm JVP.  Matches functorch: norm in fp32 (PyTorch RMSNorm casts
    internally), tangent formula in the same precision."""
    x_fp32 = x.float()
    dx_fp32 = dx.float()
    variance = x_fp32.pow(2).mean(-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    x_norm = x_fp32 * inv_rms

    dot = (x_fp32 * dx_fp32).mean(-1, keepdim=True)
    dx_norm = inv_rms * (dx_fp32 - x_norm * dot * inv_rms)

    if weight is not None:
        # Match diffusers RMSNorm.forward() precision path exactly:
        # cast to input dtype BEFORE weight multiply (line 560 in normalization.py)
        x_norm_cast = x_norm.to(x.dtype)
        dx_norm_cast = dx_norm.to(dx.dtype)
        y = x_norm_cast * weight
        dy = dx_norm_cast * weight
    else:
        y = x_norm.to(x.dtype)
        dy = dx_norm.to(dx.dtype)
    return y, dy


def layernorm_jvp(
    x: torch.Tensor,
    dx: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """LayerNorm JVP.  Computes in fp32."""
    x_fp32 = x.float()
    dx_fp32 = dx.float()
    D = x_fp32.shape[-1]

    mean_x = x_fp32.mean(-1, keepdim=True)
    centered = x_fp32 - mean_x
    var_x = centered.pow(2).mean(-1, keepdim=True)
    inv_std = torch.rsqrt(var_x + eps)
    x_norm = centered * inv_std

    d_mean = dx_fp32.mean(-1, keepdim=True)
    dot = (centered * dx_fp32).mean(-1, keepdim=True)
    dx_norm = inv_std * (dx_fp32 - d_mean - x_norm * dot * inv_std)

    if weight is not None:
        w = weight.float()
        y = (x_norm * w).to(x.dtype)
        dy = (dx_norm * w).to(dx.dtype)
        if bias is not None:
            y = y + bias
    else:
        y = x_norm.to(x.dtype)
        dy = dx_norm.to(dx.dtype)
    return y, dy


def gelu_tanh_jvp(
    x: torch.Tensor,
    dx: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GELU(tanh approximation) JVP.  Computed in fp32 to match PyTorch's F.gelu
    CUDA kernel which internally upcasts bf16→fp32."""
    k = math.sqrt(2.0 / math.pi)
    c = 0.044715
    x_fp32 = x.float()
    dx_fp32 = dx.float()
    u = k * (x_fp32 + c * x_fp32.pow(3))
    tanh_u = torch.tanh(u)
    y = (0.5 * x_fp32 * (1.0 + tanh_u)).to(x.dtype)

    sech2_u = 1.0 - tanh_u.pow(2)
    du_dx = k * (1.0 + 3.0 * c * x_fp32.pow(2))
    gelu_deriv = 0.5 * (1.0 + tanh_u) + 0.5 * x_fp32 * sech2_u * du_dx
    dy = (gelu_deriv * dx_fp32).to(dx.dtype)
    return y, dy


def silu_jvp(
    x: torch.Tensor,
    dx: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """SiLU(x) = x * sigmoid(x) JVP.  Computed in fp32 to match PyTorch's
    internal sigmoid precision for bf16 inputs."""
    x_fp32 = x.float()
    dx_fp32 = dx.float()
    sig = torch.sigmoid(x_fp32)
    y = (x_fp32 * sig).to(x.dtype)
    dy = (sig * (1.0 + x_fp32 * (1.0 - sig)) * dx_fp32).to(dx.dtype)
    return y, dy


def rotary_emb_jvp(
    x: torch.Tensor,
    dx: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """RoPE JVP — linear in x, so tangent uses same rotation."""
    x_real, x_imag = x.unflatten(2, (-1, 2)).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(2)
    y = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

    dx_real, dx_imag = dx.unflatten(2, (-1, 2)).unbind(-1)
    dx_rotated = torch.stack([-dx_imag, dx_real], dim=-1).flatten(2)
    dy = (dx.float() * cos + dx_rotated.float() * sin).to(dx.dtype)
    return y, dy


def sdpa_jvp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    scale: float,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Attention JVP with self-consistent primal and tangent.

    Matches SDPA math backend exactly: pre-scales both Q and K by sqrt(scale)
    before the matmul (the C++ kernel applies scale this way, which changes
    floating-point rounding relative to post-multiply).
    """
    orig_dtype = q.dtype
    sqrt_scale = math.sqrt(scale)
    q, k, v = q.float(), k.float(), v.float()
    dq, dk, dv = dq.float(), dk.float(), dv.float()

    q_s = q * sqrt_scale
    k_s = k * sqrt_scale
    dq_s = dq * sqrt_scale
    dk_s = dk * sqrt_scale

    s = torch.matmul(q_s, k_s.transpose(-2, -1))
    if mask is not None:
        s = s + mask.float()
    p = torch.softmax(s, dim=-1)

    o = torch.matmul(p, v)

    ds = (torch.matmul(dq_s, k_s.transpose(-2, -1))
          + torch.matmul(q_s, dk_s.transpose(-2, -1)))
    dp = p * (ds - (p * ds).sum(dim=-1, keepdim=True))

    do = torch.matmul(dp, v) + torch.matmul(p, dv)
    return o.to(orig_dtype), do.to(orig_dtype)


def sdpa_jvp_cross(
    q: torch.Tensor,
    dq: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cross-attention JVP (dK=0, dV=0).
    Matches SDPA math backend: pre-scales both Q and K by sqrt(scale)."""
    orig_dtype = q.dtype
    sqrt_scale = math.sqrt(scale)
    q, dq = q.float(), dq.float()
    k, v = k.float(), v.float()

    q_s = q * sqrt_scale
    k_s = k * sqrt_scale
    dq_s = dq * sqrt_scale

    s = torch.matmul(q_s, k_s.transpose(-2, -1))
    if mask is not None:
        s = s + mask.float()
    p = torch.softmax(s, dim=-1)

    o = torch.matmul(p, v)

    ds = torch.matmul(dq_s, k_s.transpose(-2, -1))
    dp = p * (ds - (p * ds).sum(dim=-1, keepdim=True))

    do = torch.matmul(dp, v)
    return o.to(orig_dtype), do.to(orig_dtype)


# ---------------------------------------------------------------------------
# Block-level JVP
# ---------------------------------------------------------------------------

def _attention_jvp(
    attn,
    x: torch.Tensor,
    dx: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor],
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]],
    backend: str,
    attention_mask: Optional[torch.Tensor],
    n_view: int,
    cross_view_attn: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Full attention sublayer JVP (QKV proj → norm → RoPE → SDPA → out proj)."""

    _linear = linear_jvp_stacked if backend == "fast" else linear_jvp
    is_cross = encoder_hidden_states is not None
    kv_input = encoder_hidden_states if is_cross else x
    dkv_input = torch.zeros_like(kv_input) if is_cross else dx

    # QKV projections
    q, dq = _linear(x, dx, attn.to_q.weight, attn.to_q.bias)
    k, dk = _linear(kv_input, dkv_input, attn.to_k.weight, attn.to_k.bias)
    v, dv = _linear(kv_input, dkv_input, attn.to_v.weight, attn.to_v.bias)

    # QK norm
    norm_q_weight = getattr(attn.norm_q, 'weight', None)
    norm_k_weight = getattr(attn.norm_k, 'weight', None)
    norm_q_eps = attn.norm_q.eps if hasattr(attn.norm_q, 'eps') else 1e-5
    norm_k_eps = attn.norm_k.eps if hasattr(attn.norm_k, 'eps') else 1e-5
    q, dq = rmsnorm_jvp(q, dq, norm_q_weight, norm_q_eps)
    k, dk = rmsnorm_jvp(k, dk, norm_k_weight, norm_k_eps)

    # RoPE (self-attention only)
    if image_rotary_emb is not None:
        cos, sin = image_rotary_emb
        batch_size = q.shape[0]
        if cos.shape[0] == 1 and batch_size > 1:
            cos = cos.repeat(batch_size, 1, 1)
        if sin.shape[0] == 1 and batch_size > 1:
            sin = sin.repeat(batch_size, 1, 1)
        q, dq = rotary_emb_jvp(q, dq, cos, sin)
        k, dk = rotary_emb_jvp(k, dk, cos, sin)

        if cross_view_attn:
            q = rearrange(q, '(b v) l c -> b (v l) c', v=n_view)
            dq = rearrange(dq, '(b v) l c -> b (v l) c', v=n_view)
            k = rearrange(k, '(b v) l c -> b (v l) c', v=n_view)
            dk = rearrange(dk, '(b v) l c -> b (v l) c', v=n_view)
            v = rearrange(v, '(b v) l c -> b (v l) c', v=n_view)
            dv = rearrange(dv, '(b v) l c -> b (v l) c', v=n_view)
    else:
        # Cross-attention: Q merges views
        q = rearrange(q, '(b v) l c -> b (v l) c', v=n_view)
        dq = rearrange(dq, '(b v) l c -> b (v l) c', v=n_view)

    # Reshape to multi-head: (B, S, H*D) -> (B, H, S, D)
    q = q.unflatten(2, (attn.heads, -1)).transpose(1, 2)
    dq = dq.unflatten(2, (attn.heads, -1)).transpose(1, 2)
    k = k.unflatten(2, (attn.heads, -1)).transpose(1, 2)
    dk = dk.unflatten(2, (attn.heads, -1)).transpose(1, 2)
    v = v.unflatten(2, (attn.heads, -1)).transpose(1, 2)
    dv = dv.unflatten(2, (attn.heads, -1)).transpose(1, 2)

    # Prepare attention mask
    batch_size_q = q.shape[0]
    seq_len = q.shape[2]
    if attention_mask is not None:
        if cross_view_attn:
            real_batch = batch_size_q
            attention_mask = attention_mask.repeat(real_batch, n_view, n_view)
        elif image_rotary_emb is not None:
            attention_mask = attention_mask.repeat(batch_size_q, 1, 1)
        if attention_mask.ndim == 3:
            attention_mask = attention_mask.unsqueeze(1)

    # SDPA JVP
    scale = attn.scale
    if is_cross:
        o, do = sdpa_jvp_cross(q, dq, k, v, scale, attention_mask)
    else:
        o, do = sdpa_jvp(q, k, v, dq, dk, dv, scale, attention_mask)

    # Reshape back: (B, H, S, D) -> (B, S, H*D)
    o = o.transpose(1, 2).flatten(2, 3).to(q.dtype)
    do = do.transpose(1, 2).flatten(2, 3).to(dq.dtype)

    # Undo cross-view merge
    if cross_view_attn or image_rotary_emb is None:
        o = rearrange(o, 'b (v l) c -> (b v) l c', v=n_view)
        do = rearrange(do, 'b (v l) c -> (b v) l c', v=n_view)

    # Output projection
    out_linear = attn.to_out[0]
    o, do = _linear(o, do, out_linear.weight, out_linear.bias)

    return o, do


def ltx_block_jvp(
    block,
    x: torch.Tensor,
    dx: torch.Tensor,
    temb: torch.Tensor,
    dtemb: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]],
    encoder_attention_mask: Optional[torch.Tensor],
    n_view: int,
    cross_view_attn: bool,
    backend: str = "fast",
    self_attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One LTXVideoTransformerBlock JVP."""
    B = x.shape[0]

    # --- AdaLN parameters ---
    num_ada = block.scale_shift_table.shape[0]  # 6
    sst = block.scale_shift_table  # (6, D)

    temb_reshape = temb.reshape(B, temb.size(1), num_ada, -1)
    dtemb_reshape = dtemb.reshape(B, dtemb.size(1), num_ada, -1)

    ada = sst[None, None] + temb_reshape  # (B, L, 6, D)
    dada = dtemb_reshape  # scale_shift_table has zero tangent

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada.unbind(dim=2)
    dshift_msa, dscale_msa, dgate_msa, dshift_mlp, dscale_mlp, dgate_mlp = dada.unbind(dim=2)

    # --- Self-attention sublayer ---
    norm1_weight = getattr(block.norm1, 'weight', None)
    norm1_eps = block.norm1.eps if hasattr(block.norm1, 'eps') else 1e-6
    nx, dnx = rmsnorm_jvp(x, dx, norm1_weight, norm1_eps)

    # AdaLN modulation
    nx_mod = nx * (1.0 + scale_msa) + shift_msa
    dnx_mod = dnx * (1.0 + scale_msa) + nx * dscale_msa + dshift_msa

    # Self-attention
    attn_out, dattn_out = _attention_jvp(
        block.attn1, nx_mod, dnx_mod,
        encoder_hidden_states=None,
        image_rotary_emb=image_rotary_emb,
        backend=backend,
        attention_mask=self_attention_mask,
        n_view=n_view,
        cross_view_attn=cross_view_attn,
    )

    # Gated residual
    x = x + attn_out * gate_msa
    dx = dx + dattn_out * gate_msa + attn_out * dgate_msa

    # --- Cross-attention sublayer ---
    cattn_out, dcattn_out = _attention_jvp(
        block.attn2, x, dx,
        encoder_hidden_states=encoder_hidden_states,
        image_rotary_emb=None,
        backend=backend,
        attention_mask=encoder_attention_mask,
        n_view=n_view,
        cross_view_attn=False,
    )
    # Cross-attention has no gate
    x = x + cattn_out
    dx = dx + dcattn_out

    # --- FFN sublayer ---
    norm2_weight = getattr(block.norm2, 'weight', None)
    norm2_eps = block.norm2.eps if hasattr(block.norm2, 'eps') else 1e-6
    nx2, dnx2 = rmsnorm_jvp(x, dx, norm2_weight, norm2_eps)

    # AdaLN modulation
    nx2_mod = nx2 * (1.0 + scale_mlp) + shift_mlp
    dnx2_mod = dnx2 * (1.0 + scale_mlp) + nx2 * dscale_mlp + dshift_mlp

    # FeedForward: Linear -> GELU(tanh) -> Dropout(0) -> Linear
    _linear = linear_jvp_stacked if backend == "fast" else linear_jvp
    ff = block.ff
    ff_linear1 = ff.net[0].proj  # GELU wraps a linear
    ff_linear2 = ff.net[2]       # final linear (nn.Linear)

    h, dh = _linear(nx2_mod, dnx2_mod, ff_linear1.weight, ff_linear1.bias)
    h, dh = gelu_tanh_jvp(h, dh)
    h, dh = _linear(h, dh, ff_linear2.weight, ff_linear2.bias)

    # Gated residual
    x = x + h * gate_mlp
    dx = dx + dh * gate_mlp + h * dgate_mlp

    return x, dx


# ---------------------------------------------------------------------------
# Model-level JVP
# ---------------------------------------------------------------------------

def ltx_dit_jvp(
    model,
    z: torch.Tensor,
    dz: torch.Tensor,
    timesteps: torch.Tensor,
    d_timesteps: torch.Tensor,
    *,
    backend: str = "fast",
    timestep_r: Optional[torch.Tensor] = None,
    encoder_hidden_states: torch.Tensor = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    video_attention_mask: Optional[torch.Tensor] = None,
    video_rope_coords: Optional[torch.Tensor] = None,
    n_view: int = 1,
    num_frames: int = 1,
    height: int = 1,
    width: int = 1,
    view_ids: Optional[torch.Tensor] = None,
    precomputed_vlm_hidden_states: Optional[List[torch.Tensor]] = None,
    vlm_attention_mask: Optional[torch.Tensor] = None,
    goal_mode_mask: Optional[torch.Tensor] = None,
    num_frames_wo_dropout: int = 0,
    cond_latents: Optional[torch.Tensor] = None,
    cond_mode_ids: Optional[torch.Tensor] = None,
    compute_v_head: bool = False,
) -> torch.Tensor:
    """Full LTX-DiT forward-mode JVP.

    Args:
        model: LTXVideoTransformer3DModel (unwrapped, not DDP)
        z, dz: latent primal and tangent (B, S, C)
        timesteps, d_timesteps: per-token timestep primal and tangent (B, S)
        timestep_r: constant r-endpoint (no tangent)
        encoder_hidden_states: text/VLM conditioning (no tangent)
        ... other kwargs match model.forward()

    Returns:
        du_dt: tangent of the model output, same shape as z
    """
    batch_size = z.shape[0]
    t2v_prefix_tokens = 0

    # --- RoPE (no tangent — depends only on position, not on z or t) ---
    if bool(getattr(model, "t2v_mode", False)):
        input_num_frames = int(num_frames)
        if input_num_frames < 2:
            raise ValueError(
                f"t2v_mode requires memory + future frames, got num_frames={num_frames}."
            )
        if (
            cond_mode_ids is not None
        ):
            raise NotImplementedError(
                "t2v_mode is incompatible with frame-concat conditioning."
            )
        if video_attention_mask is not None:
            raise NotImplementedError(
                "t2v_mode currently supports the dense unmasked video path only."
            )

        t2v_prefix_tokens = int(height) * int(width)
        expected_tokens = input_num_frames * t2v_prefix_tokens
        if z.ndim != 3 or z.shape[1] != expected_tokens:
            raise ValueError(
                "t2v_mode expects a dense frame-major token sequence: "
                f"got z={tuple(z.shape)}, expected token length "
                f"{input_num_frames}*{height}*{width}={expected_tokens}."
            )
        if dz.shape != z.shape:
            raise ValueError(
                f"t2v_mode expects dz to match z, got dz={tuple(dz.shape)} "
                f"and z={tuple(z.shape)}."
            )

        def _drop_first_timestep_frame(value, name):
            if value is None or value.ndim == 0:
                return value
            if value.ndim == 1:
                if value.shape[0] != batch_size:
                    raise ValueError(
                        f"{name} has shape {tuple(value.shape)}; expected one value "
                        f"per batch item ({batch_size})."
                    )
                return value
            if value.shape[-1] == expected_tokens:
                return value[..., t2v_prefix_tokens:]
            if value.shape[-1] == input_num_frames:
                return value[..., 1:]
            if value.shape[-1] == 1:
                return value
            raise ValueError(
                f"t2v_mode cannot align {name} shape {tuple(value.shape)} with "
                f"{input_num_frames} frames / {expected_tokens} tokens."
            )

        if video_rope_coords is not None and (
            video_rope_coords.ndim != 3
            or video_rope_coords.shape[1] != expected_tokens
            or video_rope_coords.shape[2] != 3
        ):
            raise ValueError(
                "t2v_mode expects video_rope_coords [batch, dense_tokens, 3], "
                f"got {tuple(video_rope_coords.shape)}."
            )

        full_image_rotary_emb = model.rope(
            z, (1.0, 1.0, 1.0), input_num_frames, height, width,
            video_rope_coords=video_rope_coords,
        )
        image_rotary_emb = tuple(
            value[:, t2v_prefix_tokens:] for value in full_image_rotary_emb
        )
        z = z[:, t2v_prefix_tokens:]
        dz = dz[:, t2v_prefix_tokens:]
        timesteps = _drop_first_timestep_frame(timesteps, "timesteps")
        d_timesteps = _drop_first_timestep_frame(d_timesteps, "d_timesteps")
        timestep_r = _drop_first_timestep_frame(timestep_r, "timestep_r")
        num_frames = input_num_frames - 1
    else:
        image_rotary_emb = model.rope(
            z, (1.0, 1.0, 1.0), num_frames, height, width,
            video_rope_coords=video_rope_coords,
        )

    # --- Attention masks ---
    if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
        encoder_attention_mask = (1 - encoder_attention_mask.to(z.dtype)) * -10000.0
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
    if video_attention_mask is not None and video_attention_mask.ndim == 2:
        video_attention_mask = (1 - video_attention_mask.to(z.dtype)) * -10000.0
        video_attention_mask = video_attention_mask.unsqueeze(0)

    # --- Patch embedding (linear, no activation) ---
    if model.patch_size > 1 or model.patch_size_t > 1:
        raise NotImplementedError(
            "fast JVP for patch_size > 1 not implemented yet"
        )

    proj_in = model.proj_in
    _linear = linear_jvp_stacked if backend == "fast" else linear_jvp
    x, dx = _linear(z, dz, proj_in.weight, proj_in.bias)

    # --- Timestep embedding JVP (hand-rolled, eliminates torch.func.jvp) ---
    # Chain: sinusoidal → Linear → SiLU → Linear → SiLU → Linear(D→6D)
    ts_flat = timesteps.flatten()
    dts_flat = d_timesteps.flatten()

    # Sinusoidal embedding (fixed function, no learnable params)
    time_proj = model.time_embed.emb.time_proj
    t_proj = time_proj(ts_flat)  # (N, 256), float32

    # Tangent of sinusoidal embedding: d(sin(f*t))/dt = cos(f*t)*f,
    # d(cos(f*t))/dt = -sin(f*t)*f.  Reuse sin/cos from forward output.
    half_dim = time_proj.num_channels // 2
    freq = torch.exp(
        -math.log(10000) * torch.arange(
            half_dim, device=ts_flat.device, dtype=torch.float32,
        ) / (half_dim - time_proj.downscale_freq_shift)
    )
    d_inner = dts_flat[:, None].float() * freq[None, :] * time_proj.scale
    if time_proj.flip_sin_to_cos:
        cos_part, sin_part = t_proj[:, :half_dim], t_proj[:, half_dim:]
        d_t_proj = torch.cat([-sin_part * d_inner, cos_part * d_inner], dim=-1)
    else:
        sin_part, cos_part = t_proj[:, :half_dim], t_proj[:, half_dim:]
        d_t_proj = torch.cat([cos_part * d_inner, -sin_part * d_inner], dim=-1)
    if time_proj.num_channels % 2 == 1:
        d_t_proj = F.pad(d_t_proj, (0, 1, 0, 0))

    # Cast to model dtype
    t_proj_h = t_proj.to(dtype=x.dtype)
    d_t_proj_h = d_t_proj.to(dtype=x.dtype)

    # TimestepEmbedding: Linear(256→D) → SiLU → Linear(D→D)
    ts_emb = model.time_embed.emb.timestep_embedder
    h, dh = linear_jvp(t_proj_h, d_t_proj_h, ts_emb.linear_1.weight, ts_emb.linear_1.bias)
    h, dh = silu_jvp(h, dh)
    embedded_timestep, dembedded_timestep = linear_jvp(h, dh, ts_emb.linear_2.weight, ts_emb.linear_2.bias)

    # AdaLayerNormSingle final: SiLU → Linear(D → 6D)
    h2, dh2 = silu_jvp(embedded_timestep, dembedded_timestep)
    temb, dtemb = linear_jvp(h2, dh2, model.time_embed.linear.weight, model.time_embed.linear.bias)

    if model.use_t_and_r_embedding and timestep_r is not None:
        temb_r, embedded_timestep_r = model.time_embed_r(
            timestep_r.flatten(), batch_size=batch_size,
            hidden_dtype=x.dtype,
        )
        temb = temb + temb_r
        embedded_timestep = embedded_timestep + embedded_timestep_r
        # r has zero tangent, so dtemb and dembedded_timestep unchanged

    temb = temb.view(batch_size, -1, temb.size(-1))
    embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))
    dtemb = dtemb.view(batch_size, -1, dtemb.size(-1))
    dembedded_timestep = dembedded_timestep.view(batch_size, -1, dembedded_timestep.size(-1))

    # --- View embedding (no tangent — depends on view_ids, not z or t) ---
    if model.use_view_embed:
        if view_ids is None:
            view_ids = torch.arange(n_view, device=x.device, dtype=torch.long).unsqueeze(0)
        else:
            view_ids = view_ids.to(device=x.device, dtype=torch.long)
        if view_ids.shape[0] == 1 and (batch_size // n_view) > 1:
            view_ids = view_ids.expand(batch_size // n_view, -1)
        oh = F.one_hot(view_ids, model.view_embed.shape[0]).to(dtype=model.view_embed.dtype)
        embedded_view = oh @ model.view_embed
        embedded_view = rearrange(embedded_view, 'b v c -> (b v) c').unsqueeze(1)
        vemb = model.view_ada(embedded_view)
        temb = temb + vemb
        embedded_timestep = embedded_timestep + embedded_view
        # view embedding has zero tangent, dtemb unchanged

    # --- VLM / text conditioning (no tangent) ---
    if model.vlm_interface is not None and precomputed_vlm_hidden_states is not None:
        _fuser_mask = vlm_attention_mask if getattr(model, "vlm_text_tokens_only", False) else None
        enc_hs = model.fuser(precomputed_vlm_hidden_states, attention_mask=_fuser_mask)
        if goal_mode_mask is not None:
            enc_hs, vlm_attention_mask = model._apply_goal_mode_null_text(
                enc_hs, vlm_attention_mask, goal_mode_mask
            )
    elif hasattr(model, 'caption_projection') and model.caption_projection is not None:
        enc_hs = model.caption_projection(encoder_hidden_states)
        enc_hs = enc_hs.view(batch_size, -1, enc_hs.shape[-1])
    else:
        enc_hs = encoder_hidden_states

    if vlm_attention_mask is not None and vlm_attention_mask.ndim == 2:
        enc_attn_mask = (1 - vlm_attention_mask.to(x.dtype)) * -10000.0
        enc_attn_mask = enc_attn_mask.unsqueeze(1)
    else:
        enc_attn_mask = encoder_attention_mask

    # --- Transformer blocks ---
    v_head_depth = getattr(model, 'v_head_depth', 0) or 0
    total_blocks = len(model.transformer_blocks)
    shared_depth = total_blocks - v_head_depth

    for block_idx in range(total_blocks):
        block = model.transformer_blocks[block_idx]
        x, dx = ltx_block_jvp(
            block, x, dx, temb, dtemb,
            encoder_hidden_states=enc_hs,
            image_rotary_emb=image_rotary_emb,
            encoder_attention_mask=enc_attn_mask,
            n_view=n_view,
            cross_view_attn=(block_idx % 3 == 0),
            backend=backend,
            self_attention_mask=video_attention_mask,
        )

    # --- Output head ---
    ssv = model.scale_shift_table[None, None] + embedded_timestep[:, :, None]
    shift, scale = ssv[:, :, 0], ssv[:, :, 1]
    dshift = dembedded_timestep
    dscale = dembedded_timestep

    x, dx = layernorm_jvp(
        x, dx,
        weight=None, bias=None,
        eps=1e-6,
    )
    x_mod = x * (1.0 + scale) + shift
    dx_mod = dx * (1.0 + scale) + x * dscale + dshift

    proj_out = model.proj_out
    _, du_dt = _linear(x_mod, dx_mod, proj_out.weight, proj_out.bias)

    if t2v_prefix_tokens:
        zero_prefix = torch.zeros_like(du_dt[:, :t2v_prefix_tokens])
        du_dt = torch.cat((zero_prefix, du_dt), dim=1)

    return du_dt


# ---------------------------------------------------------------------------
# Top-level entry point (drop-in replacement for _run_imf_jvp)
# ---------------------------------------------------------------------------

def run_imf_jvp_fast(
    model,
    z: torch.Tensor,
    t: torch.Tensor,
    v_tan: torch.Tensor,
    h_tan: torch.Tensor,
    jvp_mode: str,
    *,
    backend: str = "fast",
    conditioning_kwargs: Dict,
    jvp_fn=None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Drop-in replacement for _run_imf_jvp in fast/strict mode.

    Args:
        backend: "fast" (stacked GEMM) or "strict" (separate GEMMs, bit-exact)

    Returns (du_dt, advection_component, time_component) matching
    the _imf_jvp signature. advection/time are None unless jvp_mode
    produces them.
    """
    _jvp = jvp_fn if jvp_fn is not None else ltx_dit_jvp
    def _run_one(v_tangent, h_tangent):
        return _jvp(
            model, z, v_tangent, t, h_tangent,
            backend=backend,
            **conditioning_kwargs,
        )

    if jvp_mode == "combined":
        d = _run_one(v_tan, h_tan)
        return d, None, None

    zero_h = torch.zeros_like(h_tan)
    zero_v = torch.zeros_like(v_tan)
    adv = tim = None

    if jvp_mode in ("advection", "both"):
        adv = _run_one(v_tan, zero_h)
    if jvp_mode in ("time", "both"):
        tim = _run_one(zero_v, h_tan)

    if jvp_mode == "advection":
        d = adv
    elif jvp_mode == "time":
        d = tim
    else:
        d = adv + tim
    return d, adv, tim
