import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange
from wall_x.utils.io import hash_state_dict_keys
from wall_x.model.vga.modeling.wan_camera_controller import SimpleAdapter
from wall_x.model.core.ops import rmsnorm, rope

# SP (Sequence Parallelism) support
from wall_x.distributed import (
    get_sp_world_size,
    get_sp_rank,
    sp_shard,
    sp_gather,
    sp_all_to_all_4d,
    sp_all_gather_4d,
)

# SP functions are imported above; mark SP as available
SP_AVAILABLE = True

USE_PARALLEL_VIEW_ATTN = os.getenv("USE_PARALLEL_VIEW_ATTN", "False").lower() == "true"


def _assert_rope_freqs_fp32(cos: torch.Tensor, sin: torch.Tensor):
    assert cos.dtype == torch.float32 and sin.dtype == torch.float32, (
        "xcompute RoPE requires fp32 cos/sin. If this trips under FSDP2, "
        "check MixedPrecisionPolicy.cast_forward_inputs."
    )


try:
    import flash_attn_interface

    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn

    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn

    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    compatibility_mode=False,
    mask: Optional[torch.Tensor] = None,
):
    if mask is not None:
        compatibility_mode = True
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x, tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v, deterministic=True)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


def _sp_local_seq_chunk(t: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Slice a full-length tensor to this SP rank's local sequence chunk.

    Zero-pads to a multiple of ``sp_size`` first, exactly like ``sp_shard``, so
    the chunk length matches the sequence-sharded activation even when the
    sequence is not divisible by ``sp_size`` (the padding is trimmed by
    ``sp_gather``). Identity when sequence parallelism is off.
    """
    sp = get_sp_world_size()
    if sp <= 1:
        return t
    rem = t.shape[dim] % sp
    if rem:
        pad_shape = list(t.shape)
        pad_shape[dim] = sp - rem
        t = torch.cat([t, t.new_zeros(pad_shape)], dim=dim)
    chunk = t.shape[dim] // sp
    return t.narrow(dim, get_sp_rank() * chunk, chunk)


def sinusoidal_embedding_1d(dim, position, output_dtype=None):
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(
            10000,
            -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(
                dim // 2
            ),
        ),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(output_dtype if output_dtype is not None else position.dtype)


def precompute_freqs_cis(
    dim: int,
    end: int = 1024,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
):
    """1d rope precompute — returns (cos, sin) each of shape [end, dim//2].

    Args:
        dim: head dimension.
        end: sequence length.
        theta: RoPE base.
        device: optional device to place the precomputed frequencies on. If None,
            uses PyTorch default device (keeps previous behavior, typically CPU).
    """
    if device is None:
        base = torch.arange(0, dim, 2)[: (dim // 2)].float()
        freqs = 1.0 / (theta ** (base / dim))
        positions = torch.arange(end, device=freqs.device).float()
    else:
        base = torch.arange(0, dim, 2, device=device)[: (dim // 2)].float()
        freqs = 1.0 / (theta ** (base / dim))
        positions = torch.arange(end, device=device).float()

    freqs = torch.outer(positions, freqs)
    return torch.cos(freqs), torch.sin(freqs)


def precompute_freqs_cis_3d(
    dim: int,
    end: int = 1024,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
):
    """3d rope precompute — each element is a (cos, sin) tuple."""
    f_cs = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta, device)
    h_cs = precompute_freqs_cis(dim // 3, end, theta, device)
    w_cs = precompute_freqs_cis(dim // 3, end, theta, device)
    return f_cs, h_cs, w_cs


def precompute_freqs_cis_4d(
    dim: int,
    end: int = 1024,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
):
    """4D rope precompute (f, h, w, view) — each element is a (cos, sin) tuple."""
    quarter_dim = dim // 4
    f_cs = precompute_freqs_cis(quarter_dim, end, theta, device)
    h_cs = precompute_freqs_cis(quarter_dim, end, theta, device)
    w_cs = precompute_freqs_cis(quarter_dim, end, theta, device)
    view_cs = precompute_freqs_cis(quarter_dim, end, theta, device)
    return f_cs, h_cs, w_cs, view_cs


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        # dtype = x.dtype
        # return self.norm(x.float()).to(dtype) * self.weight
        return rmsnorm(x, self.weight, self.eps)


class AttentionModule(nn.Module):
    def __init__(self, num_heads, enable_sp: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.enable_sp = enable_sp

    def compute_flops(self, q):
        B, S, _ = q.shape
        head_dim = self.num_heads
        total_dim = q.shape[-1]
        head_dim = total_dim // self.num_heads
        flops = 4 * B * self.num_heads * S * S * head_dim
        self.__flops__ += flops

    def forward(self, q, k, v, mask: Optional[torch.Tensor] = None):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, mask=mask)
        if getattr(self, "__flops__", False):
            self.compute_flops(q)
        return x


class SelfAttention(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, eps: float = 1e-6, enable_sp: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.enable_sp = enable_sp

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs, mask: Optional[torch.Tensor] = None):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        cos_freqs, sin_freqs = freqs
        # Expand cos/sin to match batch size (xcompute kernel doesn't broadcast on batch dim)
        B = q.shape[0]
        if cos_freqs.shape[0] != B:
            cos_freqs = cos_freqs.expand(B, -1, -1).contiguous()
            sin_freqs = sin_freqs.expand(B, -1, -1).contiguous()
        _assert_rope_freqs_fp32(cos_freqs, sin_freqs)
        # TODO: support action dit using sequence parallelism?
        # TODO: decouple this part to a separate module
        # Currently video dit and action dit both use this self attention module,
        # so we need to handle the sequence parallelism here, currently action dit is not
        # using sequence parallelism.
        if get_sp_world_size() > 1 and self.enable_sp is True:
            _sp_world_size = get_sp_world_size()  # noqa: F841
            num_heads = self.num_heads

            # Reshape to [B, S_local, H, D_head]
            q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
            k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
            v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)

            # All-to-all: sequence sharding -> head sharding
            # [B, S_local, H, D_head] -> [B, S, H/sp, D_head]
            q = sp_all_to_all_4d(q)
            k = sp_all_to_all_4d(k)
            v = sp_all_to_all_4d(v)

            # Apply RoPE using xcompute kernel (q, k stay [B, S, H/sp, D])
            # Convert interleaved pairs → half-split pairs (complex→rotate_half layout)
            q = rearrange(q, "... (d two) -> ... (two d)", two=2)
            k = rearrange(k, "... (d two) -> ... (two d)", two=2)
            q, k = rope(q, k, cos_freqs, sin_freqs)
            # Convert half-split pairs → interleaved pairs (restore original layout)
            q = rearrange(q, "... (two d) -> ... (d two)", two=2)
            k = rearrange(k, "... (two d) -> ... (d two)", two=2)

            x = flash_attn.flash_attn_func(q, k, v, deterministic=True)

            # All-to-all backward: head sharding -> sequence sharding
            # [B, S, H/sp, D_head] -> [B, S_local, H, D_head]
            x = sp_all_gather_4d(x)

            # Reshape back to [B, S_local, D]
            x = rearrange(x, "b s n d -> b s (n d)")
            o = self.o(x)
        else:
            # Apply RoPE using xcompute kernel
            # Reshape 3D→4D + interleaved→half-split in one step
            q = rearrange(q, "b s (n d two) -> b s n (two d)", n=self.num_heads, two=2)
            k = rearrange(k, "b s (n d two) -> b s n (two d)", n=self.num_heads, two=2)
            q, k = rope(q, k, cos_freqs, sin_freqs)
            # Half-split→interleaved + 4D→3D in one step
            q = rearrange(q, "b s n (two d) -> b s (n d two)", two=2)
            k = rearrange(k, "b s n (two d) -> b s (n d two)", two=2)

            x = self.attn(q, k, v, mask=mask)
            o = self.o(x)

        return o


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        eps: float = 1e-6,
        has_image_input: bool = False,
        multi_views: bool = False,
        view_as_tokens: bool = False,
        num_views: int = 3,
        enable_sp: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_views = num_views
        self.enable_sp = enable_sp
        self.view_as_tokens = view_as_tokens

        self.q = nn.Linear(dim, dim)
        # self.k = nn.Linear(dim, dim)
        # self.v = nn.Linear(dim, dim)
        if multi_views and (not view_as_tokens):
            self.k = nn.Linear(dim * num_views, dim)
            self.v = nn.Linear(dim * num_views, dim)
        else:
            self.k = nn.Linear(dim, dim)
            self.v = nn.Linear(dim, dim)

        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)

        # Cross-attention does not use SP (context is not sharded)
        self.attn = AttentionModule(self.num_heads, enable_sp=False)

    def forward(self, x: torch.Tensor, y: torch.Tensor, img_seq_len: int = 257):
        if self.has_image_input:
            img = y[:, :img_seq_len]
            ctx = y[:, img_seq_len:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)


class GateModule(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual


class DiTBlock(nn.Module):
    def __init__(
        self,
        has_image_input: bool,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
        enable_sp: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.enable_sp = enable_sp

        self.self_attn = SelfAttention(dim, num_heads, eps, enable_sp=enable_sp)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input, enable_sp=enable_sp
        )
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(
        self,
        x,
        context,
        t_mod,
        freqs,
        view_freqs,
        v,
        f,
        h,
        w,
        img_seq_len=257,
        view_mask: Optional[torch.Tensor] = None,
    ):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        if has_seq and get_sp_world_size() > 1:
            # x is Ulysses sequence-sharded; slice each full-length per-token
            # modulation tensor to this rank's chunk to match x.
            shift_msa = _sp_local_seq_chunk(shift_msa)
            scale_msa = _sp_local_seq_chunk(scale_msa)
            gate_msa = _sp_local_seq_chunk(gate_msa)
            shift_mlp = _sp_local_seq_chunk(shift_mlp)
            scale_mlp = _sp_local_seq_chunk(scale_mlp)
            gate_mlp = _sp_local_seq_chunk(gate_mlp)
        original_input_x = x
        # origin wan attn
        input_x = modulate(self.norm1(original_input_x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))

        # multi view attn (skip if this block has no view attention modules)
        if not hasattr(self, "view_attn"):
            x = x + self.cross_attn(self.norm3(x), context)
            input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
            x = self.gate(x, gate_mlp, self.ffn(input_x))
            return x

        if has_seq:
            # Wan 2.2: t_mod is (1, seq_len, 6, dim) — take first 3 modulation params along dim=2
            shift_view, scale_view, gate_view = (
                self.modulation_view.unsqueeze(1).to(
                    dtype=t_mod.dtype, device=t_mod.device
                )
                + t_mod[:, :, :3, :]
            ).chunk(3, dim=2)
            shift_view = shift_view.squeeze(2)
            scale_view = scale_view.squeeze(2)
            gate_view = gate_view.squeeze(2)
            if get_sp_world_size() > 1:
                # x is sequence-sharded; slice the per-token view shift/scale to
                # the local chunk for the modulate() below. gate_view stays full
                # (re-sharded at its point of use in the SP branch).
                shift_view = _sp_local_seq_chunk(shift_view)
                scale_view = _sp_local_seq_chunk(scale_view)
        else:
            # Wan 2.1: t_mod is (1, 6, dim) — take first 3 modulation params along dim=1
            shift_view, scale_view, gate_view = (
                self.modulation_view.to(dtype=t_mod.dtype, device=t_mod.device)
                + t_mod[:, :3, :]
            ).chunk(3, dim=1)

        # Reform multi view attention, it's mathematical equivalent, mainly for overlap all2all communication.
        if not USE_PARALLEL_VIEW_ATTN:
            input_x = modulate(self.norm_view(x), shift_view, scale_view)
        else:
            input_x = modulate(self.norm_view(original_input_x), shift_view, scale_view)

        # TODO: decouple this part to a separate module
        if get_sp_world_size() > 1:
            if view_mask is not None:
                raise NotImplementedError(
                    "View/Patch masking only supports non-SP mode. "
                    "Set sp_size=1 or disable masking."
                )
            q = self.view_attn.norm_q(self.view_attn.q(input_x))
            k = self.view_attn.norm_k(self.view_attn.k(input_x))
            val = self.view_attn.v(input_x)
            num_heads = self.num_heads

            # Reshape to [B, S_local, H, D_head]
            q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
            k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
            val = rearrange(val, "b s (n d) -> b s n d", n=num_heads)

            # All-to-all: sequence sharding -> head sharding
            # [B, S_local, H, D_head] -> [B, S, H/sp, D_head]
            q = sp_all_to_all_4d(q)
            k = sp_all_to_all_4d(k)
            val = sp_all_to_all_4d(val)

            q = rearrange(
                q, "(b v) (f h w) n d -> (b f) (v h w) n d", v=v, f=f, h=h, w=w
            )
            k = rearrange(
                k, "(b v) (f h w) n d -> (b f) (v h w) n d", v=v, f=f, h=h, w=w
            )
            val = rearrange(
                val, "(b v) (f h w) n d -> (b f) (v h w) n d", v=v, f=f, h=h, w=w
            )

            # Apply RoPE using xcompute kernel (q, k stay [B, S, N, D])
            view_cos, view_sin = view_freqs
            # Expand cos/sin to match batch size (xcompute kernel doesn't broadcast on batch dim)
            B_view = q.shape[0]
            if view_cos.shape[0] != B_view:
                view_cos = view_cos.expand(B_view, -1, -1).contiguous()
                view_sin = view_sin.expand(B_view, -1, -1).contiguous()
            _assert_rope_freqs_fp32(view_cos, view_sin)
            # Convert interleaved pairs → half-split pairs (complex→rotate_half layout)
            q = rearrange(q, "... (d two) -> ... (two d)", two=2)
            k = rearrange(k, "... (d two) -> ... (two d)", two=2)
            q, k = rope(q, k, view_cos, view_sin)
            # Convert half-split pairs → interleaved pairs (restore original layout)
            q = rearrange(q, "... (two d) -> ... (d two)", two=2)
            k = rearrange(k, "... (two d) -> ... (d two)", two=2)

            # x = flash_attention(q, k, val, num_heads=h_local)
            attn_out = flash_attn.flash_attn_func(q, k, val, deterministic=True)

            attn_out = rearrange(
                attn_out, "(b f) (v h w) n d -> (b v) (f h w) n d", v=v, f=f, h=h, w=w
            )

            # All-to-all backward: head sharding -> sequence sharding
            # [B, S, H/sp, D_head] -> [B, S_local, H, D_head]
            attn_out = sp_all_gather_4d(attn_out)

            # Reshape back to [B, S_local, D]
            attn_out = rearrange(attn_out, "b s n d -> b s (n d)")
            attn_out = self.view_attn.o(attn_out)
            if has_seq:
                # Wan 2.2: gate_view is (1, S_full, dim) per-view layout, but
                # attn_out is (v, S_local, dim) after SP sharding. Slice gate_view
                # to the local sequence chunk to match.
                gate_view_local = _sp_local_seq_chunk(gate_view)
                multi_view_output_attn = gate_view_local * self.view_projector(attn_out)
            else:
                # Wan 2.1: gate_view is (1, 1, dim), broadcasts freely.
                multi_view_output_attn = gate_view * self.view_projector(attn_out)
        else:
            input_x = rearrange(
                input_x, "(b v) (f h w) d -> (b f) (v h w) d", v=v, f=f, h=h, w=w
            )
            if view_mask is not None:
                view_mask = (
                    view_mask[:, None, :, :]
                    .repeat(1, f, 1, 1)
                    .flatten(0, 1)[:, None, :, :]
                )
            attn_out = self.view_projector(
                self.view_attn(input_x, view_freqs, mask=view_mask)
            )
            if has_seq:
                # Wan 2.2: gate_view is per-token in per-view layout (1, f*h*w, dim),
                # but attn_out is in per-frame layout (f, v*h*w, dim).
                # Rearrange back to per-view layout before applying gate.
                attn_out = rearrange(
                    attn_out,
                    "(b f) (v h w) d -> (b v) (f h w) d",
                    v=v,
                    f=f,
                    h=h,
                    w=w,
                )
                multi_view_output_attn = gate_view * attn_out
            else:
                # Wan 2.1: gate_view is (1, 1, dim), broadcasts freely.
                multi_view_output_attn = gate_view * attn_out
                multi_view_output_attn = rearrange(
                    multi_view_output_attn,
                    "(b f) (v h w) d -> (b v) (f h w) d",
                    v=v,
                    f=f,
                    h=h,
                    w=w,
                )

        x = x + multi_view_output_attn

        x = x + self.cross_attn(self.norm3(x), context, img_seq_len=img_seq_len)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(
        self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float
    ):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (
                self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device)
                + t_mod.unsqueeze(2)
            ).chunk(2, dim=2)
            x = self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2))
        else:
            shift, scale = (
                self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
            ).chunk(2, dim=1)
            x = self.head(self.norm(x) * (1 + scale) + shift)
        return x


class WanModel(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
        enable_sp: bool = True,  # Sequence Parallelism
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.enable_sp = enable_sp
        self.num_heads = num_heads

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        # text_dim: UMT5 / T5 outputs are typically 4096; Qwen VL adapters project backbone hidden_size
        # to qwen_proj_out_dim (often 4096) via project_out so context matches this Linear's in_features.
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    has_image_input, dim, num_heads, ffn_dim, eps, enable_sp=enable_sp
                )
                for _ in range(num_layers)
            ]
        )
        self.head = Head(dim, out_dim, patch_size, eps)

        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(
                1280, dim, has_pos_emb=has_image_pos_emb
            )  # clip_feature_dim = 1280
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        if add_control_adapter:
            self.control_adapter = SimpleAdapter(
                in_dim_control_adapter,
                dim,
                kernel_size=patch_size[1:],
                stride=patch_size[1:],
            )
        else:
            self.control_adapter = None

    def patchify(
        self,
        x: torch.Tensor,
        control_camera_latents_input: Optional[torch.Tensor] = None,
    ):
        x = x.to(
            device=self.patch_embedding.weight.device,
            dtype=self.patch_embedding.weight.dtype,
        )
        x = self.patch_embedding(x)
        if (
            self.control_adapter is not None
            and control_camera_latents_input is not None
        ):
            control_camera_latents_input = control_camera_latents_input.to(
                device=self.patch_embedding.weight.device,
                dtype=self.patch_embedding.weight.dtype,
            )
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        return x

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **kwargs,
    ):
        timestep = timestep.to(device=x.device, dtype=torch.float32)
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep, output_dtype=x.dtype)
        )
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))

        # context shape: [B, L, text_dim] -> [B, L, dim]; text_dim must match nn.Linear(text_dim, dim) in_features
        context = self.text_embedding(context)

        img_seq_len = 257
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            img_seq_len = clip_embdding.shape[1]
            context = torch.cat([clip_embdding, context], dim=1)

        x, (f, h, w) = self.patchify(x)

        # Build cos/sin freqs for xcompute rope kernel
        (f_cos, f_sin), (h_cos, h_sin), (w_cos, w_sin) = self.freqs
        cos_freqs = (
            torch.cat(
                [
                    f_cos[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    h_cos[:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    w_cos[:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(1, f * h * w, -1)
            .to(x.device)
        )
        sin_freqs = (
            torch.cat(
                [
                    f_sin[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    h_sin[:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    w_sin[:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(1, f * h * w, -1)
            .to(x.device)
        )
        freqs = (cos_freqs, sin_freqs)

        # ★★★ SP: Sequence Parallelism - shard hidden states ★★★
        original_seq_len = None
        if self.enable_sp and SP_AVAILABLE and get_sp_world_size() > 1:
            x, original_seq_len = sp_shard(x, dim=1)
            # Also shard freqs (RoPE)
            sp_rank = get_sp_rank()
            sp_world_size = get_sp_world_size()
            cos_freqs, sin_freqs = freqs
            seq_len = cos_freqs.shape[1]
            # Pad freqs if needed
            remainder = seq_len % sp_world_size
            if remainder != 0:
                pad_len = sp_world_size - remainder
                cos_freqs = torch.cat(
                    [cos_freqs, cos_freqs[:, -1:].expand(-1, pad_len, -1)], dim=1
                )
                sin_freqs = torch.cat(
                    [sin_freqs, sin_freqs[:, -1:].expand(-1, pad_len, -1)], dim=1
                )
                seq_len = cos_freqs.shape[1]
            chunk_size = seq_len // sp_world_size
            cos_freqs = cos_freqs[
                :, sp_rank * chunk_size : (sp_rank + 1) * chunk_size, :
            ]
            sin_freqs = sin_freqs[
                :, sp_rank * chunk_size : (sp_rank + 1) * chunk_size, :
            ]
            freqs = (cos_freqs, sin_freqs)

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu(pin_memory=True):
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x,
                            context,
                            t_mod,
                            freqs,
                            img_seq_len,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        context,
                        t_mod,
                        freqs,
                        img_seq_len,
                        use_reentrant=False,
                    )
            else:
                x = block(x, context, t_mod, freqs, img_seq_len=img_seq_len)

        # ★★★ SP: gather hidden states ★★★
        if (
            self.enable_sp
            and SP_AVAILABLE
            and get_sp_world_size() > 1
            and original_seq_len is not None
        ):
            x = sp_gather(x, original_seq_len, dim=1)

        # RGB output
        rgb_output = self.head(x, t)
        rgb_output = self.unpatchify(rgb_output, (f, h, w))

        return rgb_output

    @staticmethod
    def state_dict_converter():
        return WanModelStateDictConverter()


class WanModelStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        rename_dict = {
            "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
            "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
            "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
            "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
            "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
            "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
            "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
            "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
            "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
            "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
            "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
            "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
            "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
            "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
            "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
            "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
            "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
            "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
            "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
            "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
            "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
            "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
            "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
            "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
            "blocks.0.norm2.bias": "blocks.0.norm3.bias",
            "blocks.0.norm2.weight": "blocks.0.norm3.weight",
            "blocks.0.scale_shift_table": "blocks.0.modulation",
            "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
            "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
            "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
            "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
            "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
            "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
            "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
            "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
            "condition_embedder.time_proj.bias": "time_projection.1.bias",
            "condition_embedder.time_proj.weight": "time_projection.1.weight",
            "patch_embedding.bias": "patch_embedding.bias",
            "patch_embedding.weight": "patch_embedding.weight",
            "scale_shift_table": "head.modulation",
            "proj_out.bias": "head.head.bias",
            "proj_out.weight": "head.head.weight",
        }
        state_dict_ = {}
        for name, param in state_dict.items():
            if name in rename_dict:
                state_dict_[rename_dict[name]] = param
            else:
                name_ = ".".join(name.split(".")[:1] + ["0"] + name.split(".")[2:])
                if name_ in rename_dict:
                    name_ = rename_dict[name_]
                    name_ = ".".join(
                        name_.split(".")[:1]
                        + [name.split(".")[1]]
                        + name_.split(".")[2:]
                    )
                    state_dict_[name_] = param
        if hash_state_dict_keys(state_dict) == "cb104773c6c2cb6df4f9529ad5c60d0b":
            config = {
                "model_type": "t2v",
                "patch_size": (1, 2, 2),
                "text_len": 512,
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "window_size": (-1, -1),
                "qk_norm": True,
                "cross_attn_norm": True,
                "eps": 1e-6,
            }
        else:
            config = {}
        return state_dict_, config

    def from_civitai(self, state_dict):
        state_dict = {
            name: param
            for name, param in state_dict.items()
            if not name.startswith("vace")
        }
        state_dict = {
            name: param
            for name, param in state_dict.items()
            if name.split(".")[0]
            not in [
                "pose_patch_embedding",
                "face_adapter",
                "face_encoder",
                "motion_encoder",
            ]
        }
        if hash_state_dict_keys(state_dict) == "9269f8db9040a9d860eaca435be61814":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6,
            }
        elif hash_state_dict_keys(state_dict) == "aafcfd9672c3a2456dc46e1cb6e52c70":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
            }
        elif hash_state_dict_keys(state_dict) == "6bfcfb3b342cb286ce886889d519a77e":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
            }
        elif hash_state_dict_keys(state_dict) == "6d6ccde6845b95ad9114ab993d917893":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6,
            }
        elif hash_state_dict_keys(state_dict) == "349723183fc063b2bfc10bb2835cf677":
            # 1.3B PAI control
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6,
            }
        elif hash_state_dict_keys(state_dict) == "efa44cddf936c70abd0ea28b6cbe946c":
            # 14B PAI control
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
            }
        elif hash_state_dict_keys(state_dict) == "3ef3b1f8e1dab83d5b71fd7b617f859f":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
                "has_image_pos_emb": True,
            }
        elif hash_state_dict_keys(state_dict) == "70ddad9d3a133785da5ea371aae09504":
            # 1.3B PAI control v1.1
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6,
                "has_ref_conv": True,
            }
        elif hash_state_dict_keys(state_dict) == "26bde73488a92e64cc20b0a7485b9e5b":
            # 14B PAI control v1.1
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
                "has_ref_conv": True,
            }
        elif hash_state_dict_keys(state_dict) == "ac6a5aa74f4a0aab6f64eb9a72f19901":
            # 1.3B PAI control-camera v1.1
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 32,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6,
                "has_ref_conv": False,
                "add_control_adapter": True,
                "in_dim_control_adapter": 24,
            }
        elif hash_state_dict_keys(state_dict) == "b61c605c2adbd23124d152ed28e049ae":
            # 14B PAI control-camera v1.1
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 32,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
                "has_ref_conv": False,
                "add_control_adapter": True,
                "in_dim_control_adapter": 24,
            }
        elif hash_state_dict_keys(state_dict) == "1f5ab7703c6fc803fdded85ff040c316":
            # Wan-AI/Wan2.2-TI2V-5B
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 3072,
                "ffn_dim": 14336,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 48,
                "num_heads": 24,
                "num_layers": 30,
                "eps": 1e-6,
                "seperated_timestep": True,
                "require_clip_embedding": False,
                "require_vae_embedding": False,
                "fuse_vae_embedding_in_latents": True,
            }
        elif hash_state_dict_keys(state_dict) == "5b013604280dd715f8457c6ed6d6a626":
            # Wan-AI/Wan2.2-I2V-A14B
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
                "require_clip_embedding": False,
            }
        elif hash_state_dict_keys(state_dict) == "2267d489f0ceb9f21836532952852ee5":
            # Wan2.2-Fun-A14B-Control
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 52,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
                "has_ref_conv": True,
                "require_clip_embedding": False,
            }
        elif hash_state_dict_keys(state_dict) == "47dbeab5e560db3180adf51dc0232fb1":
            # Wan2.2-Fun-A14B-Control-Camera
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
                "has_ref_conv": False,
                "add_control_adapter": True,
                "in_dim_control_adapter": 24,
                "require_clip_embedding": False,
            }
        else:
            config = {}
        return state_dict, config
