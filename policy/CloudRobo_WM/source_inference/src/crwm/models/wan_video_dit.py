import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Any, Dict, Tuple, Optional
from crwm.utils.logging_config import get_logger
logger = get_logger(__name__)

def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, ctx_mask: Optional[torch.Tensor]=None):
    B, S_q, _ = q.shape
    S_k = k.shape[1]
    head_dim = q.shape[2] // num_heads
    q = q.view(B, S_q, num_heads, head_dim).transpose(1, 2)
    k = k.view(B, S_k, num_heads, head_dim).transpose(1, 2)
    v = v.view(B, S_k, num_heads, head_dim).transpose(1, 2)
    if ctx_mask is not None and ctx_mask.dim() == 3:
        ctx_mask = ctx_mask.unsqueeze(1)
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=ctx_mask)
    x = x.transpose(1, 2).reshape(B, S_q, -1)
    return x

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift

def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(10000, -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(dim // 2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)

def precompute_freqs_cis_3d(dim: int, end: int=1024, theta: float=10000.0):
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return (f_freqs_cis, h_freqs_cis, w_freqs_cis)

def precompute_freqs_cis(dim: int, end: int=1024, theta: float=10000.0):
    freqs = 1.0 / theta ** (torch.arange(0, dim, 2)[:dim // 2].double() / dim)
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def rope_apply(x, freqs, num_heads):
    B, S, D = x.shape
    x = x.view(B, S, num_heads, -1)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(B, S, x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)

class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-05):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight

class AttentionModule(nn.Module):

    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, q, k, v, ctx_mask=None):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=ctx_mask)
        return x

class SelfAttention(nn.Module):

    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float=1e-06):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = self.num_heads * self.attn_head_dim
        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)

    def forward(self, x, freqs, self_attn_mask: Optional[torch.Tensor]=None):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=self_attn_mask)
        return self.o(x)

class CrossAttention(nn.Module):

    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float=1e-06):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = self.num_heads * self.attn_head_dim
        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, ctx_mask: Optional[torch.Tensor]=None):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=ctx_mask)
        return self.o(x)

class GateModule(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual

class DiTBlock(nn.Module):

    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, ffn_dim: int, eps: float=1e-06):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_head_dim = attn_head_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.self_attn = SelfAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.cross_attn = CrossAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, ffn_dim), nn.GELU(approximate='tanh'), nn.Linear(ffn_dim, hidden_dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim ** 0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs, context_mask=None, self_attn_mask: Optional[torch.Tensor]=None):
        if context_mask is not None and context_mask.dim() == 3:
            context_mask = context_mask.unsqueeze(1)
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2), shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2))
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs, self_attn_mask=self_attn_mask))
        x = x + self.cross_attn(self.norm3(x), context, ctx_mask=context_mask)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x

class Head(nn.Module):

    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = self.head(self.norm(x) * (1 + scale) + shift)
        return x

class WanVideoDiT(torch.nn.Module):

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        f, h, w = (grid_size[0], grid_size[1], grid_size[2])
        px, py, pz = (self.patch_size[0], self.patch_size[1], self.patch_size[2])
        B = x.shape[0]
        c = x.shape[-1] // (px * py * pz)
        x = x.view(B, f, h, w, px, py, pz, c)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        x = x.view(B, c, f * px, h * py, w * pz)
        return x

    def post_dit(self, x_tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        f, h, w = pre_state['meta']['grid_size']
        x = self.head(x_tokens, pre_state['t'])
        x = self.unpatchify(x, (f, h, w))
        return x

    def __init__(self, hidden_dim, in_dim, ffn_dim, out_dim, text_dim, freq_dim, eps, patch_size, num_heads, attn_head_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.patch_size = tuple(patch_size)
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.patch_embedding = nn.Conv3d(in_dim, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, hidden_dim), nn.GELU(approximate='tanh'), nn.Linear(hidden_dim, hidden_dim))
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, attn_head_dim, num_heads, ffn_dim, eps) for _ in range(num_layers)])
        self.head = Head(hidden_dim, out_dim, patch_size, eps)
        self.freqs = precompute_freqs_cis_3d(attn_head_dim)

    @staticmethod
    def build_video_to_video_mask(video_seq_len, video_tokens_per_frame, device):
        mask = torch.ones(video_seq_len, video_seq_len, dtype=torch.bool, device=device)
        mask[:video_tokens_per_frame, video_tokens_per_frame:] = False
        return mask

    def pre_dit(self, x, timestep, context, context_mask):
        B = x.shape[0]
        tpf = x.shape[3] // self.patch_size[1] * (x.shape[4] // self.patch_size[2])
        tt = timestep.view(B, 1, 1).expand(B, x.shape[2], tpf).clone()
        tt[:, 0] = 0
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, tt.reshape(-1))).reshape(B, -1, self.hidden_dim)
        t_mod = self.time_projection(t).unflatten(2, (6, self.hidden_dim))
        x = self.patch_embedding(x)
        f, h, w = x.shape[2:]
        tokens = x.permute(0, 2, 3, 4, 1).reshape(B, -1, x.shape[1]).contiguous()
        embedded_context = self.text_embedding(context)
        mask = context_mask[:, None].expand(-1, f * h * w, -1)
        freqs = torch.cat([self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1), self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1), self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)], -1).reshape(f * h * w, 1, -1).to(tokens.device)
        return dict(tokens=tokens, freqs=freqs, t=t, t_mod=t_mod, context=embedded_context, context_mask=mask, meta=dict(grid_size=(f, h, w), tokens_per_frame=tpf, batch_size=B))
