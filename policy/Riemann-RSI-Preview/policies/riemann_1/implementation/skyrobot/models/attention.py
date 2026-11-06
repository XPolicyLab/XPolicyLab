import torch
import torch.nn as nn
from skyrobot.models.attn_op import FlexAttnFunc

class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        dt = x.dtype
        return torch.nn.functional.layer_norm(
            x.float(), self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps).to(dt)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.eps = eps

        self.attn_op = FlexAttnFunc(is_cross=False)

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    @staticmethod
    def apply_rotary_emb(x, freqs):
        x_out = torch.view_as_complex(
            x.to(torch.float64).reshape(x.shape[0], x.shape[1],
                                        x.shape[2], -1, 2))
        x_out = torch.view_as_real(x_out * freqs).flatten(3)
        return x_out.to(x.dtype)

    def forward(self, x, rotary_emb):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if rotary_emb is not None:
            q = self.apply_rotary_emb(q, rotary_emb)
            k = self.apply_rotary_emb(k, rotary_emb)

        x = self.attn_op(q, k, v)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x

    def forward_with_cache(self, x, rotary_emb, cached_k=None, cached_v=None, update_cache=True):
        """Self-attention with KV cache for inference.

        Args:
            x: [B, L_new, C] new tokens (clean append or noisy denoise).
            rotary_emb: [B, L_new, 1, head_dim/2] RoPE for the new tokens only.
            cached_k: [B, L_cached, heads, head_dim] or None.
            cached_v: [B, L_cached, heads, head_dim] or None.
            update_cache: if True, append new K/V to cache and return updated
                          cache (for clean token append); if False, do not
                          update cache (for noisy denoise steps where KV is
                          temporary).

        Returns:
            out: [B, L_new, C]
            new_cached_k: [B, L_cached + L_new, heads, head_dim] (or original
                          cached_k when update_cache=False).
            new_cached_v: same as above.
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        if rotary_emb is not None:
            q = self.apply_rotary_emb(q, rotary_emb)
            k = self.apply_rotary_emb(k, rotary_emb)

        # Concat cached prefix KV with current KV for attention
        if cached_k is not None:
            full_k = torch.cat([cached_k, k], dim=1)
            full_v = torch.cat([cached_v, v], dim=1)
        else:
            full_k = k
            full_v = v

        # Standard scaled dot-product attention (no mask needed)
        q = q.transpose(1, 2)       # [B, heads, L_new, head_dim]
        fk = full_k.transpose(1, 2) # [B, heads, L_full, head_dim]
        fv = full_v.transpose(1, 2) # [B, heads, L_full, head_dim]
        out = torch.nn.functional.scaled_dot_product_attention(q, fk, fv)
        out = out.transpose(1, 2)    # [B, L_new, heads, head_dim]

        out = out.flatten(2)
        out = self.o(out)

        # Update cache
        if update_cache:
            new_cached_k = full_k
            new_cached_v = full_v
        else:
            new_cached_k = cached_k
            new_cached_v = cached_v

        return out, new_cached_k, new_cached_v


class WanCrossAttention(nn.Module):

    # Inference-only text key-padding mask, shape [B, 1, 1, L_text] bool
    # (True = attend, False = pad). Set once per episode by the inference
    # wrapper (model_dp.encode_text); shared by all blocks like the FlexAttn
    # masks above. None => attend all text positions (legacy behaviour).
    infer_text_kpm = None

    def __init__(self,
                 dim,
                 num_heads,
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.eps = eps

        self.attn_op = FlexAttnFunc(is_cross=True)

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = self.attn_op(q, k, v)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x

    def forward_inference(self, x, context):
        """Cross-attention using standard SDPA (no FlexAttn block mask needed).

        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # SDPA: [B, heads, L, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # Hard-exclude T5 pad tokens (parity with the FlexAttn cross mask used
        # in training). attn_mask [B,1,1,L_text] bool broadcasts over heads/L1.
        m = WanCrossAttention.infer_text_kpm
        if m is not None and m.shape[0] == b and m.shape[-1] == k.shape[2]:
            attn_mask = m.to(device=q.device)
        else:
            attn_mask = None
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask)
        x = x.transpose(1, 2)  # [B, L1, heads, head_dim]

        x = x.flatten(2)
        x = self.o(x)
        return x



if __name__ == "__main__":
    import time
    from skyrobot.utils.utils import get_mesh_id
    from skyrobot.models.rotary import WanRotaryPosEmbed

    device = "cuda"
    dtype = torch.bfloat16
    dim = 2048
    num_heads = 16
    head_dim = dim // num_heads  # 128
    patch_size = (1, 2, 2)
    patch_f, patch_h, patch_w = patch_size
    chunk_size = 1
    num_iters = 1000
    warmup_iters = 5

    # --- Two samples with different F ---
    context_shapes = [(4, 1, 40, 30), (4, 1, 40, 30)]
    latent_shapes = [(4, 16, 40, 30), (4, 32, 40, 30)]
    state_shapes = [(4, 17, 1, 1), (4, 33, 1, 1)]   # F = ctx_patched + lat_patched
    action_shapes = [(4, 16, 128, 1), (4, 32, 128, 1)]
    B = len(latent_shapes)

    # --- Init FlexAttn masks ---
    FlexAttnFunc.init_mask(
        context_shapes=context_shapes,
        latent_shapes=latent_shapes,
        state_shapes=state_shapes,
        action_shapes=action_shapes,
        chunk_size=chunk_size,
        patch_size=patch_size,
        device=device,
    )
    seq_len = FlexAttnFunc.attention_mask.shape[-1]
    text_len = FlexAttnFunc.cross_attention_mask.shape[-1]
    print(f"Self attn mask: {FlexAttnFunc.attention_mask.shape}")
    print(f"Cross attn mask: {FlexAttnFunc.cross_attention_mask.shape}")
    print(f"seq_len={seq_len}, text_len={text_len}, B={B}")

    # --- Build per-sample grid_ids ---
    grid_id_list = []
    per_sample_tokens = []
    for i in range(B):
        _, C_F, C_H, C_W = context_shapes[i]
        _, L_F, L_H, L_W = latent_shapes[i]
        _, S_F, S_H, S_W = state_shapes[i]
        _, A_F, A_H, A_W = action_shapes[i]
        cf, ch, cw = C_F // patch_f, C_H // patch_h, C_W // patch_w
        lf, lh, lw = L_F // patch_f, L_H // patch_h, L_W // patch_w
        context_grid = get_mesh_id(cf, ch, cw, f_w=1, f_shift=0, action=False)
        latent_grid = get_mesh_id(lf, lh, lw, f_w=1, f_shift=0, action=False)
        state_grid = get_mesh_id(S_F, S_H, S_W, f_w=1, f_shift=0, action=True)
        action_grid = get_mesh_id(A_F, A_H, A_W, f_w=1, f_shift=0, action=True)
        # Order: context, latent_clean, latent_noisy, state_clean, action_clean, action_noisy
        full_grid = torch.cat([context_grid, latent_grid, latent_grid, state_grid, action_grid, action_grid], dim=1)
        grid_id_list.append(full_grid.to(device))
        per_sample_tokens.append(full_grid.shape[1])
    print(f"Per-sample tokens: {per_sample_tokens}")

    # --- RoPE ---
    batched_grid_ids = WanRotaryPosEmbed.pad_and_stack(grid_id_list, pad_to=seq_len)
    rope = WanRotaryPosEmbed(
        attention_head_dim=head_dim, patch_size=patch_size, max_seq_len=4096
    ).to(device)
    rotary_emb = rope(batched_grid_ids)[:, :, None]

    # --- Models ---
    self_attn = WanSelfAttention(dim=dim, num_heads=num_heads).to(device).to(dtype)
    cross_attn = WanCrossAttention(dim=dim, num_heads=num_heads).to(device).to(dtype)

    # --- Benchmark: Self Attention forward + backward ---
    print(f"\n--- Self Attention Benchmark ({num_iters} iters, {warmup_iters} warmup) ---")
    for it in range(warmup_iters + num_iters):
        if it == warmup_iters:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()

        x = torch.randn(B, seq_len, dim, device=device, dtype=dtype, requires_grad=True)
        out = self_attn(x, rotary_emb=rotary_emb)
        loss = out.sum()
        loss.backward()

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    self_attn_time = (t1 - t0) / num_iters * 1000
    self_attn_mem = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  Avg time: {self_attn_time:.2f} ms/iter")
    print(f"  Peak GPU memory: {self_attn_mem:.2f} GB")

    # --- Benchmark: Cross Attention forward + backward ---
    print(f"\n--- Cross Attention Benchmark ({num_iters} iters, {warmup_iters} warmup) ---")
    for it in range(warmup_iters + num_iters):
        if it == warmup_iters:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()

        x = torch.randn(B, seq_len, dim, device=device, dtype=dtype, requires_grad=True)
        ctx = torch.randn(B, text_len, dim, device=device, dtype=dtype, requires_grad=True)
        out = cross_attn(x, ctx)
        loss = out.sum()
        loss.backward()

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    cross_attn_time = (t1 - t0) / num_iters * 1000
    cross_attn_mem = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  Avg time: {cross_attn_time:.2f} ms/iter")
    print(f"  Peak GPU memory: {cross_attn_mem:.2f} GB")

    # --- GPU utilization stats ---
    print(f"\n--- GPU Stats ---")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  Total memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  Current allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # --- Correctness: RoPE consistency ---
    print(f"\n--- RoPE Consistency Check ---")
    for i in range(B):
        n = per_sample_tokens[i]
        single_freqs = rope(grid_id_list[i][None])
        diff = (rotary_emb[i:i+1, :n, 0] - single_freqs).abs().max()
        print(f"  Sample {i}: max diff = {diff:.2e}")
        assert diff < 1e-6, f"RoPE mismatch for sample {i}!"

    # --- Correctness: gradient flow ---
    print(f"\n--- Gradient Flow Check ---")
    x = torch.randn(B, seq_len, dim, device=device, dtype=dtype, requires_grad=True)
    out = self_attn(x, rotary_emb=rotary_emb)
    loss = out[:, :per_sample_tokens[0]].sum()
    loss.backward()
    print(f"  x.grad norm: {x.grad.norm():.4f}")
    print(f"  x.grad valid region norm: {x.grad[0, :per_sample_tokens[0]].norm():.4f}")
    print(f"  x.grad pad region norm: {x.grad[0, per_sample_tokens[0]:].norm():.4f}")

    print("\nAll tests passed!")
    import os
    os._exit(0)
