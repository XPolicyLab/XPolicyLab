import torch
import torch.nn as nn
from einops import rearrange
from skyrobot.models.attention import WanLayerNorm, WanSelfAttention, WanCrossAttention




class WanAttentionBlock(nn.Module):

    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6
    ):
        super().__init__()
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, qk_norm, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim)
        )
        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        rotary_emb,
    ) -> torch.Tensor:
        dt = hidden_states.dtype
        # Modulation: compute in input dtype to avoid bf16->fp32 copies
        temb_mod = self.modulation[None].to(dt) + temb.to(dt)
        shift_msa, scale_msa, gate_msa, c_shift, c_scale, c_gate = \
            rearrange(temb_mod, 'b l n c -> b n l c').chunk(6, dim=1)
        shift_msa, scale_msa, gate_msa = shift_msa.squeeze(1).float(), scale_msa.squeeze(1).float(), gate_msa.squeeze(1).float()
        c_shift, c_scale, c_gate = c_shift.squeeze(1).float(), c_scale.squeeze(1).float(), c_gate.squeeze(1).float()

        # 1. Self-attention (norm internally casts to fp32 and back)
        x = self.norm1(hidden_states).float() * (1. + scale_msa) + shift_msa
        x = x.to(dt)  # cast back to input dtype for attention
        x = self.self_attn(x, rotary_emb)
        hidden_states = hidden_states.float() + x.float() * gate_msa
        hidden_states = hidden_states.to(dt)  # cast back to input dtype
        # 2. Cross-attention
        x = self.cross_attn(self.norm3(hidden_states), encoder_hidden_states)
        hidden_states = hidden_states + x

        # 3. Feed-forward
        x = self.norm2(hidden_states).float() * (1. + c_scale) + c_shift
        x = self.ffn(x.to(dt)).float()
        hidden_states = hidden_states.float() + x * c_gate
        hidden_states = hidden_states.to(dt)  # cast back to input dtype

        return hidden_states


if __name__ == "__main__":
    import time
    from skyrobot.utils.utils import get_mesh_id
    from skyrobot.models.rotary import WanRotaryPosEmbed
    from skyrobot.models.attn_op import FlexAttnFunc

    device = "cuda"
    dtype = torch.bfloat16
    dim = 2048
    ffn_dim = dim * 4
    num_heads = 16
    head_dim = dim // num_heads
    patch_size = (1, 2, 2)
    patch_f, patch_h, patch_w = patch_size
    chunk_size = 1
    num_iters = 100
    warmup_iters = 5

    # --- Shapes ---
    context_shapes = [(4, 1, 40, 30), (4, 1, 40, 30), (4, 1, 40, 30), (4, 1, 40, 30)]
    latent_shapes = [(4, 16, 40, 30), (4, 32, 40, 30), (4, 16, 40, 30), (4, 32, 40, 30)]
    state_shapes = [(4, 17, 1, 1), (4, 33, 1, 1), (4, 17, 1, 1), (4, 33, 1, 1)]
    action_shapes = [(4, 16, 128, 1), (4, 32, 128, 1), (4, 16, 128, 1), (4, 32, 128, 1)]
    B = len(latent_shapes)

    # --- Init masks ---
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
    print(f"seq_len={seq_len}, text_len={text_len}, B={B}")

    # --- Build RoPE ---
    grid_id_list = []
    per_sample_tokens = []
    for i in range(B):
        _, cf, ch, cw = context_shapes[i]
        _, lf, lh, lw = latent_shapes[i]
        _, sf, sh, sw = state_shapes[i]
        _, af, ah, aw = action_shapes[i]
        ctx_grid = get_mesh_id(cf // patch_f, ch // patch_h, cw // patch_w, f_w=1, f_shift=0, action=False)
        lat_grid = get_mesh_id(lf // patch_f, lh // patch_h, lw // patch_w, f_w=1, f_shift=0, action=False)
        st_grid = get_mesh_id(sf, sh, sw, f_w=1, f_shift=0, action=True)
        act_grid = get_mesh_id(af, ah, aw, f_w=1, f_shift=0, action=True)
        full_grid = torch.cat([ctx_grid, lat_grid, lat_grid, st_grid, act_grid, act_grid], dim=1)
        grid_id_list.append(full_grid.to(device))
        per_sample_tokens.append(full_grid.shape[1])

    batched_grid_ids = WanRotaryPosEmbed.pad_and_stack(grid_id_list, pad_to=seq_len)
    rope = WanRotaryPosEmbed(attention_head_dim=head_dim, patch_size=patch_size, max_seq_len=4096).to(device)
    rotary_emb = rope(batched_grid_ids)[:, :, None]  # [B, seq_len, 1, D/2]
    print(f"Per-sample tokens: {per_sample_tokens}")
    print(f"Rotary emb: {rotary_emb.shape}")

    # --- Create block ---
    block = WanAttentionBlock(
        dim=dim, ffn_dim=ffn_dim, num_heads=num_heads, qk_norm=True, cross_attn_norm=False
    ).to(device).to(dtype)
    print(f"Block params: {sum(p.numel() for p in block.parameters()) / 1e6:.1f}M")

    # Compile the block
    # block = torch.compile(block, dynamic=True)

    # --- Forward test ---
    print("\n--- Forward Test ---")
    hidden_states = torch.randn(B, seq_len, dim, device=device, dtype=dtype)
    encoder_hidden_states = torch.randn(B, text_len, dim, device=device, dtype=dtype)
    temb = torch.randn(B, seq_len, 6, dim, device=device, dtype=dtype)

    out = block(hidden_states, encoder_hidden_states, temb, rotary_emb)
    print(f"Input: {hidden_states.shape}, Output: {out.shape}")
    for i in range(B):
        n = per_sample_tokens[i]
        print(f"  Sample {i} (tokens={n}): valid_norm={out[i,:n].norm():.4f}, pad_norm={out[i,n:].norm():.4f}")

    # --- Backward test ---
    print("\n--- Backward Test ---")
    hidden_states = torch.randn(B, seq_len, dim, device=device, dtype=dtype, requires_grad=True)
    out = block(hidden_states, encoder_hidden_states, temb, rotary_emb)
    loss = out[:, :per_sample_tokens[0]].sum()
    loss.backward()
    print(f"  x.grad norm: {hidden_states.grad.norm():.4f}")
    print(f"  x.grad valid: {hidden_states.grad[0, :per_sample_tokens[0]].norm():.4f}")
    print(f"  x.grad pad: {hidden_states.grad[0, per_sample_tokens[0]:].norm():.4f}")

    # --- Mask correctness verification ---
    print("\n--- Mask Correctness Verification ---")
    mask_mod = FlexAttnFunc.attention_mask.mask_mod
    for bi in range(B):
        _, cf, ch, cw = context_shapes[bi]
        _, lf, lh, lw = latent_shapes[bi]
        _, sf, sh, sw = state_shapes[bi]
        _, af, ah, aw = action_shapes[bi]
        nc = (cf // patch_f) * (ch // patch_h) * (cw // patch_w)
        nl = (lf // patch_f) * (lh // patch_h) * (lw // patch_w)
        ns = sf * sh * sw
        na = af * ah * aw
        n_total = nc + nl * 2 + ns + na * 2

        b_t = torch.tensor(bi, device=device)
        h_t = torch.tensor(0, device=device)
        q_idx = torch.arange(seq_len, device=device)
        kv_idx = torch.arange(seq_len, device=device)
        q_grid, kv_grid = torch.meshgrid(q_idx, kv_idx, indexing="ij")
        dense = mask_mod(b_t, h_t, q_grid, kv_grid)
        m = dense[:n_total, :n_total]

        # Boundaries: context, lat_clean, lat_noisy, state, act_clean, act_noisy
        ctx_end = nc
        lc_end = nc + nl
        ln_end = nc + nl * 2
        st_end = nc + nl * 2 + ns
        ac_end = nc + nl * 2 + ns + na

        def attend_ratio(q_s, q_e, k_s, k_e):
            return m[q_s:q_e, k_s:k_e].float().mean().item()

        print(f"\n  Sample {bi} (nc={nc}, nl={nl}, ns={ns}, na={na}):")
        labels = ["Context", "L_Clean", "L_Noisy", "State", "A_Clean", "A_Noisy"]
        bounds = [0, ctx_end, lc_end, ln_end, st_end, ac_end, n_total]

        header = f"  {'':>10}" + "".join(f"{c:>10}" for c in labels)
        print(header)
        for ri, rl in enumerate(labels):
            row = f"  {rl:>10}"
            for ci in range(len(labels)):
                val = attend_ratio(bounds[ri], bounds[ri+1], bounds[ci], bounds[ci+1])
                row += f"{val:>10.1%}"
            print(row)

        pad_attend = dense[n_total:, :].float().mean().item()
        pad_attend_col = dense[:, n_total:].float().mean().item()

        print(f"\n  Key checks:")
        print(f"    L_Clean -> State:   {attend_ratio(ctx_end, lc_end, ln_end, st_end):.1%} (should be 0%)")
        print(f"    State -> L_Clean:   {attend_ratio(ln_end, st_end, ctx_end, lc_end):.1%} (should be > 0%)")
        print(f"    State -> A_Clean:   {attend_ratio(ln_end, st_end, st_end, ac_end):.1%} (should be > 0%)")
        print(f"    L_Noisy -> L_Clean: {attend_ratio(lc_end, ln_end, ctx_end, lc_end):.1%} (should be < 100%)")
        print(f"    L_Noisy -> A_Clean: {attend_ratio(lc_end, ln_end, st_end, ac_end):.1%} (should be > 0%)")
        print(f"    L_Noisy -> A_Noisy: {attend_ratio(lc_end, ln_end, ac_end, n_total):.1%} (should be 0%)")
        print(f"    A_Noisy -> L_Noisy: {attend_ratio(ac_end, n_total, lc_end, ln_end):.1%} (should be 0%)")
        print(f"    A_Noisy -> State:   {attend_ratio(ac_end, n_total, ln_end, st_end):.1%} (should be 0%)")
        print(f"    Padding rows: {pad_attend:.2%} (should be 0%)")
        print(f"    Padding cols: {pad_attend_col:.2%} (should be 0%)")

        errors = []
        if pad_attend > 0: errors.append(f"pad rows: {pad_attend:.2%}")
        if pad_attend_col > 0: errors.append(f"pad cols: {pad_attend_col:.2%}")
        print(f"  {'ERRORS: ' + str(errors) if errors else 'All checks OK!'}")

    # --- Benchmark ---
    print(f"\n--- Benchmark ({num_iters} iters, {warmup_iters} warmup) ---")
    h_buf = torch.randn(B, seq_len, dim, device=device, dtype=dtype)
    e_buf = torch.randn(B, seq_len, 6, dim, device=device, dtype=torch.float32)
    ctx_buf = torch.randn(B, text_len, dim, device=device, dtype=dtype)

    for it in range(warmup_iters + num_iters):
        if it == warmup_iters:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()

        h = h_buf.detach().requires_grad_(True)
        out = block(h, ctx_buf, e_buf, rotary_emb)
        out.sum().backward()

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    avg_time = (t1 - t0) / num_iters * 1000
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  Avg time: {avg_time:.2f} ms/iter")
    print(f"  Peak GPU memory: {peak_mem:.2f} GB")
    print(f"  GPU: {torch.cuda.get_device_name()}")

    print("\nAll tests passed!")
    import os
    os._exit(0)
