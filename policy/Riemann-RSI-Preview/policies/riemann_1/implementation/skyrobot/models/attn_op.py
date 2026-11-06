import torch
import torch.nn as nn
from einops import rearrange
from typing import Callable, ClassVar
from torch.nn.attention.flex_attention import (
    BlockMask, create_block_mask, flex_attention, and_masks, or_masks
)

BLOCK_ALIGN = 128
SEGMENTS = ("context", "latent_clean", "latent_noisy", "state", "action_clean", "action_noisy")


def _pad_to_block(n):
    return n + (BLOCK_ALIGN - n % BLOCK_ALIGN) % BLOCK_ALIGN


def _expand_frame(t, h, w):
    """Expand temporal indices to spatial grid: (F,) -> (F*H*W,)"""
    return t[:, None, None].expand(-1, h, w).flatten()


class FlexAttnFunc(nn.Module):
    flex_attn: ClassVar[Callable] = torch.compile(flex_attention, dynamic=True)
    compiled_create_block_mask: ClassVar[Callable] = torch.compile(create_block_mask)
    attention_mask: ClassVar[BlockMask] = None
    cross_attention_mask: ClassVar[BlockMask] = None
    segment_offsets: ClassVar[dict] = None

    def __init__(self, is_cross=False):
        super().__init__()
        self.is_cross = is_cross

    def forward(self, query, key, value, dtype=torch.bfloat16):
        q = rearrange(query, "b s n d -> b n s d")
        k = rearrange(key, "b s n d -> b n s d")
        v = rearrange(value, "b s n d -> b n s d")

        half_dtypes = (torch.float16, torch.bfloat16)
        assert dtype in half_dtypes
        to_half = lambda x: x if x.dtype in half_dtypes else x.to(dtype)
        q, k, v = to_half(q), to_half(k), to_half(v)
        q, k = q.to(v.dtype), k.to(v.dtype)

        block_mask = FlexAttnFunc.cross_attention_mask if self.is_cross else FlexAttnFunc.attention_mask
        out = FlexAttnFunc.flex_attn(q, k, v, block_mask=block_mask, kernel_options={
            "BLOCK_M": 32, "BLOCK_N": 32,
            "BLOCK_M1": 16, "BLOCK_N1": 32,
            "BLOCK_M2": 32, "BLOCK_N2": 16,
        })
        return rearrange(out, "b n s d -> b s n d")

    @staticmethod
    @torch.no_grad()
    def init_mask(context_shapes, latent_shapes, state_shapes, action_shapes,
                  chunk_size, patch_size, device, text_len=512,
                  text_mask=None):
        """
        Build attention masks with per-sample variable lengths.

        Sequence order:
            context, latent_clean, latent_noisy, state_clean, action_clean, action_noisy

        Visibility within same timestep t:
            clean2clean (frame_ids <=):
                action_clean(5t) < latent_clean(5t+1) < state(5t+2)
            noise2clean (time_ids <):
                latent_noisy sees prev-step clean; action_noisy sees prev-step clean
            noise2noise (frame_ids ==):
                self only

        Args:
            context_shapes: list of B tuples (C, F, H, W)
            latent_shapes:  list of B tuples (C, F, H, W)
            state_shapes:   list of B tuples (C, F, H, W), F = ctx_frames_patched + lat_frames_patched
            action_shapes:  list of B tuples (C, F, H, W)
            chunk_size: temporal chunk size
            patch_size: (pF, pH, pW)
            device: torch device
            text_len: text sequence length for cross attention (default 512)
        """
        torch._inductor.config.realize_opcount_threshold = 100
        B = len(latent_shapes)
        pF, pH, pW = patch_size

        # Compute per-sample token counts
        n_ctx, n_lat, n_state, n_act = [], [], [], []
        for i in range(B):
            _, cf, ch, cw = context_shapes[i]
            _, lf, lh, lw = latent_shapes[i]
            _, sf, sh, sw = state_shapes[i]
            _, af, ah, aw = action_shapes[i]
            n_ctx.append((cf // pF) * (ch // pH) * (cw // pW))
            n_lat.append((lf // pF) * (lh // pH) * (lw // pW))
            n_state.append(sf * sh * sw)
            n_act.append(af * ah * aw)

        max_ctx, max_lat = max(n_ctx), max(n_lat)
        max_state, max_act = max(n_state), max(n_act)
        seg_max = [max_ctx, max_lat, max_lat, max_state, max_act, max_act]
        offsets = []
        pos = 0
        for m in seg_max:
            offsets.append(pos)
            pos += m
        seq_len = _pad_to_block(pos)

        FlexAttnFunc.segment_offsets = {
            name: (off, mx) for name, off, mx in zip(SEGMENTS, offsets, seg_max)
        }
        FlexAttnFunc.segment_offsets.update(
            seq_len=seq_len, n_ctx=n_ctx, n_lat=n_lat, n_state=n_state, n_act=n_act)

        # Allocate (B, seq_len) ID tensors
        L = torch.long
        all_frame = torch.full((B, seq_len), -1, dtype=L)
        all_time = torch.full((B, seq_len), -1, dtype=L)
        all_noise = torch.full((B, seq_len), -1, dtype=L)
        all_valid = torch.zeros((B, seq_len), dtype=torch.bool)

        for i in range(B):
            _, cf, _, _ = context_shapes[i]
            _, lf, lh, lw = latent_shapes[i]
            _, sf, sh, sw = state_shapes[i]
            _, af, ah, aw = action_shapes[i]

            cf_p = cf // pF
            lf, lh, lw = lf // pF, lh // pH, lw // pW
            nc_i, nl_i, ns_i, na_i = n_ctx[i], n_lat[i], n_state[i], n_act[i]

            lat_t = _expand_frame(torch.arange(1, lf + 1), lh, lw) // chunk_size
            act_t = _expand_frame(torch.arange(1, af + 1), ah, aw) // chunk_size
            state_t = _expand_frame(
                torch.cat([torch.zeros(cf_p, dtype=L), torch.arange(1, lf + 1, dtype=L)]),
                sh, sw) // chunk_size
            ctx_z = torch.zeros(nc_i, dtype=L)

            # Fill each segment at its aligned offset
            # (offset, count, frame_ids, time_ids, noise_id)
            seg_data = [
                (offsets[0], nc_i,  ctx_z,          ctx_z,         1),  # context
                (offsets[1], nl_i,  lat_t*5+1,      lat_t*3+1,     1),  # latent clean
                (offsets[2], nl_i,  lat_t*5+4,      lat_t*3+1,     0),  # latent noisy
                (offsets[3], ns_i,  state_t*5+2,    state_t*3+1,   1),  # state
                (offsets[4], na_i,  act_t*5,        act_t*3,       1),  # action clean
                (offsets[5], na_i,  act_t*5+3,      act_t*3,       0),  # action noisy
            ]
            for off, n, f, t, nz in seg_data:
                all_frame[i, off:off+n] = f
                all_time[i, off:off+n] = t
                all_noise[i, off:off+n] = nz
                all_valid[i, off:off+n] = True

        all_frame = all_frame.to(device)
        all_time = all_time.to(device)
        all_noise = all_noise.to(device)
        all_valid = all_valid.to(device)

        # Self-attention mask
        mask_mod = FlexAttnFunc._get_mask_mod(all_valid, all_frame, all_time, all_noise)
        FlexAttnFunc.attention_mask = FlexAttnFunc.compiled_create_block_mask(
            mask_mod, B, 1, seq_len, seq_len, device=device, _compile=True)

        # Cross-attention mask. Per-sample text validity when a tokenizer
        # keep-mask is given: short prompts then only attend their real
        # tokens instead of all `text_len` (incl. ~500 pad) positions.
        text_padded = _pad_to_block(text_len)
        text_valid = torch.zeros((B, text_padded), dtype=torch.bool, device=device)
        if text_mask is None:
            text_valid[:, :text_len] = True
        else:
            tm = text_mask.to(device=device, dtype=torch.bool)
            L = min(text_len, tm.shape[1])
            text_valid[:, :L] = tm[:, :L]
            # Never leave a fully-masked row (would NaN the softmax).
            empty = ~text_valid.any(dim=1)
            if empty.any():
                text_valid[empty, 0] = True
        cross_mod = FlexAttnFunc._get_cross_mask_mod(all_valid, text_valid)
        FlexAttnFunc.cross_attention_mask = FlexAttnFunc.compiled_create_block_mask(
            cross_mod, B, 1, seq_len, text_padded, device=device, _compile=True)

    @staticmethod
    @torch.no_grad()
    def _get_cross_mask_mod(q_valid, kv_valid):
        def mask(b, h, q_idx, kv_idx):
            return q_valid[b, q_idx] & kv_valid[b, kv_idx]
        return mask

    @staticmethod
    @torch.no_grad()
    def _get_mask_mod(valid, frame_ids, time_ids, noise_ids):
        def clean2clean(b, h, q, kv):
            return (noise_ids[b, q] == 1) & (noise_ids[b, kv] == 1) & \
                   (frame_ids[b, kv] <= frame_ids[b, q])

        def noise2clean(b, h, q, kv):
            return (noise_ids[b, q] == 0) & (noise_ids[b, kv] == 1) & \
                   (time_ids[b, kv] < time_ids[b, q])

        def noise2noise(b, h, q, kv):
            return (noise_ids[b, q] == 0) & (noise_ids[b, kv] == 0) & \
                   (frame_ids[b, kv] == frame_ids[b, q])

        def valid_mask(b, h, q, kv):
            return valid[b, q] & valid[b, kv]

        return and_masks(or_masks(clean2clean, noise2clean, noise2noise), valid_mask)
