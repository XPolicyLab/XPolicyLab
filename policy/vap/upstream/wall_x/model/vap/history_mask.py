"""Cached structural sparsity plus per-forward Memory validity score masking.

Only the layout/device/prefix length enter the global BlockMask cache. Validity
is an immutable per-forward tensor captured by score_mod, never globally cached.
"""
from functools import lru_cache

import torch


@lru_cache(maxsize=32)
def _cached_memory_block_mask(base, memory_tokens, device_type, device_index):
    from torch.nn.attention.flex_attention import create_block_mask

    device = torch.device(device_type, device_index)
    base_mod = base.flex_block_mask(device).mask_mod
    n = base.sequence_length

    def mask_mod(b, h, q, k):
        original = base_mod(b, h, q.clamp(max=n-1), k.clamp(max=n-1))
        # All possible Memory slots exist structurally. Memory is read-only
        # w.r.t. the main sequence; self edges include invalid-query fallback.
        allowed = torch.where(q < memory_tokens, k < memory_tokens, original)
        return (q < n) & (k < n) & allowed

    return create_block_mask(
        mask_mod, B=None, H=None, Q_LEN=n, KV_LEN=n, device=device,
        BLOCK_SIZE=128, _compile=True,
    )


class MemoryValidityLayout:
    def __init__(self, base, valid):
        if valid.ndim != 2 or valid.dtype != torch.bool:
            raise ValueError("Memory validity must be bool [B,M]")
        if not 0 < valid.shape[1] < base.condition_video_tokens:
            raise ValueError("Memory prefix must precede nonempty current I")
        self.base = base
        self.valid = valid.detach().clone()  # immutable through GC backward
        valid_snapshot = self.valid
        m = valid_snapshot.shape[1]

        def score_mod(score, b, h, q, k):
            q_valid = (q >= m) | valid_snapshot[b, q.clamp(max=m-1)]
            k_valid = (k >= m) | valid_snapshot[b, k.clamp(max=m-1)]
            # Unlike mask_mod, score_mod runs on fully allowed blocks too.
            # No empty rows: invalid queries retain only their own key.
            allowed = torch.where(q_valid, k_valid, q == k)
            return torch.where(allowed, score, float("-inf"))

        self.flex_score_mod = score_mod

    def __getattr__(self, name):
        return getattr(self.base, name)

    def attention_mask(self, device=None):
        base = self.base.attention_mask(device=device)
        b, m = self.valid.shape
        n = base.shape[0]
        valid = torch.cat((self.valid.to(device), torch.ones(b, n-m, dtype=torch.bool, device=device)), 1)
        q = torch.arange(n, device=device)
        allowed = base[None] & valid[:, None, :]
        allowed = torch.where((q < m)[None, :, None],
                              ((q < m)[None, None, :] & valid[:, None, :]), allowed)
        return torch.where(valid[:, :, None], allowed, torch.eye(n, dtype=torch.bool, device=device)[None])

    def flex_block_mask(self, device):
        device = torch.device(device)
        return _cached_memory_block_mask(
            self.base, self.valid.shape[1], device.type, device.index,
        )
