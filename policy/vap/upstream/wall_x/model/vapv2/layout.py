"""Static joint sparsity + forward-local text/history validity.

Text/Memory/I are a bidirectional clean prefix. Generated routing is inherited
verbatim, including robust-A0 roles. Batch tensors never enter the global cache.
"""
from functools import lru_cache
import torch

@lru_cache(maxsize=32)
def _structural_block_mask(base, text_len, device_type, device_index):
    from torch.nn.attention.flex_attention import create_block_mask
    device = torch.device(device_type, device_index)
    base_mod = base.flex_block_mask(device).mask_mod
    n = base.sequence_length
    condition_end = text_len + base.condition_video_tokens
    def mask_mod(b, h, q, k):
        old = base_mod(b, h, (q - text_len).clamp(0, n - 1), (k - text_len).clamp(0, n - 1))
        allowed = torch.where(q < condition_end, k < condition_end, (k < text_len) | old)
        return (q < n + text_len) & (k < n + text_len) & allowed
    return create_block_mask(
        mask_mod, B=None, H=None, Q_LEN=n + text_len, KV_LEN=n + text_len,
        device=device, BLOCK_SIZE=128, _compile=True,
    )

class JointConditionLayout:
    def __init__(self, base, text_valid, memory_valid=None):
        if text_valid.ndim != 2 or text_valid.dtype != torch.bool or text_valid.shape[1] == 0:
            raise ValueError("Text validity must be nonempty bool [B,L]")
        from wall_x.model.vap.history_mask import MemoryValidityLayout
        if isinstance(base, MemoryValidityLayout):
            if memory_valid is not None:
                raise ValueError("Memory validity was provided twice")
            memory_valid, base = base.valid, base.base
        self.base = base
        b, self.text_len = text_valid.shape
        self.condition_end = self.text_len + base.condition_video_tokens
        self.sequence_length = self.text_len + base.sequence_length
        self.text_valid = text_valid.detach().clone()
        self.memory_valid = None
        valid = torch.ones(b, self.sequence_length, dtype=torch.bool, device=text_valid.device)
        valid[:, :self.text_len] = self.text_valid
        if memory_valid is not None:
            if (memory_valid.dtype != torch.bool or memory_valid.ndim != 2
                    or memory_valid.shape[0] != b or memory_valid.device != text_valid.device):
                raise ValueError("Memory validity must be bool [B,M] on text device")
            if not 0 < memory_valid.shape[1] < base.condition_video_tokens:
                raise ValueError("Memory slots must precede nonempty current I")
            self.memory_valid = memory_valid.detach().clone()
            valid[:, self.text_len:self.text_len + memory_valid.shape[1]] = self.memory_valid
        self.valid = valid
        snapshot, n = valid, self.sequence_length
        def score_mod(score, b, h, q, k):
            q_valid = snapshot[b, q.clamp(max=n - 1)]
            k_valid = snapshot[b, k.clamp(max=n - 1)]
            allowed = torch.where(q_valid, k_valid, q == k)
            return torch.where(allowed & (q < n) & (k < n), score, float("-inf"))
        self.flex_score_mod = score_mod

    def attention_mask(self, device=None):
        device = self.valid.device if device is None else device
        n, t, c = self.sequence_length, self.text_len, self.condition_end
        structure = torch.zeros(n, n, dtype=torch.bool, device=device)
        structure[t:, t:] = self.base.attention_mask(device=device)
        structure[:, :t] = True
        structure[:c, :c] = True
        structure[:c, c:] = False
        valid = self.valid.to(device)
        allowed = structure[None] & valid[:, None, :]
        return torch.where(valid[:, :, None], allowed, torch.eye(n, dtype=torch.bool, device=device)[None])

    def flex_block_mask(self, device):
        device = torch.device(device)
        return _structural_block_mask(self.base, self.text_len, device.type, device.index)
