from contextlib import nullcontext
import os
import random
import numpy as np
import torch

class Backend:

    def __init__(self, kind='npu', local_rank=0):
        if kind != 'npu':
            raise ValueError('Production training requires the NPU backend')
        import torch_npu
        os.environ['CLOUDROBO_WM_REAL_ROPE'] = '1'
        self.api = torch.npu
        if not self.api.is_available() or not 0 <= local_rank < self.api.device_count():
            raise ValueError('Requested logical NPU is unavailable')
        self.api.set_device(local_rank)
        self.kind = 'npu'
        self.device = torch.device(f'npu:{local_rank}')
        self.communication = 'hccl'

    def autocast(self, enabled=True):
        return torch.autocast('npu', dtype=torch.bfloat16, enabled=enabled)

    def synchronize(self):
        self.api.synchronize()

    def peak_memory(self):
        return int(self.api.max_memory_allocated())

    def rng(self):
        state = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(), backend=self.kind)
        state.update(rng_schema='cloudrobo_wm.npu.local_rng.v2', local_device_index=self.device.index, device=self.api.get_rng_state(self.device))
        return state

    def validate_rng_state(self, state):
        """Validate before any model/optimizer/RNG mutation during full resume."""
        if state['backend'] != self.kind:
            raise ValueError('Full resume cannot cross device backends; use weight migration')
        if state.get('rng_schema') != 'cloudrobo_wm.npu.local_rng.v2':
            raise ValueError('NPU full resume requires local RNG schema v2; legacy all-device RNG checkpoints require weights-only initialization')
        if state.get('local_device_index') != self.device.index:
            raise ValueError('NPU full resume requires the saved logical LOCAL_RANK mapping')
        value = state.get('device')
        if not torch.is_tensor(value) or value.dtype != torch.uint8 or value.device.type != 'cpu':
            raise ValueError('NPU local RNG state must be a CPU ByteTensor')

    def set_rng(self, state):
        self.validate_rng_state(state)
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch'])
        self.api.set_rng_state(state['device'], self.device)

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.random.default_generator.manual_seed(seed)
        self.api.set_device(self.device)
        self.api.manual_seed(seed)

def attention(q, k, v, mask=None, **kwargs):
    """B,H,Q,D; bool True means ALLOWED at this API, including rectangular masks."""
    if q.device.type != 'npu':
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, **kwargs)
    import torch_npu
    if kwargs.get('dropout_p', 0.0) or kwargs.get('is_causal', False):
        raise ValueError('NPU attention requires an explicit exact mask and zero dropout')
    if mask is not None and mask.dtype != torch.bool:
        raise TypeError('NPU adapter requires boolean visibility masks')
    if q.dtype not in (torch.bfloat16, torch.float16):
        scale = kwargs.get('scale') or q.shape[-1] ** (-0.5)
        logits = q.float() @ k.float().transpose(-2, -1) * scale
        valid_rows = None if mask is None else mask.any(-1, keepdim=True)
        if mask is not None:
            logits = logits.masked_fill(~mask, -torch.inf)
            logits = torch.where(valid_rows, logits, 0)
        weights = logits.softmax(-1)
        if mask is not None:
            weights = torch.where(valid_rows, weights, 0)
        return (weights @ v.float()).to(q.dtype)
    valid_rows = None if mask is None else mask.any(-1, keepdim=True)
    safe = None if mask is None else mask | ~valid_rows & (torch.arange(k.shape[-2], device=k.device) == 0)
    blocked = None if safe is None else (~safe).contiguous()
    result = torch_npu.npu_fusion_attention(q.contiguous(), k.contiguous(), v.contiguous(), head_num=q.shape[1], input_layout='BNSD', atten_mask=blocked, scale=kwargs.get('scale') or q.shape[-1] ** (-0.5), keep_prob=1.0, pre_tockens=2147483647, next_tockens=2147483647, sparse_mode=0)[0]
    return result if mask is None else torch.where(valid_rows, result, 0)

def real_rope(x, freqs, num_heads):
    b, s, d = x.shape
    pairs = x.float().reshape(b, s, num_heads, -1, 2)
    if freqs.is_complex():
        if freqs.device.type != 'cpu':
            raise ValueError('Complex frequency conversion is CPU-only')
        f = torch.view_as_real(freqs.resolve_conj()).float().to(x.device)
    else:
        f = freqs.float().reshape(*freqs.shape[:-1], -1, 2)
    even, odd = (pairs[..., 0], pairs[..., 1])
    c, sn = (f[..., 0], f[..., 1])
    return torch.stack((even * c - odd * sn, even * sn + odd * c), -1).reshape(b, s, d).to(x.dtype)
