"""Inference-only BF16 RoPE preserving eager intermediate rounding.

Adapted from colin's V1 acceleration (2e5d699). Four-axis phases are still
constructed by the existing rope.py; no positional convention is changed.
Unlike the training xcompute fused kernel this explicitly rounds each BF16
product before addition, and disables FMA contraction. Inputs are not mutated.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rotate(Q, K, C, S, OQ, OK, N: tl.constexpr, SEQ: tl.constexpr,
            HEADS: tl.constexpr, D: tl.constexpr,
            QB: tl.constexpr, QS: tl.constexpr, QH: tl.constexpr,
            KB: tl.constexpr, KS: tl.constexpr, KH: tl.constexpr,
            CB: tl.constexpr, CS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pair = i % (D // 2)
    head = i // (D // 2) % HEADS
    seq = i // (D // 2 * HEADS) % SEQ
    batch = i // (D // 2 * HEADS * SEQ)
    mask = i < N
    co = tl.load(C + batch * CB + seq * CS + pair, mask, other=0).to(tl.float32)
    si = tl.load(S + batch * CB + seq * CS + pair, mask, other=0).to(tl.float32)
    qoff = batch * QB + seq * QS + head * QH + pair * 2
    koff = batch * KB + seq * KS + head * KH + pair * 2
    a = tl.load(Q + qoff, mask, other=0).to(tl.float32)
    b = tl.load(Q + qoff + 1, mask, other=0).to(tl.float32)
    ac = (a * co).to(tl.bfloat16).to(tl.float32)
    bs = (b * si).to(tl.bfloat16).to(tl.float32)
    bc = (b * co).to(tl.bfloat16).to(tl.float32)
    ass = (a * si).to(tl.bfloat16).to(tl.float32)
    tl.store(OQ + i * 2, ac - bs, mask)
    tl.store(OQ + i * 2 + 1, bc + ass, mask)
    a = tl.load(K + koff, mask, other=0).to(tl.float32)
    b = tl.load(K + koff + 1, mask, other=0).to(tl.float32)
    ac = (a * co).to(tl.bfloat16).to(tl.float32)
    bs = (b * si).to(tl.bfloat16).to(tl.float32)
    bc = (b * co).to(tl.bfloat16).to(tl.float32)
    ass = (a * si).to(tl.bfloat16).to(tl.float32)
    tl.store(OK + i * 2, ac - bs, mask)
    tl.store(OK + i * 2 + 1, bc + ass, mask)


def rotate(q, k, cos, sin):
    if torch.is_grad_enabled():
        raise RuntimeError("VAPv2 inference RoPE requires no_grad")
    if (not q.is_cuda or q.dtype != torch.bfloat16 or k.dtype != q.dtype
            or k.device != q.device or q.shape != k.shape or q.ndim != 4
            or q.stride(-1) != 1 or k.stride(-1) != 1):
        raise ValueError("Expected CUDA BF16 [B,S,H,D] Q/K, contiguous channels")
    expected = (*q.shape[:2], q.shape[-1] // 2)
    if (q.shape[-1] % 2 or cos.shape != expected or sin.shape != expected
            or cos.dtype != q.dtype or sin.dtype != q.dtype
            or cos.device != q.device or sin.device != q.device
            or cos.stride(-1) != 1 or sin.stride() != cos.stride()):
        raise ValueError("Expected matching BF16 [B,S,D/2] phases")
    oq = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    ok = torch.empty_like(oq)
    n = q.numel() // 2
    _rotate[(triton.cdiv(n, 256),)](
        q, k, cos, sin, oq, ok, n, q.shape[1], q.shape[2], q.shape[3],
        *q.stride()[:3], *k.stride()[:3], *cos.stride()[:2], 256,
        enable_fp_fusion=False,
    )
    return oq, ok
