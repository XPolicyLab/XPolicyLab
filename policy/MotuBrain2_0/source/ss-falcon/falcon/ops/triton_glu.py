"""Triton-fused GLU activation kernels (SwiGLU, ReGLU).

Single-pass forward and backward: no torch.cat, no intermediate buffers.
Forward:  read [M, 2F] -> write [M, F]           = 6*M*F bytes (bf16)
Backward: read grad[M,F] + input[M,2F] -> write grad_input[M,2F] = 10*M*F bytes (bf16)
"""

import torch
import triton
import triton.language as tl


def _configs():
    return [triton.Config({"BLOCK_SIZE": bs}, num_warps=nw) for bs in [512, 1024, 2048, 4096] for nw in [4, 8]]


# ---------------------------------------------------------------------------
# SwiGLU: silu(a) * b = a * sigmoid(a) * b
# ---------------------------------------------------------------------------


@triton.autotune(configs=_configs(), key=["F"])
@triton.jit
def _swiglu_fwd_kernel(X, OUT, F, stride_x, stride_out, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    x_row = X + row * stride_x
    out_row = OUT + row * stride_out
    for off in range(0, F, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        a = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(x_row + F + cols, mask=mask, other=0.0).to(tl.float32)
        sig = 1.0 / (1.0 + tl.exp(-a))
        tl.store(out_row + cols, (a * sig * b).to(out_row.dtype.element_ty), mask=mask)


@triton.autotune(configs=_configs(), key=["F"])
@triton.jit
def _swiglu_bwd_kernel(G, X, DX, F, stride_g, stride_x, stride_dx, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    g_row = G + row * stride_g
    x_row = X + row * stride_x
    dx_row = DX + row * stride_dx
    for off in range(0, F, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        g = tl.load(g_row + cols, mask=mask, other=0.0).to(tl.float32)
        a = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(x_row + F + cols, mask=mask, other=0.0).to(tl.float32)
        sig = 1.0 / (1.0 + tl.exp(-a))
        da = g * sig * (1.0 + a * (1.0 - sig)) * b
        db = g * a * sig
        tl.store(dx_row + cols, da.to(dx_row.dtype.element_ty), mask=mask)
        tl.store(dx_row + F + cols, db.to(dx_row.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# ReGLU: relu(a) * b
# ---------------------------------------------------------------------------


@triton.autotune(configs=_configs(), key=["F"])
@triton.jit
def _reglu_fwd_kernel(X, OUT, F, stride_x, stride_out, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    x_row = X + row * stride_x
    out_row = OUT + row * stride_out
    for off in range(0, F, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        a = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(x_row + F + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_row + cols, tl.where(a > 0.0, a * b, 0.0).to(out_row.dtype.element_ty), mask=mask)


@triton.autotune(configs=_configs(), key=["F"])
@triton.jit
def _reglu_bwd_kernel(G, X, DX, F, stride_g, stride_x, stride_dx, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    g_row = G + row * stride_g
    x_row = X + row * stride_x
    dx_row = DX + row * stride_dx
    for off in range(0, F, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        g = tl.load(g_row + cols, mask=mask, other=0.0).to(tl.float32)
        a = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(x_row + F + cols, mask=mask, other=0.0).to(tl.float32)
        pos = a > 0.0
        tl.store(dx_row + cols, tl.where(pos, g * b, 0.0).to(dx_row.dtype.element_ty), mask=mask)
        tl.store(dx_row + F + cols, tl.where(pos, g * a, 0.0).to(dx_row.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------


def triton_swiglu_fwd(x):
    M, two_F = x.shape
    F = two_F // 2
    out = torch.empty(M, F, device=x.device, dtype=x.dtype)
    _swiglu_fwd_kernel[(M,)](x, out, F, x.stride(0), out.stride(0))
    return out


def triton_swiglu_bwd(grad_out, x):
    M, two_F = x.shape
    F = two_F // 2
    dx = torch.empty_like(x)
    _swiglu_bwd_kernel[(M,)](grad_out, x, dx, F, grad_out.stride(0), x.stride(0), dx.stride(0))
    return dx


def triton_reglu_fwd(x):
    M, two_F = x.shape
    F = two_F // 2
    out = torch.empty(M, F, device=x.device, dtype=x.dtype)
    _reglu_fwd_kernel[(M,)](x, out, F, x.stride(0), out.stride(0))
    return out


def triton_reglu_bwd(grad_out, x):
    M, two_F = x.shape
    F = two_F // 2
    dx = torch.empty_like(x)
    _reglu_bwd_kernel[(M,)](grad_out, x, dx, F, grad_out.stride(0), x.stride(0), dx.stride(0))
    return dx


# ---------------------------------------------------------------------------
# Autograd Functions
# ---------------------------------------------------------------------------


class TritonSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, fp8_input_store):
        input_save = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        ctx.save_for_backward(input_save)
        ctx.ori_dtype = input.dtype
        ctx.fp8 = fp8_input_store
        return triton_swiglu_fwd(input)

    @staticmethod
    def backward(ctx, grad_output):
        (saved,) = ctx.saved_tensors
        inp = saved.to(ctx.ori_dtype) if ctx.fp8 else saved
        return triton_swiglu_bwd(grad_output, inp), None


class TritonReGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return triton_reglu_fwd(input)

    @staticmethod
    def backward(ctx, grad_output):
        (inp,) = ctx.saved_tensors
        return triton_reglu_bwd(grad_output, inp)


# ---------------------------------------------------------------------------
# Drop-in replacements for fused_kernel.py
# ---------------------------------------------------------------------------


def triton_bias_swiglu_impl(input, fp8_input_store=False):
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    input = input.view(-1, ori_shape[-1])
    output = TritonSwiGLUFunction.apply(input, fp8_input_store)
    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)


def triton_bias_reglu_impl(input, fp8_input_store=False):
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    input = input.view(-1, ori_shape[-1])
    output = TritonReGLUFunction.apply(input)
    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)
