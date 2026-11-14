"""Bound temporary gradient buffers without changing native reduction math.

The native BF16 optimizer exposes entire flat parameter groups. DeepSpeed's
fallback batches tensors but does not split an oversized tensor. Split those
contiguous tensors into views, then use the engine's own cast/reduce/average/
copy implementation for each bounded bucket. Optimizer groups stay intact.
"""
from contextlib import contextmanager

import torch


DEFAULT_REDUCTION_ELEMENTS = 33554432


def _limit(value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError('Gradient reduction element limit must be a positive integer')
    return value


def bounded_buckets(tensors, elements):
    """Yield bounded lists of views, in the original element order."""
    elements = _limit(elements)
    pending = []
    used = 0
    identity = None
    for tensor in tensors:
        if not torch.is_tensor(tensor) or tensor.layout != torch.strided or not tensor.is_contiguous():
            raise RuntimeError('Bounded BF16 reduction requires contiguous dense gradient tensors')
        current = (tensor.device, tensor.dtype)
        if identity is not None and identity != current:
            raise RuntimeError('A native gradient bucket must have one device and dtype')
        identity = current
        flat = tensor.view(-1)
        offset = 0
        while offset < flat.numel():
            length = min(elements - used, flat.numel() - offset)
            pending.append(flat.narrow(0, offset, length))
            used += length
            offset += length
            if used == elements:
                yield pending
                pending = []
                used = 0
    if pending:
        yield pending


@contextmanager
def bounded_native_reduction(engine, elements):
    """Temporarily replace only this engine's tensor-batching operation.

    Native allreduce_and_copy still owns communication dtype, pre/postscale,
    process group, averaging and write-back. Class/global state is untouched.
    """
    elements = _limit(elements)
    if 'allreduce_no_retain' in engine.__dict__:
        raise RuntimeError('Refusing an unreviewed or nested gradient reduction override')
    if not callable(getattr(engine, 'allreduce_and_copy', None)) or not callable(getattr(engine, 'allreduce_no_retain', None)):
        raise RuntimeError('Native engine lacks the reviewed gradient reduction interface')

    def reduce(bucket, dp_group, numel_per_bucket=500000000, dp_world_size=None):
        limit = min(elements, _limit(numel_per_bucket))
        for chunk in bounded_buckets(bucket, limit):
            engine.allreduce_and_copy(chunk, dp_group, dp_world_size)

    engine.allreduce_no_retain = reduce
    try:
        yield
    finally:
        del engine.allreduce_no_retain


def finite_gradients(gradients, elements):
    """Preserve the exact finite predicate with bounded temporary bool masks."""
    for gradient in gradients:
        for bucket in bounded_buckets([gradient], elements):
            if not all(bool(torch.isfinite(view).all()) for view in bucket):
                return False
    return True
