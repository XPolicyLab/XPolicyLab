import functools
import importlib
from contextlib import contextmanager
from itertools import accumulate
from types import ModuleType

import torch
from torch.utils._pytree import tree_leaves


@functools.cache
def _load_nvtx() -> ModuleType | None:
    """Load optional NVTX bindings only when profiling is first used."""

    try:
        return importlib.import_module("nvtx")
    except ModuleNotFoundError as error:
        if error.name != "nvtx":
            raise
        return None


def get_accumulate_tensor(x):
    return torch.tensor(list(accumulate(x, initial=0)), dtype=torch.int32)


def tensor_storage_bytes(value):
    """Return logical tensor bytes in a nested DataLoader batch."""
    return sum(x.numel() * x.element_size() for x in tree_leaves(value) if isinstance(x, torch.Tensor))


def nvtx_profiled_function(fn=None, *, label=None, color=None):
    if fn is None:
        return functools.partial(nvtx_profiled_function, label=label, color=color)
    if isinstance(fn, str):

        @contextmanager
        def profile_block():
            nvtx = _load_nvtx()
            if nvtx is None:
                yield
                return
            nvtx.push_range(fn, color=color)
            try:
                yield
            finally:
                nvtx.pop_range()

        return profile_block()

    range_label = label or fn.__qualname__

    @functools.wraps(fn)
    def decorator(*args, **kwargs):
        nvtx = _load_nvtx()
        if nvtx is None:
            return fn(*args, **kwargs)
        nvtx.push_range(range_label, color=color)
        try:
            return fn(*args, **kwargs)
        finally:
            nvtx.pop_range()

    return decorator
