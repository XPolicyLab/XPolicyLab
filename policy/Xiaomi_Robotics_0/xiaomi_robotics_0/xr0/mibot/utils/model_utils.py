# Copyright (C) 2026 Xiaomi Corporation.
from functools import wraps
from typing import Any, Callable

import torch


def auto_cast(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that automatically casts float32 tensors in the batch to bfloat16.

    This is commonly used to reduce memory usage and accelerate training on
    hardware that supports bf16 (e.g., NVIDIA A100/H100).

    Args:
        func: A method whose second positional argument is a ``batch`` dict.

    Returns:
        Wrapped function with automatic float32 → bfloat16 casting.
    """

    @wraps(func)
    def wrapper(model_self: Any, batch: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                batch[key] = value.to(torch.bfloat16)
        return func(model_self, batch, *args, **kwargs)

    return wrapper
