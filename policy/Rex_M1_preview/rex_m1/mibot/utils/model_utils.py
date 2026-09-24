# Copyright (C) 2026 Xiaomi Corporation.
from functools import wraps
from typing import Any, Callable

import torch


def auto_cast(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(model_self: Any, batch: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                batch[key] = value.to(torch.bfloat16)
        return func(model_self, batch, *args, **kwargs)

    return wrapper
