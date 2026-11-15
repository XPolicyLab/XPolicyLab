import contextlib
from typing import Callable

import torch
import torch.nn as nn


def get_autocast(
    precision: str,
) -> contextlib.nullcontext | Callable[[], torch.amp.autocast]:
    if precision in ["amp", "amp_fp16"]:

        def ctx():
            return torch.amp.autocast(dtype=torch.float16)

    elif precision in ["amp_bf16"]:

        def ctx():
            return torch.amp.autocast(dtype=torch.bfloat16)

    else:
        ctx = contextlib.nullcontext

    return ctx


class AbstractEncoder(nn.Module):
    """
    Abstract base class for text encoders.
    """

    @property
    def out_channels(self) -> int:
        raise NotImplementedError

    def encode(self, *args, **kwargs):
        raise NotImplementedError
