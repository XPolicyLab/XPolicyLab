import math
from typing import Optional, Union

import torch
import torch.nn as nn

from .config import InitFnType, ModelConfig
from .util import StrEnum

__all__ = ["init_weights", "init_normal", "ModuleType"]


class ModuleType(StrEnum):
    in_module = "in"
    out_module = "out"
    emb = "emb"
    final_out = "final_out"


def init_weights(
    config: ModelConfig,
    module: Union[nn.Linear, nn.Embedding],
    d: Optional[int] = None,
    layer_id: Optional[int] = None,
    std_factor: float = 1.0,
    type_of_module: Optional[ModuleType] = None,
) -> None:
    """
    Initialize weights of a linear or embedding module.

    :param config: The model config.
    :param module: The linear or embedding submodule to initialize.
    :param d: The effective input dimensionality of the weights. This could be smaller than the actual dimensions
        for fused layers.
    :param layer_id: When set, the standard deviation for the "mitchell" method will be adjusted by
        ``1 / sqrt(2 * (layer_id + 1))``.
    """
    d = d if d is not None else config.d_model
    if config.init_fn == InitFnType.normal:
        std = config.init_std * std_factor
        if config.init_cutoff_factor is not None:
            cutoff_value = config.init_cutoff_factor * std
            if hasattr(module, "weight"):
                nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-cutoff_value, b=cutoff_value)
            else:
                nn.init.trunc_normal_(module, mean=0.0, std=std, a=-cutoff_value, b=cutoff_value)
        else:
            if hasattr(module, "weight"):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            else:
                nn.init.normal_(module, mean=0.0, std=std)
    else:
        raise NotImplementedError(config.init_fn)

    if isinstance(module, nn.Linear):
        if module.bias is not None:
            nn.init.zeros_(module.bias)

        if config.init_fn == InitFnType.normal and getattr(module, "_is_residual", False):
            with torch.no_grad():
                module.weight.div_(math.sqrt(2 * config.n_layers))


def init_normal(
    module: Union[nn.Linear, nn.Embedding],
    std: float,
    init_cutoff_factor: Optional[float] = None,
):
    # weights
    if init_cutoff_factor is not None:
        cutoff_value = init_cutoff_factor * std
        if hasattr(module, "weight"):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-cutoff_value, b=cutoff_value)
        else:
            nn.init.trunc_normal_(module, mean=0.0, std=std, a=-cutoff_value, b=cutoff_value)
    else:
        if hasattr(module, "weight"):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        else:
            nn.init.normal_(module, mean=0.0, std=std)

    # biases
    if isinstance(module, nn.Linear) and module.bias is not None:
        nn.init.zeros_(module.bias)