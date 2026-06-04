# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import contextlib
from typing import Any, Literal

import torch
from robo_orchard_core.datatypes.dataclass import TensorToMixin

__all__ = ["to_device", "switch_model_mode"]


def to_device(data: Any, device: torch.device) -> Any:
    """Recursively moves tensors in a nested data structure to a specified device.

    This function traverses common nested data structures such as dictionaries,
    lists, and tuples. For each `torch.Tensor` or `TensorToMixin` value it
    finds, it applies the `.to(device)` method. Other data types within the
    structure are returned unmodified.

    Args:
        data (Any): The data structure to move. It can be a `torch.Tensor`,
            a dictionary, a list, a tuple, or other basic types.
        device (torch.device): The target device to which tensors should be
            moved (e.g., `torch.device('cuda:0')` or `torch.device('cpu')`).

    Returns:
        Any: A new data structure of the same type and shape as `data`, but
             with all `torch.Tensor` and `TensorToMixin` instances located on
             the target device.
    """  # noqa: E501
    # Base case: If data is a tensor, move it directly to the device.
    if isinstance(data, torch.Tensor):
        if data.device == device:
            return data
        return data.to(device)

    elif isinstance(data, TensorToMixin):
        return data.to(device=device)

    # Recursive step for dictionaries: apply to_device to each value.
    elif isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}

    # Recursive step for lists and tuples: apply to_device to each item.
    # This preserves the original container type (list or tuple).
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(item, device) for item in data)

    # If data is not a tensor or a supported container, return it as is.
    else:
        return data


@contextlib.contextmanager
def switch_model_mode(
    model: torch.nn.Module, target_mode: Literal["eval", "train"] = "eval"
):
    """A context manager to temporarily switch a model to a target mode.

    This context manager ensures that the model is in the `target_mode`
    within the `with` block, and restores its original training state
    (`train()` or `eval()`) after exiting the block.

    Args:
        model (torch.nn.Module): The PyTorch model to manage.
        target_mode (Literal["eval", "train"], optional): The mode to switch
            the model to within the context. Defaults to "eval".

    Yields:
        None: This context manager does not yield any value.
    """
    original_mode_is_training = model.training

    if (target_mode == "train" and original_mode_is_training) or (
        target_mode == "eval" and not original_mode_is_training
    ):
        yield
        return

    try:
        if target_mode == "eval":
            model.eval()
        elif target_mode == "train":
            model.train()
        else:
            raise ValueError(
                f"Invalid target_mode: '{target_mode}'. "
                "Must be 'eval' or 'train'."
            )
        yield
    finally:
        if original_mode_is_training:
            model.train()
        else:
            model.eval()
