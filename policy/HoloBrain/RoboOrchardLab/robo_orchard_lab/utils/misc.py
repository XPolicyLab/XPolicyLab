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

from typing import Any, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["as_sequence", "to_tensor", "stack_batch"]


def as_sequence(
    objs: Optional[Union[Any, Sequence[Any]]],
    check_type: bool = False,
    required_types: Union[Type, Tuple[Type, ...]] = (),
) -> Sequence[Any]:
    """Make input as sequence.

    Converts an input object or a sequence of objects into a sequence (list).

    If `objs` is not already a sequence, it wraps `objs` into a single-element
    list. Optionally, checks if each element in the resulting sequence matches
    the specified `required_types`.

    Args:
        objs (Optional[Union[Any, Sequence[Any]]]): An object or a sequence of
            objects to be converted into a sequence. If `None`, an empty list
            is returned.
        check_type (bool): If `True`, checks each element in the resulting
            sequence to ensure it is an instance of `required_types`.
            Defaults to `False`.
        required_types (Union[Type, Tuple[Type, ...]]): A type or tuple of
            types that each element in the sequence should match.
            Only checked if `check_type` is `True`. Defaults to an empty tuple.

    Returns:
        Sequence[Any]: A sequence (list) of objects, with `objs` wrapped in a
        list if it was a single object or `None`.

    Raises:
        TypeError: If `check_type` is `True` and an element in `objs` does
        not match `required_types`.

    Examples:
        >>> as_sequence(5)
        [5]

        >>> as_sequence([1, 2, 3])
        [1, 2, 3]

        >>> as_sequence(None)
        []

        >>> as_sequence("hello", check_type=True, required_types=str)
        ["hello"]

        >>> as_sequence([1, "text"], check_type=True,
        >>>              required_types=(int, str))
        [1, "text"]

        >>> as_sequence([1, 2.5], check_type=True, required_types=int)
        TypeError: The 2-th entry of sequence should be instance of type
        <class 'int'>, but get <class 'float'>
    """

    if objs is None:
        return []

    if not isinstance(objs, (list, tuple)):
        objs = [objs]

    if check_type:
        for idx, obj_i in enumerate(objs):
            if not isinstance(obj_i, required_types):
                raise TypeError(
                    "The {}-th entry of sequence should be instance of type {}, but get {}".format(  # noqa: E501
                        idx + 1, required_types, type(obj_i)
                    )
                )

    return objs


def to_tensor(
    data: Union[torch.Tensor, np.ndarray, Sequence, int, float],
) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        if data.dtype is np.dtype("float64"):
            data = data.astype(np.float32)
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f"type {type(data)} cannot be converted to tensor.")


def stack_batch(
    tensor_list: List[torch.Tensor],
    pad_size_divisor: int = 1,
    pad_value: Union[int, float] = 0,
) -> torch.Tensor:
    """Stack batch.

    Stack multiple tensors to form a batch and pad the tensor to the max
    shape use the right bottom padding mode in these images. If
    ``pad_size_divisor > 0``, add padding to ensure the shape of each dim is
    divisible by ``pad_size_divisor``.

    Args:
        tensor_list (List[Tensor]): A list of tensors with the same dim.
        pad_size_divisor (int): If ``pad_size_divisor > 0``, add padding
            to ensure the shape of each dim is divisible by
            ``pad_size_divisor``. This depends on the model, and many
            models need to be divisible by 32. Defaults to 1
        pad_value (int, float): The padding value. Defaults to 0.

    Returns:
       torch.Tensor: The n dim tensor.

    """
    assert isinstance(tensor_list, list), (
        f"Expected input type to be list, but got {type(tensor_list)}"
    )
    assert tensor_list, "`tensor_list` could not be an empty list"
    assert len({tensor.ndim for tensor in tensor_list}) == 1, (
        f"Expected the dimensions of all tensors must be the same, "
        f"but got {[tensor.ndim for tensor in tensor_list]}"
    )

    dim = tensor_list[0].dim()
    num_img = len(tensor_list)
    all_sizes: torch.Tensor = torch.Tensor(
        [tensor.shape for tensor in tensor_list]
    )
    max_sizes = (
        torch.ceil(torch.max(all_sizes, dim=0)[0] / pad_size_divisor)
        * pad_size_divisor
    )
    padded_sizes = max_sizes - all_sizes
    # The first dim normally means channel,  which should not be padded.
    padded_sizes[:, 0] = 0
    if padded_sizes.sum() == 0:
        return torch.stack(tensor_list)
    # `pad` is the second arguments of `F.pad`. If pad is (1, 2, 3, 4),
    # it means that padding the last dim with 1(left) 2(right), padding the
    # penultimate dim to 3(top) 4(bottom). The order of `pad` is opposite of
    # the `padded_sizes`. Therefore, the `padded_sizes` needs to be reversed,
    # and only odd index of pad should be assigned to keep padding "right" and
    # "bottom".
    pad = torch.zeros(num_img, 2 * dim, dtype=torch.int)
    pad[:, 1::2] = padded_sizes[:, range(dim - 1, -1, -1)]
    batch_tensor = []
    for idx, tensor in enumerate(tensor_list):
        batch_tensor.append(
            F.pad(tensor, tuple(pad[idx].tolist()), value=pad_value)
        )
    return torch.stack(batch_tensor)
