# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import signal
import types
from typing import Any, Optional

import torch
import torch.distributed


def get_device(local_rank: Optional[int] = None) -> torch.device:
    """
    Get the appropriate torch device based on the distributed backend.

    Args:
        local_rank: The local rank, used to specify the CUDA device index for NCCL.
                    If None, uses the default CUDA device.

    Returns:
        The torch.device ('cuda' for NCCL, 'cpu' for Gloo).

    Raises:
        RuntimeError: If the distributed backend is neither 'nccl' nor 'gloo'.
    """
    backend = torch.distributed.get_backend()
    if backend == "nccl":
        if local_rank is None:
            device = torch.device("cuda")
        else:
            device = torch.device(f"cuda:{local_rank}")
    elif backend == "gloo":
        device = torch.device("cpu")
    else:
        raise RuntimeError
    return device


def all_gather_item(
    item: Any,
    dtype: torch.dtype,
    group: Optional[torch.distributed.ProcessGroup] = None,
    async_op: bool = False,
    local_rank: Optional[int] = None,
) -> list[Any]:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return [item]

    device = get_device(local_rank)

    if group is not None:
        group_size = group.size()
    else:
        group_size = torch.distributed.get_world_size()

    tensor = torch.tensor([item], device=device, dtype=dtype)
    output_tensors = [torch.zeros(1, dtype=tensor.dtype, device=tensor.device) for _ in range(group_size)]
    torch.distributed.all_gather(output_tensors, tensor, group, async_op)
    output = [elem.item() for elem in output_tensors]
    return output


class DistributedSignalHandler:
    """Context manager to handle signals gracefully in a distributed setting."""

    def __init__(self, sig: int = signal.SIGTERM) -> None:
        self.sig = sig
        self._signal_received = False
        self.released = False
        self.original_handler = None

    def signals_received(self) -> list[bool]:
        all_received = all_gather_item(self._signal_received, dtype=torch.int32)
        return all_received

    def __enter__(self) -> "DistributedSignalHandler":
        self._signal_received = False
        self.released = False
        self.original_handler = signal.getsignal(self.sig)

        def handler(signum: int, frame: Optional[Any]) -> None:
            logging.info("Received signal {}, initiating graceful stop".format(signum))
            self._signal_received = True

        signal.signal(self.sig, handler)
        logging.info("Signal handler installed for {}".format(self.sig))

        return self

    def __exit__(
        self, exc_type: Optional[type], exc_val: BaseException | None, exc_tb: types.TracebackType | None
    ) -> None:  # noqa: E501
        self.release()

    def release(self) -> bool:
        if self.released:
            return False

        signal.signal(self.sig, self.original_handler)
        self.released = True
        return True
