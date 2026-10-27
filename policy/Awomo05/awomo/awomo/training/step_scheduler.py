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
from math import ceil
from typing import Optional

from awomo.awomo.training.signal_handler import DistributedSignalHandler
from torch.distributed.checkpoint.stateful import Stateful

logger = logging.getLogger(__name__)


def _calculate_max_steps(
    num_epochs: int,
    epoch_len: Optional[int],
    default_max_steps: int = 9223372036854775807,
) -> int:
    if epoch_len is None:
        return default_max_steps
    return num_epochs * epoch_len


def _calculate_num_epochs(max_steps: Optional[int], epoch_len: Optional[int], default_num_epochs: int = 10) -> int:
    if epoch_len is None or max_steps is None:
        return default_num_epochs
    return ceil(max_steps / epoch_len)


class StepScheduler(Stateful):
    """Scheduler for managing gradient accumulation and checkpointing steps."""

    def __init__(
        self,
        global_batch_size: int,
        local_batch_size: int,
        dp_size: int,
        dataloader: Optional[int],
        ckpt_every_steps: Optional[int] = None,
        save_checkpoint_every_epoch: bool = True,
        val_every_steps: Optional[int] = None,
        log_remote_every_steps: int = 1,
        gc_every_steps: Optional[int] = None,
        start_step: int = 0,
        start_epoch: int = 0,
        num_epochs: Optional[int] = None,
        max_steps: Optional[int] = None,
    ):
        assert (
            global_batch_size % (local_batch_size * dp_size) == 0
        ), f"global_batch_size ({global_batch_size}) must be divisible by local_batch_size * dp_size ({local_batch_size} * {dp_size})"
        self.grad_acc_steps = global_batch_size // (local_batch_size * dp_size)
        assert (
            self.grad_acc_steps >= 1
        ), f"grad_acc_steps ({self.grad_acc_steps}) must be greater than or equal to 1. Please ensure that global_batch_size >= (local_batch_size * dp_size)"
        self.dataloader = dataloader
        self.step = start_step
        assert start_step >= 0, "start_step must be greater than or equal to 0"
        self.epoch = start_epoch
        assert start_epoch >= 0, "start_epoch must be greater than or equal to 0"

        try:
            self.epoch_len = ceil(len(dataloader) / self.grad_acc_steps)
        except:
            self.epoch_len = None

        if num_epochs is None:
            num_epochs = _calculate_num_epochs(
                max_steps,
                self.epoch_len,
            )

        self.num_epochs = num_epochs
        assert (
            num_epochs is None or num_epochs > 0
        ), "num_epochs must be greater than 0 or None if max_steps is provided"

        self.val_every_steps = val_every_steps
        assert val_every_steps is None or val_every_steps > 0, "val_every_steps must be greater than 0 if not None"
        self.log_remote_every_steps = log_remote_every_steps
        assert log_remote_every_steps > 0, "log_remote_every_steps must be greater than 0"
        self.gc_every_steps = gc_every_steps
        assert gc_every_steps is None or gc_every_steps > 0, "gc_every_steps must be greater than 0 if not None"
        if max_steps is None:
            assert self.epoch_len is not None, "epoch_len must be provided if max_steps is not provided"
            max_steps = _calculate_max_steps(self.num_epochs, self.epoch_len)
            logger.info("max_steps not provided; will run for up to {} steps".format(max_steps))
        self.max_steps = max_steps
        assert max_steps > 0, "max_steps must be greater than 0"

        if ckpt_every_steps is None:
            if self.epoch_len is None:
                ckpt_every_steps = self.max_steps // 2
            else:
                ckpt_every_steps = self.epoch_len
            logger.info("ckpt_every_steps not provided; will save checkpoint every {} steps".format(ckpt_every_steps))
        self.ckpt_every_steps = ckpt_every_steps
        self.save_checkpoint_every_epoch = save_checkpoint_every_epoch

        self.sig_handler = DistributedSignalHandler().__enter__()
        self.sigterm_flag = False
        self.debug_print = None

    def _debug(self, message: str) -> None:
        debug_print = self.debug_print
        if debug_print is not None:
            debug_print(message)

    def __iter__(self):
        self._debug(
            f"step_scheduler iter start step={self.step} max_steps={self.max_steps} "
            f"grad_acc_steps={self.grad_acc_steps}"
        )
        if self.step >= self.max_steps:
            self._debug("step_scheduler iter stop: step >= max_steps")
            return
        batch_buffer = []
        self._debug("step_scheduler before dataloader iteration")
        for batch in self.dataloader:
            if self.step < 2 and len(batch_buffer) == 0:
                self._debug(f"step_scheduler got dataloader batch for step={self.step}")
            batch_buffer.append(batch)
            if len(batch_buffer) == self.grad_acc_steps:
                if self.step < 2:
                    self._debug(
                        f"step_scheduler yielding batch_group step={self.step} " f"micro_batches={len(batch_buffer)}"
                    )
                yield batch_buffer
                self.step += 1
                batch_buffer = []
                if self.step >= self.max_steps or self.sigterm_flag:
                    self._debug(
                        f"step_scheduler iter stop after yield step={self.step} " f"sigterm_flag={self.sigterm_flag}"
                    )
                    return
        if batch_buffer:
            self._debug(
                f"step_scheduler yielding tail batch_group step={self.step} " f"micro_batches={len(batch_buffer)}"
            )
            yield batch_buffer
            self.step += 1
        self.epoch += 1
        self._debug(f"step_scheduler epoch incremented to {self.epoch}")

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        if hasattr(getattr(self.dataloader, "sampler", None), "set_epoch"):
            self.dataloader.sampler.set_epoch(epoch)

    @property
    def is_remote_logging_step(self):
        return self.step % self.log_remote_every_steps == 0

    @property
    def is_val_step(self):
        is_val = False
        if self.val_every_steps and self.val_every_steps > 0:
            is_val = self.step % self.val_every_steps == self.val_every_steps - 1
        return (is_val or self.is_ckpt_step) and not self.sigterm_flag

    @property
    def is_ckpt_step(self):
        is_ckpt_step = (self.step % self.ckpt_every_steps) == self.ckpt_every_steps - 1
        is_epoch_boundary = self.save_checkpoint_every_epoch and self.is_last_batch
        return is_ckpt_step or is_epoch_boundary or self.is_last_step or self.sigterm_received

    @property
    def is_gc_step(self):
        return self.gc_every_steps is not None and self.step % self.gc_every_steps == 0

    @property
    def is_last_step(self):
        return self.step + 1 >= self.max_steps

    @property
    def is_last_batch(self):
        if self.epoch_len is None:
            return False
        return (self.step % self.epoch_len) == self.epoch_len - 1

    @property
    def sigterm_received(self):
        self.sigterm_flag = self.sigterm_flag or any(self.sig_handler.signals_received())
        return self.sigterm_flag

    @property
    def epochs(self):
        epoch = self.epoch
        while self.step < self.max_steps:
            self._debug(f"epochs before stop check epoch={epoch} step={self.step}")
            stop_for_sigterm = self.sigterm_received
            self._debug(f"epochs after stop check epoch={epoch} " f"stop_for_sigterm={stop_for_sigterm}")
            if stop_for_sigterm:
                return
            step_before_epoch = self.step
            self._debug(f"epochs yielding epoch={epoch}")
            yield epoch
            if self.step == step_before_epoch:
                raise RuntimeError("dataloader produced no optimizer steps before max_steps was reached")
            epoch += 1

    def state_dict(self):
        return {"step": min(self.max_steps, self.step + 1), "epoch": self.epoch}

    def load_state_dict(self, s):
        self.step, self.epoch = s["step"], s["epoch"]
