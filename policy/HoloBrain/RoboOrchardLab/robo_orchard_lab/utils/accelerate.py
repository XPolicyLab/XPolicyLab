# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
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

import warnings

from accelerate import Accelerator
from accelerate.data_loader import (
    DataLoaderDispatcher,
    DataLoaderShard,
    IterableDatasetShard,
)
from torch.utils.data import DataLoader

__all__ = ["configure_data_loader_for_accelerate", "prepare_data_loader"]

_DISPATCHER_SLOW_PATH_WARNING = (
    "The prepared dataloader fell back to DataLoaderDispatcher. "
    "This dispatcher-based path is very inefficient for "
    "IterableDatasetMixin in multi-process training. To avoid this "
    "slow path, keep `dispatch_batches=False` and avoid "
    "`put_on_device=True` in the accelerate prepare path for "
    "iterable datasets."
)

_ITERABLE_SHARD_SLOW_PATH_WARNING = (
    "The prepared dataloader fell back to accelerate "
    "IterableDatasetShard. This shard wrapper is very inefficient "
    "for IterableDatasetMixin. To avoid this slow path, ensure the "
    "dataset exposes an `n_shards` value larger than the current "
    "number of processes so accelerate can use dataset-native "
    "sharding."
)


def _warn_and_reset_dataloader_config(
    dataloader_config: object,
    field_name: str,
    expected_value: object,
    warning_message: str,
) -> None:
    field_value = getattr(dataloader_config, field_name)
    if field_value == expected_value:
        return

    warnings.warn(
        warning_message
        + f" Found {field_name}={field_value!r}; "
        + f"reset it to {expected_value!r}.",
        UserWarning,
    )
    setattr(dataloader_config, field_name, expected_value)


def _warn_if_prepare_falls_back_to_slow_path(
    data_loader: object,
) -> None:
    if isinstance(data_loader, DataLoaderDispatcher):
        warnings.warn(_DISPATCHER_SLOW_PATH_WARNING, UserWarning)
        return

    if not isinstance(data_loader, DataLoaderShard):
        return

    base_dataloader = getattr(data_loader, "base_dataloader", None)
    if base_dataloader is None:
        return

    if isinstance(
        getattr(base_dataloader, "dataset", None),
        IterableDatasetShard,
    ):
        warnings.warn(_ITERABLE_SHARD_SLOW_PATH_WARNING, UserWarning)


def configure_data_loader_for_accelerate(
    accelerator: Accelerator,
    data_loader: DataLoader,
) -> bool:
    """Normalize dataloader settings before calling `Accelerator.prepare`.

    Recommended usage:
        When the dataloader should be prepared together with model,
        optimizer, and lr_scheduler, call this helper first, then call
        `accelerator.prepare(...)` exactly once on all objects.

    Pattern:
        `configure_data_loader_for_accelerate(accelerator, dataloader)`
        `dataloader, model, optimizer, scheduler = accelerator.prepare(`
        `    dataloader, model, optimizer, scheduler`
        `)`

    Returns:
        bool: Whether the prepared dataloader should be checked for
            IterableDataset slow-path warnings.
    """

    from robo_orchard_lab.dataset.robot.dataset_ex import IterableDatasetMixin

    dataset = data_loader.dataset
    if not (
        isinstance(dataset, IterableDatasetMixin)
        and accelerator.num_processes > 1
    ):
        return False

    if dataset.shard_kwargs.shard_strategy is None:
        warnings.warn(
            "The dataset is an iterable dataset and the shard strategy "
            "is not set for multi-process training. This may lead to "
            "unbalanced data loading and potential system hang. "
            "Reset the shard strategy to 'pad_last'. ",
            UserWarning,
        )
        dataset.shard_kwargs.shard_strategy = "pad_last"
    _warn_and_reset_dataloader_config(
        dataloader_config=accelerator.dataloader_config,
        field_name="dispatch_batches",
        expected_value=False,
        warning_message=(
            "Using IterableDatasetMixin with multi-process training and "
            "dispatch_batches != False will lead to inefficient data "
            "loading."
        ),
    )
    _warn_and_reset_dataloader_config(
        dataloader_config=accelerator.dataloader_config,
        field_name="even_batches",
        expected_value=False,
        warning_message=(
            "even_batches in accelerator dataloader config is not "
            "supported for IterableDataset. Set drop_last in the "
            "dataloader instead if you need to drop the last incomplete "
            "batch."
        ),
    )
    _warn_and_reset_dataloader_config(
        dataloader_config=accelerator.dataloader_config,
        field_name="split_batches",
        expected_value=False,
        warning_message=(
            "Using IterableDatasetMixin with multi-process training and "
            "split_batches != False will lead to inefficient data "
            "loading."
        ),
    )
    return True


def prepare_data_loader(
    accelerator: Accelerator,
    data_loader: DataLoader,
    **kwargs,
) -> DataLoader:
    """Prepare the dataloader using accelerator.prepare_data_loader.

    This function is a wrapper around accelerator.prepare_data_loader to handle
    the case when the dataset is `IterableDatasetMixin`.

    Warning:
        This function already calls `accelerator.prepare_data_loader()`.
        Do not pass the returned dataloader to `accelerator.prepare(...)`
        again, or it may be wrapped and sharded a second time.

        If the dataloader must be prepared together with model, optimizer,
        and lr_scheduler, call `configure_data_loader_for_accelerate(...)`
        first, then call `accelerator.prepare(...)` once on all objects.

    Recommended usage:
        Use this helper only when the dataloader must be prepared on its own.

    Pattern:
        `dataloader = prepare_data_loader(accelerator, dataloader)`

    Avoid:
        `dataloader = prepare_data_loader(accelerator, dataloader)`
        `dataloader, model, optimizer, scheduler = accelerator.prepare(`
        `    dataloader, model, optimizer, scheduler`
        `)`

    """

    should_check_slow_path = configure_data_loader_for_accelerate(
        accelerator=accelerator,
        data_loader=data_loader,
    )
    ret = accelerator.prepare_data_loader(data_loader, **kwargs)

    if should_check_slow_path:
        _warn_if_prepare_falls_back_to_slow_path(ret)

    return ret
