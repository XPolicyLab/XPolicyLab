"""Alias stub: the AnyData streaming dataset was NOT ported to FlexPi.

Legacy training configs reference `DroidAnyDataStreamingDataset` under
`data.train._target_`. Eval never instantiates it — `get_model` copies only
`cfg.model` and `cfg.data.train.processor` out of a checkpoint's config — so its
absence does not block loading a checkpoint that was trained with it.

It is stubbed rather than omitted so the failure, if something does try to build
it, names the cause instead of raising a bare ModuleNotFoundError.
"""

from __future__ import annotations

_MSG = (
    "DroidAnyDataStreamingDataset was not ported to FlexPi (which ships "
    "RobotVideoDataset only). Checkpoints trained with it still LOAD fine — eval "
    "uses only cfg.model and cfg.data.train.processor. You are seeing this because "
    "something tried to construct the training dataset: point data.train._target_ "
    "at flexpi.datasets.lerobot.robot_video_dataset.RobotVideoDataset, or train on "
    "the original FastWAM codebase."
)


class DroidAnyDataStreamingDataset:  # noqa: D101
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(_MSG)


__all__ = ["DroidAnyDataStreamingDataset"]
