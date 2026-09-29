"""CogWAM data layer.

One benchmark, one mixture, one sample ABI.  Upstream dispatched
``build_dataloader`` over several ``dataset_py`` backends (a VLM-only
instruction dataset, a plain LeRobot path, and the JointFlow dual-query path);
only the JointFlow path reaches this recipe, so the dispatch is gone and the
argument survives purely as a config assertion.

The dataset stack is imported lazily on purpose.  ``cogwam.models.cogwam`` and
the policy server only need :mod:`cogwam.data.composite` and the event-memory
decision constants; pulling in PyAV/OpenCV/albumentations and the parquet
readers at model-import time is what upstream avoided by keeping the
compositor outside the dataloader package entirely.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch.distributed as dist

from cogwam.data.collate import collate_fn
from cogwam.data.composite import (
    TRI_VIEW_COMPOSITE_LAYOUT,
    TRI_VIEW_COMPOSITE_SIZE,
    TRI_VIEW_COMPOSITE_VIEW_KEY,
    build_tri_view_composite,
    to_pil_preserve,
)
from cogwam.data.event_memory import KEEP_DECISION, UPDATE_DECISION

_SUPPORTED_DATASET_PY = {"jointflow", "jointflow_lerobot"}

_LAZY_EXPORTS = {
    "JointFlowDataset": "cogwam.data.dataset",
    "build_event_memory_dataloader": "cogwam.data.dataset",
    "build_joint_dataloader": "cogwam.data.dataset",
    "build_joint_dataset": "cogwam.data.dataset",
    "resolve_data_mix": "cogwam.data.mixtures",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)


def build_dataloader(
    cfg,
    dataset_py: str = "jointflow",
    *,
    save_statistics: bool = True,
):
    """Build the physical training/validation dataloader and persist statistics.

    ``dataset_py`` is kept in the signature because it names a config field the
    recipe pins; it is validated rather than dispatched on.
    """

    if str(dataset_py) not in _SUPPORTED_DATASET_PY:
        raise ValueError(
            f"Unsupported datasets.vla_data.dataset_py={dataset_py!r}; "
            f"this release provides {sorted(_SUPPORTED_DATASET_PY)}."
        )

    from cogwam.data.dataset import build_joint_dataloader

    vla_train_dataloader = build_joint_dataloader(cfg)
    if save_statistics and ((not dist.is_initialized()) or dist.get_rank() == 0):
        output_dir = Path(cfg.output_dir)
        vla_train_dataloader.dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
    return vla_train_dataloader


__all__ = [
    "KEEP_DECISION",
    "TRI_VIEW_COMPOSITE_LAYOUT",
    "TRI_VIEW_COMPOSITE_SIZE",
    "TRI_VIEW_COMPOSITE_VIEW_KEY",
    "UPDATE_DECISION",
    "JointFlowDataset",
    "build_dataloader",
    "build_event_memory_dataloader",
    "build_joint_dataloader",
    "build_joint_dataset",
    "build_tri_view_composite",
    "collate_fn",
    "resolve_data_mix",
    "to_pil_preserve",
]
