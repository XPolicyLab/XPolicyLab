"""Map-style Dataset adapter for the unchanged model batch contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch.utils.data import Dataset

from leap.data.me_u0.map_style_robot_video_dataset import MapStyleIndex
from leap.data.world_unified.adapter import canonical_sample_from_mapping
from leap.data.world_unified.collator import PlannedWorldUnifiedSample
from leap.data.world_unified.plan import build_task_plan


class WorldUnifiedMapDataset(Dataset):
    """Plan and validate one directly indexed ME-U0 anchor at a time."""

    def __init__(
        self,
        dataset: Dataset,
    ) -> None:
        if not isinstance(dataset, Dataset):
            raise TypeError("world-unified map-style data must wrap a torch Dataset")
        if not hasattr(dataset, "source_ranges") or not hasattr(dataset, "get_source"):
            raise TypeError("world-unified map-style data requires source_ranges/get_source")
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        setter = getattr(self.dataset, "set_epoch", None)
        if callable(setter):
            setter(int(epoch))

    @property
    def source_ranges(self):
        return self.dataset.source_ranges

    @property
    def source_counts(self):
        return self.dataset.source_counts

    def get_source(self, index: Any) -> str:
        physical_index = (
            index.physical_index if isinstance(index, MapStyleIndex) else int(index)
        )
        return self.dataset.get_source(physical_index)

    def __getitem__(self, index: Any) -> PlannedWorldUnifiedSample:
        if isinstance(index, MapStyleIndex):
            physical_index = index.physical_index
            stream_position = index.stream_position
        else:
            physical_index = int(index)
            stream_position = physical_index

        raw = self.dataset[physical_index]
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"map-style ME-U0 returned {type(raw).__name__}, expected mapping"
            )
        source_name = raw.get("source")
        if not isinstance(source_name, str):
            source_name = self.dataset.get_source(physical_index)
        domain_id = int(raw["domain_id"])
        num_cameras = int(raw["num_cameras"])

        converted = dict(raw)
        converted.update(
            source=source_name,
            domain_id=domain_id,
            stream_position=stream_position,
        )
        metadata_value = converted.get("metadata")
        if metadata_value is None:
            metadata = {}
        elif isinstance(metadata_value, Mapping):
            metadata = dict(metadata_value)
        else:
            raise TypeError("sample metadata must be a mapping")
        provenance = {
            "source": source_name,
            "domain_id": domain_id,
            "stream_position": stream_position,
            "task": "video_action_pred",
            "task_id": 2,
            "physical_index": physical_index,
        }
        for key, expected in provenance.items():
            if key in metadata and metadata[key] != expected:
                raise ValueError(
                    f"metadata {key!r} conflicts with map identity: "
                    f"{metadata[key]!r} != {expected!r}"
                )
        metadata.update(provenance)
        converted["metadata"] = metadata

        if "proprio_dim_is_pad" in converted:
            proprio_mask = torch.as_tensor(
                converted["proprio_dim_is_pad"], dtype=torch.bool
            )
            if "state_dim_is_pad" in converted:
                state_mask = torch.as_tensor(
                    converted["state_dim_is_pad"], dtype=torch.bool
                )
                if not torch.equal(proprio_mask, state_mask):
                    raise ValueError(
                        "proprio_dim_is_pad and state_dim_is_pad conflict"
                    )
            converted["state_dim_is_pad"] = proprio_mask

        sample = canonical_sample_from_mapping(
            converted,
            domain_id=domain_id,
            num_cameras=num_cameras,
        )
        plan = build_task_plan(
            rgb_frames=sample.rgb_frames,
            action_steps=sample.action_steps,
        )
        if sample.main_image is None:
            raise ValueError("map-style route requires main_image")
        return PlannedWorldUnifiedSample(sample=sample, plan=plan)


def build_world_unified_map_dataset(dataset: Dataset) -> WorldUnifiedMapDataset:
    if isinstance(dataset, WorldUnifiedMapDataset):
        return dataset
    return WorldUnifiedMapDataset(dataset)
