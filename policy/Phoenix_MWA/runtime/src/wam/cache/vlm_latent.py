"""Frozen-VLM artifact adapter for the shared cache manager and shard backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from ._format import FORMAT_VERSION, revision_tag
from .contracts import ArtifactContract, TensorSpec
from .errors import CacheError
from .fields import cache_key_for_sample, vlm_artifact_id
from .fingerprint import build_dataset_fingerprint, stable_fingerprint


VLM_CONTEXT_VALUE_NAME = "context"
VLM_MASK_VALUE_NAME = "mask"
KEY_CODEC = revision_tag("sample_int", 1)


def output_key_mapping(value_codec: str) -> dict[str, dict[str, str]]:
    return {
        vlm_artifact_id(value_codec): {
            VLM_CONTEXT_VALUE_NAME: "vlm_context_cache",
            VLM_MASK_VALUE_NAME: "vlm_mask_cache",
        }
    }


def _dataset_fingerprint(dataset: Any) -> str:
    return build_dataset_fingerprint(
        dataset_dirs=dataset.dataset_dirs,
        dataset_length=len(dataset.lerobot_dataset),
        episode_ranges=zip(dataset._episode_starts, dataset._episode_ends),
        image_shape_meta=dataset.shape_meta.get("images", []),
        global_sample_stride=dataset.lerobot_dataset.global_sample_stride,
        val_set_proportion=dataset.lerobot_dataset.val_set_proportion,
        is_training_set=dataset.lerobot_dataset.is_training_set,
        episode_selection=dataset.lerobot_dataset.episode_selection,
    )


@dataclass(frozen=True)
class VlmLatentSource:
    dataset: Any
    indices: tuple[int, ...]

    @classmethod
    def create(
        cls, dataset: Any, *, indices: Iterable[int] | None = None
    ) -> "VlmLatentSource":
        required = (
            "lerobot_dataset",
            "dataset_dirs",
            "shape_meta",
            "video_view_names",
        )
        missing = [name for name in required if not hasattr(dataset, name)]
        if missing:
            raise TypeError(
                "VLM cache requires a RobotVideoDataset-compatible source; "
                f"missing={missing}."
            )
        if not bool(getattr(dataset, "return_vlm_current_images", False)):
            raise CacheError(
                "VLM cache requires return_vlm_current_images=true on the dataset."
            )
        selected = (
            tuple(range(len(dataset)))
            if indices is None
            else tuple(int(value) for value in indices)
        )
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("VLM cache source requires unique sample indices.")
        if min(selected) < 0 or max(selected) >= len(dataset):
            raise ValueError("VLM cache source index is outside the Dataset range.")
        from .vae_latent import VaeLatentSource
        from .fields import CacheField

        VaeLatentSource.create(dataset, (CacheField.CURRENT,), indices=selected)
        processor = getattr(dataset.lerobot_dataset, "processor", None)
        drop_probability = float(getattr(processor, "drop_high_level_prob", 1.0))
        if getattr(
            dataset, "override_instruction", None
        ) is None and drop_probability not in (0.0, 1.0):
            raise CacheError(
                "VLM caching requires deterministic instructions; instruction dropout is enabled."
            )
        return cls(dataset=dataset, indices=selected)

    def expected_keys(self) -> tuple[str, ...]:
        return tuple(cache_key_for_sample(index) for index in self.indices)

    def get_sample(self, index: int) -> dict[str, Any]:
        # RobotVideoDataset exposes a projection that decodes only the current
        # image from each camera.  VLM cache generation needs exactly that
        # timestamp; using the default __getitem__ path would decode the full
        # training video window (for example 33 frames per camera on RoboTwin).
        # Compatible sources may provide their own sample loader.
        loader = getattr(self.dataset, "get_item_with_vlm_current_images", None)
        sample = loader(int(index)) if callable(loader) else self.dataset[int(index)]
        required = ("sample_id", "vlm_current_images", "prompt")
        missing = [name for name in required if name not in sample]
        if missing:
            raise CacheError(f"VLM generation sample is missing fields: {missing}.")
        if int(sample["sample_id"]) != int(index):
            raise CacheError(
                "VLM cache source returned another sample after a decode error."
            )
        result = {name: sample[name] for name in required}
        for name in (
            "video_canvas_view_names",
            "image_is_pad",
            "vlm_current_view_is_pad",
        ):
            if name in sample:
                result[name] = sample[name]
        return result

    def contract(
        self,
        *,
        producer_fingerprint: str,
        context_dim: int,
        dtype: torch.dtype,
        understanding_config: Mapping[str, Any],
        value_codec: str,
    ) -> ArtifactContract:
        if bool(understanding_config.get("train_vlm", False)):
            raise CacheError(
                "VLM cache is only valid when model.understanding.train_vlm=false."
            )
        dataset_fingerprint = _dataset_fingerprint(self.dataset)
        task_tables = []
        multi_dataset = getattr(self.dataset.lerobot_dataset, "multi_dataset", None)
        for child in getattr(multi_dataset, "_datasets", ()):
            tasks = getattr(getattr(child, "meta", None), "tasks", {})
            task_tables.append(
                sorted((int(key), str(value)) for key, value in tasks.items())
            )
        processor = getattr(self.dataset.lerobot_dataset, "processor", None)
        processor_signature = (
            None
            if processor is None
            else {
                "class": f"{type(processor).__module__}.{type(processor).__qualname__}",
                "drop_high_level_prob": float(
                    getattr(processor, "drop_high_level_prob", 1.0)
                ),
                "use_zh_instruction": bool(
                    getattr(processor, "use_zh_instruction", False)
                ),
                "train_transforms": repr(getattr(processor, "train_transforms", None)),
                "val_transforms": repr(getattr(processor, "val_transforms", None)),
                "vlm_current_image_position": getattr(
                    processor, "vlm_current_image_position", None
                ),
                "vlm_use_processed_images": bool(
                    getattr(processor, "vlm_use_processed_images", False)
                ),
            }
        )
        preprocess = stable_fingerprint(
            {
                "version": revision_tag("vlm_current_preprocess", FORMAT_VERSION),
                "images": self.dataset.shape_meta.get("images", []),
                "view_names": list(self.dataset.video_view_names),
                "processor": processor_signature,
                "tasks": task_tables,
                "override_instruction": getattr(
                    self.dataset, "override_instruction", None
                ),
                "max_pixels": int(understanding_config.get("max_pixels", 65536)),
                "prompt": understanding_config.get("prompt"),
                "default_view_names": understanding_config.get(
                    "default_view_names", ""
                ),
                "precompute_metadata": bool(
                    understanding_config.get("precompute_metadata", True)
                ),
            }
        )
        keyspace = stable_fingerprint(
            {
                "version": revision_tag("vlm_sample_keyspace", FORMAT_VERSION),
                "dataset": dataset_fingerprint,
                "key_codec": KEY_CODEC,
                "indices": self.indices,
            }
        )
        artifact_id = vlm_artifact_id(value_codec)
        return ArtifactContract(
            artifact_id=artifact_id,
            sample_count=len(self.indices),
            keyspace_fingerprint=keyspace,
            key_codec=KEY_CODEC,
            dependencies={
                "dataset": dataset_fingerprint,
                "image_prompt_preprocess": preprocess,
                "producer": producer_fingerprint,
            },
            values={
                VLM_CONTEXT_VALUE_NAME: TensorSpec(
                    dtype=dtype, shape=(-1, int(context_dim))
                ),
                VLM_MASK_VALUE_NAME: TensorSpec(dtype=torch.bool, shape=(-1,)),
            },
            value_codec=value_codec,
        )


class VlmLatentGenerationDataset(Dataset):
    def __init__(self, source: VlmLatentSource) -> None:
        self.source = source

    def __len__(self) -> int:
        return len(self.source.indices)

    def __getitem__(self, position: int) -> dict[str, Any]:
        return self.source.get_sample(self.source.indices[int(position)])


def encode_vlm_artifacts(
    understanding: Any,
    batch: Mapping[str, Any],
    *,
    dtype: torch.dtype,
) -> list[dict[str, torch.Tensor]]:
    images = batch.get("vlm_current_images")
    if not isinstance(images, torch.Tensor) or images.ndim != 5:
        raise ValueError("VLM cache generation expects vlm_current_images [B,V,C,H,W].")
    frames = images.unsqueeze(1).to(device=understanding.vlm.device, non_blocking=True)
    prompts = [str(value) for value in batch["prompt"]]
    view_names_value = batch.get("video_canvas_view_names")
    if isinstance(view_names_value, (list, tuple)):
        view_names: Sequence[str] | None = [str(value) for value in view_names_value]
    elif view_names_value is None:
        view_names = None
    else:
        view_names = [str(view_names_value)] * int(images.shape[0])
    output = understanding(
        frames=frames,
        prompts=prompts,
        view_names=view_names,
        frame_labels=["current"],
        valid_mask=(
            ~batch["image_is_pad"][:, 0].to(device="cpu", dtype=torch.bool)
            if isinstance(batch.get("image_is_pad"), torch.Tensor)
            else None
        ),
        view_valid_mask=(
            ~batch["vlm_current_view_is_pad"]
            .to(device="cpu", dtype=torch.bool)
            .unsqueeze(1)
            if isinstance(batch.get("vlm_current_view_is_pad"), torch.Tensor)
            else None
        ),
    )
    context = output["vlm_context"].to(dtype=dtype)
    mask = output["vlm_mask"].to(dtype=torch.bool)
    rows: list[dict[str, torch.Tensor]] = []
    for index in range(int(context.shape[0])):
        valid = mask[index].nonzero(as_tuple=False).flatten()
        if valid.numel() == 0:
            rows.append(
                {
                    VLM_CONTEXT_VALUE_NAME: context.new_zeros(
                        (1, 1, int(context.shape[-1]))
                    ),
                    VLM_MASK_VALUE_NAME: mask.new_zeros((1, 1)),
                }
            )
            continue
        start, stop = int(valid[0]), int(valid[-1]) + 1
        rows.append(
            {
                VLM_CONTEXT_VALUE_NAME: context[index : index + 1, start:stop],
                VLM_MASK_VALUE_NAME: mask[index : index + 1, start:stop],
            }
        )
    return rows
