"""VAE-latent domain plugin for the generic artifact cache."""

from __future__ import annotations

import bisect
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from wam.model.modules.codecs import video_latent_codec as video_codec

from ._format import FORMAT_VERSION, revision_tag
from .contracts import ArtifactContract, TensorSpec
from .errors import CacheError
from .fields import (
    vae_artifact_id,
    FIELD_SAMPLE_KEYS,
    CacheField,
    cache_key_for_sample,
    expected_vae_latent_shapes,
    normalize_cache_fields,
)
from .fingerprint import (
    build_common_image_fingerprint,
    build_dataset_fingerprint,
    build_vae_field_fingerprints,
    stable_fingerprint,
)


LATENT_VALUE_NAME = "latent"
KEY_CODEC = revision_tag("sample_int", 1)


def output_key_mapping(
    fields: Sequence[CacheField],
    *,
    value_codec: str = "raw",
) -> dict[str, dict[str, str]]:
    return {
        vae_artifact_id(field, value_codec): {
            LATENT_VALUE_NAME: FIELD_SAMPLE_KEYS[field]
        }
        for field in fields
    }


def artifact_ids(
    fields: Sequence[CacheField], *, value_codec: str = "raw"
) -> tuple[str, ...]:
    return tuple(vae_artifact_id(field, value_codec) for field in fields)


def _processor_signature(dataset: Any) -> dict[str, Any] | None:
    processor = getattr(dataset.lerobot_dataset, "processor", None)
    if processor is None:
        return None
    return {
        "class": f"{type(processor).__module__}.{type(processor).__qualname__}",
        "train_transforms": repr(getattr(processor, "train_transforms", None)),
        "val_transforms": repr(getattr(processor, "val_transforms", None)),
    }


def _stochastic_processor_transforms(dataset: Any) -> list[str]:
    processor = getattr(dataset.lerobot_dataset, "processor", None)
    if processor is None:
        return []
    transforms = getattr(
        processor,
        "train_transforms" if dataset.is_training_set else "val_transforms",
        None,
    )
    stochastic_names = {
        "colorjitter",
        "gaussianblur",
        "randomadjustsharpness",
        "randomaffine",
        "randomapply",
        "randomautocontrast",
        "randomchoice",
        "randomcrop",
        "randomequalize",
        "randomerasing",
        "randomgrayscale",
        "randomhorizontalflip",
        "randominvert",
        "randomorder",
        "randomperspective",
        "randomposterize",
        "randomresizedcrop",
        "randomrotation",
        "randomsolarize",
        "randomverticalflip",
    }
    found: list[str] = []

    def visit(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
            return
        name = type(value).__name__
        normalized = name.replace("_", "").lower()
        if normalized.startswith("random") or normalized in stochastic_names:
            found.append(name)
        nested = getattr(value, "transforms", None)
        if nested is not None and nested is not value:
            visit(nested)

    visit(transforms)
    return sorted(set(found))


@dataclass
class VaeLatentSource:
    """Cache-domain adapter around RobotVideoDataset data semantics."""

    dataset: Any
    fields: tuple[CacheField, ...]
    indices: tuple[int, ...]
    frame_cache_capacity: int = 0
    _frame_cache: OrderedDict[int, torch.Tensor] = field(
        default_factory=OrderedDict,
        init=False,
        repr=False,
    )
    _frame_cache_episode: tuple[int, int] | None = field(
        default=None,
        init=False,
        repr=False,
    )

    @classmethod
    def create(
        cls,
        dataset: Any,
        fields: Any = None,
        *,
        indices: Iterable[int] | None = None,
    ) -> "VaeLatentSource":
        selected = normalize_cache_fields(fields)
        selected_indices = (
            tuple(range(len(dataset)))
            if indices is None
            else tuple(int(value) for value in indices)
        )
        if not selected_indices:
            raise ValueError("VAE cache source requires at least one sample index.")
        if len(set(selected_indices)) != len(selected_indices):
            raise ValueError("VAE cache source indices must be unique.")
        if min(selected_indices) < 0 or max(selected_indices) >= len(dataset):
            raise ValueError("VAE cache source index is outside the Dataset range.")
        source = cls(
            dataset=dataset,
            fields=selected,
            indices=selected_indices,
        )
        source.validate_cacheability()
        return source

    def configure_frame_cache(self, capacity: int) -> None:
        """Enable a bounded, worker-local cache of deterministic RGB frames."""

        capacity = max(0, int(capacity))
        max_query_frames = len(self._selected_offsets(self.indices[0])[0])
        if capacity and capacity < max_query_frames:
            raise ValueError(
                "VAE generate frame cache must hold at least one complete "
                f"query; got capacity={capacity}, query_frames={max_query_frames}."
            )
        self.frame_cache_capacity = capacity
        self._frame_cache.clear()
        self._frame_cache_episode = None

    @property
    def layout(self) -> str:
        return (
            "single"
            if bool(self.dataset.single_canvas)
            else str(self.dataset.concat_multi_camera or "single")
        )

    def validate_cacheability(self) -> None:
        dataset = self.dataset
        required = (
            "lerobot_dataset",
            "video_size",
            "video_view_names",
            "video_sample_indices",
            "concat_multi_camera",
            "single_canvas",
            "_memory_query_layout",
            "_format_video_window",
            "_slice_video_time",
        )
        missing = [name for name in required if not hasattr(dataset, name)]
        if missing:
            raise TypeError(
                "VAE latent cache requires a RobotVideoDataset-compatible "
                f"source; missing={missing}."
            )
        if bool(dataset.skip_padding_as_possible):
            raise CacheError(
                "VAE latent cache requires skip_padding_as_possible=false so "
                "sample_id remains stable."
            )
        if (
            CacheField.MEMORY_ANCHOR in self.fields
            and int(dataset.memory_video_anchor_size) != 1
        ):
            raise CacheError(
                "memory_anchor cache requires memory_video_anchor_size=1; "
                f"got {dataset.memory_video_anchor_size}."
            )
        random_augments = [
            name
            for name in (
                "image_color_jitter",
                "image_camera_jitter",
                "image_sensor_noise",
            )
            if getattr(dataset, name, None) is not None
        ]
        if getattr(dataset, "is_training_set", True) and (
            getattr(dataset, "image_domain_jitter", None) or {}
        ).get("enabled", False):
            random_augments.append("image_domain_jitter")
        random_augments.extend(
            f"processor:{name}"
            for name in _stochastic_processor_transforms(dataset)
        )
        if random_augments:
            raise CacheError(
                "VAE latent cache requires deterministic image preprocessing; "
                f"active random transforms={random_augments}."
            )

    def expected_keys(self) -> tuple[str, ...]:
        return tuple(cache_key_for_sample(index) for index in self.indices)

    def contracts(
        self,
        *,
        vae: Any,
        vae_fingerprint: str,
        dtype: torch.dtype,
        value_codec: str = "raw",
    ) -> dict[str, ArtifactContract]:
        if not vae_fingerprint:
            raise CacheError("VAE artifact contract requires a producer fingerprint.")
        dataset = self.dataset
        dataset_fingerprint = build_dataset_fingerprint(
            dataset_dirs=dataset.dataset_dirs,
            dataset_length=len(dataset.lerobot_dataset),
            episode_ranges=zip(dataset._episode_starts, dataset._episode_ends),
            image_shape_meta=dataset.shape_meta.get("images", []),
            global_sample_stride=dataset.lerobot_dataset.global_sample_stride,
            val_set_proportion=dataset.lerobot_dataset.val_set_proportion,
            is_training_set=dataset.lerobot_dataset.is_training_set,
            episode_selection=dataset.lerobot_dataset.episode_selection,
        )
        image_fingerprint = build_common_image_fingerprint(
            video_size=dataset.video_size,
            video_view_names=dataset.video_view_names,
            concat_multi_camera=dataset.concat_multi_camera,
            single_canvas=dataset.single_canvas,
            image_color_jitter=dataset.image_color_jitter,
            image_camera_jitter=dataset.image_camera_jitter,
            image_sensor_noise=dataset.image_sensor_noise,
            processor_video_preprocess=_processor_signature(dataset),
        )
        field_fingerprints = build_vae_field_fingerprints(
            num_frames=dataset.num_frames,
            video_sample_indices=dataset.video_sample_indices,
            action_video_freq_ratio=dataset.action_video_freq_ratio,
            memory_video_anchor_size=dataset.memory_video_anchor_size,
            memory_recent_frame_offset=dataset.memory_recent_frame_offset,
            video_layout=self.layout,
        )
        shapes = expected_vae_latent_shapes(
            vae,
            video_size=dataset.video_size,
            current_video_frames=len(dataset.video_sample_indices),
            video_layout=self.layout,
            num_views=len(dataset.video_view_names),
            fields=self.fields,
        )
        keyspace_fingerprint = stable_fingerprint(
            {
                "version": revision_tag(
                    "vae_sample_keyspace", FORMAT_VERSION
                ),
                "dataset": dataset_fingerprint,
                "key_codec": KEY_CODEC,
                "indices": self.indices,
            }
        )
        return {
            vae_artifact_id(field, value_codec): ArtifactContract(
                artifact_id=vae_artifact_id(field, value_codec),
                sample_count=len(self.indices),
                keyspace_fingerprint=keyspace_fingerprint,
                key_codec=KEY_CODEC,
                dependencies={
                    "dataset": dataset_fingerprint,
                    "image_preprocess": image_fingerprint,
                    "field_selection": field_fingerprints[field],
                    "producer": vae_fingerprint,
                },
                values={
                    LATENT_VALUE_NAME: TensorSpec(
                        dtype=dtype,
                        shape=shapes[field],
                    )
                },
                value_codec=value_codec,
            )
            for field in self.fields
        }

    def _selected_offsets(
        self, sample_idx: int
    ) -> tuple[list[int], list[int], list[int]]:
        image_offsets, state_offsets, action_offsets, _, _ = (
            self.dataset._memory_query_layout(sample_idx)
        )
        anchor_count = int(self.dataset.memory_video_anchor_size)
        selected_image_offsets: list[int] = []
        if CacheField.MEMORY_ANCHOR in self.fields:
            selected_image_offsets.extend(image_offsets[:anchor_count])
        if CacheField.MEMORY_RECENT in self.fields:
            selected_image_offsets.append(image_offsets[anchor_count])
        if CacheField.CURRENT in self.fields:
            selected_image_offsets.extend(image_offsets[anchor_count + 1 :])
        return selected_image_offsets, state_offsets, action_offsets

    def _episode_bounds(self, sample_idx: int) -> tuple[int, int]:
        starts = self.dataset._episode_starts
        ends = self.dataset._episode_ends
        episode_index = bisect.bisect_right(starts, int(sample_idx)) - 1
        if episode_index < 0 or int(sample_idx) >= int(ends[episode_index]):
            raise CacheError(
                f"Sample {sample_idx} is not covered by episode metadata."
            )
        return int(starts[episode_index]), int(ends[episode_index])

    def _frame_cache_plan(
        self,
        sample_idx: int,
    ) -> tuple[
        list[int],
        list[int],
        list[int],
        list[int],
        list[int],
    ]:
        image_offsets, state_offsets, action_offsets = self._selected_offsets(
            sample_idx
        )
        episode_bounds = self._episode_bounds(sample_idx)
        if self._frame_cache_episode != episode_bounds:
            self._frame_cache.clear()
            self._frame_cache_episode = episode_bounds
        episode_start, episode_end = episode_bounds
        stride = int(self.dataset.lerobot_dataset.global_sample_stride)
        requested_indices: list[int] = []
        offset_by_index: dict[int, int] = {}
        for offset in image_offsets:
            absolute_index = max(
                episode_start,
                min(episode_end - 1, int(sample_idx) + int(offset) * stride),
            )
            requested_indices.append(absolute_index)
            offset_by_index.setdefault(absolute_index, int(offset))
            if absolute_index in self._frame_cache:
                self._frame_cache.move_to_end(absolute_index)
        missing_indices = list(
            dict.fromkeys(
                index
                for index in requested_indices
                if index not in self._frame_cache
            )
        )
        missing_offsets = [offset_by_index[index] for index in missing_indices]
        return (
            requested_indices,
            missing_indices,
            missing_offsets,
            state_offsets,
            action_offsets,
        )

    def _get_sample_with_frame_cache(self, index: int) -> dict[str, Any]:
        plans: dict[
            int,
            tuple[list[int], list[int], list[int]],
        ] = {}

        def offsets_factory(
            sample_idx: int,
        ) -> tuple[list[int], list[int], list[int]]:
            (
                requested_indices,
                missing_indices,
                missing_offsets,
                state_offsets,
                action_offsets,
            ) = self._frame_cache_plan(int(sample_idx))
            # A fully cached request is unusual for the current sliding window.
            # Keep the generic image-processing path valid by refreshing one
            # frame instead of sending an empty timestamp list to TorchCodec.
            decode_indices = missing_indices or requested_indices[:1]
            decode_offsets = missing_offsets or [
                self._selected_offsets(int(sample_idx))[0][0]
            ]
            plans[int(sample_idx)] = (
                requested_indices,
                missing_indices,
                decode_indices,
            )
            return decode_offsets, state_offsets, action_offsets

        sample = self.dataset.lerobot_dataset.get_item_with_offset_factory(
            index,
            offsets_factory=offsets_factory,
        )
        actual_sample_idx = self.dataset._as_int(sample.get("idx", index))
        try:
            requested_indices, missing_indices, decode_indices = plans[
                actual_sample_idx
            ]
        except KeyError as exc:
            raise CacheError(
                "VAE frame cache did not record the retried sample layout for "
                f"sample {actual_sample_idx}."
            ) from exc
        video_loaded = sample["pixel_values"]
        if int(video_loaded.shape[1]) != len(decode_indices):
            raise CacheError(
                "VAE frame cache decoded frame count mismatch: "
                f"expected={len(decode_indices)} actual={video_loaded.shape[1]}."
            )
        for absolute_index, frame in zip(
            decode_indices,
            video_loaded.unbind(dim=1),
            strict=True,
        ):
            self._frame_cache[absolute_index] = frame.clone()
            self._frame_cache.move_to_end(absolute_index)
        while len(self._frame_cache) > self.frame_cache_capacity:
            self._frame_cache.popitem(last=False)
        try:
            sample["pixel_values"] = torch.stack(
                [self._frame_cache[value] for value in requested_indices],
                dim=1,
            )
        except KeyError as exc:
            raise CacheError(
                "VAE frame cache evicted a frame required by the active query; "
                f"capacity={self.frame_cache_capacity}."
            ) from exc
        sample["frame_cache_hits"] = len(requested_indices) - len(
            missing_indices
        )
        sample["frame_cache_misses"] = len(missing_indices)
        return sample

    def get_sample(self, index: int) -> dict[str, Any]:
        dataset = self.dataset
        index = int(index)
        if self.frame_cache_capacity:
            sample = self._get_sample_with_frame_cache(index)
        else:
            sample = dataset.lerobot_dataset.get_item_with_offset_factory(
                index,
                offsets_factory=lambda sample_idx: self._selected_offsets(
                    sample_idx
                ),
            )
        actual_sample_idx = dataset._as_int(sample.get("idx", index))
        anchor_count = int(dataset.memory_video_anchor_size)
        color_jitter_params = dataset._sample_color_jitter_params()
        camera_jitter_params = dataset._sample_camera_jitter_params()
        sensor_noise_params = dataset._sample_sensor_noise_params()
        result: dict[str, Any] = {
            "sample_id": actual_sample_idx,
            "frame_cache_hits": int(sample.get("frame_cache_hits", 0)),
            "frame_cache_misses": int(sample.get("frame_cache_misses", 0)),
            "video_layout": "single" if dataset.single_canvas else self.layout,
            "video_view_names": (
                "canvas"
                if dataset.single_canvas
                else "|".join(dataset.video_view_names)
            ),
        }
        video_full = sample["pixel_values"]
        cursor = 0
        if CacheField.MEMORY_ANCHOR in self.fields:
            anchor_raw = dataset._slice_video_time(
                video_full, slice(cursor, cursor + anchor_count)
            )
            cursor += anchor_count
            result["memory_anchor_video"] = dataset._format_video_window(
                anchor_raw,
                color_jitter_params=color_jitter_params,
                camera_jitter_params=camera_jitter_params,
                sensor_noise_params=sensor_noise_params,
            )
        if CacheField.MEMORY_RECENT in self.fields:
            recent_raw = dataset._slice_video_time(
                video_full, slice(cursor, cursor + 1)
            )
            cursor += 1
            result["memory_recent_video"] = dataset._format_video_window(
                recent_raw,
                color_jitter_params=color_jitter_params,
                camera_jitter_params=camera_jitter_params,
                sensor_noise_params=sensor_noise_params,
            )
        if CacheField.CURRENT in self.fields:
            current_raw = dataset._slice_video_time(
                video_full,
                slice(cursor, cursor + len(dataset.video_sample_indices)),
            )
            result["current_video"] = dataset._format_video_window(
                current_raw,
                color_jitter_params=color_jitter_params,
                camera_jitter_params=camera_jitter_params,
                sensor_noise_params=sensor_noise_params,
            )
        return result


class VaeLatentGenerationDataset(Dataset):
    def __init__(self, source: VaeLatentSource) -> None:
        self.source = source

    def __len__(self) -> int:
        return len(self.source.indices)

    def __getitem__(self, position: int) -> dict[str, Any]:
        return self.source.get_sample(self.source.indices[int(position)])


def _select_metadata(value: Any, indices: Sequence[int]) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [value[index] for index in indices]
    return value


def _repeat_metadata(value: Any, indices: Sequence[int], repeats: int) -> Any:
    selected = _select_metadata(value, indices)
    if selected is None or not isinstance(selected, list):
        return selected
    return selected * int(repeats)


def encode_vae_artifacts(
    model_proxy: Any,
    batch: Mapping[str, Any],
    *,
    fields: Sequence[CacheField],
    keep: Sequence[int],
    dtype: torch.dtype,
    value_codec: str = "raw",
) -> dict[str, dict[str, torch.Tensor]]:
    """Encode selected artifacts; anchor/recent share one VAE batch launch."""

    selected = tuple(fields)
    keep = tuple(int(index) for index in keep)
    if not keep:
        raise ValueError("VAE artifact encode requires at least one row.")
    values: dict[str, dict[str, torch.Tensor]] = {}
    if CacheField.CURRENT in selected:
        video = batch["current_video"][list(keep)].to(
            device=model_proxy.device,
            dtype=dtype,
            non_blocking=True,
        )
        latent = video_codec.encode_video_latents(
            model_proxy,
            video,
            tiled=False,
            video_layout=_select_metadata(batch.get("video_layout"), keep),
            video_view_names=_select_metadata(
                batch.get("video_view_names"), keep
            ),
        )
        values[vae_artifact_id(CacheField.CURRENT, value_codec)] = {
            LATENT_VALUE_NAME: latent.to(dtype=dtype)
        }

    memory_fields = [
        field
        for field in (CacheField.MEMORY_ANCHOR, CacheField.MEMORY_RECENT)
        if field in selected
    ]
    if memory_fields:
        input_keys = {
            CacheField.MEMORY_ANCHOR: "memory_anchor_video",
            CacheField.MEMORY_RECENT: "memory_recent_video",
        }
        memory_video = torch.cat(
            [batch[input_keys[field]][list(keep)] for field in memory_fields],
            dim=0,
        ).to(device=model_proxy.device, dtype=dtype, non_blocking=True)
        memory_latents = video_codec.encode_video_latents(
            model_proxy,
            memory_video,
            tiled=False,
            video_layout=_repeat_metadata(
                batch.get("video_layout"), keep, len(memory_fields)
            ),
            video_view_names=_repeat_metadata(
                batch.get("video_view_names"), keep, len(memory_fields)
            ),
        )
        rows = len(keep)
        for field_index, field in enumerate(memory_fields):
            start = field_index * rows
            values[vae_artifact_id(field, value_codec)] = {
                LATENT_VALUE_NAME: memory_latents[start : start + rows].to(
                    dtype=dtype
                )
            }
    return values
