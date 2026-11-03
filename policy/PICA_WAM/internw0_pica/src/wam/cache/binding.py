"""Pickle-safe external DataLoader bindings for validated cache snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from torch.utils.data._utils.collate import default_collate

from .contracts import ArtifactContract, PreflightReport, ValidatedCacheSnapshot
from .errors import CacheError
from .fields import cache_key_for_sample
from .repository import CacheReadSession, CacheRepository


def _sample_id(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise CacheError(
                "Cache sample_id tensor must contain one value, got "
                f"{tuple(value.shape)}."
            )
        value = value.item()
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise CacheError(
            f"Cache sample_id must be integer-like, got {value!r}."
        ) from exc


@dataclass(frozen=True)
class CacheDevicePlacement:
    """How Trainer should let Accelerate place a role's DataLoader batch."""

    accelerator_places_batch: bool

    @property
    def input_builder_owns_h2d(self) -> bool:
        return not self.accelerator_places_batch


class CacheDataProjection(str, Enum):
    """Business-sample projection selected for a cache-bound Dataset."""

    DEFAULT = "default"
    LATENT_ONLY = "latent_only"
    VLM_CURRENT = "vlm_current"

    @classmethod
    def parse(cls, value: Any) -> "CacheDataProjection":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except ValueError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unsupported cache data projection {value!r}; allowed: {allowed}."
            ) from exc


_PROJECTION_LOADERS: dict[CacheDataProjection, str] = {
    CacheDataProjection.LATENT_ONLY: "get_item_without_images",
    CacheDataProjection.VLM_CURRENT: "get_item_with_vlm_current_images",
}


class ProjectedDatasetView(torch.utils.data.Dataset):
    """Proxy a Dataset through one explicit, capability-checked projection."""

    def __init__(
        self,
        dataset: Any,
        projection: CacheDataProjection | str,
    ) -> None:
        projection = CacheDataProjection.parse(projection)
        try:
            loader_name = _PROJECTION_LOADERS[projection]
        except KeyError as exc:
            raise ValueError(
                "ProjectedDatasetView requires a non-default projection."
            ) from exc
        loader = getattr(dataset, loader_name, None)
        if not callable(loader):
            raise TypeError(
                f"{type(dataset).__name__} does not support cache data "
                f"projection {projection.value!r}; missing {loader_name}()."
            )
        self.dataset = dataset
        self.projection = projection
        self.loader_name = loader_name

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        return getattr(self.dataset, self.loader_name)(int(index))

    def __getattr__(self, name: str) -> Any:
        if name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)


class LatentOnlyDatasetView(ProjectedDatasetView):
    """Backward-compatible name for the original image-free projection."""

    def __init__(self, dataset: Any) -> None:
        super().__init__(dataset, CacheDataProjection.LATENT_ONLY)


@dataclass
class CacheBinding:
    """One dataset role bound to a validated immutable snapshot."""

    snapshot: ValidatedCacheSnapshot | None = None
    contracts: Mapping[str, ArtifactContract] = field(default_factory=dict)
    output_keys: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    max_cached_shards: int = 32
    report: PreflightReport | None = None
    sample_id_key: str = "sample_id"
    data_projection: CacheDataProjection = CacheDataProjection.DEFAULT
    partial_artifact_hit_keys: Mapping[str, str] = field(default_factory=dict)
    # Compatibility alias for callers created before data_projection existed.
    skip_source_images: bool = False
    _session: CacheReadSession | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        projection = CacheDataProjection.parse(self.data_projection)
        if self.skip_source_images:
            if projection not in (
                CacheDataProjection.DEFAULT,
                CacheDataProjection.LATENT_ONLY,
            ):
                raise ValueError(
                    "skip_source_images=true conflicts with cache data "
                    f"projection {projection.value!r}."
                )
            projection = CacheDataProjection.LATENT_ONLY
        self.data_projection = projection
        self.skip_source_images = projection is CacheDataProjection.LATENT_ONLY

    @property
    def enabled(self) -> bool:
        return self.snapshot is not None

    @property
    def device_placement(self) -> CacheDevicePlacement:
        return CacheDevicePlacement(accelerator_places_batch=not self.enabled)

    def _reader(self) -> CacheReadSession:
        if self.snapshot is None:
            raise RuntimeError("Disabled cache binding has no reader.")
        if self._session is None:
            self._session = CacheRepository(
                self.snapshot.root,
                max_cached_shards=self.max_cached_shards,
            ).open_snapshot(self.snapshot, self.contracts)
        return self._session

    def contiguous_shard_sample_ranges(
        self,
        artifact_id: str = "vae_latent.current",
    ) -> tuple[tuple[int, int], ...]:
        """Return the sample ranges written into consecutive cache shards.

        VAE cache generation partitions the monotonically increasing sample-id
        sequence into contiguous rank blocks, and publication preserves
        rank/shard filename order.  The manifest row counts therefore define
        the cache-local sample partition without loading the large index.
        """
        if self.snapshot is None:
            raise CacheError("Disabled cache binding has no shard ranges.")
        artifact_id = str(artifact_id)
        try:
            artifact = self.snapshot.artifacts[artifact_id]
            contract = self.contracts[artifact_id]
        except KeyError as exc:
            raise CacheError(
                f"Cache snapshot does not contain artifact {artifact_id!r}."
            ) from exc
        root = Path(self.snapshot.root).resolve()
        manifest_path = (root / artifact.manifest).resolve()
        try:
            manifest_path.relative_to(root)
        except ValueError as exc:
            raise CacheError(
                f"Snapshot manifest escapes cache root: {artifact.manifest!r}."
            ) from exc
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except Exception as exc:
            raise CacheError(
                f"Unable to read cache shard layout from {manifest_path}: {exc}"
            ) from exc
        shards = manifest.get("shards") if isinstance(manifest, dict) else None
        if not isinstance(shards, list) or not shards:
            raise CacheError(f"Cache manifest has no shards: {manifest_path}")
        filenames = [str(item.get("file", "")) for item in shards]
        if filenames != sorted(filenames):
            raise CacheError(
                "Cache-locality sampling requires manifest shards in "
                "rank/shard filename order."
            )
        ranges: list[tuple[int, int]] = []
        cursor = 0
        for item in shards:
            try:
                rows = int(item["rows"])
            except (KeyError, TypeError, ValueError) as exc:
                raise CacheError(f"Invalid cache shard metadata: {item!r}.") from exc
            if rows <= 0:
                raise CacheError(f"Invalid cache shard row count: {item!r}.")
            ranges.append((cursor, cursor + rows))
            cursor += rows
        if cursor != contract.sample_count:
            raise CacheError(
                f"Cache shard rows cover {cursor} samples, expected "
                f"{contract.sample_count}."
            )
        return tuple(ranges)

    def enrich_batch(
        self,
        samples: Sequence[Mapping[str, Any]],
        collated: Mapping[str, Any],
    ) -> dict[str, Any]:
        result = dict(collated)
        if not self.enabled:
            return result
        try:
            sample_ids = [sample[self.sample_id_key] for sample in samples]
        except KeyError as exc:
            raise CacheError(
                f"Cache-enabled samples must contain {self.sample_id_key!r}."
            ) from exc
        keys = [cache_key_for_sample(_sample_id(value)) for value in sample_ids]
        session = self._reader()
        for artifact_id in self.contracts:
            hit_output_key = self.partial_artifact_hit_keys.get(artifact_id)
            if hit_output_key is None:
                bundle = session._reader(artifact_id).get_many(keys)
            else:
                bundle, hit_mask = session.get_available_many(artifact_id, keys)
                result[hit_output_key] = torch.tensor(hit_mask, dtype=torch.bool)
                if bundle is None:
                    continue
            mapping = self.output_keys.get(artifact_id)
            if mapping is None:
                raise CacheError(
                    f"No runtime output mapping for cache artifact {artifact_id!r}."
                )
            if set(mapping) != set(bundle):
                raise CacheError(
                    f"Runtime tensor mapping mismatch for {artifact_id}: "
                    f"mapping={sorted(mapping)} values={sorted(bundle)}."
                )
            for tensor_name, output_key in mapping.items():
                result[output_key] = bundle[tensor_name]
        return result

    def enrich_item(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(sample)
        if not self.enabled:
            return result
        if self.sample_id_key not in sample:
            raise CacheError(
                f"Cache-enabled sample must contain {self.sample_id_key!r}."
            )
        key = cache_key_for_sample(_sample_id(sample[self.sample_id_key]))
        session = self._reader()
        for artifact_id in self.contracts:
            hit_output_key = self.partial_artifact_hit_keys.get(artifact_id)
            if hit_output_key is None:
                bundle = session._reader(artifact_id).get_many([key])
            else:
                bundle, hit_mask = session.get_available_many(artifact_id, [key])
                result[hit_output_key] = torch.tensor(hit_mask[0], dtype=torch.bool)
                if bundle is None:
                    continue
            mapping = self.output_keys.get(artifact_id)
            if mapping is None or set(mapping) != set(bundle):
                raise CacheError(
                    f"Runtime tensor mapping mismatch for {artifact_id}."
                )
            for tensor_name, output_key in mapping.items():
                result[output_key] = bundle[tensor_name][0]
        return result

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        # Readers and shard LRU tensors are process-local.  A spawned/forked
        # DataLoader worker always opens its own reader from the pinned snapshot.
        state["_session"] = None
        return state


@dataclass
class CacheCollator:
    binding: CacheBinding
    base_collate: Callable[[Sequence[Any]], Any] = default_collate

    def __call__(self, samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        collated = self.base_collate(samples)
        if not isinstance(collated, Mapping):
            raise TypeError(
                f"Cache collate base must return a mapping, got {type(collated)}."
            )
        return self.binding.enrich_batch(samples, collated)


@dataclass
class CacheRuntimeBindings:
    """Role-indexed bindings returned by CacheManager.prepare_training()."""

    bindings: Mapping[str, CacheBinding]

    @classmethod
    def disabled(cls) -> "CacheRuntimeBindings":
        return cls({"train": CacheBinding(), "validation": CacheBinding()})

    def binding(self, role: str) -> CacheBinding:
        return self.bindings.get(str(role), CacheBinding())

    def wrap_collate(
        self,
        role: str,
        base_collate: Callable[[Sequence[Any]], Any] | None,
    ) -> Callable[[Sequence[Any]], Any]:
        binding = self.binding(role)
        base = base_collate if callable(base_collate) else default_collate
        if not binding.enabled:
            return base
        return CacheCollator(binding=binding, base_collate=base)

    def wrap_dataset(self, role: str, dataset: Any) -> Any:
        binding = self.binding(role)
        if (
            not binding.enabled
            or binding.data_projection is CacheDataProjection.DEFAULT
        ):
            return dataset
        return ProjectedDatasetView(dataset, binding.data_projection)

    def load_item(self, role: str, dataset: Any, index: int) -> dict[str, Any]:
        sample = dataset[int(index)]
        return self.binding(role).enrich_item(sample)

    def device_placement(self, role: str) -> CacheDevicePlacement:
        return self.binding(role).device_placement

    def contiguous_shard_sample_ranges(
        self,
        role: str,
        artifact_id: str = "vae_latent.current",
    ) -> tuple[tuple[int, int], ...]:
        return self.binding(role).contiguous_shard_sample_ranges(artifact_id)

    def close(self) -> None:
        seen: set[int] = set()
        for binding in self.bindings.values():
            if id(binding) in seen:
                continue
            seen.add(id(binding))
            binding.close()
