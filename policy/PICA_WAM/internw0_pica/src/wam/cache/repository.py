"""Generic artifact repository, immutable snapshots and build sessions."""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import replace
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ._format import (
    CATALOG_TYPE,
    FORMAT_VERSION,
    TENSOR_SHARD_BACKEND,
)
from .contracts import (
    ArtifactContract,
    ArtifactSnapshot,
    PreflightReport,
    PublishReport,
    ValidatedCacheSnapshot,
    normalize_artifact_id,
)
from .config import (
    DEFAULT_VALIDATION_MAX_INFLIGHT_BYTES,
    DEFAULT_VALIDATION_PROGRESS_INTERVAL_SECONDS,
    DEFAULT_VALIDATION_WORKERS,
    CacheValidationMode,
)
from .errors import CacheError
from .tensor_shard import (
    TensorShardReader,
    TensorShardWriter,
    atomic_json_dump,
    file_sha256,
    merge_rank_fragments,
    write_index,
    write_manifest,
)


def _read_manifest_contract(path: Path) -> ArtifactContract:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        raise CacheError(f"Unable to read cache artifact manifest {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("contract"), dict):
        raise CacheError(f"Invalid cache artifact contract in manifest: {path}")
    return ArtifactContract.from_dict(payload["contract"])


def _validate_subset_contract(
    *, cached: ArtifactContract, expected: ArtifactContract
) -> None:
    """Validate producer/schema identity while allowing a smaller keyspace."""

    comparisons = {
        "artifact_id": (cached.artifact_id, expected.artifact_id),
        "key_codec": (cached.key_codec, expected.key_codec),
        "dependencies": (cached.dependencies, expected.dependencies),
        "values": (cached.values, expected.values),
        "value_codec": (cached.value_codec, expected.value_codec),
    }
    mismatched = [name for name, (left, right) in comparisons.items() if left != right]
    if mismatched:
        raise CacheError(
            f"{expected.artifact_id} partial cache contract mismatch: "
            f"fields={mismatched}."
        )
    if cached.sample_count > expected.sample_count:
        raise CacheError(
            f"{expected.artifact_id} partial cache sample count exceeds runtime "
            f"keyspace: cached={cached.sample_count} expected={expected.sample_count}."
        )


def _normalize_contracts(
    contracts: Mapping[str, ArtifactContract] | Sequence[ArtifactContract],
) -> dict[str, ArtifactContract]:
    if isinstance(contracts, Mapping):
        normalized = {
            normalize_artifact_id(name): contract
            for name, contract in contracts.items()
        }
    else:
        normalized = {contract.artifact_id: contract for contract in contracts}
    if not normalized:
        raise ValueError("Cache repository operation requires at least one artifact.")
    for artifact_id, contract in normalized.items():
        if artifact_id != contract.artifact_id:
            raise ValueError(
                f"Artifact contract key/id mismatch: {artifact_id} vs "
                f"{contract.artifact_id}."
            )
    return normalized


class CacheReadSession:
    """Readers for a pinned, already validated snapshot."""

    def __init__(
        self,
        snapshot: ValidatedCacheSnapshot,
        contracts: Mapping[str, ArtifactContract],
        *,
        max_cached_shards: int,
        validation_workers: int = DEFAULT_VALIDATION_WORKERS,
        validation_max_inflight_bytes: int = DEFAULT_VALIDATION_MAX_INFLIGHT_BYTES,
        validation_progress_interval_seconds: float = (
            DEFAULT_VALIDATION_PROGRESS_INTERVAL_SECONDS
        ),
    ) -> None:
        self.snapshot = snapshot
        self.contracts = _normalize_contracts(contracts)
        self.max_cached_shards = max(1, int(max_cached_shards))
        self.validation_workers = max(1, int(validation_workers))
        self.validation_max_inflight_bytes = max(
            1, int(validation_max_inflight_bytes)
        )
        self.validation_progress_interval_seconds = max(
            0.0, float(validation_progress_interval_seconds)
        )
        if set(snapshot.artifacts) != set(self.contracts):
            raise CacheError(
                "Snapshot/contract artifact mismatch: "
                f"snapshot={sorted(snapshot.artifacts)} "
                f"contracts={sorted(self.contracts)}."
            )
        self._readers: dict[str, TensorShardReader] = {}

    def _reader(self, artifact_id: str) -> TensorShardReader:
        artifact_id = normalize_artifact_id(artifact_id)
        reader = self._readers.get(artifact_id)
        if reader is not None:
            return reader
        if artifact_id not in self.contracts:
            raise CacheError(f"Artifact is not part of this snapshot: {artifact_id}")
        artifact_snapshot = self.snapshot.artifacts[artifact_id]
        root = Path(self.snapshot.root).resolve()
        manifest_path = (root / artifact_snapshot.manifest).resolve()
        try:
            manifest_path.relative_to(root)
        except ValueError as exc:
            raise CacheError(
                f"Snapshot manifest escapes cache root: {artifact_snapshot.manifest!r}."
            ) from exc
        if manifest_path.name != "manifest.json":
            raise CacheError(
                "Snapshot artifact manifest must end in manifest.json: "
                f"{artifact_snapshot.manifest!r}."
            )
        reader = TensorShardReader(
            manifest_path.parent,
            snapshot=artifact_snapshot,
            contract=artifact_snapshot.contract or self.contracts[artifact_id],
            max_cached_shards=self.max_cached_shards,
            validation_workers=self.validation_workers,
            validation_max_inflight_bytes=self.validation_max_inflight_bytes,
            validation_progress_interval_seconds=(
                self.validation_progress_interval_seconds
            ),
        )
        self._readers[artifact_id] = reader
        return reader

    def get_many(self, keys: Sequence[str]) -> dict[str, dict[str, Any]]:
        keys = [str(key) for key in keys]
        if not keys:
            raise ValueError("CacheReadSession.get_many requires at least one key.")
        return {
            artifact_id: self._reader(artifact_id).get_many(keys)
            for artifact_id in self.contracts
        }

    def get_available_many(
        self, artifact_id: str, keys: Sequence[str]
    ) -> tuple[dict[str, Any] | None, list[bool]]:
        """Return compact rows for present keys plus an input-order hit mask."""

        reader = self._reader(artifact_id)
        present_keys, hit_mask = reader.available_keys(keys)
        return (
            reader.get_many(present_keys) if present_keys else None,
            hit_mask,
        )

    def stats(self) -> dict[str, dict[str, int]]:
        return {
            artifact_id: reader.stats() for artifact_id, reader in self._readers.items()
        }

    def close(self) -> None:
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()

    def __enter__(self) -> "CacheReadSession":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


class CacheBuildSession:
    """Rank-local writers plus rank-0 atomic publication for one build."""

    def __init__(
        self,
        repository: "CacheRepository",
        *,
        contracts: Mapping[str, ArtifactContract],
        build_id: str,
        rank: int,
        shard_size: int,
    ) -> None:
        self.repository = repository
        self.contracts = _normalize_contracts(contracts)
        self.build_id = str(build_id)
        self.rank = int(rank)
        self.shard_size = max(1, int(shard_size))
        if not self.build_id:
            raise ValueError("Cache build session requires a build id.")
        self._writers = {
            artifact_id: TensorShardWriter(
                repository.temporary_artifact_root(artifact_id, self.build_id),
                contract=contract,
                build_id=self.build_id,
                rank=self.rank,
                shard_size=self.shard_size,
            )
            for artifact_id, contract in self.contracts.items()
        }
        self._finalized = False

    def put(
        self,
        keys: Sequence[str],
        values: Mapping[str, Mapping[str, Any]],
    ) -> None:
        expected = set(self.contracts)
        actual = {normalize_artifact_id(name) for name in values}
        if expected != actual:
            raise ValueError(
                "Cache build artifact mismatch: "
                f"missing={sorted(expected - actual)} "
                f"extra={sorted(actual - expected)}."
            )
        for artifact_id in self.contracts:
            self._writers[artifact_id].add(keys, values[artifact_id])

    def put_artifact(
        self,
        artifact_id: str,
        keys: Sequence[str],
        values: Mapping[str, Any],
    ) -> None:
        """Write one artifact through the same build/session infrastructure."""

        artifact_id = normalize_artifact_id(artifact_id)
        writer = self._writers.get(artifact_id)
        if writer is None:
            raise ValueError(
                f"Artifact {artifact_id!r} is not registered in this build."
            )
        writer.add(keys, values)

    def finalize_rank(self) -> dict[str, str]:
        paths = {
            artifact_id: str(writer.finalize())
            for artifact_id, writer in self._writers.items()
        }
        self._finalized = True
        return paths

    def publish(
        self,
        *,
        world_size: int,
        expected_keys: Iterable[str],
    ) -> PublishReport:
        if self.rank != 0:
            raise RuntimeError("Cache publish is only available on rank 0.")
        if not self._finalized:
            raise RuntimeError("Finalize rank writers before publishing.")
        expected_keys = tuple(str(key) for key in expected_keys)
        counts = {contract.sample_count for contract in self.contracts.values()}
        if len(counts) != 1:
            raise ValueError(f"Build artifact sample counts disagree: {counts}.")
        expected_count = next(iter(counts))
        if (
            len(expected_keys) != expected_count
            or len(set(expected_keys)) != expected_count
        ):
            raise ValueError(
                "Cache publish keyspace must contain exactly the unique runtime "
                f"keys: keys={len(expected_keys)} unique={len(set(expected_keys))} "
                f"expected={expected_count}."
            )

        catalog_artifacts: dict[str, Any] = {}
        if self.repository.catalog_path.exists():
            catalog_artifacts.update(self.repository.read_catalog()["artifacts"])

        prepared: dict[str, dict[str, Any]] = {}
        for artifact_id, contract in self.contracts.items():
            temporary_root = self.repository.temporary_artifact_root(
                artifact_id, self.build_id
            )
            entries, shards = merge_rank_fragments(
                temporary_root,
                contract=contract,
                build_id=self.build_id,
                world_size=world_size,
            )
            if len(entries) != expected_count:
                raise CacheError(
                    f"{artifact_id} cache coverage mismatch while publishing: "
                    f"cached={len(entries)} expected={expected_count}."
                )
            if set(entries) != set(expected_keys):
                raise CacheError(
                    f"{artifact_id} cache key coverage mismatch while publishing."
                )
            index_path, index_sha256 = write_index(
                temporary_root,
                contract=contract,
                build_id=self.build_id,
                entries=entries,
            )
            manifest_path = write_manifest(
                temporary_root,
                {
                    "artifact_id": artifact_id,
                    "backend": TENSOR_SHARD_BACKEND,
                    "build_id": self.build_id,
                    "contract": contract.to_dict(),
                    "index_file": index_path.name,
                    "index_sha256": index_sha256,
                    "shards": shards,
                    "precompute_world_size": int(world_size),
                },
            )
            manifest_sha256 = file_sha256(manifest_path)
            validator = TensorShardReader(
                temporary_root,
                snapshot=ArtifactSnapshot(
                    artifact_id=artifact_id,
                    build_id=self.build_id,
                    manifest="manifest.json",
                    manifest_sha256=manifest_sha256,
                ),
                contract=contract,
                max_cached_shards=self.repository.max_cached_shards,
                validation_workers=self.repository.validation_workers,
                validation_max_inflight_bytes=(
                    self.repository.validation_max_inflight_bytes
                ),
                validation_progress_interval_seconds=(
                    self.repository.validation_progress_interval_seconds
                ),
            )
            validation = validator.validate_full(expected_keys=expected_keys)
            prepared[artifact_id] = {
                "temporary_root": temporary_root,
                "manifest_sha256": manifest_sha256,
                "validation": validation,
            }

        # Build directories are immutable.  Moving them before the catalog is
        # safe: a failure leaves only unreachable orphan builds; readers keep
        # observing the old atomically published catalog.
        for artifact_id, metadata in prepared.items():
            source = Path(metadata["temporary_root"])
            destination = self.repository.final_artifact_root(
                artifact_id, self.build_id
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise CacheError(
                    f"Cache build destination already exists: {destination}"
                )
            os.replace(source, destination)
            manifest_ref = (
                destination.relative_to(self.repository.root) / "manifest.json"
            ).as_posix()
            catalog_artifacts[artifact_id] = {
                "backend": TENSOR_SHARD_BACKEND,
                "build_id": self.build_id,
                "manifest": manifest_ref,
                "manifest_sha256": metadata["manifest_sha256"],
            }

        atomic_json_dump(
            {
                "schema_version": FORMAT_VERSION,
                "cache_type": CATALOG_TYPE,
                "artifacts": catalog_artifacts,
            },
            self.repository.catalog_path,
        )
        temporary_run_root = self.repository.root / ".tmp" / self.build_id
        if temporary_run_root.exists():
            try:
                shutil.rmtree(temporary_run_root)
            except OSError:
                pass
        return PublishReport(
            build_id=self.build_id,
            artifacts={
                artifact_id: metadata["validation"]
                for artifact_id, metadata in prepared.items()
            },
            catalog=str(self.repository.catalog_path),
        )


class CacheRepository:
    """Catalog and artifact lifecycle, independent of InternW0-delta domains."""

    def __init__(
        self,
        root: str | Path,
        *,
        max_cached_shards: int = 32,
        validation_workers: int = DEFAULT_VALIDATION_WORKERS,
        validation_max_inflight_bytes: int = DEFAULT_VALIDATION_MAX_INFLIGHT_BYTES,
        validation_progress_interval_seconds: float = (
            DEFAULT_VALIDATION_PROGRESS_INTERVAL_SECONDS
        ),
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.max_cached_shards = max(1, int(max_cached_shards))
        self.validation_workers = max(1, int(validation_workers))
        self.validation_max_inflight_bytes = max(
            1, int(validation_max_inflight_bytes)
        )
        self.validation_progress_interval_seconds = max(
            0.0, float(validation_progress_interval_seconds)
        )

    @property
    def catalog_path(self) -> Path:
        return self.root / "catalog.json"

    def temporary_artifact_root(self, artifact_id: str, build_id: str) -> Path:
        return (
            self.root
            / ".tmp"
            / str(build_id)
            / "artifacts"
            / normalize_artifact_id(artifact_id)
        )

    def final_artifact_root(self, artifact_id: str, build_id: str) -> Path:
        return (
            self.root
            / "artifacts"
            / normalize_artifact_id(artifact_id)
            / "builds"
            / str(build_id)
        )

    def read_catalog(self) -> dict[str, Any]:
        if not self.catalog_path.is_file():
            raise CacheError(f"Missing cache catalog: {self.catalog_path}")
        try:
            with self.catalog_path.open("r", encoding="utf-8") as handle:
                catalog = json.load(handle)
        except Exception as exc:
            raise CacheError(
                f"Unable to read cache catalog {self.catalog_path}: {exc}"
            ) from exc
        if not isinstance(catalog, dict):
            raise CacheError(
                f"Cache catalog must be a JSON object: {self.catalog_path}"
            )
        try:
            schema = int(catalog.get("schema_version", -1))
        except (TypeError, ValueError):
            schema = -1
        if schema != FORMAT_VERSION:
            raise CacheError(
                "Cache storage format is incompatible: "
                f"root={self.root}. "
                "Existing cache data is not migrated automatically; generate "
                "the cache in an empty/new root."
            )
        if str(catalog.get("cache_type")) != CATALOG_TYPE:
            raise CacheError(f"Unexpected cache type: {catalog.get('cache_type')!r}.")
        artifacts = catalog.get("artifacts")
        if not isinstance(artifacts, dict):
            raise CacheError("Cache catalog artifacts must be an object.")
        return catalog

    def validate_write_target(
        self,
        artifact_ids: Iterable[str],
        *,
        overwrite: bool,
    ) -> None:
        if not self.catalog_path.exists():
            return
        catalog = self.read_catalog()
        selected = {normalize_artifact_id(value) for value in artifact_ids}
        conflicts = sorted(selected.intersection(catalog["artifacts"]))
        if conflicts and not overwrite:
            raise CacheError(
                f"Cache artifacts already exist at {self.root}: {conflicts}. "
                "Set +precompute.overwrite=true to publish replacement builds."
            )

    def begin_build(
        self,
        contracts: Mapping[str, ArtifactContract] | Sequence[ArtifactContract],
        *,
        build_id: str,
        rank: int,
        shard_size: int,
    ) -> CacheBuildSession:
        return CacheBuildSession(
            self,
            contracts=_normalize_contracts(contracts),
            build_id=build_id,
            rank=rank,
            shard_size=shard_size,
        )

    def preflight(
        self,
        contracts: Mapping[str, ArtifactContract] | Sequence[ArtifactContract],
        *,
        expected_keys: Iterable[str],
        validation_mode: CacheValidationMode | str = CacheValidationMode.META_ONLY,
        allow_partial_artifacts: Iterable[str] = (),
    ) -> tuple[ValidatedCacheSnapshot, PreflightReport]:
        contracts = _normalize_contracts(contracts)
        expected_keys = tuple(str(key) for key in expected_keys)
        validation_mode = CacheValidationMode.parse(validation_mode)
        allow_partial = {
            normalize_artifact_id(value) for value in allow_partial_artifacts
        }
        unknown_partial = allow_partial - set(contracts)
        if unknown_partial:
            raise ValueError(
                f"Partial coverage policy references unselected artifacts: "
                f"{sorted(unknown_partial)}."
            )
        started = time.perf_counter()
        catalog = self.read_catalog()
        catalog_sha256 = file_sha256(self.catalog_path)
        snapshots: dict[str, ArtifactSnapshot] = {}
        for artifact_id in contracts:
            entry = catalog["artifacts"].get(artifact_id)
            if not isinstance(entry, dict):
                raise CacheError(
                    f"Selected cache artifact {artifact_id!r} is missing from catalog."
                )
            if str(entry.get("backend")) != TENSOR_SHARD_BACKEND:
                raise CacheError(
                    f"Unsupported catalog backend for {artifact_id}: "
                    f"{entry.get('backend')!r}."
                )
            build_id = str(entry.get("build_id", ""))
            manifest_ref = entry.get("manifest")
            manifest_sha256 = str(entry.get("manifest_sha256", ""))
            if not build_id or not isinstance(manifest_ref, str) or not manifest_ref:
                raise CacheError(f"Invalid catalog entry for {artifact_id}: {entry!r}")
            if len(manifest_sha256) != 64:
                raise CacheError(
                    f"Invalid catalog manifest checksum for {artifact_id}."
                )
            manifest_path = (self.root / manifest_ref).resolve()
            try:
                manifest_path.relative_to(self.root)
            except ValueError as exc:
                raise CacheError(
                    f"Catalog manifest escapes cache root: {manifest_ref!r}."
                ) from exc
            if not manifest_path.is_file():
                raise CacheError(f"Missing artifact manifest: {manifest_path}")
            if file_sha256(manifest_path) != manifest_sha256:
                raise CacheError(
                    f"Catalog manifest checksum mismatch for {artifact_id}: "
                    f"{manifest_path}"
                )
            stored_contract = _read_manifest_contract(manifest_path)
            if artifact_id in allow_partial:
                _validate_subset_contract(
                    cached=stored_contract,
                    expected=contracts[artifact_id],
                )
            snapshots[artifact_id] = ArtifactSnapshot(
                artifact_id=artifact_id,
                build_id=build_id,
                manifest=manifest_ref,
                manifest_sha256=manifest_sha256,
                contract=(stored_contract if artifact_id in allow_partial else None),
            )
        snapshot = ValidatedCacheSnapshot(
            root=str(self.root),
            catalog_sha256=catalog_sha256,
            artifacts=snapshots,
        )
        reports = {}
        with CacheReadSession(
            snapshot,
            contracts,
            max_cached_shards=self.max_cached_shards,
            validation_workers=self.validation_workers,
            validation_max_inflight_bytes=self.validation_max_inflight_bytes,
            validation_progress_interval_seconds=(
                self.validation_progress_interval_seconds
            ),
        ) as session:
            for artifact_id in contracts:
                reader = session._reader(artifact_id)
                artifact_expected_keys = expected_keys
                missing_count = 0
                missing_examples: tuple[str, ...] = ()
                if artifact_id in allow_partial:
                    cached_keys = tuple(reader.available_keys(expected_keys)[0])
                    cached_key_set = set(cached_keys)
                    # Loading the index above also verifies its checksum and row count.
                    actual_keys = set(reader.index_keys())
                    expected_key_set = set(expected_keys)
                    extra_keys = actual_keys - expected_key_set
                    if extra_keys:
                        raise CacheError(
                            f"{artifact_id} partial cache contains keys outside the "
                            f"runtime keyspace: examples={sorted(extra_keys)[:8]}."
                        )
                    if not actual_keys:
                        raise CacheError(f"{artifact_id} partial cache contains no rows.")
                    if actual_keys != cached_key_set:
                        raise CacheError(
                            f"{artifact_id} partial cache key lookup is inconsistent."
                        )
                    artifact_expected_keys = cached_keys
                    missing_count = len(expected_key_set) - len(cached_key_set)
                    missing_examples = tuple(
                        islice(
                            (
                                key
                                for key in expected_keys
                                if key not in cached_key_set
                            ),
                            8,
                        )
                    )
                report = reader.validate(
                    expected_keys=artifact_expected_keys,
                    validation_mode=validation_mode,
                )
                reports[artifact_id] = replace(
                    report,
                    expected_sample_count=len(expected_keys),
                    missing_sample_count=missing_count,
                    missing_examples=missing_examples,
                )
        return snapshot, PreflightReport(
            artifacts=reports,
            wall_seconds=time.perf_counter() - started,
        )

    def open_snapshot(
        self,
        snapshot: ValidatedCacheSnapshot,
        contracts: Mapping[str, ArtifactContract] | Sequence[ArtifactContract],
    ) -> CacheReadSession:
        if Path(snapshot.root).resolve() != self.root:
            raise CacheError(
                f"Snapshot root mismatch: snapshot={snapshot.root} "
                f"repository={self.root}."
            )
        return CacheReadSession(
            snapshot,
            _normalize_contracts(contracts),
            max_cached_shards=self.max_cached_shards,
            validation_workers=self.validation_workers,
            validation_max_inflight_bytes=self.validation_max_inflight_bytes,
            validation_progress_interval_seconds=(
                self.validation_progress_interval_seconds
            ),
        )
