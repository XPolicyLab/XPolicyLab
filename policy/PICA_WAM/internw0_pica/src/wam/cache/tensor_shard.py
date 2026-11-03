"""Domain-independent tensor-bundle shard backend."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
import hashlib
import io
import json
import logging
import os
import stat
import time
import uuid
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from ._format import ARTIFACT_TYPE, FORMAT_VERSION, TENSOR_SHARD_BACKEND
from .contracts import (
    ArtifactContract,
    ArtifactSnapshot,
    ArtifactValidationReport,
)
from .config import (
    DEFAULT_VALIDATION_MAX_INFLIGHT_BYTES,
    DEFAULT_VALIDATION_PROGRESS_INTERVAL_SECONDS,
    DEFAULT_VALIDATION_WORKERS,
    CacheValidationMode,
)
from .errors import CacheError


logger = logging.getLogger(__name__)


def atomic_json_dump(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp.{uuid.uuid4().hex}"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp.{uuid.uuid4().hex}"
    torch.save(dict(payload), str(temporary))
    os.replace(temporary, path)


def _torch_load_cpu(source: Any) -> Any:
    try:
        return torch.load(source, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older torch compatibility.
        seek = getattr(source, "seek", None)
        if callable(seek):
            seek(0)
        return torch.load(source, map_location="cpu")


def torch_load_cpu(path: Path) -> Any:
    return _torch_load_cpu(str(path))


def _torch_load_cpu_bytes(payload: bytes) -> Any:
    return _torch_load_cpu(io.BytesIO(payload))


def file_sha256(path: Path, *, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text.lower()
    )


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        raise CacheError(f"Unable to read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CacheError(f"{label} must be a JSON object: {path}")
    return payload


_OFFSETS_SUFFIX = ".__offsets"


def _storage_names(contract: ArtifactContract) -> set[str]:
    names: set[str] = set()
    for name, spec in contract.values.items():
        names.add(name)
        if spec.ragged:
            names.add(f"{name}{_OFFSETS_SUFFIX}")
    return names


def _pad_ragged_rows(rows: Sequence[torch.Tensor]) -> torch.Tensor:
    if not rows:
        raise ValueError("Cannot pad an empty ragged cache batch.")
    max_length = max(int(row.shape[0]) for row in rows)
    result = rows[0].new_zeros((len(rows), max_length, *rows[0].shape[1:]))
    for index, row in enumerate(rows):
        result[index, : int(row.shape[0])] = row
    return result.contiguous()


@dataclass(frozen=True)
class _ShardValidationResult:
    rows: int
    total_bytes: int
    stat_seconds: float
    read_seconds: float
    hash_seconds: float
    load_seconds: float
    check_seconds: float


class TensorShardWriter:
    """Write rank-local immutable shards for one artifact contract."""

    def __init__(
        self,
        root: str | Path,
        *,
        contract: ArtifactContract,
        build_id: str,
        rank: int,
        shard_size: int,
    ) -> None:
        self.root = Path(root)
        self.contract = contract
        self.artifact_id = contract.artifact_id
        self.build_id = str(build_id)
        self.rank = int(rank)
        self.shard_size = max(1, int(shard_size))
        if not self.build_id:
            raise ValueError("TensorShardWriter requires a build id.")
        self.shard_root = self.root / "shards"
        self.fragment_root = self.root / "fragments"
        self.shard_root.mkdir(parents=True, exist_ok=True)
        self.fragment_root.mkdir(parents=True, exist_ok=True)
        self._keys: list[str] = []
        self._values: dict[str, list[torch.Tensor]] = {
            name: [] for name in contract.values
        }
        self._all_keys: set[str] = set()
        self._entries: dict[str, dict[str, Any]] = {}
        self._shards: list[dict[str, Any]] = []
        self._shard_ordinal = 0
        self._closed = False

    def add(
        self,
        keys: Sequence[str],
        values: Mapping[str, torch.Tensor],
    ) -> None:
        if self._closed:
            raise RuntimeError(f"{self.artifact_id} writer is already finalized.")
        keys = self.contract.validate_batch(keys, values)
        if len(set(keys)) != len(keys):
            raise CacheError(f"{self.artifact_id} write batch contains duplicate keys.")
        duplicates = [key for key in keys if key in self._all_keys]
        if duplicates:
            raise CacheError(
                f"{self.artifact_id} rank {self.rank} wrote duplicate keys: "
                f"{duplicates[:8]}."
            )
        self._all_keys.update(keys)

        for row_index, key in enumerate(keys):
            if len(self._keys) == self.shard_size:
                self._flush()
            self._keys.append(key)
            for name in self.contract.values:
                self._values[name].append(
                    values[name][row_index].detach().to(device="cpu").contiguous()
                )
            if len(self._keys) == self.shard_size:
                self._flush()

    def _flush(self) -> None:
        if not self._keys:
            return
        shard_name = f"rank{self.rank:05d}-shard{self._shard_ordinal:06d}.pt"
        shard_path = self.shard_root / shard_name
        bundle: dict[str, torch.Tensor] = {}
        for name, rows in self._values.items():
            spec = self.contract.values[name]
            if spec.ragged:
                lengths = [int(row.shape[0]) for row in rows]
                offsets = torch.zeros(len(rows) + 1, dtype=torch.int64)
                offsets[1:] = torch.as_tensor(lengths, dtype=torch.int64).cumsum(0)
                logical = torch.cat(rows, dim=0).contiguous()
                bundle[f"{name}{_OFFSETS_SUFFIX}"] = offsets
            else:
                logical = torch.stack(rows, dim=0).contiguous()
            bundle[name] = logical
        _atomic_torch_save(
            {
                "schema_version": FORMAT_VERSION,
                "artifact_id": self.artifact_id,
                "build_id": self.build_id,
                "keys": list(self._keys),
                "values": bundle,
            },
            shard_path,
        )
        size = int(shard_path.stat().st_size)
        checksum = file_sha256(shard_path)
        for offset, key in enumerate(self._keys):
            self._entries[key] = {"shard": shard_name, "offset": offset}
        self._shards.append(
            {
                "file": shard_name,
                "rows": len(self._keys),
                "bytes": size,
                "sha256": checksum,
                "rank": self.rank,
            }
        )
        self._keys.clear()
        self._values = {name: [] for name in self.contract.values}
        self._shard_ordinal += 1

    def finalize(self) -> Path:
        path = self.fragment_root / f"rank{self.rank:05d}.json"
        if self._closed:
            return path
        self._flush()
        if not self._entries:
            raise CacheError(
                f"{self.artifact_id} rank {self.rank} produced no cache rows."
            )
        atomic_json_dump(
            {
                "schema_version": FORMAT_VERSION,
                "artifact_id": self.artifact_id,
                "build_id": self.build_id,
                "rank": self.rank,
                "contract": self.contract.to_dict(),
                "sample_count": len(self._entries),
                "entries": self._entries,
                "shards": self._shards,
            },
            path,
        )
        self._closed = True
        return path


def merge_rank_fragments(
    root: str | Path,
    *,
    contract: ArtifactContract,
    build_id: str,
    world_size: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    root = Path(root)
    entries: dict[str, dict[str, Any]] = {}
    shards: list[dict[str, Any]] = []
    for rank in range(int(world_size)):
        path = root / "fragments" / f"rank{rank:05d}.json"
        if not path.is_file():
            raise CacheError(f"Missing {contract.artifact_id} rank fragment: {path}")
        fragment = _read_json(path, label="cache rank fragment")
        if int(fragment.get("schema_version", -1)) != FORMAT_VERSION:
            raise CacheError(f"Cache rank fragment schema mismatch: {path}")
        if str(fragment.get("artifact_id")) != contract.artifact_id:
            raise CacheError(f"Cache rank fragment artifact mismatch: {path}")
        if str(fragment.get("build_id")) != str(build_id):
            raise CacheError(f"Cache rank fragment build mismatch: {path}")
        if int(fragment.get("rank", -1)) != rank:
            raise CacheError(f"Cache rank fragment rank mismatch: {path}")
        if fragment.get("contract") != contract.to_dict():
            raise CacheError(f"Cache rank fragment contract mismatch: {path}")
        fragment_entries = fragment.get("entries")
        fragment_shards = fragment.get("shards")
        if not isinstance(fragment_entries, dict) or not isinstance(
            fragment_shards, list
        ):
            raise CacheError(f"Invalid cache rank fragment payload: {path}")
        duplicates = sorted(set(entries).intersection(fragment_entries))
        if duplicates:
            raise CacheError(
                f"{contract.artifact_id} duplicate keys across ranks: {duplicates[:8]}."
            )
        entries.update({str(key): value for key, value in fragment_entries.items()})
        shards.extend(fragment_shards)
    return entries, sorted(shards, key=lambda item: str(item.get("file", "")))


def write_index(
    root: str | Path,
    *,
    contract: ArtifactContract,
    build_id: str,
    entries: Mapping[str, Mapping[str, Any]],
) -> tuple[Path, str]:
    path = Path(root) / "index.json"
    atomic_json_dump(
        {
            "schema_version": FORMAT_VERSION,
            "artifact_id": contract.artifact_id,
            "build_id": str(build_id),
            "sample_count": len(entries),
            "entries": dict(entries),
        },
        path,
    )
    return path, file_sha256(path)


def write_manifest(root: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(root) / "manifest.json"
    body = dict(payload)
    body.update(
        {
            "schema_version": FORMAT_VERSION,
            "cache_type": ARTIFACT_TYPE,
            "complete": True,
        }
    )
    atomic_json_dump(body, path)
    return path


class TensorShardReader:
    """Strict reader pinned to one immutable artifact snapshot."""

    def __init__(
        self,
        root: str | Path,
        *,
        snapshot: ArtifactSnapshot,
        contract: ArtifactContract,
        max_cached_shards: int = 32,
        validation_workers: int = DEFAULT_VALIDATION_WORKERS,
        validation_max_inflight_bytes: int = DEFAULT_VALIDATION_MAX_INFLIGHT_BYTES,
        validation_progress_interval_seconds: float = (
            DEFAULT_VALIDATION_PROGRESS_INTERVAL_SECONDS
        ),
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.snapshot = snapshot
        self.contract = contract
        self.artifact_id = contract.artifact_id
        self.build_id = snapshot.build_id
        self.max_cached_shards = max(1, int(max_cached_shards))
        self.validation_workers = max(1, int(validation_workers))
        self.validation_max_inflight_bytes = max(1, int(validation_max_inflight_bytes))
        self.validation_progress_interval_seconds = max(
            0.0, float(validation_progress_interval_seconds)
        )
        self._manifest: dict[str, Any] | None = None
        self._index: dict[str, Any] | None = None
        self._shards: OrderedDict[str, tuple[dict[str, torch.Tensor], list[str]]] = (
            OrderedDict()
        )
        self._stats: defaultdict[str, int] = defaultdict(int)
        if snapshot.artifact_id != contract.artifact_id:
            raise CacheError(
                "Snapshot/contract artifact mismatch: "
                f"{snapshot.artifact_id} vs {contract.artifact_id}."
            )

    @property
    def manifest(self) -> dict[str, Any]:
        if self._manifest is None:
            path = self.root / "manifest.json"
            if not path.is_file():
                raise CacheError(f"Missing {self.artifact_id} manifest: {path}")
            if file_sha256(path) != self.snapshot.manifest_sha256:
                raise CacheError(
                    f"{self.artifact_id} manifest changed after snapshot "
                    f"validation: {path}"
                )
            manifest = _read_json(path, label="cache artifact manifest")
            self._validate_manifest(manifest)
            self._manifest = manifest
        return self._manifest

    def _validate_manifest(self, manifest: Mapping[str, Any]) -> None:
        required = {
            "schema_version",
            "cache_type",
            "complete",
            "artifact_id",
            "backend",
            "build_id",
            "contract",
            "index_file",
            "index_sha256",
            "shards",
        }
        missing = sorted(required - set(manifest))
        if missing:
            raise CacheError(
                f"{self.artifact_id} manifest is missing fields: {missing}."
            )
        if int(manifest["schema_version"]) != FORMAT_VERSION:
            raise CacheError(
                f"{self.artifact_id} manifest storage format is incompatible."
            )
        if str(manifest["cache_type"]) != ARTIFACT_TYPE or not bool(
            manifest["complete"]
        ):
            raise CacheError(f"Incomplete/invalid artifact manifest at {self.root}.")
        if str(manifest["artifact_id"]) != self.artifact_id:
            raise CacheError(f"Artifact manifest id mismatch at {self.root}.")
        if str(manifest["backend"]) != TENSOR_SHARD_BACKEND:
            raise CacheError(
                f"Unsupported {self.artifact_id} backend: {manifest['backend']!r}."
            )
        if str(manifest["build_id"]) != self.build_id:
            raise CacheError(
                f"{self.artifact_id} snapshot/manifest build mismatch: "
                f"snapshot={self.build_id} manifest={manifest['build_id']}."
            )
        manifest_contract = manifest["contract"]
        if not isinstance(manifest_contract, dict):
            raise CacheError(f"Invalid {self.artifact_id} manifest contract.")
        if manifest_contract != self.contract.to_dict():
            self._raise_contract_mismatch(ArtifactContract.from_dict(manifest_contract))
        index_file = manifest["index_file"]
        if not isinstance(index_file, str) or Path(index_file).name != index_file:
            raise CacheError(f"Invalid {self.artifact_id} index file: {index_file!r}.")
        if not _is_sha256(manifest["index_sha256"]):
            raise CacheError(f"Invalid {self.artifact_id} index checksum metadata.")
        shards = manifest["shards"]
        if not isinstance(shards, list) or not shards:
            raise CacheError(f"Invalid {self.artifact_id} shard metadata.")
        names: set[str] = set()
        rows = 0
        for item in shards:
            if not isinstance(item, dict):
                raise CacheError(f"Invalid {self.artifact_id} shard metadata entry.")
            name = item.get("file")
            if not isinstance(name, str) or Path(name).name != name or name in names:
                raise CacheError(
                    f"Invalid/duplicate {self.artifact_id} shard name: {name!r}."
                )
            names.add(name)
            try:
                item_rows = int(item["rows"])
                item_bytes = int(item["bytes"])
            except (KeyError, TypeError, ValueError) as exc:
                raise CacheError(
                    f"Invalid {self.artifact_id} shard rows/bytes: {item!r}."
                ) from exc
            if item_rows <= 0 or item_bytes <= 0 or not _is_sha256(item.get("sha256")):
                raise CacheError(
                    f"Invalid {self.artifact_id} shard metadata: {item!r}."
                )
            rows += item_rows
        if rows != self.contract.sample_count:
            raise CacheError(
                f"{self.artifact_id} manifest rows/sample mismatch: "
                f"rows={rows} expected={self.contract.sample_count}."
            )

    def _raise_contract_mismatch(self, cached: ArtifactContract) -> None:
        expected = self.contract
        if cached.sample_count != expected.sample_count:
            raise CacheError(
                f"{self.artifact_id} sample_count mismatch: "
                f"cache={cached.sample_count} expected={expected.sample_count}."
            )
        if cached.keyspace_fingerprint != expected.keyspace_fingerprint:
            raise CacheError(f"{self.artifact_id} keyspace fingerprint mismatch.")
        if cached.key_codec != expected.key_codec:
            raise CacheError(f"{self.artifact_id} key codec mismatch.")
        for name, value in expected.dependencies.items():
            actual = cached.dependencies.get(name)
            if actual != value:
                raise CacheError(
                    f"{self.artifact_id} dependency {name!r} mismatch: "
                    f"cache={actual!r} expected={value!r}."
                )
        if cached.dependencies != expected.dependencies:
            raise CacheError(f"{self.artifact_id} dependency set mismatch.")
        if cached.values != expected.values:
            raise CacheError(
                f"{self.artifact_id} tensor schema mismatch: "
                f"cache={cached.values} expected={expected.values}."
            )
        if cached.value_codec != expected.value_codec:
            raise CacheError(
                f"{self.artifact_id} value codec mismatch: "
                f"cache={cached.value_codec!r} expected={expected.value_codec!r}."
            )
        raise CacheError(f"{self.artifact_id} artifact contract mismatch.")

    def _ensure_index(self) -> None:
        if self._index is not None:
            return
        manifest = self.manifest
        path = self.root / str(manifest["index_file"])
        if not path.is_file():
            raise CacheError(f"Missing {self.artifact_id} index: {path}")
        if file_sha256(path) != str(manifest["index_sha256"]):
            raise CacheError(f"{self.artifact_id} index checksum mismatch: {path}")
        payload = _read_json(path, label="cache artifact index")
        if int(payload.get("schema_version", -1)) != FORMAT_VERSION:
            raise CacheError(f"{self.artifact_id} index schema mismatch: {path}")
        if str(payload.get("artifact_id")) != self.artifact_id:
            raise CacheError(f"{self.artifact_id} index artifact mismatch: {path}")
        if str(payload.get("build_id")) != self.build_id:
            raise CacheError(f"{self.artifact_id} index build mismatch: {path}")
        entries = payload.get("entries")
        if not isinstance(entries, dict):
            raise CacheError(f"{self.artifact_id} index entries must be an object.")
        if int(payload.get("sample_count", -1)) != self.contract.sample_count:
            raise CacheError(f"{self.artifact_id} index sample count mismatch.")
        if len(entries) != self.contract.sample_count:
            raise CacheError(f"{self.artifact_id} index entry count mismatch.")
        self._index = entries

    @staticmethod
    def _parse_entry(artifact_id: str, key: str, entry: Any) -> tuple[str, int]:
        if not isinstance(entry, dict):
            raise CacheError(f"Invalid {artifact_id} index entry for {key!r}.")
        shard = entry.get("shard")
        offset = entry.get("offset")
        if not isinstance(shard, str) or Path(shard).name != shard:
            raise CacheError(f"Invalid {artifact_id} shard for {key!r}: {shard!r}.")
        if isinstance(offset, bool):
            raise CacheError(f"Invalid {artifact_id} offset for {key!r}: {offset!r}.")
        try:
            offset = int(offset)
        except (TypeError, ValueError) as exc:
            raise CacheError(
                f"Invalid {artifact_id} offset for {key!r}: {offset!r}."
            ) from exc
        return shard, offset

    def _load_shard(self, shard_name: str) -> tuple[dict[str, torch.Tensor], list[str]]:
        cached = self._shards.pop(shard_name, None)
        if cached is not None:
            self._stats["shard_lru_hits"] += 1
            self._shards[shard_name] = cached
            return cached
        path = self.root / "shards" / shard_name
        if not path.is_file():
            raise CacheError(f"Missing {self.artifact_id} shard: {path}")
        try:
            payload = torch_load_cpu(path)
        except Exception as exc:
            raise CacheError(
                f"Unable to load {self.artifact_id} shard: {path}: {exc}"
            ) from exc
        normalized_values, keys = self._validate_shard_payload(path, payload)
        cached = (normalized_values, keys)
        self._shards[shard_name] = cached
        self._stats["shard_loads"] += 1
        self._stats["bytes_loaded"] += int(path.stat().st_size)
        while len(self._shards) > self.max_cached_shards:
            self._shards.popitem(last=False)
        return cached

    def _validate_shard_payload(
        self, path: Path, payload: Any
    ) -> tuple[dict[str, torch.Tensor], list[str]]:
        if not isinstance(payload, dict):
            raise CacheError(f"Invalid {self.artifact_id} shard payload: {path}")
        if int(payload.get("schema_version", -1)) != FORMAT_VERSION:
            raise CacheError(f"{self.artifact_id} shard schema mismatch: {path}")
        if str(payload.get("artifact_id")) != self.artifact_id:
            raise CacheError(f"{self.artifact_id} shard artifact mismatch: {path}")
        if str(payload.get("build_id")) != self.build_id:
            raise CacheError(f"{self.artifact_id} shard build mismatch: {path}")
        keys = payload.get("keys")
        values = payload.get("values")
        if not isinstance(keys, list) or not isinstance(values, dict):
            raise CacheError(f"Invalid {self.artifact_id} shard values: {path}")
        expected_storage_names = _storage_names(self.contract)
        if set(values) != expected_storage_names:
            raise CacheError(f"{self.artifact_id} shard tensor names mismatch: {path}")
        normalized_values: dict[str, torch.Tensor] = {}
        for name, spec in self.contract.values.items():
            tensor = values[name]
            if not isinstance(tensor, torch.Tensor):
                raise CacheError(f"{self.artifact_id}.{name} is not a tensor: {path}")
            logical_prefix: tuple[int, ...]
            if spec.ragged:
                offsets_name = f"{name}{_OFFSETS_SUFFIX}"
                offsets = values[offsets_name]
                if (
                    not isinstance(offsets, torch.Tensor)
                    or offsets.dtype != torch.int64
                    or tuple(offsets.shape) != (len(keys) + 1,)
                    or (int(offsets[0]) != 0)
                    or bool((offsets[1:] < offsets[:-1]).any().item())
                    or (int(offsets[-1]) != int(tensor.shape[0]))
                ):
                    raise CacheError(
                        f"{self.artifact_id}.{name} ragged offsets are invalid: {path}"
                    )
                logical_prefix = (int(offsets[-1]), *spec.shape[1:])
                normalized_values[offsets_name] = offsets.contiguous()
            else:
                logical_prefix = (len(keys), *spec.shape)
            expected_dtype = spec.dtype
            if tuple(tensor.shape) != logical_prefix:
                raise CacheError(
                    f"{self.artifact_id}.{name} shard shape mismatch: {path}"
                )
            if tensor.dtype != expected_dtype:
                raise CacheError(
                    f"{self.artifact_id}.{name} shard dtype mismatch: {path}"
                )
            normalized_values[name] = tensor.contiguous()
        return (normalized_values, [str(key) for key in keys])

    def get_many(self, keys: Sequence[str]) -> dict[str, torch.Tensor]:
        self._ensure_index()
        assert self._index is not None
        keys = [str(key) for key in keys]
        if not keys:
            raise ValueError(f"{self.artifact_id} lookup requires at least one key.")
        self._stats["lookups"] += len(keys)
        grouped: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
        missing: list[str] = []
        for position, key in enumerate(keys):
            entry = self._index.get(key)
            if entry is None:
                if len(missing) < 8:
                    missing.append(key)
                continue
            shard, offset = self._parse_entry(self.artifact_id, key, entry)
            grouped[shard].append((position, offset, key))
        if missing:
            raise CacheError(
                f"{self.artifact_id} cache is missing requested keys; examples={missing}."
            )
        rows: dict[str, list[torch.Tensor | None]] = {
            name: [None] * len(keys) for name in self.contract.values
        }
        for shard_name, requests in grouped.items():
            values, shard_keys = self._load_shard(shard_name)
            for position, offset, key in requests:
                if offset < 0 or offset >= len(shard_keys):
                    raise CacheError(
                        f"{self.artifact_id} offset out of range for {key!r}: {offset} in {shard_name}."
                    )
                if shard_keys[offset] != key:
                    raise CacheError(
                        f"{self.artifact_id} index/shard key mismatch: index={key!r} shard={shard_keys[offset]!r}."
                    )
                for name in rows:
                    spec = self.contract.values[name]
                    if spec.ragged:
                        offsets = values[f"{name}{_OFFSETS_SUFFIX}"]
                        start = int(offsets[offset])
                        stop = int(offsets[offset + 1])
                        value = values[name][start:stop]
                    else:
                        value = values[name][offset]
                    rows[name][position] = value
        result: dict[str, torch.Tensor] = {}
        for name, component_rows in rows.items():
            if any((row is None for row in component_rows)):
                raise CacheError(
                    f"{self.artifact_id}.{name} lookup produced empty rows."
                )
            present = [row for row in component_rows if row is not None]
            spec = self.contract.values[name]
            tensor_rows = [row for row in present if isinstance(row, torch.Tensor)]
            result[name] = (
                _pad_ragged_rows(tensor_rows)
                if spec.ragged
                else torch.stack(tensor_rows, dim=0)
            )
        return result

    def index_keys(self) -> tuple[str, ...]:
        self._ensure_index()
        assert self._index is not None
        return tuple(self._index)

    def available_keys(self, keys: Sequence[str]) -> tuple[list[str], list[bool]]:
        self._ensure_index()
        assert self._index is not None
        normalized = [str(key) for key in keys]
        return (
            [key for key in normalized if key in self._index],
            [key in self._index for key in normalized],
        )

    def _validate_one_shard(
        self,
        shard_name: str,
        metadata: Mapping[str, Any],
        references: Mapping[int, str],
    ) -> _ShardValidationResult:
        path = self.root / "shards" / shard_name

        stat_started = time.perf_counter()
        try:
            file_stat = path.stat()
        except FileNotFoundError as exc:
            raise CacheError(f"Missing {self.artifact_id} shard: {path}") from exc
        if not stat.S_ISREG(file_stat.st_mode):
            raise CacheError(f"Missing {self.artifact_id} shard: {path}")
        expected_bytes = int(metadata["bytes"])
        if int(file_stat.st_size) != expected_bytes:
            raise CacheError(f"{self.artifact_id} shard size mismatch: {path}")
        stat_seconds = time.perf_counter() - stat_started

        read_started = time.perf_counter()
        try:
            serialized = path.read_bytes()
        except FileNotFoundError as exc:
            raise CacheError(f"Missing {self.artifact_id} shard: {path}") from exc
        read_seconds = time.perf_counter() - read_started
        if len(serialized) != expected_bytes:
            raise CacheError(f"{self.artifact_id} shard size mismatch: {path}")

        hash_started = time.perf_counter()
        checksum = hashlib.sha256(serialized).hexdigest()
        hash_seconds = time.perf_counter() - hash_started
        if checksum != str(metadata["sha256"]):
            raise CacheError(f"{self.artifact_id} shard checksum mismatch: {path}")

        load_started = time.perf_counter()
        try:
            payload = _torch_load_cpu_bytes(serialized)
        except Exception as exc:
            raise CacheError(
                f"Unable to load {self.artifact_id} shard {path}: {exc}"
            ) from exc
        load_seconds = time.perf_counter() - load_started

        check_started = time.perf_counter()
        values, shard_keys = self._validate_shard_payload(path, payload)
        del values, payload, serialized
        if set(references) != set(range(len(shard_keys))):
            raise CacheError(
                f"{self.artifact_id} index does not cover every row: {path}"
            )
        if int(metadata["rows"]) != len(shard_keys):
            raise CacheError(f"{self.artifact_id} shard row count mismatch: {path}")
        for offset, key in references.items():
            if shard_keys[offset] != key:
                raise CacheError(f"{self.artifact_id} index/shard key mismatch: {path}")
        check_seconds = time.perf_counter() - check_started
        return _ShardValidationResult(
            rows=len(shard_keys),
            total_bytes=expected_bytes,
            stat_seconds=stat_seconds,
            read_seconds=read_seconds,
            hash_seconds=hash_seconds,
            load_seconds=load_seconds,
            check_seconds=check_seconds,
        )

    def _effective_validation_workers(
        self,
        metadata_by_name: Mapping[str, Mapping[str, Any]],
    ) -> int:
        largest_shard = max(
            int(metadata["bytes"]) for metadata in metadata_by_name.values()
        )
        # During read-once validation each worker can hold both serialized bytes
        # and the deserialized tensor storage at the same time.
        estimated_bytes_per_worker = max(1, largest_shard * 2)
        memory_limited_workers = max(
            1,
            self.validation_max_inflight_bytes // estimated_bytes_per_worker,
        )
        return min(
            self.validation_workers,
            os.cpu_count() or 1,
            memory_limited_workers,
            len(metadata_by_name),
        )

    def _validate_shards(
        self,
        *,
        references: Mapping[str, Mapping[int, str]],
        metadata_by_name: Mapping[str, Mapping[str, Any]],
    ) -> tuple[int, int, dict[str, float], int]:
        shard_names = sorted(references)
        workers = self._effective_validation_workers(metadata_by_name)
        phase_started = time.perf_counter()
        last_progress = phase_started
        completed = 0
        validated_rows = 0
        validated_bytes = 0
        total_bytes = sum(int(item["bytes"]) for item in metadata_by_name.values())
        timings = {
            "stat": 0.0,
            "read": 0.0,
            "hash": 0.0,
            "load": 0.0,
            "check": 0.0,
        }

        logger.info(
            "Cache shard validation start: artifact=%s shards=%d bytes=%.2f GiB "
            "workers=%d requested_workers=%d max_inflight=%.2f GiB",
            self.artifact_id,
            len(shard_names),
            total_bytes / (1024**3),
            workers,
            self.validation_workers,
            self.validation_max_inflight_bytes / (1024**3),
        )

        def record(result: _ShardValidationResult) -> None:
            nonlocal completed, validated_rows, validated_bytes, last_progress
            completed += 1
            validated_rows += result.rows
            validated_bytes += result.total_bytes
            timings["stat"] += result.stat_seconds
            timings["read"] += result.read_seconds
            timings["hash"] += result.hash_seconds
            timings["load"] += result.load_seconds
            timings["check"] += result.check_seconds
            now = time.perf_counter()
            if (
                self.validation_progress_interval_seconds > 0
                and now - last_progress >= self.validation_progress_interval_seconds
            ):
                wall = max(now - phase_started, 1e-9)
                logger.info(
                    "Cache shard validation progress: artifact=%s "
                    "shards=%d/%d bytes=%.2f/%.2f GiB wall=%.1fs "
                    "throughput=%.2f GiB/s",
                    self.artifact_id,
                    completed,
                    len(shard_names),
                    validated_bytes / (1024**3),
                    total_bytes / (1024**3),
                    wall,
                    validated_bytes / (1024**3) / wall,
                )
                last_progress = now

        if workers == 1:
            for shard_name in shard_names:
                record(
                    self._validate_one_shard(
                        shard_name,
                        metadata_by_name[shard_name],
                        references[shard_name],
                    )
                )
        else:
            executor = ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="wam-cache-validate",
            )
            pending: dict[Future[_ShardValidationResult], str] = {}
            shard_iterator = iter(shard_names)

            def submit_one() -> bool:
                try:
                    shard_name = next(shard_iterator)
                except StopIteration:
                    return False
                future = executor.submit(
                    self._validate_one_shard,
                    shard_name,
                    metadata_by_name[shard_name],
                    references[shard_name],
                )
                pending[future] = shard_name
                return True

            try:
                for _ in range(workers):
                    if not submit_one():
                        break
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        pending.pop(future)
                        record(future.result())
                        submit_one()
            except Exception:
                for future in pending:
                    future.cancel()
                raise
            finally:
                executor.shutdown(wait=True, cancel_futures=True)

        timings["wall"] = time.perf_counter() - phase_started
        return validated_rows, validated_bytes, timings, workers

    def validate(
        self,
        *,
        expected_keys: Iterable[str],
        validation_mode: CacheValidationMode | str = CacheValidationMode.META_ONLY,
    ) -> ArtifactValidationReport:
        validation_mode = CacheValidationMode.parse(validation_mode)
        total_started = time.perf_counter()
        manifest_started = time.perf_counter()
        manifest = self.manifest
        manifest_seconds = time.perf_counter() - manifest_started

        index_started = time.perf_counter()
        self._ensure_index()
        index_seconds = time.perf_counter() - index_started
        assert self._index is not None
        expected_key_set = {str(key) for key in expected_keys}
        if len(expected_key_set) != self.contract.sample_count:
            raise CacheError(
                f"{self.artifact_id} runtime keyspace count mismatch: "
                f"keys={len(expected_key_set)} expected={self.contract.sample_count}."
            )
        actual_key_set = set(self._index)
        if expected_key_set != actual_key_set:
            raise CacheError(
                f"{self.artifact_id} key coverage mismatch: "
                f"missing={sorted(expected_key_set - actual_key_set)[:8]} "
                f"extra={sorted(actual_key_set - expected_key_set)[:8]}."
            )

        structure_started = time.perf_counter()
        references: dict[str, dict[int, str]] = defaultdict(dict)
        for key, entry in self._index.items():
            shard, offset = self._parse_entry(self.artifact_id, str(key), entry)
            if offset < 0 or offset in references[shard]:
                raise CacheError(
                    f"{self.artifact_id} duplicate/negative shard offset: "
                    f"shard={shard} offset={offset}."
                )
            references[shard][offset] = str(key)
        metadata_by_name = {str(item.get("file")): item for item in manifest["shards"]}
        if set(metadata_by_name) != set(references):
            raise CacheError(f"{self.artifact_id} manifest/index shard set mismatch.")
        total_bytes = sum(int(item["bytes"]) for item in manifest["shards"])
        validated_rows = 0
        for shard_name in sorted(references):
            path = self.root / "shards" / shard_name
            metadata = metadata_by_name[shard_name]
            refs = references[shard_name]
            expected_rows = int(metadata["rows"])
            if set(refs) != set(range(expected_rows)):
                raise CacheError(
                    f"{self.artifact_id} index does not cover every row: {path}"
                )
            if validation_mode is CacheValidationMode.META_ONLY:
                try:
                    file_stat = path.stat()
                except FileNotFoundError as exc:
                    raise CacheError(
                        f"Missing {self.artifact_id} shard: {path}"
                    ) from exc
                if not stat.S_ISREG(file_stat.st_mode):
                    raise CacheError(f"Missing {self.artifact_id} shard: {path}")
                if int(file_stat.st_size) != int(metadata["bytes"]):
                    raise CacheError(f"{self.artifact_id} shard size mismatch: {path}")
            validated_rows += expected_rows
        if validated_rows != self.contract.sample_count:
            raise CacheError(
                f"{self.artifact_id} validated row mismatch: "
                f"rows={validated_rows} expected={self.contract.sample_count}."
            )
        structure_seconds = time.perf_counter() - structure_started

        validation_workers = 0
        shard_timings = {
            "wall": 0.0,
            "stat": 0.0,
            "read": 0.0,
            "hash": 0.0,
            "load": 0.0,
            "check": 0.0,
        }
        if validation_mode is CacheValidationMode.FULL:
            validated_rows, validated_bytes, shard_timings, validation_workers = (
                self._validate_shards(
                    references=references,
                    metadata_by_name=metadata_by_name,
                )
            )
            if validated_bytes != total_bytes:
                raise CacheError(
                    f"{self.artifact_id} validated byte mismatch: "
                    f"bytes={validated_bytes} expected={total_bytes}."
                )
            if validated_rows != self.contract.sample_count:
                raise CacheError(
                    f"{self.artifact_id} validated row mismatch: "
                    f"rows={validated_rows} expected={self.contract.sample_count}."
                )
        report = ArtifactValidationReport(
            artifact_id=self.artifact_id,
            build_id=self.build_id,
            sample_count=self.contract.sample_count,
            shard_count=len(references),
            total_bytes=total_bytes,
            manifest_seconds=manifest_seconds,
            index_seconds=index_seconds,
            structure_seconds=structure_seconds,
            shard_seconds=shard_timings["wall"],
            total_seconds=time.perf_counter() - total_started,
            validation_mode=validation_mode.value,
            validation_workers=validation_workers,
            shard_stat_seconds=shard_timings["stat"],
            shard_read_seconds=shard_timings["read"],
            shard_hash_seconds=shard_timings["hash"],
            shard_load_seconds=shard_timings["load"],
            shard_check_seconds=shard_timings["check"],
        )
        self.close(keep_manifest=True)
        return report

    def validate_meta_only(
        self, *, expected_keys: Iterable[str]
    ) -> ArtifactValidationReport:
        return self.validate(
            expected_keys=expected_keys,
            validation_mode=CacheValidationMode.META_ONLY,
        )

    def validate_full(
        self, *, expected_keys: Iterable[str]
    ) -> ArtifactValidationReport:
        return self.validate(
            expected_keys=expected_keys,
            validation_mode=CacheValidationMode.FULL,
        )

    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def close(self, *, keep_manifest: bool = False) -> None:
        self._index = None
        self._shards.clear()
        if not keep_manifest:
            self._manifest = None
