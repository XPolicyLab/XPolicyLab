"""Typed, domain-independent contracts for the artifact cache."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from .codec import RAW_CODEC, normalize_value_codec
from .errors import CacheError


_DTYPE_NAMES: dict[torch.dtype, str] = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.uint8: "uint8",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.bool: "bool",
}
_NAME_DTYPES = {name: dtype for dtype, name in _DTYPE_NAMES.items()}


def dtype_name(dtype: torch.dtype) -> str:
    try:
        return _DTYPE_NAMES[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported cache dtype: {dtype}") from exc


def dtype_from_name(name: str) -> torch.dtype:
    try:
        return _NAME_DTYPES[str(name)]
    except KeyError as exc:
        raise CacheError(f"Unsupported cache manifest dtype: {name!r}") from exc


def normalize_artifact_id(value: str) -> str:
    artifact_id = str(value).strip()
    if not artifact_id:
        raise ValueError("Cache artifact id cannot be empty.")
    allowed = frozenset(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    )
    if any(character not in allowed for character in artifact_id):
        raise ValueError(
            "Cache artifact id may contain only letters, digits, '.', '_' and '-': "
            f"{artifact_id!r}."
        )
    return artifact_id


@dataclass(frozen=True)
class TensorSpec:
    """Schema for one named tensor, excluding its leading batch dimension."""

    dtype: torch.dtype
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.dtype not in _DTYPE_NAMES:
            raise ValueError(f"Unsupported cache tensor dtype: {self.dtype}")
        shape = tuple(int(value) for value in self.shape)
        if not shape or any(value == 0 or value < -1 for value in shape):
            raise ValueError(
                "Cache tensor dimensions must be positive or -1 for one ragged "
                f"leading dimension: {self.shape!r}"
            )
        if shape.count(-1) > 1 or (-1 in shape and shape[0] != -1):
            raise ValueError(
                "Cache tensors support at most one ragged per-entry dimension, "
                f"and it must be first: {self.shape!r}"
            )
        object.__setattr__(self, "shape", shape)

    @property
    def ragged(self) -> bool:
        return bool(self.shape and self.shape[0] == -1)

    def validate_tensor(self, tensor: torch.Tensor, *, batch_size: int) -> None:
        if tensor.dtype != self.dtype:
            raise ValueError(
                f"Cache tensor dtype mismatch: got={tensor.dtype} expected={self.dtype}."
            )
        actual = tuple(tensor.shape)
        expected = (int(batch_size), *self.shape)
        if len(actual) != len(expected) or any(
            wanted != -1 and got != wanted for got, wanted in zip(actual, expected)
        ):
            raise ValueError(
                f"Cache tensor shape mismatch: got={actual} expected={expected}."
            )

    def to_dict(self) -> dict[str, Any]:
        return {"dtype": dtype_name(self.dtype), "shape": list(self.shape)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TensorSpec":
        shape = payload.get("shape")
        if not isinstance(shape, list):
            raise CacheError(f"Tensor spec shape must be a list: {payload!r}")
        return cls(
            dtype=dtype_from_name(str(payload.get("dtype"))),
            shape=tuple(int(value) for value in shape),
        )


@dataclass(frozen=True)
class ArtifactContract:
    """Complete compatibility contract for one independently stored artifact."""

    artifact_id: str
    sample_count: int
    keyspace_fingerprint: str
    key_codec: str
    dependencies: Mapping[str, str]
    values: Mapping[str, TensorSpec]
    value_codec: str = RAW_CODEC

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", normalize_artifact_id(self.artifact_id))
        count = int(self.sample_count)
        if count <= 0:
            raise ValueError("Artifact sample_count must be positive.")
        object.__setattr__(self, "sample_count", count)
        if not str(self.keyspace_fingerprint):
            raise ValueError("Artifact keyspace fingerprint cannot be empty.")
        if not str(self.key_codec):
            raise ValueError("Artifact key codec cannot be empty.")
        dependencies = {
            str(name): str(value) for name, value in sorted(self.dependencies.items())
        }
        if not dependencies:
            raise ValueError("Artifact contract requires dependency fingerprints.")
        object.__setattr__(self, "dependencies", dependencies)
        values: dict[str, TensorSpec] = {}
        for raw_name, spec in sorted(self.values.items()):
            name = str(raw_name).strip()
            if not name:
                raise ValueError("Artifact tensor name cannot be empty.")
            if not isinstance(spec, TensorSpec):
                raise TypeError(f"Artifact tensor {name!r} must use TensorSpec.")
            values[name] = spec
        if not values:
            raise ValueError("Artifact contract requires at least one tensor value.")
        object.__setattr__(self, "values", values)
        codec = normalize_value_codec(self.value_codec)
        object.__setattr__(self, "value_codec", codec)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "artifact_id": self.artifact_id,
            "sample_count": self.sample_count,
            "keyspace_fingerprint": self.keyspace_fingerprint,
            "key_codec": self.key_codec,
            "dependencies": dict(self.dependencies),
            "values": {name: spec.to_dict() for name, spec in self.values.items()},
        }
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactContract":
        dependencies = payload.get("dependencies")
        values = payload.get("values")
        if not isinstance(dependencies, dict) or not isinstance(values, dict):
            raise CacheError(f"Invalid artifact contract payload: {payload!r}")
        if any(not isinstance(spec, dict) for spec in values.values()):
            raise CacheError(f"Invalid artifact tensor schema: {values!r}")
        return cls(
            artifact_id=str(payload.get("artifact_id", "")),
            sample_count=int(payload.get("sample_count", -1)),
            keyspace_fingerprint=str(payload.get("keyspace_fingerprint", "")),
            key_codec=str(payload.get("key_codec", "")),
            dependencies={str(key): str(value) for key, value in dependencies.items()},
            values={
                str(name): TensorSpec.from_dict(spec) for name, spec in values.items()
            },
            value_codec=str(payload.get("value_codec", RAW_CODEC)),
        )

    def validate_batch(
        self,
        keys: Iterable[str],
        values: Mapping[str, torch.Tensor],
    ) -> tuple[str, ...]:
        keys = tuple(str(key) for key in keys)
        if not keys:
            raise ValueError(f"{self.artifact_id} write batch cannot be empty.")
        expected_names = set(self.values)
        actual_names = {str(name) for name in values}
        if actual_names != expected_names:
            raise ValueError(
                f"{self.artifact_id} tensor bundle mismatch: "
                f"missing={sorted(expected_names - actual_names)} "
                f"extra={sorted(actual_names - expected_names)}."
            )
        for name, spec in self.values.items():
            tensor = values[name]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"{self.artifact_id}.{name} must be a torch.Tensor, "
                    f"got {type(tensor)}."
                )
            try:
                spec.validate_tensor(tensor, batch_size=len(keys))
            except ValueError as exc:
                raise ValueError(f"{self.artifact_id}.{name}: {exc}") from exc
        return keys


@dataclass(frozen=True)
class ArtifactSnapshot:
    artifact_id: str
    build_id: str
    manifest: str
    manifest_sha256: str
    contract: ArtifactContract | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", normalize_artifact_id(self.artifact_id))
        if not self.build_id or not self.manifest or len(self.manifest_sha256) != 64:
            raise ValueError(f"Invalid artifact snapshot: {self!r}")
        if self.contract is not None and self.contract.artifact_id != self.artifact_id:
            raise ValueError(
                "Artifact snapshot/contract id mismatch: "
                f"{self.artifact_id} vs {self.contract.artifact_id}."
            )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "artifact_id": self.artifact_id,
            "build_id": self.build_id,
            "manifest": self.manifest,
            "manifest_sha256": self.manifest_sha256,
        }
        if self.contract is not None:
            payload["contract"] = self.contract.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactSnapshot":
        raw_contract = payload.get("contract")
        return cls(
            artifact_id=str(payload.get("artifact_id", "")),
            build_id=str(payload.get("build_id", "")),
            manifest=str(payload.get("manifest", "")),
            manifest_sha256=str(payload.get("manifest_sha256", "")),
            contract=(
                ArtifactContract.from_dict(raw_contract)
                if isinstance(raw_contract, dict)
                else None
            ),
        )


@dataclass(frozen=True)
class ValidatedCacheSnapshot:
    root: str
    catalog_sha256: str
    artifacts: Mapping[str, ArtifactSnapshot]

    def __post_init__(self) -> None:
        root = str(Path(self.root).expanduser().resolve())
        if len(self.catalog_sha256) != 64:
            raise ValueError("Validated snapshot requires a catalog SHA256.")
        artifacts = {
            normalize_artifact_id(name): snapshot
            for name, snapshot in self.artifacts.items()
        }
        if not artifacts:
            raise ValueError("Validated cache snapshot requires artifacts.")
        for name, snapshot in artifacts.items():
            if name != snapshot.artifact_id:
                raise ValueError(
                    f"Snapshot artifact key/id mismatch: {name} vs "
                    f"{snapshot.artifact_id}."
                )
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "artifacts", artifacts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "catalog_sha256": self.catalog_sha256,
            "artifacts": {
                name: snapshot.to_dict() for name, snapshot in self.artifacts.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ValidatedCacheSnapshot":
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, dict):
            raise CacheError("Validated cache snapshot artifacts must be an object.")
        return cls(
            root=str(payload.get("root", "")),
            catalog_sha256=str(payload.get("catalog_sha256", "")),
            artifacts={
                str(name): ArtifactSnapshot.from_dict(value)
                for name, value in artifacts.items()
                if isinstance(value, dict)
            },
        )


@dataclass(frozen=True)
class ArtifactValidationReport:
    artifact_id: str
    build_id: str
    sample_count: int
    shard_count: int
    total_bytes: int
    manifest_seconds: float
    index_seconds: float
    structure_seconds: float
    shard_seconds: float
    total_seconds: float
    validation_mode: str = "full"
    expected_sample_count: int | None = None
    missing_sample_count: int = 0
    coverage_ratio: float = 1.0
    missing_examples: tuple[str, ...] = ()
    validation_workers: int = 0
    shard_stat_seconds: float = 0.0
    shard_read_seconds: float = 0.0
    shard_hash_seconds: float = 0.0
    shard_load_seconds: float = 0.0
    shard_check_seconds: float = 0.0

    def __post_init__(self) -> None:
        expected = (
            int(self.sample_count)
            if self.expected_sample_count is None
            else int(self.expected_sample_count)
        )
        if expected <= 0 or int(self.sample_count) <= 0:
            raise ValueError("Cache validation sample counts must be positive.")
        missing = int(self.missing_sample_count)
        if missing < 0 or int(self.sample_count) + missing != expected:
            raise ValueError(
                "Cache validation coverage counts do not align: "
                f"cached={self.sample_count} missing={missing} expected={expected}."
            )
        object.__setattr__(self, "expected_sample_count", expected)
        object.__setattr__(self, "missing_sample_count", missing)
        object.__setattr__(self, "coverage_ratio", float(self.sample_count) / expected)
        object.__setattr__(
            self,
            "missing_examples",
            tuple(str(value) for value in self.missing_examples),
        )
        object.__setattr__(
            self, "validation_workers", max(0, int(self.validation_workers))
        )

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactValidationReport":
        return cls(
            artifact_id=str(payload["artifact_id"]),
            build_id=str(payload["build_id"]),
            sample_count=int(payload["sample_count"]),
            shard_count=int(payload["shard_count"]),
            total_bytes=int(payload["total_bytes"]),
            manifest_seconds=float(payload.get("manifest_seconds", 0.0)),
            index_seconds=float(payload.get("index_seconds", 0.0)),
            structure_seconds=float(payload.get("structure_seconds", 0.0)),
            shard_seconds=float(payload.get("shard_seconds", 0.0)),
            total_seconds=float(payload.get("total_seconds", 0.0)),
            validation_mode=str(payload.get("validation_mode", "full")),
            expected_sample_count=int(
                payload.get("expected_sample_count", payload["sample_count"])
            ),
            missing_sample_count=int(payload.get("missing_sample_count", 0)),
            coverage_ratio=float(payload.get("coverage_ratio", 1.0)),
            missing_examples=tuple(payload.get("missing_examples", ())),
            validation_workers=int(payload.get("validation_workers", 0)),
            shard_stat_seconds=float(payload.get("shard_stat_seconds", 0.0)),
            shard_read_seconds=float(payload.get("shard_read_seconds", 0.0)),
            shard_hash_seconds=float(payload.get("shard_hash_seconds", 0.0)),
            shard_load_seconds=float(payload.get("shard_load_seconds", 0.0)),
            shard_check_seconds=float(payload.get("shard_check_seconds", 0.0)),
        )


@dataclass(frozen=True)
class PreflightReport:
    artifacts: Mapping[str, ArtifactValidationReport]
    wall_seconds: float

    @property
    def sample_count(self) -> int:
        counts = {
            int(report.expected_sample_count) for report in self.artifacts.values()
        }
        if len(counts) != 1:
            raise CacheError(
                f"Selected artifact expected sample counts disagree: {counts}."
            )
        return next(iter(counts))

    @property
    def shard_count(self) -> int:
        return sum(report.shard_count for report in self.artifacts.values())

    @property
    def total_bytes(self) -> int:
        return sum(report.total_bytes for report in self.artifacts.values())

    @property
    def total_seconds(self) -> float:
        return sum(report.total_seconds for report in self.artifacts.values())

    @property
    def validation_mode(self) -> str:
        modes = {report.validation_mode for report in self.artifacts.values()}
        if len(modes) != 1:
            raise CacheError(f"Selected artifact validation modes disagree: {modes}.")
        return next(iter(modes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifacts": {
                name: report.to_dict() for name, report in self.artifacts.items()
            },
            "sample_count": self.sample_count,
            "shard_count": self.shard_count,
            "total_bytes": self.total_bytes,
            "total_seconds": self.total_seconds,
            "validation_mode": self.validation_mode,
            "wall_seconds": float(self.wall_seconds),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PreflightReport":
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, dict):
            raise CacheError("Preflight report artifacts must be an object.")
        return cls(
            artifacts={
                str(name): ArtifactValidationReport.from_dict(report)
                for name, report in artifacts.items()
                if isinstance(report, dict)
            },
            wall_seconds=float(payload.get("wall_seconds", 0.0)),
        )


@dataclass(frozen=True)
class PublishReport:
    build_id: str
    artifacts: Mapping[str, ArtifactValidationReport]
    catalog: str

    @property
    def total_bytes(self) -> int:
        return sum(report.total_bytes for report in self.artifacts.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "build_id": self.build_id,
            "artifacts": {
                name: report.to_dict() for name, report in self.artifacts.items()
            },
            "catalog": self.catalog,
            "total_bytes": self.total_bytes,
        }
