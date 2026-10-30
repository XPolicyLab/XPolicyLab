from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


TRANSFORM_BUNDLE_FILENAME = "transform_bundle.json"
TRANSFORM_BUNDLE_SCHEMA_VERSION = "1"
_NORMALIZATION_MODES = {"q99", "mean_std", "min_max", "binary", "scale"}


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


@dataclass(frozen=True)
class TransformField:
    key: str
    size: int
    normalization: str | None
    statistics: dict[str, list[Any]]
    per_horizon_statistics: dict[str, list[Any]] | None
    absolute: bool

    def __post_init__(self) -> None:
        if not self.key.startswith(("state.", "action.")):
            raise ValueError(f"Invalid transform field key: {self.key!r}")
        if isinstance(self.size, bool) or self.size <= 0:
            raise ValueError(f"{self.key} size must be a positive integer")
        if self.normalization is not None and self.normalization not in _NORMALIZATION_MODES:
            raise ValueError(
                f"{self.key} has unsupported normalization {self.normalization!r}"
            )
        required = {
            "q99": {"q01", "q99"},
            "mean_std": {"mean", "std"},
            "min_max": {"min", "max"},
            "scale": {"min", "max"},
            "binary": set(),
        }.get(self.normalization, set())
        if not required.issubset(self.statistics):
            raise ValueError(
                f"{self.key} is missing normalization statistics: "
                f"{sorted(required - self.statistics.keys())}"
            )
        for name, values in self.statistics.items():
            array = np.asarray(values)
            if array.ndim != 1 or array.shape[0] != self.size:
                raise ValueError(
                    f"{self.key}.{name} must have shape [{self.size}], got {array.shape}"
                )
            if not np.isfinite(array).all():
                raise ValueError(f"{self.key}.{name} contains non-finite values")
        if self.per_horizon_statistics is not None:
            if self.normalization is None:
                raise ValueError(f"{self.key} has per-horizon stats without normalization")
            for name in required:
                if name not in self.per_horizon_statistics:
                    raise ValueError(
                        f"{self.key} per-horizon statistics are missing {name!r}"
                    )
            horizons: set[int] = set()
            for name, values in self.per_horizon_statistics.items():
                array = np.asarray(values)
                if array.ndim != 2 or array.shape[1] != self.size or array.shape[0] < 1:
                    raise ValueError(
                        f"{self.key} per-horizon {name} must have shape [H,{self.size}]"
                    )
                if not np.isfinite(array).all():
                    raise ValueError(
                        f"{self.key} per-horizon {name} contains non-finite values"
                    )
                horizons.add(array.shape[0])
            if len(horizons) != 1:
                raise ValueError(f"{self.key} per-horizon statistic lengths differ")

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "size": self.size,
            "normalization": self.normalization,
            "statistics": self.statistics,
            "per_horizon_statistics": self.per_horizon_statistics,
            "absolute": self.absolute,
        }

    @classmethod
    def from_dict(cls, value: object) -> TransformField:
        if not isinstance(value, dict):
            raise ValueError("Transform field must be an object")
        expected = {
            "key",
            "size",
            "normalization",
            "statistics",
            "per_horizon_statistics",
            "absolute",
        }
        if set(value) != expected:
            raise ValueError(
                f"Transform field keys differ: expected {sorted(expected)}, "
                f"got {sorted(value)}"
            )
        if (
            not isinstance(value["key"], str)
            or not isinstance(value["size"], int)
            or (
                value["normalization"] is not None
                and not isinstance(value["normalization"], str)
            )
            or not isinstance(value["statistics"], dict)
            or (
                value["per_horizon_statistics"] is not None
                and not isinstance(value["per_horizon_statistics"], dict)
            )
            or not isinstance(value["absolute"], bool)
        ):
            raise ValueError(f"Transform field {value.get('key')!r} has invalid types")
        return cls(
            key=value["key"],
            size=value["size"],
            normalization=value["normalization"],
            statistics=value["statistics"],
            per_horizon_statistics=value["per_horizon_statistics"],
            absolute=value["absolute"],
        )


@dataclass(frozen=True)
class EmbodimentTransform:
    embodiment_id: int
    max_state_dim: int
    max_action_dim: int
    state_fields: tuple[TransformField, ...]
    action_fields: tuple[TransformField, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.embodiment_id, bool)
            or self.embodiment_id < 0
            or self.max_state_dim < 1
            or self.max_action_dim < 1
        ):
            raise ValueError("Embodiment IDs and dimensions must be valid positive values")
        if not self.state_fields or not self.action_fields:
            raise ValueError("Embodiment transform requires state and action fields")
        if sum(item.size for item in self.state_fields) > self.max_state_dim:
            raise ValueError("State fields exceed max_state_dim")
        if sum(item.size for item in self.action_fields) > self.max_action_dim:
            raise ValueError("Action fields exceed max_action_dim")
        state_keys = [item.key for item in self.state_fields]
        action_keys = [item.key for item in self.action_fields]
        if len(state_keys) != len(set(state_keys)) or len(action_keys) != len(set(action_keys)):
            raise ValueError("Transform field keys must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "embodiment_id": self.embodiment_id,
            "max_state_dim": self.max_state_dim,
            "max_action_dim": self.max_action_dim,
            "state_fields": [item.to_dict() for item in self.state_fields],
            "action_fields": [item.to_dict() for item in self.action_fields],
        }

    @classmethod
    def from_dict(cls, value: object) -> EmbodimentTransform:
        if not isinstance(value, dict):
            raise ValueError("Embodiment transform must be an object")
        expected = {
            "embodiment_id",
            "max_state_dim",
            "max_action_dim",
            "state_fields",
            "action_fields",
        }
        if set(value) != expected:
            raise ValueError("Embodiment transform has unknown or missing fields")
        if not all(
            isinstance(value[name], int)
            for name in ("embodiment_id", "max_state_dim", "max_action_dim")
        ) or not all(
            isinstance(value[name], list) for name in ("state_fields", "action_fields")
        ):
            raise ValueError("Embodiment transform has invalid field types")
        return cls(
            embodiment_id=value["embodiment_id"],
            max_state_dim=value["max_state_dim"],
            max_action_dim=value["max_action_dim"],
            state_fields=tuple(
                TransformField.from_dict(item) for item in value["state_fields"]
            ),
            action_fields=tuple(
                TransformField.from_dict(item) for item in value["action_fields"]
            ),
        )


@dataclass(frozen=True)
class CheckpointTransformArtifact:
    image_input_range: str
    embodiments: dict[str, EmbodimentTransform]
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        if self.image_input_range not in {"uint8", "zero_one", "minus_one_one"}:
            raise ValueError(f"Invalid image input range: {self.image_input_range!r}")
        if not self.embodiments or any(
            not isinstance(tag, str) or not tag for tag in self.embodiments
        ):
            raise ValueError("Transform bundle must contain named embodiments")
        if not isinstance(self.provenance, dict) or not self.provenance:
            raise ValueError("Transform bundle provenance is required")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": TRANSFORM_BUNDLE_SCHEMA_VERSION,
            "image": {
                "raw_range": "uint8",
                "model_input_range": self.image_input_range,
            },
            "embodiments": {
                tag: item.to_dict() for tag, item in sorted(self.embodiments.items())
            },
            "provenance": self.provenance,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        return {**payload, "checksum": _sha256_bytes(_canonical_json(payload))}

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def read(cls, path: str | Path) -> CheckpointTransformArtifact:
        bundle_path = Path(path)
        try:
            value = json.loads(bundle_path.read_text())
        except json.JSONDecodeError as error:
            raise ValueError(f"{bundle_path.name} is not valid JSON") from error
        if not isinstance(value, dict):
            raise ValueError(f"{bundle_path.name} must contain an object")
        expected = {
            "schema_version",
            "image",
            "embodiments",
            "provenance",
            "checksum",
        }
        if set(value) != expected:
            raise ValueError("Transform bundle has unknown or missing top-level fields")
        if value["schema_version"] != TRANSFORM_BUNDLE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported transform bundle schema {value['schema_version']!r}"
            )
        checksum = value.pop("checksum")
        actual = _sha256_bytes(_canonical_json(value))
        if not isinstance(checksum, str) or checksum != actual:
            raise ValueError("Transform bundle checksum mismatch")
        image = value["image"]
        if (
            not isinstance(image, dict)
            or set(image) != {"raw_range", "model_input_range"}
            or image["raw_range"] != "uint8"
            or not isinstance(image["model_input_range"], str)
        ):
            raise ValueError("Transform bundle image schema is invalid")
        if not isinstance(value["embodiments"], dict):
            raise ValueError("Transform bundle embodiments must be an object")
        if not isinstance(value["provenance"], dict):
            raise ValueError("Transform bundle provenance must be an object")
        return cls(
            image_input_range=image["model_input_range"],
            embodiments={
                tag: EmbodimentTransform.from_dict(item)
                for tag, item in value["embodiments"].items()
            },
            provenance=value["provenance"],
        )


class NativeCheckpointTransform:
    """WSP-owned image/state/action transform with no Groot runtime imports."""

    def __init__(
        self,
        *,
        image_input_range: str,
        embodiment: EmbodimentTransform,
    ) -> None:
        self.image_input_range = image_input_range
        self.embodiment = embodiment
        self.relative_action_keys = frozenset(
            item.key for item in embodiment.action_fields if not item.absolute
        )

    def eval(self) -> NativeCheckpointTransform:
        return self

    def apply_image(self, value: torch.Tensor) -> torch.Tensor:
        if value.dtype != torch.uint8:
            raise TypeError("Native image transform expects uint8 input")
        if self.image_input_range == "uint8":
            return value
        result = value.float().div(255.0)
        if self.image_input_range == "minus_one_one":
            result = result.mul(2.0).sub(1.0)
        return result

    def apply_state(self, data: Mapping[str, Any]) -> torch.Tensor:
        values = []
        for field in self.embodiment.state_fields:
            if field.key not in data:
                raise KeyError(f"State input is missing {field.key!r}")
            value = torch.as_tensor(data[field.key]).float()
            if value.shape[-1] != field.size:
                raise ValueError(
                    f"{field.key} expected width {field.size}, got {value.shape[-1]}"
                )
            values.append(_normalize(value, field, inverse=False))
        return torch.cat(values, dim=-1)

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        result = dict(data)
        if any(field.key in result for field in self.embodiment.state_fields):
            for field in self.embodiment.state_fields:
                if field.key not in result:
                    raise KeyError(f"State input is missing {field.key!r}")
            result["state"] = self.apply_state(result)
            for field in self.embodiment.state_fields:
                result.pop(field.key)
        if any(field.key in result for field in self.embodiment.action_fields):
            values = []
            for field in self.embodiment.action_fields:
                if field.key not in result:
                    raise KeyError(f"Action input is missing {field.key!r}")
                value = torch.as_tensor(result.pop(field.key)).float()
                values.append(_normalize(value, field, inverse=False))
            result["action"] = torch.cat(values, dim=-1)
        return result

    __call__ = apply

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        result = dict(data)
        if "action" in result:
            packed = torch.as_tensor(result.pop("action"))
            width = sum(field.size for field in self.embodiment.action_fields)
            if packed.shape[-1] < width:
                raise ValueError(
                    f"Packed action width {packed.shape[-1]} is smaller than {width}"
                )
            offset = 0
            for field in self.embodiment.action_fields:
                value = packed[..., offset : offset + field.size]
                result[field.key] = _normalize(value, field, inverse=True)
                offset += field.size
        if "state" in result:
            packed = torch.as_tensor(result.pop("state"))
            width = sum(field.size for field in self.embodiment.state_fields)
            if packed.shape[-1] < width:
                raise ValueError(
                    f"Packed state width {packed.shape[-1]} is smaller than {width}"
                )
            offset = 0
            for field in self.embodiment.state_fields:
                value = packed[..., offset : offset + field.size]
                result[field.key] = _normalize(value, field, inverse=True)
                offset += field.size
        return result


def _normalize(
    value: torch.Tensor,
    field: TransformField,
    *,
    inverse: bool,
) -> torch.Tensor:
    mode = field.normalization
    if mode is None:
        return value
    statistics = field.per_horizon_statistics or field.statistics
    tensors = {
        name: torch.as_tensor(item, dtype=value.dtype, device=value.device)
        for name, item in statistics.items()
    }
    if field.per_horizon_statistics is not None:
        available = next(iter(tensors.values())).shape[0]
        if value.ndim == 1:
            tensors = {name: item[0] for name, item in tensors.items()}
        elif value.ndim == 2 and value.shape[0] > available:
            if value.shape[0] % available:
                raise ValueError(
                    f"{field.key} flattened horizon {value.shape[0]} is not a "
                    f"multiple of statistics horizon {available}"
                )
            return _normalize(
                value.reshape(-1, available, value.shape[-1]),
                field,
                inverse=inverse,
            ).reshape(value.shape)
        horizon = value.shape[-2] if value.ndim >= 2 else 1
        if horizon > available:
            raise ValueError(
                f"{field.key} horizon {horizon} exceeds statistics horizon {available}"
            )
        if value.ndim >= 2:
            tensors = {name: item[:horizon] for name, item in tensors.items()}
    if mode == "q99":
        low, high = tensors["q01"], tensors["q99"]
        if inverse:
            return (value + 1) * 0.5 * (high - low) + low
        safe = torch.where(high == low, torch.ones_like(high), high - low)
        result = 2 * (value - low) / safe - 1
        return torch.where(high == low, value, result).clamp(-1, 1)
    if mode == "mean_std":
        mean, std = tensors["mean"], tensors["std"]
        if inverse:
            return value * std + mean
        safe = torch.where(std == 0, torch.ones_like(std), std)
        result = torch.where(std == 0, value, (value - mean) / safe)
        if "clip_min" in tensors or "clip_max" in tensors:
            if "clip_min" not in tensors or "clip_max" not in tensors:
                raise ValueError(
                    f"{field.key} mean_std clipping requires clip_min and clip_max"
                )
            result = torch.maximum(
                torch.minimum(result, tensors["clip_max"]),
                tensors["clip_min"],
            )
        return result
    if mode == "min_max":
        low, high = tensors["min"], tensors["max"]
        if inverse:
            return (value + 1) * 0.5 * (high - low) + low
        safe = torch.where(high == low, torch.ones_like(high), high - low)
        return torch.where(
            high == low, torch.zeros_like(value), 2 * (value - low) / safe - 1
        ).clamp(-1, 1)
    if mode == "scale":
        if inverse:
            scale = torch.maximum(tensors["min"].abs(), tensors["max"].abs())
            return value * scale
        scale = torch.maximum(tensors["min"].abs(), tensors["max"].abs())
        safe = torch.where(scale == 0, torch.ones_like(scale), scale)
        return torch.where(scale == 0, torch.zeros_like(value), value / safe)
    if mode == "binary":
        return (value > 0.5).to(value.dtype)
    raise ValueError(f"Unsupported normalization mode: {mode!r}")
