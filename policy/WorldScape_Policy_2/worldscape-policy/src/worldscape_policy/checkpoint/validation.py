from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


NATIVE_CHECKPOINT_FORMAT_VERSION = "2"


REQUIRED_NATIVE_GROUPS = frozenset(
    {
        "auto_conditioner",
        "event_memory",
        "interactive_conditioner",
        "visual_codec",
        "wam_image_encoder",
        "wam_core",
    }
)

INTERACTIVE_NATIVE_GROUPS = frozenset(
    {
        "interactive_conditioner",
        "visual_codec",
        "wam_image_encoder",
        "wam_core",
    }
)


class CheckpointConversionError(RuntimeError):
    pass


@dataclass
class GroupCoverage:
    tensors: int = 0
    parameters: int = 0
    dtypes: set[str] = field(default_factory=set)


@dataclass
class ConversionResult:
    state_dict: dict[str, Any]
    report: "ConversionReport"
    key_mapping_version: str

    def manifest(
        self,
        *,
        model_variant: str,
        source_checkpoint_hash: str,
        source_files: list[dict[str, Any]],
        git_commit: str | None = None,
    ) -> dict[str, Any]:
        return {
            "format_version": NATIVE_CHECKPOINT_FORMAT_VERSION,
            "model_variant": model_variant,
            "wam_plugin": "wan22",
            "source_checkpoint_hash": source_checkpoint_hash,
            "source_files": source_files,
            "key_mapping_version": self.key_mapping_version,
            "source_tensors": self.report.source_tensors,
            "converted_tensors": self.report.converted_tensors,
            "coverage": self.report.coverage,
            "unmapped_keys": sorted(self.report.unmapped_keys),
            "rank_mismatches": sorted(self.report.rank_mismatches),
            "shape_mismatches": sorted(self.report.shape_mismatches),
            "missing_target_keys": sorted(self.report.missing_target_keys),
            "collisions": sorted(self.report.collisions),
            "loaded_keys": sorted(self.state_dict),
            "missing_keys": sorted(self.report.missing_target_keys),
            "unexpected_keys": sorted(self.report.unmapped_keys),
            "groups": {
                name: {
                    "tensors": coverage.tensors,
                    "parameters": coverage.parameters,
                    "dtypes": sorted(coverage.dtypes),
                }
                for name, coverage in sorted(self.report.groups.items())
            },
            "git_commit": git_commit,
        }


@dataclass
class ConversionReport:
    source_tensors: int = 0
    converted_tensors: int = 0
    groups: dict[str, GroupCoverage] = field(default_factory=dict)
    unmapped_keys: list[str] = field(default_factory=list)
    rank_mismatches: list[str] = field(default_factory=list)
    shape_mismatches: list[str] = field(default_factory=list)
    missing_target_keys: list[str] = field(default_factory=list)
    collisions: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        if self.source_tensors == 0:
            return 1.0
        return self.converted_tensors / self.source_tensors

    def validate(self, *, required_groups: frozenset[str] = REQUIRED_NATIVE_GROUPS) -> None:
        failures: list[str] = []
        if self.unmapped_keys:
            failures.append(f"{len(self.unmapped_keys)} unmapped tensor(s)")
        if self.rank_mismatches:
            failures.append(f"{len(self.rank_mismatches)} rank mismatch(es)")
        if self.shape_mismatches:
            failures.append(
                f"{len(self.shape_mismatches)} shape mismatch(es): "
                + ", ".join(self.shape_mismatches[:5])
            )
        if self.missing_target_keys:
            failures.append(
                f"{len(self.missing_target_keys)} converted key(s) absent from target model"
            )
        if self.collisions:
            failures.append(f"{len(self.collisions)} target-key collision(s)")
        missing_groups = sorted(required_groups.difference(self.groups))
        if missing_groups:
            failures.append(f"missing required groups: {', '.join(missing_groups)}")
        if self.converted_tensors != self.source_tensors:
            failures.append(
                f"coverage is {self.converted_tensors}/{self.source_tensors} "
                f"({self.coverage:.2%}), expected 100%"
            )
        if failures:
            raise CheckpointConversionError("; ".join(failures))

    def validate_target(
        self,
        converted_state_dict: Mapping[str, Any],
        target_state_dict: Mapping[str, Any],
    ) -> None:
        """Validate converted tensor names and shapes against a real module tree."""

        self.shape_mismatches.clear()
        self.missing_target_keys.clear()
        for key, source_tensor in converted_state_dict.items():
            target_tensor = target_state_dict.get(key)
            if target_tensor is None:
                self.missing_target_keys.append(key)
                continue
            source_shape = getattr(source_tensor, "shape", None)
            target_shape = getattr(target_tensor, "shape", None)
            if source_shape is not None and target_shape is not None:
                if tuple(source_shape) != tuple(target_shape):
                    self.shape_mismatches.append(
                        f"{key}: source={tuple(source_shape)}, target={tuple(target_shape)}"
                    )
