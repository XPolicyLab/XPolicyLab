"""Explicit module-path freezing policies and startup reports."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from torch import nn


@dataclass(frozen=True)
class FreezeRule:
    """Set all parameters below a native ``nn.Module.get_submodule`` path."""

    module_path: str
    frozen: bool
    optimizer_group: str | None = None
    initialization_source: str | None = None

    def __post_init__(self) -> None:
        if self.module_path.startswith(".") or self.module_path.endswith("."):
            raise ValueError("module_path must not start or end with '.'")


@dataclass(frozen=True)
class ParameterFreezeRecord:
    name: str
    numel: int
    trainable: bool
    module_path: str
    optimizer_group: str | None = None
    initialization_source: str | None = None


@dataclass(frozen=True)
class ModuleFreezeSummary:
    module_path: str
    trainable_parameters: int
    frozen_parameters: int
    trainable_tensors: int
    frozen_tensors: int


@dataclass(frozen=True)
class FreezeReport:
    """Auditable parameter-level result of applying a freeze policy."""

    parameters: tuple[ParameterFreezeRecord, ...]
    modules: tuple[ModuleFreezeSummary, ...]
    unmatched_rules: tuple[str, ...] = ()
    unused_trainable_names: tuple[str, ...] = ()

    @property
    def trainable_names(self) -> tuple[str, ...]:
        return tuple(record.name for record in self.parameters if record.trainable)

    @property
    def frozen_names(self) -> tuple[str, ...]:
        return tuple(record.name for record in self.parameters if not record.trainable)

    @property
    def trainable_parameters(self) -> int:
        return sum(record.numel for record in self.parameters if record.trainable)

    @property
    def frozen_parameters(self) -> int:
        return sum(record.numel for record in self.parameters if not record.trainable)

    @property
    def optimizer_groups(self) -> Mapping[str, tuple[str, ...]]:
        groups: dict[str, list[str]] = {}
        for record in self.parameters:
            if record.trainable and record.optimizer_group is not None:
                groups.setdefault(record.optimizer_group, []).append(record.name)
        return {name: tuple(parameters) for name, parameters in groups.items()}

    def as_dict(self) -> dict[str, object]:
        return {
            "trainable_parameter_names": list(self.trainable_names),
            "frozen_parameter_names": list(self.frozen_names),
            "trainable_parameters": self.trainable_parameters,
            "frozen_parameters": self.frozen_parameters,
            "modules": [
                {
                    "module_path": summary.module_path,
                    "trainable_parameters": summary.trainable_parameters,
                    "frozen_parameters": summary.frozen_parameters,
                    "trainable_tensors": summary.trainable_tensors,
                    "frozen_tensors": summary.frozen_tensors,
                }
                for summary in self.modules
            ],
            "optimizer_groups": {
                name: list(parameters)
                for name, parameters in self.optimizer_groups.items()
            },
            "initialization_sources": {
                record.name: record.initialization_source
                for record in self.parameters
                if record.initialization_source is not None
            },
            "unmatched_rules": list(self.unmatched_rules),
            "unused_trainable_parameter_names": list(self.unused_trainable_names),
        }


class FreezePolicy:
    """Apply longest-prefix module rules without legacy class-name matching."""

    def __init__(
        self,
        rules: tuple[FreezeRule, ...] | list[FreezeRule],
        *,
        strict: bool = True,
    ) -> None:
        paths = [rule.module_path for rule in rules]
        if len(paths) != len(set(paths)):
            raise ValueError("freeze rules must have unique module paths")
        self.rules = tuple(rules)
        self.strict = strict

    @classmethod
    def from_mapping(
        cls,
        frozen_by_path: Mapping[str, bool],
        *,
        strict: bool = True,
    ) -> FreezePolicy:
        return cls(
            [
                FreezeRule(module_path=path, frozen=frozen)
                for path, frozen in frozen_by_path.items()
            ],
            strict=strict,
        )

    def apply(
        self,
        model: nn.Module,
        *,
        optimizer_group: Callable[[str], str | None] | None = None,
        initialization_source: Callable[[str], str | None] | None = None,
        unused_module_paths: tuple[str, ...] | list[str] = (),
    ) -> FreezeReport:
        discovered_unused = list(unused_module_paths)
        for module_path, module in model.named_modules():
            detector = getattr(module, "unused_parameter_module_paths", None)
            if not callable(detector):
                continue
            for relative_path in detector():
                discovered_unused.append(
                    ".".join(part for part in (module_path, relative_path) if part)
                )
        matched_paths: set[str] = set()
        for rule in self.rules:
            try:
                model.get_submodule(rule.module_path)
            except AttributeError:
                continue
            matched_paths.add(rule.module_path)
        unmatched = tuple(
            rule.module_path
            for rule in self.rules
            if rule.module_path not in matched_paths
        )
        if unmatched and self.strict:
            raise ValueError(
                "freeze policy paths do not exist: " + ", ".join(repr(p) for p in unmatched)
            )

        records: list[ParameterFreezeRecord] = []
        for name, parameter in model.named_parameters():
            rule = self._rule_for_parameter(name, matched_paths)
            if rule is not None:
                parameter.requires_grad_(not rule.frozen)
            records.append(
                ParameterFreezeRecord(
                    name=name,
                    numel=parameter.numel(),
                    trainable=parameter.requires_grad,
                    module_path=rule.module_path if rule is not None else "",
                    optimizer_group=(
                        optimizer_group(name)
                        if optimizer_group is not None
                        else rule.optimizer_group if rule is not None else None
                    ),
                    initialization_source=(
                        initialization_source(name)
                        if initialization_source is not None
                        else rule.initialization_source if rule is not None else None
                    ),
                )
            )
        return FreezeReport(
            parameters=tuple(records),
            modules=self._summaries(records),
            unmatched_rules=unmatched,
            unused_trainable_names=tuple(
                record.name
                for record in records
                if record.trainable
                and any(
                    record.name == path or record.name.startswith(path + ".")
                    for path in discovered_unused
                )
            ),
        )

    def _rule_for_parameter(
        self,
        parameter_name: str,
        matched_paths: set[str],
    ) -> FreezeRule | None:
        candidates = [
            rule
            for rule in self.rules
            if rule.module_path in matched_paths
            and (
                not rule.module_path
                or parameter_name == rule.module_path
                or parameter_name.startswith(rule.module_path + ".")
            )
        ]
        return max(candidates, key=lambda rule: len(rule.module_path), default=None)

    @staticmethod
    def _summaries(
        records: list[ParameterFreezeRecord],
    ) -> tuple[ModuleFreezeSummary, ...]:
        grouped: dict[str, list[ParameterFreezeRecord]] = {}
        for record in records:
            grouped.setdefault(record.module_path, []).append(record)
        return tuple(
            ModuleFreezeSummary(
                module_path=path,
                trainable_parameters=sum(r.numel for r in values if r.trainable),
                frozen_parameters=sum(r.numel for r in values if not r.trainable),
                trainable_tensors=sum(r.trainable for r in values),
                frozen_tensors=sum(not r.trainable for r in values),
            )
            for path, values in sorted(grouped.items())
        )


@dataclass(frozen=True)
class NativeFreezeConfig:
    """High-level flags resolved exclusively to native registered module paths."""

    vlm: bool = True
    qformer: bool | None = None
    t5: bool = True
    vae: bool = True
    image_encoder: bool = True
    wam: bool = False
    action_adapters: bool = False
    condition_projector: bool = False
    event_memory: bool = False
    path_overrides: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


NATIVE_MODULE_PATHS: Mapping[str, tuple[str, ...]] = {
    "vlm": ("condition_router.auto.vlm",),
    "qformer": ("condition_router.auto.vlm.qformer",),
    "t5": ("condition_router.interactive.t5",),
    "vae": ("visual_memory.codec.vae",),
    "image_encoder": ("wam.image_encoder",),
    "wam": ("wam",),
    "action_adapters": (
        "wam.core.action_encoder",
        "wam.core.state_encoder",
        "wam.core.action_decoder",
    ),
    "condition_projector": (
        "condition_router.auto.projector",
        "condition_router.interactive.projector",
    ),
    "event_memory": ("condition_router.auto.event_memory",),
}


def native_freeze_policy(
    config: NativeFreezeConfig,
    *,
    strict: bool = True,
) -> FreezePolicy:
    """Resolve public freeze flags to explicit native module paths."""

    rules: list[FreezeRule] = []
    for component in NATIVE_MODULE_PATHS:
        if component == "qformer" and config.qformer is None:
            continue
        paths = config.path_overrides.get(component, NATIVE_MODULE_PATHS[component])
        frozen = bool(getattr(config, component))
        rules.extend(FreezeRule(path, frozen) for path in paths)
    return FreezePolicy(rules, strict=strict)


ModulePathFreezePolicy = FreezePolicy


__all__ = [
    "NATIVE_MODULE_PATHS",
    "FreezePolicy",
    "FreezeReport",
    "FreezeRule",
    "ModuleFreezeSummary",
    "ModulePathFreezePolicy",
    "NativeFreezeConfig",
    "ParameterFreezeRecord",
    "native_freeze_policy",
]
