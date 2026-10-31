"""Benchmark-neutral contracts. Implementations own environment-specific details."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Mapping, Protocol, Sequence


def safe_id(value: str) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value))


@dataclass(frozen=True)
class ActionContract:
    representation: str
    fields: tuple[str, ...]
    frame: str
    units: str
    limits: Mapping[str, Any]
    max_chunk_steps: int

    def __post_init__(self) -> None:
        if not all((self.representation, self.fields, self.frame, self.units, self.limits)):
            raise ValueError("action contract needs representation, fields, frame, units and limits")
        if self.max_chunk_steps < 1:
            raise ValueError("max_chunk_steps must be positive")


@dataclass(frozen=True)
class ScenarioSpec:
    benchmark: str
    task: str
    seed: int
    reset: Mapping[str, Any]
    permitted_observations: tuple[str, ...]
    action: ActionContract
    budgets: Mapping[str, Any]
    terminal_conditions: tuple[str, ...]
    assets: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.benchmark or not self.task or self.seed < 0:
            raise ValueError("benchmark, task and nonnegative seed are required")
        if not self.reset or not self.permitted_observations or not self.budgets or not self.terminal_conditions:
            raise ValueError("reset, observations, budgets and terminal conditions are required")


@dataclass(frozen=True)
class PolicyEntry:
    kind: str
    policy_dir: Path
    policy_env: str
    checkpoint_name: str
    eval_env: str
    checkpoint_dir: Path | None = None
    endpoint: str | None = None

    def __post_init__(self) -> None:
        if not safe_id(self.kind) or not self.policy_env or not self.eval_env or not safe_id(self.checkpoint_name):
            raise ValueError("policy entry needs kind, environments and checkpoint name")
        if self.kind == "oeh" and not self.endpoint:
            raise ValueError("OEH policy entry needs an endpoint")


@dataclass(frozen=True)
class ConditionSpec:
    id: str
    display_name: str
    policy: PolicyEntry
    model: Mapping[str, Any]
    inputs: Mapping[str, Any]
    budgets: Mapping[str, Any]
    compute: Mapping[str, Any]
    configuration: Mapping[str, Any] = field(default_factory=dict)
    evidence_requirements: tuple[str, ...] = ()
    identity_files: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        if not safe_id(self.id) or not self.display_name.strip():
            raise ValueError("condition needs a safe ID and display name")
        if not self.model or not self.inputs or not self.budgets or not self.compute:
            raise ValueError("condition needs model identity, inputs, budgets and compute")
        if not self.evidence_requirements:
            raise ValueError("condition needs explicit evidence requirements")


@dataclass(frozen=True)
class ComparisonSpec:
    id: str
    left: str
    right: str
    match_on: tuple[str, ...]
    require_equal: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not safe_id(self.id) or not safe_id(self.left) or not safe_id(self.right) or self.left == self.right:
            raise ValueError("comparison needs an ID and two distinct condition IDs")
        if not self.match_on or any(not isinstance(field, str) or not field for field in self.match_on) or len(set(self.match_on)) != len(self.match_on):
            raise ValueError("comparison needs unique matching fields")
        if any(not isinstance(field, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", field)
               for field in self.require_equal) or len(set(self.require_equal)) != len(self.require_equal):
            raise ValueError("comparison equality fields must be unique dotted paths")


@dataclass(frozen=True)
class AttemptPlan:
    attempt_id: str
    split: str
    scenario: ScenarioSpec
    condition: ConditionSpec
    output: Path


@dataclass(frozen=True)
class LaunchSpec:
    command: tuple[str, ...]
    cwd: Path
    environment: Mapping[str, str]
    wall_seconds: int
    # What the adapter decided about the launch beyond the command (e.g. oeh_eval.vulkan_compat's decision),
    # recorded with the attempt's run_started event; empty for every adapter that decides nothing.
    provenance: Mapping[str, Any] = field(default_factory=dict)


def external_model_id(model: Mapping[str, Any]) -> str | None:
    """The hosted model a condition consults beside its VLA: a hybrid's GPT (gpt_model_id) or a review-mode teacher
    (teacher_model_id). Its calls get the hashed-input and unique-request checks and are counted and timed."""
    return model.get("gpt_model_id") or model.get("teacher_model_id")


# A NativeOutcome reason that starts with this: the benchmark itself voided the episode (RoboDojo counts it under
# "Unstable nums" and leaves it out of its eval total). panel60's rule: record it and exclude the case, never rerun.
NATIVE_VOID = "native episode voided by the benchmark"


@dataclass(frozen=True)
class NativeOutcome:
    complete: bool
    valid: bool
    success: bool | None
    score: float | None
    actual_layout: str | None
    evidence_path: Path | None
    reason: str | None = None


class Scenario(Protocol):
    def preflight(self, plan: AttemptPlan) -> Sequence[str]: ...
    def launch(self, plan: AttemptPlan, run_id: str, sidecar: Path) -> LaunchSpec: ...
    def result_path(self, plan: AttemptPlan, run_id: str) -> Path | None: ...


class Judge(Protocol):
    def read_native(self, plan: AttemptPlan, result_path: Path | None) -> NativeOutcome: ...


class Recorder(Protocol):
    def append(self, kind: str, payload: Mapping[str, Any]) -> None: ...
    def attach(self, name: str, source: Path) -> None: ...
    def seal(self, outcome: NativeOutcome, status: str, *, status_reason: str | None = None,
             evidence_issues: list[str] | None = None) -> None: ...


class CampaignAdapterConfig(Protocol):
    """Benchmark-specific configuration consumed by the shared campaign loop."""
    source_file: Path
    benchmark: str
    benchmark_settings: Mapping[str, Any]
    task: str
    seeds: tuple[int, ...]
    split: str
    conditions: Mapping[str, ConditionSpec]
    comparisons: tuple[ComparisonSpec, ...]

    def scenario(self, seed: int, condition_id: str) -> ScenarioSpec: ...
    def make_scenario(self) -> Scenario: ...
    def make_judge(self) -> Judge: ...
    def freeze_shared(self) -> Mapping[str, Any]: ...
    def freeze_condition(self, condition: ConditionSpec) -> Mapping[str, Any]: ...
