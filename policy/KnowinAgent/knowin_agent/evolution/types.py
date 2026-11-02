"""Episode and pipeline specifications (docs/INTERFACES.md, M4)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Optional

PHASES = ("practice", "test")


@dataclass
class Budget:
    executions: int
    wall_seconds: float = 3600.0

    def __post_init__(self):
        if int(self.executions) < 0 or float(self.wall_seconds) <= 0:
            raise ValueError("budget needs executions >= 0 and wall_seconds > 0")
        self.executions = int(self.executions)
        self.wall_seconds = float(self.wall_seconds)

    @classmethod
    def of(cls, value) -> "Budget":
        if isinstance(value, Budget):
            return value
        if isinstance(value, dict):
            return cls(**value)
        if isinstance(value, (list, tuple)):
            return cls(*value)
        return cls(int(value))


@dataclass
class EpisodeSpec:
    task: str
    seed: int
    phase: str
    knowledge_from: Optional[str] = None
    shared_library: Optional[str] = None
    budget: Budget = field(default_factory=lambda: Budget(45))
    switches: dict = field(default_factory=dict)
    model: Optional[str] = None
    label: str = ""

    def __post_init__(self):
        if self.phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}")
        self.budget = Budget.of(self.budget)
        self.seed = int(self.seed)
        if self.knowledge_from is not None:
            self.knowledge_from = str(self.knowledge_from)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EpisodeSpec":
        return cls(**{f.name: data[f.name] for f in fields(cls) if f.name in data})


@dataclass
class EpisodeResult:
    score: Optional[float]
    success: Optional[bool]
    native_steps: int
    executions: int
    stop_reason: Optional[str]
    workspace_dir: Optional[str]
    frozen_dir: Optional[str]
    manifest_path: Optional[str]
    run_dir: Optional[str] = None
    task: Optional[str] = None
    seed: Optional[int] = None
    phase: Optional[str] = None
    best_score: Optional[float] = None
    score_history: list = field(default_factory=list)
    error: Optional[str] = None
    extra: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def rank_score(self) -> float:
        value = self.best_score if self.best_score is not None else self.score
        return float(value) if value is not None else -1.0

    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self), default=str))

    @classmethod
    def from_dict(cls, data: dict) -> "EpisodeResult":
        return cls(**{f.name: data[f.name] for f in fields(cls) if f.name in data})


@dataclass
class PipelineSpec:
    """Practice runs in parallel -> review merge -> continuation practice (new seeds, seeded with the merge)
    -> optional further reviews -> final tests.

    continuation_seeds: one seed list per continuation round.
    review_rounds: how many of the 1 + len(continuation_seeds) practice rounds are followed by a review
      (the first ones). None = every round. 0 = no review: knowledge passes on as the best practice run.
    validation_seeds: optional, one seed list per reviewed round: test-phase runs of that round's merge. They
      feed the next review as dev feedback and pick the merge that goes to the final test (best validation
      mean, later round on ties). Each round needs fresh seeds because its runs feed the next merge.
    total_practice_executions: optional equal-budget cap on all practice executions of the pipeline.
    stop_when_perfect: once a round's validation scored 100 on every scene, the later continuation rounds
      (practice, review, validation) are skipped and that merge goes to the final test.
    """
    task: str
    practice_seeds: list
    continuation_seeds: list = field(default_factory=list)
    review_rounds: Optional[int] = None
    test_seeds: list = field(default_factory=list)
    validation_seeds: list = field(default_factory=list)
    practice_budget: Budget = field(default_factory=lambda: Budget(45, 3600))
    test_budget: Budget = field(default_factory=lambda: Budget(100, 5400))
    review_wall_seconds: float = 2400.0
    total_practice_executions: Optional[int] = None
    shared_library: Optional[str] = None
    knowledge_from: Optional[str] = None
    switches: dict = field(default_factory=dict)
    model: Optional[str] = None
    max_parallel: int = 4
    review_sources: int = 6
    label: str = ""
    stop_when_perfect: bool = False

    def __post_init__(self):
        self.practice_seeds = [int(s) for s in self.practice_seeds]
        self.continuation_seeds = [[int(s) for s in r] for r in self.continuation_seeds]
        self.test_seeds = [int(s) for s in self.test_seeds]
        self.validation_seeds = [[int(s) for s in r] for r in self.validation_seeds]
        self.practice_budget = Budget.of(self.practice_budget)
        self.test_budget = Budget.of(self.test_budget)
        rounds = 1 + len(self.continuation_seeds)
        if self.review_rounds is None:
            self.review_rounds = rounds
        if not self.practice_seeds or any(not r for r in self.continuation_seeds):
            raise ValueError("every practice round needs at least one seed")
        if not 0 <= int(self.review_rounds) <= rounds:
            raise ValueError(f"review_rounds must be within 0..{rounds}")
        practice = set(self.practice_seeds).union(*map(set, self.continuation_seeds))
        validation = [s for r in self.validation_seeds for s in r]
        if set(self.test_seeds) & (practice | set(validation)):
            raise ValueError("test seeds must not be used for practice or validation")
        if set(validation) & practice:
            raise ValueError("validation seeds must not be used for practice")
        if self.validation_seeds and (len(self.validation_seeds) != self.review_rounds
                                      or any(not r for r in self.validation_seeds)
                                      or len(set(validation)) != len(validation)):
            raise ValueError("validation_seeds needs one non-empty list of fresh seeds per reviewed round")
        if self.max_parallel < 1 or self.review_sources < 1:
            raise ValueError("max_parallel and review_sources must be >= 1")

    @property
    def rounds(self) -> list[list[int]]:
        return [self.practice_seeds] + self.continuation_seeds

    def planned_practice_executions(self) -> int:
        planned = sum(len(r) for r in self.rounds) * self.practice_budget.executions
        if self.total_practice_executions is not None:
            planned = min(planned, int(self.total_practice_executions))
        return planned

    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self), default=str))

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineSpec":
        return cls(**{f.name: data[f.name] for f in fields(cls) if f.name in data})


def spec_from_switches(task: str, switches: dict, *, explore_seeds: list, validate_seeds: list = (),
                       test_seeds: list = (), continuation_rounds: int = 1, library_root=None,
                       **overrides) -> PipelineSpec:
    """Map the resolved `evolution.*` switches (knowin_agent.experiments.switches) onto a PipelineSpec.

    practice_scenes seeds per round are taken in order from explore_seeds; practice_budget is the total
    practice executions (split evenly over all practice runs and enforced as a cap); validation splits
    validate_seeds evenly over the reviewed rounds; shared_library 'off' -> none, 'latest' -> latest published.
    """
    from knowin_agent.workspace.knowledge import latest_published

    per_round = int(switches.get("evolution.practice_scenes", 3))
    rounds = 1 + int(continuation_rounds)
    explore = [int(s) for s in explore_seeds]
    if len(explore) < per_round * rounds:
        raise ValueError(f"{task}: {rounds} rounds x {per_round} practice scenes need {per_round * rounds} "
                         f"explore seeds, got {len(explore)}")
    chunks = [explore[i * per_round:(i + 1) * per_round] for i in range(rounds)]
    review_rounds = rounds if switches.get("evolution.review_merge", True) else 0
    validation = []
    if switches.get("evolution.validation", True) and review_rounds and validate_seeds:
        each = len(validate_seeds) // review_rounds
        if each < 1:
            raise ValueError(f"{task}: validation needs at least one validate seed per reviewed round")
        validation = [list(validate_seeds[i * each:(i + 1) * each]) for i in range(review_rounds)]
    library = switches.get("evolution.shared_library", "off")
    library = None if library in (None, "off") else (latest_published(library_root) if library == "latest" else library)
    total = int(switches.get("evolution.practice_budget", 80))
    runs = per_round * rounds
    fields_ = dict(task=task, practice_seeds=chunks[0], continuation_seeds=chunks[1:], review_rounds=review_rounds,
                   test_seeds=list(test_seeds), validation_seeds=validation,
                   practice_budget=Budget(max(1, total // runs)), total_practice_executions=total,
                   shared_library=library, knowledge_from=switches.get("evolution.seed_knowledge"),
                   switches=dict(switches), model=switches.get("agent.model"),
                   stop_when_perfect=bool(switches.get("evolution.stop_when_perfect", True)))
    fields_.update(overrides)
    return PipelineSpec(**fields_)


def equal_budget_specs(total_practice_executions: int, variants: dict[str, dict[str, Any]],
                       **common) -> dict[str, PipelineSpec]:
    """Build ablation tiers that share one total practice-execution budget.

    Each variant gives its own rounds (practice_seeds / continuation_seeds / review_rounds ...); the per-run
    budget is the total divided by the number of practice runs of that variant (floor), and the total is also
    enforced as a cap, so every tier spends at most the same number of practice executions.
    """
    out = {}
    for name, overrides in variants.items():
        merged = {**common, **overrides}
        runs = len(merged["practice_seeds"]) + sum(len(r) for r in merged.get("continuation_seeds", []))
        base = Budget.of(merged.get("practice_budget", Budget(1)))
        merged["practice_budget"] = Budget(max(1, total_practice_executions // runs), base.wall_seconds)
        merged["total_practice_executions"] = total_practice_executions
        out[name] = PipelineSpec(**merged)
    return out
