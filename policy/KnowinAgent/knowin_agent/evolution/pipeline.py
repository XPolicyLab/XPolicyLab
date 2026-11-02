"""Self-improvement pipeline: parallel practice -> review merge -> continuation practice -> ... -> tests.

root/
  pipeline.json            spec, plan, per-step records, accounting, final result (resumable)
  steps/<step-id>.json     one record per finished step
  runs/<step>-seed<s>/     one EpisodeRunner run directory per episode (+ episode_result.json)
  reviews/r<k>/            one Reviewer out_dir per review (frozen/ is the merge)

Resume: a finished step is replayed from its record, never re-run. Inside an interrupted step, episodes with
an episode_result.json are reused (except runner crashes, which run again); everything else is moved aside
as `<dir>.voided-<n>` and run again.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from knowin_agent.evolution.review import Reviewer, void_dir
from knowin_agent.evolution.runner import EpisodeRunner
from knowin_agent.evolution.types import Budget, EpisodeResult, EpisodeSpec, PipelineSpec
from knowin_agent.workspace.files import read_json, write_json
from knowin_agent.workspace.freeze import verify

SCHEMA = "knowin_agent-pipeline/1"
RUNNER_ERROR = "runner_error"


class PipelineError(RuntimeError):
    pass


def plan_steps(spec: PipelineSpec) -> list[dict]:
    steps = []
    for k, seeds in enumerate(spec.rounds):
        steps.append({"id": f"practice-r{k}", "kind": "practice", "round": k, "seeds": list(seeds)})
        if k < spec.review_rounds:
            steps.append({"id": f"review-r{k}", "kind": "review", "round": k})
            if spec.validation_seeds:
                steps.append({"id": f"validation-r{k}", "kind": "validation", "round": k,
                              "seeds": list(spec.validation_seeds[k])})
    if spec.test_seeds:
        steps.append({"id": "test", "kind": "test", "seeds": list(spec.test_seeds)})
    return steps


def ranked(results: list[EpisodeResult]) -> list[EpisodeResult]:
    """Runs that ended normally and froze, best Score first."""
    usable = [r for r in results if r.ok and r.frozen_dir]
    return sorted(usable, key=lambda r: -r.rank_score)


def mean(values: list) -> Optional[float]:
    known = [float(v) for v in values if v is not None]
    return round(sum(known) / len(known), 2) if known else None


@dataclass
class Context:
    knowledge: Optional[str] = None
    merges: list = field(default_factory=list)
    by_round: dict = field(default_factory=dict)
    feedback: list = field(default_factory=list)
    validated: list = field(default_factory=list)


class Pipeline:
    def __init__(self, spec: PipelineSpec, *, runner: EpisodeRunner, reviewer: Optional[Reviewer],
                 root: Union[str, Path]):
        self.spec = spec
        self.runner = runner
        self.reviewer = reviewer
        self.root = Path(root)
        self.plan = plan_steps(spec)
        if spec.review_rounds and reviewer is None:
            raise ValueError("this pipeline has review rounds; pass a Reviewer")
        self.ctx = Context(knowledge=spec.knowledge_from)
        self.state: dict = {}
        self._lock = threading.Lock()

    @classmethod
    def resume(cls, root: Union[str, Path], *, runner: EpisodeRunner, reviewer: Optional[Reviewer]) -> "Pipeline":
        state = read_json(Path(root) / "pipeline.json")
        if not state:
            raise FileNotFoundError(f"no pipeline.json in {root}")
        return cls(PipelineSpec.from_dict(state["spec"]), runner=runner, reviewer=reviewer, root=root)

    # ---------------- state ----------------
    def _load_or_init(self) -> dict:
        path = self.root / "pipeline.json"
        state = read_json(path)
        if state:
            if PipelineSpec.from_dict(state.get("spec") or {}).to_dict() != self.spec.to_dict():
                raise ValueError(f"{path} belongs to a different pipeline spec; use a new root")
            return state
        self.root.mkdir(parents=True, exist_ok=True)
        return {"schema": SCHEMA, "spec": self.spec.to_dict(), "plan": self.plan, "status": "created",
                "created_at": time.time(), "steps": {}}

    def _save(self) -> None:
        with self._lock:
            self.state["accounting"] = self.accounting()
            self.state["updated_at"] = time.time()
            write_json(self.root / "pipeline.json", self.state)

    # ---------------- run ----------------
    def run(self) -> dict:
        self.state = self._load_or_init()
        self.ctx = Context(knowledge=self.spec.knowledge_from)
        self.state.update(status="running", error=None)
        self._save()
        handlers = {"practice": self._practice, "review": self._review, "validation": self._validation,
                    "test": self._test}
        for step in self.plan:
            record = self.state["steps"].get(step["id"])
            if record and record.get("status") in ("done", "skipped"):
                self._apply(step, record)
                continue
            perfect = self._perfect_before(step)
            if perfect is not None:
                record = {**step, "status": "skipped", "source": None, "finished_at": time.time(),
                          "skipped": f"validation of round {perfect} already scored 100 on every scene"}
                self.state["steps"][step["id"]] = record
                self._apply(step, record)
                write_json(self.root / "steps" / f"{step['id']}.json", record)
                self._save()
                continue
            record = {**step, "status": "running", "started_at": time.time()}
            self.state["steps"][step["id"]] = record
            self._save()
            try:
                output = handlers[step["kind"]](step)
            except Exception as exc:
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished_at=time.time())
                self.state.update(status="failed", error=f"{step['id']}: {type(exc).__name__}: {exc}")
                self._save()
                raise
            record.update(output, status="skipped" if output.get("skipped") else "done", finished_at=time.time())
            self._apply(step, record)
            write_json(self.root / "steps" / f"{step['id']}.json", record)
            self._save()
        self.state.update(status="done", final=self.final(), finished_at=time.time())
        self._save()
        return self.state

    def _apply(self, step: dict, record: dict) -> None:
        """Advance the context from a finished step record (also used when resuming)."""
        kind = step["kind"]
        if kind == "practice":
            results = [EpisodeResult.from_dict(r) for r in record.get("runs") or []]
            self.ctx.by_round[step["round"]] = results
            if step["round"] >= self.spec.review_rounds and ranked(results):
                self.ctx.knowledge = ranked(results)[0].frozen_dir
        elif kind == "review" and record.get("frozen_dir"):
            self.ctx.knowledge = record["frozen_dir"]
            self.ctx.merges.append({"round": step["round"], "frozen_dir": record["frozen_dir"]})
            self.ctx.feedback = []
        elif kind == "validation":
            self.ctx.feedback = [EpisodeResult.from_dict(r) for r in record.get("runs") or []]
            scores = record.get("scores") or []
            self.ctx.validated.append({"round": step["round"], "frozen_dir": record.get("source"),
                                       "mean": record.get("mean"),
                                       "perfect": bool(scores) and all(s is not None and s >= 100 for s in scores)})

    def _perfect_before(self, step: dict) -> Optional[int]:
        """The round whose validation was perfect, when stop_when_perfect makes this later round step moot."""
        if not self.spec.stop_when_perfect or step["kind"] == "test":
            return None
        return next((v["round"] for v in self.ctx.validated if v["perfect"] and v["round"] < step["round"]), None)

    # ---------------- episodes ----------------
    def _episode(self, spec: EpisodeSpec, run_dir: Path) -> EpisodeResult:
        done = run_dir / "episode_result.json"
        previous = read_json(done)
        if previous and previous.get("stop_reason") != RUNNER_ERROR:
            return EpisodeResult.from_dict(previous)
        void_dir(run_dir)
        try:
            result = self.runner.run(spec, run_dir)
        except Exception as exc:  # noqa: BLE001 - one failed episode must not stop its siblings
            result = EpisodeResult(score=None, success=None, native_steps=0, executions=0, stop_reason=RUNNER_ERROR,
                                   workspace_dir=None, frozen_dir=None, manifest_path=None, run_dir=str(run_dir),
                                   task=spec.task, seed=spec.seed, phase=spec.phase,
                                   error=f"{type(exc).__name__}: {exc}")
        result.extra = {**(result.extra or {}), "knowledge_from": spec.knowledge_from,
                        "budget": spec.budget.executions}
        write_json(done, result.to_dict())
        return result

    def _run_many(self, specs: list[EpisodeSpec], dirs: list[Path]) -> list[EpisodeResult]:
        workers = max(1, min(self.spec.max_parallel, len(specs)))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="knowin_agent-episode") as pool:
            return list(pool.map(self._episode, specs, dirs))

    def _episode_spec(self, seed: int, phase: str, knowledge_from, shared, budget: Budget, label: str) -> EpisodeSpec:
        return EpisodeSpec(task=self.spec.task, seed=seed, phase=phase, knowledge_from=knowledge_from,
                           shared_library=shared, budget=budget, switches=dict(self.spec.switches),
                           model=self.spec.model, label=label)

    # ---------------- steps ----------------
    def _practice_used(self) -> int:
        return sum(r.executions for rs in self.ctx.by_round.values() for r in rs)

    def _practice(self, step: dict) -> dict:
        k, seeds = step["round"], step["seeds"]
        per = self.spec.practice_budget.executions
        if self.spec.total_practice_executions is not None:
            remaining = int(self.spec.total_practice_executions) - self._practice_used()
            per = min(per, max(0, remaining) // len(seeds))
        if per < 1:
            return {"runs": [], "budget_per_run": 0, "skipped": "practice execution budget exhausted"}
        if k == 0:
            knowledge_from, shared = self.spec.knowledge_from, self.spec.shared_library
        else:
            if not self.ctx.knowledge:
                raise PipelineError(f"round {k}: no knowledge to continue from")
            knowledge_from, shared = self.ctx.knowledge, None
        budget = Budget(per, self.spec.practice_budget.wall_seconds)
        specs = [self._episode_spec(s, "practice", knowledge_from, shared, budget, step["id"]) for s in seeds]
        results = self._run_many(specs, [self.root / "runs" / f"{step['id']}-seed{s}" for s in seeds])
        record = {"runs": [r.to_dict() for r in results], "budget_per_run": per, "knowledge_from": knowledge_from}
        if not ranked(results):
            self.state["steps"][step["id"]].update(record)
            raise PipelineError(f"no practice run of round {k} ended normally and froze: "
                                + "; ".join(f"seed {r.seed}: {r.error}" for r in results))
        return record

    def _review(self, step: dict) -> dict:
        k = step["round"]
        new = ranked(self.ctx.by_round.get(k) or [])
        if not new:
            return {"skipped": f"practice round {k} produced no runs to review"}
        earlier = ranked([r for j, rs in self.ctx.by_round.items() if j < k for r in rs])
        sources = (new + earlier)[:self.spec.review_sources]
        previous = self.ctx.merges[-1]["frozen_dir"] if self.ctx.merges else None
        out_dir = self.root / "reviews" / f"r{k}"
        void_dir(out_dir)
        frozen = self.reviewer.review(self.spec.task, sources, Path(previous) if previous else None, out_dir,
                                      feedback=list(self.ctx.feedback), round=k,
                                      test_budget=self.spec.test_budget.executions,
                                      wall_seconds=self.spec.review_wall_seconds)
        return {"frozen_dir": str(frozen), "out_dir": str(out_dir), "sources": [r.run_dir for r in sources],
                "previous": previous, "feedback": [r.run_dir for r in self.ctx.feedback]}

    def _tests(self, step: dict, source: str) -> dict:
        before = verify(source)["digest"]
        specs = [self._episode_spec(s, "test", source, None, self.spec.test_budget, step["id"]) for s in step["seeds"]]
        results = self._run_many(specs, [self.root / "runs" / f"{step['id']}-seed{s}" for s in step["seeds"]])
        after = verify(source)["digest"]
        if before != after:
            raise PipelineError(f"frozen knowledge changed during {step['id']}")
        scores = [r.score if r.ok else None for r in results]
        return {"source": source, "digest": after, "runs": [r.to_dict() for r in results], "scores": scores,
                "mean": mean(scores)}

    def _validation(self, step: dict) -> dict:
        merge = next((m for m in reversed(self.ctx.merges) if m["round"] == step["round"]), None)
        if merge is None:
            return {"skipped": f"round {step['round']} has no merge to validate", "source": None}
        return self._tests(step, merge["frozen_dir"])

    def test_source(self) -> Optional[str]:
        scored = [v for v in self.ctx.validated if v.get("mean") is not None]
        if scored:
            return max(scored, key=lambda v: (v["mean"], v["round"]))["frozen_dir"]
        return self.ctx.knowledge

    def _test(self, step: dict) -> dict:
        source = self.test_source()
        if not source:
            raise PipelineError("no knowledge to test")
        record = self._tests(step, source)
        if record["mean"] is None:
            self.state["steps"][step["id"]].update(record)
            raise PipelineError("no test produced a score")
        return record

    # ---------------- reporting ----------------
    def accounting(self) -> dict:
        """Equal-budget accounting: what each kind of step spent, comparable across ablation tiers."""
        steps = self.state.get("steps", {}).values()

        def runs(kind):
            return [r for s in steps if s.get("kind") == kind for r in s.get("runs") or []]

        practice = runs("practice")
        return {
            "practice_runs": len(practice),
            "practice_executions": sum(r.get("executions") or 0 for r in practice),
            "practice_native_steps": sum(r.get("native_steps") or 0 for r in practice),
            "practice_executions_planned": self.spec.planned_practice_executions(),
            "practice_execution_cap": self.spec.total_practice_executions,
            "review_sessions": sum(1 for s in steps if s.get("kind") == "review" and s.get("status") == "done"),
            "validation_runs": len(runs("validation")),
            "validation_executions": sum(r.get("executions") or 0 for r in runs("validation")),
            "test_runs": len(runs("test")),
            "test_executions": sum(r.get("executions") or 0 for r in runs("test")),
        }

    def final(self) -> dict:
        test = self.state["steps"].get("test") or {}
        return {"knowledge": test.get("source") or self.ctx.knowledge, "test_scores": test.get("scores"),
                "test_mean": test.get("mean"), "merges": self.ctx.merges, "validation": self.ctx.validated,
                "practice_executions": self.accounting()["practice_executions"]}
