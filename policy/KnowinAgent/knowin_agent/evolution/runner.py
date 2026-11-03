"""EpisodeRunner protocol, the workspace bookkeeping every runner shares, and a FakeRunner for tests.

A run directory holds:
  workspace/ + workspace.json   the agent's workspace and its host receipt
  tool_calls.jsonl ...           run records (knowin_agent.core.recorder)
  frozen/                        practice only: frozen skills + lessons + manifest.json
  result.json                    the EpisodeResult written by finish_episode
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional, Protocol, runtime_checkable

from knowin_agent.core.recorder import RunRecorder
from knowin_agent.evolution.types import EpisodeResult, EpisodeSpec
from knowin_agent.workspace.files import tree_hashes, write_json
from knowin_agent.workspace.freeze import FrozenChanged, freeze, verify
from knowin_agent.workspace.layout import Workspace, create_workspace


@runtime_checkable
class EpisodeRunner(Protocol):
    def run(self, spec: EpisodeSpec, run_dir: Path) -> EpisodeResult: ...


def default_instruction(spec: EpisodeSpec) -> str:
    if spec.phase == "practice":
        rules = ("Practice: you may reset this scene. Keep what you learn in skills/ (includable .py + .md) and "
                 "lessons/ (PLAN.md first); they are frozen at the end and are all the next agent gets.")
    else:
        rules = "Test: skills/ and lessons/ are read-only; write temporary notes only in scratch/."
    return f"# {spec.task} (seed {spec.seed}, {spec.phase})\n\n{rules}\n"


def prepare_episode(spec: EpisodeSpec, run_dir: Path, *, instruction: Optional[str] = None,
                    library_root: Optional[Path] = None) -> Workspace:
    """Verify the seed knowledge and create the workspace for one episode."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if spec.knowledge_from:
        verify(spec.knowledge_from)
    write_json(run_dir / "episode_spec.json", spec.to_dict())
    return create_workspace(run_dir / "workspace", instruction=instruction or default_instruction(spec),
                            phase=spec.phase, knowledge_from=spec.knowledge_from,
                            shared_library=spec.shared_library, library_root=library_root)


def finish_episode(spec: EpisodeSpec, workspace: Workspace, run_dir: Path, *, score: Optional[float],
                   success: Optional[bool], native_steps: int, executions: int, stop_reason: Optional[str],
                   best_score: Optional[float] = None, score_history: Optional[list] = None,
                   extra: Optional[dict] = None) -> EpisodeResult:
    """Freeze practice knowledge, or re-verify test knowledge, and write result.json."""
    run_dir = Path(run_dir)
    frozen_dir = manifest_path = None
    error = None
    if spec.phase == "practice":
        manifest_path = freeze(workspace, run_dir / "frozen", source_run=run_dir.name,
                               extra={"task": spec.task, "seed": spec.seed, "best_score": best_score,
                                      "knowledge_from": spec.knowledge_from})
        frozen_dir = str(run_dir / "frozen")
        manifest_path = str(manifest_path)
    else:
        if spec.knowledge_from:
            try:
                verify(spec.knowledge_from)
            except FrozenChanged as exc:
                error = str(exc)
        seeded = (workspace.receipt.get("seed_knowledge") or {}).get("files") or {}
        now = {**tree_hashes(workspace.skills, prefix="skills/", exclude=("shared",)),
               **tree_hashes(workspace.lessons, prefix="lessons/")}
        if now != seeded:
            error = error or "test workspace skills/ or lessons/ changed during the episode"
        if workspace.receipt.get("shared_library") and not workspace.shared_unchanged():
            error = error or "skills/shared/ changed during the episode"
    result = EpisodeResult(score=score, success=success, native_steps=int(native_steps), executions=int(executions),
                           stop_reason=stop_reason, workspace_dir=str(workspace.root), frozen_dir=frozen_dir,
                           manifest_path=manifest_path, run_dir=str(run_dir), task=spec.task, seed=spec.seed,
                           phase=spec.phase, best_score=best_score if best_score is not None else score,
                           score_history=list(score_history or []), error=error, extra=dict(extra or {}))
    write_json(run_dir / "result.json", result.to_dict())
    return result


ScoreFn = Callable[[EpisodeSpec, int, Workspace], float]


def default_score(spec: EpisodeSpec, execution: int, workspace: Workspace) -> float:
    """Deterministic toy dynamics: seeded knowledge starts higher; merged knowledge solves tests."""
    plan = workspace.lessons / "PLAN.md"
    merged = plan.is_file() and "merged" in plan.read_text(encoding="utf-8")
    base = 10.0 + 10.0 * (spec.seed % 3) + (30.0 if spec.knowledge_from else 0.0)
    if spec.phase == "test":
        return 100.0 if merged else min(90.0, base + 20.0)
    return min(100.0, base + 5.0 * (execution // 3))


class FakeRunner:
    """Simulates an episode without a robot: exec rows with score feedback, plus repeated tool-mode calls so
    trajectory mining finds candidates; practice writes a skill and lessons, then freezes."""

    def __init__(self, score_fn: ScoreFn = default_score, *, executions: Optional[int] = None,
                 delay: float = 0.0, fail_seeds: tuple = ()):
        self.score_fn = score_fn
        self.executions = executions
        self.delay = delay
        self.fail_seeds = set(fail_seeds)
        self.calls: list[EpisodeSpec] = []

    def run(self, spec: EpisodeSpec, run_dir: Path) -> EpisodeResult:
        self.calls.append(spec)
        if spec.seed in self.fail_seeds:
            raise RuntimeError(f"simulator failed to start for seed {spec.seed}")
        ws = prepare_episode(spec, run_dir)
        recorder = RunRecorder(Path(run_dir), save_images=False)
        limit = spec.budget.executions if self.executions is None else min(self.executions, spec.budget.executions)
        includes = sorted(f"skills/{p.name}" for p in ws.skills.glob("*.py"))
        includes += sorted(f"skills/shared/{p.name}" for p in ws.shared.glob("*.py"))[:1]
        history, best, score, native, n = [], None, None, 0, 0
        for n in range(1, limit + 1):
            if self.delay:
                time.sleep(self.delay)
            for decision, (tool, args) in enumerate((("move_to", {"arm": "left", "target": [0.1 * n, 0, 0.2]}),
                                                     ("gripper", {"arm": "left", "command": "close"}))):
                recorder.record("tool_calls", scope="task", delegation=None, decision=n * 10 + decision, tool=tool,
                                arguments=args, rationale="fake", ok=True, status="reached", result={})
            score = float(self.score_fn(spec, n, ws))
            stop = "checkpoint_failed" if n % 4 == 0 and score < 100 else None
            recorder.record("tool_calls", scope="task", delegation=None, decision=n * 10 + 5, tool="exec",
                            arguments={"code": f"# attempt {n}\nmove_to('left', [0.3, 0.0, 0.2])\n"
                                               f"checkpoint('lifted', True)\n", "include": includes},
                            rationale="fake", ok=stop is None, status="done",
                            result={"ok": stop is None, "stop_reason": stop, "native_steps": 7,
                                    "score_feedback": {"score": score} if spec.phase == "practice" else None})
            native += 7
            if best is None or score > best:
                best = score
                history.append({"execution": n, "score": score})
            if score >= 100:
                break
        if spec.phase == "practice":
            ws.write_text(f"skills/approach_{spec.task}.py".replace("-", "_"),
                          '"""Approach helper."""\n\n\ndef approach_height(top_z, clearance=0.08):\n'
                          '    return float(top_z) + float(clearance)\n')
            ws.write_text(f"skills/approach_{spec.task}.md".replace("-", "_"),
                          f"approach_height: hover above a measured top. Evidence: seed {spec.seed}, Score {best}.\n")
            ws.write_text("lessons/PLAN.md", f"# PLAN\n\n| # | What | Verification | Recovery |\n|---|------|--------------|----------|\n"
                                             f"| 1 | approach; approach_height; ~7 | lifted | re-measure top |\n\n"
                                             f"seed {spec.seed}\n")
            ws.write_text("lessons/EVIDENCE.md", f"seed {spec.seed}: best {best} at execution {history[-1]['execution'] if history else None}\n")
            ws.write_text("lessons/shared_proposals.md", f"- geometry.py: seed {spec.seed} proposes a tolerance option\n")
        else:
            ws.write_text("scratch/notes.md", f"test seed {spec.seed}: score {score}\n")
        return finish_episode(spec, ws, run_dir, score=score, success=bool(score is not None and score >= 100),
                              native_steps=native, executions=n, stop_reason="success" if (score or 0) >= 100
                              else "budget_exhausted", best_score=best, score_history=history)
