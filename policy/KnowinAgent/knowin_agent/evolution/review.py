"""Review merge: a simulator-free coding-agent session merges practice knowledge into one frozen dir.

out_dir/
  workspace/            the agent's working directory (everything read-only except out/)
    instruction.md
    practice/run<i>_seed<s>/{skills,lessons,summary.json,timeline.md,executions/,views/}
    practice/mined_candidates.json
    dev/seed<s>[-cut_off]/{summary.json,timeline.md,executions/,views/,scratch_notes/}   tests of the previous merge
    previous/{skills,lessons,REPORT.md}                       the previous merge
    reference/{scoring.md,task_source.md,recipe.md,api.md,...}
    out/{skills,lessons,REPORT.md}                            written by the agent

views/ holds the views after the key executions; with a raw mount (the console binds the runs directory
read-only into the review container) the timeline points at every execution's views there instead.
  frozen/               the merge as frozen knowledge (skills + lessons + the sources' skills/shared copy)
  REPORT.md             copied from out/REPORT.md
  review.json           state: status, launcher return code, timing, warnings, errors
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence, Union

from knowin_agent.evolution import instructions as text
from knowin_agent.evolution import traces
from knowin_agent.evolution.snippet import check_snippet
from knowin_agent.evolution.types import EpisodeResult
from knowin_agent.workspace.files import copy_tree, read_json, regular_files, set_read_only, tree_hashes, write_json
from knowin_agent.workspace.freeze import freeze_dirs, load_manifest, verify
from knowin_agent.workspace.knowledge import SHARED_REL

OUT_MAX_FILES, OUT_MAX_BYTES = 60, 2_000_000
SKILL_SUFFIXES = (".py", ".md")
LESSON_SUFFIXES = (".md", ".txt", ".json")

Launcher = Callable[[str, Path], object]


class ReviewError(RuntimeError):
    pass


class FinishedProcess:
    """What a synchronous launcher returns: the session already ran."""

    def __init__(self, returncode: int = 0):
        self.returncode = returncode

    def wait(self, timeout: Optional[float] = None) -> int:
        return self.returncode

    def kill(self) -> None:
        pass


def command_launcher(argv: Optional[Sequence[str]] = None, *, log_name: str = "agent.log") -> Launcher:
    """Launcher running `argv + [prompt]` in the workdir (stdout/stderr to workdir/../agent.log).

    argv defaults to $KNOWIN_AGENT_REVIEW_AGENT_CMD (shell-split), e.g. "codex exec --skip-git-repo-check".
    """
    command = list(argv) if argv is not None else shlex.split(os.environ.get("KNOWIN_AGENT_REVIEW_AGENT_CMD", ""))
    if not command:
        raise ValueError("no review agent command: pass argv or set KNOWIN_AGENT_REVIEW_AGENT_CMD")

    def launch(prompt: str, workdir: Path):
        log = open(Path(workdir).parent / log_name, "ab")
        try:
            return subprocess.Popen(command + [prompt], cwd=workdir, stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT)
        finally:
            log.close()

    return launch


def wait_session(proc, wall_seconds: float) -> tuple[Optional[int], bool]:
    try:
        return proc.wait(timeout=wall_seconds), False
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            return proc.wait(timeout=30), True
        except subprocess.TimeoutExpired:
            return None, True


def scoring_reference(task: str) -> str:
    base = resources.files("knowin_agent.assets").joinpath("robodojo", "scoring")
    name = task[:-len("_random")] if task.endswith("_random") else task
    parts = [base.joinpath(n).read_text(encoding="utf-8") for n in ("_common.md", f"{name}.md")
             if base.joinpath(n).is_file()]
    return "\n\n".join(parts) or "No scoring note for this task.\n"


def task_material(task: str, kind: str) -> Optional[str]:
    """The official task source (kind 'code') or the supplied recipe ('recipe') the task agent also gets."""
    name = task[:-len("_random")] if task.endswith("_random") else task
    folder = ("reference", "code") if kind == "code" else ("recipe",)
    path = resources.files("knowin_agent.assets").joinpath("robodojo", *folder, f"{name}.md")
    return path.read_text(encoding="utf-8") if path.is_file() else None


def api_reference() -> str:
    """The program layer's own reference if it publishes one, else the summary shipped here."""
    try:
        from knowin_agent import program  # noqa: PLC0415 - optional, developed separately
    except ImportError:
        return text.API_REFERENCE
    provider = getattr(program, "API_REFERENCE", None) or getattr(program, "api_reference", None)
    if callable(provider):
        provider = provider()
    return provider if isinstance(provider, str) and provider.strip() else text.API_REFERENCE


@dataclass
class RunInput:
    run_dir: Path
    frozen_dir: Optional[Path]
    seed: Optional[int]
    result: dict

    @classmethod
    def of(cls, item: Union[EpisodeResult, str, Path]) -> "RunInput":
        if isinstance(item, EpisodeResult):
            data = item.to_dict()
        else:
            run = Path(item)
            data = read_json(run / "episode_result.json") or read_json(run / "result.json") or {"run_dir": str(run)}
            data["run_dir"] = str(run)
            hosted = data.get("evaluation_host_only") or {}
            for key in ("seed", "score", "success", "native_steps"):
                if data.get(key) is None and hosted.get(key) is not None:
                    data[key] = hosted[key]
            if data.get("phase") is None:
                data["phase"] = (read_json(run / "config.json") or {}).get("phase")
        run_dir = Path(data["run_dir"])
        frozen = data.get("frozen_dir") or (str(run_dir / "frozen") if (run_dir / "frozen").is_dir() else None)
        return cls(run_dir, Path(frozen) if frozen else None, data.get("seed"), data)


def seeded_from(run: "RunInput", manifest: Optional[Mapping] = None) -> Optional[str]:
    """The merged knowledge a practice run started from: hosted runs record it in protocol.json
    (knowledge_from) and workspace.json (seed_knowledge.source)."""
    run_dir = Path(run.run_dir)
    candidates = [(read_json(run_dir / "protocol.json") or {}).get("knowledge_from"),
                  ((read_json(run_dir / "workspace.json") or {}).get("seed_knowledge") or {}).get("source"),
                  (run.result.get("extra") or {}).get("knowledge_from"), (manifest or {}).get("knowledge_from")]
    return next((str(c) for c in candidates if c), None)


def same_knowledge(a: Optional[Union[str, Path]], b: Optional[Union[str, Path]]) -> bool:
    """Whether two frozen-knowledge paths are the same merge (<run>/frozen), wherever the runs dir is mounted."""
    if not a or not b:
        return False
    a, b = Path(a), Path(b)
    if a == b:
        return True
    try:
        if a.resolve() == b.resolve():
            return True
    except OSError:
        pass
    key = lambda p: (p.parent.name, p.name) if p.name == "frozen" else (p.name, "")  # noqa: E731
    return key(a) == key(b)


def knowledge_name(path: Optional[Union[str, Path]]) -> Optional[str]:
    if not path:
        return None
    path = Path(path)
    return path.parent.name if path.name == "frozen" else path.name


def traces_has_files(folder: Path) -> bool:
    return Path(folder).is_dir() and bool(regular_files(folder))


CHANGES_HEADING = re.compile(r"^#{1,4}\s*Changes\b", re.I | re.M)


def change_table_problem(report: str) -> Optional[str]:
    """REPORT.md must carry a `## Changes` table (one row per kept/rewritten/deleted/added PLAN row or file, with
    the run, execution and what was seen there), so every edit of the knowledge can be audited."""
    match = CHANGES_HEADING.search(report)
    if not match:
        return "REPORT.md lacks the `## Changes` table"
    rows = [line for line in report[match.end():].splitlines() if line.strip().startswith("|")]
    body = [r for r in rows if not set(r.replace("|", "").strip()) <= set("-: ")]
    if len(body) < 2:
        return "the `## Changes` table in REPORT.md has no rows"
    return None


def inputs_hash(workspace: Path) -> dict[str, str]:
    return tree_hashes(workspace, exclude=("out",))


def protect_inputs(workspace: Path) -> None:
    """Everything but out/ read-only on disk (the agent's sandbox may still allow more; inputs_hash checks)."""
    for child in Path(workspace).iterdir():
        if child.name != "out":
            set_read_only(child)
    path = Path(workspace)
    path.chmod(path.stat().st_mode & ~0o222)


class Reviewer:
    def __init__(self, launcher: Launcher, *, wall_seconds: float = 2400.0, test_budget: int = 100,
                 max_files: int = OUT_MAX_FILES, max_bytes: int = OUT_MAX_BYTES,
                 references: Optional[Mapping[str, str]] = None, scoring: Optional[Callable[[str], str]] = None,
                 mine: bool = True, raw_mount: Optional[tuple[str, str]] = None, oracle: bool = False):
        self.launcher = launcher
        self.wall_seconds = wall_seconds
        self.test_budget = test_budget
        self.max_files, self.max_bytes = max_files, max_bytes
        self.references = dict(references or {})
        self.scoring = scoring or scoring_reference
        self.mine = mine
        # (host prefix, path in the review container) of a read-only mount of the raw runs directory.
        self.raw_mount = raw_mount
        # Oracle input (switch evolution.condition_report): the host's per-predicate record of each run.
        self.oracle = oracle

    def raw_dir(self, run_dir: Path) -> Optional[str]:
        if not self.raw_mount:
            return None
        host, inside = self.raw_mount
        run = str(Path(run_dir))
        host = str(Path(host))
        if run != host and not run.startswith(host.rstrip("/") + "/"):
            return None
        return inside.rstrip("/") + run[len(host):]

    def _evidence(self, run: "RunInput", d: Path, header: list[str]) -> dict:
        summary = traces.run_summary(run.run_dir, run.result)
        written = traces.write_timeline(run.run_dir, summary, d, raw_dir=self.raw_dir(run.run_dir), header=header)
        if self.oracle:
            report = traces.ground_truth_report(run.run_dir)
            if report:
                (d / "ground_truth.md").write_text(report, encoding="utf-8")
        summary["key_executions"] = written["key_executions"]
        return summary

    # ---------------- inputs ----------------
    def build_inputs(self, task: str, practice_runs: Iterable, previous: Optional[Path], workspace: Path, *,
                     feedback: Iterable = (), test_budget: Optional[int] = None) -> Path:
        ws = Path(workspace)
        if ws.exists():
            raise ValueError(f"review workspace already exists: {ws}")
        runs = [RunInput.of(r) for r in practice_runs]
        if not runs:
            raise ReviewError("a review needs at least one practice run")
        (ws / "practice").mkdir(parents=True)
        previous_name = knowledge_name(previous)
        for i, run in enumerate(runs, 1):
            d = ws / "practice" / f"run{i}_seed{run.seed}"
            if run.frozen_dir is None:
                raise ReviewError(f"practice run has no frozen knowledge: {run.run_dir}")
            manifest = verify(run.frozen_dir)
            for sub in ("skills", "lessons"):
                copy_tree(run.frozen_dir / sub, d / sub)
            source = seeded_from(run, manifest)
            from_previous = same_knowledge(source, previous)
            header = [f"- Started from: {knowledge_name(source) or 'no merged knowledge (cold start)'}"
                      + (" -- your previous merge (`previous/`)" if from_previous
                         else (f" -- NOT your previous merge ({previous_name})" if previous and source else ""))]
            summary = self._evidence(run, d, header)
            summary.update(started_from=knowledge_name(source), started_from_previous_merge=from_previous)
            write_json(d / "summary.json", summary)
        if self.mine:
            from knowin_agent.skills.consolidation import mine
            candidates = [c.as_dict() for c in mine([r.run_dir for r in runs])]
            write_json(ws / "practice" / "mined_candidates.json", candidates[:20])
        dev = [RunInput.of(r) for r in feedback]
        for run in dev:
            status = traces.run_status(run.run_dir, run.result)
            d = ws / "dev" / (f"seed{run.seed}" + ("" if status["complete"] else "-cut_off"))
            source = seeded_from(run)
            header = [f"- Tested knowledge: {knowledge_name(source) or '?'}"
                      + (" -- your previous merge (`previous/`)" if same_knowledge(source, previous) else "")]
            write_json(d / "summary.json", self._evidence(run, d, header))
            scratch = run.run_dir / "workspace" / "scratch"
            if traces_has_files(scratch):
                copy_tree(scratch, d / "scratch_notes", max_files=20)
        if previous:
            previous = Path(previous)
            verify(previous)
            copy_tree(previous / "skills", ws / "previous" / "skills", exclude=("shared",))
            copy_tree(previous / "lessons", ws / "previous" / "lessons")
            report = previous.parent / "REPORT.md"
            if report.is_file():
                (ws / "previous" / "REPORT.md").write_bytes(report.read_bytes())
        ref = ws / "reference"
        ref.mkdir()
        (ref / "scoring.md").write_text(self.scoring(task), encoding="utf-8")
        (ref / "api.md").write_text(api_reference(), encoding="utf-8")
        extra = []
        for name, kind, what in (("task_source.md", "code", "the official task source: the exact predicates of "
                                  "every tier and of success"), ("recipe.md", "recipe", "the task recipe the "
                                  "task agent also reads (where it disagrees with scoring.md, scoring wins)")):
            body = task_material(task, kind)
            if body:
                (ref / name).write_text(body, encoding="utf-8")
                extra.append(f"- `reference/{name}`: {what}.\n")
        for name, body in self.references.items():
            (ref / name).write_text(body, encoding="utf-8")
            extra.append(f"- `reference/{name}`: supplied reference ({name}).\n")
        (ws / "instruction.md").write_text(text.REVIEW_INSTRUCTION.format(
            task=task, test_budget=test_budget or self.test_budget, how_agents_act=text.HOW_AGENTS_ACT,
            dev_section=text.REVIEW_DEV_SECTION if dev else "",
            previous_section=text.REVIEW_PREVIOUS_SECTION if previous else "",
            oracle_section=text.REVIEW_ORACLE_SECTION if self.oracle else "",
            extra_references="".join(extra), max_files=self.max_files, max_kb=self.max_bytes // 1000),
            encoding="utf-8")
        for sub in ("skills", "lessons"):
            (ws / "out" / sub).mkdir(parents=True)
        return ws

    # ---------------- validation ----------------
    @staticmethod
    def tidy_out(out: Path) -> list[str]:
        """Remove what a merge can never keep (copied camera views, scratch outside skills/, lessons/ and
        REPORT.md, files of a kind those folders do not hold) instead of rejecting the whole merge for it."""
        removed = []
        for path in sorted(regular_files(out)):
            rel = path.relative_to(out)
            keep = (rel.as_posix() == "REPORT.md"
                    or (rel.parts[0] == "skills" and path.suffix in SKILL_SUFFIXES)
                    or (rel.parts[0] == "lessons" and path.suffix in LESSON_SUFFIXES))
            if not keep:
                path.unlink()
                removed.append(f"{rel}: removed (only skills/*.py|.md, lessons/{'|'.join(LESSON_SUFFIXES)} and "
                               "REPORT.md are kept)")
        for folder in sorted((p for p in out.rglob("*") if p.is_dir() and not p.is_symlink()), reverse=True):
            if folder.parts[len(out.parts)] not in ("skills", "lessons") and not any(folder.iterdir()):
                folder.rmdir()
        return removed

    def validate_out(self, out: Path) -> list[str]:
        """Raise ReviewError for an unusable merge; return warnings."""
        out = Path(out)
        if any(p.is_symlink() for p in out.rglob("*")):
            raise ReviewError("review output contains symlinks")
        tidied = self.tidy_out(out)
        files = regular_files(out)
        if len(files) > self.max_files or sum(p.stat().st_size for p in files) > self.max_bytes:
            raise ReviewError(f"review output exceeds {self.max_files} files / {self.max_bytes // 1000} KB")
        if any(regular_files(out / SHARED_REL)) or (out / SHARED_REL).exists():
            raise ReviewError("review output writes out/skills/shared/; the shared library is restored unchanged")
        knowledge = [p for p in files if p.relative_to(out).parts[0] in ("skills", "lessons")]
        if not knowledge:
            raise ReviewError("review wrote no skills/ or lessons/")
        if not (out / "lessons" / "PLAN.md").is_file():
            raise ReviewError("review output lacks out/lessons/PLAN.md")
        if not (out / "REPORT.md").is_file():
            raise ReviewError("review output lacks out/REPORT.md")
        problem = change_table_problem((out / "REPORT.md").read_text(encoding="utf-8", errors="replace"))
        if problem:
            raise ReviewError(problem)
        problems, warnings = [], list(tidied)
        for path in files:
            rel = path.relative_to(out)
            if rel.parts[0] == "skills" and path.suffix not in SKILL_SUFFIXES:
                problems.append(f"{rel}: skills/ holds only .py and .md files")
            elif rel.parts[0] == "lessons" and path.suffix not in LESSON_SUFFIXES:
                problems.append(f"{rel}: lessons/ holds only {', '.join(LESSON_SUFFIXES)} files")
            elif rel.parts[0] not in ("skills", "lessons") and rel.as_posix() != "REPORT.md":
                warnings.append(f"{rel}: outside skills/, lessons/ and REPORT.md; not kept")
            if rel.parts[0] == "skills" and path.suffix == ".py":
                problems += [f"{rel}: {p}" for p in check_snippet(path.read_text(encoding="utf-8"), str(rel))]
        if problems:
            raise ReviewError("review skills are not includable: " + "; ".join(problems[:12]))
        return warnings

    # ---------------- session ----------------
    def review(self, task: str, practice_runs: Iterable, previous: Optional[Path], out_dir: Path, *,
               feedback: Iterable = (), round: Optional[int] = None, test_budget: Optional[int] = None,
               wall_seconds: Optional[float] = None) -> Path:
        out_dir = Path(out_dir)
        if out_dir.exists() and any(out_dir.iterdir()):
            raise ValueError(f"review dir is not empty: {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
        practice_runs, feedback = list(practice_runs), list(feedback)
        state = {"status": "preparing", "task": task, "round": round, "started_at": time.time(),
                 "previous": str(previous) if previous else None}
        write_json(out_dir / "review.json", state)
        try:
            ws = self.build_inputs(task, practice_runs, previous, out_dir / "workspace", feedback=feedback,
                                   test_budget=test_budget)
            runs = [RunInput.of(r) for r in practice_runs]
            state.update(merged_from=[str(r.run_dir) for r in runs],
                         feedback=[str(RunInput.of(r).run_dir) for r in feedback])
            protect_inputs(ws)
            before = inputs_hash(ws)
            state.update(status="running", running_at=time.time())
            write_json(out_dir / "review.json", state)
            proc = self.launcher(text.REVIEW_PROMPT, ws)
            rc, timed_out = wait_session(proc, wall_seconds or self.wall_seconds)
            state.update(rc=rc, timed_out=timed_out, status="checking")
            if inputs_hash(ws) != before:
                raise ReviewError("review session modified its read-only inputs")
            state["warnings"] = self.validate_out(ws / "out")
            frozen = self.to_frozen(ws / "out", out_dir / "frozen", runs, state)
            state.update(status="finished", finished_at=time.time(), frozen_dir=str(frozen))
            write_json(out_dir / "review.json", state)
            return frozen
        except Exception as exc:
            state.update(status="failed", finished_at=time.time(), error=f"{type(exc).__name__}: {exc}")
            write_json(out_dir / "review.json", state)
            raise

    def to_frozen(self, out: Path, frozen_dir: Path, runs: list[RunInput], state: dict) -> Path:
        shared, version, versions = None, None, set()
        for run in runs:
            copy = run.frozen_dir / SHARED_REL
            if copy.is_dir():
                manifest = load_manifest(run.frozen_dir)
                lib_version = (manifest.get("shared_library") or {}).get("version")
                versions.add((lib_version, tuple(sorted(tree_hashes(copy).items()))))
                if shared is None:
                    shared, version = copy, lib_version
        if len(versions) > 1:
            state.setdefault("warnings", []).append("practice runs used different skills/shared copies; "
                                                    f"kept the one of the best-ranked run ({version})")
        freeze_dirs(frozen_dir, skills=out / "skills", lessons=out / "lessons", shared=shared,
                    source_run=f"review:{frozen_dir.parent.name}", library_version=version,
                    extra={"review": {"task": state.get("task"), "round": state.get("round"),
                                      "merged_from": state.get("merged_from"), "feedback": state.get("feedback"),
                                      "previous": state.get("previous")}})
        (frozen_dir.parent / "REPORT.md").write_bytes((out / "REPORT.md").read_bytes())
        return frozen_dir


def void_dir(path: Path) -> Optional[Path]:
    """Move an interrupted attempt aside (kept for inspection) so the step can run again."""
    path = Path(path)
    if not path.exists():
        return None
    n = 1
    while path.with_name(f"{path.name}.voided-{n}").exists():
        n += 1
    target = path.with_name(f"{path.name}.voided-{n}")
    path.rename(target)
    return target


__all__ = ["FinishedProcess", "Reviewer", "ReviewError", "command_launcher", "protect_inputs", "void_dir"]
