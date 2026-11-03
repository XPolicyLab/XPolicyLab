"""Cross-task shared-library curation: draft the next version, validate it offline, publish on request.

out_dir/
  workspace/   current/ (base library), tasks/<task>/{skills,lessons,usage.json}, reference/api.md,
               instruction.md, out/{library,tests,CHANGES.md}   (everything read-only except out/)
  candidate/   library/, tests/, CHANGES.md, candidate.json, validation.json
  curate.json  state

A candidate is published as the next `v<N>` under the library root only when its validation passes
(snippet rules, compiles, offline unit tests run with this interpreter, no task names in library code,
size limits) and its base is still the latest published version. Frozen task knowledge keeps the library
copy it was trained with, so publishing never changes a finished task.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Union

from knowin_agent.evolution import instructions as text
from knowin_agent.evolution import traces
from knowin_agent.evolution.review import Launcher, RunInput, api_reference, inputs_hash, protect_inputs, wait_session
from knowin_agent.evolution.snippet import check_snippet
from knowin_agent.workspace import knowledge
from knowin_agent.workspace.files import copy_tree, read_json, tree_hashes, write_json
from knowin_agent.workspace.freeze import verify


class CurationError(RuntimeError):
    pass


def robodojo_task_names() -> list[str]:
    base = resources.files("knowin_agent.assets").joinpath("robodojo", "scoring")
    return sorted(p.name[:-3] for p in base.iterdir() if p.name.endswith(".md") and not p.name.startswith("_"))


@dataclass
class TaskEvidence:
    """One task's input to curation: its best frozen knowledge and the runs that used it."""
    frozen_dir: Path
    runs: list = field(default_factory=list)

    def __post_init__(self):
        self.frozen_dir = Path(self.frozen_dir)

    @classmethod
    def of(cls, value) -> "TaskEvidence":
        if isinstance(value, TaskEvidence):
            return value
        if isinstance(value, dict):
            return cls(Path(value["frozen_dir"]), list(value.get("runs") or []))
        return cls(Path(value))


def usage_rows(runs: Iterable) -> list[dict]:
    rows = []
    for item in runs:
        run = RunInput.of(item)
        summary = traces.run_summary(run.run_dir, run.result)
        row = {"run": run.run_dir.name, "phase": run.result.get("phase"), "seed": run.seed,
               "executions": summary["executions"], "includes": summary["include_usage"]}
        if run.result.get("phase") == "practice":
            row.update(best_score=summary["best_score"], final_score=summary["final_score"])
        else:
            row.update(score=run.result.get("score"))
        rows.append(row)
    return rows


class Curator:
    def __init__(self, launcher: Launcher, *, library_root: Optional[Union[str, Path]] = None,
                 task_names: Optional[Iterable[str]] = None, python: str = sys.executable,
                 wall_seconds: float = 3600.0, test_timeout: float = 300.0,
                 sandbox: Sequence[str] = ()):
        self.launcher = launcher
        self.root = knowledge.library_root(library_root)
        self.task_names = sorted(task_names) if task_names is not None else robodojo_task_names()
        self.python = python
        self.wall_seconds = wall_seconds
        self.test_timeout = test_timeout
        self.sandbox = list(sandbox)

    # ---------------- inputs ----------------
    def build_inputs(self, base_version: str, task_runs: Mapping[str, object], workspace: Path) -> Path:
        ws = Path(workspace)
        if ws.exists():
            raise ValueError(f"curation workspace already exists: {ws}")
        base = knowledge.library_dir(base_version, self.root)
        knowledge.check_library(base)
        copy_tree(base, ws / "current")
        for task, value in sorted(task_runs.items()):
            evidence = TaskEvidence.of(value)
            manifest = verify(evidence.frozen_dir)
            d = ws / "tasks" / task
            for sub in ("skills", "lessons"):
                copy_tree(evidence.frozen_dir / sub, d / sub)
            write_json(d / "usage.json", {"source": manifest.get("source_run"),
                                          "shared_library": (manifest.get("shared_library") or {}).get("version"),
                                          "rows": usage_rows(evidence.runs)})
        (ws / "reference").mkdir(parents=True)
        (ws / "reference" / "api.md").write_text(api_reference(), encoding="utf-8")
        (ws / "instruction.md").write_text(text.CURATE_INSTRUCTION.format(
            base=base_version, how_agents_act=text.HOW_AGENTS_ACT, max_files=knowledge.LIBRARY_MAX_FILES,
            max_kb=knowledge.LIBRARY_MAX_BYTES // 1000, python=f"{sys.version_info[0]}.{sys.version_info[1]}"),
            encoding="utf-8")
        for sub in ("library", "tests"):
            (ws / "out" / sub).mkdir(parents=True)
        return ws

    # ---------------- validation ----------------
    def run_tests(self, version_dir: Path) -> dict:
        argv = self.sandbox + [self.python, "-m", "unittest", "discover", "-s", "tests"]
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        try:
            r = subprocess.run(argv, cwd=version_dir, capture_output=True, text=True, timeout=self.test_timeout,
                               env=env)
        except subprocess.TimeoutExpired:
            return {"ok": False, "output": "tests timed out"}
        output = r.stdout + r.stderr
        ran = re.search(r"Ran (\d+) tests?", output)
        return {"ok": r.returncode == 0 and bool(ran) and int(ran.group(1)) > 0, "output": output[-3000:]}

    def validate(self, version_dir: Path, *, write: bool = True) -> dict:
        """Check a candidate (or version) directory: library/, tests/, CHANGES.md."""
        version_dir = Path(version_dir)
        library = version_dir / "library"
        checks: dict = {"layout": [], "snippet": {}, "task_names": [], "tests": {"ok": False, "output": ""}}
        try:
            files = knowledge.check_library(library)
        except ValueError as exc:
            checks["layout"].append(str(exc))
            files = sorted(p for p in library.iterdir() if p.is_file()) if library.is_dir() else []
        if any(p.is_symlink() for p in version_dir.rglob("*")):
            checks["layout"].append("candidate contains symlinks")
        if not (library / "README.md").is_file():
            checks["layout"].append("library/README.md is missing")
        if not list((version_dir / "tests").glob("test_*.py")):
            checks["layout"].append("no unit tests tests/test_*.py")
        if not (version_dir / "CHANGES.md").is_file():
            checks["layout"].append("CHANGES.md is missing")
        sources = {p.name: p.read_text(encoding="utf-8", errors="replace") for p in files if p.suffix == ".py"}
        for name, source in sources.items():
            problems = check_snippet(source, name)
            checks["snippet"][name] = "ok" if not problems else problems
        joined = "\n".join(sources.values())
        checks["task_names"] = [t for t in self.task_names if re.search(rf"\b{re.escape(t)}\b", joined)]
        if not checks["layout"]:
            checks["tests"] = self.run_tests(version_dir)
        checks["files"] = tree_hashes(library) if library.is_dir() else {}
        checks["ok"] = (not checks["layout"] and not checks["task_names"] and checks["tests"]["ok"]
                        and all(v == "ok" for v in checks["snippet"].values()))
        checks["validated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        if write:
            write_json(version_dir / "validation.json", checks)
        return checks

    # ---------------- session ----------------
    def curate(self, base_version: str, task_runs: Mapping[str, object], out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        if out_dir.exists() and any(out_dir.iterdir()):
            raise ValueError(f"curation dir is not empty: {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
        state = {"status": "preparing", "base": base_version, "tasks": sorted(task_runs), "started_at": time.time()}
        write_json(out_dir / "curate.json", state)
        try:
            ws = self.build_inputs(base_version, task_runs, out_dir / "workspace")
            protect_inputs(ws)
            before = inputs_hash(ws)
            state.update(status="running")
            write_json(out_dir / "curate.json", state)
            rc, timed_out = wait_session(self.launcher(text.CURATE_PROMPT, ws), self.wall_seconds)
            state.update(rc=rc, timed_out=timed_out, status="checking")
            if inputs_hash(ws) != before:
                raise CurationError("curation session modified its read-only inputs")
            out = ws / "out"
            if any(p.is_symlink() for p in out.rglob("*")):
                raise CurationError("curation output contains symlinks")
            candidate = out_dir / "candidate"
            copy_tree(out / "library", candidate / "library")
            copy_tree(out / "tests", candidate / "tests", exclude=("__pycache__",))
            if (out / "CHANGES.md").is_file():
                (candidate / "CHANGES.md").write_bytes((out / "CHANGES.md").read_bytes())
            write_json(candidate / "candidate.json", {"base": base_version, "tasks": sorted(task_runs),
                                                      "curation": str(out_dir)})
            checks = self.validate(candidate)
            state.update(status="finished", finished_at=time.time(), candidate=str(candidate),
                         validation_ok=checks["ok"])
            write_json(out_dir / "curate.json", state)
            return candidate
        except Exception as exc:
            state.update(status="failed", finished_at=time.time(), error=f"{type(exc).__name__}: {exc}")
            write_json(out_dir / "curate.json", state)
            raise

    # ---------------- publish ----------------
    def publish(self, candidate_dir: Path, *, base: Optional[str] = None, allow_stale_base: bool = False) -> Path:
        """Validate again and copy the candidate to the library root as the next v<N>; update VERSIONS.json."""
        if knowledge.is_packaged_root(self.root):
            raise CurationError("the packaged shared library is immutable; choose an external training registry")
        candidate_dir = Path(candidate_dir)
        meta = read_json(candidate_dir / "candidate.json", {}) or {}
        base = base or meta.get("base")
        checks = self.validate(candidate_dir)
        if not checks["ok"]:
            raise CurationError("validation failed (layout / snippet rules / unit tests / task names); not published")
        latest = knowledge.latest_published(self.root)
        if latest and base != latest and not allow_stale_base:
            raise CurationError(f"candidate is based on {base}, but the latest published version is {latest}; "
                                "curate again")
        number = (knowledge.version_number(latest) or 0) + 1 if latest else 1
        name = f"v{number}"
        target = self.root / name
        if target.exists():
            raise CurationError(f"{target} already exists")
        copy_tree(candidate_dir, target, exclude=("__pycache__",))
        knowledge.register_version(name, "published", root=self.root, base=base, source_candidate=str(candidate_dir),
                                   published_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        index = knowledge.load_versions(self.root)
        for other, value in index["versions"].items():
            if other != name and value.get("status") == "candidate" and \
                    (self.root / other).resolve() == candidate_dir.resolve():
                value["published_as"] = name
                knowledge.save_versions(index, self.root)
        write_json(candidate_dir / "candidate.json", {**meta, "published_as": name})
        return target


__all__ = ["CurationError", "Curator", "TaskEvidence", "robodojo_task_names", "usage_rows"]
