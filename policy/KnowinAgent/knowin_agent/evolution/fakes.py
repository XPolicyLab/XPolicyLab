"""Fake coding-agent launchers for tests and dry runs: they write a plausible out/ synchronously.

FakeReviewAgent merges the practice inputs it finds (cites each run's best Score, merges shared proposals);
FakeCurationAgent copies the current library and adds one tested helper. `mutate` lets a test break the
output on purpose (e.g. write out/skills/shared/, add an import, tamper with inputs).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

from knowin_agent.evolution.review import FinishedProcess

SKILL = '''"""Hover/contact heights over a measured top surface. Merged from practice."""
import numpy as np


def approach_heights(top_z, clearance=0.08, depth=0.01):
    top = float(top_z)
    return {"hover": top + float(clearance), "contact": top - float(depth)}


def midpoint(a, b):
    return ((np.asarray(a, dtype=float) + np.asarray(b, dtype=float)) / 2).tolist()
'''

HELPER = '''"""Clamp a planned motion allowance to a native-action reserve. No actions at include time."""


def reserved_allowance(max_steps, remaining_steps, reserve_steps=20):
    return max(0, min(int(max_steps), int(remaining_steps) - int(reserve_steps)))
'''

HELPER_TEST = '''import unittest
from pathlib import Path

LIBRARY = Path(__file__).resolve().parents[1] / "library"


class ReserveTests(unittest.TestCase):
    def test_allowance(self):
        ns = {}
        exec((LIBRARY / "reserve.py").read_text(), ns)
        self.assertEqual(ns["reserved_allowance"](10, 25), 5)
        self.assertEqual(ns["reserved_allowance"](10, 15), 0)
        self.assertEqual(ns["reserved_allowance"](3, 100), 3)


if __name__ == "__main__":
    unittest.main()
'''


class FakeReviewAgent:
    def __init__(self, mutate: Optional[Callable[[Path], None]] = None):
        self.mutate = mutate
        self.calls: list[tuple[str, Path]] = []

    def __call__(self, prompt: str, workdir: Path):
        workdir = Path(workdir)
        self.calls.append((prompt, workdir))
        out = workdir / "out"
        runs = sorted((workdir / "practice").glob("run*_seed*"))
        evidence = []
        proposals = []
        for run in runs:
            summary = json.loads((run / "summary.json").read_text())
            evidence.append(f"- {run.name}: best {summary['best_score']} at execution "
                            f"{summary['best_at_execution']} of {summary['executions']}")
            prop = run / "lessons" / "shared_proposals.md"
            if prop.is_file():
                proposals.append(f"## from {run.name}\n{prop.read_text()}")
        dev = sorted((workdir / "dev").glob("seed*")) if (workdir / "dev").is_dir() else []
        (out / "skills").mkdir(parents=True, exist_ok=True)
        (out / "lessons").mkdir(parents=True, exist_ok=True)
        (out / "skills" / "approach.py").write_text(SKILL)
        (out / "skills" / "approach.md").write_text("approach_heights(top_z, clearance, depth): hover/contact "
                                                    "heights. Evidence: " + "; ".join(r.name for r in runs) + "\n")
        (out / "lessons" / "PLAN.md").write_text(
            "# PLAN (merged)\n\n| # | What | Verification | Recovery |\n|---|------|--------------|----------|\n"
            "| 1 | approach; `skills/approach.py` `approach_heights`; ~7 | `checkpoint('above', ...)` | re-measure top |\n"
            "| 2 | grasp with `pick`; ~12 | `checkpoint('lifted', ...)` | shift 1 cm, other yaw |\n"
            f"\nPrevious merge used: {(workdir / 'previous').is_dir()}; dev feedback runs: {len(dev)}\n")
        (out / "lessons" / "EVIDENCE.md").write_text("# Evidence\n" + "\n".join(evidence) + "\n")
        (out / "lessons" / "RECOVERY.md").write_text("# Recovery\n- checkpoint_failed after lift: re-measure, retry once\n")
        (out / "lessons" / "shared_proposals.md").write_text("# Shared proposals (merged)\n" + "\n".join(proposals))
        (out / "REPORT.md").write_text(
            f"# Report\nMerged {len(runs)} runs; kept approach_heights.\n\n## Changes\n\n"
            "| PLAN row or file | change | evidence |\n|---|---|---|\n"
            + "".join(f"| 1 approach | kept | {r.name}: best {json.loads((r / 'summary.json').read_text())['best_score']} |\n"
                      for r in runs))
        if self.mutate:
            self.mutate(workdir)
        return FinishedProcess(0)


class FakeCurationAgent:
    def __init__(self, mutate: Optional[Callable[[Path], None]] = None):
        self.mutate = mutate
        self.calls: list[tuple[str, Path]] = []

    def __call__(self, prompt: str, workdir: Path):
        workdir = Path(workdir)
        self.calls.append((prompt, workdir))
        out = workdir / "out"
        library = out / "library"
        library.mkdir(parents=True, exist_ok=True)
        for path in sorted((workdir / "current").iterdir()):
            (library / path.name).write_bytes(path.read_bytes())
        (library / "reserve.py").write_text(HELPER)
        (library / "reserve.md").write_text("reserved_allowance(max_steps, remaining_steps, reserve_steps=20). "
                                            "Evidence: usage.json of every task.\n")
        readme = library / "README.md"
        readme.write_text((readme.read_text() if readme.is_file() else "# Shared library\n")
                          + "\n## reserve.py\n`reserved_allowance` keeps a native-action reserve.\n")
        (out / "tests").mkdir(parents=True, exist_ok=True)
        (out / "tests" / "test_reserve.py").write_text(HELPER_TEST)
        tasks = sorted(p.name for p in (workdir / "tasks").iterdir()) if (workdir / "tasks").is_dir() else []
        (out / "CHANGES.md").write_text("# Changes\n- reserve.py: added; evidence from " + ", ".join(tasks) + "\n")
        if self.mutate:
            self.mutate(workdir)
        return FinishedProcess(0)
