"""Practice gates as pure functions over a score history.

Stall rule (from H5 practice sessions): after `threshold` (default 15, $KNOWIN_AGENT_STALL_EXECUTIONS) accepted
executions without a native Score gain, while the Score is below 100 and budget remains, a stall notice is
raised. From then on submissions are refused (without using an execution slot) until lessons/ has changed
since the notice AND the submission calls reset(); the accepted reset clears the notice and restarts the
no-gain count. A reset starts a new attempt from Score 0, so the first positive Score after it is a gain.

PG80 rule (H1 PG80/150 protocol): practice runs at least to execution 80; a run with Score < 10 before 80
may stop for a harness fix; a run with Score 100 at execution 80 is complete; otherwise it continues to
150 (or until Score 100). A missing native Score holds the run for host review.
"""

from __future__ import annotations

import ast
import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union

STALL_EXECUTIONS = 15
PG80_CHECKPOINT = 80
PG80_CAP = 150


@dataclass(frozen=True)
class ScorePoint:
    """One accepted execution: its native Score (None if unknown), whether it reset the scene, and whether it
    was the reset accepted to clear a stall notice."""
    execution: int
    score: Optional[float]
    reset: bool = False
    cleared_stall: bool = False


def stall_threshold(phase: str = "practice", env: Optional[dict] = None) -> int:
    raw = (env if env is not None else os.environ).get("KNOWIN_AGENT_STALL_EXECUTIONS", str(STALL_EXECUTIONS))
    if not str(raw).isdigit():
        raise ValueError("KNOWIN_AGENT_STALL_EXECUTIONS must be a non-negative integer")
    return int(raw) if phase == "practice" else 0


def calls_reset(code: str) -> bool:
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return False
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "reset"
               for n in ast.walk(tree))


def lessons_digest(lessons_dir: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    root = Path(lessons_dir)
    for path in sorted(root.rglob("*")) if root.is_dir() else ():
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def as_points(history: Iterable) -> list[ScorePoint]:
    """Accept ScorePoints, dicts, (execution, score) pairs or a plain list of scores (execution = index)."""
    points = []
    for index, item in enumerate(history, 1):
        if isinstance(item, ScorePoint):
            points.append(item)
        elif isinstance(item, dict):
            points.append(ScorePoint(int(item.get("execution", index)), item.get("score"),
                                     bool(item.get("reset")), bool(item.get("cleared_stall"))))
        elif isinstance(item, (tuple, list)):
            points.append(ScorePoint(int(item[0]), item[1], *(bool(x) for x in item[2:4])))
        else:
            points.append(ScorePoint(index, item))
    return points


def stall_notice(history: Iterable, *, threshold: int = STALL_EXECUTIONS, budget: Optional[int] = None,
                 lessons: Optional[str] = None) -> Optional[dict]:
    """The stall notice in force after the history, or None. `lessons` is the lessons digest to latch."""
    if threshold <= 0:
        return None
    notice, last_gain, last_score = None, 0, 0.0
    for point in as_points(history):
        if point.cleared_stall:
            notice, last_gain = None, point.execution
        if point.reset:
            last_score = 0.0
        if point.score is None:
            continue
        if point.score > last_score:
            last_gain = point.execution
        last_score = point.score
        used = point.execution
        if (notice is None and point.score < 100 and used - last_gain >= threshold
                and (budget is None or used < budget)):
            notice = {"since_execution": used, "last_score_gain_execution": last_gain,
                      "executions_without_gain": used - last_gain, "lessons_digest": lessons,
                      "required": "Write what keeps failing and what to try instead into lessons/ (new or changed "
                                  "file), then submit code that calls reset(). Other submissions are refused "
                                  "without using an execution slot until then."}
    return notice


class StallRefused(RuntimeError):
    pass


def admit_after_stall(notice: Optional[dict], *, lessons_now: str, code: str = "") -> bool:
    """Check a submission against a stall notice. Returns True if it clears the notice, False if no notice
    applies; raises StallRefused (no execution slot used) otherwise."""
    if notice is None:
        return False
    if lessons_now == notice.get("lessons_digest"):
        raise StallRefused(f"stall_notice: no native Score gain in {notice['executions_without_gain']} executions. "
                           "Update lessons/ first; this refusal used no execution slot.")
    if not calls_reset(code):
        raise StallRefused("stall_notice: the next accepted submission must call reset(); "
                           "this refusal used no execution slot.")
    return True


# "Keep the progress you have" alone (10-30) kept every recorded insert_key / pour_balls practice salvaging a
# dropped key or spilled balls for 15-45 executions; resets came only in the last minutes.
ADVICE = ("No native Score gain in {n} executions. Before the next program, compare the scene with the plan "
          "(images, measurements), name the sub-goal that blocks progress and what to change about it, and note "
          "it in lessons/. If the scene is now worse than at the start (a part dropped, spilled, or moved into "
          "an awkward pose) and a recovery has already failed, reset() and redo the attempt with the change; "
          "otherwise keep the progress you have and change how you do the blocking sub-goal.")
ADVICE_STUCK = ("No native Score gain in {n} executions: this attempt is stuck. Write what keeps failing and what "
                "to try instead into lessons/, then reset() and redo the attempt with that change, unless your "
                "last program passed a checkpoint it had never passed before.")


class StallGate:
    """Stateful wrapper for a practice session: record each accepted execution, check each submission.

    mode "enforce" is the H5 rule above. mode "advise" never refuses a submission: after `threshold` executions
    without a gain it attaches ADVICE once (reset only after a setback whose recovery failed), and again only
    after another `threshold` executions without a gain, then ADVICE_STUCK (reset unless the last program
    passed a new checkpoint). Tiered scores rise in steps, so progress can be real without a gain.
    """

    def __init__(self, lessons_dir: Union[str, Path], *, threshold: Optional[int] = None,
                 budget: Optional[int] = None, mode: str = "enforce"):
        if mode not in ("advise", "enforce"):
            raise ValueError("mode must be 'advise' or 'enforce'")
        self.lessons_dir = Path(lessons_dir)
        self.threshold = stall_threshold("practice") if threshold is None else int(threshold)
        self.budget = budget
        self.mode = mode
        self.history: list[ScorePoint] = []
        self.notice: Optional[dict] = None
        self._clearing = False
        self._last_gain = 0
        self._last_score = 0.0
        self._advised = 0

    def admit(self, code: str) -> None:
        """Raise StallRefused if the submission may not run now."""
        if self.mode == "advise":
            return
        self._clearing = admit_after_stall(self.notice, lessons_now=lessons_digest(self.lessons_dir), code=code)

    def record(self, execution: int, score: Optional[float], *, reset: bool = False) -> Optional[dict]:
        if self.mode == "advise":
            return self._advise(execution, score, reset)
        self.history.append(ScorePoint(execution, score, reset, self._clearing))
        self._clearing = False
        before = self.notice
        self.notice = stall_notice(self.history, threshold=self.threshold, budget=self.budget)
        if self.notice is not None:
            self.notice["lessons_digest"] = (before or {}).get("lessons_digest") or lessons_digest(self.lessons_dir)
        return self.notice

    def _advise(self, execution: int, score: Optional[float], reset: bool) -> Optional[dict]:
        if reset:
            self._last_score = 0.0
        if score is not None:
            if score > self._last_score:
                self._last_gain = execution
            self._last_score = score
        quiet = execution - max(self._last_gain, self._advised)
        if (score is not None and score < 100 and quiet >= self.threshold
                and (self.budget is None or execution < self.budget)):
            self._advised = execution
            n = execution - self._last_gain
            return {"executions_without_gain": n,
                    "advice": (ADVICE_STUCK if n >= 2 * self.threshold else ADVICE).format(n=n)}
        return None


def _valid_score(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 100


def pg80_decision(used: int, latest_score: Optional[float], score_at_80: Optional[float] = None, *,
                  checkpoint: int = PG80_CHECKPOINT, cap: int = PG80_CAP) -> dict:
    """What the practice protocol allows after `used` accepted executions."""
    if type(used) is not int or not 0 <= used <= cap:
        raise ValueError(f"executions used must be an integer in 0..{cap}")
    out = {"executions_used": used, "latest_score": latest_score if _valid_score(latest_score) else None,
           "score_at_checkpoint": score_at_80 if _valid_score(score_at_80) else None, "target": None,
           "test_eligible": False, "finish_allowed": False, "accept_allowed": False}
    if not _valid_score(latest_score):
        return {**out, "state": "hold_missing_score"}
    if used < checkpoint:
        low = latest_score < 10
        return {**out, "state": "needs_harness_fix" if low else f"continue_to_{checkpoint}", "target": checkpoint,
                "finish_allowed": low, "accept_allowed": True}
    if not _valid_score(score_at_80):
        return {**out, "state": "hold_missing_checkpoint_score"}
    target = checkpoint if score_at_80 >= 100 else cap
    done = used >= target or latest_score >= 100
    return {**out, "state": "complete" if done else f"continue_to_{cap}", "target": target,
            "test_eligible": done, "finish_allowed": done, "accept_allowed": not done}


def pg80_from_history(history: Iterable, *, checkpoint: int = PG80_CHECKPOINT, cap: int = PG80_CAP) -> dict:
    points = as_points(history)
    used = points[-1].execution if points else 0
    latest = points[-1].score if points else None
    at80 = next((p.score for p in points if p.execution == checkpoint), None)
    return pg80_decision(used, latest, at80, checkpoint=checkpoint, cap=cap)
