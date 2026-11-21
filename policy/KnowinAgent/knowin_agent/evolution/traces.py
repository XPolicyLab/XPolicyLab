"""Read run records (tool_calls.jsonl) into review/curation evidence.

An `exec` submission is a tool_calls row with tool == 'exec', arguments {code, include} and result = the
ExecReceipt (ok, error, stop_reason, interruption, native_steps, score_feedback ...). The score of an
execution is read from result.score_feedback (a number or {"score": x}), falling back to row["score"]; runs
without feedback (tests, blind practice) carry the host's `score_host_only`, read only after the run.
Refused submissions (a stall refusal, or the exec budget used up) never ran: they used no execution slot and
have no observations folder, so they are not executions.

Execution n is the n-th submission that ran; `observations/<n:06d>/` holds the views after it.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Optional

from knowin_agent.core.recorder import read_table
from knowin_agent.evolution.gates import calls_reset
from knowin_agent.workspace.files import read_json

BEST_ATTEMPT_MAX = 40
EARLY_STOPS_MAX = 30
MOTION_EVENTS_MAX = 30
KEY_VIEWS_MAX = 40
WHY_CHARS = 160
STDOUT_CHARS = 1500
NORMAL_STOPS = (None, "", "completed")
# Console and host stop reasons of a run that was cut off rather than played out.
CUT_OFF = ("task_agent_exited", "runner_error", "agent_exited", "agent_model_error")


def _result(row: dict) -> dict:
    result = row.get("result")
    return result if isinstance(result, dict) else {}


def _number(value) -> Optional[float]:
    if isinstance(value, dict):
        value = value.get("score")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def score_of(row: dict) -> Optional[float]:
    feedback = _result(row).get("score_feedback", row.get("score_feedback"))
    if isinstance(feedback, dict):
        feedback = feedback.get("score")
    if feedback is None:
        feedback = row.get("score")
    return _number(feedback)


def host_score_of(row: dict) -> Optional[float]:
    return _number(row.get("score_host_only"))


def steps_used(steps) -> Optional[int]:
    """Native actions one execution spent: hosted receipts give {"before", "after", "used", "resets"}."""
    if isinstance(steps, dict):
        steps = steps.get("used")
    return int(steps) if isinstance(steps, (int, float)) and not isinstance(steps, bool) else None


def did_reset(row: dict) -> bool:
    """A reset that happened (the receipt counts it); a refused `reset()` call is not one."""
    receipt = _result(row)
    steps = receipt.get("native_steps")
    if isinstance(steps, dict) and "resets" in steps:
        return bool(steps.get("resets"))
    return bool(receipt.get("reset")) or calls_reset((row.get("arguments") or {}).get("code", ""))


def total_steps(result: dict, per_execution: list) -> int:
    """Native actions of the scored attempt: the evaluation's count, else the budget's, else the receipts' sum."""
    for value in (result.get("native_steps"), (result.get("evaluation_host_only") or {}).get("native_steps"),
                  (result.get("budget") or {}).get("ticks_used")):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
    return sum(s for s in per_execution if s)


def refused(row: dict) -> bool:
    result = _result(row)
    if result.get("status") == "refused" or row.get("status") == "refused" or result.get("stop_reason") == "stall_refused":
        return True
    # The exec tool raised before running anything (e.g. "the execution budget (80) is used up").
    return bool(result) and set(result) <= {"error"} and isinstance(result.get("error"), str)


def task_rows(run_dir: Path) -> list[dict]:
    return [r for r in read_table(Path(run_dir), "tool_calls") if r.get("scope", "task") == "task"]


def exec_rows(run_dir: Path) -> list[dict]:
    return [r for r in task_rows(run_dir) if r.get("tool") == "exec" and not refused(r)]


def tool_counts(run_dir: Path) -> dict[str, int]:
    return dict(sorted(Counter(r.get("tool") for r in read_table(Path(run_dir), "tool_calls")).items()))


def include_counts(run_dir: Path) -> dict[str, int]:
    counts: Counter = Counter()
    for row in exec_rows(run_dir):
        for name in set((row.get("arguments") or {}).get("include") or []):
            counts[str(name)] += 1
    return dict(sorted(counts.items()))


def motion_health(rows: list[dict]) -> dict:
    """How the run's moves and grasps ended, from the motion tracks of each execution receipt, with the
    executions where a move stalled, timed out or dropped its part, or a close found nothing."""
    counts: Counter = Counter()
    events = []
    for n, row in enumerate(rows, 1):
        for track in _result(row).get("tracks") or []:
            out = track.get("result") or {}
            label = str(track.get("label") or "")
            if label.startswith("gripper"):
                commanded = out.get("commanded")
                if isinstance(commanded, (int, float)) and commanded < 0.5:
                    counts["closes"] += 1
                    if out.get("holding") == "no":
                        counts["closes_on_nothing"] += 1
                        events.append({"execution": n, "arms": track.get("arms"), "status": "closed_on_nothing"})
                continue
            if not label.startswith("move_to"):
                continue
            status = out.get("status")
            counts["moves"] += 1
            counts[f"moves_{status}"] += 1
            if out.get("reason") == "arm_not_following":
                counts["arm_not_following"] += 1
            if status in ("stalled", "timeout", "dropped"):
                events.append({"execution": n, "arm": out.get("arm"), "status": status, "reason": out.get("reason"),
                               "error_mm": out.get("position_error_mm"), "path": out.get("path")})
    return {"counts": dict(sorted(counts.items())), "events": events[-MOTION_EVENTS_MAX:]}


def run_status(run_dir: Path, result: dict) -> dict:
    """Whether the run played out: a run the console or an operator cut off, or whose agent died, is not a
    measurement of its knowledge, whatever partial tier it held when it stopped."""
    run_dir = Path(run_dir)
    protocol = read_json(run_dir / "protocol.json") or {}
    stop = str(result.get("stop_reason") or protocol.get("reason") or "")
    notes = []
    if any(stop.startswith(s) for s in CUT_OFF):
        notes.append(f"stopped early ({stop}): the agent did not finish")
    if protocol.get("interrupted"):
        notes.append(f"interrupted: {json.dumps(protocol['interrupted'], ensure_ascii=False)[:200]}")
    if protocol.get("operator_note"):
        notes.append(f"operator: {str(protocol['operator_note'])[:200]}")
    if protocol.get("valid") is False:
        notes.append("the console marked this run invalid")
    return {"complete": not notes, "notes": notes}


def attempts(rows: list[dict], scores: list) -> list[dict]:
    """The run split at each reset that happened: executions, best and last score, native actions of each."""
    spans, start = [], 1
    for n, row in enumerate(rows, 1):
        if did_reset(row) and n > 1:
            spans.append((start, n - 1))
            start = n
    if rows:
        spans.append((start, len(rows)))
    out = []
    for k, (first, last) in enumerate(spans, 1):
        known = [s for s in scores[first - 1:last] if s is not None]
        steps = [steps_used(_result(r).get("native_steps")) for r in rows[first - 1:last]]
        out.append({"attempt": k, "executions": [first, last], "best_score": max(known, default=None),
                    "final_score": known[-1] if known else None, "native_steps": sum(s for s in steps if s)})
    return out


def delegations(run_dir: Path) -> list[dict]:
    """Each `interact` hand-over to the Local agent: after which execution, the goal, how it ended, the native
    actions it spent, and the Local agent's own calls."""
    out, executions = [], 0
    local: dict[str, list] = {}
    rows = read_table(Path(run_dir), "tool_calls")
    for row in rows:
        if row.get("scope") == "local":
            local.setdefault(str(row.get("delegation")), []).append(row)
    for row in rows:
        if row.get("scope", "task") != "task":
            continue
        if row.get("tool") == "exec" and not refused(row):
            executions += 1
        elif row.get("tool") == "interact":
            result = _result(row)
            args = row.get("arguments") or {}
            name = str(result.get("delegation") or "")
            steps = [{"tool": r.get("tool"), "status": r.get("status") or _result(r).get("status"),
                      "why": str(r.get("rationale") or "")[:WHY_CHARS]} for r in local.get(name, [])]
            ticks = result.get("ticks_used")
            if ticks is None and isinstance(row.get("tick_after"), int) and isinstance(row.get("tick_before"), int):
                ticks = max(0, row["tick_after"] - row["tick_before"])
            out.append({"delegation": name or None, "after_execution": executions,
                        "goal": str(args.get("goal") or "")[:300], "completion": str(args.get("completion") or "")[:300],
                        "status": result.get("status"), "native_steps": ticks,
                        "report": str(result.get("report") or "")[:600], "local_calls": steps,
                        "score_after": score_of(row) if score_of(row) is not None else host_score_of(row)})
    return out


def run_summary(run_dir: Path, result: Optional[dict] = None) -> dict:
    """Score history and execution statistics of one run (practice or test)."""
    run_dir = Path(run_dir)
    result = result or read_json(run_dir / "episode_result.json") or read_json(run_dir / "result.json") or {}
    rows = exec_rows(run_dir)
    feedback = [score_of(r) for r in rows]
    host = [host_score_of(r) for r in rows]
    during_run = any(s is not None for s in feedback)
    # A run without feedback (a test, blind practice) is read after the fact from the host's record.
    scores = feedback if during_run else host
    source = "feedback" if during_run else ("host_after_run" if any(s is not None for s in host) else "none")
    history, last, resets, stops, native = [], None, [], [], []
    for n, row in enumerate(rows, 1):
        score = scores[n - 1]
        if score is not None and score != last:
            history.append({"execution": n, "score": score})
            last = score
        if did_reset(row):
            resets.append(n)
        receipt = _result(row)
        if receipt.get("stop_reason") not in NORMAL_STOPS:
            stops.append({"execution": n, "reason": receipt.get("stop_reason"),
                          "interruption": receipt.get("interruption")})
        native.append(steps_used(receipt.get("native_steps")))
    known = [(n, s) for n, s in enumerate(scores, 1) if s is not None]
    # The evaluation's own score also counts: a success reached inside a delegated `interact` session leaves
    # no exec score feedback, and was summarized as 0 (a 100-scoring conveyor PG was demoted by its review).
    evaluated = _number((result.get("evaluation_host_only") or {}).get("score"))
    candidates = [s for _, s in known] + ([evaluated] if evaluated is not None else [])
    best = max(candidates, default=result.get("best_score", result.get("score")))
    best_at = next((n for n, s in known if s == best), len(rows) if evaluated is not None and best == evaluated else None)
    start = max([n for n in resets if best_at is not None and n <= best_at], default=1)
    final = evaluated if evaluated is not None else next((s for s in reversed(scores) if s is not None), None)
    parts = attempts(rows, scores)
    handed = delegations(run_dir)
    last_receipt = _result(rows[-1]) if rows else {}
    return {
        "run": run_dir.name,
        "seed": result.get("seed", (result.get("evaluation_host_only") or {}).get("seed")),
        "phase": result.get("phase"),
        "status": run_status(run_dir, result),
        "executions": len(rows),
        "best_score": best,
        "best_at_execution": best_at,
        "final_score": final if final is not None else result.get("score"),
        "score_source": source,
        "score_history": history,
        "resets_at": resets,
        "attempts": parts,
        "first_attempt_best_score": parts[0]["best_score"] if parts else None,
        "best_attempt_executions": [start, best_at] if best_at else None,
        "success": result.get("success"),
        "stop_reason": result.get("stop_reason"),
        "finish_report": result.get("finish_report"),
        "native_steps": total_steps(result, native),
        "native_steps_note": "the scored (last) attempt; attempts[] and delegations[] give every attempt",
        "native_step_limit": (result.get("evaluation_host_only") or {}).get("native_step_limit"),
        "native_steps_per_execution": native,
        "native_steps_delegated": sum(d["native_steps"] or 0 for d in handed),
        "executions_counted": last_receipt.get("executions_used"),
        "measure_only_uncounted": last_receipt.get("measure_only_uncounted"),
        "decisions": result.get("decisions"),
        "wall_seconds": result.get("wall_seconds"),
        "score_feedback_during_run": during_run,
        "early_stops": stops[-EARLY_STOPS_MAX:],
        "tool_usage": tool_counts(run_dir),
        "include_usage": include_counts(run_dir),
        "motion_health": motion_health(rows),
        "delegations": [{k: d[k] for k in ("delegation", "after_execution", "status", "native_steps", "score_after")}
                        for d in handed],
    }


# ---------------- programs, views and the timeline ----------------

def _short(value, limit: int = 120) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _error_text(receipt: dict) -> str:
    error = receipt.get("error")
    if isinstance(error, dict):
        return f"{error.get('type', 'Error')}: {error.get('message', '')}"
    return str(error or "")


def _checkpoints(receipt: dict) -> list[str]:
    return [f"{c.get('name')}={'ok' if c.get('passed') else 'FAILED'}" for c in receipt.get("checkpoints") or []
            if isinstance(c, dict) and c.get("kind") != "mark"]


def _motion(receipt: dict) -> str:
    counts: Counter = Counter()
    notes = []
    for track in receipt.get("tracks") or []:
        out = track.get("result") or {}
        label = str(track.get("label") or "")
        if label.startswith("gripper"):
            if out.get("holding") == "no" and isinstance(out.get("commanded"), (int, float)) and out["commanded"] < 0.5:
                notes.append(f"close on nothing {''.join(a[0].upper() for a in track.get('arms') or [])}")
            continue
        if not label.startswith("move_to"):
            continue
        status = out.get("status")
        counts[status] += 1
        if status not in ("reached", None):
            arm = str(out.get("arm") or "?")[:1].upper()
            err = out.get("position_error_mm")
            notes.append(f"{status} {arm}" + (f" {err:.0f}mm" if isinstance(err, (int, float)) else ""))
    head = f"{counts.get('reached', 0)}/{sum(counts.values())} moves reached" if counts else ""
    return "; ".join([head] + notes[:4]).strip("; ")


def _grip(run_dir: Path, n: int) -> str:
    state = (read_json(Path(run_dir) / "observations" / f"{int(n):06d}" / "state.json") or {}).get("state") or {}
    parts = []
    for arm in ("left", "right"):
        joints = state.get(f"{arm}_ee_joint_state")
        if isinstance(joints, list) and joints:
            parts.append(f"{arm[0].upper()}{float(joints[0]):.2f}")
    return " ".join(parts)


def render_submission(row: dict, n: int, *, attempt: Optional[int] = None, score_before=None) -> str:
    """One execution as a .py file: what was submitted, why, and what the receipt reported."""
    args = row.get("arguments") or {}
    receipt = _result(row)
    score = score_of(row)
    host = host_score_of(row)
    head = [f"# execution {n}" + (f" (attempt {attempt})" if attempt else ""),
            f"# include: {list(args.get('include') or [])}"]
    if row.get("rationale"):
        head.append(f"# why (the agent): {_short(row['rationale'], 600)}")
    if score is not None:
        head.append(f"# score after: {score}" + (f" (before: {score_before})" if score_before is not None else ""))
    elif host is not None:
        head.append(f"# score after (host record, the agent saw none): {host}")
    steps = receipt.get("native_steps")
    if isinstance(steps, dict):
        head.append(f"# native actions: used {steps.get('used')}, tick {steps.get('before')} -> {steps.get('after')}"
                    + (", reset" if steps.get("resets") else ""))
    if receipt.get("stop_reason") not in NORMAL_STOPS:
        head.append(f"# stopped early: {receipt.get('stop_reason')}")
    if _error_text(receipt):
        head.append(f"# error: {_short(_error_text(receipt), 400)}")
    if receipt.get("interruption"):
        head.append(f"# interruption: {_short(receipt['interruption'], 400)}")
    if _checkpoints(receipt):
        head.append(f"# checkpoints: {', '.join(_checkpoints(receipt))}")
    if _motion(receipt):
        head.append(f"# motion: {_motion(receipt)}")
    stdout = str(receipt.get("stdout") or "")
    if stdout.strip():
        tail = stdout[-STDOUT_CHARS:]
        head.append("# stdout" + (" (tail)" if len(stdout) > STDOUT_CHARS else "") + ":")
        head += ["#   " + line for line in tail.splitlines()]
    return "\n".join(head) + "\n" + str(args.get("code") or "") + "\n"


def copy_views(run_dir: Path, n: int, target: Path) -> int:
    """The camera views recorded after execution n (observations/<n>/current_cam_*.png) as <n>_<camera>.png."""
    folder = Path(run_dir) / "observations" / f"{int(n):06d}"
    copied = 0
    for image in sorted(folder.glob("current_cam_*.png")) if folder.is_dir() else []:
        target.mkdir(parents=True, exist_ok=True)
        (target / f"{int(n):06d}_{image.stem[len('current_cam_'):]}.png").write_bytes(image.read_bytes())
        copied += 1
    return copied


def key_executions(rows: list[dict], scores: list, limit: int = KEY_VIEWS_MAX) -> list[int]:
    """Executions whose views matter most: score changes (and the one before), resets, early stops, motion
    failures and failed checkpoints, the first two and the last five."""
    picked: list[int] = []
    last = None

    def add(n):
        if 1 <= n <= len(rows) and n not in picked:
            picked.append(n)

    for n, row in enumerate(rows, 1):
        receipt = _result(row)
        score = scores[n - 1]
        if score is not None and score != last:
            add(n), add(n - 1)
            last = score
        if (did_reset(row) or receipt.get("stop_reason") not in NORMAL_STOPS
                or any("FAILED" in c for c in _checkpoints(receipt))
                or any(w in _motion(receipt) for w in ("stalled", "timeout", "dropped", "nothing"))):
            add(n)
    for n in (1, 2, *range(len(rows) - 4, len(rows) + 1)):
        add(n)
    if len(picked) > limit:
        tail = set(range(len(rows) - 4, len(rows) + 1))
        picked = [n for n in picked if n in tail] + [n for n in picked if n not in tail][:limit - len(tail)]
    return sorted(picked)


def write_timeline(run_dir: Path, summary: dict, target: Path, *, raw_dir: Optional[str] = None,
                   header: Optional[list[str]] = None) -> dict:
    """The review's entry to one run: target/timeline.md (one line per execution, delegation and single
    tool call), target/executions/<n>.py for every execution, and the views after each one -- referenced in
    the read-only raw run directory when it is mounted (raw_dir), else copied for the key executions."""
    run_dir, target = Path(run_dir), Path(target)
    rows = task_rows(run_dir)
    execs = [r for r in rows if r.get("tool") == "exec" and not refused(r)]
    scores = [score_of(r) for r in execs] if summary.get("score_feedback_during_run") else [host_score_of(r) for r in execs]
    attempt_of = {}
    for part in summary.get("attempts") or []:
        for n in range(part["executions"][0], part["executions"][1] + 1):
            attempt_of[n] = part["attempt"]
    (target / "executions").mkdir(parents=True, exist_ok=True)
    last = None
    for n, row in enumerate(execs, 1):
        (target / "executions" / f"{n:06d}.py").write_text(
            render_submission(row, n, attempt=attempt_of.get(n), score_before=last), encoding="utf-8")
        last = scores[n - 1] if scores[n - 1] is not None else last
    keys = key_executions(execs, scores)
    if raw_dir is None:
        for n in keys:
            copy_views(run_dir, n, target / "views")
    by_delegation = {d["delegation"]: d for d in delegations(run_dir)}

    status = summary.get("status") or {}
    lines = [f"# Timeline of {summary.get('run')}", ""]
    lines += header or []
    lines += [
        f"- Phase: {summary.get('phase') or '?'}; seed {summary.get('seed')}; score shown to the agent: "
        f"{'yes' if summary.get('score_feedback_during_run') else 'no'} (scores below: {summary.get('score_source')})",
        f"- Played out: {'yes' if status.get('complete', True) else 'NO -- ' + '; '.join(status.get('notes') or [])}",
        f"- Best {summary.get('best_score')} at execution {summary.get('best_at_execution')}; final "
        f"{summary.get('final_score')}; stop: {summary.get('stop_reason')}",
        "- Attempts (split at each reset): " + "; ".join(
            f"#{a['attempt']} executions {a['executions'][0]}-{a['executions'][1]} best {a['best_score']} "
            f"final {a['final_score']} ({a['native_steps']} native actions)" for a in summary.get("attempts") or []),
        f"- Budget: {summary.get('executions')} executions ({summary.get('executions_counted')} counted, "
        f"measure-only uncounted {summary.get('measure_only_uncounted')}); native actions of the scored attempt "
        f"{summary.get('native_steps')} of {summary.get('native_step_limit')}, delegated "
        f"{summary.get('native_steps_delegated')}; decisions {summary.get('decisions')}; wall {summary.get('wall_seconds')} s",
    ]
    if summary.get("finish_report"):
        lines.append(f"- The agent's finish report: {_short(summary['finish_report'], 800)}")
    if raw_dir:
        lines.append(f"- Views after execution n: `{raw_dir}/observations/<n:06d>/current_cam_{{head,left_wrist,"
                     f"right_wrist}}.png` (read-only, every execution); robot state in `state.json` next to them.")
    else:
        lines.append(f"- Views after the key executions: `views/<n:06d>_<camera>.png` ({len(keys)} executions).")
    lines += ["- Programs with their receipts: `executions/<n:06d>.py`. `*` marks a key execution.", "",
              "| # | att | score | steps | result | motion | grip L/R | why (the agent) |",
              "|---|---|---|---|---|---|---|---|"]
    n, last = 0, None
    for row in rows:
        tool = row.get("tool")
        if tool == "exec":
            if refused(row):
                lines.append(f"| - | | | | refused: {_short(_error_text(_result(row)) or _result(row).get('stop_reason'), 80)} "
                             f"| | | |")
                continue
            n += 1
            receipt = _result(row)
            score = scores[n - 1]
            shown = "" if score is None else (f"{last}→{score}" if last is not None and score != last else f"{score}")
            last = score if score is not None else last
            outcome = []
            if did_reset(row):
                outcome.append("RESET")
            if receipt.get("stop_reason") not in NORMAL_STOPS:
                outcome.append(str(receipt.get("stop_reason")))
            if _error_text(receipt):
                outcome.append(_short(_error_text(receipt), 70))
            outcome += [c for c in _checkpoints(receipt) if "FAILED" in c]
            steps = steps_used(receipt.get("native_steps"))
            lines.append(f"| {'*' if n in keys else ''}{n} | {attempt_of.get(n, '')} | {shown} | "
                         f"{'' if steps is None else steps} | {_short('; '.join(outcome), 90)} | {_short(_motion(receipt), 70)} | "
                         f"{_grip(run_dir, n)} | {_short(row.get('rationale') or '', WHY_CHARS)} |".replace("\n", " "))
        elif tool == "interact":
            d = by_delegation.get(str(_result(row).get("delegation") or "")) or {}
            lines.append(f"| D{d.get('delegation') or ''} | | {'' if d.get('score_after') is None else d['score_after']} | "
                         f"{d.get('native_steps') or ''} | delegated: {d.get('status')} | {len(d.get('local_calls') or [])} "
                         f"Local calls | | {_short(d.get('goal') or '', WHY_CHARS)} |")
        elif tool in ("remember", "api_reference"):
            continue
        else:
            receipt = _result(row)
            found = receipt.get("status") or ("ok" if row.get("ok") else "failed")
            lines.append(f"| · | | | | {tool}: {_short(found, 40)} | | | {_short(row.get('rationale') or '', WHY_CHARS)} |")
    notes = {}
    for row in rows:
        if row.get("tool") == "remember":
            args = row.get("arguments") or {}
            notes[str(args.get("key"))] = str(args.get("text") or "")
    if notes:
        lines += ["", "## The agent's notes (last version of each key)"]
        lines += [f"- `{k}`: {_short(v, 600)}" for k, v in notes.items()]
    if by_delegation:
        lines += ["", "## Delegations to the Local agent"]
        for d in by_delegation.values():
            lines.append(f"- D{d['delegation']} after execution {d['after_execution']}: {d['status']}, "
                         f"{d['native_steps']} native actions. Goal: {_short(d['goal'], 300)}. Completion: "
                         f"{_short(d['completion'], 200)}. Report: {_short(d['report'], 400)}")
            lines += [f"  - {c['tool']}: {c['status']} -- {_short(c['why'], 120)}" for c in d["local_calls"]]
    (target / "timeline.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"executions": len(execs), "key_executions": keys}


def write_best_attempt(run_dir: Path, summary: dict, target: Path) -> int:
    """Write the submissions of the attempt that reached the best Score (at most the last 40), each with the
    camera views recorded after it."""
    span = summary.get("best_attempt_executions")
    if not span or not summary.get("best_score"):
        # No program ever scored: the last programs are still the evidence of what was tried and how it failed.
        return write_submissions(run_dir, target)
    first, last = span
    rows = exec_rows(run_dir)
    written = 0
    for n in range(max(first, last - BEST_ATTEMPT_MAX + 1), last + 1):
        target.mkdir(parents=True, exist_ok=True)
        (target / f"{n:06d}.py").write_text(render_submission(rows[n - 1], n), encoding="utf-8")
        copy_views(run_dir, n, target)
        written += 1
    return written


def write_submissions(run_dir: Path, target: Path, *, last: int = BEST_ATTEMPT_MAX) -> int:
    rows = exec_rows(run_dir)
    start = max(1, len(rows) - last + 1)
    for n in range(start, len(rows) + 1):
        target.mkdir(parents=True, exist_ok=True)
        (target / f"{n:06d}.py").write_text(render_submission(rows[n - 1], n), encoding="utf-8")
        copy_views(run_dir, n, target)
    return len(rows) - start + 1 if rows else 0


def ground_truth_report(run_dir: Path) -> Optional[str]:
    """Oracle review input (switch evolution.condition_report): after each execution, which scoring predicates
    held, from the host's record. None when the run has no such record."""
    lines, n, previous = [], 0, None
    for row in task_rows(run_dir):
        if row.get("tool") == "exec" and refused(row):
            continue
        if row.get("tool") == "exec":
            n += 1
        truth = row.get("ground_truth_host_only")
        conditions = (truth or {}).get("conditions") if isinstance(truth, dict) else None
        if not isinstance(conditions, dict) or not conditions.get("predicates"):
            continue
        state = {p["pred"]: p.get("ok") for p in conditions["predicates"]}
        if previous is None:
            lines.append(f"- after execution {n} ({row.get('tool')}): " + "; ".join(
                f"{k}: {'yes' if v else ('no' if v is False else '?')}" for k, v in state.items()))
        else:
            changed = {k: v for k, v in state.items() if previous.get(k) != v}
            if changed:
                lines.append(f"- after execution {n} ({row.get('tool')}): " + "; ".join(
                    f"{k} -> {'yes' if v else ('no' if v is False else '?')}" for k, v in changed.items()))
        previous = state
    if not lines:
        return None
    return ("# Scoring predicates over the run (host record; the agent never saw it)\n\n"
            "First line: every predicate; later lines: only the ones that changed.\n\n" + "\n".join(lines) + "\n")
