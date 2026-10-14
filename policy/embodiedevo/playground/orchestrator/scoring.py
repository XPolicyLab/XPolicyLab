"""Persist evaluator data outside every agent-visible workspace."""

import math
import time

from orchestrator.files import write_json


def capture(env, run, *, execution, final=False):
    getter = getattr(env, "private_scoring_snapshot", None)
    if getter is None:
        return None
    record = getter()
    if record is None:
        return None
    score = record.get("score")
    if isinstance(score, bool) or not isinstance(score, (float, int)):
        raise ValueError("Native scoring must return a numeric Score")
    if not math.isfinite(score) or not 0 <= score <= 100:
        raise ValueError("Native Score must be finite and between 0 and 100")
    relative = "scoring/final.json" if final else f"scoring/{execution:06d}.json"
    record = {**record, "execution": execution, "final": final, "recorded_at": time.time()}
    write_json(run / relative, record)
    summary = {
        "score": score,
        "score_schema": record["schema"],
        "score_record": relative,
        "score_execution": execution,
    }
    write_json(run / "scoring/latest.json", summary)
    return summary


def result_score(run, state, *, valid):
    """An idle, clean agent exit can use the last completed execution's score."""
    import json

    path = run / "scoring/latest.json"
    if not valid or not path.exists():
        return {"score": None}
    summary = json.loads(path.read_text())
    if summary.get("score_execution") != state.get("executions_used"):
        return {"score": None, "score_error": "score_execution_mismatch"}
    return summary


def aggregate(results, *, complete):
    scores = [r.get("score") for r in results if r.get("valid")]
    present = [
        s
        for s in scores
        if not isinstance(s, bool)
        and isinstance(s, (int, float))
        and math.isfinite(s)
        and 0 <= s <= 100
    ]
    return {
        "scored_scenes": len(present),
        "score": sum(present) / len(results)
        if complete and results and len(present) == len(results)
        else None,
    }
