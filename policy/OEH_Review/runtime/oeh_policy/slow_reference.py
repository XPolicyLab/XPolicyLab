"""Opt-in Q1 common context and instruction contract; no model calls or inferred robot facts."""
from __future__ import annotations

import copy

from .contracts import default_context, digest, validate_context

STATIC_KEYS = {"task": {"note", "recipe_file"}, "state": {"embodiment", "embodiment_file"}}


def validate_static_context(value, task=None):
    if not isinstance(value, dict):
        raise ValueError("brain.context must be a mapping of public static context")
    for namespace, fields in value.items():
        if namespace not in STATIC_KEYS or not isinstance(fields, dict) or set(fields) - STATIC_KEYS[namespace]:
            raise ValueError("brain.context supports only task note/recipe and state embodiment; no experience")
    if task is not None and (not isinstance(task, str) or not task.strip()):
        raise ValueError("brain.task must be a nonempty public task name")
    if (value.get("task") or {}).get("note") == "recipe" and task is None:
        raise ValueError("brain.context recipe requires brain.task")
    manifest = validate_context(value, {"task": task})
    # Keep only this deliberately limited shared surface; do not expose Review-only switches to slow mode.
    return {namespace: {key: copy.deepcopy(manifest[namespace][key]) for key in fields}
            for namespace, fields in STATIC_KEYS.items()}


def context_packet(manifest, task):
    """Same task/embodiment text Review reads, snapshotted at episode construction, with content hashes."""
    from .teacher import prompt_dir, task_context
    normalized = validate_static_context(manifest, task)
    full = default_context()
    for namespace, fields in normalized.items():
        full[namespace].update(fields)
    packet = {"task_context": task_context(task, context=full), "embodiment": None}
    files = {}
    for namespace, key in (("task", "recipe_file"), ("state", "embodiment_file")):
        relative = normalized[namespace][key]
        if relative is not None:
            text = (prompt_dir("robodojo") / relative).read_text(encoding="utf-8")
            files[relative] = digest(text)
            if key == "embodiment_file":
                packet["embodiment"] = text
    return packet, {"manifest": normalized, "files": files, "content_sha256": digest(packet)}


def reference_persona():
    """One canonical instruction source. Only the opt-in Q1 row selects this section."""
    from .llm_contract import reference_section
    return reference_section("q1-reference")


def episode_summary(events, *, decisions, fills, native_steps, active_seconds, wall_seconds,
                    close_reason, agent_claim, environment_terminal):
    """Measured accounting only; missing provider usage remains explicitly incomplete."""
    def rows(kind):
        return [row["payload"] for row in events if row["kind"] == kind]

    requests = [row for row in rows("model_call") if row.get("method") == "internal_native_brain"]
    request_ids = {row["request_id"] for row in requests}
    dispatches = [row for row in rows("provider_call") if row.get("request_id") in request_ids]
    usage = [row for row in rows("usage") if row.get("request_id") in request_ids]
    covered = {row["request_id"] for row in usage}
    timings = {row["request_id"]: row["duration_ms"] for row in rows("model_timing")
               if row.get("request_id") in request_ids}
    reported = {key: sum(row.get(key) or 0 for row in usage) if usage else None
                for key in ("input_tokens", "output_tokens", "cached_input_tokens", "total_tokens")}
    reasoning = [row["reasoning_output_tokens"] for row in usage if row.get("reasoning_output_tokens") is not None]
    reported["reasoning_output_tokens"] = sum(reasoning) if reasoning else None
    # close() appends its own stop record; retain earlier stops even if they have
    # the same spelling as the caller's final reason.
    stopped = [row["reason"] for row in rows("execution_stopped")[:-1]]
    prior_stop = stopped[0] if stopped else None
    guard_reasons = {"model_call_budget", "budget_exhausted", "cancelled_or_expired", "llm_token_budget_exceeded"}
    if agent_claim is not None:
        termination = {"kind": "agent_requested", "reason": "agent_requested_termination",
                       "agent_claim": copy.deepcopy(agent_claim)}
    elif prior_stop in guard_reasons or close_reason in guard_reasons:
        termination = {"kind": "guard_stop", "reason": prior_stop or close_reason}
    elif prior_stop is not None:
        termination = {"kind": "abnormal_stop", "reason": prior_stop}
    elif environment_terminal or close_reason == "native_terminal":
        termination = {"kind": "environment_terminal", "reason": close_reason}
    elif close_reason not in {"episode_closed", "agent_requested_termination"}:
        termination = {"kind": "abnormal_stop", "reason": close_reason}
    else:
        termination = {"kind": "external_close", "reason": close_reason}
    termination["benchmark_outcome"] = "owned_by_evaluator"
    return {"schema": "oeh.slow-summary.v1", "termination": termination,
            "decision_turns": decisions, "fallback_fills": fills, "native_steps": native_steps,
            "llm_requests": len(requests), "provider_dispatches": len(dispatches),
            "usage_records": len(usage), "requests_without_usage": sorted(request_ids - covered),
            "usage_complete": (covered == request_ids and len(dispatches) == len(requests)
                               and all(row.get("call_complete") is True for row in usage)),
            "reported_tokens": reported, "cost_usd": None,
            "cost_status": "no billing price supplied; tokens are provider-reported",
            "timed_requests": len(timings), "model_request_seconds": sum(timings.values()) / 1000,
            "active_seconds": active_seconds, "wall_seconds_including_cleanup": wall_seconds,
            "total_token_hard_limit": None}
