"""Evidence gates and paired metrics for localized closed-loop recovery."""
from __future__ import annotations

import math
from collections import Counter, defaultdict


def qualify_case(case, training_layouts=()):
    reasons = []
    required = ("case_id", "task", "layout_sha256", "prefix_sha256", "state_sha256",
                "history_sha256", "runtime_sha256", "failure_type", "protected_subgoals")
    for key in required:
        if key not in case or case[key] is None or (key != "protected_subgoals" and not case[key]):
            reasons.append("missing:" + key)
    flags = ("failure_predicate_verified", "failure_persistence_verified", "nonterminal",
             "two_replays_match", "recoverability_witness_success", "teacher_uses_native_controls",
             "frozen_before_candidate_evaluation", "native_rules_unchanged")
    reasons.extend("unverified:" + flag for flag in flags if case.get(flag) is not True)
    budget = case.get("remaining_action_budget")
    witness = case.get("witness_action_count")
    if type(budget) is not int or budget <= 0 or type(witness) is not int or not 0 < witness <= budget:
        reasons.append("invalid_or_exceeded_witness_budget")
    if case.get("layout_sha256") in training_layouts:
        reasons.append("training_layout_overlap")
    if case.get("candidate_outcomes_used_for_selection") is not False:
        reasons.append("candidate_selection_leakage")
    if case.get("terminal") is True:
        reasons.append("terminal_state")
    return {"eligible": not reasons, "reasons": reasons}


def validate_continuation(case, result):
    if result.get("case_id") != case["case_id"]:
        raise ValueError("Wrong case")
    for key in ("layout_sha256", "prefix_sha256", "state_sha256", "history_sha256", "runtime_sha256"):
        if result.get(key) != case[key]:
            raise ValueError("Mismatched continuation identity: " + key)
    if result.get("remaining_action_budget") != case["remaining_action_budget"]:
        raise ValueError("Mismatched budget")
    if result.get("agent_used") is not False:
        raise ValueError("Recovery must run inside the learned policy")
    if result.get("status") != "valid_native_terminal":
        raise ValueError("Infrastructure failures are not policy failures")
    for key in ("native_success", "local_error_cleared", "protected_subgoals_preserved"):
        if type(result.get(key)) is not bool:
            raise ValueError("Unmeasured endpoint: " + key)
    if type(result.get("executed_actions")) is not int or not 0 <= result["executed_actions"] <= case["remaining_action_budget"]:
        raise ValueError("Invalid action count")
    score = result.get("score")
    if type(score) not in (float, int) or not math.isfinite(score) or not 0 <= score <= 100:
        raise ValueError("Score must use the declared 0-100 scale")
    if not result.get("native_result_sha256") or not result.get("predicate_trace_sha256"):
        raise ValueError("Missing native result or local predicate evidence")


def paired_metrics(cases, results, candidate, training_layouts=()):
    accepted, excluded = {}, {}
    for case in cases:
        cid = case["case_id"]
        if cid in accepted or cid in excluded:
            raise ValueError("Duplicate frozen case")
        decision = qualify_case(case, training_layouts)
        (accepted if decision["eligible"] else excluded)[cid] = case if decision["eligible"] else decision
    observed, invalid = {}, []
    seen = set()
    for result in results:
        key = (result["case_id"], result["variant"])
        if key in seen:
            raise ValueError("Multiple outcomes for one case/variant; no best-attempt selection")
        seen.add(key)
        if key[0] not in accepted:
            continue
        try:
            validate_continuation(accepted[key[0]], result)
        except ValueError as exc:
            invalid.append(dict(case_id=key[0], variant=key[1], reason=str(exc)))
            continue
        observed[key] = result
    grouped = defaultdict(list)
    missing = []
    for cid, case in accepted.items():
        pair = [observed.get((cid, v)) for v in ("baseline", candidate)]
        if any(x is None for x in pair):
            missing.append(cid)
        else:
            grouped[case["task"]].append(pair)
    def summarize(pairs):
        out = dict(pairs=len(pairs), fail_to_success=0, success_to_fail=0)
        for j, variant in enumerate(("baseline", candidate)):
            rows = [p[j] for p in pairs]
            out[variant] = {"n": len(rows)}
            for key in ("native_success", "local_error_cleared", "protected_subgoals_preserved", "score"):
                out[variant][key] = sum(r[key] for r in rows) / len(rows) if rows else None
            out[variant]["local_repair_without_damage"] = (sum(r["local_error_cleared"] and r["protected_subgoals_preserved"] for r in rows) / len(rows)) if rows else None
        out["fail_to_success"] = sum(not a["native_success"] and b["native_success"] for a,b in pairs)
        out["success_to_fail"] = sum(a["native_success"] and not b["native_success"] for a,b in pairs)
        return out
    pairs = [p for rows in grouped.values() for p in rows]
    return dict(schema="recoverwam-paired-recovery-v1", candidate=candidate,
                eligible_cases=len(accepted), excluded_cases=excluded, invalid_results=invalid,
                missing_pairs=missing, complete=bool(accepted) and not missing and not invalid,
                per_task={t: summarize(p) for t,p in grouped.items()}, overall=summarize(pairs),
                interpretation="Partial matched-pair estimates are not full-cohort or leaderboard SR.")
