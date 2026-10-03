"""Paired survivor selection for a previously frozen comparison.

This checks supplied identities and local evidence, not the truthfulness of a
producer, policy isolation, layout novelty, or completeness of its dependency
closure. Admission, full candidate freezing and layout registration remain
separate prerequisites. No benchmark-specific names or score scales live here.
"""

import math
from datetime import datetime, timezone
from pathlib import Path

from PhysicalRSI_core.infra.storage import (
    atomic_json,
    digest,
    file_digest,
    identifier,
    locked,
    relative_path,
)


def _number(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError("Expected a finite numeric value")
    return value


def _sha(value):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("Expected a SHA256 identity")
    return value


def _time(value):
    value = datetime.fromisoformat(value)
    if value.tzinfo is None or value > datetime.now(timezone.utc):
        raise ValueError("Invalid event timestamp")
    return value


def comparison_identity(comparison):
    """Validate the predeclared pool and metric profile; return its identity."""
    if comparison.get("schema_version") != 1:
        raise ValueError("Unsupported comparison schema")
    identifier(comparison["round_id"])
    _time(comparison["frozen_at"])
    candidates = comparison["candidates"]
    parent = comparison["parent_id"]
    if len(candidates) < 2 or parent not in candidates:
        raise ValueError("Comparison requires the parent and at least one child")
    for name, freeze in candidates.items():
        identifier(name)
        _sha(freeze)
    profile = comparison["profile"]
    tasks = profile["tasks"]
    if not tasks:
        raise ValueError("Empty task scope")
    for task, spec in tasks.items():
        identifier(task)
        if _number(spec["weight"]) <= 0:
            raise ValueError("Task weights must be positive")
        if type(spec["episodes"]) is not int or spec["episodes"] < 1:
            raise ValueError("Invalid cohort size")
        low, high = map(_number, spec["score_range"])
        if low >= high:
            raise ValueError("Invalid native score range")
        if not 0 <= _number(spec["maximum_regression"]) <= 1:
            raise ValueError("Invalid normalized regression limit")
    if not 0 <= _number(profile["minimum_gain"]) <= 1:
        raise ValueError("Invalid normalized improvement margin")
    if not 0 <= _number(profile["tie_tolerance"]) < 1:
        raise ValueError("Invalid tie tolerance")
    if "cost_reduction" in profile or "capability_qualification" in profile:
        raise ValueError(
            "Optional cost/qualification policies are not admitted in the minimal core"
        )
    return digest(comparison)


def validate_admission_binding(
    admission, *, candidate_id, freeze_sha256, comparison, alias, current_revision
):
    """Validate a producer receipt bound to the current release interaction.

    Paired score selection intentionally stays agnostic to how an external
    evaluator was run.  A release operation that opts into this stricter gate
    must nevertheless prove that each CODE_POLICY admission was produced for this
    candidate, comparison, protocol/profile snapshot and current parent.  The
    check is opt-in so old synthetic/core fixtures remain valid and cannot be
    mistaken for the stronger external-admission contract.
    """
    if (
        not isinstance(admission, dict)
        or admission.get("status", "completed") != "completed"
    ):
        raise ValueError("Admission binding requires a completed producer receipt")
    if (
        admission.get("candidate_id") != candidate_id
        or admission.get("freeze_sha256") != freeze_sha256
    ):
        raise ValueError("Admission binding candidate identity mismatch")
    comparison_sha = digest(comparison)
    if admission.get("comparison_sha256") != comparison_sha:
        raise ValueError("Admission binding comparison identity mismatch")
    protocol = admission.get("protocol_identity")
    if not isinstance(protocol, str) or not protocol.strip():
        raise ValueError("Admission binding protocol identity required")
    profile_sha = admission.get("profile_sha256")
    if not _sha(profile_sha) or profile_sha != digest(comparison.get("profile")):
        raise ValueError("Admission binding profile identity required")
    resource_profile_sha = admission.get("resource_profile_sha256")
    if not _sha(resource_profile_sha):
        raise ValueError("Admission binding resource profile identity required")
    if (
        comparison.get("resource_plan_sha256") is not None
        and resource_profile_sha != comparison["resource_plan_sha256"]
    ):
        raise ValueError("Admission binding resource profile mismatch")
    producer = admission.get("producer")
    if (
        not isinstance(producer, dict)
        or not isinstance(producer.get("id"), str)
        or not producer["id"].strip()
        or not _sha(producer.get("receipt_sha256"))
    ):
        raise ValueError("Admission binding producer receipt required")
    if (
        comparison.get("protocol_sha256") is not None
        and protocol != comparison["protocol_sha256"]
    ):
        raise ValueError("Admission binding protocol identity mismatch")
    binding = admission.get("current_binding")
    if not isinstance(binding, dict):
        raise ValueError("Admission binding current release identity required")
    expected = dict(
        alias=alias,
        revision=current_revision,
        parent_id=comparison.get("parent_id"),
        parent_freeze_sha256=comparison.get("candidates", {}).get(
            comparison.get("parent_id")
        ),
    )
    if any(binding.get(key) != value for key, value in expected.items()):
        raise ValueError("Admission binding differs from current release parent")
    skill_choice_sha = admission.get("skill_choice_policy_sha256")
    if not _sha(skill_choice_sha):
        raise ValueError("Admission binding skill_choice policy identity required")
    return True


def select_survivor(
    comparison, cohort, results, *, evidence_root, qualification_repository=None
):
    """Require complete paired results; retain parent unless a child improves.

    Scores are normalized by predeclared task ranges, then task-weighted. Every
    result must use the exact same task/layout pairs and evaluator identity.
    Missing episodes, runtime errors, changed evidence and duplicate layouts
    invalidate selection rather than being silently excluded or scored as zero.
    """
    comparison_sha = comparison_identity(comparison)
    if cohort.get("comparison_sha256") != comparison_sha:
        raise ValueError("Cohort belongs to another frozen comparison")
    if _time(cohort["generated_at"]) <= _time(comparison["frozen_at"]):
        raise ValueError(
            "Validation must be generated after the entire pool/profile freezes"
        )
    tasks = comparison["profile"]["tasks"]
    layouts = cohort["layouts"]
    if set(layouts) != set(tasks):
        raise ValueError("Cohort task coverage differs")
    seen = set()
    expected = set()
    for task, spec in tasks.items():
        if len(layouts[task]) != spec["episodes"]:
            raise ValueError("Incomplete validation cohort")
        for sha in layouts[task]:
            _sha(sha)
            if sha in seen:
                raise ValueError("Duplicate geometry across task cohorts")
            seen.add(sha)
            expected.add((task, sha))
    cohort_sha = digest(cohort)
    pool = comparison["candidates"]
    if len(results) != len(pool) or {r["candidate_id"] for r in results} != set(pool):
        raise ValueError("Missing, duplicate or unexpected candidate result")
    root = Path(evidence_root).resolve()
    metrics = {}
    revisions = set()
    for result in results:
        name = result["candidate_id"]
        if (
            result.get("kind") != comparison.get("evaluation_kind", "policy_evaluation")
            or type(result.get("native_exit_code")) is not int
            or result["native_exit_code"] != 0
        ):
            raise ValueError(
                "Expected a successfully completed native policy evaluation"
            )
        if (
            result.get("comparison_sha256") != comparison_sha
            or result.get("cohort_sha256") != cohort_sha
            or result.get("freeze_sha256") != pool[name]
        ):
            raise ValueError("Result/candidate identity mismatch")
        revision = result.get("evaluator_revision")
        if not isinstance(revision, str) or not revision:
            raise ValueError("Missing evaluator identity")
        revisions.add(revision)
        rows = result["episodes"]
        if len(rows) != len(expected):
            raise ValueError("Incomplete candidate evaluation")
        used = set()
        scores = {task: [] for task in tasks}
        success = {task: [] for task in tasks}
        for row in rows:
            pair = (row["task"], row["layout_sha256"])
            if pair not in expected or pair in used:
                raise ValueError("Unpaired or repeated episode")
            used.add(pair)
            if row.get("state") != "completed" or type(row.get("success")) is not bool:
                raise ValueError("Runtime failure or missing native outcome")
            score = _number(row["score"])
            low, high = tasks[pair[0]]["score_range"]
            if not low <= score <= high:
                raise ValueError("Native score outside declared range")
            evidence = row.get("evidence_sha256")
            if not isinstance(evidence, dict) or not evidence:
                raise ValueError("Missing native evidence")
            for relative, sha in evidence.items():
                path = (root / relative_path(relative)).resolve()
                if (
                    not path.is_relative_to(root)
                    or not path.is_file()
                    or file_digest(path) != _sha(sha)
                ):
                    raise ValueError(
                        "Native evidence missing, changed or outside the evidence root"
                    )
            scores[pair[0]].append((score - low) / (high - low))
            success[pair[0]].append(row["success"])
        per_task = {task: sum(values) / len(values) for task, values in scores.items()}
        total_weight = sum(spec["weight"] for spec in tasks.values())
        metrics[name] = dict(
            normalized_score=sum(per_task[t] * tasks[t]["weight"] for t in tasks)
            / total_weight,
            per_task=per_task,
            success_rate={t: sum(v) / len(v) for t, v in success.items()},
        )
    if len(revisions) != 1:
        raise ValueError("Mixed evaluator revisions")
    parent = comparison["parent_id"]
    parent_metrics = metrics[parent]
    profile = comparison["profile"]
    eligible = []
    rejection = {}
    cost_rule = profile.get("cost_reduction")
    for name, metric in metrics.items():
        if name == parent:
            continue
        regressed = [
            t
            for t in tasks
            if parent_metrics["per_task"][t] - metric["per_task"][t]
            > tasks[t]["maximum_regression"] + profile["tie_tolerance"]
        ]
        gain = metric["normalized_score"] - parent_metrics["normalized_score"]
        if metric.get("capability_qualification", {}).get("state") == "not_qualified":
            rejection[name] = dict(
                reason="capability_not_qualified",
                qualification=metric["capability_qualification"],
            )
        elif regressed:
            rejection[name] = dict(reason="task_regression", tasks=regressed)
        elif cost_rule is not None:
            success_regressed = [
                t
                for t in tasks
                if parent_metrics["success_rate"][t] - metric["success_rate"][t]
                > cost_rule.get("maximum_success_regression", 0)
                + profile["tie_tolerance"]
            ]
            before = parent_metrics["selection_cost"]["value"]
            after = metric["selection_cost"]["value"]
            if success_regressed:
                rejection[name] = dict(
                    reason="success_regression", tasks=success_regressed
                )
            elif gain + profile["tie_tolerance"] < profile["minimum_gain"]:
                rejection[name] = dict(
                    reason="insufficient_quality", normalized_gain=gain
                )
            elif before is None or after is None:
                rejection[name] = dict(reason="unknown_cost")
            elif (
                before == 0
                or (before - after) / before
                <= cost_rule["minimum_relative_reduction"] + profile["tie_tolerance"]
            ):
                rejection[name] = dict(
                    reason="no_sufficient_cost_reduction",
                    relative_reduction=(before - after) / before if before else None,
                )
            else:
                eligible.append(name)
        elif gain <= profile["minimum_gain"] + profile["tie_tolerance"]:
            rejection[name] = dict(
                reason="no_sufficient_improvement", normalized_gain=gain
            )
        else:
            eligible.append(name)
    winner = parent
    if eligible:
        if cost_rule is not None:
            # Costs here are integer calls/tokens. Prefer quality among equal
            # costs, then use the same deterministic candidate identity tie.
            best_cost = min(metrics[n]["selection_cost"]["value"] for n in eligible)
            eligible = [
                n
                for n in eligible
                if metrics[n]["selection_cost"]["value"] == best_cost
            ]
        best = max(metrics[n]["normalized_score"] for n in eligible)
        winner = min(
            n
            for n in eligible
            if best - metrics[n]["normalized_score"] <= profile["tie_tolerance"]
        )
    return dict(
        schema_version=1,
        comparison_sha256=comparison_sha,
        cohort_sha256=cohort_sha,
        parent_id=parent,
        survivor_id=winner,
        survivor_freeze_sha256=pool[winner],
        decision="retain_parent" if winner == parent else "inherit_child",
        metrics=metrics,
        rejected=rejection,
        result_sha256={r["candidate_id"]: digest(r) for r in results},
        scope=comparison.get(
            "scope",
            "paired evidence integrity only; no automatic physical qualification",
        ),
    )


def write_selection(path, selection):
    """Persist one decision without overwriting an earlier survivor record."""
    path = Path(path)
    with locked(path.with_suffix(".lock")):
        if path.exists():
            raise FileExistsError("Selection already exists")
        atomic_json(path, selection)
