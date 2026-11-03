"""Develop skill_choice candidates without consulting benchmark evaluation scores."""

import math

from PhysicalRSI_core.infra.storage import digest

VLA_EXPERTS = ("pi05", "pi05-sparse-memory")


def assess(task, expert, revision, episodes, *, minimum_episodes):
    if expert not in VLA_EXPERTS or not revision:
        raise ValueError("Expected a pinned VLA expert")
    if type(minimum_episodes) is not int or minimum_episodes < 1:
        raise ValueError("minimum_episodes must be positive")
    if len(episodes) < minimum_episodes:
        raise ValueError("Insufficient development episodes")
    identities = set()
    for row in episodes:
        if row["split"] != "development" or row["task"] != task:
            raise ValueError("Skill_selection accepts only this task's development evidence")
        if row["expert"] != expert or row["revision"] != revision:
            raise ValueError("Expert identity mismatch")
        if row["state"] != "completed" or row["natural_terminal"] is not True:
            raise ValueError("Infrastructure failure or truncation is not inability")
        score = row["episode_score"]
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            raise ValueError("Native episode_score must be finite and within [0, 1]")
        if not row["evidence"] or not row["layout_sha256"]:
            raise ValueError("Layout and execution evidence required")
        if row["layout_sha256"] in identities:
            raise ValueError("Duplicate development layout")
        identities.add(row["layout_sha256"])
    return dict(
        task=task,
        name=expert,
        revision=revision,
        mean_score=sum(row["episode_score"] for row in episodes) / len(episodes),
        layouts=sorted(identities),
        evidence_sha256=digest(episodes),
    )


def propose_skill_choice(task, assessments, *, minimum_score, code_policy=None):
    """Return a candidate skill_choice, never a deployment decision or test-set fallback."""
    if isinstance(minimum_score, bool) or not 0 <= minimum_score <= 1:
        raise ValueError("minimum_score must lie within [0, 1]")
    if len(assessments) != 2 or {a["name"] for a in assessments} != set(VLA_EXPERTS):
        raise ValueError(
            "Both VLA assessments are required before developing a code policy skill"
        )
    if (
        any(a["task"] != task for a in assessments)
        or assessments[0]["layouts"] != assessments[1]["layouts"]
    ):
        raise ValueError("VLA assessments must use paired development layouts")
    qualified = [a for a in assessments if a["mean_score"] >= minimum_score]
    if qualified:
        selected = max(
            qualified, key=lambda a: (a["mean_score"], -VLA_EXPERTS.index(a["name"]))
        )
        return dict(
            task=task,
            state="skill_choice_candidate",
            name=selected["name"],
            revision=selected["revision"],
            evidence=assessments,
        )
    if code_policy is None:
        return dict(
            task=task, state="needs_code_policy_development", evidence=assessments
        )
    if not code_policy.get("name") or not code_policy.get("revision"):
        raise ValueError("Code policy must have a pinned identity")
    return dict(task=task, state="skill_choice_candidate", **code_policy, evidence=assessments)
