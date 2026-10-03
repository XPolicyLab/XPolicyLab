"""Small program proposal boundary for RoboDojo.

The artifact records the reusable primitive contract, development-only failure
feedback, code policy and memory together. It is a proposal artifact: an
independent Self-Harness validation still decides whether it can replace the
parent. No validation layouts or hidden scores enter it.
"""

import ast
from pathlib import Path

from PhysicalRSI_core.infra.storage import atomic_json, digest

from .primitive_api import for_task

SCHEMA = "physicalrsi.robodojo.program-proposal/v1"


def propose(*, task, source, memory, feedback, skill_api, proposer, output):
    """Create one auditable code-skill proposal from development feedback."""
    if not isinstance(task, str) or not task:
        raise ValueError("Program proposal requires a task")
    if not feedback or any(
        row.get("split") != "development" or row.get("task") != task
        for row in feedback
    ):
        raise ValueError("Program proposal accepts development feedback only")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    api = for_task(skill_api, task)
    request = dict(
        schema=SCHEMA,
        task=task,
        primitive_api=api,
        development_feedback=feedback,
        parent_source_sha256=digest(source),
        parent_memory_sha256=digest(memory),
        proposer=proposer.identity(),
    )
    atomic_json(output / "program-request.json", request)
    candidate = proposer.propose(
        task=task,
        source=source,
        memory=memory,
        feedback=feedback,
        skill_api=skill_api,
        output=output,
    )
    if set(candidate) != {"source", "memory", "rationale"} or not isinstance(
        candidate["memory"], dict
    ):
        raise ValueError("Program proposer must return source, memory and rationale")
    tree = ast.parse(candidate["source"])
    if not any(
        isinstance(node, ast.FunctionDef) and node.name == "policy"
        for node in tree.body
    ):
        raise ValueError("Program candidate must define policy(robot, memory)")
    artifact = dict(
        schema=SCHEMA,
        task=task,
        source=candidate["source"],
        source_sha256=digest(candidate["source"]),
        memory=candidate["memory"],
        memory_sha256=digest(candidate["memory"]),
        primitive_api_sha256=digest(api),
        development_feedback_sha256=digest(feedback),
        rationale=candidate["rationale"],
        proposer=proposer.identity(),
    )
    atomic_json(output / "program-proposal.json", artifact)
    return artifact


def verify(artifact, *, task=None):
    """Verify an artifact without executing its policy or contacting a model."""
    required = {
        "schema",
        "task",
        "source",
        "source_sha256",
        "memory",
        "memory_sha256",
        "primitive_api_sha256",
        "development_feedback_sha256",
        "rationale",
        "proposer",
    }
    if set(artifact) != required or artifact.get("schema") != SCHEMA:
        raise ValueError("Unknown program proposal schema")
    if task is not None and artifact.get("task") != task:
        raise ValueError("Program proposal task identity differs")
    if not isinstance(artifact["task"], str) or not isinstance(
        artifact["memory"], dict
    ):
        raise ValueError("Program proposal task and memory types differ")
    if not isinstance(artifact["rationale"], str) or not isinstance(
        artifact["proposer"], dict
    ):
        raise ValueError("Program proposal rationale and proposer types differ")
    if digest(artifact["source"]) != artifact["source_sha256"]:
        raise ValueError("Program proposal source digest differs")
    if digest(artifact["memory"]) != artifact["memory_sha256"]:
        raise ValueError("Program proposal memory digest differs")
    ast.parse(artifact["source"])
    return True
