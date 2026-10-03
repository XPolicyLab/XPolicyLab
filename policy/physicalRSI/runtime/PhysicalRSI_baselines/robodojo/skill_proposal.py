"""Versioned code-policy proposal boundary.

This is the public name for the development loop: a proposer receives only
development feedback and the task's primitive contract, then returns a new
policy source and reusable memory.  The older proposal module remains as a
compatibility import for existing workspaces.
"""

import ast
from pathlib import Path

from PhysicalRSI_core.infra.storage import atomic_json, digest

from .primitive_api import for_task

SCHEMA = "physicalrsi.robodojo.skill-proposal/v1"

# Code skills receive only the declared primitive client.  Keep the release
# boundary explicit even though execution is isolated: a task policy must not
# inspect files, import a simulator package, or evaluate hidden task data.
FORBIDDEN_POLICY_NAMES = frozenset(
    {
        "open",
        "eval",
        "exec",
        "compile",
        "__import__",
        "input",
        "help",
        "dir",
        "globals",
        "locals",
        "vars",
    }
)


def validate_policy_source(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("Code-policy source may not import modules")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_POLICY_NAMES:
            raise ValueError("Code-policy source uses forbidden capability: " + node.id)
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("Code-policy source may not access dunder attributes")
    return tree


def propose(*, task, source, memory, feedback, skill_api, proposer, output):
    if not isinstance(task, str) or not task:
        raise ValueError("Code-policy proposal requires a task")
    if not feedback or any(
        row.get("split") != "development" or row.get("task") != task
        for row in feedback
    ):
        raise ValueError("Code-policy proposal accepts development feedback only")
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
    atomic_json(output / "skill-request.json", request)
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
        raise ValueError(
            "Code-policy proposer must return source, memory and rationale"
        )
    tree = validate_policy_source(candidate["source"])
    if not any(
        isinstance(node, ast.FunctionDef) and node.name == "policy"
        for node in tree.body
    ):
        raise ValueError("Code-policy candidate must define policy(robot, memory)")
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
    atomic_json(output / "skill-proposal.json", artifact)
    return artifact


def verify(artifact, *, task=None):
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
        raise ValueError("Unknown code-policy proposal schema")
    if task is not None and artifact.get("task") != task:
        raise ValueError("Code-policy task identity differs")
    if not isinstance(artifact["task"], str) or not isinstance(
        artifact["memory"], dict
    ):
        raise ValueError("Code-policy task and memory types differ")
    if not isinstance(artifact["rationale"], str) or not isinstance(
        artifact["proposer"], dict
    ):
        raise ValueError("Code-policy rationale and proposer types differ")
    if digest(artifact["source"]) != artifact["source_sha256"]:
        raise ValueError("Code-policy source digest differs")
    if digest(artifact["memory"]) != artifact["memory_sha256"]:
        raise ValueError("Code-policy memory digest differs")
    validate_policy_source(artifact["source"])
    return True
