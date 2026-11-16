"""Frozen, pure tool recipes. Generated code never runs in the policy process."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

from orimapi.evaluation.storage import atomic_json, digest
from orimapi.skills.code_skills.align_blocks.sandbox import admit
from .sequence import validate_sequence

SCHEMA = "orimapi.tool-recipe/v1"


def validate_recipe(document):
    required = {"schema", "name", "interface", "arguments_example", "source", "tests"}
    if (
        not isinstance(document, dict)
        or set(document) != required
        or document["schema"] != SCHEMA
    ):
        raise ValueError("Unsupported tool recipe fields/schema")
    name = document["name"]
    if (
        not isinstance(name, str)
        or not name.isidentifier()
        or name.startswith("_")
        or len(name) > 64
    ):
        raise ValueError("Recipe name must be a bounded public identifier")
    if (
        not isinstance(document["interface"], str)
        or not 20 <= len(document["interface"]) <= 4000
    ):
        raise ValueError("Recipe requires a bounded interface description")
    if not isinstance(document["arguments_example"], dict):
        raise ValueError("Recipe arguments example must be an object")
    admit(document["source"])
    tests = document["tests"]
    if not isinstance(tests, list) or not 1 <= len(tests) <= 32:
        raise ValueError("Recipe requires 1..32 data-only test cases")
    names = set()
    for case in tests:
        if not isinstance(case, dict) or set(case) != {"name", "scene", "expected"}:
            raise ValueError("Test cases require name, scene and expected sequence")
        if (
            not isinstance(case["name"], str)
            or not case["name"]
            or case["name"] in names
        ):
            raise ValueError("Test case names must be unique")
        names.add(case["name"])
        if not isinstance(case["scene"], dict) or set(case["scene"]) != {
            "arguments",
            "state",
        }:
            raise ValueError("Test inputs contain only public arguments and state")
        validate_sequence(case["expected"])
    if len(json.dumps(document, allow_nan=False).encode()) > 240000:
        raise ValueError("Recipe exceeds its size limit")
    return deepcopy(document)


def execute_recipe(document, scene):
    document = validate_recipe(document)
    request = json.dumps(
        {"source": document["source"], "scene": scene}, allow_nan=False
    )
    if len(request.encode()) > 480000:
        raise ValueError("Recipe input exceeds sandbox limit")
    worker = Path(__file__).parents[1] / "code_skills/align_blocks/sandbox_worker.py"
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-S", str(worker)],
            input=request,
            text=True,
            capture_output=True,
            timeout=8,
            env={},
            cwd="/tmp",
            close_fds=True,
        )
    except subprocess.TimeoutExpired as error:
        raise ValueError("Tool recipe sandbox timed out") from error
    if result.returncode or len(result.stdout) > 256000:
        raise ValueError("Tool recipe sandbox failed: " + result.stderr[-1500:])
    return validate_sequence(json.loads(result.stdout))


def check_recipe(document, previous=None):
    """Preserve all prior regression cases; tests do not establish task success."""
    document = validate_recipe(document)
    if previous is not None:
        previous = validate_recipe(previous)
        if document["name"] != previous["name"]:
            raise ValueError("A repair must retain the tool name")
        old = {case["name"]: case for case in previous["tests"]}
        new = {case["name"]: case for case in document["tests"]}
        if any(new.get(name) != case for name, case in old.items()):
            raise ValueError(
                "A repair cannot remove or change existing regression cases"
            )
        if document["source"] == previous["source"] or not new.keys() - old.keys():
            raise ValueError("A repair must change code and add a regression case")
        discriminating = False
        for name in new.keys() - old.keys():
            try:
                discriminating |= (
                    execute_recipe(previous, new[name]["scene"])
                    != new[name]["expected"]
                )
            except ValueError:
                discriminating = True
        if not discriminating:
            raise ValueError(
                "New regression cases must expose a failure of the old source"
            )
    for case in document["tests"]:
        if execute_recipe(document, case["scene"]) != case["expected"]:
            raise ValueError("Tool recipe regression failed: " + case["name"])
    return {
        "passed": True,
        "test_count": len(document["tests"]),
        "sha256": digest(document),
        "scope": "Pure contract tests, not native task success",
    }


class RecipeLibrary:
    def __init__(self, documents=None):
        documents = documents or {}
        if not isinstance(documents, dict) or len(documents) > 16:
            raise ValueError("Use at most 16 frozen tool recipes")
        self.documents = {
            name: validate_recipe(value) for name, value in documents.items()
        }
        if any(name != value["name"] for name, value in self.documents.items()):
            raise ValueError("Recipe key and declared name disagree")
        self.sha256 = digest(self.documents)

    def describe(self):
        return {
            name: {
                "interface": row["interface"],
                "arguments_example": row["arguments_example"],
                "sha256": digest(row),
            }
            for name, row in self.documents.items()
        }

    def build(self, name, arguments, state):
        if digest(self.documents) != self.sha256:
            raise ValueError("Frozen tool recipes changed during the episode")
        if name not in self.documents or not isinstance(arguments, dict):
            raise ValueError("Unknown recipe or invalid arguments")
        return execute_recipe(
            self.documents[name], {"arguments": arguments, "state": state}
        )

    def snapshot(self, root):
        if self.documents:
            path = Path(root) / "tool_recipes" / (self.sha256 + ".json")
            if path.exists() and json.loads(path.read_text()) != self.documents:
                raise ValueError("Recipe snapshot changed")
            if not path.exists():
                atomic_json(path, self.documents)
