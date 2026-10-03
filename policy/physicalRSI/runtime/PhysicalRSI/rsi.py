"""Versioned, application-level RSI plans and harness descriptors.

The plan is deliberately data-only.  It selects how an installed task should
search for a capability and carries a reviewed harness description, without
executing user supplied code in the application process.
"""

from copy import deepcopy
from pathlib import Path

from PhysicalRSI_core.infra.storage import digest, read_json


SCHEMA = "physicalrsi.rsi-plan/v1"
HARNESS_SCHEMA = "physicalrsi.harness/v1"
SCHEMES = ("hybrid", "learned_skill", "code_policy")
_ALIASES = {
    "auto": "hybrid",
    "learned-skill": "learned_skill",
    "learned-skill-first": "learned_skill",
    "code-skill": "code_policy",
    "code-skill-policy": "code_policy",
    "code-policy": "code_policy",
}


def canonical_scheme(value):
    if not isinstance(value, str):
        raise ValueError("RSI scheme must be a string")
    scheme = _ALIASES.get(value.strip().lower(), value.strip().lower())
    if scheme not in SCHEMES:
        raise ValueError(
            "Unknown RSI scheme: " + str(value) + "; choose hybrid, learned_skill or code_policy"
        )
    return scheme


def _definition(value):
    if not isinstance(value, dict) or not value:
        raise ValueError("Harness definition must be a non-empty JSON object")
    result = deepcopy(value)
    name = result.pop("name", None)
    if name is not None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Harness name must be a non-empty string")
        result = {"name": name.strip(), **result}
    return result


def design_harness(value):
    """Create a frozen, data-only harness descriptor from a CLI JSON object."""
    if not isinstance(value, dict):
        raise ValueError("Designed harness must be a JSON object")
    definition = value.get("definition", value.get("configuration", value))
    definition = _definition(definition)
    name = definition.get("name", "designed-harness")
    return {
        "schema": HARNESS_SCHEMA,
        "mode": "design",
        "name": name,
        "definition": definition,
        "sha256": digest(definition),
    }


def import_harness(path):
    """Import a JSON harness descriptor and freeze its content by digest."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError("Harness file does not exist: " + str(source))
    value = read_json(source)
    if value.get("schema") != HARNESS_SCHEMA:
        raise ValueError("Expected a physicalrsi.harness/v1 JSON file")
    definition = _definition(value.get("definition"))
    name = value.get("name", definition.get("name", source.stem))
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Imported harness name must be a non-empty string")
    return {
        "schema": HARNESS_SCHEMA,
        "mode": "import",
        "name": name.strip(),
        "source": str(source),
        "definition": definition,
        "sha256": digest(definition),
    }


def normalize_harness(value):
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return import_harness(value)
    if not isinstance(value, dict):
        raise ValueError("Harness must be a JSON object or a descriptor path")
    if value.get("schema") == HARNESS_SCHEMA and "definition" in value:
        definition = _definition(value["definition"])
        expected = digest(definition)
        if value.get("sha256") != expected:
            raise ValueError("Harness revision differs from its definition")
        result = {
            "schema": HARNESS_SCHEMA,
            "mode": value.get("mode"),
            "name": value.get("name", definition.get("name", "harness")),
            "definition": definition,
            "sha256": expected,
        }
        if value.get("source") is not None:
            result["source"] = str(value["source"])
        if result["mode"] not in {"import", "design"}:
            raise ValueError("Harness mode must be import or design")
        return result
    mode = value.get("mode")
    if mode == "import":
        path = value.get("path") or value.get("source")
        if not path:
            raise ValueError("Imported harness requires a path")
        return import_harness(path)
    if mode == "design" or mode is None:
        return design_harness(value)
    raise ValueError("Harness mode must be import or design")


def make_plan(value=None, *, harness=None):
    if value is None:
        value = {}
    if isinstance(value, str):
        value = {"scheme": value}
    if not isinstance(value, dict):
        raise ValueError("RSI plan must be a JSON object")
    scheme = canonical_scheme(value.get("scheme", "hybrid"))
    unknown = set(value) - {"schema", "scheme", "harness", "notes", "budget", "revision"}
    if unknown:
        raise ValueError("Unknown RSI plan field: " + sorted(unknown)[0])
    selected_harness = harness if harness is not None else value.get("harness")
    normalized = {
        "schema": SCHEMA,
        "scheme": scheme,
        "harness": normalize_harness(selected_harness),
    }
    if "notes" in value:
        if not isinstance(value["notes"], str):
            raise ValueError("RSI plan notes must be a string")
        normalized["notes"] = value["notes"]
    if "budget" in value:
        if not isinstance(value["budget"], dict):
            raise ValueError("RSI plan budget must be a JSON object")
        normalized["budget"] = deepcopy(value["budget"])
    normalized["revision"] = digest(normalized)
    return normalized


def verify_plan(value):
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("Expected a physicalrsi.rsi-plan/v1 plan")
    expected = make_plan({key: value[key] for key in ("scheme", "harness", "notes", "budget") if key in value})
    if value != expected:
        raise ValueError("RSI plan revision differs from its content")
    return deepcopy(value)
