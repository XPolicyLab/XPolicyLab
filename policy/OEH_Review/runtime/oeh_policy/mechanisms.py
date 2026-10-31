"""The condition registry's mechanism keys and what this commit realizes of each (docs/agent-icl-study/
condition-registry.md; experiment-plan-round1.md §8).

A condition of the agent ICL study is declared as one entry of the matrix spec's optional ``designs:`` block: a base
role and a mapping of mechanisms from the closed key set below. oeh-eval compiles one condition per entry and asks
this module, before anything is frozen, whether every mechanism value is one the runtime realizes on this commit.
A value that is not here is refused by name, so a lock never runs the base role's bytes under a design's name: the
same rule as ``contracts.CONTEXT_REALIZED`` for the context manifest, which the ``context`` mechanism delegates to.

Every key is an oeh-eval yaml option and states its question (QUESTIONS); none has a hidden default. Realizing a new
mechanism value means: implement it behind the option, add its acceptance test, then add the value here. Nothing here
changes a spec without a ``designs:`` block.
"""
from __future__ import annotations

import copy
from typing import Any, Mapping

MECHANISM_KEYS = ("schedule", "digest", "handback", "intent.modes", "context", "notes_store", "budgets")
BASE_ROLES = ("oeh_review", "oeh_hybrid_v4", "oeh_slow")
# The mode each base role runs (oeh_eval.matrix_emit.MODES agrees).
BASE_MODES = {"oeh_review": "review", "oeh_hybrid_v4": "review", "oeh_slow": "slow"}

# The question each key answers (R6: a controllable mechanism is a yaml option with its question stated).
QUESTIONS = {
    "schedule": "When is the LLM component called: every chunk boundary (every_chunk), on a computed event until it "
                "hands back (on_stall), after the number of chunks it named itself or an event (segment), or at "
                "every decision with the cadence set by its own execute_steps (own_cadence, slow mode)?",
    "digest": "Does the first packet after unreviewed chunks carry, per unreviewed chunk, the diagnostics a packet "
              "would have had (per_chunk), bounded whole-interval evidence (event_summary), or counts only (none)? (M2 against M2-c1)",
    "handback": "How does the LLM component give control back: when it accepts a full OpenWAM chunk judged "
                "progressing and aligned (accepted_full_chunk), or by an explicit handback{reason} (explicit, G12)?",
    "intent.modes": "Which intents may the LLM component issue in slow mode: delegate to the VLA/WAM model for k steps "
                    "(vla) and/or the bounded primitives (move_relative, gripper, wait, home), with explicit "
                    "pose_control for pose targets? (M3 against M3-c1; "
                    "what M3-c1 keeps besides vla is P7, the user's)",
    "context": "Which declared context levels does the LLM component read (review.context switches: task.note and "
               "recipe_file, state.embodiment and embodiment_file, behavior.lessons_file, ...; each a manifest "
               "switch realized through contracts.CONTEXT_REALIZED)?",
    "notes_store": "Does the LLM component read notes written on disjoint exploration episodes, from a store that is "
                   "read-only and sha256-hashed during the scored round (read_only), and at which scope? (M5 against "
                   "M5-c2; who writes the notes is P10, the user's)",
    "budgets": "Which of the three budgets does the condition set: execution (native steps: the case's limit), "
               "llm_calls (decisions per episode; the same cap for all or per design is P14, user) and wall (the "
               "campaign attempt wall)?",
}

# What this commit realizes, per key and base role: the value strings a design may declare. A key missing for a role
# means the role has no such mechanism (declaring it is refused). The realization sets grow one value at a time, each
# with its acceptance test (tests/test_mechanisms.py names them).
# Acceptance tests of the values added on 2026-10-18 (docs/agent-icl-study/acceptance/):
#   schedule oeh_review segment: tests/test_segment_schedule.py (G13). schedule oeh_slow own_cadence and intent.modes
#   oeh_slow: tests/test_slow_role.py (G4: one call per decision at the cadence of its own execute_steps, the brain
#   row's settings read from cfg["brain"]; the schema shows and admission takes only the listed modes).
#   digest per_chunk: tests/test_interval_digest.py (G5; under every_chunk it is accepted and carries []).
#   notes_store read_only: tests/test_interval_digest.py::test_every_chunk_digest_is_empty_and_the_notes_store_* and
#   ::test_a_changed_or_layout_scoped_store_refuses_the_start (G7 read path: loaded at its sha256 at start,
#   task_notes on decision 0, identity.json records path, sha256 and count; scope layout is refused at start).
MECHANISMS_REALIZED: dict[str, dict[str, frozenset[str]]] = {
    "schedule": {"oeh_review": frozenset({"every_chunk", "segment"}), "oeh_hybrid_v4": frozenset({"on_stall"}),
                 "oeh_slow": frozenset({"own_cadence"})},
    "digest": {"oeh_review": frozenset({"none", "per_chunk", "event_summary"}), "oeh_hybrid_v4": frozenset({"none", "per_chunk", "event_summary"})},
    "handback": {"oeh_hybrid_v4": frozenset({"accepted_full_chunk"}), "oeh_review": frozenset()},
    "intent.modes": {"oeh_slow": frozenset({"vla", "move_relative", "gripper", "wait", "home", "pose"})},
    "notes_store": {"oeh_review": frozenset({"read_only"}), "oeh_slow": frozenset({"read_only"})},
}
# budgets: the values each sub-key may take. execution "native" = the case's pinned native step limit (never moved by
# a design); wall "campaign" = defaults.wall_seconds; llm_calls a positive integer or explicit null (no cap).
BUDGET_KEYS = ("execution", "llm_calls", "wall")
BUDGETS_REALIZED = {"execution": frozenset({"native"}), "wall": frozenset({"campaign"})}
# Where llm_calls lands in the frozen config, per base role and schedule.
LLM_CALL_FIELDS = {("oeh_review", "every_chunk"): "review.max_decisions",
                   ("oeh_review", "segment"): "review.max_decisions",
                   ("oeh_hybrid_v4", "on_stall"): "review.on_stall.max_teacher_decisions",
                   ("oeh_slow", "own_cadence"): "brain.max_calls"}
# Schedule values given as a mapping take these settings beside ``name`` (bounds inclusive); a plain string is the
# name alone. segment's bound is P4 (settled by the G13 call trace), so it is required, with no default.
# default_after_chunks: what cadence applies when the LLM component names no next_look (P4's second half).
SCHEDULE_SETTINGS = {"segment": {"max_segment_chunks": (1, 64), "default_after_chunks": (1, 64)}}
NOTES_STORE_KEYS = {"mode", "path", "sha256", "scope"}
NOTES_STORE_SCOPES = ("global", "task", "layout")


class MechanismError(ValueError):
    """A design declares a mechanism value this commit does not realize, or a malformed one."""


def _schedule(role: str, value: Any) -> dict:
    name = value.get("name") if isinstance(value, Mapping) else value
    realized = MECHANISMS_REALIZED["schedule"].get(role, frozenset())
    if not isinstance(name, str) or name not in realized:
        raise MechanismError(f"mechanism schedule={_short(value)} is not realized on this commit for base role "
                             f"{role} (realized: {sorted(realized)}; oeh_policy.mechanisms.MECHANISMS_REALIZED)")
    settings = SCHEDULE_SETTINGS.get(name, {})
    given = {key: item for key, item in value.items() if key != "name"} if isinstance(value, Mapping) else {}
    if set(given) != set(settings):
        raise MechanismError(f"schedule {name} takes exactly {sorted(settings)} beside name, got {sorted(given)}")
    for key, (low, high) in settings.items():
        if type(given[key]) is not int or not low <= given[key] <= high:
            raise MechanismError(f"schedule.{key} must be an integer in [{low}, {high}]")
    if name == "segment" and given["default_after_chunks"] > given["max_segment_chunks"]:
        raise MechanismError("schedule.default_after_chunks cannot exceed max_segment_chunks")
    return {"name": name, **given}


def _enum(key: str, role: str, value: Any) -> str:
    realized = MECHANISMS_REALIZED[key].get(role)
    if realized is None:
        raise MechanismError(f"mechanism {key} has no meaning for base role {role}")
    if not isinstance(value, str) or value not in realized:
        raise MechanismError(f"mechanism {key}={_short(value)} is not realized on this commit for base role {role} "
                             f"(realized: {sorted(realized)}; oeh_policy.mechanisms.MECHANISMS_REALIZED)")
    return value


def _intent_modes(role: str, value: Any) -> list[str]:
    realized = MECHANISMS_REALIZED["intent.modes"].get(role)
    if realized is None:
        raise MechanismError(f"mechanism intent.modes has no meaning for base role {role}")
    if not isinstance(value, list) or not value or any(not isinstance(item, str) for item in value) \
            or len(set(value)) != len(value):
        raise MechanismError("mechanism intent.modes must list distinct mode names")
    unrealized = [item for item in value if item not in realized]
    if unrealized:
        raise MechanismError(f"mechanism intent.modes {unrealized} are not realized on this commit for base role "
                             f"{role} (realized: {sorted(realized)}; oeh_policy.mechanisms.MECHANISMS_REALIZED)")
    return list(value)


def _context(role: str, value: Any) -> dict:
    """The context mechanism is a manifest fragment; realization is contracts.CONTEXT_REALIZED's call (validated in
    full, with the review block, when the design's review block is built)."""
    if role == "oeh_slow":
        from .slow_reference import STATIC_KEYS
        if not isinstance(value, Mapping) or any(namespace not in STATIC_KEYS or not isinstance(fields, Mapping)
                or set(fields) - STATIC_KEYS[namespace] for namespace, fields in value.items()):
            raise MechanismError("slow context supports public task note/recipe and state embodiment only")
        # Full value/file validation occurs on the compiled per-case configuration with its public task name.
        return copy.deepcopy(dict(value))
    if not isinstance(value, Mapping):
        raise MechanismError("mechanism context must be a mapping of review.context namespaces")
    from .contracts import CONTEXT_SWITCHES
    for namespace, switches in value.items():
        if namespace not in CONTEXT_SWITCHES or not isinstance(switches, Mapping):
            raise MechanismError(f"mechanism context names no review.context namespace {namespace!r}")
        unknown = set(switches) - set(CONTEXT_SWITCHES[namespace])
        if unknown:
            raise MechanismError(f"mechanism context.{namespace} names no switch {sorted(unknown)}")
    return copy.deepcopy(dict(value))


def _notes_store(role: str, value: Any) -> dict:
    realized = MECHANISMS_REALIZED["notes_store"].get(role)
    if realized is None:
        raise MechanismError(f"mechanism notes_store has no meaning for base role {role}")
    if not isinstance(value, Mapping) or set(value) != NOTES_STORE_KEYS:
        raise MechanismError(f"mechanism notes_store takes exactly {sorted(NOTES_STORE_KEYS)}")
    if value["mode"] not in realized:
        raise MechanismError(f"mechanism notes_store.mode={_short(value['mode'])} is not realized on this commit for "
                             f"base role {role} (realized: {sorted(realized)}; "
                             "oeh_policy.mechanisms.MECHANISMS_REALIZED)")
    if not isinstance(value["path"], str) or not value["path"]:
        raise MechanismError("mechanism notes_store.path must be the store directory")
    digest = value["sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise MechanismError("mechanism notes_store.sha256 must be the store's sha256 (64 hex digits)")
    if value["scope"] not in NOTES_STORE_SCOPES:
        raise MechanismError(f"mechanism notes_store.scope must be one of {list(NOTES_STORE_SCOPES)}")
    return dict(value)


def _budgets(role: str, schedule: str, value: Any) -> dict:
    if not isinstance(value, Mapping) or set(value) != set(BUDGET_KEYS):
        raise MechanismError(f"mechanism budgets takes exactly {list(BUDGET_KEYS)} (every design states all three)")
    for key, realized in BUDGETS_REALIZED.items():
        if value[key] not in realized:
            raise MechanismError(f"mechanism budgets.{key}={_short(value[key])} is not realized on this commit "
                                 f"(realized: {sorted(realized)})")
    calls = value["llm_calls"]
    if calls is not None and (type(calls) is not int or calls < 1):
        raise MechanismError("mechanism budgets.llm_calls must be a positive integer or explicit null (no cap)")
    if (role, schedule) not in LLM_CALL_FIELDS:
        raise MechanismError(f"no frozen field holds budgets.llm_calls for base role {role} under schedule {schedule}")
    return {"execution": value["execution"], "llm_calls": calls, "wall": value["wall"],
            "llm_calls_field": LLM_CALL_FIELDS[(role, schedule)]}


def check_mechanisms(base_role: str, mechanisms: Any) -> dict[str, Any]:
    """The normalized mechanisms of one design entry, or MechanismError naming the first problem. Required for every
    base role: schedule and budgets. The other keys are optional and refused where the role has no such mechanism."""
    if base_role not in BASE_ROLES:
        raise MechanismError(f"base_role must be one of {list(BASE_ROLES)}, got {base_role!r}")
    if not isinstance(mechanisms, Mapping):
        raise MechanismError("mechanisms must be a mapping")
    unknown = sorted(set(mechanisms) - set(MECHANISM_KEYS))
    if unknown:
        raise MechanismError(f"unknown mechanisms {unknown}; the closed key set is {list(MECHANISM_KEYS)}")
    missing = [key for key in ("schedule", "budgets") if key not in mechanisms]
    if missing:
        raise MechanismError(f"every design states {missing} (schedule: when the LLM component is called; budgets: "
                             "execution, llm_calls, wall)")
    out: dict[str, Any] = {"schedule": _schedule(base_role, mechanisms["schedule"])}
    if "digest" in mechanisms:
        out["digest"] = _enum("digest", base_role, mechanisms["digest"])
    if "handback" in mechanisms:
        out["handback"] = _enum("handback", base_role, mechanisms["handback"])
    if "intent.modes" in mechanisms:
        out["intent.modes"] = _intent_modes(base_role, mechanisms["intent.modes"])
    if "context" in mechanisms:
        out["context"] = _context(base_role, mechanisms["context"])
    if "notes_store" in mechanisms:
        out["notes_store"] = _notes_store(base_role, mechanisms["notes_store"])
    out["budgets"] = _budgets(base_role, out["schedule"]["name"], mechanisms["budgets"])
    return out


def realized_summary() -> dict[str, Any]:
    """What this commit realizes, as JSON (the acceptance record and condition-registry.md quote it)."""
    return {key: {role: sorted(values) for role, values in roles.items()} for key, roles in MECHANISMS_REALIZED.items()} \
        | {"budgets": {key: sorted(values) for key, values in BUDGETS_REALIZED.items()}}


def _short(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 60 else text[:57] + "..."
