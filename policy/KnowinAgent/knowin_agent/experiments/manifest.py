"""Append-only run manifest: who the run is, under which switches and code, on which scene, and what it cost.

``manifest.json`` is written once. Corrections are full manifests written as ``manifest-<n>.json``
(n = 1, 2, ...) that name the file they amend; readers take the highest n.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Optional

from knowin_agent.backends.robodojo.layouts import SEED_STRIDE
from knowin_agent.core.recorder import read_table
from knowin_agent.experiments import switches as registry
from knowin_agent.experiments.splits import SPLITS

SCHEMA = "knowin_agent.manifest/1"
MANIFEST = "manifest.json"
PHASES = ("practice", "review", "validate", "test", "curate")
SCENE_SPLITS = SPLITS + ("official",)
IDENTITY_KEYS = ("run_id", "task", "phase", "parent_run", "pipeline_id", "round", "experiment")
UNKNOWN = "unknown"

EXEC_TOOLS = {"exec", "execute", "exec_code", "run_program"}
STALL_MARKERS = {"stalled", "stuck", "stall"}
OVER_BUDGET_MARKERS = {"over_budget", "budget_refused", "refused_budget", "budget_refusal"}
RUN_TABLES = ("episode", "turns", "tool_calls", "llm_calls", "events", "delegations", "skill_runs")

_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@+-]*")
_AMENDMENT = re.compile(r"manifest-(\d+)\.json")
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class ManifestError(ValueError):
    pass


def harness_digest(root: Optional[Path] = None) -> str:
    """sha256 over every ``knowin_agent/**/*.py`` (relative path and bytes, sorted by path)."""
    root = Path(root) if root else _PACKAGE_ROOT
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts):
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


_harness_digest = harness_digest


def tree_digest(path: Path) -> Optional[str]:
    """sha256 over a directory's files (for skills/ and lessons/ snapshots); None if it does not exist."""
    path = Path(path)
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(file.relative_to(path).as_posix().encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(file.read_bytes()).digest())
    return digest.hexdigest()


def manifest_paths(run_dir: Path) -> list[Path]:
    """The run's manifest and its amendments, oldest first."""
    run_dir = Path(run_dir)
    found = []
    if (run_dir / MANIFEST).is_file():
        found.append((0, run_dir / MANIFEST))
    for path in run_dir.glob("manifest-*.json"):
        match = _AMENDMENT.fullmatch(path.name)
        if match:
            found.append((int(match.group(1)), path))
    return [p for _, p in sorted(found)]


def read_manifest(run_dir: Path) -> dict:
    """The newest manifest (or amendment) of a run."""
    paths = manifest_paths(run_dir)
    if not paths:
        raise FileNotFoundError(f"no manifest in {run_dir}")
    return json.loads(paths[-1].read_text(encoding="utf-8"))


# -- collect: derive results, costs and process statistics from the run's JSONL tables ----------

def _markers(result: Any) -> set[str]:
    if not isinstance(result, dict):
        return set()
    out = set()
    for key in ("status", "stop_reason", "interruption", "refused"):
        value = result.get(key)
        if isinstance(value, dict):
            value = value.get("kind") or value.get("status")
        if isinstance(value, str):
            out.add(value.split(":")[0])
    return out


def _async_tracks(call: dict) -> int:
    count = 1 if isinstance(call.get("arguments"), dict) and call["arguments"].get("wait") is False else 0
    result = call.get("result")
    tracks = result.get("tracks") if isinstance(result, dict) else None
    if isinstance(tracks, list):
        count += sum(1 for t in tracks if isinstance(t, dict) and (t.get("async") is True or t.get("wait") is False))
    return count


def _end_row(run_dir: Path) -> dict:
    ends = [row for row in read_table(run_dir, "episode") if row.get("phase") == "end"]
    if ends:
        return ends[-1]
    path = Path(run_dir) / "result.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _tokens(run_dir: Path, end: dict) -> Any:
    calls = read_table(run_dir, "llm_calls")
    totals = Counter()
    if calls:
        for row in calls:
            usage = row.get("usage") or {}
            totals["input"] += int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
            totals["output"] += int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
            details = usage.get("prompt_tokens_details") or {}
            totals["cached"] += int(details.get("cached_tokens") or usage.get("cached_tokens") or 0)
    else:
        for scope in (end.get("usage") or {}).values():
            if isinstance(scope, dict):
                totals["input"] += int(scope.get("prompt_tokens") or 0)
                totals["output"] += int(scope.get("completion_tokens") or 0)
    if not any(totals.values()):
        return UNKNOWN
    return {"input": totals["input"], "cached": totals["cached"] if calls else UNKNOWN, "output": totals["output"]}


def collect(run_dir: Path) -> dict:
    """``{'results', 'costs', 'process'}`` derived from tool_calls, delegations, events, episode, llm_calls."""
    run_dir = Path(run_dir)
    calls = read_table(run_dir, "tool_calls")
    events = read_table(run_dir, "events")
    delegations = read_table(run_dir, "delegations")
    end = _end_row(run_dir)

    by_tool, by_scope, failed = Counter(), Counter(), 0
    stalls = refusals = async_tracks = executions = 0
    for call in calls:
        by_tool[str(call.get("tool"))] += 1
        by_scope[str(call.get("scope") or "task")] += 1
        failed += call.get("ok") is False
        markers = _markers(call.get("result")) | ({call["status"]} if isinstance(call.get("status"), str) else set())
        stalls += bool(markers & STALL_MARKERS)
        refusals += bool(markers & OVER_BUDGET_MARKERS)
        async_tracks += _async_tracks(call)
        executions += call.get("tool") in EXEC_TOOLS
    event_kinds = Counter(str(e.get("kind")) for e in events)
    stalls += sum(n for kind, n in event_kinds.items() if kind in STALL_MARKERS)
    refusals += sum(n for kind, n in event_kinds.items() if kind in OVER_BUDGET_MARKERS)
    async_tracks += event_kinds.get("async_track", 0)
    returns = Counter(str((row.get("report") or {}).get("status")) for row in delegations if row.get("phase") == "return")

    evaluation = end.get("evaluation_host_only") or {}
    budget = end.get("budget") or evaluation.get("budget") or {}
    tiers = evaluation.get("tiers", evaluation.get("tier"))
    results = {"score": evaluation.get("score"), "success": evaluation.get("success"),
               "excluded": bool(evaluation.get("excluded", False))}
    if tiers is not None:
        results["tiers"] = tiers
    if evaluation.get("error"):
        results["error"] = evaluation["error"]
    costs = {
        "native_steps": evaluation.get("native_steps", budget.get("ticks_used")),
        "native_step_limit": evaluation.get("native_step_limit", budget.get("ticks_limit")),
        "executions": executions,
        "decisions": end.get("decisions") or UNKNOWN,
        "tokens": _tokens(run_dir, end),
        "wall_seconds": end.get("wall_seconds"),
    }
    process = {
        "tool_calls": {"total": len(calls), "failed": failed, "by_tool": dict(sorted(by_tool.items())),
                       "by_scope": dict(sorted(by_scope.items()))},
        "delegations": sum(1 for row in delegations if row.get("phase") == "open"),
        "delegation_returns": dict(sorted(returns.items())),
        "async_tracks": async_tracks,
        "stalls": stalls,
        "over_budget_refusals": refusals,
        "model_errors": event_kinds.get("model_error", 0),
        "stop_reason": end.get("stop_reason"),
    }
    return {"results": results, "costs": costs, "process": process}


# -- write --------------------------------------------------------------------------------------

def _identity(identity: Mapping[str, Any]) -> dict:
    unknown = set(identity) - set(IDENTITY_KEYS)
    if unknown:
        raise ManifestError(f"unknown identity keys {sorted(unknown)}")
    out = {key: identity.get(key) for key in IDENTITY_KEYS}
    if not isinstance(out["run_id"], str) or not _RUN_ID.fullmatch(out["run_id"]):
        raise ManifestError(f"identity.run_id must be a plain identifier, got {out['run_id']!r}")
    if not isinstance(out["task"], str) or not out["task"]:
        raise ManifestError("identity.task is required")
    if out["phase"] not in PHASES:
        raise ManifestError(f"identity.phase must be one of {PHASES}, got {out['phase']!r}")
    if out["round"] is not None and (isinstance(out["round"], bool) or not isinstance(out["round"], int)):
        raise ManifestError(f"identity.round must be an integer or null, got {out['round']!r}")
    return out


def _scene(scene: Optional[Mapping[str, Any]]) -> Optional[dict]:
    if scene is None:
        return None
    out = {"split": scene.get("split"), "seed": scene.get("seed"), "bank": scene.get("bank"),
           "ordinal": scene.get("ordinal")}
    if out["split"] is not None and out["split"] not in SCENE_SPLITS:
        raise ManifestError(f"scene.split must be one of {SCENE_SPLITS}, got {out['split']!r}")
    if isinstance(out["seed"], int) and not isinstance(out["seed"], bool):
        bank, ordinal = divmod(out["seed"], SEED_STRIDE)
        if out["bank"] is None:
            out["bank"] = bank
        if out["ordinal"] is None:
            out["ordinal"] = ordinal
        if (out["bank"], out["ordinal"]) != (bank, ordinal):
            raise ManifestError(f"scene seed {out['seed']} is bank {bank} ordinal {ordinal}, not "
                                f"{out['bank']}/{out['ordinal']}")
    return out


def _switches(values: Mapping[str, Any]) -> dict:
    normalized = registry.validate(values)
    missing = set(registry.REGISTRY) - set(normalized)
    if missing:
        raise ManifestError(f"switches must be fully expanded (use switches.resolve); missing {sorted(missing)}")
    return {name: normalized[name] for name in registry.REGISTRY}


def _default_artifacts(run_dir: Path) -> dict:
    files = {}
    for name in sorted([f"{t}.jsonl" for t in RUN_TABLES] + ["config.json", "result.json"]):
        path = run_dir / name
        if path.is_file():
            files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"files": files, "video": "video" if (run_dir / "video").is_dir() else None}


def write_manifest(run_dir: Path, *, identity: Mapping[str, Any], switches: Mapping[str, Any],
                   profile: Optional[str] = None, harness_digest: Optional[str] = None,
                   library_version: Optional[str] = None, knowledge_source: Optional[str] = None,
                   brief: Optional[Mapping[str, Any]] = None, scene: Optional[Mapping[str, Any]] = None,
                   model: Optional[Mapping[str, Any]] = None, results: Optional[Mapping[str, Any]] = None,
                   costs: Optional[Mapping[str, Any]] = None, process: Optional[Mapping[str, Any]] = None,
                   artifacts: Optional[Mapping[str, Any]] = None, notes: Optional[str] = None,
                   amend: bool = False) -> Path:
    """Write the run's manifest; results/costs/process default to (and override) ``collect(run_dir)``.

    Refuses to overwrite: an existing manifest needs ``amend=True``, which writes ``manifest-<n>.json``.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    existing = manifest_paths(run_dir)
    if existing and not amend:
        raise FileExistsError(f"{existing[-1]} exists; manifests are append-only (pass amend=True)")
    if amend and not existing:
        raise ManifestError(f"nothing to amend in {run_dir}")
    expanded = _switches(switches)
    collected = collect(run_dir)
    manifest = {
        "schema": SCHEMA,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "identity": _identity(identity),
        "profile": profile,
        "switches": expanded,
        "compliance": registry.compliance_report(expanded),
        "harness_digest": harness_digest or _harness_digest(),
        "library_version": library_version,
        "knowledge_source": knowledge_source,
        "brief": dict(brief) if brief is not None else None,
        "scene": _scene(scene),
        "model": dict(model) if model is not None else {"name": expanded["agent.model"],
                                                         "effort": expanded["agent.effort"]},
        "results": {**collected["results"], **(results or {})},
        "costs": {**collected["costs"], **(costs or {})},
        "process": {**collected["process"], **(process or {})},
        "artifacts": dict(artifacts) if artifacts is not None else _default_artifacts(run_dir),
        "notes": notes,
    }
    if amend:
        match = _AMENDMENT.fullmatch(existing[-1].name)
        number = int(match.group(1)) + 1 if match else 1
        manifest["amendment"] = {"number": number, "amends": existing[-1].name}
        path = run_dir / f"manifest-{number}.json"
    else:
        path = run_dir / MANIFEST
    with open(path, "x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return path
