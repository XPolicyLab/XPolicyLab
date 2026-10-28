"""Validate policy execution evidence without deciding task success."""
from __future__ import annotations

import json
from pathlib import Path

from .contracts import digest, file_digest


def validate_attempt(directory: Path, sidecar: Path | None = None) -> dict:
    directory = directory.resolve()
    issues = []
    try:
        seal = json.loads((directory / "seal.json").read_text(encoding="utf-8"))
        inventory = seal["files"]
        if digest(inventory) != seal["sha256"]:
            issues.append("seal inventory hash mismatch")
        actual = {str(p.relative_to(directory)) for p in directory.rglob("*")
                  if p.is_file() and p.name != "seal.json"}
        if actual != set(inventory):
            issues.append("sealed file inventory differs from disk")
        for name, expected in inventory.items():
            path = (directory / name).resolve()
            if not path.is_relative_to(directory) or file_digest(path) != expected:
                issues.append(f"sealed file changed: {name}")
        events = [json.loads(line) for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        kernel = [json.loads(line) for line in (directory / "kernel-events.jsonl").read_text(encoding="utf-8").splitlines()]
        cfg = json.loads((directory / "configuration.json").read_text(encoding="utf-8"))
        identity = json.loads((directory / "identity.json").read_text(encoding="utf-8"))
        if digest(cfg) != identity["configuration_sha256"]:
            issues.append("frozen configuration identity mismatch")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"integrity_ok": False, "issues": [f"unreadable evidence: {exc}"],
                "native_outcome": "not evaluated"}

    def index(kind, key):
        result = {}
        for row in events:
            if row["kind"] != kind:
                continue
            payload = row["payload"]
            identifier = payload[key]
            if identifier in result:
                issues.append(f"duplicate {kind}: {identifier}")
            result[identifier] = payload
        return result

    proposed = index("action_proposed", "index")
    admitted = index("oeh_admission", "trace_id")
    authorized = index("native_action_authorized", "trace_id")
    receipts = index("oeh_receipt", "trace_id")
    executed = index("action_executed", "trace_id")
    kernel_admissions = {(e.get("trace_id"), e["payload"].get("proposal_id"))
                         for e in kernel if e["kind"] == "act/admitted"}
    for trace, admission in admitted.items():
        if (admission["kernel_trace_id"], admission["kernel_exec_id"]) not in kernel_admissions:
            issues.append(f"admission lacks real kernel evidence: {trace}")
        proposal = proposed.get(admission["action_index"])
        if not proposal or digest(proposal["action"]) != admission["proposed_action_sha256"]:
            issues.append(f"admission differs from proposal: {trace}")
    for trace, authorization in authorized.items():
        admission = admitted.get(trace, {})
        if authorization["action_sha256"] != admission.get("proposed_action_sha256"):
            issues.append(f"native authorization lacks matching admission: {trace}")
    for trace, execution in executed.items():
        expected = digest(execution["action"])
        if (expected != execution["sha256"] or
                expected != authorized.get(trace, {}).get("action_sha256") or
                expected != receipts.get(trace, {}).get("executed_action_sha256")):
            issues.append(f"executed action lacks matching authorization and receipt: {trace}")
    if set(receipts) != set(executed):
        issues.append("receipt and execution sets differ")
    unacknowledged = sorted(set(authorized) - set(receipts))
    if unacknowledged:
        issues.append("native submissions remain unacknowledged")
    closed = [row["payload"] for row in events if row["kind"] == "attempt_closed"]
    if len(closed) != 1 or closed[0]["native_steps"] != len(executed):
        issues.append("episode closure or executed step count inconsistent")
    for row in events:
        if row["kind"] == "model_call" and "input_artifact" in row["payload"]:
            payload = row["payload"]
            request = json.loads((directory / payload["input_artifact"]).read_text(encoding="utf-8"))
            if digest(request) != payload["input_sha256"]:
                issues.append("model input differs from recorded request")
        if row["kind"] == "cleanup_failure":
            issues.append("runtime cleanup failed")
    if sidecar is not None:
        native = [json.loads(line) for line in sidecar.read_text(encoding="utf-8").splitlines()]
        submitted = [row["payload"] for row in native if row["kind"] == "native_action_submitted"]
        if len({row["trace_id"] for row in submitted}) != len(submitted):
            issues.append("duplicate native submissions")
        if {row["trace_id"] for row in submitted} != set(authorized):
            issues.append("native submissions differ from authorized commands")
        for row in submitted:
            if digest(row["action"]) != authorized.get(row["trace_id"], {}).get("action_sha256"):
                issues.append("native submitted action differs from authorization")
    return {"integrity_ok": not issues, "issues": issues, "executed_steps": len(executed),
            "unacknowledged": unacknowledged, "native_boundary_checked": sidecar is not None,
            "close_reason": closed[0]["reason"] if closed else None,
            "native_outcome": f"owned by {cfg.get('benchmark', 'robodojo')}; not evaluated here"}
