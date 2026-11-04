"""Check frozen prompt/code inputs before and after the upstream native pin step."""

import hashlib
import json

# These are the four declared protocol wording changes at the pinned upstream
# revision. No executable input or other prompt text may change during pinning.
NATIVE_BUDGET_WORDING = (
    ("Native `step(action)` calls have no step-count limit. Actions are counted for telemetry only.",
     "Native `step(action)` calls obey the official task action limit. Check status for native_steps_remaining."),
    ("Native actions have no step-count limit;",
     "Native actions obey the official per-task step-count limit;"),
    ("`native_steps_remaining: null` (unlimited)",
     "`native_steps_remaining` (official remaining actions)"),
    ("There is no action-count budget.", "The official action-count budget applies."),
)
WORDING_PATHS = ("test.md", "primitives/primitives.md")


def validate_source_inputs(source, state, hashes):
    expected = state.get("inputs")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Frozen protocol must contain a nonempty inputs hash map")
    # Upstream hashes excludes a root freeze.json for harness bookkeeping.
    # Such a file has no role in inputs and would otherwise escape verification.
    if (source / "inputs/freeze.json").exists():
        raise ValueError("Unexpected untracked inputs/freeze.json")
    if hashes(source / "inputs") != expected:
        raise ValueError("Frozen source inputs differ from protocol.json")
    return dict(expected)


def verify_pinned_inputs(source, run, state, hashes):
    expected = validate_source_inputs(source, state, hashes)
    changes = []
    for relative in WORDING_PATHS:
        before = (source / "inputs" / relative).read_text()
        after = before
        for old, new in NATIVE_BUDGET_WORDING:
            after = after.replace(old, new)
        expected[relative] = hashlib.sha256(after.encode()).hexdigest()
        changes.append({"path": relative, "before": before, "after": after})
    if hashes(run / "inputs") != expected or (run / "inputs/freeze.json").exists():
        raise ValueError("Pinned inputs differ beyond the allowed native budget wording")
    recorded = json.loads((run / "protocol-description-changes.json").read_text())
    if recorded != changes:
        raise ValueError("Native budget wording change receipt does not match verified inputs")
    return expected
