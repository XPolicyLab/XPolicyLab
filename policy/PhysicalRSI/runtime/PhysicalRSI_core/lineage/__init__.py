"""Local CAS lineage for a scoped harness, with immutable history and rollback.

This records a selection, not a physical capability certificate. The caller owns
independent evaluation; one host with working POSIX flock owns this store.
"""

from copy import deepcopy
from pathlib import Path

from PhysicalRSI_core.infra.storage import (
    atomic_json,
    digest,
    file_digest,
    locked,
    read_json,
)
from PhysicalRSI_core.self_harness.artifacts import verify_harness


class StateConflict(ValueError):
    pass


class HarnessState:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def _record(self, revision):
        value = read_json(self.root / "history" / (revision + ".json"))
        if digest(value) != revision:
            raise ValueError("Lineage record changed")
        if verify_harness(value["harness"]) != value["freeze_sha256"]:
            raise ValueError("Committed harness changed")
        for path, sha in value.get("evidence", {}).items():
            if file_digest(Path(path)) != sha:
                raise ValueError("Committed evidence changed")
        return value

    def resolve(self):
        revision = read_json(self.root / "current.json")["revision"]
        record = self._record(revision)
        return dict(revision=revision, **record)

    def initialize(self, initial, *, policy, scope):
        with locked(self.root / ".state.lock"):
            if (self.root / "current.json").exists():
                current = self.resolve()
                if current["policy"] != policy or current["scope"] != scope:
                    raise ValueError("State scope or frozen policy changed")
                return current
            body = dict(
                harness=deepcopy(initial),
                freeze_sha256=verify_harness(initial),
                policy=policy,
                scope=scope,
                previous=None,
                action="bootstrap",
                evidence={},
            )
            return self._publish(body)

    def _publish(self, body):
        revision = digest(body)
        path = self.root / "history" / (revision + ".json")
        if path.exists() and read_json(path) != body:
            raise ValueError("Conflicting lineage history")
        atomic_json(path, body)
        atomic_json(self.root / "current.json", dict(revision=revision))
        return dict(revision=revision, **deepcopy(body))

    def commit(
        self,
        expected,
        survivor,
        *,
        evidence,
        decision,
        selector=None,
        comparison_root=None,
    ):
        with locked(self.root / ".state.lock"):
            current = self.resolve()
            freeze = verify_harness(survivor)
            # Idempotent after a crash between CAS and the loop's commit receipt.
            if (
                current["previous"] == expected
                and current.get("decision") == decision
                and current["evidence"] == evidence
                and current["freeze_sha256"] == freeze
            ):
                return current
            if current["revision"] != expected:
                raise StateConflict(
                    "Committed parent changed; preserve this comparison"
                )
            if selector is None or comparison_root is None:
                raise ValueError(
                    "Commit requires the frozen selector and full comparison"
                )
            if selector.identity() != current["policy"]["identities"]["selector"]:
                raise ValueError("Selection policy changed")
            directory = Path(comparison_root).resolve()

            def completed(name):
                path = directory / (name + ".json")
                receipt = read_json(path)
                if (
                    evidence.get(str(path)) != file_digest(path)
                    or receipt.get("state") != "completed"
                    or digest(receipt["output"]) != receipt["output_sha256"]
                ):
                    raise ValueError("Missing or changed comparison receipt")
                return receipt["output"]

            comparison = completed("freeze")
            if (
                comparison["profile"] != current["policy"]["profile"]
                or comparison["protocol_sha256"]
                != digest(current["policy"]["protocol"])
                or comparison["scope"] != current["scope"]
                or comparison["candidates"].get(current["harness"]["id"])
                != current["freeze_sha256"]
            ):
                raise ValueError(
                    "Comparison policy or parent differs from committed state"
                )
            cohort = completed("validation")
            results = []
            for name, sha in comparison["candidates"].items():
                admission = completed("admit_" + name)
                if (
                    admission.get("accepted") is not True
                    or admission.get("freeze_sha256") != sha
                ):
                    raise ValueError("Candidate not admitted")
                result = completed("evaluate_" + name)
                if (
                    result.get("evaluator_revision")
                    != current["policy"]["protocol"]["identity"]
                ):
                    raise ValueError("Evaluator protocol changed")
                results.append(result)
            if (
                completed("selection") != decision
                or selector.select(comparison, cohort, results, evidence_root=directory)
                != decision
            ):
                raise ValueError("Decision differs from full paired evidence")
            if decision["parent_id"] != current["harness"]["id"]:
                raise StateConflict("Decision does not compare the expected parent")
            if (
                decision["survivor_id"] != survivor["id"]
                or decision["survivor_freeze_sha256"] != freeze
            ):
                raise ValueError("Decision and survivor differ")
            if (
                survivor["components"]["foundation"]
                != current["harness"]["components"]["foundation"]
            ):
                raise ValueError("Foundation F changed")
            if not evidence:
                raise ValueError("Commit requires comparison evidence")
            for path, sha in evidence.items():
                if file_digest(Path(path)) != sha:
                    raise ValueError("Comparison evidence changed before commit")
            if decision["decision"] == "retain_parent":
                if freeze != current["freeze_sha256"]:
                    raise ValueError("Retain must preserve the parent")
                return current
            if decision["decision"] != "inherit_child":
                raise ValueError("Unknown decision")
            body = dict(
                harness=deepcopy(survivor),
                freeze_sha256=freeze,
                policy=current["policy"],
                scope=current["scope"],
                previous=expected,
                action="selection",
                evidence=deepcopy(evidence),
                decision=deepcopy(decision),
            )
            result = self._publish(body)
            self._record(result["revision"])
            return result

    def rollback(self, expected):
        with locked(self.root / ".state.lock"):
            current = self.resolve()
            if current["revision"] != expected:
                raise StateConflict("Committed parent changed before rollback")
            if current["previous"] is None:
                raise ValueError("No previous harness")
            previous = self._record(current["previous"])
            return self._publish(
                dict(
                    previous,
                    previous=expected,
                    action="rollback",
                    rollback_to=current["previous"],
                )
            )

    def reserve_validation(self, comparison, layouts):
        with locked(self.root / ".validation.lock"):
            path = self.root / "validation.json"
            values = read_json(path) if path.exists() else {}
            for key, used in values.items():
                if key != comparison and set(used).intersection(layouts):
                    raise ValueError("Validation inputs reused across interactions")
            if comparison in values and values[comparison] != layouts:
                raise ValueError("Validation inputs changed")
            values[comparison] = layouts
            atomic_json(path, values)
