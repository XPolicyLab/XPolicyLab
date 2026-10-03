"""One bounded Self-Harness iteration; concrete proposal/evaluation are ports.

The application can invoke subsequent iterations using the committed parent.
Memory, skills and tool use are all edits to the same frozen H. No second loop.
"""

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from PhysicalRSI_core.infra.storage import (
    atomic_json,
    digest,
    file_digest,
    locked,
    read_json,
)

from .artifacts import verify_harness


def now():
    return datetime.now(timezone.utc).isoformat()


class Proposer(Protocol):
    def identity(self) -> dict: ...
    def develop(self, parent: dict, output: Path) -> dict: ...
    def propose(self, parent: dict, feedback: dict, output: Path) -> list[dict]: ...


class Evaluator(Protocol):
    def identity(self) -> dict: ...
    def admit(self, candidate: dict, output: Path) -> dict: ...
    def validation(self, comparison: dict, output: Path) -> dict: ...
    def evaluate(
        self, candidate: dict, comparison: dict, cohort: dict, output: Path
    ) -> dict: ...


class ReconciliationRequired(RuntimeError):
    """A started external effect must be reconciled before any retry."""


class SelfHarness:
    def __init__(
        self,
        root,
        *,
        state,
        proposer: Proposer,
        evaluator: Evaluator,
        selector,
        profile,
        protocol,
        scope,
        max_candidates=3,
    ):
        if type(max_candidates) is not int or not 1 <= max_candidates <= 32:
            raise ValueError("Bounded candidate budget required")
        self.root = Path(root).resolve()
        self.state, self.proposer, self.evaluator, self.selector = (
            state,
            proposer,
            evaluator,
            selector,
        )
        self.config = dict(
            profile=deepcopy(profile),
            protocol=deepcopy(protocol),
            scope=scope,
            max_candidates=max_candidates,
            identities=self.identities(),
        )
        with locked(self.root / ".init.lock"):
            path = self.root / "config.json"
            if path.exists() and read_json(path) != self.config:
                raise ValueError("Frozen experiment configuration changed")
            atomic_json(path, self.config)

    def identities(self):
        return dict(
            proposer=self.proposer.identity(),
            evaluator=self.evaluator.identity(),
            selector=self.selector.identity(),
            core=file_digest(Path(__file__)),
            artifacts=file_digest(Path(__file__).with_name("artifacts.py")),
        )

    def _stable(self):
        if read_json(self.root / "config.json") != self.config:
            raise ValueError("Frozen experiment configuration changed")
        if self.identities() != self.config["identities"]:
            raise ValueError("Executable port or core identity changed")

    def _step(self, name, inputs, call):
        # Adapted from v1 RSI._step: completed work resumes, unknown work blocks.
        self._stable()
        path = self.root / (name + ".json")
        identity = digest(inputs)
        if path.exists():
            record = read_json(path)
            if record["input_sha256"] != identity:
                raise ValueError("Checkpoint inputs changed: " + name)
            if record["state"] != "completed":
                raise ReconciliationRequired(
                    "Reconcile uncertain stage before retry: " + str(path)
                )
            if digest(record["output"]) != record["output_sha256"]:
                raise ValueError("Checkpoint output changed")
            return deepcopy(record["output"])
        record = dict(state="started", input_sha256=identity, started_at=now())
        atomic_json(path, record)
        try:
            output = call()
            self._stable()
        except BaseException as error:
            atomic_json(
                path, dict(record, error=type(error).__name__ + ": " + str(error))
            )
            raise
        atomic_json(
            path,
            dict(
                record,
                state="completed",
                output=output,
                output_sha256=digest(output),
                finished_at=now(),
            ),
        )
        return deepcopy(output)

    def run(self):
        with locked(self.root / ".run.lock"):
            return self._run()

    def _run(self):
        anchor_path = self.root / "parent.json"
        if anchor_path.exists():
            anchor = read_json(anchor_path)
        else:
            anchor = self.state.resolve()
            atomic_json(anchor_path, anchor)
        if anchor["policy"] != self.config or anchor["scope"] != self.config["scope"]:
            raise ValueError("Committed state and experiment policy differ")
        parent = anchor["harness"]
        parent_sha = verify_harness(parent)
        if parent_sha != anchor["freeze_sha256"]:
            raise ValueError("Parent changed")
        done = self.root / "commit.json"
        if done.exists():
            value = read_json(done)
            if value != self.state.resolve():
                raise ValueError(
                    "Interaction completed; current state has since changed"
                )
            return value
        feedback = self._step(
            "develop",
            parent_sha,
            lambda: self.proposer.develop(deepcopy(parent), self.root),
        )
        if feedback.get("split") != "evolve" or not feedback.get("evidence"):
            raise ValueError("Development evidence, including failures, required")
        children = self._step(
            "propose",
            dict(parent=parent_sha, feedback=feedback),
            lambda: self.proposer.propose(
                deepcopy(parent), deepcopy(feedback), self.root
            ),
        )
        if (
            not isinstance(children, list)
            or not 0 <= len(children) <= self.config["max_candidates"]
        ):
            raise ValueError("Candidate pool is malformed or exceeds declared budget")
        if not children:
            if verify_harness(parent) != parent_sha:
                raise ValueError("Parent changed during development")
            current = self.state.resolve()
            if current["revision"] != anchor["revision"]:
                raise ValueError("Committed parent changed during development")
            atomic_json(
                self.root / "retained.json",
                dict(reason="no_changed_candidate", qualification=None),
            )
            return current
        pool = [parent, *children]
        hashes = {h["id"]: verify_harness(h) for h in pool}
        if len(hashes) != len(pool) or len(set(hashes.values())) != len(pool):
            raise ValueError("Duplicate candidate ID or executable content")
        if hashes[parent["id"]] != parent_sha:
            raise ValueError("Parent edited during proposal")
        for child in children:
            if child["components"]["foundation"] != parent["components"]["foundation"]:
                raise ValueError("Foundation F must stay fixed")
            if child.get("parent_sha256") != parent_sha or not all(
                child.get(k)
                for k in ("method", "changes", "environment", "evidence", "costs")
            ):
                raise ValueError("Candidate provenance or costs missing")
            if Path(child["root"]).resolve() == Path(parent["root"]).resolve():
                raise ValueError("Materialize children separately")
        accepted, rejected = [], {}
        for h in pool:
            receipt = self._step(
                "admit_" + h["id"],
                hashes[h["id"]],
                lambda h=h: self.evaluator.admit(deepcopy(h), self.root),
            )
            if receipt.get("freeze_sha256") != hashes[h["id"]] or not receipt.get(
                "evidence"
            ):
                raise ValueError("Admission is not bound to candidate evidence")
            if type(receipt.get("accepted")) is not bool:
                raise ValueError("Admission must complete explicitly")
            if receipt["accepted"]:
                accepted.append(h)
            else:
                if h["id"] == parent["id"]:
                    raise ValueError("Incumbent failed admission")
                if not receipt.get("reason"):
                    raise ValueError("Keep rejection reasons")
                rejected[h["id"]] = receipt
        self._step(
            "admission",
            hashes,
            lambda: dict(rejected=rejected, accepted=[h["id"] for h in accepted]),
        )
        # Even all-rejected pools evaluate the incumbent on fresh inputs only if
        # there is a comparison. Otherwise retain it without a new qualification.
        if len(accepted) == 1:
            if {h["id"]: verify_harness(h) for h in pool} != hashes:
                raise ValueError("Candidate changed during admission")
            atomic_json(
                self.root / "retained.json",
                dict(reason="all_candidates_rejected", qualification=None),
            )
            current = self.state.resolve()
            if current["revision"] != anchor["revision"]:
                raise ValueError("Committed parent changed during admission")
            return current
        comparison = self._step(
            "freeze",
            dict(pool=hashes, config=self.config),
            lambda: dict(
                schema_version=1,
                round_id=self.root.name,
                parent_id=parent["id"],
                frozen_at=now(),
                candidates={h["id"]: hashes[h["id"]] for h in accepted},
                profile=self.config["profile"],
                protocol_sha256=digest(self.config["protocol"]),
                scope=self.config["scope"],
                evaluation_kind=self.config["protocol"]["evaluation_kind"],
            ),
        )
        cohort = self._step(
            "validation",
            comparison,
            lambda: self.evaluator.validation(deepcopy(comparison), self.root),
        )
        if (
            cohort.get("split") != "validation"
            or not cohort.get("admission_evidence")
            or cohort.get("comparison_sha256") != digest(comparison)
        ):
            raise ValueError("Independent validation must bind the frozen pool")
        generated = datetime.fromisoformat(cohort["generated_at"])
        if (
            generated.tzinfo is None
            or generated > datetime.now(timezone.utc)
            or generated <= datetime.fromisoformat(comparison["frozen_at"])
        ):
            raise ValueError("Validation must follow full pool freeze")
        tasks = self.config["profile"]["tasks"]
        if set(cohort["layouts"]) != set(tasks) or any(
            len(cohort["layouts"][t]) != tasks[t]["episodes"] for t in tasks
        ):
            raise ValueError("Validation coverage mismatch")
        layouts = [s for v in cohort["layouts"].values() for s in v]
        if len(set(layouts)) != len(layouts):
            raise ValueError("Duplicate validation inputs")
        self.state.reserve_validation(digest(comparison), layouts)
        results = []
        for h in accepted:
            if {p["id"]: verify_harness(p) for p in pool} != hashes:
                raise ValueError("Executable changed before evaluation")
            result = self._step(
                "evaluate_" + h["id"],
                dict(comparison=comparison, cohort=cohort, candidate=hashes[h["id"]]),
                lambda h=h: self.evaluator.evaluate(
                    deepcopy(h), deepcopy(comparison), deepcopy(cohort), self.root
                ),
            )
            if {p["id"]: verify_harness(p) for p in pool} != hashes:
                raise ValueError("Executable changed during evaluation")
            if result.get("evaluator_revision") != self.config["protocol"]["identity"]:
                raise ValueError("Evaluator protocol changed")
            results.append(result)
        if {h["id"]: verify_harness(h) for h in pool} != hashes:
            raise ValueError("Executable changed during comparison")
        # Recompute selection on resume, rechecking every raw evidence digest.
        decision = self.selector.select(
            comparison, cohort, results, evidence_root=self.root
        )
        saved = self._step(
            "selection",
            dict(comparison=comparison, cohort=cohort, results=results),
            lambda: decision,
        )
        if saved != decision:
            raise ValueError("Selection no longer agrees with evidence")
        survivor = next(h for h in accepted if h["id"] == decision["survivor_id"])
        evidence = {
            str(p): file_digest(p)
            for p in self.root.rglob("*.json")
            if p.name != "commit.json"
        }
        # Trajectories contain binary observations/proposals as well as JSON.
        # Preserve the evaluator's complete evidence closure through inheritance.
        from PhysicalRSI_core.infra.storage import relative_path

        for result in results:
            for episode in result["episodes"]:
                for name, sha in episode["evidence_sha256"].items():
                    path = (self.root / relative_path(name)).resolve()
                    if not path.is_relative_to(self.root) or file_digest(path) != sha:
                        raise ValueError("Evaluation evidence changed before commit")
                    evidence[str(path)] = sha
        committed = self.state.commit(
            anchor["revision"],
            survivor,
            evidence=evidence,
            decision=decision,
            selector=self.selector,
            comparison_root=self.root,
        )
        atomic_json(done, committed)
        return committed
