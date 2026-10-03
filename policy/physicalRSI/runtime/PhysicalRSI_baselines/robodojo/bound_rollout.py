"""Bind a frozen policy service to native execution and publish joint evidence.

This orchestration owns the outer policy service; native_execution owns the
simulator process. Neither code completion nor transport success is a task score.
"""

from copy import deepcopy
from pathlib import Path
from urllib.parse import urlsplit

from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, read_json
from PhysicalRSI_core.self_harness.artifacts import verify_harness

from . import native_execution
from .evidence import episode_receipt, verify_case
from .policy_service import PolicyService, verify as verify_policy


def execute(
    request,
    *,
    policy_spec,
    policy_python,
    policy_environment,
    simulator_python,
    simulator_environment,
    output,
    startup_timeout_s=600,
    cleanup_timeout_s=30,
    timeout_s=1800,
    cancelled=None,
):
    request, policy_spec = deepcopy(request), deepcopy(policy_spec)
    root = Path(output).resolve()
    settings = policy_spec["config"]["physicalrsi"]
    if settings.get("mode") != "candidate":
        raise ValueError("Bound rollout requires an explicitly frozen candidate")
    harness = settings["harness"]
    freeze = verify_harness(harness)
    if (
        freeze != settings["expected_sha256"]
        or freeze != request["harness_sha256"]
        or settings["task"] != request["task"]
        or settings["split"] != request["split"]
    ):
        raise ValueError(
            "Policy and native request differ in frozen task/split/identity"
        )
    layouts = request["layouts"]
    if not layouts or len({row["layout_sha256"] for row in layouts}) != len(layouts):
        raise ValueError("Nonempty distinct fresh layout cohort required")
    for row in layouts:
        if (
            row["task"] != request["task"]
            or row["split"] != request["split"]
            or row.get("validity") != "passed"
            or not row.get("validation_artifacts")
        ):
            raise ValueError("Native initialization evidence required for each layout")
        verify_case(row, request["layout_root"])
    if root.exists() and any(root.iterdir()):
        raise ValueError("Bound rollout output must be empty")
    root.mkdir(parents=True, exist_ok=True)
    settings["output"] = str(root / "model")
    atomic_json(root / "requested.json", request)
    try:
        with PolicyService(
            policy_spec,
            python=policy_python,
            environment=policy_environment,
            output=root / "policy-server",
            startup_timeout_s=startup_timeout_s,
            cleanup_timeout_s=cleanup_timeout_s,
        ) as service:
            identity = service.ready["model_identity"]
            if (
                identity["harness_sha256"] != freeze
                or identity["task"] != request["task"]
                or identity["execution_mode"] != request["split"]
                or identity["configuration_sha256"] != digest(settings)
                or identity["expert"] != request["expert"]
            ):
                raise ValueError(
                    "Owned policy identity differs from native execution request"
                )
            endpoint = urlsplit(service.ready["endpoint"])
            if (
                endpoint.scheme != "ws"
                or endpoint.hostname != "127.0.0.1"
                or not endpoint.port
            ):
                raise ValueError("Expected owned localhost WebSocket endpoint")
            request.update(
                endpoint=service.ready["endpoint"],
                host=endpoint.hostname,
                port=endpoint.port,
                policy_name="physicalRSI",
                policy_service_identity=service.identity,
            )
            atomic_json(root / "bound-request.json", request)
            receipts = native_execution.execute(
                request,
                python=simulator_python,
                environment=simulator_environment,
                output=root / "native",
                timeout_s=timeout_s,
                cancelled=cancelled,
            )
        # Only publish outer receipts after both process lifecycles have ended.
        if verify_policy(policy_spec) != service.identity:
            raise ValueError("Policy service dependencies changed during rollout")
        if verify_harness(harness) != freeze:
            raise ValueError("Frozen harness changed during rollout")
        for row in layouts:
            verify_case(row, request["layout_root"])
        if len(receipts) != len(layouts):
            raise ValueError("Native receipt coverage differs from requested cohort")
        by_layout = {read_json(path)["layout_sha256"]: path for path in receipts}
        if set(by_layout) != {row["layout_sha256"] for row in layouts}:
            raise ValueError("Native receipt identities differ from cohort")
        native = []
        for row in layouts:
            path = by_layout[row["layout_sha256"]]
            episode_receipt(
                path,
                directory=root / "native",
                task=request["task"],
                split=request["split"],
                layout_sha256=row["layout_sha256"],
                expert=request["expert"],
                harness_sha256=freeze,
            )
            native.append(read_json(path))
        artifacts = {}
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError("Joint rollout evidence must not contain symlinks")
            if path.is_file():
                artifacts[str(path.relative_to(root))] = file_digest(path)
        staging = root / ".episodes.pending"
        published = []
        for i, (receipt, row) in enumerate(zip(native, layouts)):
            receipt["artifacts"] = artifacts
            path = staging / f"{i}.json"
            atomic_json(path, receipt)
            episode_receipt(
                path,
                directory=root,
                task=request["task"],
                split=request["split"],
                layout_sha256=row["layout_sha256"],
                expert=request["expert"],
                harness_sha256=freeze,
            )
            published.append(root / "episodes" / path.name)
        staging.rename(root / "episodes")
        return published
    except BaseException as error:
        atomic_json(
            root / "failed.json", dict(error=type(error).__name__ + ": " + str(error))
        )
        raise
