"""Concrete workflow adapter for fresh layouts, native reset and bound rollouts.

All paths describe installed code/public assets. No historical rollout or test
layout is imported. Native qualification requires executing this runtime, not
merely loading its configuration or passing its software admission checks.
"""

import ast
import os
import secrets
from copy import deepcopy
from pathlib import Path

from PhysicalRSI_core.infra.storage import (
    atomic_json,
    digest,
    file_digest,
    read_json,
    relative_path,
)
from PhysicalRSI_core.self_harness.artifacts import verify_harness

from . import bound_rollout, layouts, native_execution
from .isolated_factory import verify_spec as verify_policy_assembly
from .expert_service import verify_spec as verify_expert
from .policy_service import verify as verify_policy
from .native_preflight import client_files

verify_policy = verify_policy_assembly


class NativeRuntime:
    def __init__(self, configuration):
        self.config = deepcopy(configuration)
        if self.config.get("schema") != "physicalrsi.robodojo.native-runtime/v1":
            raise ValueError("Versioned native runtime configuration required")
        if set(self.config["experts"]) != {"pi05", "pi05-sparse-memory"}:
            raise ValueError("Both trained VLA expert configurations required")
        n = self.config["initialization_batch_size"]
        if type(n) is not int or not 1 <= n <= 10:
            raise ValueError("Initialization batch size must be one to ten")
        workers = self.config.get("generation_workers", 1)
        if type(workers) is not int or not 1 <= workers <= 10:
            raise ValueError("Layout generation workers must be one to ten")
        if not self.config.get("files"):
            raise ValueError("Pinned native runtime dependencies required")
        for field in ("camera_calibration", "robot_calibration"):
            if field in self.config and type(self.config[field]) is not bool:
                raise ValueError(f"{field} must be boolean")
        self._verify()

    def _assembly_spec(self):
        """Resolve the neutral field while accepting old runtime manifests."""
        return self.config["policy_spec"]

    def _verify(self):
        for name, sha in self.config["files"].items():
            path = Path(name)
            if not path.is_absolute() or file_digest(path) != sha:
                raise ValueError("Native runtime dependency changed")

    def identity(self):
        self._verify()
        return dict(
            scope="native_robodojo; qualification requires native evidence",
            configuration_sha256=digest(self.config),
            files=deepcopy(self.config["files"]),
            sampler=layouts.implementation(),
            expert_specs={
                name: digest(entry["spec"])
                for name, entry in self.config["experts"].items()
            },
            policy_spec=digest(self._assembly_spec()),
        )

    def check(self):
        try:
            self._verify()
            if self.config["initialization_policy_name"] != "physicalRSI":
                raise ValueError(
                    "Native initialization must use the physicalRSI adapter"
                )
            for path, sha in client_files(self.config["simulator_root"]).items():
                if self.config["files"].get(path) != sha:
                    raise ValueError("Native client deployment files must be pinned")
            for field in ("simulator_python", "policy_python"):
                if not Path(self.config[field]).is_file() or not os.access(
                    self.config[field], os.X_OK
                ):
                    raise ValueError(
                        "Native and policy interpreters must be executable"
                    )
            verify_policy(self._assembly_spec())
            self.skill_api()
            verify_policy(
                dict(
                    schema="physicalrsi.robodojo.policy-service/v1",
                    config={},
                    files=self.config["policy_files"],
                )
            )
            for name, entry in self.config["experts"].items():
                if entry["spec"]["expert"]["name"] != name:
                    raise ValueError(
                        "Expert identity differs from runtime registration"
                    )
                verify_expert(entry["spec"])
            for path in (
                self.config["source"],
                self.config["assets"],
                self.config["simulator_root"],
            ):
                if not Path(path).is_dir():
                    raise ValueError(
                        "Native configuration/assets/simulator directory missing"
                    )
        except (ValueError, OSError, KeyError, ImportError) as error:
            return dict(ready=False, error=str(error), physical_qualification=False)
        return dict(
            ready=True,
            scope="artifact preflight only; no simulation started",
            initialization_diagnostics=dict(
                camera_calibration=self.config.get("camera_calibration", False),
                robot_calibration=self.config.get("robot_calibration", False),
                mode="reset_only",
            ),
            physical_qualification=False,
        )

    def skill_api(self):
        from .primitive_api import ASSEMBLER, describe

        spec = self._assembly_spec()
        if spec.get("entrypoint") == ASSEMBLER:
            return describe(spec)
        return deepcopy(self.config["skill_api"])

    def _request(self, *, task, split, rows, root):
        return dict(
            simulator_root=self.config["simulator_root"],
            assets_root=self.config["assets"],
            task=task,
            split=split,
            layouts=deepcopy(rows),
            layout_root=str(Path(root).resolve()),
            gpu=self.config["gpu"],
            seed=self.config["simulator_seed"],
            policy_name=self.config["initialization_policy_name"],
            camera_calibration=self.config.get("camera_calibration", False),
            robot_calibration=self.config.get("robot_calibration", False),
            port=1,
            endpoint="ws://127.0.0.1:1",
        )

    def cases(self, *, tasks, count, split, output, comparison=None):
        self._verify()
        root = Path(output).resolve()
        comparison_sha = digest(comparison) if comparison is not None else None
        if split == "validation" and (
            not comparison or not comparison.get("frozen_at")
        ):
            raise ValueError("Validation layouts require a frozen comparison")
        seed = secrets.randbits(63)
        generated = root / "generated"
        layouts.generate_batch(
            self.config["source"],
            self.config["assets"],
            generated,
            tasks=tasks,
            count=count,
            split=split,
            seed=seed,
            comparison_sha256=comparison_sha,
            timeout_s=self.config["generation_timeout_s"],
            workers=self.config.get("generation_workers", 1),
        )
        batch = layouts.verify_batch(
            generated / "manifest.json", split=split, comparison_sha256=comparison_sha
        )
        if any(row.get("eligibility_blocker") for row in batch["entries"]):
            raise ValueError(
                "Generated task requires additional native support/semantic eligibility validation"
            )
        rows = [
            dict(
                task=row["task"],
                split=split,
                file="generated/" + row["file"],
                layout_sha256=row["content_sha256"],
            )
            for row in batch["entries"]
        ]
        for task in tasks:
            group = [row for row in rows if row["task"] == task]
            size = self.config["initialization_batch_size"]
            for start in range(0, len(group), size):
                chunk = group[start : start + size]
                destination = root / "initialization" / task / str(start)
                receipt = native_execution.initialize(
                    self._request(task=task, split=split, rows=chunk, root=root),
                    python=self.config["simulator_python"],
                    environment=self.config["simulator_environment"],
                    output=destination,
                    timeout_s=self.config["initialization_timeout_s"],
                )
                report = read_json(receipt)
                if (
                    report.get("schema") != "physicalrsi.robodojo.initialization/v1"
                    or report["task"] != task
                    or report["split"] != split
                    or len(report["initializations"]) != len(chunk)
                    or any(
                        r.get("state") != "native_reset_checked"
                        or r.get("policy_actions") != 0
                        for r in report["initializations"]
                    )
                    or {r["layout_sha256"] for r in report["initializations"]}
                    != {r["layout_sha256"] for r in chunk}
                ):
                    raise ValueError(
                        "Initialization evidence differs from generated cohort"
                    )
                evidence = {str(Path(receipt).relative_to(root)): file_digest(receipt)}
                for name, sha in report["artifacts"].items():
                    path = destination / relative_path(name)
                    if (
                        path.is_symlink()
                        or not path.resolve().is_relative_to(destination.resolve())
                        or file_digest(path) != sha
                    ):
                        raise ValueError("Initialization artifact escaped or changed")
                    evidence[str(path.relative_to(root))] = sha
                for row in chunk:
                    row.update(validity="passed", validation_artifacts=dict(evidence))
        self._verify()
        atomic_json(root / "cases.json", rows)
        return rows

    def admit(self, candidate, output):
        try:
            self._verify()
            verify_harness(candidate)
            root = Path(candidate["root"])
            if read_json(root / "assets.json") != self.identity():
                raise ValueError("Candidate does not pin this runtime closure")
            for skill in read_json(root / "skills.json").values():
                source = root / relative_path(skill["source"])
                if file_digest(source) != skill["revision"]:
                    raise ValueError("Skill revision differs from source")
                tree = ast.parse(source.read_text())
                if not any(
                    isinstance(node, ast.FunctionDef) and node.name == "policy"
                    for node in tree.body
                ):
                    raise ValueError("Code skill needs a policy function")
            report = dict(
                accepted=True,
                reason="Artifact and syntax admission; generated code executes only in the isolated code-policy runtime",
                physical_qualification=False,
            )
        except (ValueError, OSError, SyntaxError) as error:
            report = dict(
                accepted=False, reason=str(error), physical_qualification=False
            )
        atomic_json(Path(output) / "admission.json", report)
        return report

    def rollout(self, *, candidate, expert, case, layout_root, output):
        return self.rollouts(
            candidate=candidate,
            expert=expert,
            cases=[case],
            layout_root=layout_root,
            output=output,
        )[0]

    def rollouts(self, *, candidate, expert, cases, layout_root, output):
        self._verify()
        if (
            not 1 <= len(cases) <= 10
            or len({(r["task"], r["split"]) for r in cases}) != 1
        ):
            raise ValueError(
                "One task/split and up to ten cases per native cohort required"
            )
        task, split = cases[0]["task"], cases[0]["split"]
        if read_json(Path(candidate["root"]) / "assets.json") != self.identity():
            raise ValueError("Candidate runtime closure differs")
        name = expert["name"]
        if name.startswith("code."):
            entry = dict(
                kind="code_policy", expert=deepcopy(expert),
                spec=self._assembly_spec()
            )
        else:
            configured = self.config["experts"][name]
            if configured["spec"]["expert"] != expert:
                raise ValueError("Requested VLA differs from pinned runtime expert")
            entry = dict(kind="vla", expert=deepcopy(expert), **deepcopy(configured))
        policy = dict(
            schema="physicalrsi.robodojo.policy/v1",
            mode="candidate",
            task=task,
            harness=deepcopy(candidate),
            expected_sha256=verify_harness(candidate),
            split=split,
            resources=self.config["resources"],
            factories=[entry],
        )
        if split == "development":
            policy["expert"] = deepcopy(expert)
        service = dict(
            schema="physicalrsi.robodojo.policy-service/v1",
            config={"physicalrsi": policy},
            files=self.config["policy_files"],
        )
        request = self._request(task=task, split=split, rows=cases, root=layout_root)
        request.update(
            expert=deepcopy(expert),
            harness_sha256=policy["expected_sha256"],
            num_envs=len(cases),
        )
        return bound_rollout.execute(
            request,
            policy_spec=service,
            policy_python=self.config["policy_python"],
            policy_environment=self.config["policy_environment"],
            simulator_python=self.config["simulator_python"],
            simulator_environment=self.config["simulator_environment"],
            output=output,
            startup_timeout_s=self.config["policy_startup_timeout_s"],
            cleanup_timeout_s=self.config["policy_cleanup_timeout_s"],
            timeout_s=self.config["rollout_timeout_s"],
        )


def create(*, configuration, source_directory):
    # All artifact paths are explicit/absolute; task location is not a search path.
    return NativeRuntime(configuration)
