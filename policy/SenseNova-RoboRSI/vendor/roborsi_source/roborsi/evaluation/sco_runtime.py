# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Launch one Benchmark HTTP service per case through SCO."""

from __future__ import annotations

import math
import secrets
import shlex
import time
from collections.abc import Callable
from pathlib import Path

from roborsi.benchmarks.registry import BenchmarkManifestLock, BenchmarkRegistry
from roborsi.evaluation.benchmark_client import BenchmarkClient
from roborsi.evaluation.deadline import CaseCancelled, CaseDeadline, CaseTimeout
from roborsi.evaluation.messages import (
    BenchmarkRuntimeInfo,
    BenchmarkRuntimeRecord,
)
from roborsi.evaluation.sco_jobs import ScoConfig, ScoJob, _sco_command
from roborsi.utils.rpc import RpcTransportError
from roborsi.utils.rpc.http_abort import interrupt_http_calls


class ScoBenchmarkRegistry(BenchmarkRegistry):
    """Validate local task manifests; verify remote assets through the service."""

    _lock_type = BenchmarkManifestLock


class ScoAdapter:
    """Own the HTTP client, remote job, and final runtime checks for one case."""

    def __init__(
        self,
        client: BenchmarkClient,
        job: ScoJob,
        record: BenchmarkRuntimeRecord,
        verify: Callable[[], BenchmarkRuntimeRecord],
        deadline: CaseDeadline | None,
    ) -> None:
        self.client, self.job = client, job
        self.runtime_record = record
        self._verify, self._deadline = verify, deadline
        self._closed = False

    def __enter__(self) -> ScoAdapter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Keep cleanup independent of the case deadline and preserve failures."""
        if self._closed:
            return
        checked = None
        try:
            if self._deadline:
                self._deadline.check()
            checked = self._verify()
            if checked != self.runtime_record:
                raise ValueError("SCO Benchmark runtime changed during evaluation")
        except (CaseCancelled, CaseTimeout):
            checked = None
        finally:
            try:
                self.client.close()
            finally:
                self.job.close()
                self._closed = True
        if checked is not None:
            self.runtime_record = checked.model_copy(
                update={"verified_after_close": True}
            )


class ScoExecutor:
    """Run pinned images with SCO; all Benchmark output is fetched over HTTP."""

    def __init__(
        self,
        config: ScoConfig,
        *,
        command_runner: Callable[[list[str], float], str] = _sco_command,
    ) -> None:
        self.config, self._command_runner = config, command_runner

    def start_adapter(
        self,
        lock: BenchmarkManifestLock,
        case_dir: Path,
        *,
        deadline: CaseDeadline | None = None,
    ) -> ScoAdapter:
        """Submit, discover a worker, check HTTP metadata, and own its cleanup."""
        lock.verify()
        directory = Path(case_dir)
        if not directory.is_dir():
            raise FileNotFoundError(f"case directory is missing: {directory}")
        manifest = lock.as_dict()
        command = shlex.join(manifest["adapter_command"])
        job = ScoJob(
            self.config, directory / "sco-job.json", command_runner=self._command_runner
        )
        token = secrets.token_urlsafe(32)
        client = None
        startup = CaseDeadline(self.config.startup_timeout_s)

        def timeout(limit=30.0):
            return min(
                startup.remaining(limit),
                deadline.remaining(limit) if deadline else limit,
            )

        try:
            job.submit(
                lock.image_reference,
                command,
                token=token,
                deadline=deadline,
                startup_deadline=startup,
            )
            endpoint = job.wait_endpoint(
                manifest["service_port"], deadline=deadline, startup_deadline=startup
            )
            client = BenchmarkClient(endpoint, token=token)
            while True:
                try:
                    with interrupt_http_calls(deadline.check if deadline else None):
                        response = client.transport.call(
                            "healthz", timeout_s=timeout(5)
                        )
                    if (
                        response.get("status") != "ok"
                        or response.get("busy") is not False
                    ):
                        raise ValueError(
                            "new SCO Benchmark service is not idle and healthy"
                        )
                    break
                except RpcTransportError:
                    time.sleep(min(self.config.poll_interval_s, timeout()))
            with interrupt_http_calls(deadline.check if deadline else None):
                initial = self._record(
                    lock, job, client, command, timeout(self.config.command_timeout_s)
                )
        except BaseException:
            try:
                if client:
                    client.close()
            finally:
                job.close()
            raise

        def verify():
            with interrupt_http_calls(deadline.check if deadline else None):
                return self._record(
                    lock,
                    job,
                    client,
                    command,
                    deadline.remaining(self.config.command_timeout_s)
                    if deadline
                    else self.config.command_timeout_s,
                )

        return ScoAdapter(client, job, initial, verify, deadline)

    def _record(self, lock, job, client, command, timeout_s):
        lock.verify()
        manifest = lock.as_dict()
        budget = CaseDeadline(timeout_s)
        actual = job.describe(budget.remaining())
        roles = actual.get("roles", [])
        workers = job.workers(budget.remaining())
        if (
            len(roles) != 1
            or actual["state"] not in {"RUNNING", "QUEUEING"}
            or len(workers) != 1
            or workers[0]["phase"] != "Running"
        ):
            raise ValueError("SCO Benchmark requires one running role")
        role = roles[0]
        if (
            role.get("image_path") != lock.image_reference
            or role.get("startup_script") != command
            or role.get("total_replicas") != 1
        ):
            raise ValueError(
                "SCO image, command or replica count differs from manifest"
            )
        pool = actual.get("resource_pool", {})
        if (pool.get("name"), pool.get("vpc_id"), pool.get("zone")) != (
            self.config.cluster,
            self.config.vpc_id,
            self.config.zone,
        ):
            raise ValueError("SCO resource pool or network differs from configuration")
        specs = role.get("resource_spec", [])
        if (
            len(specs) != 1
            or specs[0].get("name") != self.config.worker_spec
            or specs[0].get("replicas") != 1
        ):
            raise ValueError("SCO worker specification differs from configuration")
        limits = specs[0].get("limits", {})
        resources = manifest["resources"]
        scheduling = actual.get("scheduling", {})
        if (scheduling.get("priority"), scheduling.get("quota_type")) != (
            self.config.priority,
            self.config.quota_type.upper(),
        ):
            raise ValueError("SCO scheduling differs from configuration")
        cpu, memory = float(limits.get("cpu", 0)), float(limits.get("memory", 0))
        if (
            not math.isfinite(cpu)
            or not math.isfinite(memory)
            or cpu < resources["cpus"]
            or memory * 1024 < resources["memory_mb"]
            or int(limits.get("nvidia.com/gpu", 0)) < resources["gpus"]
        ):
            raise ValueError(
                "SCO worker does not satisfy manifest resource requirements"
            )
        expected_mounts = set()
        for mount in self.config.mounts:
            source, destination = mount.split(":", 1)
            volume, _, subdir = source.partition("/")
            expected_mounts.add((volume, subdir, destination))
        mounts = actual.get("mount", [])
        if (
            len(mounts) != len(expected_mounts)
            or any(m.get("type") != "PV_AFS" for m in mounts)
            or {(m.get("id"), m.get("subdir", ""), m.get("mount_path")) for m in mounts}
            != expected_mounts
        ):
            raise ValueError("SCO mounts differ from configuration")
        _, description = client.transport.request(
            "GET", "/v1/benchmark", timeout_s=budget.remaining()
        )
        info = BenchmarkRuntimeInfo.from_dict(description["runtime"])
        if description["benchmark"] != lock.benchmark:
            raise ValueError("SCO service Benchmark identity differs from manifest")
        for field in (
            "benchmark",
            "benchmark_version",
            "benchmark_commit",
            "adapter_version",
            "evaluation",
        ):
            if getattr(info, field) != manifest[field]:
                raise ValueError(f"SCO runtime {field} differs from manifest")
        expected_assets = [
            {"name": a["name"], "kind": a.get("kind", "file"), "sha256": a["sha256"]}
            for a in manifest["assets"]
        ]
        if sorted(description["assets"], key=lambda a: a["name"]) != sorted(
            expected_assets, key=lambda a: a["name"]
        ):
            raise ValueError("SCO service asset hashes differ from manifest")
        tasks = client.tasks(timeout_s=budget.remaining())
        by_identity = {(t.task_id, t.seed): t for t in tasks}
        if len(by_identity) != len(tasks):
            raise ValueError("SCO task catalog contains duplicate identities")
        for identity in lock.list_tasks():
            if by_identity.get((identity.task_id, identity.seed)) != lock.task_message(
                identity.task_id, identity.seed
            ):
                raise ValueError("SCO task catalog differs from manifest")
        return BenchmarkRuntimeRecord(
            info=info,
            # SCO describe reports the submitted image, not the running worker's
            # resolved image ID. Preserve this distinction until attested.
            docker_image_digest=None,
            assets_version=manifest["assets_version"],
            assets=tuple(
                {k: v for k, v in a.items() if k != "path"} for a in manifest["assets"]
            ),
            configuration={
                "transport": "sco_http",
                "verification": "job_spec_and_service_hashes",
                "requested_image": role["image_path"],
                "command": command,
                "resource_pool": pool,
                "resource_spec": specs,
                "mounts": mounts,
                "root_mapping": actual.get("root_mapping"),
                "scheduling": actual.get("scheduling"),
            },
        )
