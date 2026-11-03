"""Owned planner process using core service supervision and bounded HTTP RPC."""

import inspect
import math
import time
from pathlib import Path
from threading import Lock

from PhysicalRSI_core.infra.rpc.deadline_http import DeadlineHttpRpcClient
from PhysicalRSI_core.infra.services import Services, ServiceSpec
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest
from .curobo_backend import NoPathFound


def verify(spec):
    if spec.get("schema") != "physicalrsi.planner-service/v1" or not spec.get("files"):
        raise ValueError("Pinned planner service specification required")
    for name, expected in spec["files"].items():
        if file_digest(Path(name)) != expected:
            raise ValueError("Planner dependency changed: " + name)
    return {"planner": digest(spec)}


def check_factory(factory, spec):
    source = inspect.getsourcefile(factory)
    if source is None or str(Path(source).resolve()) not in spec["files"]:
        raise ValueError("Planner factory source is not pinned")


class PlannerService:
    def __init__(
        self,
        spec,
        *,
        python,
        environment,
        output,
        pool=None,
        resources=None,
        startup_timeout_s=180,
    ):
        if not math.isfinite(startup_timeout_s) or startup_timeout_s <= 0:
            raise ValueError("Positive planner startup timeout required")
        self.identity = verify(spec)
        self.joint_names = list(spec["configuration"].get("joint_names", []))
        self.root = Path(output).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        atomic_json(self.root / "spec.json", spec)
        self.group = Services(self.root / "service", pool=pool)
        self.specification = ServiceSpec(
            name="planner",
            expected=self.identity,
            command=(
                str(python),
                "-m",
                "PhysicalRSI_baselines.robodojo.planner_worker",
                str(self.root / "spec.json"),
            ),
            environment=dict(environment, PYTHONDONTWRITEBYTECODE="1"),
            resources=resources or {},
            timeout=startup_timeout_s,
        )
        self.client = None
        self.calls = 0
        self._lock = Lock()

    def __enter__(self):
        self.group.start([self.specification])
        handle = self.group.handles["planner"]
        from PhysicalRSI_core.infra.storage import read_json

        endpoint = read_json(handle["announcement"])["endpoint"]
        self.client = DeadlineHttpRpcClient(endpoint, response_bytes=8 * 1024 * 1024)
        return self

    def plan(self, *, deadline, **arguments):
        return self._invoke("plan", deadline=deadline, **arguments)

    def tool_pose(self, joints, *, deadline):
        return self._invoke("tool_pose", joints=joints, deadline=deadline)

    def collision_spheres(self, joints, *, deadline):
        return self._invoke("collision_spheres", joints=joints, deadline=deadline)

    def _invoke(self, method, *, deadline, **arguments):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Planner already has an active request")
        try:
            return self._request(method, deadline=deadline, **arguments)
        finally:
            self._lock.release()

    def _request(self, method, *, deadline, **arguments):
        if self.client is None:
            raise RuntimeError("Planner service is not running")
        remaining = deadline - time.monotonic()
        if not math.isfinite(remaining) or remaining <= 0:
            raise TimeoutError("Planner request deadline reached")
        self.calls += 1
        receipt = self.root / "requests" / (str(self.calls) + ".json")
        record = dict(
            identity=self.identity, method=method, arguments=arguments, state="started"
        )
        atomic_json(receipt, record)
        try:
            result = self.client.call(
                method,
                kwargs=dict(arguments=arguments, timeout_s=remaining),
                timeout_s=remaining,
            )
            if method == "plan" and result.get("state") == "no_path":
                raise NoPathFound(result["reason"])
            field = "positions" if method == "plan" else "result"
            if set(result) != {"state", field} or result["state"] != "completed":
                raise ValueError("Invalid planner result envelope")
            result = result[field]
            if method != "plan":
                from .primitives import vector
                from .planner_bridge import quaternion

                if method == "tool_pose":
                    if not isinstance(result, list) or len(result) != 2:
                        raise ValueError("Expected position and quaternion")
                    result = [
                        vector(result[0], 3, "tool position"),
                        quaternion(result[1]),
                    ]
                else:
                    if not isinstance(result, list) or not result:
                        raise ValueError("Expected nonempty collision spheres")
                    result = [vector(row, 4, "collision sphere") for row in result]
                    if any(row[3] <= 0 for row in result):
                        raise ValueError("Collision sphere radii must be positive")
            record.update(state="completed", result=result)
            atomic_json(receipt, record)
            return result
        except NoPathFound as error:
            record.update(state="no_path", reason=str(error))
            atomic_json(receipt, record)
            raise
        except BaseException as error:
            record.update(
                state="failed", error=type(error).__name__ + ": " + str(error)
            )
            atomic_json(receipt, record)
            self.close()
            raise

    def close(self):
        self.client = None
        self.group.close()
        atomic_json(self.root / "cleanup.json", dict(state="stopped"))

    def __exit__(self, *args):
        self.close()
