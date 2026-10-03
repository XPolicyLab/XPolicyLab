"""Shared owned RGB inference worker for depth arrays and target records."""

import importlib
import inspect
import math
import os
import sys
import time
from pathlib import Path
from threading import Lock

import numpy as np

from PhysicalRSI_core.infra.processes import watch_parent_death
from PhysicalRSI_core.infra.rpc.deadline_http import DeadlineHttpRpcClient
from PhysicalRSI_core.infra.rpc.http_rpc import HttpRpcServer
from PhysicalRSI_core.infra.services import Services, ServiceSpec
from PhysicalRSI_core.infra.storage import (
    atomic_json,
    digest,
    file_digest,
    identifier,
    read_json,
)


def kind(spec):
    schemas = {
        "physicalrsi.rgb-depth-service/v1": "depth",
        "physicalrsi.rgb-targets-service/v1": "targets",
    }
    if spec.get("schema") not in schemas:
        raise ValueError("Unknown RGB service schema")
    return schemas[spec["schema"]]


def verify(spec):
    service_kind = kind(spec)
    if not spec.get("files"):
        raise ValueError("Pinned RGB depth specification required")
    if service_kind == "targets":
        queries = spec.get("queries")
        if (
            not isinstance(queries, list)
            or not 1 <= len(queries) <= 8
            or len(set(queries)) != len(queries)
        ):
            raise ValueError("Declare bounded unique target queries")
        for query in queries:
            identifier(query)
    shape = spec["image_shape"]
    if (
        len(shape) != 3
        or any(type(n) is not int or n < 1 for n in shape)
        or shape[2] != 3
        or shape[0] * shape[1] > 1024 * 1024
    ):
        raise ValueError("Declare bounded RGB image shape")
    for name, sha in spec["files"].items():
        if not Path(name).is_absolute() or file_digest(Path(name)) != sha:
            raise ValueError("Depth dependency changed: " + name)
    return {"rgb_" + service_kind: digest(spec)}


def validate_result(value, service_kind, shape, queries=None):
    if service_kind == "depth":
        if (
            not isinstance(value, np.ndarray)
            or value.shape != tuple(shape[:2])
            or value.dtype.kind != "f"
            or not np.isfinite(value).all()
        ):
            raise ValueError("Invalid depth result")
        return
    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "selected", "unresolved", "candidates"}
        or value["schema"] != "physicalrsi.rgb-targets/v1"
    ):
        raise ValueError("Invalid target result schema")
    selected, unresolved, rows = (
        value["selected"],
        value["unresolved"],
        value["candidates"],
    )
    if (
        not isinstance(selected, dict)
        or not isinstance(unresolved, dict)
        or set(selected) & set(unresolved)
        or set(selected) | set(unresolved) != set(queries)
        or not isinstance(rows, list)
        or len(rows) > 32
    ):
        raise ValueError("Invalid target query coverage or capacity")
    for reason in unresolved.values():
        if not isinstance(reason, str) or not 1 <= len(reason) <= 512:
            raise ValueError("Invalid unresolved target reason")
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row)
            != {
                "query",
                "box_xyxy",
                "score",
                "clip_score",
                "appearance_score",
                "rank_score",
                "embedding",
            }
            or row["query"] not in queries
        ):
            raise ValueError("Invalid target candidate")
        box = np.asarray(row["box_xyxy"])
        embedding = np.asarray(row["embedding"])
        if (
            box.shape != (4,)
            or box.dtype.kind not in "ifu"
            or not np.isfinite(box).all()
            or not 0 <= box[0] < box[2] <= shape[1] - 1
            or not 0 <= box[1] < box[3] <= shape[0] - 1
            or embedding.ndim != 1
            or not 1 <= len(embedding) <= 1024
            or embedding.dtype.kind not in "ifu"
            or not np.isfinite(embedding).all()
            or not np.isclose(np.linalg.norm(embedding), 1, atol=1e-4)
        ):
            raise ValueError("Invalid target geometry or embedding")
        from .clip_ranking import rank_scores

        expected = rank_scores([row], [row["clip_score"]], [row["appearance_score"]])[0]
        if not math.isfinite(row["rank_score"]) or not math.isclose(
            row["rank_score"], expected["rank_score"], abs_tol=1e-6
        ):
            raise ValueError("Invalid target rank score")
    for query, row in selected.items():
        if row not in rows or row["query"] != query:
            raise ValueError("Selected target missing from evidence")


def image(value, shape):
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.uint8
        or value.shape != tuple(shape)
    ):
        raise ValueError("RGB array differs from frozen camera shape")
    return value


class RGBService:
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
            raise ValueError("Positive depth startup timeout required")
        self.identity = verify(spec)
        self.kind = kind(spec)
        self.queries = spec.get("queries")
        self.name = "rgb-" + self.kind
        self.shape = tuple(spec["image_shape"])
        self.root = Path(output).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        atomic_json(self.root / "spec.json", spec)
        self.group = Services(self.root / "service", pool=pool)
        self.specification = ServiceSpec(
            name=self.name,
            expected=self.identity,
            command=(str(python), "-m", __name__, str(self.root / "spec.json")),
            environment=dict(environment, PYTHONDONTWRITEBYTECODE="1"),
            resources=resources or {},
            timeout=startup_timeout_s,
        )
        self.client = None
        self.calls = 0
        self.lock = Lock()

    def __enter__(self):
        self.group.start([self.specification])
        ready = read_json(self.group.handles[self.name]["announcement"])
        self.client = DeadlineHttpRpcClient(
            ready["endpoint"],
            response_bytes=max(2_000_000, self.shape[0] * self.shape[1] * 16 + 65536),
        )
        return self

    def predict(self, rgb, *, deadline):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Depth service already has an active request")
        try:
            if self.client is None:
                raise RuntimeError("Depth service is not running")
            remaining = deadline - time.monotonic()
            if not math.isfinite(remaining) or remaining <= 0:
                raise TimeoutError("Depth request deadline reached")
            rgb = image(rgb, self.shape).copy()
            self.calls += 1
            root = self.root / "requests" / str(self.calls)
            root.mkdir(parents=True)
            np.save(root / "rgb.npy", rgb, allow_pickle=False)
            record = dict(
                state="started",
                identity=self.identity,
                input_sha256=file_digest(root / "rgb.npy"),
            )
            atomic_json(root / "receipt.json", record)
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Depth input recording exceeded deadline")
                result = self.client.call(
                    "predict",
                    kwargs=dict(rgb=rgb, timeout_s=remaining),
                    timeout_s=remaining,
                )
                validate_result(result, self.kind, self.shape, self.queries)
                if time.monotonic() >= deadline:
                    raise TimeoutError("RGB result arrived after deadline")
                if self.kind == "depth":
                    artifact = root / "depth.npy"
                    np.save(artifact, result, allow_pickle=False)
                else:
                    artifact = root / "targets.json"
                    atomic_json(artifact, result)
                record.update(state="completed", output_sha256=file_digest(artifact))
                atomic_json(root / "receipt.json", record)
                return result
            except BaseException as error:
                record.update(state="failed", error=str(error))
                atomic_json(root / "receipt.json", record)
                self.close()
                raise
        finally:
            self.lock.release()

    def close(self):
        self.client = None
        self.group.close()
        atomic_json(self.root / "cleanup.json", dict(state="stopped"))

    def __exit__(self, *args):
        self.close()


def worker(path):
    watch_parent_death(lambda: os._exit(70))
    spec = read_json(path)
    identity = verify(spec)
    module, name = spec["entrypoint"].split(":")
    factory = getattr(importlib.import_module(module), name)
    source = inspect.getsourcefile(factory)
    if source is None or str(Path(source).resolve()) not in spec["files"]:
        raise ValueError("Depth factory source must be pinned")
    backend = factory(**spec["configuration"])
    try:
        verify(spec)
        lock = Lock()

        def dispatch(method, args, kwargs):
            if args:
                raise ValueError("Named arguments required")
            if method == "healthz" and not kwargs:
                return {"ready": True}
            if method == "service.describe" and not kwargs:
                return identity
            if method != "predict" or set(kwargs) != {"rgb", "timeout_s"}:
                raise ValueError("Unknown depth request")
            timeout = kwargs["timeout_s"]
            if not math.isfinite(timeout) or timeout <= 0:
                raise ValueError("Positive depth timeout required")
            rgb = image(kwargs["rgb"], spec["image_shape"])
            if not lock.acquire(blocking=False):
                raise RuntimeError("Depth service busy")
            try:
                result = backend.predict(rgb, deadline=time.monotonic() + timeout)
                if kind(spec) == "targets":
                    validate_result(
                        result, kind(spec), spec["image_shape"], spec.get("queries")
                    )
                return result
            finally:
                lock.release()

        server = HttpRpcServer(("127.0.0.1", 0), dispatch)
        atomic_json(
            os.environ["PHYSICALRSI_ENDPOINT_FILE"],
            dict(
                pid=os.getpid(), endpoint="http://127.0.0.1:" + str(server.server_port)
            ),
        )
        try:
            server.serve_forever()
        finally:
            server.server_close()
    finally:
        backend.close()


if __name__ == "__main__":
    worker(sys.argv[1])
