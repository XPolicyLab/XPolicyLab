"""Owned, revision-bound VLA service exposed through the Model interface."""

import math
import uuid
from pathlib import Path

from PhysicalRSI_core.infra.storage import atomic_json

from .expert_probe import validate_actions
from .expert_service import ExpertService


def websocket_client(*, endpoint, timeout_s):
    # This adapter targets the pinned XPolicyLab protocol. Disable its automatic
    # same-request reconnect retry: uncertain inference changes memory state.
    from client_server.ws.model_client import WsModelClient
    from client_server.ws.protocol.client import (
        PolicyEvalClient,
        PolicyEvalClientConfig,
    )

    class NoReplayClient(PolicyEvalClient):
        async def request(self, *args, **kwargs):
            kwargs["_reconnect_attempted"] = True
            return await super().request(*args, **kwargs)

    session = uuid.uuid4().hex
    client = NoReplayClient(
        PolicyEvalClientConfig(
            url=endpoint,
            evaluation_id=session,
            request_timeout_s=timeout_s,
            connect_timeout_s=timeout_s,
            handshake_timeout_s=timeout_s,
            max_connect_attempts=1,
            max_connect_seconds=timeout_s,
            close_timeout_s=2,
        )
    )
    return WsModelClient(
        url=endpoint, evaluation_id=session, trial_id=session, client=client
    )


class ExpertModel:
    def __init__(
        self,
        *,
        spec,
        binding,
        python,
        environment,
        dimensions,
        output,
        timeout_s=120,
        startup_timeout_s=600,
    ):
        if spec["expert"] != {
            key: binding["expert"][key] for key in ("name", "revision")
        }:
            raise ValueError("Service differs from committed expert revision")
        if not dimensions or any(
            type(n) is not int or n < 1 for n in dimensions.values()
        ):
            raise ValueError("Explicit positive action dimensions required")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Positive inference timeout required")
        self.root = Path(output).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.dimensions = dict(dimensions)
        self.client = None
        self.closed = False
        self.failed = False
        self.order = []
        self.environments = set()
        self.calls = 0
        self.service = ExpertService(
            spec,
            python=python,
            environment=environment,
            output=self.root / "service",
            startup_timeout_s=startup_timeout_s,
        )
        atomic_json(
            self.root / "binding.json",
            dict(binding=binding, expert_identity=self.service.identity),
        )
        try:
            self.service.__enter__()
            self.client = websocket_client(
                endpoint=self.service.ready["endpoint"], timeout_s=timeout_s
            )
            identity = self.client.call(func_name="physicalrsi_identity")
            if identity != dict(identity=self.service.identity, expert=spec["expert"]):
                raise ValueError("Connected expert identity mismatch")
        except BaseException:
            self.close()
            raise

    def _call(self, method, payload=None, *, project=lambda value: value):
        if self.closed or self.failed:
            raise RuntimeError("Expert stopped or interrupted; create a new deployment")
        self.calls += 1
        receipt = self.root / "calls" / f"{self.calls:06d}.json"
        atomic_json(receipt, dict(method=method, state="started"))
        try:
            result = project(self.client.call(func_name=method, obs=payload))
            atomic_json(receipt, dict(method=method, state="completed"))
            return result
        except BaseException as error:
            self.failed = True
            atomic_json(
                receipt, dict(method=method, state="uncertain", error=str(error))
            )
            self.close()
            raise

    def update_obs(self, obs):
        self.update_obs_batch([dict(obs, env_idx=obs.get("env_idx", 0))])

    def update_obs_batch(self, obs_list):
        rows = list(obs_list)
        indices = [row["env_idx"] for row in rows]
        if (
            not indices
            or any(type(i) is not int or i < 0 for i in indices)
            or len(indices) != len(set(indices))
            or len(self.environments | set(indices)) > 10
        ):
            raise ValueError("One to ten distinct environment indices required")
        self._call("update_obs_batch", rows)
        self.order = indices
        self.environments.update(indices)

    def get_action(self):
        if len(self.order) != 1:
            raise ValueError("Single action requires one observed environment")
        return self.get_action_batch(self.order)[0]

    def get_action_batch(self, env_idx_list=None):
        indices = list(self.order if env_idx_list is None else env_idx_list)
        if (
            not indices
            or any(type(i) is not int or i not in self.environments for i in indices)
            or len(set(indices)) != len(indices)
        ):
            raise ValueError("Actions require distinct observed environment indices")

        def project(value):
            if not isinstance(value, list) or len(value) != len(indices):
                raise ValueError("Expert omitted batch environments")
            return [validate_actions(actions, self.dimensions) for actions in value]

        return self._call("get_action_batch", indices, project=project)

    def reset(self):
        self._call("reset")
        self.order = []
        self.environments.clear()

    def close(self):
        if self.closed:
            return
        self.failed = True
        # Stop remote execution before releasing the local websocket loop.
        self.service.close()
        if self.client is not None:
            self.client.close()
        self.closed = True


def expert_factory(
    *, spec, python, environment, dimensions, timeout_s=120, startup_timeout_s=600
):
    """Factory for CommittedModel; H must pin the complete expert spec digest."""
    from copy import deepcopy
    from PhysicalRSI_core.infra.storage import digest, read_json
    from PhysicalRSI_core.self_harness.artifacts import verify_harness

    frozen_spec = deepcopy(spec)

    def create(*, harness, binding, output):
        if verify_harness(harness) != binding["harness_sha256"]:
            raise ValueError("Deployment harness identity mismatch")
        root = Path(harness["root"])
        name, revision = (frozen_spec["expert"][key] for key in ("name", "revision"))
        if read_json(root / "foundation.json").get(name) != revision or read_json(
            root / "assets.json"
        ).get("expert_specs", {}).get(name) != digest(frozen_spec):
            raise ValueError("Harness does not pin this expert artifact specification")
        return ExpertModel(
            spec=frozen_spec,
            binding=binding,
            python=python,
            environment=environment,
            dimensions=dimensions,
            output=output,
            timeout_s=timeout_s,
            startup_timeout_s=startup_timeout_s,
        )

    return create
