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

"""Connect the existing Pi0.5 RPC protocol without starting a model service."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from ..errors import VLAConfigError, VLAInferenceError, VLAUnsupportedError
from ..factory import _positive_timeout
from ..types import VLA, VLAPrediction, VLASpec


def _parameters(parameters: Mapping[str, Any]) -> tuple[str, str, float]:
    from roborsi.utils.rpc import parse_endpoint

    if not isinstance(parameters, Mapping) or set(parameters) - {
        "endpoint",
        "embodiment",
        "default_timeout_s",
    }:
        raise VLAConfigError("Pi0.5 parameters must use only declared fields")
    endpoint = parameters.get("endpoint")
    if (
        not isinstance(endpoint, str)
        or not endpoint
        or any(char.isspace() for char in endpoint)
        or any(char in endpoint for char in ("@", "?", "#", "\\"))
    ):
        raise VLAConfigError("endpoint must be [http://|socket://]host:port")
    rest = endpoint.partition("://")[2] if "://" in endpoint else endpoint
    if "/" in rest:
        raise VLAConfigError("endpoint cannot include a URL path")
    try:
        protocol, host, port = parse_endpoint(endpoint)
    except (ValueError, TypeError):
        # The shared parser includes the rejected input in its exception.
        raise VLAConfigError("endpoint must be [http://|socket://]host:port") from None
    if protocol not in {"http", "socket"} or not host or not 1 <= port <= 65535:
        raise VLAConfigError("endpoint requires http/socket and port 1-65535")
    embodiment = parameters.get("embodiment")
    if not isinstance(embodiment, str) or embodiment not in {
        "libero",
        "franka",
        "dual_franka",
    }:
        raise VLAConfigError("embodiment must be libero, franka or dual_franka")
    timeout = _positive_timeout(parameters.get("default_timeout_s", 120.0))
    return endpoint, embodiment, timeout


def _observation(observation: Mapping[str, Any], embodiment: str) -> dict[str, Any]:
    import numpy as np

    if not isinstance(observation, Mapping):
        raise VLAConfigError("observation must be a mapping")
    if "main_images" not in observation or "states" not in observation:
        raise VLAConfigError("observation requires main_images and states")
    result = dict(observation)
    description = result.get("task_descriptions")
    if description is not None and not isinstance(description, str):
        raise VLAConfigError("task_descriptions must be text or None")
    views = {"main_images": (3,)}
    if embodiment == "libero":
        views.update(wrist_images=(3,), extra_view_images=(3,))
    else:
        views["extra_view_images"] = (4,) if embodiment == "dual_franka" else (3, 4)
    for key, dimensions in views.items():
        value = result.get(key)
        required = key == "main_images" or (
            embodiment == "dual_franka" and key == "extra_view_images"
        )
        if value is None and not required:
            continue
        try:
            array = np.asarray(value)
            valid = (
                array.ndim in dimensions
                and array.shape[-1] == 3
                and all(array.shape)
                and array.dtype.kind in "uif"
                and np.isfinite(array).all()
                and (array >= 0).all()
                and (array <= 255).all()
            )
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise VLAConfigError(f"{key} must contain finite RGB images in [0, 255]")
    try:
        states = np.asarray(result["states"])
        valid = (
            states.ndim == 1
            and states.size > 0
            and states.dtype.kind in "uif"
            and np.isfinite(states).all()
            and (np.abs(states) <= np.finfo(np.float32).max).all()
        )
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise VLAConfigError(
            "states must be a nonempty single-environment vector of finite float32 values"
        )
    return result


class _Pi05VLA:
    def __init__(
        self, rpc: Any, client: Any, endpoint: str, embodiment: str, timeout: float
    ):
        self._rpc = rpc
        self._client = client
        self._embodiment = embodiment
        self._timeout = timeout
        self.spec = VLASpec(
            name="pi05",
            observation_format=f"pi05.{embodiment}.single",
            action_format="pi05.action_chunk",
            supports_timeout=True,
            identity={
                "provider": "pi05",
                "endpoint": endpoint,
                "embodiment": embodiment,
                "model": None,
                "checkpoint": None,
            },
        )

    def predict(
        self,
        observation: Mapping[str, Any],
        *,
        instruction: str | None = None,
        options: Mapping[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> VLAPrediction:
        import numpy as np

        from roborsi.utils.rpc import RpcTransportError

        if options is not None and (
            not isinstance(options, Mapping)
            or set(options) - {"mode"}
            or options.get("mode", "eval") != "eval"
        ):
            raise VLAConfigError('Pi0.5 options support only mode="eval"')
        if instruction is not None and not isinstance(instruction, str):
            raise VLAConfigError("instruction must be text or None")
        values = dict(observation) if isinstance(observation, Mapping) else observation
        if instruction is not None and isinstance(values, dict):
            values["task_descriptions"] = instruction
        values = _observation(values, self._embodiment)
        timeout = self._timeout if timeout_s is None else _positive_timeout(timeout_s)
        start = time.monotonic()
        try:
            actions = self._client.predict(values, {"mode": "eval"}, timeout_s=timeout)
        except TimeoutError as exc:
            raise TimeoutError(
                "Pi0.5 inference wait timed out; remote computation may continue"
            ) from exc
        except RpcTransportError as exc:
            cause = exc.__cause__
            if isinstance(cause, TimeoutError) or isinstance(
                getattr(cause, "reason", None), TimeoutError
            ):
                raise TimeoutError(
                    "Pi0.5 inference wait timed out; remote computation may continue"
                ) from exc
            raise
        except (ValueError, TypeError, IndexError) as exc:
            raise VLAInferenceError("Pi0.5 returned an invalid action chunk") from exc
        try:
            actions = np.asarray(actions)
            valid = (
                actions.ndim == 2
                and all(actions.shape)
                and actions.dtype.kind in "uif"
                and np.isfinite(actions).all()
            )
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise VLAInferenceError(
                "Pi0.5 actions must be a nonempty finite numeric matrix"
            )
        return VLAPrediction(actions, {"inference_s": time.monotonic() - start})

    def reset(self) -> None:
        raise VLAUnsupportedError("Pi0.5 RPC inference is stateless and has no reset")

    def close(self) -> None:
        self._rpc.close()


def create(*, parameters: Mapping[str, Any]) -> VLA:
    """Build a Pi0.5 connection using endpoint, embodiment and optional timeout.

    This protocol is RoboRSI RPC, not OpenPI WebSocket/msgpack. Request timeouts
    bound waiting; they do not cancel remote inference. Socket connection setup
    retains the RPC client's separate ten-second connection timeout.
    """
    endpoint, embodiment, timeout = _parameters(parameters)
    from roborsi.robots.components.pi05_vla_client import Pi05VLAClient
    from roborsi.utils.rpc import make_rpc_client

    rpc = make_rpc_client(endpoint, enable_sessions=False)
    try:
        client = Pi05VLAClient(rpc, embodiment=embodiment)
        return _Pi05VLA(rpc, client, endpoint, embodiment, timeout)
    except Exception as creation_error:
        try:
            rpc.close()
        except Exception as cleanup_error:
            raise creation_error from cleanup_error
        raise
