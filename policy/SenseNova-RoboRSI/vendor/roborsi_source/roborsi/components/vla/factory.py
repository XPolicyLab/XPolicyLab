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

"""Create and serialize access to independently supplied VLA providers."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from threading import Lock
from typing import Any

from .errors import (
    VLABusyError,
    VLAClosedError,
    VLAConfigError,
    VLAInferenceError,
    VLAUnsupportedError,
)
from .types import VLA, VLAConfig, VLAPrediction, VLASpec


def _positive_timeout(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VLAConfigError("timeout_s must be a positive finite number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise VLAConfigError("timeout_s must be a positive finite number") from exc
    if not math.isfinite(value) or value <= 0:
        raise VLAConfigError("timeout_s must be a positive finite number")
    return float(value)


class _ManagedVLA:
    def __init__(self, provider: VLA):
        from roborsi.utils.config_values import json_value, reject_secrets

        spec = getattr(provider, "spec", None)
        if not isinstance(spec, VLASpec):
            raise VLAConfigError("provider.spec must be a VLASpec")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (spec.name, spec.observation_format, spec.action_format)
        ):
            raise VLAConfigError("provider formats and name must be nonempty text")
        if any(
            not isinstance(value, bool)
            for value in (spec.stateful, spec.supports_reset, spec.supports_timeout)
        ):
            raise VLAConfigError("provider capabilities must be booleans")
        if not isinstance(spec.identity, Mapping):
            raise VLAConfigError("provider identity must be a JSON object")
        try:
            identity = dict(spec.identity)
            json_value(identity)
            reject_secrets(identity)
        except ValueError as exc:
            raise VLAConfigError(
                "provider identity must contain safe finite JSON"
            ) from exc
        if any(
            not callable(getattr(provider, method, None))
            for method in ("predict", "reset", "close")
        ):
            raise VLAConfigError("provider must implement predict, reset and close")
        self._provider = provider
        self._spec = spec
        self._lock = Lock()
        self._closed = False
        self._uncertain = False

    @property
    def spec(self) -> VLASpec:
        return self._spec

    @contextmanager
    def _operation(self, *, closing: bool = False) -> Iterator[None]:
        if not self._lock.acquire(blocking=False):
            raise VLABusyError("VLA component has an operation in progress")
        try:
            if self._closed and not closing:
                raise VLAClosedError("VLA component is closed")
            yield
        finally:
            self._lock.release()

    def predict(
        self,
        observation: Mapping[str, Any],
        *,
        instruction: str | None = None,
        options: Mapping[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> VLAPrediction:
        with self._operation():
            if not isinstance(observation, Mapping):
                raise VLAConfigError("observation must be a mapping")
            if instruction is not None and not isinstance(instruction, str):
                raise VLAConfigError("instruction must be text or None")
            if options is not None and not isinstance(options, Mapping):
                raise VLAConfigError("options must be a mapping or None")
            if timeout_s is not None:
                timeout_s = _positive_timeout(timeout_s)
                if not self.spec.supports_timeout:
                    raise VLAUnsupportedError("provider does not support timeout_s")
            if self._uncertain:
                raise VLAInferenceError(
                    "stateful inference timed out; reset or create a new component"
                )
            try:
                prediction = self._provider.predict(
                    observation,
                    instruction=instruction,
                    options=options,
                    timeout_s=timeout_s,
                )
            except TimeoutError:
                self._uncertain = self.spec.stateful
                raise
            if not isinstance(prediction, VLAPrediction):
                raise VLAInferenceError("provider must return a VLAPrediction")
            return prediction

    def reset(self) -> None:
        with self._operation():
            if not self.spec.supports_reset:
                raise VLAUnsupportedError("provider does not support reset")
            self._provider.reset()
            self._uncertain = False

    def close(self) -> None:
        with self._operation(closing=True):
            if self._closed:
                return
            self._provider.close()
            self._closed = True


def create_vla(config: VLAConfig) -> VLA:
    """Validate configuration and create an owned, serial-access component.

    Provider factories are trusted Python code. They must release any resources
    acquired before raising; ownership transfers when they return a provider.
    """
    from roborsi.utils.config_values import json_value, reject_secrets

    if not isinstance(config, VLAConfig):
        raise VLAConfigError("config must be a VLAConfig")
    if not isinstance(config.factory, str):
        raise VLAConfigError("factory must be a module:function reference")
    module, separator, symbol = config.factory.partition(":")
    if (
        not separator
        or not all(part.isidentifier() for part in module.split("."))
        or not symbol.isidentifier()
    ):
        raise VLAConfigError("factory must be a module:function reference")
    if not isinstance(config.parameters, Mapping):
        raise VLAConfigError("parameters must be a finite JSON object")
    try:
        parameters = dict(config.parameters)
        encoded = json_value(parameters)
        reject_secrets(parameters)
    except ValueError as exc:
        raise VLAConfigError("parameters must contain safe finite JSON values") from exc

    from roborsi.evaluation.service_host import load_factory

    try:
        factory = load_factory(config.factory)
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        raise VLAConfigError("cannot resolve VLA provider factory") from exc
    provider = factory(parameters=json.loads(encoded))
    try:
        return _ManagedVLA(provider)
    except Exception as creation_error:
        close = getattr(provider, "close", None)
        if callable(close):
            try:
                close()
            except Exception as cleanup_error:
                raise creation_error from cleanup_error
        raise
