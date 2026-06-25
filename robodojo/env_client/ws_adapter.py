"""Synchronous environment-side adapter for RoboDojo policy WebSocket."""

from __future__ import annotations

import asyncio
from typing import Any, cast

from robodojo.protocol.client import PolicyEvalClient, PolicyEvalClientConfig


class RoboDojoModelClient:
    def __init__(
        self,
        *,
        url: str,
        evaluation_id: str,
        trial_id: str,
        action_case_id: str | None = None,
        repeat_index: int | None = None,
        client: Any | None = None,
    ):
        self.action_case_id = action_case_id
        self.trial_id = trial_id
        self.repeat_index = repeat_index
        self._step = 0
        self._loop = asyncio.new_event_loop()
        self._client = client or PolicyEvalClient(
            PolicyEvalClientConfig(url=url, evaluation_id=evaluation_id)
        )
        self._loop.run_until_complete(self._client.connect(handshake=True))

    def call(self, func_name: str | None = None, obs: Any = None, **kwargs: Any) -> Any:
        if func_name == "prepare_case":
            if self.action_case_id is None:
                raise ValueError("prepare_case requires action_case_id")
            response = self._loop.run_until_complete(
                self._client.prepare_case(
                    self.action_case_id,
                    case_meta=obs if isinstance(obs, dict) else None,
                )
            )
            return response.payload.get("result")

        if func_name == "reset":
            self._step = 0
            response = self._loop.run_until_complete(
                self._client.reset(
                    trial_id=self.trial_id,
                    action_case_id=self.action_case_id,
                    repeat_index=self.repeat_index,
                    payload=obs if isinstance(obs, dict) else None,
                )
            )
            return response.payload.get("result")

        if func_name == "update_obs":
            self._loop.run_until_complete(
                self._client.update_obs(
                    cast(dict[str, Any], obs),
                    trial_id=self.trial_id,
                    action_case_id=self.action_case_id,
                    step=self._step,
                )
            )
            return None

        if func_name == "get_action":
            response = self._loop.run_until_complete(
                self._client.infer(
                    cast(dict[str, Any], obs) if obs is not None else None,
                    trial_id=self.trial_id,
                    action_case_id=self.action_case_id,
                    step=self._step,
                )
            )
            self._step += 1
            return response.payload.get("actions")

        if func_name == "update_obs_batch":
            observations = list(obs) if obs is not None else []
            self._loop.run_until_complete(
                self._client.update_obs_batch(
                    cast(list[dict[str, Any]], observations),
                    trial_id=self.trial_id,
                    action_case_id=self.action_case_id,
                    step=self._step,
                )
            )
            return None

        if func_name == "get_action_batch":
            env_idx_list = obs if obs is not None else kwargs.get("env_idx_list")
            if env_idx_list is None:
                raise ValueError("get_action_batch requires env_idx_list")
            response = self._loop.run_until_complete(
                self._client.get_action_batch(
                    list(env_idx_list),
                    trial_id=self.trial_id,
                    action_case_id=self.action_case_id,
                    step=self._step,
                )
            )
            self._step += 1
            return response.payload.get("actions")

        if func_name == "trial_end":
            response = self._loop.run_until_complete(
                self._client.trial_end(
                    trial_id=self.trial_id,
                    action_case_id=self.action_case_id,
                    result=obs if isinstance(obs, dict) else None,
                )
            )
            return response.payload.get("result")

        raise NotImplementedError(f"unsupported RoboDojo model call: {func_name}")

    def close(self) -> None:
        if self._loop.is_closed():
            return
        try:
            self._loop.run_until_complete(self._client.close())
        finally:
            self._loop.close()

    def __enter__(self) -> RoboDojoModelClient:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()
