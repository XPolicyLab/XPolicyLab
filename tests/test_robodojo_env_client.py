from __future__ import annotations

from typing import Any

import pytest

from robodojo.env_client import RoboDojoModelClient
from robodojo.protocol.messages import MessageType
from robodojo.protocol.schemas import Frame


class FakePolicyEvalClient:
    def __init__(self):
        self.connected = False
        self.closed = False
        self.prepared_cases: list[dict[str, Any]] = []
        self.resets: list[dict[str, Any]] = []
        self.update_obs_calls: list[dict[str, Any]] = []
        self.update_obs_batch_calls: list[dict[str, Any]] = []
        self.infers: list[dict[str, Any]] = []
        self.get_action_batch_calls: list[dict[str, Any]] = []
        self.trial_ends: list[dict[str, Any]] = []

    async def connect(self, *, handshake: bool = True, evaluation_plan=None) -> None:
        self.connected = True
        if handshake:
            await self.hello(evaluation_plan=evaluation_plan)

    async def hello(self, *, evaluation_plan=None, client_name="robodojo-eval-client", client_version="1.0.0"):
        return Frame(
            message_type=MessageType.HELLO_ACK,
            request_id="hello-1",
            evaluation_id="eval-1",
            payload={"ok": True},
        )

    async def close(self) -> None:
        self.closed = True

    async def prepare_case(
        self, action_case_id: str, case_meta: dict[str, Any] | None = None
    ) -> Frame:
        self.prepared_cases.append(
            {"action_case_id": action_case_id, "case_meta": case_meta}
        )
        return Frame(
            message_type=MessageType.PREPARE_CASE_ACK,
            request_id="prepare-1",
            evaluation_id="eval-1",
            payload={"result": {"prepared": action_case_id}},
        )

    async def reset(self, **kwargs: Any) -> Frame:
        self.resets.append(kwargs)
        return Frame(
            message_type=MessageType.RESET_RESULT,
            request_id="reset-1",
            evaluation_id="eval-1",
            payload={"result": {"ok": True}},
        )

    async def update_obs(self, observation: dict[str, Any], **kwargs: Any) -> Frame:
        self.update_obs_calls.append({"observation": observation, **kwargs})
        return Frame(
            message_type=MessageType.UPDATE_OBS_ACK,
            request_id=f"update-obs-{len(self.update_obs_calls)}",
            evaluation_id="eval-1",
            payload={"ok": True},
        )

    async def update_obs_batch(
        self, observations: list[dict[str, Any]], **kwargs: Any
    ) -> Frame:
        self.update_obs_batch_calls.append({"observations": observations, **kwargs})
        return Frame(
            message_type=MessageType.UPDATE_OBS_BATCH_ACK,
            request_id=f"update-obs-batch-{len(self.update_obs_batch_calls)}",
            evaluation_id="eval-1",
            payload={"ok": True},
        )

    async def infer(self, observation: dict[str, Any] | None = None, **kwargs: Any) -> Frame:
        self.infers.append({"observation": observation, **kwargs})
        return Frame(
            message_type=MessageType.INFER_RESULT,
            request_id=f"infer-{len(self.infers)}",
            evaluation_id="eval-1",
            payload={
                "actions": [
                    {
                        "observation": observation,
                        "step": kwargs["step"],
                    }
                ]
            },
        )

    async def get_action_batch(self, env_idx_list: list[int], **kwargs: Any) -> Frame:
        self.get_action_batch_calls.append({"env_idx_list": env_idx_list, **kwargs})
        return Frame(
            message_type=MessageType.GET_ACTION_BATCH_RESULT,
            request_id=f"get-action-batch-{len(self.get_action_batch_calls)}",
            evaluation_id="eval-1",
            payload={
                "actions": [
                    [{"env_idx": env_idx, "step": kwargs["step"]}]
                    for env_idx in env_idx_list
                ]
            },
        )

    async def trial_end(self, **kwargs: Any) -> Frame:
        self.trial_ends.append(kwargs)
        return Frame(
            message_type=MessageType.TRIAL_END_ACK,
            request_id="trial-end-1",
            evaluation_id="eval-1",
            payload={"result": {"ended": kwargs["trial_id"]}},
        )


def _client(fake: FakePolicyEvalClient) -> RoboDojoModelClient:
    return RoboDojoModelClient(
        url="ws://example.test",
        evaluation_id="eval-1",
        trial_id="trial-1",
        action_case_id="case-1",
        repeat_index=2,
        client=fake,
    )


def test_model_client_forwards_update_obs_and_get_action_separately():
    fake = FakePolicyEvalClient()

    with _client(fake) as client:
        assert fake.connected
        assert client.call(func_name="update_obs", obs={"state": 1}) is None
        actions = client.call(func_name="get_action")

    assert actions == [{"observation": None, "step": 0}]
    assert fake.update_obs_calls == [
        {
            "observation": {"state": 1},
            "trial_id": "trial-1",
            "action_case_id": "case-1",
            "step": 0,
        }
    ]
    assert fake.infers == [
        {
            "observation": None,
            "trial_id": "trial-1",
            "action_case_id": "case-1",
            "step": 0,
        }
    ]
    assert fake.closed


def test_model_client_forwards_intermediate_update_obs_during_action_chunk():
    fake = FakePolicyEvalClient()

    with _client(fake) as client:
        client.call(func_name="update_obs", obs={"state": 0})
        client.call(func_name="get_action")
        client.call(func_name="update_obs", obs={"state": 1})
        client.call(func_name="update_obs", obs={"state": 2})
        client.call(func_name="update_obs", obs={"state": 3})
        client.call(func_name="get_action")

    assert [call["observation"]["state"] for call in fake.update_obs_calls] == [
        0,
        1,
        2,
        3,
    ]
    assert len(fake.infers) == 2
    assert fake.infers[0]["step"] == 0
    assert fake.infers[1]["step"] == 1


def test_model_client_reset_sends_trial_metadata_and_resets_step():
    fake = FakePolicyEvalClient()

    with _client(fake) as client:
        assert client.call(func_name="get_action", obs={"state": 1}) == [
            {"observation": {"state": 1}, "step": 0}
        ]
        assert client.call(func_name="reset", obs={"seed": 7}) == {"ok": True}
        assert client.call(func_name="get_action", obs={"state": 2}) == [
            {"observation": {"state": 2}, "step": 0}
        ]

    assert fake.resets == [
        {
            "trial_id": "trial-1",
            "action_case_id": "case-1",
            "repeat_index": 2,
            "payload": {"seed": 7},
        }
    ]


def test_model_client_maps_prepare_case_and_trial_end():
    fake = FakePolicyEvalClient()

    with _client(fake) as client:
        assert client.call(func_name="prepare_case", obs={"seed": 7}) == {
            "prepared": "case-1"
        }
        assert client.call(func_name="trial_end", obs={"success": True}) == {
            "ended": "trial-1"
        }

    assert fake.prepared_cases == [
        {"action_case_id": "case-1", "case_meta": {"seed": 7}}
    ]
    assert fake.trial_ends == [
        {
            "trial_id": "trial-1",
            "action_case_id": "case-1",
            "result": {"success": True},
        }
    ]


def test_model_client_forwards_update_obs_batch_and_get_action_batch():
    fake = FakePolicyEvalClient()

    with _client(fake) as client:
        assert client.call(
            func_name="update_obs_batch",
            obs=({"env_idx": 0}, {"env_idx": 1}),
        ) is None

        actions = client.call(func_name="get_action_batch", obs=[0, 1])

    assert actions == [
        [{"env_idx": 0, "step": 0}],
        [{"env_idx": 1, "step": 0}],
    ]
    assert fake.update_obs_batch_calls == [
        {
            "observations": [{"env_idx": 0}, {"env_idx": 1}],
            "trial_id": "trial-1",
            "action_case_id": "case-1",
            "step": 0,
        }
    ]
    assert fake.get_action_batch_calls == [
        {
            "env_idx_list": [0, 1],
            "trial_id": "trial-1",
            "action_case_id": "case-1",
            "step": 0,
        }
    ]
    assert fake.infers == []


def test_model_client_rejects_unsupported_legacy_commands():
    fake = FakePolicyEvalClient()

    with _client(fake) as client:
        with pytest.raises(NotImplementedError, match="unsupported RoboDojo"):
            client.call(func_name="set_language", obs="pick up the cube")
