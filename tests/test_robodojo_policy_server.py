import asyncio
import threading
from typing import Any, cast

import numpy as np
import pytest

from model_template import ModelTemplate
from robodojo.policy_server import PolicyServer
from robodojo.protocol.codec import decode_envelope, encode_frame
from robodojo.protocol.exceptions import ErrorCode
from robodojo.protocol.messages import MessageType
from robodojo.protocol.schemas import Frame


class DummyModel:
    def __init__(self):
        self.prepared = None
        self.observation = None
        self.reset_count = 0
        self.trial_result = None

    def prepare_case(self, case_meta):
        self.prepared = case_meta

    def reset(self):
        self.reset_count += 1

    def update_obs(self, obs):
        self.observation = obs

    def get_action(self):
        return {"arm_joint_state": np.zeros(2, dtype=np.float32)}

    def on_trial_end(self, result):
        self.trial_result = result


def _frame(message_type, payload=None, **kwargs):
    return Frame(
        message_type=message_type,
        request_id=kwargs.pop("request_id", "req-1"),
        evaluation_id=kwargs.pop("evaluation_id", "eval-1"),
        payload=payload or {},
        **kwargs,
    )


def _expect_frame(response: Frame | None) -> Frame:
    assert response is not None
    return response


class QueueWebSocket:
    def __init__(self):
        self.closed = False
        self._incoming: asyncio.Queue[bytes | str | None] = asyncio.Queue()
        self._responses: asyncio.Queue[Frame] = asyncio.Queue()

    def __aiter__(self):
        return self

    async def __anext__(self):
        raw = await self._incoming.get()
        if raw is None:
            raise StopAsyncIteration
        return raw

    async def send(self, message):
        await self._responses.put(decode_envelope(message))

    async def close(self):
        self.closed = True
        await self._incoming.put(None)

    async def feed(self, frame: Frame) -> None:
        await self._incoming.put(encode_frame(frame))

    async def wait_for_response(self, message_type: MessageType) -> Frame:
        while True:
            response = await self._responses.get()
            if response.message_type == message_type:
                return response


def test_model_template_robodojo_hooks_are_noop():
    model = ModelTemplate()

    assert model.prepare_case({"case": "demo"}) is None
    assert model.on_trial_end({"success": True}) is None


@pytest.mark.asyncio
async def test_policy_server_hello_and_heartbeat():
    server = PolicyServer(DummyModel())

    hello = _expect_frame(await server.process_frame(_frame(MessageType.HELLO)))
    heartbeat = _expect_frame(await server.process_frame(_frame(MessageType.HEARTBEAT)))

    assert hello.message_type == MessageType.HELLO_ACK
    assert heartbeat.message_type == MessageType.HEARTBEAT_ACK
    assert heartbeat.payload["ok"] is True


@pytest.mark.asyncio
async def test_policy_server_prepare_reset_infer_trial_end_flow():
    model = DummyModel()
    server = PolicyServer(model)

    prepare = _expect_frame(
        await server.process_frame(
            _frame(
                MessageType.PREPARE_CASE,
                {"difficulty": "easy"},
                action_case_id="case-1",
            )
        )
    )
    reset = _expect_frame(
        await server.process_frame(
            _frame(MessageType.RESET, {"trial_id": "trial-1"}, trial_id="trial-1")
        )
    )
    infer = _expect_frame(
        await server.process_frame(
            _frame(
                MessageType.INFER,
                {"observation": {"state": np.ones(3, dtype=np.float32)}},
                trial_id="trial-1",
                step=3,
            )
        )
    )
    trial_end = _expect_frame(
        await server.process_frame(
            _frame(
                MessageType.TRIAL_END,
                {"success": True},
                trial_id="trial-1",
            )
        )
    )

    assert prepare.message_type == MessageType.PREPARE_CASE_ACK
    assert prepare.payload["ok"] is True
    assert model.prepared == {"difficulty": "easy", "action_case_id": "case-1"}
    assert reset.message_type == MessageType.RESET_RESULT
    assert model.reset_count == 1
    assert infer.message_type == MessageType.INFER_RESULT
    assert infer.step == 3
    assert "latency_ms" in infer.payload
    np.testing.assert_array_equal(
        infer.payload["actions"]["arm_joint_state"],
        np.zeros(2, dtype=np.float32),
    )
    assert model.observation is not None
    np.testing.assert_array_equal(
        model.observation["state"],
        np.ones(3, dtype=np.float32),
    )
    assert trial_end.message_type == MessageType.TRIAL_END_ACK
    assert model.trial_result == {"success": True}


@pytest.mark.asyncio
async def test_policy_server_infer_method_can_return_full_payload():
    class InferModel:
        def infer(self, observation):
            return {"actions": [observation["x"]], "model_meta": {"name": "dummy"}}

    server = PolicyServer(InferModel())
    response = _expect_frame(
        await server.process_frame(_frame(MessageType.INFER, {"observation": {"x": 7}}))
    )

    assert response.message_type == MessageType.INFER_RESULT
    assert response.payload["actions"] == [7]
    assert response.payload["model_meta"] == {"name": "dummy"}
    assert "latency_ms" in response.payload


@pytest.mark.asyncio
async def test_policy_server_prefers_legacy_update_get_action_for_compatibility():
    class LegacyModel:
        def __init__(self):
            self.observation = None

        def update_obs(self, observation):
            self.observation = observation

        def get_action(self):
            assert self.observation is not None
            return {"from": "legacy", "x": self.observation["x"]}

        def infer(self, payload=None):
            return {"from": "direct", "payload": payload}

    server = PolicyServer(LegacyModel())
    response = _expect_frame(
        await server.process_frame(_frame(MessageType.INFER, {"observation": {"x": 7}}))
    )

    assert response.message_type == MessageType.INFER_RESULT
    assert response.payload["actions"] == {"from": "legacy", "x": 7}


@pytest.mark.asyncio
async def test_policy_server_heartbeat_responds_while_sync_infer_is_running():
    class BlockingInferModel:
        def __init__(self):
            self.started = threading.Event()
            self.release = threading.Event()

        def infer(self, observation):
            self.started.set()
            assert self.release.wait(timeout=2)
            return [observation["x"]]

    model = BlockingInferModel()
    server = PolicyServer(model)
    websocket = QueueWebSocket()
    handler = asyncio.create_task(server._handle_connection(cast(Any, websocket)))

    try:
        await websocket.feed(
            _frame(
                MessageType.INFER,
                {"observation": {"x": 7}},
                request_id="infer-1",
            )
        )
        assert await asyncio.wait_for(
            asyncio.to_thread(model.started.wait, 1.0),
            timeout=1.5,
        )

        await websocket.feed(_frame(MessageType.HEARTBEAT, request_id="heartbeat-1"))

        heartbeat = await asyncio.wait_for(
            websocket.wait_for_response(MessageType.HEARTBEAT_ACK),
            timeout=0.5,
        )
        assert heartbeat.request_id == "heartbeat-1"
        assert not model.release.is_set()

        model.release.set()
        infer = await asyncio.wait_for(
            websocket.wait_for_response(MessageType.INFER_RESULT),
            timeout=1.0,
        )
        assert infer.request_id == "infer-1"
        assert infer.payload["actions"] == [7]

        await websocket.feed(_frame(MessageType.CLOSE, request_id="close-1"))
        await asyncio.wait_for(handler, timeout=1.0)
    finally:
        model.release.set()
        if not websocket.closed:
            await websocket.close()
        if not handler.done():
            await asyncio.wait_for(handler, timeout=1.0)


@pytest.mark.asyncio
async def test_policy_server_returns_error_frame_for_bad_infer_payload():
    server = PolicyServer(DummyModel())

    response = _expect_frame(await server.process_frame(_frame(MessageType.INFER, {})))

    assert response.message_type == MessageType.ERROR
    assert response.payload["code"] == ErrorCode.INVALID_FRAME.value
    assert "observation" in response.payload["message"]
