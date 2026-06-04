import asyncio
from collections.abc import AsyncIterator

import pytest

from robodojo.protocol.client import PolicyEvalClient, PolicyEvalClientConfig
from robodojo.protocol.codec import decode_envelope
from robodojo.protocol.exceptions import ErrorCode, WsError
from robodojo.protocol.messages import MessageType
from robodojo.protocol.schemas import Frame


class FakeWebSocket:
    def __init__(self):
        self.sent: list[bytes] = []
        self.closed = False

    async def send(self, message: bytes, text: bool | None = None) -> None:
        self.sent.append(message)

    async def close(self) -> None:
        self.closed = True

    def __aiter__(self) -> AsyncIterator[bytes | str]:
        return self

    async def __anext__(self) -> bytes | str:
        raise StopAsyncIteration


class EmptyAsyncWebSocket(FakeWebSocket):
    pass


@pytest.mark.asyncio
async def test_client_config():
    cfg = PolicyEvalClientConfig(
        url="ws://127.0.0.1:9",
        evaluation_id="eval-test",
        max_connect_attempts=1,
        connect_retry_delay_s=0.01,
    )
    client = PolicyEvalClient(config=cfg)
    assert client.config.evaluation_id == "eval-test"
    frame = Frame(
        message_type=MessageType.RESET,
        request_id="req-1",
        evaluation_id="eval-test",
    )
    assert frame.message_type == MessageType.RESET


@pytest.mark.asyncio
async def test_send_close_does_not_wait_for_response():
    client = PolicyEvalClient(
        PolicyEvalClientConfig(url="ws://example.test", evaluation_id="eval-test")
    )
    ws = FakeWebSocket()
    client._ws = ws

    sent_frame = await client.send_close("done")

    assert sent_frame.message_type == MessageType.CLOSE
    assert client._pending == {}
    decoded = decode_envelope(ws.sent[0])
    assert decoded.message_type == MessageType.CLOSE
    assert decoded.payload["reason"] == "done"


@pytest.mark.asyncio
async def test_dispatch_incoming_resolves_and_removes_pending_future():
    client = PolicyEvalClient(
        PolicyEvalClientConfig(url="ws://example.test", evaluation_id="eval-test")
    )
    fut = asyncio.get_running_loop().create_future()
    client._pending["req-1"] = fut

    frame = Frame(
        message_type=MessageType.INFER_RESULT,
        request_id="req-1",
        evaluation_id="eval-test",
        payload={"actions": [1, 2, 3]},
    )
    client._dispatch_incoming(frame)

    assert fut.result().payload["actions"] == [1, 2, 3]
    assert "req-1" not in client._pending


@pytest.mark.asyncio
async def test_dispatch_error_with_unknown_code_does_not_crash_receiver():
    client = PolicyEvalClient(
        PolicyEvalClientConfig(url="ws://example.test", evaluation_id="eval-test")
    )
    fut = asyncio.get_running_loop().create_future()
    client._pending["req-1"] = fut

    frame = Frame(
        message_type=MessageType.ERROR,
        request_id="req-1",
        evaluation_id="eval-test",
        payload={"code": "new_server_code", "message": "bad remote frame"},
    )
    client._dispatch_incoming(frame)

    with pytest.raises(WsError) as exc_info:
        fut.result()
    assert exc_info.value.code == ErrorCode.INTERNAL
    assert client._pending == {}


@pytest.mark.asyncio
async def test_recv_loop_fails_pending_futures_on_clean_disconnect():
    client = PolicyEvalClient(
        PolicyEvalClientConfig(url="ws://example.test", evaluation_id="eval-test")
    )
    client._ws = EmptyAsyncWebSocket()
    fut = asyncio.get_running_loop().create_future()
    client._pending["req-1"] = fut

    await client._recv_loop()

    with pytest.raises(WsError) as exc_info:
        fut.result()
    assert exc_info.value.code == ErrorCode.INTERNAL
    assert client._pending == {}
