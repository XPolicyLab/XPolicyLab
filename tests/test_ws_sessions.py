"""Session ownership and reconnect replay regressions (no model weights needed).

Run from the checkout with ``python -m unittest discover -s tests -p test_ws_sessions.py``.
Socket tests bind only to loopback and use an ephemeral port.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np
from websockets.asyncio.client import connect

from client_server.ws.protocol.codec import encode_frame, decode_envelope

from client_server.ws.model_server import PolicyServer, PolicyServerConfig
from client_server.ws.protocol.client import PolicyEvalClient, PolicyEvalClientConfig
from client_server.ws.protocol.exceptions import ErrorCode, WsError
from client_server.ws.protocol.messages import MessageType
from client_server.ws.protocol.schemas import Frame


def call_frame(request_id, *, evaluation_id="eval-a", method="get_action", **context):
    return Frame(
        message_type=MessageType.CALL,
        request_id=request_id,
        evaluation_id=evaluation_id,
        payload={"func_name": method},
        **context,
    )


class ReusedBufferModel:
    def __init__(self):
        self.calls = 0
        self.buffer = np.zeros(2, dtype=np.float32)

    def get_action(self):
        self.calls += 1
        self.buffer[:] = self.calls
        return [{"nested": {"action": self.buffer}}]


class BlockingModel:
    def __init__(self):
        self.calls = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def get_action(self):
        self.calls += 1
        self.entered.set()
        await self.release.wait()
        return np.array([self.calls], dtype=np.float32)


class ReplayTests(unittest.IsolatedAsyncioTestCase):
    async def test_cached_result_survives_model_buffer_reuse(self):
        model = ReusedBufferModel()
        server = PolicyServer(model)
        first = await server.process_frame(call_frame("first"))
        await server.process_frame(call_frame("second"))
        replay = await server.process_frame(call_frame("first"))
        self.assertEqual(model.calls, 2)
        for response in (first, replay):
            np.testing.assert_array_equal(
                response.payload["result"][0]["nested"]["action"], [1, 1]
            )
        # A caller mutating the returned container cannot poison future replays.
        replay.payload["result"][0]["nested"]["action"] = np.array([99, 99])
        replay_again = await server.process_frame(call_frame("first"))
        np.testing.assert_array_equal(
            replay_again.payload["result"][0]["nested"]["action"], [1, 1]
        )

    async def test_same_request_inflight_and_cancelled_waiter_execute_once(self):
        model = BlockingModel()
        server = PolicyServer(model)
        first = asyncio.create_task(server.process_frame(call_frame("same")))
        await asyncio.wait_for(model.entered.wait(), 2)
        replay = asyncio.create_task(server.process_frame(call_frame("same")))
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        model.release.set()
        response = await asyncio.wait_for(replay, 2)
        self.assertEqual(model.calls, 1)
        np.testing.assert_array_equal(response.payload["result"], [1])
        await server.process_frame(call_frame("same"))
        self.assertEqual(model.calls, 1)

    async def test_inflight_request_id_cannot_change_payload(self):
        model = BlockingModel()
        server = PolicyServer(model)
        first = asyncio.create_task(server.process_frame(call_frame("same")))
        try:
            await asyncio.wait_for(model.entered.wait(), 2)
            with self.assertLogs("client_server.ws.model_server", level="ERROR"):
                rejected = await server.process_frame(call_frame("same", method="reset"))
            self.assertEqual(rejected.message_type, MessageType.ERROR)
            self.assertEqual(rejected.payload["code"], ErrorCode.INVALID_FRAME.value)
            self.assertEqual(model.calls, 1)
        finally:
            model.release.set()
            await first

    async def test_cached_request_id_binds_content_and_trial_context(self):
        model = ReusedBufferModel()
        server = PolicyServer(model)
        original = call_frame("same", trial_id="trial-a", step=1)
        await server.process_frame(original)
        changed = [
            call_frame("same", method="reset", trial_id="trial-a", step=1),
            call_frame("same", trial_id="trial-b", step=1),
            call_frame("same", trial_id="trial-a", step=2),
            call_frame("same", trial_id="trial-a", step=1, repeat_index=2),
        ]
        for request in changed:
            with self.subTest(request=request.to_wire_dict()):
                with self.assertLogs("client_server.ws.model_server", level="ERROR"):
                    response = await server.process_frame(request)
                self.assertEqual(response.message_type, MessageType.ERROR)
                self.assertEqual(response.payload["code"], ErrorCode.INVALID_FRAME.value)
        # Transmission timestamps may change on a legitimate retry.
        retry = call_frame("same", trial_id="trial-a", step=1, sent_at="new-transmission")
        self.assertEqual((await server.process_frame(retry)).message_type, MessageType.CALL_RESULT)
        self.assertEqual(model.calls, 1)

    async def test_infer_snapshot_survives_buffer_reuse(self):
        class InferModel:
            def __init__(self):
                self.calls = 0
                self.buffer = np.zeros(1, dtype=np.float32)

            def infer(self, observation):
                self.calls += 1
                self.buffer[:] = observation["value"]
                return {"actions": [{"joint": self.buffer}]}

        model = InferModel()
        server = PolicyServer(model)

        def request(request_id, value):
            return Frame(
                message_type=MessageType.INFER,
                request_id=request_id,
                evaluation_id="eval-a",
                payload={"observation": {"value": value}},
            )

        original = await server.process_frame(request("first", 1))
        await server.process_frame(request("second", 2))
        replay = await server.process_frame(request("first", 1))
        for response in (original, replay):
            self.assertEqual(response.message_type, MessageType.INFER_RESULT)
            np.testing.assert_array_equal(response.payload["actions"][0]["joint"], [1])
        self.assertEqual(model.calls, 2)

    async def test_unserializable_result_is_cached_error(self):
        class BadResultModel:
            def __init__(self):
                self.calls = 0

            def get_action(self):
                self.calls += 1
                return object()

        model = BadResultModel()
        server = PolicyServer(model)
        with self.assertLogs("client_server.ws.model_server", level="ERROR"):
            first = await server.process_frame(call_frame("bad-result"))
        replay = await server.process_frame(call_frame("bad-result"))
        self.assertEqual(first.message_type, MessageType.ERROR)
        self.assertEqual(first.payload["code"], ErrorCode.CALL_FAILED.value)
        self.assertIn("msgpack", first.payload["message"])
        self.assertEqual(replay.to_wire_dict(), first.to_wire_dict())
        self.assertEqual(model.calls, 1)

    async def test_empty_or_foreign_evaluation_never_calls_model(self):
        model = ReusedBufferModel()
        server = PolicyServer(model)
        blank = await server.process_frame(call_frame("blank", evaluation_id="  "))
        self.assertEqual(blank.message_type, MessageType.ERROR)
        await server.process_frame(call_frame("owner"))
        foreign = await server.process_frame(call_frame("other", evaluation_id="eval-b"))
        self.assertEqual(foreign.message_type, MessageType.ERROR)
        self.assertEqual(model.calls, 1)


class ConnectionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.model = ReusedBufferModel()
        self.server = PolicyServer(
            self.model, PolicyServerConfig(host="127.0.0.1", port=0)
        )
        await self.server.start()
        self.clients = []
        self.status_patch = patch("client_server.ws.protocol.client._status")
        self.status_patch.start()

    async def asyncTearDown(self):
        for client in self.clients:
            await client.close()
        await self.server.stop()
        self.status_patch.stop()

    def client(self, evaluation_id="eval-a"):
        client = PolicyEvalClient(PolicyEvalClientConfig(
            url=self.server.url,
            evaluation_id=evaluation_id,
            connect_timeout_s=2,
            handshake_timeout_s=2,
            request_timeout_s=2,
            max_connect_attempts=3,
            connect_retry_delay_s=0.01,
            max_connect_seconds=5,
            close_timeout_s=2,
        ))
        self.clients.append(client)
        return client

    async def test_conflicting_connection_retries_after_owner_disconnect(self):
        owner = self.client()
        await owner.connect()
        reconnect = self.client()
        # The first handshake must be rejected. Release ownership in the retry
        # delay, so the second real WebSocket handshake can succeed deterministically.
        async def disconnect_owner(*_):
            await owner.close()

        with patch.object(reconnect, "_sleep_with_countdown", new=AsyncMock(
            side_effect=disconnect_owner
        )) as retry:
            ack = await reconnect.connect()
        retry.assert_awaited_once()
        self.assertEqual(ack.message_type, MessageType.HELLO_ACK)
        self.assertEqual(ack.payload["server_instance_id"], owner._server_instance_id)

    async def test_closed_connection_releases_owner_during_backpressure(self):
        model = BlockingModel()
        self.server.model = model
        # Production permits 128 pending replies; lower the threshold to reach
        # the same inline backpressure path with just two requests.
        with patch("client_server.ws.model_server._MAX_INFLIGHT_RESPONSES", 1):
            try:
                async with connect(self.server.url) as owner:
                    await owner.send(encode_frame(call_frame("first")))
                    await asyncio.wait_for(model.entered.wait(), 2)
                    await owner.send(encode_frame(call_frame("queued")))
                    async def wait_for_queued_request():
                        while len(self.server._inflight) < 2:
                            await asyncio.sleep(0.001)

                    await asyncio.wait_for(wait_for_queued_request(), 2)
                # The reader is blocked in an inline response, so its finally
                # clause cannot release ownership yet, despite the closed socket.
                async with connect(self.server.url) as reconnect:
                    hello = Frame(
                        message_type=MessageType.HELLO,
                        request_id="reconnect",
                        evaluation_id="eval-a",
                    )
                    await reconnect.send(encode_frame(hello))
                    ack = decode_envelope(await asyncio.wait_for(reconnect.recv(), 2))
                    self.assertEqual(ack.message_type, MessageType.HELLO_ACK)
                    await reconnect.send(encode_frame(call_frame("first")))
                    model.release.set()
                    response = decode_envelope(await asyncio.wait_for(reconnect.recv(), 2))
                    self.assertEqual(response.message_type, MessageType.CALL_RESULT)
                    np.testing.assert_array_equal(response.payload["result"], [1])
                # Both original requests execute, but replaying "first" adds no call.
                async def wait_for_executions():
                    while self.server._inflight:
                        await asyncio.sleep(0.001)

                await asyncio.wait_for(wait_for_executions(), 2)
                self.assertEqual(model.calls, 2)
            finally:
                model.release.set()

    async def test_foreign_evaluation_fails_without_retry_even_after_disconnect(self):
        owner = self.client()
        await owner.connect()
        await owner.close()
        foreign = self.client("eval-b")
        with patch.object(foreign, "_sleep_with_countdown", new=AsyncMock()) as retry:
            with self.assertRaises(WsError) as caught:
                await foreign.connect()
        self.assertEqual(caught.exception.code, ErrorCode.INVALID_FRAME)
        retry.assert_not_awaited()
        self.assertEqual(self.model.calls, 0)

    async def test_reconnected_client_replays_cached_result(self):
        original = self.client()
        await original.connect()
        first = await original.request(
            MessageType.CALL, {"func_name": "get_action"}, _request_id="survives-disconnect"
        )
        await original.close()
        reconnect = self.client()
        await reconnect.connect()
        replay = await reconnect.request(
            MessageType.CALL, {"func_name": "get_action"}, _request_id="survives-disconnect"
        )
        self.assertEqual(self.model.calls, 1)
        np.testing.assert_array_equal(
            first.payload["result"][0]["nested"]["action"],
            replay.payload["result"][0]["nested"]["action"],
        )


if __name__ == "__main__":
    unittest.main()
