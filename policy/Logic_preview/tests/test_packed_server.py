import asyncio
from pathlib import Path
import sys
import unittest
from uuid import uuid4

from websockets.asyncio.client import connect

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packed_server import PackedPolicyServer, PolicyServerConfig
from client_server.ws.protocol.codec import decode_envelope, encode_frame
from client_server.ws.protocol.messages import MessageType
from client_server.ws.protocol.schemas import Frame


class CountingModel:
    def __init__(self):
        self.calls = 0
        self.active = 0

    async def count(self):
        self.active += 1
        if self.active != 1:
            raise RuntimeError("Concurrent access to shared model")
        await asyncio.sleep(0.01)
        self.calls += 1
        self.active -= 1
        return self.calls


async def request(socket, evaluation, kind=MessageType.HELLO, request_id=None, **payload):
    frame = Frame(message_type=kind, evaluation_id=evaluation, request_id=request_id or uuid4().hex,
                  payload=payload)
    await socket.send(encode_frame(frame))
    return decode_envelope(await socket.recv())


class PackedServerTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_shards_and_reconnect_share_model_and_isolate_replay(self):
        model = CountingModel()
        server = PackedPolicyServer(model, PolicyServerConfig(host="127.0.0.1", port=0))
        await server.start()
        self.addAsyncCleanup(server.stop)
        url = server.url
        async with connect(url) as a, connect(url) as b:
            first = await request(a, "shard-a")
            second = await request(b, "shard-b")
            self.assertEqual([first.message_type, second.message_type], [MessageType.HELLO_ACK] * 2)
            a1, b1 = await asyncio.gather(request(a, "shard-a", MessageType.CALL, "same", func_name="count"),
                                          request(b, "shard-b", MessageType.CALL, "same", func_name="count"))
            self.assertEqual({a1.payload["result"], b1.payload["result"]}, {1, 2})
            if hasattr(server.sessions["shard-a"], "_active_connection"):
                async with connect(url) as duplicate:
                    self.assertEqual((await request(duplicate, "shard-a")).message_type, MessageType.ERROR)
        async with connect(url) as a:
            hello = await request(a, "shard-a")
            self.assertEqual(hello.payload.get("server_instance_id"), first.payload.get("server_instance_id"))
            replay = await request(a, "shard-a", MessageType.CALL, "same", func_name="count")
            self.assertEqual(replay.payload, a1.payload)
            self.assertEqual(model.calls, 2)
        async with connect(url) as c:
            self.assertEqual((await request(c, "shard-c")).message_type, MessageType.HELLO_ACK)
