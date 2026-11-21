"""Exercise real RPC framing/server decode in memory, without opening a socket."""

import asyncio

import cv2
import numpy as np
import pytest

from client_server.ws import model_server
from client_server.ws.model_client import WsModelClient
from client_server.ws.protocol import client as protocol_client
from client_server.ws.protocol.codec import decode_envelope, encode_frame
from XPolicyLab.utils import process_data


class CaptureModel:
    def update_obs(self, obs):
        self.observation = obs

    def update_obs_batch(self, obs):
        self.observation = obs

    def begin_episode(self, obs):
        self.observation = obs


class InMemoryWebSocket:
    """Replace only network I/O; run the real client and PolicyServer dispatch."""

    def __init__(self, server):
        self.server = server
        self.replies = asyncio.Queue()

    async def send(self, data, text=None):
        assert isinstance(data, bytes)
        request = decode_envelope(data)
        response = await self.server.process_frame(request)
        assert response is not None
        await self.replies.put(encode_frame(response))

    async def close(self):
        await self.replies.put(None)

    def __aiter__(self):
        return self

    async def __anext__(self):
        value = await self.replies.get()
        if value is None:
            raise StopAsyncIteration
        return value


@pytest.fixture
def rpc(monkeypatch):
    model = CaptureModel()
    server = model_server.PolicyServer(model)

    async def connect(_url, **_kwargs):
        return InMemoryWebSocket(server)

    monkeypatch.setattr(protocol_client.websockets, "connect", connect)
    with WsModelClient(
        url="ws://in-memory.invalid", evaluation_id="rgb-test", trial_id="trial",
        max_connect_attempts=1,
    ) as client:
        yield model, client


def image():
    rows, cols = np.indices((37, 49))
    return np.stack(((rows * 19 + cols * 7) % 256,
                     (cols * 13 + rows) % 256,
                     (rows * 3 + cols * 29) % 256), axis=-1).astype(np.uint8)


@pytest.mark.parametrize("method", ["update_obs", "begin_episode"])
def test_raw_rgb_native_rpc_preserves_every_pixel_and_color_order(rpc, monkeypatch, method):
    model, client = rpc
    rgb = image()[:, ::-1]  # Strided native arrays also survive msgpack serialization.
    depth = np.arange(37 * 49, dtype=np.uint16).reshape(37, 49)
    observation = {"vision": {"head_camera": {"color": rgb, "depth": depth}},
                   "state": {"joint_state": np.arange(6, dtype=np.float32)}}

    def no_image_encoding(*_args, **_kwargs):
        raise AssertionError("Native ndarray observations must not be JPEG encoded")

    monkeypatch.setattr(process_data, "encode_image_bit", no_image_encoding)
    monkeypatch.setattr(cv2, "imencode", no_image_encoding)
    client.call(func_name=method, obs=observation)
    received = model.observation
    np.testing.assert_array_equal(received["vision"]["head_camera"]["color"], rgb)
    assert received["vision"]["head_camera"]["color"].dtype == np.uint8
    assert not np.shares_memory(received["vision"]["head_camera"]["color"], rgb)
    np.testing.assert_array_equal(received["vision"]["head_camera"]["depth"], depth)
    np.testing.assert_array_equal(received["state"]["joint_state"], observation["state"]["joint_state"])


def test_raw_rgb_batch_rpc_preserves_all_cameras_and_environments(rpc):
    model, client = rpc
    observations = [
        {"env_idx": index, "vision": {
            "head_camera": {"color": np.roll(image(), index, axis=1)},
            "left_camera": {"colors": np.roll(image(), index, axis=0)},
            "right_camera": image().copy(),
        }} for index in range(3)
    ]
    client.call(func_name="update_obs_batch", obs=observations)
    for expected, actual in zip(observations, model.observation):
        assert actual["env_idx"] == expected["env_idx"]
        for camera in ("head_camera", "left_camera", "right_camera"):
            target = expected["vision"][camera]
            received = actual["vision"][camera]
            if isinstance(target, dict):
                key = next(iter(target))
                target, received = target[key], received[key]
            np.testing.assert_array_equal(received, target)


@pytest.mark.parametrize("legacy", [False, True])
def test_optional_encoded_jpeg_rpc_keeps_shared_decoder_compatibility(rpc, legacy):
    model, client = rpc
    rgb = image()
    if legacy:
        # Deliberately reproduce the pre-marker fixture format in memory only.
        ok, buffer = cv2.imencode(".jpg", rgb)
        assert ok
        encoded = buffer.tobytes()
    else:
        encoded = process_data.encode_image_bit(rgb, quality=100)
    expected = process_data.decode_image_bit(encoded)
    client.call(func_name="update_obs", obs={"vision": {"head_camera": {"color": encoded}}})
    np.testing.assert_array_equal(model.observation["vision"]["head_camera"]["color"], expected)
