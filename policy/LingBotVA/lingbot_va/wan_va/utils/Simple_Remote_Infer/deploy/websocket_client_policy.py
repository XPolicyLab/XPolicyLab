import logging
import time
from typing import Dict, Optional, Tuple

import websockets.sync.client
from typing_extensions import override

from .msgpack_numpy import Packer, unpackb


class WebsocketClientPolicy:
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self,
                 host: str = "0.0.0.0",
                 port: Optional[int] = None,
                 api_key: Optional[str] = None) -> None:
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    # def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
    #     logging.info(f"Waiting for server at {self._uri}...")
    #     while True:
    #         try:
    #             headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
    #             conn = websockets.sync.client.connect(
    #                 self._uri, compression=None, max_size=None, additional_headers=headers
    #             )
    #             metadata = unpackb(conn.recv())
    #             return conn, metadata
    #         except ConnectionRefusedError:
    #             logging.info("Still waiting for server...")
    #             time.sleep(5)

    def _wait_for_server(
            self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {
                    "Authorization": f"Api-Key {self._api_key}"
                } if self._api_key else None
                # 禁用 ping 机制，防止推理时间过长导致超时
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    ping_interval=None,
                    close_timeout=10)
                metadata = unpackb(conn.recv())
                return conn, metadata
            except (ConnectionRefusedError, Exception) as e:
                logging.info(f"Still waiting for server... (Error: {e})")
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return unpackb(response)

    @override
    def reset(self) -> None:
        self.infer(dict(reset=True))


if __name__ == "__main__":
    policy_on_device = WebsocketClientPolicy(port=8000)
    import numpy as np
    import torch
    from PIL import Image

    from .image_tools import convert_to_uint8
    device = torch.device("cuda")

    base_0_rgb = np.random.randint(0,
                                   256,
                                   size=(1, 3, 224, 224),
                                   dtype=np.uint8)
    left_wrist_0_rgb = np.random.randint(0,
                                         256,
                                         size=(1, 3, 224, 224),
                                         dtype=np.uint8)
    state = np.random.rand(1, 8).astype(np.float32)
    prompt = ["do something"]

    # observation = {
    #     "image": {
    #         "base_0_rgb": torch.from_numpy(base_0_rgb).to(device)[None],
    #         "left_wrist_0_rgb": torch.from_numpy(left_wrist_0_rgb).to(device)[None],
    #     },
    #     "state": torch.from_numpy(state).to(device)[None],
    #     "prompt": prompt,
    # }

    observation = {
        "image": {
            "base_0_rgb": convert_to_uint8(base_0_rgb),
            "left_wrist_0_rgb": convert_to_uint8(left_wrist_0_rgb),
            "right_wrist_0_rgb": convert_to_uint8(left_wrist_0_rgb),
        },
        "state": state,
        "prompt": prompt,
    }

    policy_on_device.infer(observation)
    from IPython import embed
    embed()
