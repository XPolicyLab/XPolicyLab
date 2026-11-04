"""Optional OpenPI-compatible WebSocket transport."""
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from .client.websocket_policy_server import WebsocketPolicyServer
# from openpi_client import msgpack_numpy
from .client import msgpack_numpy
import traceback
import websockets
import websockets.asyncio.server as _ws_server
import websockets.frames

logger = logging.getLogger(__name__)

class TimedWebsocketPolicyServer(WebsocketPolicyServer):
    """Websocket server that logs server-side wait / unpack / infer / pack / send times."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._request_index = 0

    async def _handler(self, websocket: _ws_server.ServerConnection):
        logger.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))
        prev_pack_ms = None
        prev_send_ms = None

        while True:
            try:
                wait_t0 = time.perf_counter()
                raw_obs = await websocket.recv()
                wait_obs_ms = (time.perf_counter() - wait_t0) * 1000.0

                unpack_t0 = time.perf_counter()
                obs = msgpack_numpy.unpackb(raw_obs)
                unpack_ms = (time.perf_counter() - unpack_t0) * 1000.0
                payload_in_kb = len(raw_obs) / 1024.0

                infer_t0 = time.perf_counter()
                action = self._policy.infer(obs)
                infer_ms = (time.perf_counter() - infer_t0) * 1000.0

                server_timing = {
                    "infer_ms": infer_ms,
                    "wait_obs_ms": wait_obs_ms,
                    "unpack_ms": unpack_ms,
                }
                if prev_pack_ms is not None:
                    server_timing["prev_pack_ms"] = prev_pack_ms
                if prev_send_ms is not None:
                    server_timing["prev_send_ms"] = prev_send_ms
                action["server_timing"] = server_timing

                pack_t0 = time.perf_counter()
                packed = packer.pack(action)
                pack_ms = (time.perf_counter() - pack_t0) * 1000.0
                payload_out_kb = len(packed) / 1024.0

                send_t0 = time.perf_counter()
                await websocket.send(packed)
                send_ms = (time.perf_counter() - send_t0) * 1000.0
                prev_pack_ms = pack_ms
                prev_send_ms = send_ms

                self._request_index += 1
                comm_ms = unpack_ms + pack_ms + send_ms
                logger.info(
                    "Comm #%d | wait_obs=%.1fms (idle, includes robot/client) | "
                    "unpack=%.1fms pack=%.1fms send=%.1fms server_comm=%.1fms | "
                    "infer=%.1fms | in=%.1fKB out=%.1fKB",
                    self._request_index,
                    wait_obs_ms,
                    unpack_ms,
                    pack_ms,
                    send_ms,
                    comm_ms,
                    infer_ms,
                    payload_in_kb,
                    payload_out_kb,
                )
            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def main():
    from deploy.multiview_policy import main as serve
    serve()

if __name__ == "__main__":
    main()
