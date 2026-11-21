"""Run the original local VLA server and report its bound port to its owner."""

import json
import logging
import os

from server import openpi_batch_server as server
from server.protocol import health_check, log_server_ready


class LocalPolicyServer(server.BatchedWebsocketPolicyServer):
    async def run(self):
        import websockets.asyncio.server as websocket_server

        try:
            async with websocket_server.serve(
                self._handler,
                self._host,
                self._port,
                compression=None,
                max_size=None,
                process_request=health_check,
            ) as listener:
                port = listener.sockets[0].getsockname()[1]
                with os.fdopen(self._ready_fd, "w") as stream:
                    self._ready_fd = None
                    stream.write(json.dumps({"port": port}) + "\n")
                    stream.flush()
                log_server_ready(
                    server.logger, service="OpenPI", host=self._host, port=port
                )
                await listener.serve_forever()
        finally:
            if self._ready_fd is not None:
                os.close(self._ready_fd)
                self._ready_fd = None
            await self._batcher.close()


def main():
    parser = server.build_arg_parser()
    parser.add_argument("--ready-fd", type=int, required=True)
    args = parser.parse_args()
    if args.host != "127.0.0.1":
        parser.error("the local VLA server requires --host 127.0.0.1")
    if args.max_batch_size <= 0:
        parser.error("--max-batch-size must be positive")
    if args.batch_wait_ms < 0:
        parser.error("--batch-wait-ms cannot be negative")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    policy = server.create_policy(
        config_name=args.config_name, checkpoint_dir=args.checkpoint_dir
    )
    service = LocalPolicyServer(
        policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
        max_batch_size=args.max_batch_size,
        batch_wait_ms=args.batch_wait_ms,
        pad_to_max=args.batch_pad_to_max,
    )
    service._ready_fd = args.ready_fd
    service.serve_forever()


if __name__ == "__main__":
    main()
