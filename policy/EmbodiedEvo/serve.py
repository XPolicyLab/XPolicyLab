"""Run all configured task agents behind one policy service address."""

import asyncio
import logging

from client_server.ws.model_server import PolicyServerConfig
from setup_policy_server import parse_args_and_config
from XPolicyLab.policy.EmbodiedEvo.runtime.serving import (
    TaskRoutingServer, close_models, prepare_models, server_settings,
)


def main():
    cfg = parse_args_and_config()
    models, run = prepare_models(cfg)
    try:
        config = PolicyServerConfig(
            host=cfg["host"], port=int(cfg["port"]),
            ws_ping_interval_s=cfg.get("ws_ping_interval_s", 20.0),
            ws_ping_timeout_s=cfg.get("ws_ping_timeout_s", 120.0),
            max_message_bytes=cfg.get("server_max_message_bytes", 64 * 1024 * 1024),
            max_receive_queue=2,
            max_pending_responses=4,
        )
        server = TaskRoutingServer(models, config, run_dir=run, **server_settings(cfg))
        print(f"[SERVER] {len(models)} task templates at {server.url}; independent layout/episode agents "
              f"start on demand (limit={server.capacity.limit}); run_dir={run}", flush=True)
        asyncio.run(server.serve_forever())
    except KeyboardInterrupt:
        print("[SERVER] shutting down task agents", flush=True)
    finally:
        close_models(models)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
