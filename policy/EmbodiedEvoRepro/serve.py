"""Use XPolicyLab's server with explicit policy cleanup on owned process shutdown."""

import asyncio
import signal

from client_server.ws.model_server import PolicyServer, PolicyServerConfig
from setup_policy_server import parse_args_and_config
from XPolicyLab.policy.EmbodiedEvoRepro.model import Model


def main():
    cfg = parse_args_and_config()
    model = Model(cfg)

    def interrupted(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        server = PolicyServer(model, PolicyServerConfig(host=cfg["host"], port=int(cfg["port"])))
        asyncio.run(server.serve_forever())
    except KeyboardInterrupt:
        pass
    finally:
        model.close()


if __name__ == "__main__":
    main()
