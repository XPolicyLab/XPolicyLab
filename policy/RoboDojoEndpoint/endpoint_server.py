"""Official policy server with cleanup for remote batch sessions."""
import asyncio
import signal

from XPolicyLab.client_server.ws.model_server import PolicyServer, PolicyServerConfig
from XPolicyLab.policy.RoboDojoEndpoint.model import Model
from XPolicyLab.setup_policy_server import parse_args_and_config


async def run(cfg):
    model = Model(cfg)
    server = PolicyServer(model, PolicyServerConfig(host=cfg["host"], port=int(cfg["port"])))
    task = asyncio.create_task(server.serve_forever())
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, task.cancel)
    try:
        await task
    except asyncio.CancelledError:
        pass
    finally:
        await asyncio.to_thread(model.close)


if __name__ == "__main__":
    asyncio.run(run(parse_args_and_config()))
