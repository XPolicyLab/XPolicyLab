"""Load a pinned VLA in its own Python environment and serve XPolicyLab WS."""

import asyncio
import importlib
import inspect
import os
import sys
from pathlib import Path

from PhysicalRSI_core.infra.processes import watch_parent_death
from PhysicalRSI_core.infra.storage import atomic_json, read_json

from .expert_service import verify_spec


async def serve(spec, ready_path):
    identity = verify_spec(spec)
    module, _, factory_name = spec["entrypoint"].partition(":")
    factory = getattr(importlib.import_module(module), factory_name)
    source = Path(inspect.getsourcefile(factory)).resolve()
    if not any(
        source.is_relative_to(Path(root["root"]))
        and str(source.relative_to(Path(root["root"]))) in root["files"]
        for root in spec["code"]
    ):
        raise ValueError("Loaded model factory is not in the pinned code inventory")
    model = factory(dict(spec["deploy"]))
    # Detect artifacts modified during model initialization before announcing.
    if verify_spec(spec) != identity:
        raise ValueError("Expert identity changed during initialization")

    class IdentifiedModel:
        def physicalrsi_identity(self):
            return dict(identity=identity, expert=spec["expert"])

        def __getattr__(self, name):
            return getattr(model, name)

    from client_server.ws.model_server import PolicyServer, PolicyServerConfig

    server = PolicyServer(
        IdentifiedModel(), PolicyServerConfig(host="127.0.0.1", port=0)
    )
    await server.start()
    atomic_json(
        ready_path,
        dict(
            identity=identity,
            expert=spec["expert"],
            pid=os.getpid(),
            endpoint=server.url,
        ),
    )
    try:
        await server.serve_forever()
    finally:
        await server.stop()


if __name__ == "__main__":
    watch_parent_death(lambda: os._exit(70))
    asyncio.run(serve(read_json(sys.argv[1]), sys.argv[2]))
