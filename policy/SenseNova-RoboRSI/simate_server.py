"""Node-local Simate_beta service compatible with the pinned VLA proposal API."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

POLICY_PACKAGE = "XPolicyLab.policy." + Path(__file__).resolve().parent.name
SimateBridge = importlib.import_module(POLICY_PACKAGE + ".simate_bridge").SimateBridge

logger = logging.getLogger("Simate_beta")


async def serve(bridge, *, host, port, ready_fd=None, metadata=None):
    from websockets.asyncio.server import serve as websocket_serve
    from websockets.datastructures import Headers
    from websockets.http11 import Response
    load_openpi_client = importlib.import_module(POLICY_PACKAGE + ".pi05_source").load_openpi_client

    codec = load_openpi_client()
    owners = {}

    async def health(connection, request):
        if request.path == "/healthz":
            return Response(200, "OK", Headers({"Content-Type": "text/plain"}), b"ok\n")

    async def handler(connection):
        await connection.send(codec.Packer().pack({
            "policy": "Simate_beta", "action_horizon": 50, "action_dim": 14,
            "history": "25 frames at 20 native action steps; isolated episode UUID",
            "deployment": metadata or {},
        }))
        owned_scopes = set()
        try:
            async for message in connection:
                try:
                    payload = codec.unpackb(message)
                    scope, operation = payload.get("evaluation_id"), payload.get("op", "infer")
                    if operation in ("observe", "release") and scope in owners and owners[scope] is not connection:
                        raise ValueError("This episode's history belongs to another observation connection")
                    result = await asyncio.to_thread(bridge.request, payload)
                    if operation == "observe":
                        owners[scope] = connection
                        owned_scopes.add(scope)
                    elif operation == "release":
                        owners.pop(scope, None)
                        owned_scopes.discard(scope)
                    await connection.send(codec.Packer().pack(result))
                except Exception as error:
                    logger.exception("Simate request failed")
                    await connection.send(f"{type(error).__name__}: {error}")
        finally:
            for scope in owned_scopes:
                await asyncio.to_thread(bridge.request, {"op": "release", "evaluation_id": scope})
                owners.pop(scope, None)

    async with websocket_serve(
        handler, host, port, compression=None, max_size=None,
        process_request=health, ping_interval=20, ping_timeout=300,
    ) as listener:
        bound = listener.sockets[0].getsockname()[1]
        if ready_fd is not None:
            with os.fdopen(ready_fd, "w") as stream:
                stream.write(json.dumps({"port": bound}) + "\n")
                stream.flush()
        logger.info("Simate_beta ready on %s:%s", host, bound)
        await listener.serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1"])
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--ready-fd", type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    import yaml
    importlib.import_module(POLICY_PACKAGE + ".simate_assets").validate_source()
    from XPolicyLab.policy.Simate_beta.model import Model

    policy_dir = Path(__file__).resolve().parent.parent / "Simate_beta"
    cfg = yaml.safe_load((policy_dir / "deploy.yml").read_text())
    cfg.update(
        checkpoint_path=str(args.checkpoint_dir.resolve()),
        env_cfg_type="arx_x5", action_type="joint", bench_name="RoboDojo",
        seed=args.seed, exec_chunk_size=50,
    )
    metadata = {
        "seed": args.seed, "config": cfg,
        "model_bytes": (args.checkpoint_dir / "model.safetensors").stat().st_size,
        "tokenizer_sha256": hashlib.sha256((args.checkpoint_dir / "tokenizer.model").read_bytes()).hexdigest(),
        "service_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    logger.info("Simate deployment: %s", json.dumps(metadata, sort_keys=True))
    # Bound only this shared service's PyTorch allocator; model math is unchanged.
    import torch
    fraction = float(os.environ.get("ROBORSI_SIM_MEMORY_FRACTION", "0.45"))
    if not 0 < fraction <= 0.45:
        raise ValueError("The shared Simate allocator must fit its reserved GPU budget")
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    metadata["cuda_memory_fraction"] = fraction
    bridge = SimateBridge(Model(cfg))
    asyncio.run(serve(bridge, host=args.host, port=args.port, ready_fd=args.ready_fd, metadata=metadata))


if __name__ == "__main__":
    main()
