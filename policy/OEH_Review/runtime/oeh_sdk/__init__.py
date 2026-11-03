"""`oeh-sdk` — the Python side of the OpenEmbodied-Harness boundary protocol.

Four things live here, and they are the four roles a process outside the kernel can
play:

* :class:`~oeh_sdk.rpc.Client` — the connection itself, JSON-RPC 2.0 over a Unix
  socket, symmetric (the kernel calls back).
* :class:`~oeh_sdk.plugin.Plugin` — a cross-process fiber: declares streams and
  capabilities, heartbeats, and runs `capability/execute` bodies.
* :class:`~oeh_sdk.brain.Brain` — a brain the kernel wakes, proposes through, and
  steers.
* :class:`~oeh_sdk.approval.Approver` — the one connection that answers `Ask`.

:mod:`oeh_sdk.protocol` holds the method table and the wire shapes, checked against
the Rust fixtures by `tests/test_protocol.py`.

Frames are the fifth thing, and they are deliberately not here: they never travel over
JSON-RPC. :mod:`oeh_sdk.frames` (`FramePublisher` / `FrameSubscriber`) moves them
through iceoryx2 shared memory and is the one module in this package with a dependency
— the `frames` extra — so it is imported explicitly, never from this namespace.

    import asyncio
    from oeh_sdk import Client

    async def main():
        client = await Client.connect_uds("./run/oeh.sock")
        print(await client.call("robot/state"))
        await client.close()

    asyncio.run(main())
"""

from . import protocol
from .approval import Approver
from .brain import Brain, TurnCtx
from .plugin import CapabilityDecl, ExecCtx, Plugin, StreamDecl
from .protocol import METHODS, PARAMS, RESULTS, StreamChannel, code
from .rpc import Client, RpcError

__all__ = [
    "Approver",
    "Brain",
    "CapabilityDecl",
    "Client",
    "ExecCtx",
    "METHODS",
    "PARAMS",
    "Plugin",
    "RESULTS",
    "RpcError",
    "StreamChannel",
    "StreamDecl",
    "TurnCtx",
    "code",
    "protocol",
]
