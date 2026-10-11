"""WebSocket policy server for remote model inference.

Hosts a VLA model behind a thin websocket+msgpack protocol so that
simulation workers (possibly running in a different conda env / on another
host) can drive the policy remotely.

Protocol (aligned with starVLA's ``deployment/model_server/server_policy.py``):

    Request:
        {"payload": {"images": [ndarray (H,W,3) uint8, ...],
                     "state":  ndarray (D_s,) float32,
                     "instruction": str}}

    Response (success):
        {"status": "ok",
         "actions": ndarray (T, D_a) float32}     # env-compatible, already denormalized

    Response (error):
        {"status": "error", "message": str}

For architectural details see ``docs/leap_eval_framework.md``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import websockets

from leap.serving.transport import decode, encode

logger = logging.getLogger(__name__)


class PolicyServer:
    """Serve a VLA policy for remote action prediction over WebSocket.

    Supports **batched inference**: incoming requests are accumulated and
    dispatched together in a single forward pass to maximize GPU utilization.
    """

    def __init__(
        self,
        policy: Any,
        host: str = "0.0.0.0",
        port: int = 8765,
        device: str = "cuda",
        max_batch_size: int = 8,
        max_wait_ms: float = 10.0,
    ) -> None:
        """
        Args:
            policy: Either a policy adapter (preferred)
                or a raw ``nn.Module`` with a ``predict_action(**kwargs)``
                method (legacy path). The former handles state normalization
                + action denormalization server-side; the latter is a passthrough.
            host, port: Bind address.
            device: Only used by the legacy raw-module path; ignored for
                the policy adapter (which owns its own device).
            max_batch_size: Maximum number of requests to batch together.
            max_wait_ms: Maximum time (ms) to wait for more requests before
                dispatching an incomplete batch.
        """
        self.policy = policy
        self.host = host
        self.port = port
        self.device = torch.device(device)
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms

        # Batch queue: list of (payload_dict, asyncio.Future) tuples.
        self._queue: List[Tuple[dict, asyncio.Future]] = []
        self._queue_event: Optional[asyncio.Event] = None
        self._batch_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------

    def serve_forever(self) -> None:
        """Start the server (blocks the calling thread)."""
        logger.info("PolicyServer starting on ws://%s:%d", self.host, self.port)
        asyncio.run(self._run())

    async def _run(self) -> None:
        # Optional cosmetic-error filter, gated by env var so libero / calvin
        # / other evaluators that don't do bare TCP probes don't get their
        # legitimate handshake-failure traces silenced.
        #
        # Set ``LEAP_SILENCE_HANDSHAKE_ERRORS=1`` (eval_robotwin.sh does this
        # automatically) to drop the
        #   websockets.server: opening handshake failed
        #   EOFError: connection closed while reading HTTP request line
        #   InvalidMessage: did not receive a valid HTTP request
        # noise that fires on every probe in eval_robotwin.sh's
        # wait_for_server() loop. Real WS handshake errors from misbehaving
        # clients still surface via our own _handler exceptions.
        if os.environ.get("LEAP_SILENCE_HANDSHAKE_ERRORS", "0") == "1":
            _ws_logger = logging.getLogger("websockets.server")

            class _DropHandshakeFailures(logging.Filter):
                def filter(self, record: logging.LogRecord) -> bool:
                    return "opening handshake failed" not in record.getMessage()

            _ws_logger.addFilter(_DropHandshakeFailures())
            logger.info(
                "Handshake-failure log filter enabled "
                "(LEAP_SILENCE_HANDSHAKE_ERRORS=1)"
            )

        self._queue_event = asyncio.Event()
        self._batch_task = asyncio.ensure_future(self._batch_dispatcher())
        async with websockets.serve(
            self._handler,
            self.host,
            self.port,
            max_size=100 * 1024 * 1024,  # 100 MB — agentview+wrist images per request
            ping_interval=None,  # disable keepalive — inference blocks the event loop
            ping_timeout=None,
        ):
            logger.info("PolicyServer ready, waiting for connections...")
            await asyncio.Future()  # run forever

    async def _handler(self, ws) -> None:
        peer = ws.remote_address
        logger.info("Client connected: %s", peer)
        try:
            async for message in ws:
                try:
                    request = decode(message)
                    payload = request["payload"]
                    payload = {str(k): v for k, v in payload.items()}
                    # Enqueue and await batched result.
                    actions = await self._enqueue(payload)
                    response = encode({"status": "ok", "actions": actions})
                except Exception as e:
                    logger.exception("Prediction error")
                    response = encode({"status": "error", "message": str(e)})
                await ws.send(response)
        except websockets.ConnectionClosed:
            logger.info("Client disconnected: %s", peer)

    # ------------------------------------------------------------------
    # Batch collection
    # ------------------------------------------------------------------

    async def _enqueue(self, payload: dict) -> np.ndarray:
        """Add a request to the batch queue and wait for the result."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._queue.append((payload, fut))
        self._queue_event.set()  # wake up the dispatcher
        return await fut

    async def _batch_dispatcher(self) -> None:
        """Background task that collects requests and dispatches batches."""
        while True:
            # Wait until at least one request arrives.
            await self._queue_event.wait()
            self._queue_event.clear()

            # Wait up to max_wait_ms for more requests to arrive.
            deadline = time.monotonic() + self.max_wait_ms / 1000.0
            while len(self._queue) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    await asyncio.wait_for(
                        self._wait_for_more(), timeout=remaining
                    )
                except asyncio.TimeoutError:
                    break

            # Snapshot and clear the queue.
            batch = self._queue[:]
            self._queue.clear()
            if not batch:
                continue

            # Per-dispatch log. INFO when batching is actually doing work
            # (batch > 1), DEBUG when it's effectively a passthrough
            # (batch == 1) — without this distinction, a 1 server : 1
            # client setup like robotwin (max_batch_size=1) writes one
            # ``Dispatching batch of 1 requests`` line per inference,
            # producing thousands of lines per task. libero / calvin /
            # mixed-batch evals still see the useful "batch of N>1" lines.
            n_batch = len(batch)
            if n_batch > 1:
                logger.info("Dispatching batch of %d requests", n_batch)
            else:
                logger.debug("Dispatching single request")
            # Run inference (synchronous/CPU-bound work in the event loop
            # is acceptable here because the GPU forward pass dominates and
            # we WANT to block further dispatches until it completes).
            payloads = [item[0] for item in batch]
            futures = [item[1] for item in batch]
            try:
                results = self._predict_batch(payloads)
                for fut, result in zip(futures, results):
                    if not fut.cancelled():
                        fut.set_result(result)
            except Exception as e:
                for fut in futures:
                    if not fut.cancelled() and not fut.done():
                        fut.set_exception(e)

    async def _wait_for_more(self) -> None:
        """Helper: wait until a new item is added to the queue."""
        # Spin-check with a tiny sleep — avoids complex synchronization.
        current_len = len(self._queue)
        while len(self._queue) == current_len:
            await asyncio.sleep(0.001)  # 1 ms granularity

    # ------------------------------------------------------------------
    # Inference dispatch
    # ------------------------------------------------------------------

    def _predict_batch(self, payloads: list) -> list:
        """Dispatch a batched inference call.

        Uses ``predict_action_batch`` when available.
        Falls back to sequential ``_predict`` for legacy nn.Module policies.
        """
        if hasattr(self.policy, "predict_action_batch"):
            return self.policy.predict_action_batch(payloads)
        # Legacy fallback: process one at a time.
        return [self._predict(p) for p in payloads]

    @torch.no_grad()
    def _predict(self, payload: dict) -> np.ndarray:
        """Dispatch one inference call (legacy / fallback path).

        Preferred path — ``policy.predict_action(images, state, instruction)``. Falls back to the legacy ``**kwargs`` passthrough so
        older adapter models keep working.
        """
        if hasattr(self.policy, "predict_action"):
            actions = self.policy.predict_action(**payload)
        elif hasattr(self.policy, "forward"):
            # Raw nn.Module path — build tensor kwargs on the target device.
            torch_kwargs: dict = {}
            for k, v in payload.items():
                if isinstance(v, np.ndarray):
                    torch_kwargs[k] = torch.from_numpy(v).to(self.device)
                else:
                    torch_kwargs[k] = v
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = self.policy(action_chunks=None, **torch_kwargs)
            actions = output.actions
        else:
            raise TypeError(
                f"Policy {type(self.policy).__name__} has neither predict_action() nor forward()."
            )

        if isinstance(actions, torch.Tensor):
            actions = actions.detach().to(dtype=torch.float32).cpu().numpy()
        return actions
