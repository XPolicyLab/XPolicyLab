"""Adapter-local dynamic batching on top of the official WebSocket server."""

import asyncio
import logging
import math
import os
from collections import deque
from dataclasses import dataclass, field

from client_server.ws.model_server import PolicyServer, PolicyServerConfig

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class Pending:
    future: asyncio.Future
    count: int
    prepared: dict = field(default_factory=dict)
    cursor: int = 0
    completed: int = 0
    results: list = field(default_factory=list)
    enqueued: float = 0.0


class DynamicPolicyServer(PolicyServer):
    """Queue only atomic context calls; retain official decoding and replay handling."""

    def __init__(self, model, config=None, *, wait_ms=10.0, max_pending_envs=64):
        super().__init__(model, config or PolicyServerConfig())
        self.batch_size = model.inference_batch_size
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("inference_batch_size must be positive")
        if not math.isfinite(wait_ms) or wait_ms < 0:
            raise ValueError("batch wait_ms must be finite and nonnegative")
        if type(max_pending_envs) is not int or max_pending_envs < self.batch_size:
            raise ValueError("max_pending_envs must be at least inference_batch_size")
        self.wait_seconds = wait_ms / 1000
        self.max_pending_envs = max_pending_envs
        self._pending_envs = 0
        self._tickets = set()
        self._queue = deque()
        self._arrived = asyncio.Event()
        self._worker = None
        self._batch_closed = False
        self._failure = None
        self.batch_stats = {
            "batches": 0,
            "rows": 0,
            "mixed_request_batches": 0,
            "max_batch": 0,
            "rejected": 0,
        }

    async def _call_model_method(self, method, *args):
        if self._failure is not None:
            raise RuntimeError(
                "Dynamic batch worker failed; restart this server"
            ) from self._failure
        if (
            getattr(method, "__self__", None) is self.model
            and getattr(method, "__name__", "") == "infer_context"
        ):
            return await self._submit(*args)
        if getattr(method, "__name__", "") in {
            "update_obs",
            "update_obs_batch",
            "get_action",
            "get_action_batch",
        }:
            raise ValueError(
                "Dynamic batching requires HWM_CLIENT_CONTEXT=1 and infer_context"
            )
        # Legacy methods never enter the queue, and cannot race GPU execution.
        return await super()._call_model_method(method, *args)

    def _release(self, ticket):
        self._pending_envs -= ticket.count
        self._tickets.discard(ticket)
        if not ticket.future.cancelled():
            ticket.future.exception()  # Mark observed even if the caller disconnected.

    async def _submit(self, observations):
        if self._batch_closed:
            raise RuntimeError("Dynamic batch server is closed")
        if not isinstance(observations, list) or not observations:
            raise ValueError("Expected nonempty context observations")
        count = len(observations)
        if self._pending_envs + count > self.max_pending_envs:
            self.batch_stats["rejected"] += 1
            raise RuntimeError("Dynamic batch queue is full; request was not admitted")
        # Reserve before CPU preparation, so concurrent preparers cannot create an
        # unbounded queue of prepared image/history tensors. Includes in-flight rows.
        ticket = Pending(asyncio.get_running_loop().create_future(), count)
        self._pending_envs += count
        self._tickets.add(ticket)
        ticket.future.add_done_callback(lambda _: self._release(ticket))
        try:
            ticket.prepared = await asyncio.to_thread(
                self.model._prepare_context, observations
            )
        except BaseException:
            ticket.future.cancel()
            raise
        if ticket.future.done():
            return await ticket.future  # Shutdown/fatal error during preparation.
        ticket.results = [None] * count
        ticket.enqueued = asyncio.get_running_loop().time()
        self._queue.append(ticket)
        self._arrived.set()
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run_batches())
        # Official transport already shields executions/replays. Also shield here
        # so cancellation of a waiter does not drop another client's GPU work.
        return await asyncio.shield(ticket.future)

    def _queued_rows(self):
        return sum(
            ticket.count - ticket.cursor
            for ticket in self._queue
            if not ticket.future.done()
        )

    def _fail_pending(self, message):
        self._queue.clear()
        for ticket in list(self._tickets):
            if not ticket.future.done():
                error = RuntimeError(message)
                error.__cause__ = self._failure
                ticket.future.set_exception(error)
        self._arrived.set()

    async def _run_batches(self):
        try:
            while self._queue and not self._batch_closed:
                deadline = self._queue[0].enqueued + self.wait_seconds
                while self._queued_rows() < self.batch_size and not self._batch_closed:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        break
                    self._arrived.clear()
                    try:
                        await asyncio.wait_for(self._arrived.wait(), timeout=remaining)
                    except asyncio.TimeoutError:
                        break
                if self._batch_closed:
                    break
                selected = []
                while self._queue and len(selected) < self.batch_size:
                    ticket = self._queue.popleft()
                    if ticket.future.done():
                        continue
                    index = ticket.cursor
                    ticket.cursor += 1
                    selected.append((ticket, index))
                    if ticket.cursor < ticket.count:
                        self._queue.append(ticket)
                if not selected:
                    continue
                rows = [ticket.prepared["rows"][index] for ticket, index in selected]
                async with self._model_lock:
                    if self._batch_closed:
                        break
                    items = await asyncio.to_thread(
                        self.model._predict_context_rows, rows
                    )
                if len(items) != len(selected):
                    raise RuntimeError("Model returned a different dynamic batch size")
                requests = len({id(ticket) for ticket, _ in selected})
                self.batch_stats["batches"] += 1
                self.batch_stats["rows"] += len(selected)
                self.batch_stats["mixed_request_batches"] += int(requests > 1)
                self.batch_stats["max_batch"] = max(
                    self.batch_stats["max_batch"], len(selected)
                )
                logger.info(
                    "HWM dynamic batch: envs=%d requests=%d queued_envs=%d",
                    len(selected),
                    requests,
                    self._queued_rows(),
                )
                for (ticket, index), item in zip(selected, items, strict=True):
                    ticket.results[index] = item
                    ticket.completed += 1
                    if ticket.completed == ticket.count and not ticket.future.done():
                        ticket.future.set_result(
                            {
                                "request_id": ticket.prepared["request_id"],
                                "items": ticket.results,
                            }
                        )
        except asyncio.CancelledError:
            self._batch_closed = True
            self._fail_pending("Dynamic batch worker was cancelled")
            raise
        except Exception as exc:
            # No silent single-row/OOM fallback, and no further work on a possibly
            # poisoned CUDA context. Every pending caller receives an explicit error.
            logger.exception("HWM dynamic batch worker failed")
            self._failure = exc
            self._batch_closed = True
            self._fail_pending(f"Dynamic batch inference failed: {exc}")

    async def stop(self):
        self._batch_closed = True
        self._fail_pending("Dynamic batch server stopped before request completion")
        if self._worker is not None and not self._worker.done():
            _, pending = await asyncio.wait({self._worker}, timeout=10)
            if pending:
                logger.warning(
                    "Dynamic GPU call is still finishing after shutdown drain"
                )
        await super().stop()


def create_server(model, cfg):
    enabled = os.environ.get("HWM_DYNAMIC_BATCH", "1")
    if enabled not in {"0", "1"}:
        raise ValueError("HWM_DYNAMIC_BATCH must be 0 or 1")
    config = PolicyServerConfig(host=cfg["host"], port=int(cfg["port"]))
    if enabled == "0":
        return PolicyServer(model, config)
    server = DynamicPolicyServer(
        model,
        config,
        wait_ms=float(os.environ.get("HWM_BATCH_WAIT_MS", "10")),
        max_pending_envs=int(os.environ.get("HWM_MAX_PENDING_ENVS", "64")),
    )
    logger.info(
        "HWM dynamic batching enabled: batch=%d wait_ms=%g max_pending_envs=%d",
        server.batch_size,
        server.wait_seconds * 1000,
        server.max_pending_envs,
    )
    return server


async def main():
    from setup_policy_server import parse_args_and_config

    from ...model import Model

    logging.basicConfig(level=logging.INFO)
    cfg = parse_args_and_config()
    if cfg.get("protocol", "ws") != "ws":
        raise ValueError("Dynamic batching requires the official WebSocket protocol")
    server = create_server(Model(cfg), cfg)
    await server.start()
    print(
        f"[SERVER] server running policy=HWM_0_preview ckpt={cfg['ckpt_name']} address={server.url}",
        flush=True,
    )
    await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
