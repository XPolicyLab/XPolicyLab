"""Dynamic-batching websocket server for an OpenPI policy.

This module lives outside ``third_party/openpi`` so the upstream checkout stays
untouched.  It adapts OpenPI's single-observation ``Policy`` to its underlying
model's native batch dimension, then coalesces websocket requests into bounded
micro-batches.

Heavy OpenPI, JAX, PyTorch, and websocket imports remain lazy so this module can
still be imported by the lightweight evaluation and unit-test environments.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
import contextlib
from dataclasses import dataclass
import logging
import queue
import threading
import time
import traceback
from typing import Any, Protocol

from server.protocol import (
    health_check,
    log_server_ready,
    pack_message,
    unpack_message,
)


logger = logging.getLogger(__name__)


class BatchPolicy(Protocol):
    """Minimal policy contract consumed by :class:`DynamicBatcher`."""

    def infer_batch(
        self,
        observations: Sequence[dict[str, Any]],
        *,
        pad_to: int | None = None,
    ) -> list[dict[str, Any]]: ...


class OpenPIBatchPolicy:
    """Expose batched inference for an upstream ``openpi.policies.Policy``.

    OpenPI's public ``Policy.infer`` transforms one observation, inserts a
    singleton batch dimension, calls ``sample_actions``, and removes that
    dimension.  This adapter performs the same operations for multiple
    observations while preserving per-observation input/output transforms.

    Access to the policy's RNG and model is serialized by ``DynamicBatcher``.
    """

    _REQUIRED_ATTRIBUTES = (
        "_input_transform",
        "_output_transform",
        "_sample_actions",
        "_sample_kwargs",
        "_is_pytorch_model",
    )

    def __init__(self, policy: Any) -> None:
        missing = [
            name for name in self._REQUIRED_ATTRIBUTES if not hasattr(policy, name)
        ]
        if missing:
            names = ", ".join(missing)
            raise TypeError(
                "Dynamic batching requires an openpi.policies.policy.Policy; "
                f"missing internal attribute(s): {names}"
            )
        if not policy._is_pytorch_model and not hasattr(policy, "_rng"):
            raise TypeError("JAX OpenPI policy is missing its RNG state")
        self._policy = policy

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._policy.metadata)

    def infer_batch(
        self,
        observations: Sequence[dict[str, Any]],
        *,
        pad_to: int | None = None,
    ) -> list[dict[str, Any]]:
        if not observations:
            return []

        request_count = len(observations)
        if pad_to is not None and pad_to < request_count:
            raise ValueError(
                f"pad_to ({pad_to}) cannot be smaller than batch ({request_count})"
            )
        effective_batch_size = pad_to or request_count

        import jax
        import numpy as np
        from openpi.models import model as openpi_model

        transformed_inputs = []
        for observation in observations:
            # Match upstream Policy.infer: copy containers because transforms may
            # pop or replace fields in-place.
            inputs = jax.tree.map(lambda value: value, observation)
            transformed_inputs.append(self._policy._input_transform(inputs))

        if effective_batch_size > request_count:
            transformed_inputs.extend(
                transformed_inputs[-1]
                for _ in range(effective_batch_size - request_count)
            )

        if self._policy._is_pytorch_model:
            import torch

            device = self._policy._pytorch_device

            def stack_torch(*values):
                array = np.ascontiguousarray(
                    np.stack([np.asarray(value) for value in values], axis=0)
                )
                return torch.from_numpy(array).to(device)

            batched_inputs = jax.tree.map(
                stack_torch,
                *transformed_inputs,
            )
            sample_rng_or_device = device
        else:
            import jax.numpy as jnp

            batched_inputs = jax.tree.map(
                lambda *values: jnp.stack(
                    [jnp.asarray(value) for value in values],
                    axis=0,
                ),
                *transformed_inputs,
            )
            self._policy._rng, sample_rng_or_device = jax.random.split(
                self._policy._rng
            )

        observation = openpi_model.Observation.from_dict(batched_inputs)
        sample_kwargs = dict(self._policy._sample_kwargs)
        started = time.monotonic()
        actions = self._policy._sample_actions(
            sample_rng_or_device,
            observation,
            **sample_kwargs,
        )
        infer_ms = (time.monotonic() - started) * 1000

        batched_outputs = {
            "state": batched_inputs["state"],
            "actions": actions,
        }
        if self._policy._is_pytorch_model:
            batched_outputs = jax.tree.map(
                lambda value: np.asarray(value.detach().cpu()),
                batched_outputs,
            )
        else:
            # np.asarray synchronizes JAX device work before timing/result return.
            batched_outputs = jax.tree.map(np.asarray, batched_outputs)

        results: list[dict[str, Any]] = []
        for index in range(request_count):
            item = jax.tree.map(
                lambda value: value[index, ...],
                batched_outputs,
            )
            output = dict(self._policy._output_transform(item))
            output["policy_timing"] = {
                "infer_ms": infer_ms,
                "batch_size": request_count,
                "padded_batch_size": effective_batch_size,
            }
            results.append(output)
        return results


@dataclass(frozen=True)
class BatchResponse:
    action: dict[str, Any]
    batch_size: int
    padded_batch_size: int
    queue_ms: float
    infer_ms: float


@dataclass
class _PendingRequest:
    observation: dict[str, Any]
    future: asyncio.Future[BatchResponse]
    enqueued_at: float


@dataclass(frozen=True)
class _BatchJob:
    loop: asyncio.AbstractEventLoop
    requests: tuple[_PendingRequest, ...]
    pad_to: int | None
    started_at: float


class DynamicBatcher:
    """Collect requests briefly and run one serialized batched policy call."""

    def __init__(
        self,
        policy: BatchPolicy,
        *,
        max_batch_size: int,
        batch_wait_ms: float,
        pad_to_max: bool = False,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if batch_wait_ms < 0:
            raise ValueError("batch_wait_ms cannot be negative")

        self._policy = policy
        self._max_batch_size = int(max_batch_size)
        self._batch_wait_s = float(batch_wait_ms) / 1000.0
        self._pad_to_max = bool(pad_to_max)
        self._queue: asyncio.Queue[_PendingRequest] = asyncio.Queue()
        self._runner: asyncio.Task[None] | None = None
        self._inference_running = False
        self._active_requests: tuple[_PendingRequest, ...] = ()
        self._closed = False
        self._jobs: queue.Queue[_BatchJob | None] = queue.Queue()
        self._worker = threading.Thread(
            target=self._policy_worker,
            name="openpi-batch-infer",
            daemon=True,
        )
        self._worker.start()

    async def infer(self, observation: dict[str, Any]) -> BatchResponse:
        if self._closed:
            raise RuntimeError("OpenPI dynamic batcher is closed")

        loop = asyncio.get_running_loop()
        future: asyncio.Future[BatchResponse] = loop.create_future()
        self._queue.put_nowait(
            _PendingRequest(
                observation=observation,
                future=future,
                enqueued_at=loop.time(),
            )
        )
        self._ensure_runner()
        try:
            # The timeout keeps a bounded event-loop heartbeat while the model
            # runs in its dedicated thread. Normally Future completion wakes
            # this immediately; the heartbeat is a fallback for event loops
            # whose selector doesn't observe nested Future wakeups promptly.
            while not future.done():
                await asyncio.wait({future}, timeout=0.01)
            return future.result()
        except asyncio.CancelledError:
            future.cancel()
            raise

    def _ensure_runner(self) -> None:
        if (
            self._runner is None
            and not self._inference_running
            and not self._closed
        ):
            self._runner = asyncio.create_task(
                self._run_one_batch(),
                name="openpi-dynamic-batcher",
            )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self._runner is not None:
            self._runner.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._runner

        self._fail_requests(
            self._active_requests,
            RuntimeError("OpenPI dynamic batcher closed during inference"),
        )
        error = RuntimeError("OpenPI dynamic batcher closed before inference")
        while not self._queue.empty():
            request = self._queue.get_nowait()
            if not request.future.done():
                request.future.set_exception(error)
        self._jobs.put_nowait(None)

    async def _run_one_batch(self) -> None:
        requests: list[_PendingRequest] = []
        try:
            first = self._queue.get_nowait()
            requests.append(first)
            await self._collect_batch(requests)
            requests = [
                request for request in requests if not request.future.cancelled()
            ]
            if not requests:
                return

            pad_to = self._max_batch_size if self._pad_to_max else None
            loop = asyncio.get_running_loop()
            self._active_requests = tuple(requests)
            self._inference_running = True
            self._jobs.put_nowait(
                _BatchJob(
                    loop=loop,
                    requests=self._active_requests,
                    pad_to=pad_to,
                    started_at=loop.time(),
                )
            )
        except asyncio.CancelledError:
            self._fail_requests(
                requests,
                RuntimeError("OpenPI dynamic batcher stopped during inference"),
            )
            raise
        except Exception as exc:
            logger.exception(
                "OpenPI batch collection failed for batch_size=%d",
                len(requests),
            )
            self._fail_requests(requests, exc)
        finally:
            self._runner = None
            if not self._queue.empty():
                self._ensure_runner()

    def _policy_worker(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return

            started = time.monotonic()
            outputs: list[dict[str, Any]] | None = None
            error: BaseException | None = None
            try:
                outputs = self._policy.infer_batch(
                    [request.observation for request in job.requests],
                    pad_to=job.pad_to,
                )
                if len(outputs) != len(job.requests):
                    raise RuntimeError(
                        "Batched policy returned "
                        f"{len(outputs)} result(s) for {len(job.requests)} request(s)"
                    )
            except BaseException as exc:
                logger.exception(
                    "OpenPI batched inference failed for batch_size=%d",
                    len(job.requests),
                )
                error = exc

            infer_ms = (time.monotonic() - started) * 1000
            with contextlib.suppress(RuntimeError):
                job.loop.call_soon_threadsafe(
                    self._finish_batch,
                    job,
                    outputs,
                    error,
                    infer_ms,
                )

    def _finish_batch(
        self,
        job: _BatchJob,
        outputs: list[dict[str, Any]] | None,
        error: BaseException | None,
        infer_ms: float,
    ) -> None:
        self._inference_running = False
        self._active_requests = ()

        if error is not None:
            self._fail_requests(job.requests, error)
        elif outputs is not None:
            completed_at = job.loop.time()
            padded_batch_size = job.pad_to or len(job.requests)
            for request, output in zip(job.requests, outputs, strict=True):
                if request.future.done():
                    continue
                request.future.set_result(
                    BatchResponse(
                        action=output,
                        batch_size=len(job.requests),
                        padded_batch_size=padded_batch_size,
                        queue_ms=(job.started_at - request.enqueued_at) * 1000,
                        infer_ms=infer_ms,
                    )
                )
            logger.debug(
                "OpenPI batch completed size=%d padded=%d infer_ms=%.1f total_ms=%.1f",
                len(job.requests),
                padded_batch_size,
                infer_ms,
                (completed_at - min(r.enqueued_at for r in job.requests)) * 1000,
            )

        if not self._queue.empty():
            self._ensure_runner()

    async def _collect_batch(
        self,
        requests: list[_PendingRequest],
    ) -> None:
        deadline = asyncio.get_running_loop().time() + self._batch_wait_s

        while len(requests) < self._max_batch_size:
            try:
                requests.append(self._queue.get_nowait())
                continue
            except asyncio.QueueEmpty:
                pass

            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            try:
                request = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=remaining,
                )
            except TimeoutError:
                break
            requests.append(request)

    @staticmethod
    def _fail_requests(
        requests: Sequence[_PendingRequest],
        error: BaseException,
    ) -> None:
        for request in requests:
            if not request.future.done():
                request.future.set_exception(error)


class BatchedWebsocketPolicyServer:
    """OpenPI websocket protocol server backed by :class:`DynamicBatcher`."""

    def __init__(
        self,
        policy: Any,
        *,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict[str, Any] | None = None,
        max_batch_size: int = 4,
        batch_wait_ms: float = 10.0,
        pad_to_max: bool = False,
    ) -> None:
        batch_policy = OpenPIBatchPolicy(policy)
        self._host = host
        self._port = port
        self._metadata = metadata or batch_policy.metadata
        self._batcher = DynamicBatcher(
            batch_policy,
            max_batch_size=max_batch_size,
            batch_wait_ms=batch_wait_ms,
            pad_to_max=pad_to_max,
        )
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        import websockets.asyncio.server as websocket_server

        try:
            async with websocket_server.serve(
                self._handler,
                self._host,
                self._port,
                compression=None,
                max_size=None,
                process_request=health_check,
            ) as server:
                log_server_ready(
                    logger,
                    service="OpenPI",
                    host=self._host,
                    port=int(self._port or 8000),
                )
                await server.serve_forever()
        finally:
            await self._batcher.close()

    async def _handler(self, websocket: Any) -> None:
        import websockets
        import websockets.frames

        logger.info("Connection from %s opened", websocket.remote_address)
        await websocket.send(pack_message(self._metadata))

        previous_total_time = None
        while True:
            try:
                started = time.monotonic()
                observation = unpack_message(await websocket.recv())
                response = await self._batcher.infer(observation)
                action = dict(response.action)
                action["server_timing"] = {
                    "infer_ms": response.infer_ms,
                    "queue_ms": response.queue_ms,
                    "batch_size": response.batch_size,
                    "padded_batch_size": response.padded_batch_size,
                }
                if previous_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = (
                        previous_total_time * 1000
                    )
                await websocket.send(pack_message(action))
                previous_total_time = time.monotonic() - started
            except websockets.ConnectionClosed:
                logger.info(
                    "Connection from %s closed",
                    websocket.remote_address,
                )
                break
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "OpenPI request failed for %s",
                    websocket.remote_address,
                )
                with contextlib.suppress(websockets.ConnectionClosed):
                    await websocket.send(traceback.format_exc())
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error. Traceback included in previous frame.",
                    )
                break

def create_policy(*, config_name: str, checkpoint_dir: str) -> Any:
    """Load an OpenPI policy without importing heavyweight modules at startup."""
    from openpi.policies import policy_config
    from openpi.training import config as openpi_config

    train_config = openpi_config.get_config(config_name)
    logger.info(
        "Loaded openpi TrainConfig(name=%s): %s",
        config_name,
        train_config.model,
    )
    return policy_config.create_trained_policy(train_config, checkpoint_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve an OpenPI policy with dynamic inference batching."
    )
    parser.add_argument(
        "--config-name",
        default="pi05_libero",
        help="OpenPI TrainConfig name (default: pi05_libero).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Path to the OpenPI checkpoint directory or gs:// URI.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind host (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Bind port (default: 8000).",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=4,
        help="Maximum number of requests in one inference batch (default: 4).",
    )
    parser.add_argument(
        "--batch-wait-ms",
        type=float,
        default=10.0,
        help="Maximum time to collect a batch in milliseconds (default: 10).",
    )
    parser.add_argument(
        "--batch-pad-to-max",
        action="store_true",
        help=(
            "Pad every inference to --max-batch-size for a fixed JIT shape; "
            "uses more compute and GPU memory."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.max_batch_size <= 0:
        parser.error("--max-batch-size must be positive")
    if args.batch_wait_ms < 0:
        parser.error("--batch-wait-ms cannot be negative")

    policy = create_policy(
        config_name=args.config_name,
        checkpoint_dir=args.checkpoint_dir,
    )
    server = BatchedWebsocketPolicyServer(
        policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
        max_batch_size=args.max_batch_size,
        batch_wait_ms=args.batch_wait_ms,
        pad_to_max=args.batch_pad_to_max,
    )
    logger.info(
        "Serving batched OpenPI policy on ws://%s:%d "
        "(max_batch_size=%d, wait_ms=%.1f, pad_to_max=%s)",
        args.host,
        args.port,
        args.max_batch_size,
        args.batch_wait_ms,
        args.batch_pad_to_max,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
